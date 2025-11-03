# %% IMPORTS
import os
import yaml
import torch
import random
import typing
import argparse
import numpy as np
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score
from save_latent import extract_latents
from fetch_experiments import fetch_experiment
from sklearn.metrics import precision_recall_fscore_support
from utils import get_args_parser
class AttentionMIL(nn.Module):
    def __init__(self, input_dim=76, hidden_dim=128, att_dim=64, num_classes=7):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.feature_extractor(x)                # [N, hidden_dim]
        a = torch.softmax(self.attention(h), dim=0)  # [N, 1] attention over instances
        z = torch.sum(a * h, dim=0)                  # [hidden_dim] aggregated bag representation
        logits = self.classifier(z)                  # [num_classes]
        probs = torch.softmax(logits, dim=0)        # class probabilities
        return probs, a

class PatientDataset(Dataset):
    def __init__(self, patient_features, patient_labels):
        self.features = patient_features
        self.labels = patient_labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


parser = get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
# %%
runs_df = fetch_experiment(experiment_ids=list(range(798, 814)))
runs_df = runs_df[['sys/id',
                   'config/training_plan/parameters/norm_pix_loss',
                   'config/training_plan/parameters/include_lesion_mask',
                   'best_model_path']].copy()
runs_df['best_model_path'] = runs_df['best_model_path'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else x)

runs_df['micro_accuracy'] = 0.0
runs_df['macro_precision'] = 0.0
runs_df['macro_recall'] = 0.0
runs_df['macro_f1'] = 0.0
runs_df['weighted_precision'] = 0.0
runs_df['weighted_recall'] = 0.0
runs_df['weighted_f1'] = 0.0

# NEW: loop over runs_df rows, extract latents for each model, train MIL, store best metrics
# Deterministic seed (same for each model to ensure comparable MIL initialization & sampling)
SEED = 42

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for idx, row in runs_df.iterrows():
    model_name = row['best_model_path']
    if not isinstance(model_name, str) or model_name == 'nan':
        print(f"Skipping row {idx} because best_model_path is missing")
        continue

    print(f"\n=== Processing run {idx} - model: {model_name} ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    try:
        (patch_level_latents_df_train,
         patch_level_latents_df_test,
         latent_pooled_train,
         latent_pooled_test,
         latent_raw_train,
         latent_raw_test) = extract_latents(config, path=model_name, remove_background=False)
    except Exception as e:
        print(f"  Error extracting latents for {model_name}: {e}")
        runs_df.loc[idx, ['micro_accuracy','macro_precision','macro_recall','macro_f1',
                          'weighted_precision','weighted_recall','weighted_f1']] = [np.nan]*7
        continue

    patch_level_latents_df_train['patient_id'] = patch_level_latents_df_train['image_path'].apply(
        lambda x: os.path.basename(x).split('_')[1].split('.')[0]
    )
    patch_level_latents_df_test['patient_id'] = patch_level_latents_df_test['image_path'].apply(
        lambda x: os.path.basename(x).split('_')[1].split('.')[0]
    )

    drop_background = False
    if drop_background:
        patch_level_latents_df_train = patch_level_latents_df_train[patch_level_latents_df_train['patch_in_mask'] == 1].reset_index(drop=True)
        patch_level_latents_df_test = patch_level_latents_df_test[patch_level_latents_df_test['patch_in_mask'] == 1].reset_index(drop=True)

    # --- Prepare patient-level features/labels ---
    train_patient_features = [
        torch.tensor(np.vstack(g['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_level_latents_df_train.groupby('patient_id')
    ]
    train_patient_labels = [
        int(g['target'].mode().iat[0])
        for pid, g in patch_level_latents_df_train.groupby('patient_id')
    ]

    test_patient_features = [
        torch.tensor(np.vstack(g['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_level_latents_df_test.groupby('patient_id')
    ]
    test_patient_labels = [
        int(g['target'].mode().iat[0])
        for pid, g in patch_level_latents_df_test.groupby('patient_id')
    ]

    train_dataset = PatientDataset(train_patient_features, train_patient_labels)
    val_dataset = PatientDataset(test_patient_features, test_patient_labels)

    patient_labels = np.array(train_patient_labels)
    patient_class_counts = Counter(patient_labels)
    weights = np.array([1.0 / patient_class_counts[int(lbl)] for lbl in patient_labels], dtype=np.float64)
    sample_weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionMIL(input_dim=train_dataset[0][0].shape[1], hidden_dim=256, att_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    best_bacc = -np.inf
    best_metrics = { 'micro': np.nan, 'macro_p': np.nan, 'macro_r': np.nan, 'macro_f1': np.nan,
                     'weighted_p': np.nan, 'weighted_r': np.nan, 'weighted_f1': np.nan }
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_val_loss = 0.0
        for x, y in train_loader:
            x = x[0].to(device)
            y_long = y.to(device).long()

            optimizer.zero_grad()
            probs, att = model(x)
            logp = torch.log(probs + 1e-9).unsqueeze(0)
            loss = criterion(logp, y_long)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        y_true = []
        y_score = []
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x[0].to(device)
                y_long = y.to(device).long()
                probs, att = model(x)
                y_true.append(int(y_long.item()))
                y_score.append(probs.cpu().numpy())
                val_loss = criterion(torch.log(probs.unsqueeze(0) + 1e-9), y_long)
                total_val_loss += val_loss.item()

        if len(y_true) == 0:
            print("  No validation samples, skipping epoch metrics")
            continue

        y_true = np.array(y_true)
        y_score = np.vstack(y_score)
        try:
            auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception:
            auc = float('nan')

        y_pred_labels = np.argmax(y_score, axis=1)
        acc = accuracy_score(y_true, y_pred_labels)
        bacc = balanced_accuracy_score(y_true, y_pred_labels)

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred_labels, average='weighted', zero_division=0)

        if bacc > best_bacc:
            best_bacc = bacc
            best_metrics['micro'] = acc
            best_metrics['macro_p'] = macro_p
            best_metrics['macro_r'] = macro_r
            best_metrics['macro_f1'] = macro_f1
            best_metrics['weighted_p'] = weighted_p
            best_metrics['weighted_r'] = weighted_r
            best_metrics['weighted_f1'] = weighted_f1

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}, AUC={auc:.3f}, ACC={acc:.3f}, BACC={bacc:.3f}")

    runs_df.loc[idx, 'micro_accuracy'] = best_metrics['micro']
    runs_df.loc[idx, 'macro_precision'] = best_metrics['macro_p']
    runs_df.loc[idx, 'macro_recall'] = best_metrics['macro_r']
    runs_df.loc[idx, 'macro_f1'] = best_metrics['macro_f1']
    runs_df.loc[idx, 'weighted_precision'] = best_metrics['weighted_p']
    runs_df.loc[idx, 'weighted_recall'] = best_metrics['weighted_r']
    runs_df.loc[idx, 'weighted_f1'] = best_metrics['weighted_f1']

out_pickle = "runs_df_mil_results.pkl"
out_csv = "runs_df_mil_results.csv"
runs_df.to_pickle(out_pickle)
runs_df.to_csv(out_csv, index=False)
print(f"\nSaved runs results to {out_pickle} and {out_csv}")
# %%