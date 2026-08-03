import os
import yaml
import pickle
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter
from utils_g_mil import AttentionMIL_teacher, PatientDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from utils import get_args_parser
from save_latent import extract_latents
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support

def _sort_group_patches(g):
        """Return group DataFrame sorted by numeric patch id.

        Strategy:
        - If column `patch_id` exists, sort by it.
        - Otherwise try to parse a trailing integer from `image_path` basename
          (last underscore-separated token before extension).
        - If parsing fails, preserve original order.
        """
        if 'patch_id' in g.columns:
            try:
                return g.sort_values('patch_id')
            except Exception:
                return g

        def _extract_from_path(x):
            try:
                b = os.path.basename(x)
                name = os.path.splitext(b)[0]
                tok = name.split('_')[-1]
                return int(tok)
            except Exception:
                return None

        g = g.copy()
        g['_patch_num'] = g['image_path'].apply(_extract_from_path)
        if g['_patch_num'].notnull().all():
            return g.sort_values('_patch_num')
        # if we couldn't parse numeric ids for all rows, drop helper column and return original
        g = g.drop(columns=['_patch_num'], errors='ignore')
        return g

def _build_patient_bags(df):
    patient_features = []
    patient_labels = []
    patient_image_ids = []

    for _, g in df.groupby('patient_id'):
        g_sorted = _sort_group_patches(g)
        patient_features.append(np.vstack(g_sorted['patch_latent'].values))
        patient_labels.append(int(g_sorted['target'].mode().iat[0]))

        if 'image_id' in g_sorted.columns:
            image_id = str(g_sorted['image_id'].iloc[0])
        else:
            image_id = os.path.splitext(os.path.basename(g_sorted['image_path'].iloc[0]))[0]
        patient_image_ids.append(image_id)

    return patient_features, patient_labels, patient_image_ids

def _collect_teacher_outputs(model_obj, loader, image_ids, device_obj):
    teacher_outputs = []

    model_obj.eval()
    with torch.no_grad():
        for (x, y_batch), image_id in zip(loader, image_ids):
            x = x[0].to(device_obj)
            y_value = int(y_batch.item())
            outputs = model_obj(x)

            teacher_outputs.append({
                'image_id': image_id,
                'label': y_value,
                'patch_probs': outputs['patch_probs'].cpu().numpy(),
                'attention': outputs['attention'].cpu().numpy(),
                'patch_embeddings': x.cpu().numpy(),
            })

    return pd.DataFrame(teacher_outputs)

def _evaluate_model(state_dict):
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for x, y_batch in test_loader:
            x = x[0].to(device)
            y_long = y_batch.to(device).long()
            outputs = model(x)
            probs = outputs["bag_probs"]
            y_true.append(int(y_long.item()))
            y_score.append(probs.cpu().numpy())

    if len(y_true) == 0:
        return {k: np.nan for k in ['micro','macro_p','macro_r','macro_f1','weighted_p','weighted_r','weighted_f1']}

    y_true_arr = np.array(y_true)
    y_score_arr = np.vstack(y_score)
    y_pred_arr = np.argmax(y_score_arr, axis=1)

    micro_acc = accuracy_score(y_true_arr, y_pred_arr)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='weighted', zero_division=0)

    return {'micro': micro_acc,
            'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
            'weighted_p': weighted_p, 'weighted_r': weighted_r, 'weighted_f1': weighted_f1}
# %%

# -----------------------------------------------------------------------------------------------

# model_name="a9d7feb3402a4670bbcfa73f534acab7.pth"  # <-- the AE model basename to use 799
# model_name="ce4069521dfb4264a3ac8cc3d59971a2.pth"  # 726
# model_name="e6b29aa3b47145ec935e675a13c4b71d.pth"  # 804
model_name="c72210e208974529927e6c53d8ec890c.pth"    # 805
# model_name="4175dac48c3b4e93b4c0c82e8d8b44ff.pth"  # 806
# model_name="6d4c4f1198f0439583ffd3af0a76ef9f.pth"  # 802

# -----------------------------------------------------------------------------------------------


# %%
parser = get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# %%
load = True
if load:
    patch_train_df = '/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/dataframes_latents/patch_level_latents_train_df.pkl'
    patch_test_df = '/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/dataframes_latents/patch_level_latents_test_df.pkl'
    with open(patch_train_df, 'rb') as f:
        patch_train_df = pickle.load(f)
    with open(patch_test_df, 'rb') as f:
        patch_test_df = pickle.load(f)
else:
    patch_level_train_df, patch_level_test_df, latent_pooled_train, latent_pooled_test, latent_raw_train, latent_raw_test = extract_latents(config, model_name, remove_background=False)
    patch_train_df = patch_level_train_df
    patch_test_df = patch_level_test_df
# %%
patch_train_df['patient_id'] = patch_train_df['image_path'].apply(
    lambda x: os.path.basename(x).split('_')[1].split('.')[0]
)
patch_test_df['patient_id'] = patch_test_df['image_path'].apply(
    lambda x: os.path.basename(x).split('_')[1].split('.')[0]
)

train_patient_features, train_patient_labels, train_patient_image_ids = _build_patient_bags(patch_train_df)
test_patient_features, test_patient_labels, test_patient_image_ids = _build_patient_bags(patch_test_df)
# %%
SPLITS = 5
SEED = 42
fold_test_metrics_bacc = []
fold_test_metrics_loss = []
skf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)
y = np.array(train_patient_labels)


test_dataset = PatientDataset(test_patient_features, test_patient_labels)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

teacher_outputs_dir = os.path.join('teacher_outputs', os.path.splitext(model_name)[0])
os.makedirs(teacher_outputs_dir, exist_ok=True)

num_epochs = 200
criterion = nn.CrossEntropyLoss()

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    print(f"  Fold {fold_idx+1}/{SPLITS}: train {len(train_idx)} patients, val {len(val_idx)} patients")

    fold_train_feats = [train_patient_features[i] for i in train_idx]
    fold_train_labels = [int(train_patient_labels[i]) for i in train_idx]
    fold_train_image_ids = [train_patient_image_ids[i] for i in train_idx]
    fold_val_feats = [train_patient_features[i] for i in val_idx]
    fold_val_labels = [int(train_patient_labels[i]) for i in val_idx]
    fold_val_image_ids = [train_patient_image_ids[i] for i in val_idx]

    patient_labels_fold = np.array(fold_train_labels)
    class_counts = Counter(patient_labels_fold)
    weights = np.array([1.0 / class_counts[int(lbl)] for lbl in patient_labels_fold], dtype=np.float64)
    sample_weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    num_classes = len(set(fold_train_labels))

    train_dataset = PatientDataset(fold_train_feats, fold_train_labels)
    val_dataset = PatientDataset(fold_val_feats, fold_val_labels)

    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, drop_last=False)
    train_eval_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    torch.manual_seed(SEED + fold_idx)
    np.random.seed(SEED + fold_idx)
    random.seed(SEED + fold_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + fold_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = train_dataset[0][0].shape[1]
    best_params = config['best_params']
    model = AttentionMIL_teacher(input_dim=input_dim,
                            hidden_dim=best_params['hidden_dim'],
                            att_dim=best_params['att_dim'],
                            dropout=best_params['dropout'],
                            num_classes=num_classes).to(device)
    if best_params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                        lr=best_params['learning_rate'],
                                        weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=best_params['learning_rate'],
                                        weight_decay=best_params['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer: {best_params['optimizer']}")



    patience = config.get('training_plan', {}).get('parameters', {}).get('patience', 8)
    epochs_no_improve = 0
    best_val_bacc = -np.inf
    best_state_bacc = None
    best_val_loss = float('inf')
    best_state_loss = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        for x, y_batch in train_loader:
            x = x[0].to(device)
            y_long = y_batch.to(device).long()
            optimizer.zero_grad()
            outputs = model(x)
            probs = outputs["bag_probs"]
            logits = outputs["bag_logits"]
            loss = criterion(logits.unsqueeze(0), y_long)
            loss.backward()
            optimizer.step()
        model.eval()
        y_true = []
        y_score = []
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y_batch in val_loader:
                x = x[0].to(device)
                y_long = y_batch.to(device).long()
                outputs = model(x)
                probs = outputs["bag_probs"]
                logits = outputs["bag_logits"]
                y_true.append(int(y_long.item()))
                y_score.append(probs.cpu().numpy())
                val_loss_sum += criterion(logits.unsqueeze(0), y_long).item()
        if len(y_true) == 0:
            print("    No validation samples for this fold, skipping")
            break

        y_true = np.array(y_true)
        y_score = np.vstack(y_score)
        y_pred = np.argmax(y_score, axis=1)
        try:
            _ = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception:
            pass
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        val_loss = val_loss_sum / len(val_loader)

        if val_bacc > best_val_bacc + 1e-6:
            best_val_bacc = val_bacc
            best_state_bacc = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if val_loss < best_val_loss - 1e-6:
            epochs_no_improve = 0
            best_val_loss = val_loss
            best_state_loss = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        print(f"    Epoch {epoch:03d}: Val BAcc: {val_bacc:.4f} (best: {best_val_bacc:.4f})  | Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f} ) | Epochs no improve: {epochs_no_improve}/{patience}")
        
        if epochs_no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # ---------------- Test evaluations for both checkpoints ----------------
    
    metrics_best_bacc = _evaluate_model(best_state_bacc)
    metrics_best_loss = _evaluate_model(best_state_loss if best_state_loss is not None else best_state_bacc)
    fold_test_metrics_bacc.append(metrics_best_bacc)
    fold_test_metrics_loss.append(metrics_best_loss)

    if best_state_bacc is not None:
        model.load_state_dict(best_state_bacc)
    train_teacher_outputs = _collect_teacher_outputs(model, train_eval_loader, fold_train_image_ids, device)
    val_teacher_outputs = _collect_teacher_outputs(model, val_loader, fold_val_image_ids, device)
    test_teacher_outputs = _collect_teacher_outputs(model, test_loader, test_patient_image_ids, device)

    train_teacher_outputs.to_pickle(os.path.join(teacher_outputs_dir, f'teacher_outputs_fold_{fold_idx}_train.pkl'))
    val_teacher_outputs.to_pickle(os.path.join(teacher_outputs_dir, f'teacher_outputs_fold_{fold_idx}_val.pkl'))
    test_teacher_outputs.to_pickle(os.path.join(teacher_outputs_dir, f'teacher_outputs_fold_{fold_idx}_test.pkl'))

# %%
# TODO
# for model ...
# save the latent representations of the patches along with node statistics in a dataframe_model
###