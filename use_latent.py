# %% IMPORTS
import os
import yaml
import torch
import typing
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import Counter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score
from save_latent import extract_latents

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


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=path,
                        help=help)
    return parser

parser = get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
# %%

extract = False
if extract:
    model_names = [
    '76af6437f4424b3c95b5813c6afdaffe.pth',  # 25T     0.7692 bacc
    '30fde206242a4aebbc1d7c587d7efc75.pth',  # 50T     0.7917 bacc  0.7908 acc
    '35dc94daefc349a795202050bd226fe5.pth',  # 75T     0.7709 bacc
    '159d1cd1d724456aaedc80493cdfe2ea.pth',  # 25F     0.7430 bacc
    'fe8cdf6db4b14a51b4418ff746bb851d.pth',  # 50F     0.7816 bacc  0.7574 acc
    'ce4069521dfb4264a3ac8cc3d59971a2.pth',  # 75F     0.7877 bacc  0.7654 acc  
    ]
    
    model_name= model_names[1]  # choose model

    (patch_level_latents_df_train,
     patch_level_latents_df_test,
     latent_pooled_train,
     latent_pooled_test,
     latent_raw_train,
     latent_raw_test) = extract_latents(config, path=model_name, remove_background=False)

else:
    # LOAD DataFrame FROM pickle
    patch_level_latents_df_train = pd.read_pickle("dataframes_latents/patch_level_latents_train_df.pkl")
    patch_level_latents_df_test = pd.read_pickle("dataframes_latents/patch_level_latents_test_df.pkl")

patch_level_latents_df_train['patient_id'] = patch_level_latents_df_train['image_path'].apply(lambda x: os.path.basename(x).split('_')[1].split('.')[0])
patch_level_latents_df_test['patient_id'] = patch_level_latents_df_test['image_path'].apply(lambda x: os.path.basename(x).split('_')[1].split('.')[0])

drop_background = False  # only keep patches that overlap with lesion mask
if drop_background:
    patch_level_latents_df_train = patch_level_latents_df_train[patch_level_latents_df_train['patch_in_mask'] == 1].reset_index(drop=True)
    patch_level_latents_df_test = patch_level_latents_df_test[patch_level_latents_df_test['patch_in_mask'] == 1].reset_index(drop=True)

print(f"Number of patches in training set: {len(patch_level_latents_df_train)}")
for c in patch_level_latents_df_train['target'].unique():
    n_c = len(patch_level_latents_df_train[patch_level_latents_df_train['target'] == c])
    print(f"  Class {c}: {n_c} patches")

# %%

# --- TRAIN ---
train_patient_features = [
    torch.tensor(np.vstack(g['patch_latent'].values), dtype=torch.float32)
    for pid, g in patch_level_latents_df_train.groupby('patient_id')
]
train_patient_labels = [
    int(g['target'].mode().iat[0])
    for pid, g in patch_level_latents_df_train.groupby('patient_id')
]

# --- TEST ---
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
print("Patient class counts:", patient_class_counts)

weights = np.array([1.0 / patient_class_counts[int(lbl)] for lbl in patient_labels], dtype=np.float64)
sample_weights = torch.from_numpy(weights)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttentionMIL(input_dim=train_dataset[0][0].shape[1], hidden_dim=256, att_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()


num_epochs = 100
accuracies = []
baccuracies = []
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_val_loss = 0.0
    for x, y in train_loader:
        # x: list/tensor of instances for the patient (batch_size==1)
        x = x[0].to(device)               # [n_instances, input_dim]
        y_long = y.to(device).long()      # scalar label as LongTensor

        optimizer.zero_grad()
        probs, att = model(x)             # probs: [num_classes]
        logp = torch.log(probs + 1e-9).unsqueeze(0)   # [1, num_classes]
        loss = criterion(logp, y_long)             # expects [N, C] and [N]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # validation
    model.eval()
    y_true = []
    y_score = []   # list of probability vectors per sample
    with torch.no_grad():
        for x, y in val_loader:
            x = x[0].to(device)
            y_long = y.to(device).long()
            probs, att = model(x)        # probs: [num_classes]
            y_true.append(int(y_long.item()))
            y_score.append(probs.cpu().numpy())
            val_loss = criterion(torch.log(probs.unsqueeze(0) + 1e-9), y_long)
            total_val_loss += val_loss.item()
    y_true = np.array(y_true)
    y_score = np.vstack(y_score)        # shape (n_samples, num_classes)

    try:
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
    except Exception:
        auc = float('nan')

    y_pred_labels = np.argmax(y_score, axis=1)
    acc = accuracy_score(y_true, y_pred_labels)
    bacc = balanced_accuracy_score(y_true, y_pred_labels)
    accuracies.append(acc)
    baccuracies.append(bacc)
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_loss={avg_val_loss:.4f}, AUC={auc:.3f}, ACC={acc:.3f}, BACC={bacc:.3f}")
plt.plot(accuracies)
plt.plot(baccuracies)
plt.xlabel('Validation Samples')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Samples')
plt.legend(['Accuracy', 'Balanced Accuracy'])
plt.show()