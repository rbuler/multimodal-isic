import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from ray import tune


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=76, hidden_dim=128, att_dim=64, dropout=0.5, num_classes=7):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.feature_extractor(x)                # [N, hidden_dim]
        a = torch.softmax(self.attention(h), dim=0)  # [N, 1]
        z = torch.sum(a * h, dim=0)                  # [hidden_dim]
        logits = self.classifier(z)                  # [num_classes]
        probs = torch.softmax(logits, dim=0)
        return probs, a

class PatientDataset(Dataset):
    def __init__(self, patient_features, patient_labels):
        self.features = patient_features
        self.labels = patient_labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_datasets_from_latents(patch_train_df, patch_test_df, drop_background=False):
    patch_train_df = patch_train_df.copy()
    patch_test_df = patch_test_df.copy()
    patch_train_df['patient_id'] = patch_train_df['image_path'].apply(lambda x: os.path.basename(x).split('_')[1].split('.')[0])
    patch_test_df['patient_id'] = patch_test_df['image_path'].apply(lambda x: os.path.basename(x).split('_')[1].split('.')[0])
    if drop_background:
        patch_train_df = patch_train_df[patch_train_df['patch_in_mask'] == 1].reset_index(drop=True)
        patch_test_df = patch_test_df[patch_test_df['patch_in_mask'] == 1].reset_index(drop=True)

    train_patient_features = [
        torch.tensor(np.vstack(g['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_train_df.groupby('patient_id')
    ]
    train_patient_labels = [
        int(g['target'].mode().iat[0])
        for pid, g in patch_train_df.groupby('patient_id')
    ]

    test_patient_features = [
        torch.tensor(np.vstack(g['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_test_df.groupby('patient_id')
    ]
    test_patient_labels = [
        int(g['target'].mode().iat[0])
        for pid, g in patch_test_df.groupby('patient_id')
    ]

    return (train_patient_features, train_patient_labels,
            test_patient_features, test_patient_labels)

def get_data_loaders(train_feats, train_labels, test_feats, test_labels, num_workers=0, pin_memory=False):
    train_dataset = PatientDataset(train_feats, train_labels)
    val_dataset = PatientDataset(test_feats, test_labels)
    patient_labels = np.array(train_labels)
    patient_class_counts = Counter(patient_labels)
    weights = np.array([1.0 / patient_class_counts[int(lbl)] for lbl in patient_labels], dtype=np.float64)
    sample_weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=max(0, num_workers//2), pin_memory=pin_memory)
    return train_loader, val_loader

def train_mil(config, data=None, seed=42, num_classes=7, device_str=None, patience=8, max_epochs=50):
    set_seed(seed)
    torch_threads = int(data.get('torch_threads', 1))
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(max(1, torch_threads//2))
    device = torch.device(device_str if device_str is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

    if 'train_feats' in data and 'train_labels' in data:
        train_feats = [torch.tensor(arr, dtype=torch.float32) for arr in data['train_feats']]
        train_labels = [int(l) for l in data['train_labels']]
        test_feats = [torch.tensor(arr, dtype=torch.float32) for arr in data.get('test_feats', [])]
        test_labels = [int(l) for l in data.get('test_labels', [])]
    else:
        patch_train_df = data['patch_train_df']
        patch_test_df = data['patch_test_df']
        train_feats, train_labels, test_feats, test_labels = build_datasets_from_latents(patch_train_df, patch_test_df)

    train_loader, val_loader = get_data_loaders(train_feats, train_labels, test_feats, test_labels,
                                                num_workers=int(data.get('num_workers', 0)),
                                                pin_memory=bool(data.get('pin_memory', False)))

    input_dim = train_feats[0].shape[1] if len(train_feats) > 0 else data.get('input_dim', 76)

    model = AttentionMIL(input_dim=input_dim,
                         hidden_dim=int(config['hidden_dim']),
                         att_dim=int(config['att_dim']),
                         dropout=float(config['dropout']),
                         num_classes=num_classes).to(device)

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=config.get('weight_decay', 1e-5))
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=config.get('weight_decay', 1e-5))
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config['lr']), momentum=0.9, weight_decay=config.get('weight_decay', 1e-5))
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    criterion = nn.CrossEntropyLoss()
    best_val_bacc = -np.inf
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x[0].to(device)
            y_long = y.to(device).long()
            optimizer.zero_grad()
            probs, _ = model(x)
            loss = criterion(torch.log(probs.unsqueeze(0) + 1e-9), y_long)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x[0].to(device)
                y_long = y.to(device).long()
                probs, _ = model(x)
                y_true.append(int(y_long.item()))
                y_score.append(probs.cpu().numpy())

        if len(y_true) == 0:
            tune.report({'val_bacc': float('nan')})
            return

        y_true = np.array(y_true)
        y_score = np.vstack(y_score)
        y_pred = np.argmax(y_score, axis=1)
        try:
            val_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception:
            val_auc = float('nan')
        val_acc = accuracy_score(y_true, y_pred)
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        tune.report({
            'val_bacc': val_bacc, 'val_acc': val_acc, 'val_auc': val_auc,
            'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1,
            'weighted_precision': weighted_p, 'weighted_recall': weighted_r, 'weighted_f1': weighted_f1
        })

        if val_bacc > best_val_bacc + 1e-6:
            best_val_bacc = val_bacc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break
