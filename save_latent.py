import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import yaml
import typing
import neptune
import umap
import umap.plot as umap_plot
from bokeh.io import output_file, save
import argparse
import numpy as np
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
from dataset import DermDataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

import sys
if not hasattr(np, 'float'):
    np.float = float
sys.path.append("ConvMAE")
from ConvMAE.models_convmae import convmae_convvit_base_patch16_dec512d8b

root = os.getcwd()


def lr_lambda(current_epoch: int):
    base_lr = config['training_plan']['parameters']['lr']
    warmup_epochs = config['training_plan']['parameters']['warmup_epochs']
    total_epochs = config['training_plan']['parameters']['epochs']
    if current_epoch < warmup_epochs:
        return base_lr * (float(current_epoch + 1) / float(max(1, warmup_epochs)))
    else:
        progress = float(current_epoch + 1 - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return base_lr * 0.5 * (1. + np.cos(np.pi * progress))


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=path,
                        help=help)
    return parser

# %%
parser = get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

if config["neptune"]:
    run = neptune.init_run(project="ProjektMMG/multimodal-isic",)
    run["config"] = config

device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
df_train_val = pd.read_pickle(config['dir']['df'])
df_test = pd.read_pickle(config['dir']['df_test'])

transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)

train_val_dataset = DermDataset(df_train_val, radiomics=None, transform=transform)
test_dataset = DermDataset(df_test, radiomics=None, transform=transform)

train_val_loader = DataLoader(train_val_dataset, batch_size=1000, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# %%
ae_model = convmae_convvit_base_patch16_dec512d8b(with_decoder=False)
ae_model = ae_model.to(device)

model_names = [
'76af6437f4424b3c95b5813c6afdaffe.pth',  # 25T     0.7692 bacc
'30fde206242a4aebbc1d7c587d7efc75.pth',  # 50T     0.7917 bacc  0.7908 acc
'35dc94daefc349a795202050bd226fe5.pth',  # 75T     0.7709 bacc
'159d1cd1d724456aaedc80493cdfe2ea.pth',  # 25F     0.7430 bacc
'fe8cdf6db4b14a51b4418ff746bb851d.pth',  # 50F     0.7816 bacc  0.7574 acc
'ce4069521dfb4264a3ac8cc3d59971a2.pth',  # 75F     0.7877 bacc  0.7654 acc  
]
# model_name = '8b8fe69df52e48399c371b37e4fef502.pth'
model_name = model_names[1]

checkpoint_path = os.path.join(root, "models", model_name)
checkpoint = torch.load(checkpoint_path, map_location=device)
ae_model.load_state_dict(checkpoint, strict=False)

ae_model.eval()
latent_pooled = []
latent_raw = []

with torch.no_grad():
    for batch in train_val_loader:
        images = batch['image'].to(device)
        image_path, segmentation_path = batch['image_path'], batch['segmentation_path']
        latent, _, ids_restore = ae_model(images, mask_ratio=0)

        latent_pooled_max = torch.max(latent, dim=1).values
        latent_pooled_mean = torch.mean(latent, dim=1)

        pooled_df = pd.DataFrame({
            'image_path': image_path,
            'segmentation_path': segmentation_path,
            'target': batch['target'].numpy(),
            'latent_pooled_max': list(latent_pooled_max.cpu().numpy()),
            'latent_pooled_mean': list(latent_pooled_mean.cpu().numpy()),
            'ids_restore': list(ids_restore.cpu().numpy())
        })
        latent_pooled.append(pooled_df)

        # skin lesion mask (supports (B, H, W) or (B, 1, H, W))
        mask = batch['mask'].to(device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B,1,H,W)

        # divide image into 16x16 patches and find which patches overlap with lesion mask
        B, _, H, W = mask.shape
        patch_size = 16
        mask_patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # (B,1,H/16,W/16,16,16)
        mask_patches = mask_patches.contiguous().view(B, 1, H // patch_size, W // patch_size, -1)  # (B,1,H/16,W/16,256)
        mask_patches = mask_patches.sum(dim=-1)  # (B,1,H/16,W/16)
        mask_patches = (mask_patches > 0).squeeze(1)  # (B,H/16,W/16), bool tensor indicating if patch overlaps with lesion

        raw_df = pd.DataFrame({
            'image_path': image_path,
            'segmentation_path': segmentation_path,
            'target': batch['target'].numpy(),
            'latent': list(latent.cpu().numpy()),
            'ids_restore': list(ids_restore.cpu().numpy()),
            'lesion_mask_patches': list(mask_patches.cpu().numpy())
        })
        latent_raw.append(raw_df)

latent_pooled = pd.concat(latent_pooled, ignore_index=True) if len(latent_pooled) > 0 else pd.DataFrame()
latent_raw = pd.concat(latent_raw, ignore_index=True) if len(latent_raw) > 0 else pd.DataFrame()
# %%
i = 0
remove = True  # only keep patches that overlap with lesion mask
patch_level_latents = []

for idx, row in latent_raw.iterrows():
    image_path = row['image_path']
    segmentation_path = row['segmentation_path']
    target = row['target']
    latent = row['latent']  # shape (196, 768)
    ids_restore = row['ids_restore']  # shape (196,)
    mask_patches = row['lesion_mask_patches']  # shape (196,)

    mask_flat = np.asarray(mask_patches).ravel()  # (num_patches,)
    for patch_idx in range(len(latent)):
        patch_latent = np.asarray(latent[patch_idx])  # (768,)
        patch_id = int(ids_restore[patch_idx])
        
        if remove:
            if patch_id < mask_flat.size and mask_flat[patch_id]:
                patch_level_latents.append({
                    'image_path': image_path,
                    'segmentation_path': segmentation_path,
                    'target': target,
                    'patch_id': patch_id,
                    'patch_latent': patch_latent
                })
                i += 1
        else:
            patch_level_latents.append({
                'image_path': image_path,
                'segmentation_path': segmentation_path,
                'target': target,
                'patch_id': patch_id,
                'patch_latent': patch_latent
            })
            i += 1
print(f"Total lesion-overlapping patches: {i}")
patch_level_latents_df = pd.DataFrame(patch_level_latents)
# %%
X_patches = np.vstack(patch_level_latents_df["patch_latent"].values)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_patches)
pca = PCA(n_components=0.90, whiten=True)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA reduced dimensions from {X_patches.shape[1]} to {X_pca.shape[1]}")
patch_level_latents_df['patch_latent_pca'] = list(X_pca)

# %%

X_feat = np.vstack(patch_level_latents_df['patch_latent_pca'].values)
y = patch_level_latents_df['target'].values

k_same = 5
k_other = 10

scores = np.zeros(len(patch_level_latents_df))

for cls in np.unique(y):
    idx_cls = np.where(y == cls)[0]
    idx_other = np.where(y != cls)[0]
    X_cls = X_feat[idx_cls]
    X_other = X_feat[idx_other]

    nbrs_same = NearestNeighbors(n_neighbors=k_same+1).fit(X_cls)
    d_same, _ = nbrs_same.kneighbors(X_cls)
    intra = d_same[:,1:].mean(axis=1)

    nbrs_other = NearestNeighbors(n_neighbors=k_other).fit(X_other)
    d_other, _ = nbrs_other.kneighbors(X_cls)
    inter = d_other.mean(axis=1)

    scores_cls = inter / (intra + 1e-8)
    scores[idx_cls] = scores_cls

patch_level_latents_df['proto_score'] = scores

print("Done computing prototype scores.")

prototypes_idx = []
alpha = 0.5
proto_score_min = 1.05
for cls in np.unique(y):
    cls_idx = patch_level_latents_df.index[patch_level_latents_df['target'] == cls].tolist()
    cls_scores = patch_level_latents_df.loc[cls_idx, 'proto_score']
    mu_cls = cls_scores.mean()
    sigma_cls = cls_scores.std()
    proto_score_thresh = mu_cls + alpha * sigma_cls
    final_thresh = max(proto_score_min, proto_score_thresh)
    print(f"Class {cls}: Prototype score threshold: {final_thresh:.4f}")

    cls_prototypes = cls_scores[cls_scores >= final_thresh].index.tolist()
    prototypes_idx.extend(cls_prototypes)

patch_level_latents_df = patch_level_latents_df.loc[prototypes_idx]

print(patch_level_latents_df['target'].value_counts())
print(len(patch_level_latents_df['image_path'].unique()))
print(patch_level_latents_df.groupby('target')['image_path'].nunique())

# %% apply self-organizing map (SOM) to cluster patch-level latents
# with 100 by 100 grid
from minisom import MiniSom
from tqdm import trange


# TODO change to SOMPY 
# use batch training for faster convergence
# set n_jobs=-1 for parallel processing if possible

seed = 42
np.random.seed(seed)

X_raw = np.vstack(patch_level_latents_df["patch_latent"].values)
X_pca = np.vstack(patch_level_latents_df["patch_latent_pca"].values)
Y = patch_level_latents_df["target"].values

X = X_pca

# take 1% of data for faster training
# sample_size = max(1, X.shape[0] // 100)
# sample_indices = np.random.choice(X.shape[0], size=sample_size, replace=False)
# X = X[sample_indices]
# Y = Y[sample_indices]

som_size = (100, 100)
som = MiniSom(som_size[0], som_size[1], X.shape[1], sigma=5, learning_rate=0.5, random_seed=seed)
som.random_weights_init(X)

n_epochs = 50
plot_every = 1  # plot every n epochs

def plot_som_progress(som, X, y, epoch):
    w_x, w_y = zip(*[som.winner(d) for d in X])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    plt.figure(figsize=(10, 9))
    plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.2)
    plt.colorbar()

    for c in np.unique(y):
        idx_target = y==c
        plt.scatter(w_x[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8,
                    w_y[idx_target]+.5+(np.random.rand(np.sum(idx_target))-.5)*.8, 
                    s=50, label=c)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    
for epoch in trange(1, n_epochs + 1):
    som.train_random(X, len(X) // 2)  # train on half the data per epoch

    if epoch % plot_every == 0 or epoch == 1:
        plot_som_progress(som, X, Y, epoch)

# %%
# TODO done
# UMAP on patch-level latents for both raw and PCA-reduced features

# X_raw = np.vstack(patch_level_latents_df["patch_latent"].values)
# X_pca = np.vstack(patch_level_latents_df["patch_latent_pca"].values)

# reducer_raw = umap.UMAP(random_state=seed, n_neighbors=50, min_dist=0.4)
# embedding_raw = reducer_raw.fit_transform(X_raw)
# reducer_pca = umap.UMAP(random_state=seed, n_neighbors=50, min_dist=0.4)
# embedding_pca = reducer_pca.fit_transform(X_pca)

# plt.figure(figsize=(8, 6))
# unique_labels = sorted(set(patch_level_latents_df["target"].values))
# for i, lab in enumerate(unique_labels):
#     mask = patch_level_latents_df["target"].values == lab
#     plt.scatter(
#         embedding_raw[mask, 0], embedding_raw[mask, 1],
#         s=0.1, label=lab
#     )
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
# plt.title("UMAP Projection of Patch-level Latent Space (Raw Features)")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(8, 6))
# unique_labels = sorted(set(patch_level_latents_df["target"].values))
# for i, lab in enumerate(unique_labels):
#     mask = patch_level_latents_df["target"].values == lab
#     plt.scatter(
#         embedding_pca[mask, 0], embedding_pca[mask, 1],
#         s=0.1, label=lab
#     )
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
# plt.title("UMAP Projection of Patch-level Latent Space (PCA-reduced Features)")
# plt.tight_layout()
# plt.show()

# %%
# TODO done
# train MIL model on patch-level latents

# grouped = patch_level_latents_df.groupby(["image_path", "target"])["patch_latent"]
# x_patient = []
# y_patient = []
# for (image_path, target), patches in grouped:
#     patches_array = np.vstack(patches.values)  # shape (num_patches, latent_dim)
#     x_patient.append(patches_array)
#     y_patient.append(target)

# x_patient = np.array(x_patient, dtype=float)  # shape (num_patients,)
# y_patient = np.array(y_patient)  # shape (num_patients,)

# class MILModel(nn.Module):
#     def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.25):
#         super(MILModel, self).__init__()
#         # gated attention components
#         self.att_V = nn.Linear(input_dim, hidden_dim)
#         self.att_U = nn.Linear(input_dim, hidden_dim)
#         self.att_w = nn.Linear(hidden_dim, 1)

#         # classifier operating on the aggregated bag representation
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
#         Vx = torch.tanh(self.att_V(x))       # (B, N, H)
#         Ux = torch.sigmoid(self.att_U(x))    # (B, N, H)
#         H = Vx * Ux                          # gated activation (B, N, H)
#         att_logits = self.att_w(H).squeeze(-1)   # (B, N)
#         att_weights = torch.softmax(att_logits, dim=1).unsqueeze(-1)  # (B, N, 1)
#         z = torch.sum(att_weights * x, dim=1)    # (B, D)
#         logits = self.classifier(z)              # (B, num_classes)
#         return logits

# mil_model = MILModel(input_dim=x_patient.shape[2], num_classes=len(np.unique(y_patient)))
# mil_model = mil_model.to(device)
# num_epochs = 1000

# # reset seed for reproducibility
# np.random.seed(seed)
# torch.manual_seed(seed)
# x_train, x_val, y_train, y_val = train_test_split(x_patient, y_patient, test_size=0.2, random_state=seed, stratify=y_patient)
# class_counts = np.bincount(y_train)
# class_weights = 1.0 / (class_counts + 1e-6)
# class_weights = class_weights / class_weights.sum() * len(class_counts)
# class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
# criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
# optimizer = torch.optim.Adam(mil_model.parameters(), lr=0.001)


# train_losses = []
# val_losses = []
# train_accuracies = []
# val_accuracies = []
# for epoch in range(num_epochs):
#     mil_model.train()
#     optimizer.zero_grad()
#     outputs = mil_model(torch.tensor(x_train, dtype=torch.float32).to(device))
#     loss = criterion(outputs, torch.tensor(y_train, dtype=torch.long).to(device))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
#     train_losses.append(loss.item())
#     _, predicted = torch.max(outputs.data, 1)
#     correct = (predicted.cpu().numpy() == y_train).astype(int)
#     train_acc = balanced_accuracy_score(y_train, predicted.cpu().numpy())
#     train_accuracies.append(train_acc)


#     mil_model.eval()
#     with torch.no_grad():
#         val_outputs = mil_model(torch.tensor(x_val, dtype=torch.float32).to(device))
#         val_loss = criterion(val_outputs, torch.tensor(y_val, dtype=torch.long).to(device))
#         val_losses.append(val_loss.item())
#         _, val_predicted = torch.max(val_outputs.data, 1)
#         val_correct = (val_predicted.cpu().numpy() == y_val).astype(int)
#         # log two accuracies: overall and balanced
#         val_acc = val_correct.sum() / len(y_val)        
#         val_bacc = balanced_accuracy_score(y_val, val_predicted.cpu().numpy())
#         val_accuracies.append(val_bacc)
#         if val_bacc >= max(val_accuracies):
#             print(f"Validation BAcc: {val_bacc:.4f}, Validation Acc: {val_acc:.4f}")
            
# # plot train val loss curves
# plt.figure()
# plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
# plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Train and Validation Loss Curves')
# plt.legend()
# plt.show()

# # plot train val balanced accuracy curves
# plt.figure()
# plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Balanced Accuracy')
# plt.plot(range(1, num_epochs+1), val_accuracies, label='Val Balanced Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Balanced Accuracy')
# plt.title('Train and Validation Balanced Accuracy')
# plt.legend()
# plt.show()