import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import torch
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

model_name = '8b8fe69df52e48399c371b37e4fef502.pth'
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
        if patch_id < mask_flat.size and mask_flat[patch_id]:
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
from sklearn.neighbors import NearestNeighbors

X_feat = np.vstack(patch_level_latents_df['patch_latent_pca'].values)
y = patch_level_latents_df['target'].values

k_same = 10
k_other = 10

scores = np.zeros(len(patch_level_latents_df))

for cls in np.unique(y):
    idx_cls = np.where(y == cls)[0]
    idx_other = np.where(y != cls)[0]
    X_cls = X_feat[idx_cls]
    X_other = X_feat[idx_other]

    nbrs_same = NearestNeighbors(n_neighbors=k_same+1).fit(X_cls)
    d_same, _ = nbrs_same.kneighbors(X_cls)
    # ignore self-distance d_same[:,0]
    intra = d_same[:,1:].mean(axis=1)

    nbrs_other = NearestNeighbors(n_neighbors=k_other).fit(X_other)
    d_other, _ = nbrs_other.kneighbors(X_cls)
    inter = d_other.mean(axis=1)

    scores_cls = inter / (intra + 1e-8)
    scores[idx_cls] = scores_cls

patch_level_latents_df['prototype_score'] = scores

# %%
prototypes_idx = []
N_per_class = 200

for cls in np.unique(y):
    idx_cls = np.where(y == cls)[0]
    df_cls = df.iloc[idx_cls].copy()
    df_cls_sorted = df_cls.sort_values('prototype_score', ascending=False)
    selected = df_cls_sorted.head(N_per_class).index.tolist()
    prototypes_idx.extend(selected)

patch_level_latents_df = patch_level_latents_df.loc[prototypes_idx]

# make umap on raw and pca-reduced features
X_raw = np.vstack(patch_level_latents_df["patch_latent"].values)
X_pca = np.vstack(patch_level_latents_df["patch_latent_pca"].values)

reducer_raw = umap.UMAP(random_state=seed, n_neighbors=5, min_dist=0.9)
embedding_raw = reducer_raw.fit_transform(X_raw)
reducer_pca = umap.UMAP(random_state=seed, n_neighbors=5, min_dist=0.9)
embedding_pca = reducer_pca.fit_transform(X_pca)

plt.figure(figsize=(8, 6))
unique_labels = sorted(set(patch_level_latents_df["target"].values))
for i, lab in enumerate(unique_labels):
    mask = patch_level_latents_df["target"].values == lab
    plt.scatter(
        embedding_raw[mask, 0], embedding_raw[mask, 1],
        s=0.05, label=lab
    )
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.title("UMAP Projection of Patch-level Latent Space (Raw Features)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
unique_labels = sorted(set(patch_level_latents_df["target"].values))
for i, lab in enumerate(unique_labels):
    mask = patch_level_latents_df["target"].values == lab
    plt.scatter(
        embedding_pca[mask, 0], embedding_pca[mask, 1],
        s=0.05, label=lab
    )
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.title("UMAP Projection of Patch-level Latent Space (PCA-reduced Features)")
plt.tight_layout()
plt.show()

# %%
# for mutliple combinations of n_neighbors and min_dist
# visualize UMAP projections of patch-level latent space

for n_neighbors in [5, 15, 30, 50]:
    for min_dist in [0.0, 0.1, 0.5]:
        reducer_patches = umap.UMAP(random_state=seed, n_neighbors=n_neighbors, min_dist=min_dist)
        embedding_patches = reducer_patches.fit_transform(X_pca)

        plt.figure(figsize=(8, 6))
        unique_labels = sorted(set(patch_level_latents_df["target"].values))
        for i, lab in enumerate(unique_labels):
            mask = patch_level_latents_df["target"].values == lab
            plt.scatter(
                embedding_patches[mask, 0], embedding_patches[mask, 1],
                s=5, label=lab
            )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.title(f"UMAP Projection of Patch-level Latent Space (n_neighbors={n_neighbors}, min_dist={min_dist})")
        plt.tight_layout()
        plt.show()

# %%
model_names = ['75f15b3dee2a4fe1ad97b7a0e454cf1a.pth',
               '403ae2156c114b98b54d46ecf2594cb6.pth',
               'd6c5a23936e843fbb22538de8817bf76.pth']
for model_name in model_names:
    ae_model = convmae_convvit_base_patch16_dec512d8b(with_decoder=False)
    ae_model = ae_model.to(device)
    checkpoint_path = os.path.join(root, "models", model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ae_model.load_state_dict(checkpoint, strict=False)
    ae_model.eval()
    latent_pooled = []
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
        
        latent_pooled = pd.concat(latent_pooled, ignore_index=True) if len(latent_pooled) > 0 else pd.DataFrame()

        X_max = np.vstack(latent_pooled["latent_pooled_max"].values)
        X_mean = np.vstack(latent_pooled["latent_pooled_mean"].values)
        labels = latent_pooled["target"].values.squeeze()

        reducer_max = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.1)
        embedding_max = reducer_max.fit_transform(X_max)
        reducer_mean = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.1)
        embedding_mean = reducer_mean.fit_transform(X_mean)

        okabe_ito = [
            "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00",
            "#CC79A7", "#999999"
        ]

        plt.figure(figsize=(8, 6))
        unique_labels = sorted(set(labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            plt.scatter(
                embedding_max[mask, 0], embedding_max[mask, 1],
                s=5, color=okabe_ito[i % len(okabe_ito)], label=lab
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.title("UMAP Projection of Latent Space - Max Pooling")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        unique_labels = sorted(set(labels))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            plt.scatter(
                embedding_mean[mask, 0], embedding_mean[mask, 1],
                s=5, color=okabe_ito[i % len(okabe_ito)], label=lab
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.title("UMAP Projection of Latent Space - Mean Pooling")
        plt.tight_layout()
        plt.show()

# %%
# output_path = os.path.join(root, "patient_latent_space_data.pkl")
# latent_pooled.to_pickle(output_path)
# print(f"Latent space data saved to {output_path}")
# %%

