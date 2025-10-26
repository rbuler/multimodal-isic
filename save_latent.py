import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import torch
import torch.nn as nn
import yaml
import typing
import neptune
from tqdm import trange
from tqdm import tqdm

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
# save df to pickle
save = True
if save:
    patch_level_latents_df.to_pickle("patch_level_latents_df.pkl")
# %%
print("Finished saving patch-level latents.")