import os
import torch
import yaml
import typing
import neptune
import umap
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

train_val_loader = DataLoader(train_val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

ae_model = convmae_convvit_base_patch16_dec512d8b(with_decoder=False)
ae_model = ae_model.to(device)
root = os.getcwd()

model_name = '8b8fe69df52e48399c371b37e4fef502.pth'
checkpoint_path = os.path.join(root, "models", model_name)
checkpoint = torch.load(checkpoint_path, map_location=device)
ae_model.load_state_dict(checkpoint, strict=False)
# %%
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

        raw_df = pd.DataFrame({
            'image_path': image_path,
            'segmentation_path': segmentation_path,
            'target': batch['target'].numpy(),
            'latent': list(latent.cpu().numpy()),
            'ids_restore': list(ids_restore.cpu().numpy())
        })
        latent_raw.append(raw_df)

latent_pooled = pd.concat(latent_pooled, ignore_index=True) if len(latent_pooled) > 0 else pd.DataFrame()
latent_raw = pd.concat(latent_raw, ignore_index=True) if len(latent_raw) > 0 else pd.DataFrame()

# %%
X_max = np.vstack(latent_pooled["latent_pooled_max"].values)
X_mean = np.vstack(latent_pooled["latent_pooled_mean"].values)
labels = latent_pooled["target"].values.squeeze()

reducer_max = umap.UMAP(random_state=seed)
embedding_max = reducer_max.fit_transform(X_max)
reducer_mean = umap.UMAP(random_state=seed)
embedding_mean = reducer_mean.fit_transform(X_mean)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sc0 = axes[0].scatter(embedding_max[:, 0], embedding_max[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
axes[0].set_title("UMAP - pooled max")
axes[0].axis("off")

sc1 = axes[1].scatter(embedding_mean[:, 0], embedding_mean[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
axes[1].set_title("UMAP - pooled mean")
axes[1].axis("off")

fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
fig.colorbar(sc0, cax=cax)
plt.tight_layout(rect=[0, 0, 0.85, 1])

plt_path = os.path.join(root, "umap_latent.png")
plt.savefig(plt_path, dpi=150)
print(f"Saved UMAP plot to {plt_path}")

# %%
output_path = os.path.join(root, "patient_latent_space_data.pkl")
latent_pooled.to_pickle(output_path)
print(f"Latent space data saved to {output_path}")
# %%