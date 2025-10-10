import os
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

# %%
n_neighbors = [2, 5, 10, 15, 25, 50, 100, 200]
min_dist = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
param_grid = [(n, d) for n in n_neighbors for d in min_dist]
save_path = '/users/project1/pt01191/MMODAL_ISIC/Data/umap_figures'

for n, d in param_grid:
    reducer_max = umap.UMAP(random_state=seed, n_neighbors=n, min_dist=d)
    embedding_max = reducer_max.fit_transform(X_max)
    hover_data = pd.DataFrame({
        "image": latent_pooled["image_path"].apply(lambda x: x.split('/')[-1]),
        "target": latent_pooled["target"]
    })
    p_max = umap_plot.interactive(reducer_max, labels=labels, hover_data=hover_data, point_size=2)
    output_file_path_max = os.path.join(save_path, f"umap_max_n{n}_d{d}.html")
    output_file(output_file_path_max)
    save(p_max, filename=output_file_path_max)
    print(f"Saved UMAP max plot to {output_file_path_max}")

    reducer_mean = umap.UMAP(random_state=seed, n_neighbors=n, min_dist=d)
    embedding_mean = reducer_mean.fit_transform(X_mean)
    p_mean = umap_plot.interactive(reducer_mean, labels=labels, hover_data=hover_data, point_size=2)
    output_file_path_mean = os.path.join(save_path, f"umap_mean_n{n}_d{d}.html")
    output_file(output_file_path_mean)
    save(p_mean, filename=output_file_path_mean)
    print(f"Saved UMAP mean plot to {output_file_path_mean}")
# %%
# experiment with a specific set of parameters 
SPLITS=10
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
kf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=seed)
folds = list(kf.split(df_train_val, df_train_val['dx']))
current_fold = config['training_plan']['parameters']['fold']
train_idx, val_idx = folds[current_fold]
# set labels to 0 and 1, for training and validation set respectively
labels = np.zeros(len(df_train_val))
labels[val_idx] = 1


reducer_max = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.)
embedding_max = reducer_max.fit_transform(X_max)
reducer_mean = umap.UMAP(random_state=seed, n_neighbors=15, min_dist=0.)
embedding_mean = reducer_mean.fit_transform(X_mean)

umap_plot.output_notebook()
hover_data = pd.DataFrame({
    "image": latent_pooled["image_path"].apply(lambda x: x.split('/')[-1]),
    "target": latent_pooled["target"]
})

p_max = umap_plot.interactive(reducer_max, labels=labels, hover_data=hover_data, point_size=2, theme='fire')
umap_plot.show(p_max)

p_mean = umap_plot.interactive(reducer_mean, labels=labels, hover_data=hover_data, point_size=2, theme='fire')
umap_plot.show(p_mean)

# %%
# output_path = os.path.join(root, "patient_latent_space_data.pkl")
# latent_pooled.to_pickle(output_path)
# print(f"Latent space data saved to {output_path}")
# %%

