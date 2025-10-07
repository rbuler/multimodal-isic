import os
import torch
import yaml
import typing
import neptune
import argparse
import numpy as np
import pandas as pd
import albumentations as A
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

train_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

valid_transform = A.Compose([A.Resize(224,224),
                             A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                             ToTensorV2()])

SPLITS=10
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
kf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=seed)
folds = list(kf.split(df_train_val, df_train_val['dx']))
current_fold = config['training_plan']['parameters']['fold']
train_idx, val_idx = folds[current_fold]
df_train = df_train_val.iloc[train_idx]
df_val = df_train_val.iloc[val_idx]

train_dataset = DermDataset(df_train, radiomics=None, transform=train_transform)
val_dataset = DermDataset(df_val, radiomics=None, transform=valid_transform)
test_dataset = DermDataset(df_test, radiomics=None, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

norm_pix_loss = config['training_plan']['parameters']['norm_pix_loss']

ae_model = convmae_convvit_base_patch16_dec512d8b(norm_pix_loss=norm_pix_loss, with_decoder=False)
ae_model = ae_model.to(device)
root = os.getcwd()
model_name = '8b8fe69df52e48399c371b37e4fef502.pth'
checkpoint_path = os.path.join(root, "models", model_name)
checkpoint = torch.load(checkpoint_path, map_location=device)
ae_model.load_state_dict(checkpoint, strict=False)

# %%
num_epochs=config['training_plan']['parameters']['epochs']
best_val_loss=float('inf')
best_model_state=None


ae_model.eval()
with torch.no_grad():
    for batch in val_loader:
        images = batch['image'].to(device)
        latent, _, ids_restore = ae_model(images, mask_ratio=0)
        break


# %%