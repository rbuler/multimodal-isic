# %%
import os
import torch
import yaml
import typing
import neptune
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from dataset import DermDataset
from torch.utils.data import DataLoader
# from autoencoder import ConvAutoencoder
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import io
import sys
if not hasattr(np, 'float'):
    np.float = float
sys.path.append("ConvMAE")
from ConvMAE.models_convmae import convmae_convvit_base_patch16_dec512d8b


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

if config["neptune"]:
    run = neptune.init_run(project="ProjektMMG/multimodal-isic",)
    run["config"] = config

device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
df_train_val = pd.read_pickle(config['dir']['df'])
df_test = pd.read_pickle(config['dir']['df_test'])

train_transform = A.Compose([A.Resize(224,224), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                             A.RandomRotate90(p=0.5), A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                             A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                             A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
                            #  A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                             A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
                             ToTensorV2()])

valid_transform = A.Compose([A.Resize(224,224),
                            #  A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                             A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


ae_model = convmae_convvit_base_patch16_dec512d8b()
# ae_model = ae_model.to(device)
root = os.getcwd()
checkpoint_path = os.path.join(root, "ConvMAE", "checkpoint.pth")
# checkpoint = torch.load(checkpoint_path, map_location=device)
# ae_model.load_state_dict(checkpoint, strict=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=config['training_plan']['parameters']['lr'])

# optimizer = torch.optim.Adam([
#     {"params": ae_model.encoder.parameters(), "lr": 1e-5},
#     {"params": ae_model.decoder.parameters(), "lr": 1e-4}
# ], weight_decay=1e-5)

# %%
num_epochs=config['training_plan']['parameters']['epochs']
best_val_loss=float('inf')

for epoch in range(num_epochs):
    ae_model.train()
    running_loss=0.0
    for batch in train_loader:
        images = batch['image'].to(device)
        optimizer.zero_grad()
        loss, pred, mask = ae_model(images)
        # loss = criterion(recon, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*images.size(0)
    train_loss = running_loss/len(train_loader.dataset)
    
    ae_model.eval()
    running_loss=0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            loss, _, _ = ae_model(images)
            # loss = criterion(recon, images)
            running_loss += loss.item()*images.size(0)
    val_loss = running_loss/len(val_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if config["neptune"]:
        run["train/loss"].append(train_loss)
        run["val/loss"].append(val_loss)

        ae_model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            for i in range(4):  # Save 4 pairs of image-reconstruction
                img = batch['image'][i:i+1].to(device)
                loss, pred, _ = ae_model(img)
                if hasattr(ae_model, 'unpatchify'):
                    recon = ae_model.unpatchify(pred)
                else:
                    recon = pred
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)
                img_vis = img * std + mean
                recon_vis = recon * std + mean
                img_vis = img_vis.squeeze().cpu().numpy().transpose(1,2,0)
                recon_vis = recon_vis.squeeze().cpu().numpy().transpose(1,2,0)
                img_vis = np.clip(img_vis, 0, 1)
                recon_vis = np.clip(recon_vis, 0, 1)

                fig, axs = plt.subplots(1,2, figsize=(8,4))
                axs[0].imshow(img_vis)
                axs[0].set_title("Original")
                axs[0].axis('off')
                axs[1].imshow(recon_vis)
                axs[1].set_title("Reconstruction")
                axs[1].axis('off')
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                run[f"visuals/image_comparison_{i+1}"].append(neptune.types.File.from_content(buf.getvalue(), extension='png'))
                plt.close(fig)


    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(ae_model.state_dict(), f'best_ae_model_{device}.pth')
        print("Saved Best Model")

# %%