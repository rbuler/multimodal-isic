# %%
import os
import torch
import yaml
import uuid
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
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import io
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
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3),
    A.GaussianBlur(blur_limit=(3,7), sigma_limit=0.1, p=0.2),
    A.GaussNoise(std_range=(0.03, 0.03), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1.0, p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

valid_transform = A.Compose([A.Resize(224,224),
                             A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                            #  A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
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

train_loader = DataLoader(train_dataset, batch_size=config['training_plan']['parameters']['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

ae_model = convmae_convvit_base_patch16_dec512d8b()
ae_model = ae_model.to(device)
root = os.getcwd()
checkpoint_path = os.path.join(root, "ConvMAE", "checkpoint.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
ae_model.load_state_dict(checkpoint['model'], strict=False)
# optimizer = optim.Adam(ae_model.parameters(), lr=config['training_plan']['parameters']['lr'])
# optimizer = optim.AdamW(ae_model.parameters(), lr=config['training_plan']['parameters']['lr'], betas=(0.9, 0.95), weight_decay=0.05)

encoder_params = [param for name, param in ae_model.named_parameters() if 'decoder' not in name]
decoder_params = [param for name, param in ae_model.named_parameters() if 'decoder' in name]
encoder_lr = 1e-5
decoder_lr = 1e-3
optimizer = torch.optim.AdamW([
    {'params': encoder_params, 'lr': encoder_lr},
    {'params': decoder_params, 'lr': decoder_lr}
], betas=(0.9, 0.95), weight_decay=0.05)

# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
# scheduler.step()
scheduler = None

mask_ratio = 0.75
include_lesion_mask = True

# %%
num_epochs=config['training_plan']['parameters']['epochs']
best_val_loss=float('inf')
best_model_state=None


for epoch in range(num_epochs):
    ae_model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        
        images = batch['image'].to(device)
        lesion_mask = batch['mask'].to(device).unsqueeze(1) if include_lesion_mask else None

        loss, pred, mask = ae_model(images, mask_ratio=mask_ratio, lesion_mask=lesion_mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    
    ae_model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            loss, _, _ = ae_model(images, mask_ratio=mask_ratio)
            running_loss += loss.item() * images.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    
    if scheduler is not None:
        scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if config["neptune"]:
        run["train/loss"].append(train_loss)
        run["val/loss"].append(val_loss)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                batch = next(iter(val_loader))
                for i in range(4):
                    img = batch['image'][i:i+1].to(device)
                    loss, pred, mask = ae_model(img, mask_ratio=mask_ratio)
                    recon = ae_model.unpatchify(pred) if hasattr(ae_model, 'unpatchify') else pred
                    img_vis = img.squeeze().cpu().numpy().transpose(1, 2, 0)
                    recon_vis = recon.squeeze().cpu().numpy().transpose(1, 2, 0)
                    mask_vis = mask.cpu().numpy()
                    
                    image_patches = ae_model.patchify(img).cpu().numpy() if hasattr(ae_model, 'patchify') else img.cpu().numpy()
                    mask_expanded = mask_vis[..., None]
                    unmasked_patches = image_patches * (1 - mask_expanded)
                    binary_patches = mask_expanded * np.ones_like(image_patches)
                    binary_image = ae_model.unpatchify(torch.tensor(binary_patches, device=device)).cpu().numpy()

                    
                    img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    recon_vis = recon_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    
                    binary_image_vis = binary_image.squeeze().transpose(1, 2, 0)
                    overlay_vis = recon_vis * binary_image_vis + img_vis * (1 - binary_image_vis)
                
                    img_vis = np.clip(img_vis, 0, 1)
                    binary_image_vis = np.clip(binary_image_vis, 0, 1)
                    recon_vis = np.clip(recon_vis, 0, 1)
                    overlay_vis = np.clip(overlay_vis, 0, 1)

                    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                    axs[0].imshow(img_vis)
                    axs[0].set_title("Original")
                    axs[0].axis('off')
                    axs[1].imshow(binary_image_vis, cmap='gray')
                    axs[1].set_title("Mask")
                    axs[1].axis('off')
                    axs[2].imshow(recon_vis)
                    axs[2].set_title("Reconstruction")
                    axs[2].axis('off')
                    axs[3].imshow(overlay_vis)
                    axs[3].set_title("Overlay")
                    axs[3].axis('off')

                    plt.tight_layout()
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    run[f"visuals/image_comparison_{i+1}"].append(neptune.types.File.from_content(buf.getvalue(), extension='png'))
                    plt.close(fig)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = ae_model.state_dict()

    if epoch == num_epochs - 1:
        model_dir = os.path.join(root, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{uuid.uuid4().hex}.pth")
        torch.save(best_model_state, model_path)
        print(f"Saved Best Model at {model_path}")

# %%





# after training forward pass thorugh encoder only with mask_ratio == 0
# ae.model.eval()
# with torch.no_grad():
#     # Pass mask_ratio=0 to use the full image
#     latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio=0.0)