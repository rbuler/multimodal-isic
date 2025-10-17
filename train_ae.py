# %%
import os
import io
import sys
import copy
import torch
import yaml
import uuid
import neptune
import numpy as np
import pandas as pd
import albumentations as A
from dataset import DermDataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from utils import concat_patch_moments, get_args_parser
from utils import visualize_latent_space, visualize_model_outputs

if not hasattr(np, 'float'):
    np.float = float
sys.path.append("ConvMAE")
from ConvMAE.models_convmae import convmae_convvit_base_patch16_dec512d8b

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

# %%
use_isic2019 = True
if use_isic2019:
    path_isic2019_csv = "/users/project1/pt01191/MMODAL_ISIC/Data/ISIC2019/ISIC_2019_Training_Metadata.csv"
    path_isic2019 = "/users/project1/pt01191/MMODAL_ISIC/Data/ISIC2019/images/ISIC_2019_Training_Input"
    path_isic_gt = "/users/project1/pt01191/MMODAL_ISIC/Data/ISIC2019/ISIC_2019_Training_GroundTruth.csv"
    df_isic2019 = pd.read_csv(path_isic2019_csv)
    df_isic2019['image_path'] = df_isic2019['image'].apply(lambda x: os.path.join(path_isic2019, f"{x}.jpg"))
    df_isic_gt = pd.read_csv(path_isic_gt)
    # create one column "dx", and map cols MEL NV BCC AK BKL DF VASC SCC UNK to one column where i want: NV: 5, MEL: 4, BKL:2, BCC:1, AK:0, VASC:6, DF:3, drop unk and scc, classes are in cols, binary values
    df_isic_gt['dx'] = 0
    dx_mapping = {
        'MEL': 4,
        'NV': 5,
        'BCC': 1,
        'AK': 0,
        'BKL': 2,
        'DF': 3,
        'VASC': 6,
        'SCC': None,
        'UNK': None
    }
    for col, val in dx_mapping.items():
        df_isic_gt.loc[df_isic_gt[col] == 1, 'dx'] = val
    df_isic_gt = df_isic_gt.drop(columns=list(dx_mapping.keys()))
    df_isic_gt = df_isic_gt.dropna(subset=['dx'])
    df_isic_gt['dx'] = df_isic_gt['dx'].astype(int)
    df_isic2019 = df_isic2019.merge(df_isic_gt[['image', 'dx']], on='image', how='inner')

    df_isic2019 = df_isic2019[['image_path', 'dx']]
    df_train_val = pd.concat([df_train_val, df_isic2019], ignore_index=True, sort=False)
    df_train_val = df_train_val.reset_index(drop=True)
    df_train_val['image_id'] = df_train_val['image_path'].apply(lambda x: os.path.basename(x).split('.')[0])
    df_train_val = df_train_val.drop_duplicates(subset=['image_id'], keep='first')
    df_train_val = df_train_val.reset_index(drop=True)
    df_test['image_id'] = df_test['image_path'].apply(lambda x: os.path.basename(x).split('.')[0])
    df_train_val = df_train_val[~df_train_val['image_id'].isin(df_test['image_id'])]
    df_train_val = df_train_val.reset_index(drop=True)
    # find most freqeunt existing value in each column and fill na with it
    columns_to_fill = ['segmentation_path', 'age', 'sex', 'localization', 'hair', 'ruler_marks', 'bubbles', 'vignette', 'frame', 'other', 'age_normalized', 'sex_encoded', 'loc_encoded']
    for column in columns_to_fill:
        if column in df_train_val.columns:
            most_frequent = df_train_val[column].mode()[0]
            df_train_val[column] = df_train_val[column].fillna(most_frequent)
# %%
train_transform = A.Compose([
    # A.Resize(224,224), # not needed with RandomResizedCrop?
    A.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.3),
    # A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3),
    # A.GaussianBlur(blur_limit=(3,7), sigma_limit=0.1, p=0.2),
    # A.GaussNoise(std_range=(0.03, 0.03), mean_range=(0.0, 0.0), per_channel=True, noise_scale_factor=1.0, p=0.2),
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
# %%
# Over-sampling to handle class imbalance
class_counts = df_train['dx'].value_counts()
class_weights = 1.0 / class_counts
sample_weights = df_train['dx'].map(class_weights).astype(float).values
sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)

# train_loader = DataLoader(train_dataset, batch_size=config['training_plan']['parameters']['batch_size'], shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=config['training_plan']['parameters']['batch_size'], sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

norm_pix_loss = config['training_plan']['parameters']['norm_pix_loss']

ae_model = convmae_convvit_base_patch16_dec512d8b(norm_pix_loss=norm_pix_loss)
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

mask_ratio = config['training_plan']['parameters']['masking_ratio']
eval_mask_ratio = config['training_plan']['parameters']['eval_masking_ratio']
include_lesion_mask = False  # be aware that lesion masks for isic2019 are not available

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
    latent_feats_list, target_list = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            loss, _, _ = ae_model(images, mask_ratio=eval_mask_ratio)
            running_loss += loss.item() * images.size(0)
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                latent, _, _ = ae_model.forward_encoder(images, mask_ratio=0.0)
                feats = concat_patch_moments(latent)  # (B, 6*D)
                latent_feats_list.append(feats.cpu())
                targets = batch['target']
                target_list.append(targets.cpu())

        balance_classes = False
        visualize_latent_space(config, run, seed, num_epochs, epoch, latent_feats_list, target_list, balance_classes=balance_classes)
    val_loss = running_loss / len(val_loader.dataset)
    

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if config["neptune"]:
        run["train/loss"].append(train_loss)
        run["val/loss"].append(val_loss)
        
        visualize_model_outputs(run, device, val_loader, ae_model, mask_ratio, num_epochs, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(ae_model.state_dict())

    if epoch == num_epochs - 1:
        model_dir = os.path.join(root, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{uuid.uuid4().hex}.pth")
        torch.save(best_model_state, model_path)
        print(f"Saved Best Model at {model_path}")
        run["best_model_path"].log(model_path) if config["neptune"] else None


# %%
# after training forward pass thorugh encoder only with mask_ratio == 0
# ae.model.eval()
# with torch.no_grad():
#     # Pass mask_ratio=0 to use the full image
#     latent, mask, ids_restore = model.forward_encoder(imgs, mask_ratio=0.0)