import os
import yaml
import uuid
import torch
import typing
import pickle
import neptune
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import albumentations as A
from dataset import DermDataset
from model import MultiModalFusionNet
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from net_utils import train, validate, test, EarlyStopping
from sklearn.model_selection import StratifiedKFold


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

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
# parser.add_argument("--fold", type=int, default=None)
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

# current_fold = args.fold if args.fold else config['training_plan']['parameters']['fold']
current_fold = config['training_plan']['parameters']['fold']

if config["neptune"]:
    run = neptune.init_run(project="ProjektMMG/multimodal-isic",)
    run["sys/group_tags"].add(config["modality"])
    if not any(mod in config["modality"] for mod in ['image', 'radiomics', 'clinical', 'artifacts']):
        run["sys/group_tags"].add('baseline')
    else:
        run["sys/group_tags"].add(config['training_plan']['fusion'])
    
    run["config"] = config
    run['train/current_fold'] = current_fold
else:
    run = None

# SET FIXED SEED FOR REPRODUCIBILITY --------------------------------
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = config['device'] if torch.cuda.is_available() else 'cpu'
# %%
df_train_val = pd.read_pickle(config['dir']['df'])
df_test = pd.read_pickle(config['dir']['df_test'])

# with open(config['dir']['radiomics'], 'rb') as f:
#     pickle_train = pickle.load(f)

# with open(config['dir']['radiomics_test'], 'rb') as f:
#     pickle_test = pickle.load(f)

# %%
train_transform = A.Compose([
    A.Resize(380, 380),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

valid_transform = A.Compose([
    A.Resize(380, 380),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
# %%
SPLITS = 10
if run:
    run['train/splits'] = SPLITS

kf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=seed)
folds = list(kf.split(df_train_val, df_train_val['dx']))

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    if fold_idx != current_fold:
        continue
    train_idx, val_idx = folds[current_fold]
    df_train = df_train_val.iloc[train_idx]
    df_val = df_train_val.iloc[val_idx]
    # pickle_train = pickle_train.iloc[train_idx]
    # pickle_val = pickle_train.iloc[val_idx]
    print(f"Train set size: {len(df_train)}")
    print(f"Val set size: {len(df_val)}")
    print(f"Test set size: {len(df_test)}")

# %%
train_dataset = DermDataset(df=df_train, radiomics=None, transform=train_transform, is_train=True)
val_dataset = DermDataset(df=df_val, radiomics=None, transform=valid_transform, is_train=False)
test_dataset = DermDataset(df=df_test, radiomics=None, transform=valid_transform, is_train=False)
# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

dataloaders = {'train': train_loader,
               'val': val_loader,
               'test': test_loader}
# %%
modality = config['training_plan']['modality']
fusion = config['training_plan']['fusion']
fusion_level = config['training_plan']['fusion_level']
model = MultiModalFusionNet(modality=modality, fusion_level=fusion_level, fusion_strategy=fusion)
# model.apply(deactivate_batchnorm)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# %%
early_stopping = EarlyStopping(patience=config['training_plan']['parameters']['patience'], neptune_run=run)

for epoch in range(1, config['training_plan']['parameters']['epochs'] + 1):
    train(model, dataloaders['train'], criterion, optimizer, device, run, epoch)
    val_loss = validate(model, dataloaders['val'], criterion, device, run, epoch)
    if early_stopping(val_loss, model):
        print(f"Early stopping at epoch {epoch}")
        break
model_name = uuid.uuid4().hex
model_name = os.path.join(config['model_path'], model_name)
if not os.path.exists(config['model_path']):
    os.makedirs(config['model_path'])
torch.save(early_stopping.get_best_model_state(), model_name)
if run is not None:
    run["best_model_path"].log(model_name)


model = MultiModalFusionNet(modality=modality, fusion_level=fusion_level, fusion_strategy=fusion)
# model.apply(deactivate_batchnorm)
model.load_state_dict(torch.load(model_name))
model.to(device)
test(model, dataloaders['test'], device, run)
if run is not None:
    run.stop()
# %%