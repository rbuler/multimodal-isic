import os
import yaml
import typing
import pickle
import logging
import argparse
import warnings
import pandas as pd
from dataset import DermDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# %%
df_train = pd.read_pickle(config['dir']['df'])
df_test = pd.read_pickle(config['dir']['df_test'])

with open(config['dir']['radiomics'], 'rb') as f:
    pickle_train = pickle.load(f)

with open(config['dir']['radiomics_test'], 'rb') as f:
    pickle_test = pickle.load(f)

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
dataset_train = DermDataset(df=df_train, radiomics=pickle_train, transform=train_transform, is_train=True)
##
# TODO
# add validation dataset
##
dataset_test = DermDataset(df=df_test, radiomics=pickle_test, transform=valid_transform, is_train=False)
# %%



# %%