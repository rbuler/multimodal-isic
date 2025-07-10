import os
import yaml
import typing
import pickle
import logging
import argparse
import warnings
import pandas as pd
from dataset import DermDataset

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
df_train = pd.read_csv(config['dir']['csv'])
df_test = pd.read_csv(config['dir']['csv_test'])

with open(config['dir']['radiomics'], 'rb') as f:
    pickle_train = pickle.load(f)

with open(config['dir']['radiomics_test'], 'rb') as f:
    pickle_test = pickle.load(f)

dataset_train = DermDataset(df=df_train, radiomics=pickle_train, is_train=True)
dataset_test = DermDataset(df=df_test, radiomics=pickle_test, is_train=False)
# %%