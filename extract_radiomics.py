# %%
import os
import yaml
import typing
import argparse
import logging
import warnings
import pandas as pd
from RadiomicExtractor import RadiomicsExtractor


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings('ignore')


logger_radiomics = logging.getLogger("radiomics")
logger_radiomics.setLevel(logging.ERROR)


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

# df_train = df_train.head(100)
# df_test = df_test.head(100)

dct_train = df_train.to_dict(orient='records')
dct_test = df_test.to_dict(orient='records')

extractor_train = RadiomicsExtractor(param_file='params.yml')
extractor_test = RadiomicsExtractor(param_file='params.yml')

features_train = extractor_train.parallell_extraction(dct_train)
features_test = extractor_test.parallell_extraction(dct_test)

rad_features_train = pd.concat([
    pd.DataFrame([item['grayscale'] for item in features_train]),
    pd.DataFrame([item['red'] for item in features_train]),
    pd.DataFrame([item['green'] for item in features_train]),
    pd.DataFrame([item['blue'] for item in features_train])
], axis=1)

rad_features_test = pd.concat([
    pd.DataFrame([item['grayscale'] for item in features_test]),
    pd.DataFrame([item['red'] for item in features_test]),
    pd.DataFrame([item['green'] for item in features_test]),
    pd.DataFrame([item['blue'] for item in features_test])
], axis=1)
# %%
# name col features as original name + respectively gs, red, green, blue
num_features = len(rad_features_train.columns) // 4
rad_features_train.columns = [f"{col}_{img_type}" for col, img_type in zip(rad_features_train.columns, ['gs'] * num_features + ['red'] * num_features + ['green'] * num_features + ['blue'] * num_features)]
rad_features_test.columns = [f"{col}_{img_type}" for col, img_type in zip(rad_features_test.columns, ['gs'] * num_features + ['red'] * num_features + ['green'] * num_features + ['blue'] * num_features)]
# %%
output_radiomics_path_train = config['dir']['radiomics']
output_radiomics_path_test = config['dir']['radiomics_test']

rad_features_train.to_pickle(output_radiomics_path_train)
rad_features_test.to_pickle(output_radiomics_path_test)

logger.info(f"Train DataFrame saved to {output_radiomics_path_train}")
logger.info(f"Test DataFrame saved to {output_radiomics_path_test}")
# %%
