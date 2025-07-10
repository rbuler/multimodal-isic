import os
import yaml
import typing
import logging
import argparse
import warnings
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
csv_path_train = config['dir']['csv']
img_path_train = config['dir']['img']
seg_path_train = config['dir']['seg']
csv_path_test = config['dir']['csv_test']
img_path_test = config['dir']['img_test']
seg_path_test = config['dir']['seg_test']
# %%
df_train = pd.read_csv(csv_path_train)
df_test = pd.read_csv(csv_path_test)
df_test = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]
# drop row form df_test where id ISIC_0035068
df_test = df_test[df_test['image_id'] != 'ISIC_0035068']

# %%
for dx_class in df_train['dx'].unique():
    median_age = df_train[df_train['dx'] == dx_class]['age'].median()
    df_train.loc[df_train['dx'] == dx_class, 'age'] = df_train.loc[df_train['dx'] == dx_class, 'age'].fillna(median_age)
    df_test.loc[df_test['dx'] == dx_class, 'age'] = df_test.loc[df_test['dx'] == dx_class, 'age'].fillna(median_age)

columns_to_fill = ['hair', 'ruler_marks', 'bubbles', 'vignette', 'frame', 'other']
for column in columns_to_fill:
    if column in df_train.columns:
        df_train[column] = df_train[column].fillna(0).astype(int)
        df_test[column] = df_test[column].fillna(0).astype(int)

if 'sex' in df_train.columns:
    df_train['sex'] = df_train['sex'].fillna('unknown')
    df_test['sex'] = df_test['sex'].fillna('unknown')

if 'localization' in df_train.columns:
    df_train['localization'] = df_train['localization'].fillna('unknown')
    df_test['localization'] = df_test['localization'].fillna('unknown')

df_train['image_path'] = df_train['image_id'].apply(lambda x: os.path.join(img_path_train, f"{x}.jpg"))
df_train['segmentation_path'] = df_train['image_id'].apply(lambda x: os.path.join(seg_path_train, f"{x}_segmentation.png"))
df_test['image_path'] = df_test['image_id'].apply(lambda x: os.path.join(img_path_test, f"{x}.jpg"))
df_test['segmentation_path'] = df_test['image_id'].apply(lambda x: os.path.join(seg_path_test, f"{x}_segmentation.png"))


columns_to_drop = ['dx_type', 'dataset', 'lesion_id', 'image_id']
df_train = df_train.drop(columns=[col for col in columns_to_drop if col in df_train.columns])
df_test = df_test.drop(columns=[col for col in columns_to_drop if col in df_test.columns])


if 'image_path' in df_train.columns and 'segmentation_path' in df_train.columns:
    cols = df_train.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    df_train = df_train[cols]
    df_test = df_test[cols]

if 'age' in df_train.columns:
    age_mean = df_train['age'].mean()
    age_std = df_train['age'].std()
    df_train['age_normalized'] = (df_train['age'] - age_mean) / age_std
    df_test['age_normalized'] = (df_test['age'] - age_mean) / age_std

sex_encoder = LabelEncoder()
loc_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df_train['dx'] = target_encoder.fit_transform(df_train['dx'])
df_test['dx'] = target_encoder.transform(df_test['dx'])

df_train['sex_encoded'] = sex_encoder.fit_transform(df_train['sex'])
df_test['sex_encoded'] = sex_encoder.transform(df_test['sex'])

df_train['loc_encoded'] = loc_encoder.fit_transform(df_train['localization'])
df_test['loc_encoded'] = loc_encoder.transform(df_test['localization'])

output_csv_path_train = config['dir']['df']
output_csv_path_test = config['dir']['df_test']

df_train.to_pickle(output_csv_path_train)
df_test.to_pickle(output_csv_path_test)

logger.info(f"Train DataFrame saved to {output_csv_path_train}")
logger.info(f"Test DataFrame saved to {output_csv_path_test}")

# %%
