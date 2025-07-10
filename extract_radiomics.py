# %%
import os
import yaml
import typing
import argparse
import logging
import warnings
import numpy as np
import pandas as pd
from RadiomicExtractor import RadiomicsExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV


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

df_train = df_train.head(5000)
df_test = df_test.head(100)

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
def filter_low_variance(train_df, test_df, threshold=1e-5):
    selector = VarianceThreshold(threshold)
    train_filtered = selector.fit_transform(train_df)
    test_filtered = selector.transform(test_df)
    kept_cols = train_df.columns[selector.get_support()]
    return pd.DataFrame(train_filtered, columns=kept_cols), pd.DataFrame(test_filtered, columns=kept_cols)

def normalize_features(train_df, test_df):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return pd.DataFrame(train_scaled, columns=train_df.columns), pd.DataFrame(test_scaled, columns=train_df.columns)

def lasso_select(train_df, y_train, test_df, C_values='auto'):
    if C_values == 'auto':
        Cs = np.logspace(-2, 1, 20)
    else:
        Cs = C_values
    cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    model = LogisticRegressionCV(
        Cs=Cs,
        cv=cv_strategy,
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        scoring='f1',
        max_iter=10000,
        n_jobs=-1
    ).fit(train_df, y_train)

    selector = SelectFromModel(model, prefit=True)

    train_selected = selector.transform(train_df)
    test_selected = selector.transform(test_df)

    selected_cols = train_df.columns[selector.get_support()]
    return pd.DataFrame(train_selected, columns=selected_cols), pd.DataFrame(test_selected, columns=selected_cols)

def drop_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop
# %% 
y_train = df_train['dx']

print(f"Initial features: {rad_features_train.shape[1]}")
features_train, features_test = filter_low_variance(rad_features_train, rad_features_test)
print(f"Features after variance filtering: {features_train.shape[1]}")

dropped_gs = num_features - len([col for col in features_train.columns if '_gs' in col])
dropped_red = num_features - len([col for col in features_train.columns if '_red' in col])
dropped_green = num_features - len([col for col in features_train.columns if '_green' in col])
dropped_blue = num_features - len([col for col in features_train.columns if '_blue' in col])
print(f"Dropped due to variance filtering - gs: {dropped_gs}, red: {dropped_red}, green: {dropped_green}, blue: {dropped_blue}")

features_train, features_test = normalize_features(features_train, features_test)
features_train, features_test = lasso_select(features_train, y_train, features_test)
print(f"Features after Lasso selection: {features_train.shape[1]}")

dropped_gs = num_features - len([col for col in features_train.columns if '_gs' in col])
dropped_red = num_features - len([col for col in features_train.columns if '_red' in col])
dropped_green = num_features - len([col for col in features_train.columns if '_green' in col])
dropped_blue = num_features - len([col for col in features_train.columns if '_blue' in col])
print(f"Dropped due to Lasso selection - gs: {dropped_gs}, red: {dropped_red}, green: {dropped_green}, blue: {dropped_blue}")

features_train, dropped_features = drop_correlated_features(features_train)
print(f"Features after dropping correlated features: {features_train.shape[1]}")

dropped_gs = len([col for col in dropped_features if '_gs' in col])
dropped_red = len([col for col in dropped_features if '_red' in col])
dropped_green = len([col for col in dropped_features if '_green' in col])
dropped_blue = len([col for col in dropped_features if '_blue' in col])
print(f"Dropped due to correlation filtering - gs: {dropped_gs}, red: {dropped_red}, green: {dropped_green}, blue: {dropped_blue}")

features_test = features_test[features_train.columns]
# %%
output_radiomics_path_train = config['dir']['radiomics']
output_radiomics_path_test = config['dir']['radiomics_test']

features_train.to_pickle(output_radiomics_path_train)
features_test.to_pickle(output_radiomics_path_test)

logger.info(f"Train DataFrame saved to {output_radiomics_path_train}")
logger.info(f"Test DataFrame saved to {output_radiomics_path_test}")
# %%
