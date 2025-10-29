# %% IMPORTS
import umap
import umap.plot as umap_plot
import os
import faiss
import torch
import numpy as np
import pandas as pd
import yaml
import typing
import argparse
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import from_scipy_sparse_matrix, softmax as pyg_softmax
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
from bokeh.plotting import output_notebook
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


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

device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
# %%
# LOAD DataFrame FROM pickle
from sklearn.decomposition import PCA

patch_level_latents_df_train = pd.read_pickle("dataframes_latents/patch_level_latents_train_df.pkl")
patch_level_latents_df_test = pd.read_pickle("dataframes_latents/patch_level_latents_test_df.pkl")

patch_level_latents_df_train['patient_id'] = patch_level_latents_df_train['image_path'].apply(lambda x: os.path.basename(x).split('_')[1].split('.')[0])
patch_level_latents_df_test['patient_id'] = patch_level_latents_df_test['image_path'].apply(lambda x: os.path.basename(x).split('_')[1].split('.')[0])

X_train = np.vstack(patch_level_latents_df_train['patch_latent_pca'].values)
y_train = patch_level_latents_df_train['target'].values
X_test  = np.vstack(patch_level_latents_df_test['patch_latent_pca'].values)
y_test  = patch_level_latents_df_test['target'].values
# %%


# GPU pipeline for 800k x 144 feature patches
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from cuml.metrics import trustworthiness
import cupy as cp

X_gpu = cp.asarray(X_train)
X_red = UMAP(n_neighbors=15, min_dist=0.05, n_components=20).fit_transform(X_gpu)
score = trustworthiness(X_train, X_red)
print(f"Trustworthiness of UMAP embedding: {score:.4f}")
# %%
clusters = HDBSCAN(min_cluster_size=50, min_samples=10).fit_predict(X_red)
print(f"Number of clusters found: {len(np.unique(clusters)) - (1 if -1 in clusters else 0)}")

# go back to CPU
X_red_cpu = cp.asnumpy(X_red)
clusters_cpu = cp.asnumpy(clusters)

X_train = X_red_cpu[clusters_cpu != -1]
y_train = y_train[clusters_cpu != -1]
patch_level_latents_df_train = patch_level_latents_df_train.iloc[clusters_cpu != -1].reset_index(drop=True)
patch_level_latents_df_train['cluster'] = clusters_cpu[clusters_cpu != -1]
# %%
# use patch_level_latents_df_train['cluster'] to group by clusters and compute scores within each cluster
# patch_level_latents_df_train['target'] has the class label
X_feat = np.vstack(patch_level_latents_df_train['patch_latent_pca'].values)
y = patch_level_latents_df_train['target'].values
clusters = patch_level_latents_df_train['cluster'].values

same_counts = np.zeros(len(y), dtype=int)
other_counts = np.zeros(len(y), dtype=int)
eps = 1e-8

for clust in np.unique(clusters):
    idx_clust = np.where(clusters == clust)[0]
    if idx_clust.size == 0:
        continue
    y_clust = y[idx_clust]
    unique_vals, counts = np.unique(y_clust, return_counts=True)
    count_map = dict(zip(unique_vals, counts))
    cluster_size = len(idx_clust)
    for local_i, global_i in enumerate(idx_clust):
        cls = y_clust[local_i]
        same = count_map.get(cls, 0) - 1  # exclude the point itself
        other = cluster_size - same - 1
        same_counts[global_i] = same
        other_counts[global_i] = other

# store counts and simple scores in the dataframe
patch_level_latents_df_train['cluster_same_count'] = same_counts
patch_level_latents_df_train['cluster_other_count'] = other_counts
patch_level_latents_df_train['cluster_prop_same'] = (same_counts.astype(float) + eps) / (same_counts + other_counts + eps)
patch_level_latents_df_train['cluster_ratio_same_other'] = (same_counts.astype(float) + eps) / (other_counts.astype(float) + eps)

# %%
# drop all clusters with cluster_ratio_same_other less than 30
df_filtered = patch_level_latents_df_train[patch_level_latents_df_train['cluster_ratio_same_other'] >= 30].reset_index(drop=True)
X_feat = np.vstack(df_filtered['patch_latent_pca'].values)
y = df_filtered['target'].values

# print 10 largest clusters after filtering
print("10 largest clusters after filtering:")
print(df_filtered['cluster'].value_counts().head(10))

# drop cluster 3163 and 2976 if they exist
clusters_to_drop = [3163, 2976]
df_filtered = df_filtered[~df_filtered['cluster'].isin(clusters_to_drop)].reset_index(drop=True)
X_feat = np.vstack(df_filtered['patch_latent_pca'].values)
y = df_filtered['target'].values

# drop from X_feat and y class 5
mask = y != 5
X_feat = X_feat[mask]
y = y[mask]

# cuml umap on filtered data
X_gpu = cp.asarray(X_feat)
X_red = UMAP(n_neighbors=100, min_dist=0.9, n_components=2).fit_transform(X_gpu)
score = trustworthiness(X_feat, cp.asnumpy(X_red))
print(f"Trustworthiness of UMAP embedding after filtering: {score:.4f}")
X_feat = cp.asnumpy(X_red)

# plot umap of filtered prototypes
plt.figure(figsize=(8, 6))
plt.scatter(X_feat[:, 0], X_feat[:, 1], c=y, s=1, cmap='tab10', alpha=0.8)
plt.colorbar(ticks=np.arange(len(np.unique(y))))
plt.title("2D UMAP Projection of Filtered Prototypes Colored by Class")
plt.xlabel("UMAP dim 1")
plt.ylabel("UMAP dim 2")
plt.tight_layout()
plt.show()
# %%
# umap to visualize prototypes, 2D/3D depending on n_components

n_neighbors = [25]
min_dist = [0.1]
n_components = 2
X = X_feat
y = y

for n in n_neighbors:
    for md in min_dist:
        reducer = umap.UMAP(random_state=42, n_neighbors=n, min_dist=md, n_components=n_components, metric='cosine')
        embedding = reducer.fit_transform(X)
        dims = embedding.shape[1]

        unique_classes = np.unique(y)
        palette = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(unique_classes)}

        if dims == 2:
            plt.figure(figsize=(8, 6))
            for c in unique_classes:
                mask = y == c
                plt.scatter(embedding[mask, 0], embedding[mask, 1],
                            s=10, label=str(c), color=color_map[c], alpha=0.9, edgecolors='none')
            plt.xlabel("UMAP dim 1")
            plt.ylabel("UMAP dim 2")
            plt.title(f"2D UMAP Projection of Prototypes Colored by Class (n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()

        elif dims >= 3:
            if dims > 3:
                print(f"UMAP produced {dims} dimensions, plotting first 3.")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            for c in unique_classes:
                mask = y == c
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                        s=10, label=str(c), color=color_map[c], depthshade=True, alpha=0.9)
            ax.set_xlabel("UMAP dim 1")
            ax.set_ylabel("UMAP dim 2")
            ax.set_zlabel("UMAP dim 3")
            ax.set_title("3D UMAP Projection of Prototypes Colored by Class (n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist})")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.tight_layout()

        output_dir = "figures_umap_prototypes"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f"umap_prototypes_n{reducer.n_neighbors}_md{reducer.min_dist}.png"), dpi=300)
        plt.show()

output_notebook()
hover_data = {
    'Class': y.astype(str),
}
p = umap_plot.interactive(reducer, hover_data=hover_data, labels=y, point_size=3)
umap_plot.show(p)
# %%
