import numpy as np
from bokeh.plotting import output_notebook
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN
from cuml.metrics import trustworthiness
import cupy as cp
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import os
import umap
import umap.plot as umap_plot

# %%
patch_level_latents_df_train = pd.read_pickle("patch_level_latents_df_train.pkl")
patch_level_latents_df_test = pd.read_pickle("patch_level_latents_df_test.pkl")


X_train = np.vstack(patch_level_latents_df_train['patch_latent_pca'].values)
y_train = patch_level_latents_df_train['target'].values
X_test  = np.vstack(patch_level_latents_df_test['patch_latent_pca'].values)
y_test  = patch_level_latents_df_test['target'].values

X_gpu = cp.asarray(X_train)
X_red = UMAP(n_neighbors=15, min_dist=0.05, n_components=20).fit_transform(X_gpu)
score = trustworthiness(X_train, X_red)
print(f"Trustworthiness of UMAP embedding: {score:.4f}")

# cluster using HDBSCAN
clusters = HDBSCAN(min_cluster_size=50, min_samples=10).fit_predict(X_red)
print(f"Number of clusters found: {len(np.unique(clusters)) - (1 if -1 in clusters else 0)}")

X_red_cpu = cp.asnumpy(X_red)
clusters_cpu = cp.asnumpy(clusters)

patch_level_latents_df_train['cluster'] = clusters_cpu
df_filtered = patch_level_latents_df_train[patch_level_latents_df_train['cluster'] != -1].reset_index(drop=True)

print(f"Number of patches in training set: {len(df_filtered)} after removing noise clusters")
for c in df_filtered['target'].unique():
    n_c = len(df_filtered[df_filtered['target'] == c])
    print(f"  Class {c}: {n_c} patches")

# save filtered df
df_filtered.to_pickle("df_filtered.pkl")
# %%
# analyze clusters and compute statistics per cluster
load = True
if load:
    df_filtered = pd.read_pickle("df_filtered.pkl")

X_feat = np.vstack(df_filtered['patch_latent_pca'].values)
y = df_filtered['target'].values
clusters = df_filtered['cluster'].values

same_counts = np.zeros(len(y), dtype=int)
other_counts = np.zeros(len(y), dtype=int)
eps = 1e-8

unique_classes = np.unique(y)
same_counts = np.zeros(len(y), dtype=int)
other_counts = np.zeros(len(y), dtype=int)
counts_per_class = {c: np.zeros(len(y), dtype=int) for c in unique_classes}

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
        for c in unique_classes:
            counts_per_class[c][global_i] = count_map.get(c, 0)

df_filtered['cluster_same_count'] = same_counts
df_filtered['cluster_other_count'] = other_counts
df_filtered['cluster_prop_same'] = (same_counts.astype(float) + eps) / (same_counts + other_counts + eps)
df_filtered['cluster_ratio_same_other'] = (same_counts.astype(float) + eps) / (other_counts.astype(float) + eps)

for c in unique_classes:
    colname = f"cluster_count_class_{int(c)}"
    df_filtered[colname] = counts_per_class[c]


patient_target_counts = patch_level_latents_df_train.groupby('patient_id')['target'].agg(lambda x: Counter(x).most_common(1)[0][0])
print("Patient target counts in training dataset:")
print(patient_target_counts.value_counts())
counts = dict(patient_target_counts.value_counts())

class_weights = {}
total_patients = len(patient_target_counts)
for c in unique_classes:
    class_count = counts.get(c, 0)
    class_weights[c] = total_patients / (class_count + eps)
    print(f"Class {c}: {class_count} patients, weight: {class_weights[c]:.4f}")
weighted_same = np.zeros(len(y), dtype=float)
weighted_other = np.zeros(len(y), dtype=float)
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
        weighted_same[global_i] = same * class_weights[cls]
        weighted_other[global_i] = 0.0
        for c in unique_classes:
            if c != cls:
                weighted_other[global_i] += count_map.get(c, 0) * class_weights[c]
df_filtered['cluster_prop_same_weighted'] = (weighted_same + eps) / (weighted_same + weighted_other + eps)

# plot histogram of prop_same and weighted using one value per cluster
df_cluster = df_filtered.groupby('cluster').agg({
    'cluster_prop_same': 'first',
    'cluster_prop_same_weighted': 'first'
}).reset_index()

# take last 10 percentile of cluster_prop_same_weighted
threshold = np.percentile(df_cluster['cluster_prop_same_weighted'], 10)
print(f"10th percentile of cluster_prop_same_weighted: {threshold:.4f}")

# %%
# drop all clusters with cluster_prop less than threshold
df_filtered = df_filtered[df_filtered['cluster_prop_same_weighted'] >= threshold].reset_index(drop=True)
X_feat = np.vstack(df_filtered['patch_latent_pca'].values)
y = df_filtered['target'].values

# %%
# cuml umap on filtered data
X = np.vstack(df_filtered['patch_latent_pca'].values)
y = df_filtered['target'].values

X_gpu = cp.asarray(X)
X_red = UMAP(n_neighbors=5, min_dist=0.9, n_components=2).fit_transform(X_gpu)
score = trustworthiness(X, cp.asnumpy(X_red))
print(f"Trustworthiness of UMAP embedding after filtering: {score:.4f}")
X = cp.asnumpy(X_red)

# plot umap of filtered prototypes
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Accent', s=1, alpha=0.8)
plt.colorbar(ticks=np.arange(len(np.unique(y))))

plt.title("2D UMAP Projection of Filtered Prototypes Colored by Class")
plt.xlabel("UMAP dim 1")
plt.ylabel("UMAP dim 2")
plt.tight_layout()
plt.show()

# %%
# umap to visualize prototypes, 2D/3D depending on n_components

n_neighbors = [5]
min_dist = [0.9]
n_components = 2

X = np.vstack(df_filtered['patch_latent_pca'].values)
y = df_filtered['target'].values


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
                            s=1, label=str(c), cmap='Accent', alpha=0.9, edgecolors='none')
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