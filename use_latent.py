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

X_train = np.vstack(patch_level_latents_df_train['patch_latent'].values)
y_train = patch_level_latents_df_train['target'].values
X_test  = np.vstack(patch_level_latents_df_test['patch_latent'].values)
y_test  = patch_level_latents_df_test['target'].values
# %%

X_feat = X_train
y = y_train

scores = np.zeros(len(patch_level_latents_df_train), dtype=np.float32)
pct_for_k = 0.01            # 1% of class size
k_min = 3
eps = 1e-8

for cls in np.unique(y):
    idx_cls = np.where(y == cls)[0]
    idx_other = np.where(y != cls)[0]
    X_cls = X_feat[idx_cls]
    X_other = X_feat[idx_other]

    n_cls = len(X_cls)
    n_other = len(X_other)

    k_calc = int(round(pct_for_k * n_cls))
    k_same = max(k_min, k_calc)
    k_same = min(k_same, 10000)
    k_calc_o = int(round(pct_for_k * n_other))
    k_other = max(1, k_calc_o)
    k_other = min(k_other, 10000)

    nbrs_same = NearestNeighbors(n_neighbors=k_same + 1).fit(X_cls)
    d_same, _ = nbrs_same.kneighbors(X_cls)
    intra = d_same[:, 1:].mean(axis=1)

    nbrs_other = NearestNeighbors(n_neighbors=k_other).fit(X_other)
    d_other, _ = nbrs_other.kneighbors(X_cls)
    inter = d_other.mean(axis=1)

    scores_cls = inter / (intra + eps)
    scores[idx_cls] = scores_cls
    print(f"Class {cls}: Prototype scores computed with {k_same} same-class and {k_other} other-class neighbors.")
patch_level_latents_df_train['proto_score'] = scores
print("Done computing prototype scores.")

print(patch_level_latents_df_train['target'].value_counts())
print(len(patch_level_latents_df_train['image_path'].unique()))
print(patch_level_latents_df_train.groupby('target')['image_path'].nunique())

# %% 
# Build patient-level bags for MIL: each patient -> up to n instances (top by proto_score).

INSTANCES_PER_PATIENT = 24

def build_patient_bags(df, instances_per_patient=INSTANCES_PER_PATIENT, train=True):
    """
    Returns: bags_list (list of torch.Tensor [num_instances, D]),
             labels (torch.LongTensor [num_patients]),
             patient_ids (list)
    For train=True take up to instances_per_patient top patches by proto_score,
    for train=False take all patches for each patient.
    """
    bags = []
    labels = []
    pids = []
    for pid, grp in df.groupby('patient_id'):
        latents = np.vstack(grp.sort_values('proto_score', ascending=False)['patch_latent_pca'].values)
        if train:
            latents = latents[:instances_per_patient]
        bags.append(torch.from_numpy(latents).float())
        labels.append(int(grp['target'].iloc[0]))
        pids.append(pid)
    labels = torch.tensor(labels, dtype=torch.long)
    return bags, labels, pids

bags_train_list, labels_train, pids_train = build_patient_bags(patch_level_latents_df_train, INSTANCES_PER_PATIENT, train=True)
bags_test_list, labels_test, pids_test = build_patient_bags(patch_level_latents_df_test, INSTANCES_PER_PATIENT, train=False)

class BagDataset(torch.utils.data.Dataset):
    def __init__(self, bags, labels, pids):
        self.bags = bags
        self.labels = labels
        self.pids = pids
    def __len__(self):
        return len(self.bags)
    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx], self.pids[idx]

train_ds = BagDataset(bags_train_list, labels_train, pids_train)
test_ds  = BagDataset(bags_test_list,  labels_test,  pids_test)

def collate_pad(batch):
    bags, labels, pids = zip(*batch)
    lengths = [b.shape[0] for b in bags]
    max_len = max(lengths)
    D = bags[0].shape[1]
    padded = torch.zeros(len(bags), max_len, D, dtype=bags[0].dtype)
    mask = torch.zeros(len(bags), max_len, dtype=torch.bool)
    for i, b in enumerate(bags):
        L = b.shape[0]
        padded[i, :L] = b
        mask[i, :L] = True
    labels = torch.stack([torch.tensor(l, dtype=torch.long) if not isinstance(l, torch.Tensor) else l for l in labels])
    return padded.to(device), labels.to(device), mask.to(device), list(pids)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_pad)
val_loader   = torch.utils.data.DataLoader(test_ds,  batch_size=256, shuffle=False, collate_fn=collate_pad)


class MILAttentionClassifier(torch.nn.Module):
    def __init__(self, input_dim, instance_hidden=128, embed_dim=128, num_classes=7, dropout=0.5):
        # ...existing code...
        self.classifier = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, bags, mask=None):
        """
        bags: (B, M, D) padded tensor
        mask: (B, M) boolean tensor where True marks valid instances
        """
        B, M, D = bags.shape
        x = bags.view(B * M, D)
        inst_emb = self.instance_encoder(x)
        E = inst_emb.shape[1]
        inst_emb = inst_emb.view(B, M, E)          # (B, M, E)

        V = torch.tanh(self.attn_V(inst_emb))      # (B, M, E)
        U = torch.sigmoid(self.attn_U(inst_emb))   # (B, M, E)
        gated = V * U                              # (B, M, E)

        attn_logits = self.attn_w(gated).squeeze(-1)  # (B, M)
        if mask is not None:
            # mask is True for valid positions; set logits of padded positions to large negative
            attn_logits = attn_logits.masked_fill(~mask, float('-1e9'))
        attn_w = torch.softmax(attn_logits, dim=1)    # (B, M)
        attn_w_unsq = attn_w.unsqueeze(-1)            # (B, M, 1)
        pooled = (attn_w_unsq * inst_emb).sum(dim=1)  # (B, E)

        logits = self.classifier(pooled)              # (B, num_classes)
        return logits, attn_w


model = MILAttentionClassifier(input_dim=patch_level_latents_df_train['patch_latent_pca'].iloc[0].shape[0],
                               instance_hidden=128,
                               embed_dim=128,
                               num_classes=len(np.unique(y)),
                               dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
epochs = 50

for epoch in range(epochs):
    model.train()
    for bags, labels, mask, pids in train_loader:
        optimizer.zero_grad()
        logits, attn_w = model(bags, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for bags, labels, mask, pids in val_loader:
            logits, attn_w = model(bags, mask)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    bal_acc = balanced_accuracy_score(all_labels.numpy(), all_preds.numpy())
    print(f"Validation Balanced Accuracy: {bal_acc:.4f}")
    

# %%
# umap to visualize prototypes, 2D/3D depending on n_components

n_neighbors = [25]
min_dist = [0.9]
n_components = 2
X = None
y = None

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
