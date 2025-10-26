# %% IMPORTS
import numpy as np
import pandas as pd
import faiss
import scipy.sparse as sp
from collections import Counter
from torch_geometric.utils import from_scipy_sparse_matrix, softmax as pyg_softmax
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_add_pool
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
import umap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from bokeh.plotting import output_notebook
from torch_geometric.data import Data

# %%
# LOAD DataFrame FROM pickle
patch_level_latents_df = pd.read_pickle("patch_level_latents_df.pkl")
# %%

X_feat = np.vstack(patch_level_latents_df['patch_latent_pca'].values)
y = patch_level_latents_df['target'].values

k_same = 15
k_other = 15

scores = np.zeros(len(patch_level_latents_df))

for cls in tqdm(np.unique(y), desc="Prototype scoring"):
    idx_cls = np.where(y == cls)[0]
    idx_other = np.where(y != cls)[0]
    X_cls = X_feat[idx_cls]
    X_other = X_feat[idx_other]

    nbrs_same = NearestNeighbors(n_neighbors=k_same+1).fit(X_cls)
    d_same, _ = nbrs_same.kneighbors(X_cls)
    intra = d_same[:,1:].mean(axis=1)

    nbrs_other = NearestNeighbors(n_neighbors=k_other).fit(X_other)
    d_other, _ = nbrs_other.kneighbors(X_cls)
    inter = d_other.mean(axis=1)

    scores_cls = inter / (intra + 1e-8)
    scores[idx_cls] = scores_cls

patch_level_latents_df['proto_score'] = scores

print("Done computing prototype scores.")

prototypes_idx = []
alpha = 1.0
proto_score_min = 1.05
for cls in tqdm(np.unique(y), desc="Prototype selection"):
    cls_mask = patch_level_latents_df['target'] == cls
    cls_df = patch_level_latents_df.loc[cls_mask]
    cls_scores = cls_df['proto_score']
    mu_cls = cls_scores.mean()
    sigma_cls = cls_scores.std()
    alpha = 1.0 if cls != 5 else 1.5
    proto_score_thresh = mu_cls + alpha * sigma_cls
    final_thresh = max(proto_score_min, proto_score_thresh)
    tqdm.write(f"Class {cls}: Prototype score threshold: {final_thresh:.4f}")

    cls_prototypes = set(cls_scores[cls_scores >= final_thresh].index.tolist())

    # ensure minimum of 4 prototypes per image_path for this class
    groups = cls_df.groupby('image_path').groups
    for image_path, group_idxs in groups.items():
        group_idxs = list(group_idxs)
        selected_for_img = [i for i in group_idxs if i in cls_prototypes]
        if len(selected_for_img) < 4:
            sorted_by_score = cls_scores.loc[group_idxs].sort_values(ascending=False).index.tolist()
            for idx in sorted_by_score:
                if idx not in cls_prototypes:
                    cls_prototypes.add(idx)
                    selected_for_img.append(idx)
                if len(selected_for_img) >= 4:
                    break

    prototypes_idx.extend(list(cls_prototypes))

patch_level_latents_df = patch_level_latents_df.loc[prototypes_idx]

print(patch_level_latents_df['target'].value_counts())
print(len(patch_level_latents_df['image_path'].unique()))
print(patch_level_latents_df.groupby('target')['image_path'].nunique())

patch_level_latents_df['patient_id'] = patch_level_latents_df['image_path'].apply(lambda x: os.path.basename(x).split('_')[1].split('.')[0])
# %%

K_PER_CLASS = 1000

df = patch_level_latents_df
X = np.stack(df['patch_latent_pca'].to_numpy()).astype(np.float32)   # (N, D)
y = df['target'].to_numpy()
N, D = X.shape

classes, counts = np.unique(y, return_counts=True)

centroids_list = []
proto_class = []
assignments_global = np.full(N, -1, dtype=np.int32)
global_proto_id = 0

for cls in classes:
    idxs = np.where(y == cls)[0]
    X_cls = X[idxs]
    n = X_cls.shape[0]
    k = min(K_PER_CLASS, n)
    if k <= 0:
        continue

    if k == n:
        centroids = X_cls.copy()
        labels_local = np.arange(n, dtype=np.int32)
    else:
        kmeans = faiss.Kmeans(d=D, k=k, niter=40, nredo=1, verbose=False, seed=42)
        kmeans.train(X_cls)
        centroids = kmeans.centroids.astype(np.float32)  # (k, D)
        index = faiss.IndexFlatL2(D)
        index.add(centroids)
        _, labels_local = index.search(X_cls, 1)
        labels_local = labels_local.ravel().astype(np.int32)

    n_cent = centroids.shape[0]
    centroids_list.append(centroids)
    for local_id in range(n_cent):
        proto_class.append(cls)
    assignments_global[idxs] = global_proto_id + labels_local
    global_proto_id += n_cent

centroids_all = np.vstack(centroids_list).astype(np.float32)  # (P_total, D)
proto_meta = pd.DataFrame({
    'proto_id': np.arange(len(proto_class), dtype=np.int32),
    'class': proto_class
})

print(f"Total patches: {N}, dim: {D}")
print(f"Produced {len(centroids_all)} prototypes (requested up to {K_PER_CLASS} per class).")

counts_per_proto = np.bincount(assignments_global, minlength=len(centroids_all))
print("Per-proto counts: min/median/mean/max =", counts_per_proto.min(), np.median(counts_per_proto),
      counts_per_proto.mean(), counts_per_proto.max())


# %%

# umap to visualize prototypes, 2D/3D depending on n_components

n_neighbors = [100]
min_dist = [0.4]
n_components = 2
X = centroids_all
y = proto_meta['class'].values

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
p = umap_plot.interactive(reducer, hover_data=hover_data, labels=y, point_size=5)
umap_plot.show(p)
# %%



# Build one big graph (all nodes) for message passing, but compute loss on pooled patient-level labels.

# adjacency from UMAP (scipy sparse)
adjacency_matrix = reducer.graph_.tocsr()  # shape (N, N)
node_features = X.astype(np.float32)       # (N, D)
node_labels = y.astype(np.int64)           # per-node class
patient_ids = patch_level_latents_df['patient_id'].values  # (N,)

# convert adjacency to edge_index (COO -> torch_geometric format)
edge_index, edge_attr = from_scipy_sparse_matrix(adjacency_matrix)  # edge_index: [2, E]

# make edges binary (UMAP weights are not needed here)
edge_index = edge_index[:, edge_attr.squeeze() > 0]


# map patients to indices and compute patient-level labels (majority vote)
unique_patients, inverse_idx = np.unique(patient_ids, return_inverse=True)
num_patients = len(unique_patients)
patient_idx_per_node = torch.tensor(inverse_idx, dtype=torch.long, device=device)  # (N,)
patient_labels = []
for pid in unique_patients:
    idxs = np.where(patient_ids == pid)[0]
    lbl = int(Counter(node_labels[idxs].tolist()).most_common(1)[0][0])
    patient_labels.append(lbl)
patient_labels = torch.tensor(patient_labels, dtype=torch.long, device=device)  # (P,)

# create full graph Data (no per-graph batching here)

data = Data(x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index)
data = data.to(device)
patient_idx_per_node = patient_idx_per_node.to(device)

# split patients into train/val (we still run message passing on the full graph)
rng = np.random.default_rng(config.get('seed', 42))
perm_pat = rng.permutation(num_patients)
n_train_pat = int(0.9 * num_patients)
train_pat_idx = torch.tensor(perm_pat[:n_train_pat], dtype=torch.long, device=device)
val_pat_idx = torch.tensor(perm_pat[n_train_pat:], dtype=torch.long, device=device)

# GNN with attention-based MIL pooling (returns logits per patient)
class GNN_MIL_Attn(torch.nn.Module):
    def __init__(self, in_dim, gnn_hidden=128, gnn_heads=8, attn_dim=128, num_classes=7, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_dim, gnn_hidden, heads=gnn_heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(gnn_hidden * gnn_heads, gnn_hidden, heads=1, concat=True, dropout=dropout)

        self.attn_v = torch.nn.Linear(gnn_hidden, attn_dim)
        self.attn_u = torch.nn.Linear(attn_dim, 1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(gnn_hidden, gnn_hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(gnn_hidden // 2, num_classes)
        )

    def forward(self, data, patient_idx_per_node, num_patients):
        x, edge_index = data.x, data.edge_index
        if edge_index.numel() == 0:
            h = x
        else:
            h = F.elu(self.conv1(x, edge_index))
            h = F.elu(self.conv2(h, edge_index))

        # attention scores per node, softmaxed per patient
        a = torch.tanh(self.attn_v(h))
        a = self.attn_u(a).squeeze(-1)  # (N,)
        a = pyg_softmax(a, patient_idx_per_node)  # softmax over nodes of each patient -> (N,)

        # weighted sum pooling per patient
        h_weighted = h * a.unsqueeze(-1)  # (N, hidden)
        h_pool = global_add_pool(h_weighted, patient_idx_per_node, size=num_patients)  # (P, hidden)

        logits = self.classifier(h_pool)  # (P, num_classes)
        return logits, h_pool, a

# instantiate
in_dim = node_features.shape[1]
num_classes = len(np.unique(node_labels))
gnn_model = GNN_MIL_Attn(in_dim=in_dim, gnn_hidden=128, gnn_heads=8, attn_dim=128,
                         num_classes=num_classes, dropout=0.5).to(device)

# class weights computed over patient-level labels (to use in loss)
counts = np.bincount(patient_labels.cpu().numpy(), minlength=num_classes)
class_weights = torch.tensor([1.0 / c if c > 0 else 0.0 for c in counts], dtype=torch.float32, device=device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=config.get('gnn_lr', 1e-4), weight_decay=config.get('gnn_wd', 5e-4))

# training loop: full-graph message passing, loss computed on patient subsets
num_epochs = 1000
for epoch in range(1, num_epochs + 1):
    gnn_model.train()
    optimizer.zero_grad()
    logits_all, _, attn_scores = gnn_model(data, patient_idx_per_node, num_patients)  # (P, C)
    train_logits = logits_all[train_pat_idx]
    train_labels = patient_labels[train_pat_idx]
    loss = criterion(train_logits, train_labels)
    loss.backward()
    optimizer.step()

    # evaluation on val patients (no grad)
    gnn_model.eval()
    with torch.no_grad():
        logits_all, pooled_all, attn_scores = gnn_model(data, patient_idx_per_node, num_patients)
        val_logits = logits_all[val_pat_idx]
        val_labels = patient_labels[val_pat_idx]
        val_loss = criterion(val_logits, val_labels).item()
        preds = val_logits.argmax(dim=1).cpu().numpy()
        trues = val_labels.cpu().numpy()
        val_acc = (preds == trues).mean() if len(trues) > 0 else 0.0
        try:
            val_bal = balanced_accuracy_score(trues, preds) if len(np.unique(trues)) > 1 else val_acc
        except Exception:
            val_bal = val_acc

    print(f"Epoch {epoch}/{num_epochs} TrainLoss: {loss.item():.4f} ValLoss: {val_loss:.4f} ValAcc: {val_acc:.4f} ValBalAcc: {val_bal:.4f}")

# After training you can inspect attention scores per node (attn_scores.cpu().numpy()),
# the pooled patient embeddings (pooled_all.cpu().numpy()), and per-patient logits (logits_all.cpu().numpy()).
# Save model if desired:
torch.save(gnn_model.state_dict(), "gnn_mil_attn_full_graph.pth")

# %%
