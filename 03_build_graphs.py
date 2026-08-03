import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


NUM_NODES = 196
GRID_SIDE = 14


def _grid_edge_index(connect_diagonals=False):
    if GRID_SIDE * GRID_SIDE != NUM_NODES:
        raise ValueError("NUM_NODES must be a perfect square for grid graphs")

    edges = []
    neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connect_diagonals:
        neighbor_offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE):
            node_idx = row * GRID_SIDE + col
            for d_row, d_col in neighbor_offsets:
                neigh_row, neigh_col = row + d_row, col + d_col
                if 0 <= neigh_row < GRID_SIDE and 0 <= neigh_col < GRID_SIDE:
                    neigh_idx = neigh_row * GRID_SIDE + neigh_col
                    edges.append((node_idx, neigh_idx))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def _grid_adjacency(connect_diagonals=False, add_self_loops=True):
    edge_index = _grid_edge_index(connect_diagonals=connect_diagonals)
    adj = torch.zeros((NUM_NODES, NUM_NODES), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    if add_self_loops:
        adj.fill_diagonal_(1.0)
    return adj, edge_index


def _knn_edge_index(x, k=8):
    if x.ndim != 2:
        raise ValueError("x must be a 2D tensor [num_nodes, feature_dim]")

    num_nodes = x.size(0)
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)

    k = int(max(1, min(k, num_nodes - 1)))
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    dist = x_norm + x_norm.t() - 2.0 * x @ x.t()
    dist = torch.clamp(dist, min=0.0)
    dist.fill_diagonal_(float("inf"))

    _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)
    src = torch.arange(num_nodes, device=x.device).unsqueeze(1).expand(-1, k)
    edge_index = torch.stack([src.reshape(-1), nn_idx.reshape(-1)], dim=0)
    return edge_index.long()


def _random_edge_index(num_nodes, r=4, seed=None):
    if num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)

    r = int(max(1, min(r, num_nodes - 1)))
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))

    src, dst = [], []
    for node_idx in range(num_nodes):
        candidates = torch.arange(num_nodes)
        candidates = candidates[candidates != node_idx]
        perm = torch.randperm(candidates.numel(), generator=generator)
        chosen = candidates[perm[:r]]
        src.extend([node_idx] * chosen.numel())
        dst.extend(chosen.tolist())

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def build_graph_blueprints(row, graph_type, k=8, r=4, connect_diagonals=False, seed=42):
    patch_embeddings = torch.as_tensor(row["patch_embeddings"], dtype=torch.float32)
    if patch_embeddings.ndim != 2:
        raise ValueError(f"patch_embeddings must be 2D, got shape {tuple(patch_embeddings.shape)}")
    if patch_embeddings.size(0) != NUM_NODES:
        raise ValueError(f"Expected {NUM_NODES} nodes, got {patch_embeddings.size(0)}")

    if graph_type == "grid":
        adjacency, edge_index = _grid_adjacency(connect_diagonals=connect_diagonals)
        return {
            "graph_type": "grid",
            "connect_diagonals": bool(connect_diagonals),
            "adjacency": adjacency.numpy(),
            "edge_index": edge_index.numpy(),
        }

    if graph_type == "knn":
        edge_index = _knn_edge_index(patch_embeddings, k=k)
        return {
            "graph_type": "knn",
            "k": int(k),
            "edge_index": edge_index.numpy(),
        }

    if graph_type == "random":
        edge_index = _random_edge_index(NUM_NODES, r=r, seed=seed)
        return {
            "graph_type": "random",
            "r": int(r),
            "seed": None if seed is None else int(seed),
            "edge_index": edge_index.numpy(),
        }

    raise ValueError(f"Unsupported graph_type={graph_type!r}")


def _as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def process_teacher_pickle(input_path, output_root, k_values=8, r_values=4, connect_diagonals=False, seed=42):
    with open(input_path, "rb") as f:
        teacher_df = pickle.load(f)

    if not isinstance(teacher_df, pd.DataFrame):
        teacher_df = pd.DataFrame(teacher_df)

    input_path = Path(input_path)
    output_root = Path(output_root)
    relative_parent = input_path.parent.relative_to(input_path.parents[1]) if len(input_path.parents) > 1 else Path("")
    output_dir = output_root / relative_parent
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "grid4": [],
        "grid8": [],
    }

    k_values = _as_list(k_values)
    r_values = _as_list(r_values)

    for k in k_values:
        outputs[f"knn{k}"] = []
    for r in r_values:
        outputs[f"random{r}"] = []

    for row_idx, row in teacher_df.iterrows():
        base_record = {
            "image_id": row["image_id"],
            "label": row["label"],
            "patch_embeddings": row["patch_embeddings"],
            "patch_probs": row["patch_probs"],
            "attention": row["attention"],
            "entropy": row["entropy"],
            "confidence": row["confidence"],
            "dominant_class": row["dominant_class"],
        }

        outputs["grid4"].append({
            **base_record,
            **build_graph_blueprints(row, "grid", connect_diagonals=False),
        })
        outputs["grid8"].append({
            **base_record,
            **build_graph_blueprints(row, "grid", connect_diagonals=True),
        })
        for k in k_values:
            outputs[f"knn{k}"].append({
                **base_record,
                **build_graph_blueprints(row, "knn", k=k),
            })
        for r in r_values:
            outputs[f"random{r}"].append({
                **base_record,
                **build_graph_blueprints(row, "random", r=r, seed=seed + row_idx),
            })

    for graph_name, records in outputs.items():
        out_path = output_dir / f"{input_path.stem}_{graph_name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(pd.DataFrame(records), f)
        print(f"Saved: {out_path}")


# def main():

teacher_root = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/teacher_outputs")
output_root = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/graph_outputs")

k = [1, 2, 3, 4, 5, 6, 7, 8]
r = [1, 2, 3, 4, 5, 6, 7, 8]
seed = 42

for input_path in sorted(teacher_root.rglob("*.pkl")):
    if input_path.name.startswith("teacher_outputs_"):
        continue
    process_teacher_pickle(
        input_path=input_path,
        output_root=output_root,
        k_values=k,
        r_values=r,
        connect_diagonals=False,
        seed=seed,
    )


# if __name__ == "__main__":
#     main()