import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# %%
EPS = 1e-8
DEFAULT_K_VALUES = tuple(range(1, 9)) + (12, 16)
DEFAULT_R_VALUES = tuple(range(1, 9)) + (12, 16)
MEASURES = [
    "H_f",
    "H_e",
    "H_cls",
    "H_kl",
    "H_dirichlet",
    "H_spatial",
    "H_node",
    "H_adj",
]
GRAPH_VARIANT_RE = re.compile(r"^(?P<kind>grid4|grid8|knn|random)(?P<param>\d+)?$")
PATCH_STATS_RE = re.compile(r"^patch_stats_fold_(?P<fold>\d+)_(?P<split>train|val|test)\.pkl$")
PATCH_STATS_ROOT = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/patch_stats")


def _load_graph_dataframe(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def _load_patch_stats_dataframe(model_name):
    model_dir = PATCH_STATS_ROOT / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing patch_stats model directory: {model_dir}")

    frames = []
    for path in sorted(model_dir.glob("patch_stats_fold_*_*.pkl")):
        match = PATCH_STATS_RE.match(path.name)
        if not match:
            continue
        frame = _load_graph_dataframe(path)
        frame = frame.copy()
        frame["fold"] = int(match.group("fold"))
        frame["split"] = match.group("split")
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No patch_stats pickle files found under {model_dir}")

    return pd.concat(frames, ignore_index=True)


def _merge_graph_and_patch_stats(graph_df, patch_df):
    join_cols = ["fold", "split", "image_id"]
    missing_cols = [col for col in join_cols if col not in graph_df.columns or col not in patch_df.columns]
    if missing_cols:
        raise KeyError(f"Missing join columns for graph/patch merge: {missing_cols}")

    merged = graph_df.merge(patch_df, on=join_cols, how="inner", suffixes=("", "_patch"))

    patch_cols = [
        "patch_embeddings",
        "patch_probs",
        "dominant_class",
    ]
    for col in patch_cols:
        patch_col = f"{col}_patch"
        if patch_col in merged.columns:
            merged[col] = merged[patch_col]

    return merged


def _edge_index_from_variant(row, graph_variant):
    match = GRAPH_VARIANT_RE.match(graph_variant)
    if not match:
        raise ValueError(f"Unsupported graph variant: {graph_variant}")

    kind = match.group("kind")
    param = match.group("param")

    if kind == "grid4":
        return np.asarray(row["grid4_edge_index"])
    if kind == "grid8":
        return np.asarray(row["grid8_edge_index"])
    if kind == "knn":
        return np.asarray(row["knn_edge_indices"][int(param)])
    if kind == "random":
        return np.asarray(row["random_edge_indices"][int(param)])
    raise ValueError(f"Unsupported graph variant: {graph_variant}")


# --------------------------------------------------------------------------
# Edge-wise heterophily measures
# --------------------------------------------------------------------------

def _row_cosine_sim(a, b):
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS
    return num / denom


def compute_edge_heterophily(row, graph_variant=None):
    """Compute scalar edge heterophily measures and a class compatibility matrix for one image."""
    if graph_variant is None:
        edge_index = np.asarray(row["edge_index"])
    else:
        edge_index = _edge_index_from_variant(row, graph_variant)

    src, dst = edge_index[0], edge_index[1]
    keep = src != dst  # defensive: strip any self-loops before averaging
    src, dst = src[keep], dst[keep]

    embeddings = np.asarray(row["patch_embeddings"], dtype=np.float32)
    patch_probs = np.asarray(row["patch_probs"], dtype=np.float32)
    dominant_class = np.asarray(row["dominant_class"])

    x_src, y_src = src % 14, src // 14
    x_dst, y_dst = dst % 14, dst // 14
    num_nodes = embeddings.shape[0]
    num_classes = patch_probs.shape[1]
    same_class_edge = dominant_class[src] == dominant_class[dst]

    matching_neighbors = np.bincount(src[same_class_edge], minlength=num_nodes)
    node_degree = np.bincount(src, minlength=num_nodes)
    valid_nodes = node_degree > 0
    if np.any(valid_nodes):
        h_node = float(np.mean(matching_neighbors[valid_nodes] / node_degree[valid_nodes]))
    else:
        h_node = 0.0

    edge_homophily = float(np.mean(same_class_edge)) if len(same_class_edge) > 0 else 0.0
    node_class_counts = np.bincount(dominant_class, minlength=num_classes)
    p_k = node_class_counts / max(1, num_nodes)
    expected_homophily = float(np.sum(p_k ** 2))

    if expected_homophily < 1.0:
        h_adj = (edge_homophily - expected_homophily) / (1.0 - expected_homophily)
    else:
        h_adj = 1.0

    compat_matrix, _, _ = np.histogram2d(
        dominant_class[src],
        dominant_class[dst],
        bins=[np.arange(num_classes + 1), np.arange(num_classes + 1)],
    )
    row_sums = compat_matrix.sum(axis=1, keepdims=True)
    compat_matrix = np.divide(
        compat_matrix,
        row_sums,
        out=np.zeros_like(compat_matrix),
        where=row_sums != 0,
    )

    return {
        "H_f": 1.0 - _row_cosine_sim(embeddings[src], embeddings[dst]),
        "H_e": 1.0 - _row_cosine_sim(patch_probs[src], patch_probs[dst]),
        "H_cls": (dominant_class[src] != dominant_class[dst]).astype(np.float32),
        "H_kl": np.sum(patch_probs[src] * np.log((patch_probs[src] + EPS) / (patch_probs[dst] + EPS)), axis=1),
        "H_dirichlet": 0.5 * np.sum((embeddings[src] - embeddings[dst]) ** 2, axis=1),
        "H_spatial": np.sqrt((x_src - x_dst) ** 2 + (y_src - y_dst) ** 2),
        "H_node": h_node,
        "H_adj": h_adj,
        "H_compat_matrix": compat_matrix,
    }


def _summarize_image(em, meta):
    out = dict(meta)
    out["num_edges"] = len(em["H_f"])
    for m in MEASURES:
        vals = em[m]
        out[f"{m}_mean"] = float(np.mean(vals))
        out[f"{m}_std"] = float(np.std(vals))
        out[f"{m}_median"] = float(np.median(vals))
    out["H_compat_matrix"] = em["H_compat_matrix"]
    return out


# --------------------------------------------------------------------------
# Build master summary of H across all Images, Folds, and Graph Variants
# --------------------------------------------------------------------------

def build_master_summary(root_dir, pattern="graph_dataset.pkl", verbose=True):
    root_dir = Path(root_dir)
    files = sorted(root_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root_dir}")

    all_summaries = []
    for f in files:
        if verbose:
            print(f"Processing {f.name} ...")
            print("Loading:", f)
            print("Size:", f.stat().st_size)
        graph_df = _load_graph_dataframe(f)
        patch_df = _load_patch_stats_dataframe(f.parent.name)
        graph_df = _merge_graph_and_patch_stats(graph_df, patch_df)

        # ---- DEBUG / TEST LIMIT ----
        test_fold = 0
        graph_df = graph_df[graph_df["fold"] == test_fold]

        graph_df = (
            graph_df[graph_df["split"].isin(["train", "val", "test"])]
            .groupby("split", group_keys=False)
            .head(100)
        )
        # ----------------------------

        for _, row in graph_df.iterrows():
            meta_base = {
                "model_name": row["model_name"],
                "fold": int(row["fold"]),
                "split": row["split"],
                "image_id": row["image_id"],
                "label": row.get("label", None),
            }

            variants = ["grid4", "grid8"]
            variants.extend([f"knn{int(k)}" for k in row["knn_edge_indices"].keys()])
            variants.extend([f"random{int(r)}" for r in row["random_edge_indices"].keys()])

            for graph_variant in variants:
                em = compute_edge_heterophily(row, graph_variant=graph_variant)
                meta = {**meta_base, "graph_variant": graph_variant}
                meta["graph_type"] = "grid" if graph_variant.startswith("grid") else ("knn" if graph_variant.startswith("knn") else "random")
                if graph_variant.startswith("knn"):
                    meta["graph_param"] = int(graph_variant[3:])
                elif graph_variant.startswith("random"):
                    meta["graph_param"] = int(graph_variant[6:])
                else:
                    meta["graph_param"] = None
                all_summaries.append(pd.DataFrame([_summarize_image(em, meta)]))

    return pd.concat(all_summaries, ignore_index=True)

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def aggregate_results(df):
    """Aggregate per-image heterophily scores across folds."""
    metric_cols = [f"{m}_mean" for m in MEASURES]

    fold_level = (
        df.groupby(["fold", "split", "graph_variant"], as_index=False)[metric_cols]
        .mean()
    )

    long_df = fold_level.melt(
        id_vars=["fold", "split", "graph_variant"],
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    summary_df = (
        long_df.groupby(["split", "graph_variant", "metric"], as_index=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
        )
        .sort_values(["split", "metric", "graph_variant"])
        .reset_index(drop=True)
    )

    summary_df["std"] = summary_df["std"].fillna(0.0)
    return summary_df


def aggregate_compatibility(df):
    """Aggregate H_compat_matrix across folds for each split and graph_variant."""
    if "H_compat_matrix" not in df.columns:
        raise KeyError("Expected column 'H_compat_matrix' in dataframe")

    fold_rows = []
    for (fold, split, graph_variant), group in df.groupby(["fold", "split", "graph_variant"]):
        matrices = [np.asarray(m) for m in group["H_compat_matrix"].tolist()]
        if not matrices:
            continue
        fold_rows.append(
            {
                "fold": fold,
                "split": split,
                "graph_variant": graph_variant,
                "fold_mean_matrix": np.mean(np.stack(matrices, axis=0), axis=0),
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    if fold_df.empty:
        raise ValueError("No compatibility matrices available to aggregate")

    out_rows = []
    for (split, graph_variant), group in fold_df.groupby(["split", "graph_variant"]):
        matrices = np.stack(group["fold_mean_matrix"].to_list(), axis=0)
        out_rows.append(
            {
                "split": split,
                "graph_variant": graph_variant,
                "mean_matrix": matrices.mean(axis=0),
                "std_matrix": matrices.std(axis=0),
            }
        )

    return pd.DataFrame(out_rows).sort_values(["split", "graph_variant"]).reset_index(drop=True)


def plot_compatibility_matrices(compat_summary_df, split, output_dir):
    """Plot mean compatibility matrices for one split as a 3x6 heatmap grid."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    graph_order = ["grid4", "grid8"]
    graph_order.extend([f"knn{k}" for k in DEFAULT_K_VALUES])
    graph_order.extend([f"random{r}" for r in DEFAULT_R_VALUES])

    split_df = compat_summary_df[compat_summary_df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No compatibility matrices available for split={split!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    last_im = None

    for ax, graph_variant in zip(axes, graph_order):
        row = split_df[split_df["graph_variant"] == graph_variant]
        if row.empty:
            ax.axis("off")
            continue

        mat = np.asarray(row.iloc[0]["mean_matrix"], dtype=float)
        last_im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        ax.set_title(graph_variant, fontsize=11)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_yticks(range(mat.shape[0]))
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[len(graph_order):]:
        ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.tolist(), shrink=0.75, pad=0.01)

    fig.suptitle(f"Compatibility matrices - {split}", fontsize=14)
    fig.tight_layout()

    png_path = output_dir / f"compatibility_{split}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split(df_summary, split, output_dir):
    """Plot the fold-aggregated heterophily curves for one split.

    Parameters
    ----------
    df_summary : pandas.DataFrame
        Output of aggregate_results().
    split : str
        One of {'train', 'val', 'test'}.
    output_dir : str or pathlib.Path
        Directory where the PNG and PDF should be saved.
    """


    graph_order = ["grid4", "grid8"]
    graph_order.extend([f"knn{k}" for k in DEFAULT_K_VALUES])
    graph_order.extend([f"random{r}" for r in DEFAULT_R_VALUES])
    display_labels = ["g4", "g8"]
    display_labels.extend([f"k{k}" for k in DEFAULT_K_VALUES])
    display_labels.extend([f"r{r}" for r in DEFAULT_R_VALUES])
    metric_order = [f"{m}_mean" for m in MEASURES]
    metric_titles = {
        "H_f_mean": "Feature heterophily (H_f)",
        "H_e_mean": "Evidence heterophily (H_e)",
        "H_cls_mean": "Dominant-class heterophily (H_cls)",
        "H_kl_mean": "KL heterophily (H_kl)",
        "H_dirichlet_mean": "Dirichlet heterophily (H_dirichlet)",
        "H_spatial_mean": "Spatial heterophily (H_spatial)",
        "H_node_mean": "Node heterophily (H_node)",
        "H_adj_mean": "Adjusted homophily (H_adj)",
    }
    metric_colors = {
        "H_f_mean": "blue",
        "H_e_mean": "orange",
        "H_cls_mean": "red",
        "H_kl_mean": "purple",
        "H_dirichlet_mean": "brown",
        "H_spatial_mean": "green",
        "H_node_mean": "gray",
        "H_adj_mean": "black",
    }

    split_df = df_summary[df_summary["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No rows available for split={split!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    axes = axes.flatten()
    x_positions = np.arange(len(graph_order))
    family_slices = {
        "grid": slice(0, 2),
        "knn": slice(2, 2 + len(DEFAULT_K_VALUES)),
        "random": slice(2 + len(DEFAULT_K_VALUES), len(graph_order)),
    }

    for ax, metric in zip(axes, metric_order):
        metric_df = split_df[split_df["metric"] == metric].copy()
        metric_df = metric_df.set_index("graph_variant").reindex(graph_order)

        y = metric_df["mean"].to_numpy(dtype=float)
        yerr = metric_df["std"].to_numpy(dtype=float)

        for family, slc in family_slices.items():
            family_x = x_positions[slc]
            family_y = y[slc]
            family_yerr = yerr[slc]
            ax.errorbar(
                family_x,
                family_y,
                yerr=family_yerr,
                color=metric_colors[metric],
                marker="o",
                linewidth=2,
                markersize=6,
                capsize=4,
                alpha=0.9,
            )
        ax.set_title(metric_titles[metric], fontsize=13)
        ax.set_ylabel("Mean heterophily", fontsize=12)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axvline(1.5, linestyle="--", color="gray", linewidth=1, alpha=0.7)
        ax.axvline(11.5, linestyle="--", color="gray", linewidth=1, alpha=0.7)
        ax.set_xlim(-0.5, len(graph_order) - 0.5)

    # Family labels under the x-axis.
    axes[-1].text(
        0.5,
        -0.38,
        "Grid",
        transform=axes[-1].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=12,
    )
    axes[-1].text(
        6.5,
        -0.38,
        "kNN",
        transform=axes[-1].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=12,
    )
    axes[-1].text(
        16.5,
        -0.38,
        "Random",
        transform=axes[-1].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=12,
    )

    fig.tight_layout()

    png_path = output_dir / f"heterophily_{split}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %%
# def main():
graph_root = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/graph_outputs")
figures_root = Path("figures/heterophily")

model_dirs = sorted([p for p in graph_root.iterdir() if p.is_dir()])
if not model_dirs:
    raise FileNotFoundError(f"No model directories found under {graph_root}")

# TODO: remove this line after testing
for model_dir in model_dirs:
    model_dir = model_dirs[0]  # for testing, only process the first model directory
    results_df = build_master_summary(model_dir)
    summary_df = aggregate_results(results_df)
    compat_summary_df = aggregate_compatibility(results_df)

    model_fig_dir = figures_root / model_dir.name
    for split in ["train", "val", "test"]:
        plot_split(summary_df, split, output_dir=model_fig_dir)
        plot_compatibility_matrices(compat_summary_df, split, output_dir=model_fig_dir)
    break  # for testing, only process the first model directory

# if __name__ == "__main__":
#     main()
# %%