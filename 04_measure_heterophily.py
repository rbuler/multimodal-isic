import re
import os
import psutil
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sp
# %%
EPS = 1e-8
DEFAULT_K_VALUES = tuple(range(1, 9)) + (12, 16)
DEFAULT_R_VALUES = tuple(range(1, 9)) + (12, 16)
MEASURES = [
    "H_kl",
    "H_dirichlet",
    "H_spatial",
    "H_adj",
    "lambda_2"
]
GRAPH_VARIANT_RE = re.compile(r"^(?P<kind>grid4|grid8|knn|random)(?P<param>\d+)?$")
PATCH_STATS_RE = re.compile(r"^patch_stats_fold_(?P<fold>\d+)_(?P<split>train|val|test)\.pkl$")
PATCH_STATS_ROOT = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/patch_stats")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    
    "xtick.labelsize": 10,
    "xtick.direction": "in",
    "xtick.major.size": 4,
    
    "ytick.labelsize": 10,
    "ytick.direction": "in",
    "ytick.major.size": 4,
    
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def _load_graph_dataframe(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


def _load_patch_stats_dataframe(model_name, fold, split):
    model_dir = PATCH_STATS_ROOT / model_name

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Missing patch_stats model directory: {model_dir}"
        )

    path = model_dir / f"patch_stats_fold_{fold}_{split}.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing patch_stats file: {path}"
        )

    frame = _load_graph_dataframe(path)
    frame = frame.copy()
    frame["fold"] = int(fold)
    frame["split"] = split

    return frame


def _merge_graph_and_patch_stats(graph_df, patch_df):
    join_cols = ["fold", "split", "image_id"]
    missing_cols = [col for col in join_cols if col not in graph_df.columns or col not in patch_df.columns]
    if missing_cols:
        raise KeyError(f"Missing join columns for graph/patch merge: {missing_cols}")

    merged = graph_df.merge(patch_df, on=join_cols, how="inner", suffixes=("", "_patch"))
    return merged


def _edge_index_from_variant(row, graph_variant):
    match = GRAPH_VARIANT_RE.match(graph_variant)
    if not match:
        raise ValueError(f"Unsupported graph variant: {graph_variant}")

    kind = match.group("kind")
    param = match.group("param")

    if kind == "grid4":
        return np.asarray(row.grid4_edge_index)
    if kind == "grid8":
        return np.asarray(row.grid8_edge_index)
    if kind == "knn":
        return np.asarray(row.knn_edge_indices[int(param)])
    if kind == "random":
        return np.asarray(row.random_edge_indices[int(param)])
    raise ValueError(f"Unsupported graph variant: {graph_variant}")


def compute_edge_heterophily(row, graph_variant=None):
    """Compute scalar edge heterophily measures and a class compatibility matrix for one image."""
    if graph_variant is None:
        edge_index = np.asarray(row.edge_index)
    else:
        edge_index = _edge_index_from_variant(row, graph_variant)

    src, dst = edge_index[0], edge_index[1]
    keep = src != dst  # strip any self-loops before averaging
    src, dst = src[keep], dst[keep]

    embeddings = np.asarray(row.patch_embeddings, dtype=np.float32)
    patch_probs = np.asarray(row.patch_probs, dtype=np.float32)
    dominant_class = np.asarray(row.dominant_class, dtype=np.int32)

    x_src, y_src = src % 14, src // 14
    x_dst, y_dst = dst % 14, dst // 14

    num_nodes = embeddings.shape[0]
    num_classes = patch_probs.shape[1]
    same_class_edge = dominant_class[src] == dominant_class[dst]
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

    data = np.ones_like(src, dtype=np.float32)
    A = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)
    degrees = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(degrees)
    d_inv_sqrt[degrees > 0] = 1.0 / np.sqrt(degrees[degrees > 0])
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L = sp.eye(num_nodes) - D_inv_sqrt.dot(A).dot(D_inv_sqrt)
    eigenvalues = np.linalg.eigvalsh(L.toarray())
    lambda_2 = float(np.sort(eigenvalues)[1]) if len(eigenvalues) > 1 else 0.0

    return {
        "H_kl": np.sum(patch_probs[src] * np.log((patch_probs[src] + EPS) / (patch_probs[dst] + EPS)), axis=1),
        "H_dirichlet": 0.5 * np.sum((embeddings[src] - embeddings[dst]) ** 2, axis=1),
        "H_spatial": np.sqrt((x_src - x_dst) ** 2 + (y_src - y_dst) ** 2),
        "H_adj": h_adj,
        "lambda_2": np.array([lambda_2]),
        "H_compat_matrix": compat_matrix,
    }


def _summarize_image(em, meta):
    out = dict(meta)
    out["num_edges"] = len(em["H_kl"])
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

def build_master_summary(root_dir, pattern="graph_dataset.pkl"):
    root_dir = Path(root_dir)
    files = sorted(root_dir.rglob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} under {root_dir}")

    all_summaries = []

    for f in files:

        graph_df = _load_graph_dataframe(f)

        for (fold, split), graph_group in graph_df.groupby(["fold", "split"]):

            print(f"Processing fold={fold}, split={split}")
            patch_df = _load_patch_stats_dataframe(f.parent.name, int(fold), split)
            merged = _merge_graph_and_patch_stats(graph_group, patch_df)

            for row in merged.itertuples(index=False):
                meta_base = {
                    "model_name": row.model_name,
                    "fold": int(row.fold),
                    "split": row.split,
                    "image_id": row.image_id,
                    "label": getattr(row, "label", None),
                }
                variants = ["grid4", "grid8"]
                variants.extend([f"knn{int(k)}" for k in row.knn_edge_indices.keys()])
                variants.extend([f"random{int(r)}" for r in row.random_edge_indices.keys()])

                for graph_variant in variants:
                    em = compute_edge_heterophily(row, graph_variant=graph_variant)
                    meta = {**meta_base, "graph_variant": graph_variant}

                    meta["graph_type"] = (
                        "grid"
                        if graph_variant.startswith("grid")
                        else (
                            "knn"
                            if graph_variant.startswith("knn")
                            else "random"
                        )
                    )

                    if graph_variant.startswith("knn"):
                        meta["graph_param"] = int(graph_variant[3:])
                    elif graph_variant.startswith("random"):
                        meta["graph_param"] = int(graph_variant[6:])
                    else:
                        meta["graph_param"] = None

                    all_summaries.append(_summarize_image(em, meta))

            del merged
            del patch_df

    return pd.DataFrame.from_records(all_summaries)

# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def aggregate_results(df):
    """Aggregate per-image heterophily scores across folds."""
    metric_cols = [f"{m}_mean" for m in MEASURES] + ["H_kl_std"]

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


def mean_compatibility_matrices(matrices):
    matrices = np.stack(matrices, axis=0)

    row_support = matrices.sum(axis=2) > 0

    result = np.zeros(matrices.shape[1:], dtype=float)

    for cls in range(matrices.shape[1]):
        valid = row_support[:, cls]

        if np.any(valid):
            result[cls] = matrices[valid, cls].mean(axis=0)

    return result


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
                "fold_mean_matrix": mean_compatibility_matrices(matrices),
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
                "mean_matrix": mean_compatibility_matrices(matrices),
                "std_matrix": matrices.std(axis=0),
            }
        )

    return pd.DataFrame(out_rows).sort_values(["split", "graph_variant"]).reset_index(drop=True)


def plot_compatibility_matrices(compat_summary_df, split, output_dir):
    graph_order = ["grid4", "grid8"]
    graph_order.extend([f"knn{k}" for k in DEFAULT_K_VALUES])
    graph_order.extend([f"random{r}" for r in DEFAULT_R_VALUES])

    title_map = {"grid4": "Grid-4", "grid8": "Grid-8"}
    title_map.update({f"knn{k}": rf"$k={k}$" for k in DEFAULT_K_VALUES})
    title_map.update({f"random{r}": rf"$r={r}$" for r in DEFAULT_R_VALUES})

    split_df = compat_summary_df[compat_summary_df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(
            f"No compatibility matrices available for split={split!r}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(20, 10),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )
    axes_flat = axes.flatten()
    last_im = None

    for ax, graph_variant in zip(axes_flat, graph_order):
        ax.grid(False)

        row = split_df[split_df["graph_variant"] == graph_variant]
        if row.empty:
            ax.axis("off")
            continue

        mat = np.asarray(row.iloc[0]["mean_matrix"], dtype=float)
        num_classes = mat.shape[0]

        last_im = ax.imshow(
            mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal"
        )

        ax.set_title(
            title_map.get(graph_variant, graph_variant),
            fontsize=11,
            fontweight="bold",
            pad=6,
        )
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(range(num_classes), fontsize=8)
        ax.set_yticklabels(range(num_classes), fontsize=8)

        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=3,
            top=False,
            right=False,
        )

    for ax in axes_flat[len(graph_order) :]:
        ax.axis("off")

    fig.supxlabel(
        r"Target Class ($y_{\mathrm{dst}}$)",
        fontsize=12,
        fontweight="bold",
        y=0.03,
    )
    fig.supylabel(
        r"Source Class ($y_{\mathrm{src}}$)",
        fontsize=12,
        fontweight="bold",
        x=0.03,
    )

    if last_im is not None:
        cax = fig.add_axes([0.91, 0.12, 0.012, 0.73])
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label(
            r"Transition Probability $P(y_{\mathrm{dst}} \mid y_{\mathrm{src}})$",
            fontsize=11,
        )
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(
        rf"Class Compatibility Matrices $\mathbf{{H}}$ ({split.capitalize()} Split)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    fig.subplots_adjust(left=0.06, right=0.89, top=0.91, bottom=0.08)
    png_path = output_dir / f"compatibility_{split}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split(df_summary, split, output_dir):
    graph_order = ["grid4", "grid8"]
    graph_order.extend([f"knn{k}" for k in DEFAULT_K_VALUES])
    graph_order.extend([f"random{r}" for r in DEFAULT_R_VALUES])
    display_labels = ["Grid-4", "Grid-8"]
    display_labels.extend([rf"$k={k}$" for k in DEFAULT_K_VALUES])
    display_labels.extend([rf"$r={r}$" for r in DEFAULT_R_VALUES])
    metric_order = [
        "H_kl_mean",
        "H_kl_std",
        "H_dirichlet_mean",
        "H_spatial_mean",
        "H_adj_mean",
        "lambda_2_mean"
    ]
    metric_titles = {
        "H_kl_mean": r"KL Heterophily ($H_{\mathrm{KL}}$)",
        "H_kl_std": r"KL Edge Non-Uniformity ($\sigma_{H_{\mathrm{KL}}}$)",
        "H_dirichlet_mean": r"Dirichlet Energy ($H_{\mathrm{Dirichlet}}$)",
        "H_spatial_mean": r"Spatial Distance ($H_{\mathrm{spatial}}$)",
        "H_adj_mean": r"Adjusted Homophily ($H_{\mathrm{adj}}$)",
        "lambda_2_mean": r"Algebraic Connectivity ($\lambda_2$)",
    }
    metric_colors = {
        "H_kl_mean": "purple",
        "H_kl_std": "rebeccapurple",
        "H_dirichlet_mean": "brown",
        "H_spatial_mean": "green",
        "H_adj_mean": "black",
        "lambda_2_mean": "orange",
    }

    y_labels = {
    "H_kl_mean": r"Prediction Divergence $D_{\mathrm{KL}}$",
    "H_kl_std": r"Edge Heterogeneity $\sigma(D_{\mathrm{KL}})$",
    "H_dirichlet_mean": r"Dirichlet Energy $\mathcal{E}_{\mathrm{Dir}}$",
    "H_spatial_mean": r"Spatial Distance $d_{\mathrm{spatial}}$ [patches]",
    "H_adj_mean": r"Adjusted Homophily $h_{\mathrm{adj}}$",
    "lambda_2_mean": r"Algebraic Connectivity $\lambda_2(\mathbf{L})$",
    }

    split_df = df_summary[df_summary["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No rows available for split={split!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
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
        ax.set_title(metric_titles[metric], fontweight="bold")
        ax.set_ylabel(y_labels[metric], fontsize=11)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=10)
        ax.tick_params(top=True, right=True, which="both")
        ax.grid(True, linestyle="--", alpha=0.3)

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

for model_dir in model_dirs:
    results_df = build_master_summary(model_dir)
    summary_df = aggregate_results(results_df)
    compat_summary_df = aggregate_compatibility(results_df)

    summary_out_path = model_dir / "heterophily_summary.pkl"
    compat_out_path = model_dir / "compatibility_summary.pkl"
    results_out_path = model_dir / "heterophily_results.pkl"

    with open(summary_out_path, "wb") as f:
        pickle.dump(summary_df, f)
    with open(compat_out_path, "wb") as f:
        pickle.dump(compat_summary_df, f)
    with open(results_out_path, "wb") as f:
        pickle.dump(results_df, f)

    model_fig_dir = figures_root / model_dir.name
    for split in ["train", "val", "test"]:
        plot_split(summary_df, split, output_dir=model_fig_dir)
        plot_compatibility_matrices(compat_summary_df, split, output_dir=model_fig_dir)

# if __name__ == "__main__":
#     main()
# %%