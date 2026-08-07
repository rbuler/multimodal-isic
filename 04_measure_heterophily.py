import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
# %%
EPS = 1e-8
MEASURES = ["H_f", "H_e", "H_ent", "H_cls", "H_conf", "H_att"]
GRAPH_VARIANT_RE = re.compile(r"^(?P<kind>grid4|grid8|knn|random)(?P<param>\d+)?$")


def _load_graph_dataframe(path):
    with open(path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    return df


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
    """All six edge-wise measures for one image (one DataFrame row)."""
    if graph_variant is None:
        edge_index = np.asarray(row["edge_index"])
    else:
        edge_index = _edge_index_from_variant(row, graph_variant)
    src, dst = edge_index[0], edge_index[1]
    keep = src != dst  # defensive: strip any self-loops before averaging
    src, dst = src[keep], dst[keep]

    embeddings = np.asarray(row["patch_embeddings"], dtype=np.float32)
    patch_probs = np.asarray(row["patch_probs"], dtype=np.float32)
    entropy = np.asarray(row["entropy"], dtype=np.float32)
    confidence = np.asarray(row["confidence"], dtype=np.float32)
    dominant_class = np.asarray(row["dominant_class"])
    attention = np.asarray(row["attention"], dtype=np.float32)

    return {
        "H_f": 1.0 - _row_cosine_sim(embeddings[src], embeddings[dst]),
        "H_e": 1.0 - _row_cosine_sim(patch_probs[src], patch_probs[dst]),
        "H_ent": np.abs(entropy[src] - entropy[dst]),
        "H_cls": (dominant_class[src] != dominant_class[dst]).astype(np.float32),
        "H_conf": np.abs(confidence[src] - confidence[dst]),
        "H_att": np.abs(attention[src] - attention[dst]),
    }


def _summarize_image(em, meta):
    out = dict(meta)
    out["num_edges"] = len(em["H_f"])
    for m in MEASURES:
        vals = em[m]
        out[f"{m}_mean"] = float(np.mean(vals))
        out[f"{m}_std"] = float(np.std(vals))
        out[f"{m}_median"] = float(np.median(vals))
    return out


# --------------------------------------------------------------------------
# Corpus-level orchestration
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
        graph_df = _load_graph_dataframe(f)

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
    """Aggregate per-image heterophily scores across folds.

    The input dataframe is expected to contain at least:
        fold, split, graph_variant,
        H_f_mean, H_e_mean, H_ent_mean, H_cls_mean, H_conf_mean, H_att_mean

    We first average within each fold, then compute mean/std across the five folds
    for each split, graph_variant, and metric.
    """
    metric_cols = [
        "H_f_mean",
        "H_e_mean",
        "H_ent_mean",
        "H_cls_mean",
        "H_conf_mean",
        "H_att_mean",
    ]

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
    import matplotlib.pyplot as plt
    from pathlib import Path

    graph_order = [
        "grid4",
        "grid8",
        "knn1",
        "knn2",
        "knn3",
        "knn4",
        "knn5",
        "knn6",
        "knn7",
        "knn8",
        "random1",
        "random2",
        "random3",
        "random4",
        "random5",
        "random6",
        "random7",
        "random8",
    ]
    metric_order = [
        "H_f_mean",
        "H_e_mean",
        "H_ent_mean",
        "H_cls_mean",
        "H_conf_mean",
        "H_att_mean",
    ]
    metric_titles = {
        "H_f_mean": "Feature heterophily (H_f)",
        "H_e_mean": "Evidence heterophily (H_e)",
        "H_ent_mean": "Entropy heterophily (H_ent)",
        "H_cls_mean": "Dominant-class heterophily (H_cls)",
        "H_conf_mean": "Confidence heterophily (H_conf)",
        "H_att_mean": "Attention heterophily (H_att)",
    }
    metric_colors = {
        "H_f_mean": "blue",
        "H_e_mean": "orange",
        "H_ent_mean": "green",
        "H_cls_mean": "red",
        "H_conf_mean": "purple",
        "H_att_mean": "brown",
    }

    split_df = df_summary[df_summary["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No rows available for split={split!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    x_positions = np.arange(len(graph_order))

    for ax, metric in zip(axes, metric_order):
        metric_df = split_df[split_df["metric"] == metric].copy()
        metric_df = metric_df.set_index("graph_variant").reindex(graph_order)

        y = metric_df["mean"].to_numpy(dtype=float)
        yerr = metric_df["std"].to_numpy(dtype=float)

        ax.errorbar(
            x_positions,
            y,
            yerr=yerr,
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
        ax.set_xticklabels(graph_order, rotation=45, ha="right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.axvline(1.5, linestyle="--", color="gray", linewidth=1, alpha=0.7)
        ax.axvline(9.5, linestyle="--", color="gray", linewidth=1, alpha=0.7)
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
        5.5,
        -0.38,
        "kNN",
        transform=axes[-1].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=12,
    )
    axes[-1].text(
        13.5,
        -0.38,
        "Random",
        transform=axes[-1].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=12,
    )

    fig.tight_layout()

    png_path = output_dir / f"heterophily_{split}.png"
    pdf_path = output_dir / f"heterophily_{split}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# %%
# def main():
graph_root = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/graph_outputs")
figures_root = Path("figures/heterophily")

model_dirs = sorted([p for p in graph_root.iterdir() if p.is_dir()])
if not model_dirs:
    raise FileNotFoundError(f"No model directories found under {graph_root}")

model_dirs = model_dirs[:1]  # For testing, only process the first model directory
for model_dir in model_dirs:
    results_df = build_master_summary(model_dir)
    summary_df = aggregate_results(results_df)

    model_fig_dir = figures_root / model_dir.name
    for split in ["train", "val", "test"]:
        plot_split(summary_df, split, output_dir=model_fig_dir)


# if __name__ == "__main__":
#     main()
# %%