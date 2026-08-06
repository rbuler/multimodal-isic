import re
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
# %%
EPS = 1e-8
MEASURES = ["H_f", "H_e", "H_ent", "H_cls", "H_conf", "H_att"]
MEASURE_LABELS = {
    "H_f": "Feature (embedding cosine)",
    "H_e": "Evidence (prob cosine)",
    "H_ent": "Entropy disagreement",
    "H_cls": "Dominant-class mismatch",
    "H_conf": "Confidence disagreement",
    "H_att": "Attention disagreement",
}

FNAME_RE = re.compile(
    r"patch_stats_fold_(?P<fold>\d+)_(?P<split>train|val|test)_(?P<gtype>grid4|grid8|knn\d+|random\d+)\.pkl$"
)

GRAPH_VARIANT_RE = re.compile(r"^(?P<kind>grid4|grid8|knn|random)(?P<param>\d+)?$")


def parse_filename(path):
    m = FNAME_RE.search(Path(path).name)
    if not m:
        raise ValueError(
            f"Filename doesn't match expected pattern "
            f"'patch_stats_fold_<N>_<split>_<gtype>.pkl': {path}"
        )
    gtype = m.group("gtype")
    if gtype.startswith("knn"):
        graph_type = "knn"
        graph_param = int(gtype[3:])
    elif gtype.startswith("random"):
        graph_type = "random"
        graph_param = int(gtype[6:])
    else:
        graph_type = gtype
        graph_param = None
    return {
        "fold": int(m.group("fold")),
        "split": m.group("split"),
        "graph_type": graph_type,
        "graph_param": graph_param,
        "graph_variant": gtype,
    }


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
# Per-file processing
# --------------------------------------------------------------------------

def process_file(path, pool_edges=False, max_pool_per_file=200_000, rng=42, graph_variant=None):
    meta_file = parse_filename(path)
    df = _load_graph_dataframe(path)

    rng = rng or np.random.default_rng(0)
    summary_rows = []
    pooled_chunks = {m: [] for m in MEASURES} if pool_edges else None

    for _, row in df.iterrows():
        em = compute_edge_heterophily(row, graph_variant=graph_variant or meta_file["graph_variant"])
        meta = {**meta_file, "image_id": row["image_id"]}
        if "label" in row:
            meta["label"] = row["label"]
        summary_rows.append(_summarize_image(em, meta))
        if pool_edges:
            for m in MEASURES:
                pooled_chunks[m].append(em[m])

    summary_df = pd.DataFrame(summary_rows)

    pooled_arrays = None
    if pool_edges:
        pooled_arrays = {}
        for m in MEASURES:
            arr = np.concatenate(pooled_chunks[m]) if pooled_chunks[m] else np.array([])
            if len(arr) > max_pool_per_file:
                idx = rng.choice(len(arr), size=max_pool_per_file, replace=False)
                arr = arr[idx]
            pooled_arrays[m] = arr

    return summary_df, pooled_arrays


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


def collect_pooled_edges(root_dir, model_name, fold, split,
                          graph_types=("grid4", "grid8", "knn1", "random1"),
                          max_pool_per_file=200_000, seed=0):

    root_dir = Path(root_dir)
    rng = np.random.default_rng(seed)
    pooled_by_gtype = {}
    graph_path = root_dir / model_name / "graph_dataset.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph dataset: {graph_path}")

    graph_df = _load_graph_dataframe(graph_path)
    graph_df = graph_df[(graph_df["fold"] == fold) & (graph_df["split"] == split)]

    for gtype in graph_types:
        pooled_chunks = {m: [] for m in MEASURES}
        for _, row in graph_df.iterrows():
            if not row["graph_variant"] == gtype:
                continue
            em = compute_edge_heterophily(row, graph_variant=gtype)
            for m in MEASURES:
                pooled_chunks[m].append(em[m])
        pooled = {}
        for m in MEASURES:
            arr = np.concatenate(pooled_chunks[m]) if pooled_chunks[m] else np.array([])
            if len(arr) > max_pool_per_file:
                idx = rng.choice(len(arr), size=max_pool_per_file, replace=False)
                arr = arr[idx]
            pooled[m] = arr
        pooled_by_gtype[gtype] = pooled
    return pooled_by_gtype


def leaderboard(master_df, group_cols=("split", "graph_type")):
    """Mean heterophily per group (e.g. per split x graph_type), for every
    measure. Answers 'which graph construction induces the strongest
    evidence heterophily, and does that hold across splits?'"""
    cols = [f"{m}_mean" for m in MEASURES]
    return master_df.groupby(list(group_cols))[cols].mean().round(5)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def aggregate_results(df):
    metric_cols = [
        "H_f_mean",
        "H_e_mean",
        "H_ent_mean",
        "H_cls_mean",
        "H_conf_mean",
        "H_att_mean",
    ]

    long_df = df.melt(
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


def plot_split(summary_df, split, output_dir):
    import matplotlib.pyplot as plt
    from pathlib import Path

    metric_order = [
        "H_f_mean",
        "H_e_mean",
        "H_ent_mean",
        "H_cls_mean",
        "H_conf_mean",
        "H_att_mean",
    ]
    metric_titles = {
        "H_f_mean": "Feature heterophily",
        "H_e_mean": "Evidence heterophily",
        "H_ent_mean": "Entropy heterophily",
        "H_cls_mean": "Class heterophily",
        "H_conf_mean": "Confidence heterophily",
        "H_att_mean": "Attention heterophily",
    }
    metric_colors = {
        "H_f_mean": "blue",
        "H_e_mean": "orange",
        "H_ent_mean": "green",
        "H_cls_mean": "red",
        "H_conf_mean": "purple",
        "H_att_mean": "brown",
    }
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
    x_pos = np.arange(len(graph_order))

    split_df = summary_df[summary_df["split"] == split].copy()
    if split_df.empty:
        raise ValueError(f"No rows found for split={split!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metric_order):
        metric_df = split_df[split_df["metric"] == metric].copy()
        metric_df = metric_df.set_index("graph_variant").reindex(graph_order)

        y = metric_df["mean"].to_numpy(dtype=float)
        yerr = metric_df["std"].to_numpy(dtype=float)

        ax.errorbar(
            x_pos,
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
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.axvline(1.5, linestyle="--", color="gray", linewidth=1, alpha=0.6)
        ax.axvline(9.5, linestyle="--", color="gray", linewidth=1, alpha=0.6)
        ax.set_xlim(-0.5, len(graph_order) - 0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(graph_order, rotation=45, ha="right", fontsize=10)

    family_centers = {
        "Grid": 0.5,
        "kNN": 5.5,
        "Random": 13.5,
    }
    for label, xpos in family_centers.items():
        axes[-1].text(
            xpos,
            -0.38,
            label,
            transform=axes[-1].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
        )

    fig.tight_layout()

    png_path = output_dir / f"heterophily_{split}.png"
    pdf_path = output_dir / f"heterophily_{split}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)



# MEASURES = ["H_f", "H_e", "H_ent", "H_cls", "H_conf", "H_att"]
ROOT = "graph_outputs/c72210e208974529927e6c53d8ec890c"

master_df = build_master_summary(ROOT)
master_df.to_csv("heterophily_master_summary.csv", index=False)
print(leaderboard(master_df, group_cols=("split", "graph_variant")))                 # mean per (split, graph_type)

summary_df = aggregate_results(master_df)

for split in ["train", "val", "test"]:
    plot_split(summary_df, split, output_dir="figures/heterophily")



# %%
# TODO
# sprawdzic kolejnosc patch embeddings i potrzebe sortowania!!!!!!!!!!! wazne!