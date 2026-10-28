"""
GSRP means Graph Structural and Representation Properties
"""

from __future__ import annotations
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import linregress, spearmanr

# %%

GSRP_LABELS = {
    "H_adj_mean": "Adjusted label homophily",
    "H_kl_mean": "Prediction KL discrepancy",
    "H_dirichlet_mean": "Embedding Dirichlet energy",
    "lambda_2_mean": "Algebraic connectivity",
}
GSRP_SHORT_LABELS = {
    "H_adj_mean": "Adjusted homophily",
    "H_kl_mean": "Prediction KL",
    "H_dirichlet_mean": "Dirichlet energy",
    "lambda_2_mean": "Algebraic connectivity",
}
ARCHITECTURE_LABELS = {
    "gin": "GIN",
    "gcn": "GCN",
    "gcnii": "GCNII",
    "fagcn": "FAGCN",
    "gat": "GAT",
    "gatv2": "GATv2",
    "graphsage": "GSAGE",
    "transformer": "GT",
}
PRIMARY_GSRPS = ["H_adj_mean", "H_kl_mean", "H_dirichlet_mean", "lambda_2_mean"]
GRAPH_VARIANT_RE = re.compile(r"^(grid4|grid8|knn|random)(\d+)?$")


def graph_variant_sort_key(variant: str) -> tuple:
    """Sort graph variants by family and numeric parameter, not text order."""
    variant = str(variant)
    match = GRAPH_VARIANT_RE.match(variant)
    if not match:
        return (99, 0, variant)

    family, parameter = match.groups()
    if family.startswith("grid"):
        return (0, int(family[4:]), "")
    if family == "knn":
        return (1, int(parameter), "")
    return (2, int(parameter), "")


@dataclass
class Config:
    graph_root: Path = Path("graph_outputs")
    performance_csv: Path = Path("gnn_results/_3master_results.csv")
    detailed_performance_csv: Path = Path("gnn_results/_3master_detailed_fold_results.csv")
    model_name: str | None = None
    split: str = "test"
    output_dir: Path = Path("figures/gsrp_analysis")
    ba_column: str = "test_bacc_mean"
    seed: int = 20261016
    bootstrap_iterations: int = 2000
    show_trend_lines: bool = True


def inspect_object(value: Any, path: str = "root", depth: int = 0, max_depth: int = 4) -> list[dict[str, Any]]:
    """Recursively describe serialized objects without dumping large arrays."""
    record: dict[str, Any] = {"path": path, "type": type(value).__name__}
    if isinstance(value, pd.DataFrame):
        record.update({"shape": list(value.shape), "columns": value.columns.tolist()})
    elif isinstance(value, np.ndarray):
        record.update({"shape": list(value.shape), "dtype": str(value.dtype)})
    elif isinstance(value, dict):
        record["keys"] = [str(k) for k in list(value)[:50]]
    elif isinstance(value, (list, tuple)):
        record["length"] = len(value)
    records = [record]
    if depth >= max_depth:
        return records
    if isinstance(value, dict):
        for key, child in list(value.items())[:20]:
            records.extend(inspect_object(child, f"{path}[{key!r}]", depth + 1, max_depth))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value[:10]):
            records.extend(inspect_object(child, f"{path}[{index}]", depth + 1, max_depth))
    return records


def load_gsrp_summary(path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GSRP summary: {path}")
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    inspection = inspect_object(obj)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame in {path}, found {type(obj).__name__}")
    required = {"split", "graph_variant", "metric", "mean", "std"}
    missing = required - set(obj.columns)
    if missing:
        raise ValueError(f"GSRP summary is missing columns: {sorted(missing)}")
    frame = obj.copy()
    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    frame["std"] = pd.to_numeric(frame["std"], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean"]), inspection


def load_performance(path: Path, ba_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing performance CSV: {path}")
    frame = pd.read_csv(path)
    required = {"embedding_model", "graph_variant", "graph_model", "seed", "num_layers", ba_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Performance CSV is missing columns: {sorted(missing)}")
    frame = frame.copy()
    for column in ["seed", "num_layers", ba_column]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required))


def variant_metadata(variants: pd.Series) -> pd.DataFrame:
    rows = []
    for variant in sorted(
        variants.dropna().astype(str).unique(),
        key=graph_variant_sort_key,
    ):
        match = GRAPH_VARIANT_RE.match(variant)
        if not match:
            continue
        family, parameter = match.groups()
        if family.startswith("grid"):
            family_name, density = "grid", int(family[4:])
        else:
            family_name, density = family, int(parameter)
        rows.append({
            "graph_variant": variant,
            "graph_family": family_name,
            "graph_parameter": density,
            "graph_density_proxy": density,
        })
    return pd.DataFrame(rows)


def prepare_analysis_data(config: Config) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    performance_path = (
        config.detailed_performance_csv
        if config.detailed_performance_csv.exists()
        else config.performance_csv
    )
    performance_column = "test_bacc" if performance_path == config.detailed_performance_csv else config.ba_column
    performance = load_performance(performance_path, performance_column)
    performance["performance_value"] = performance[performance_column]
    performance_schema = {
        "path": str(performance_path),
        "type": "DataFrame",
        "shape": list(performance.shape),
        "columns": performance.columns.tolist(),
        "dtypes": performance.dtypes.astype(str).to_dict(),
        "unique_values": {
            column: performance[column].drop_duplicates().tolist()[:100]
            for column in ["embedding_model", "graph_model", "graph_variant", "seed", "fold", "split"]
            if column in performance
        },
    }
    models = sorted(performance["embedding_model"].dropna().astype(str).unique())
    model_name = str(config.model_name) if config.model_name is not None else None
    if model_name is None:
        if len(models) != 1:
            raise ValueError("Multiple embedding models are present; use --model-name: " + ", ".join(models))
        model_name = models[0]
    config.model_name = model_name
    performance = performance[performance["embedding_model"].astype(str).eq(model_name)].copy()
    if performance.empty:
        raise ValueError(f"No performance rows for embedding model {model_name!r}")

    gsrp_path = config.graph_root / model_name / "heterophily_summary.pkl"
    gsrp, inspection = load_gsrp_summary(gsrp_path)
    inspection.append(performance_schema)
    variants = set(gsrp["graph_variant"].astype(str))
    performance = performance[performance["graph_variant"].isin(variants | {"none"})].copy()
    performance["architecture"] = performance["graph_model"].astype(str)

    gsrp = gsrp[gsrp["split"].astype(str).eq(config.split)].copy()
    selected = [metric for metric in PRIMARY_GSRPS if metric in set(gsrp["metric"])]
    if not selected:
        raise ValueError(f"No selected GSRPs found. Available: {sorted(gsrp['metric'].unique())}")
    gsrp_wide = gsrp[gsrp["metric"].isin(selected)].pivot_table(
        index="graph_variant", columns="metric", values=["mean", "std"], aggfunc="first"
    )
    gsrp_wide.columns = [f"{value}_{metric}" for value, metric in gsrp_wide.columns]
    gsrp_wide = gsrp_wide.reset_index().merge(
        variant_metadata(gsrp["graph_variant"]), on="graph_variant", how="left"
    )
    gsrp_wide["_sort_order"] = gsrp_wide["graph_variant"].map(graph_variant_sort_key)
    gsrp_wide = (
        gsrp_wide.sort_values("_sort_order", kind="stable")
        .drop(columns="_sort_order")
        .reset_index(drop=True)
    )
    return performance, gsrp_wide, inspection


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if valid.any():
        order = np.argsort(values[valid])
        ranked = values[valid][order]
        corrected = np.minimum.accumulate(
            (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
        )[::-1]
        result = np.empty(len(ranked))
        result[order] = np.clip(corrected, 0.0, 1.0)
        adjusted[valid] = result
    return pd.Series(adjusted, index=p_values.index)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, iterations: int) -> tuple[float, float, float]:
    observed = float(spearmanr(x, y).statistic)
    if len(x) < 4:
        return observed, np.nan, np.nan
    estimates = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), len(x))
        if len(np.unique(x[indices])) < 2 or len(np.unique(y[indices])) < 2:
            continue
        estimates.append(float(spearmanr(x[indices], y[indices]).statistic))
    if not estimates:
        return observed, np.nan, np.nan
    return observed, float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def correlation_analysis(performance: pd.DataFrame, gsrp: pd.DataFrame, config: Config) -> pd.DataFrame:
    key_columns = ["embedding_model", "seed", "fold", "num_layers"]
    available_keys = [column for column in key_columns if column in performance.columns]
    if "fold" not in available_keys:
        raise ValueError("Detailed fold results are required for matched GNN-minus-MLP analysis")
    gnn = performance[~performance["architecture"].eq("mlp")].copy()
    mlp = performance[performance["architecture"].eq("mlp")].copy()
    mlp = mlp[available_keys + ["performance_value"]].rename(
        columns={"performance_value": "mlp_bacc"}
    )
    paired = gnn.merge(mlp, on=available_keys, how="inner", validate="many_to_one")
    paired["delta_bacc"] = paired["performance_value"] - paired["mlp_bacc"]
    advantage = (
        paired.groupby(["graph_variant", "architecture", "num_layers"], as_index=False)
        .agg(
            delta_bacc=("delta_bacc", "mean"),
            delta_bacc_std=("delta_bacc", "std"),
            n_pairs=("delta_bacc", "size"),
        )
        .merge(variant_metadata(paired["graph_variant"]), on="graph_variant", how="left")
    )
    gsrp_columns = ["graph_variant"] + [
        f"mean_{metric}" for metric in PRIMARY_GSRPS
        if f"mean_{metric}" in gsrp.columns
    ]
    merged = advantage.merge(
        gsrp[gsrp_columns],
        on="graph_variant",
        how="inner",
    )
    rows = []
    rng = np.random.default_rng(config.seed)
    for (architecture, num_layers), group in merged.groupby(["architecture", "num_layers"]):
        for scope, family in [("pooled", "all"), ("within_family", "grid"),
                              ("within_family", "knn"), ("within_family", "random")]:
            scoped_group = group if scope == "pooled" else group[group["graph_family"].eq(family)]
            for metric in PRIMARY_GSRPS:
                property_column = f"mean_{metric}"
                if property_column not in scoped_group:
                    continue
                usable = scoped_group[[property_column, "delta_bacc"]].dropna()
                if len(usable) < 4 or usable[property_column].nunique() < 2 or usable["delta_bacc"].nunique() < 2:
                    continue
                rho, ci_low, ci_high = bootstrap_spearman(
                    usable[property_column].to_numpy(), usable["delta_bacc"].to_numpy(), rng, config.bootstrap_iterations
                )
                p_value = float(spearmanr(usable[property_column], usable["delta_bacc"]).pvalue)
                rows.append({
                    "analysis": "exploratory_gsrp_delta_bacc_correlation",
                    "architecture": architecture,
                    "num_layers": num_layers,
                    "scope": scope,
                    "graph_family": family,
                    "gsrp": metric,
                    "gsrp_name": GSRP_LABELS.get(metric, metric),
                    "n_graph_variants": len(usable),
                    "n_pairs": int(scoped_group["n_pairs"].sum()),
                    "n_folds": int(performance["fold"].nunique()),
                    "n_seeds": int(performance["seed"].nunique()),
                    "mean_delta_bacc_sd": float(scoped_group["delta_bacc_std"].mean()),
                    "rho_spearman": rho,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "unit_of_analysis": "distinct graph variants; fold- and seed-matched GNN-minus-MLP test BAcc aggregated per graph variant",
                    "note": "Pooled results may reflect differences between graph families; within-family results are exploratory. Grid is omitted when fewer than four variants are available.",
                })
    results = pd.DataFrame(rows)
    if results.empty:
        return results
    results["p_fdr_bh"] = results.groupby("scope")["p_value"].transform(benjamini_hochberg)
    results["interpretation"] = np.select(
        [results["rho_spearman"] > 0.3, results["rho_spearman"] < -0.3],
        ["positive association", "negative association"],
        default="weak or negligible monotonic association",
    )
    return results


def make_correlation_heatmap(correlations: pd.DataFrame, config: Config) -> None:
    if correlations.empty:
        return
    heatmap = correlations[correlations["scope"].eq("pooled")].pivot_table(
        index="gsrp_name", columns=["architecture", "num_layers"], values="rho_spearman"
    )
    architecture_order = ["gin", "gcn", "gcnii", "fagcn", "gat", "gatv2", "graphsage", "transformer"]
    columns = [
        (architecture, layer)
        for architecture in architecture_order
        for layer in [1, 2, 3, 4, 5]
        if (architecture, layer) in heatmap.columns
    ]
    heatmap = heatmap.reindex(columns=columns)
    heatmap.columns = [
        f"{ARCHITECTURE_LABELS.get(architecture, architecture)}\nL{layer}"
        for architecture, layer in heatmap.columns
    ]
    heatmap.index = [GSRP_SHORT_LABELS.get(label, label) for label in heatmap.index]
    cell_width = 0.72
    cell_height = 0.72
    fig_width = max(18, len(heatmap.columns) * cell_width + 2.4)
    fig_height = max(5.5, len(heatmap.index) * cell_height + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heatmap,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Spearman correlation with Delta BAcc"},
    )
    ax.set_xlabel("GNN architecture and number of layers")
    ax.set_ylabel("GSRP measure")
    ax.set_title("GSRP measures versus GNN advantage over MLP")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8, pad=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10, pad=6)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.24, top=0.86)
    fig.savefig(config.output_dir / "figure3_delta_bacc_correlation_heatmap.png", dpi=400)
    plt.close(fig)


def descriptive_tables(performance: pd.DataFrame, gsrp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = [f"mean_{m}" for m in PRIMARY_GSRPS if f"mean_{m}" in gsrp]
    property_long = gsrp.melt(
        id_vars=["graph_variant", "graph_family", "graph_parameter"],
        value_vars=value_columns, var_name="gsrp_column", value_name="value"
    )
    property_long["gsrp"] = property_long["gsrp_column"].str.replace("mean_", "", regex=False)
    property_summary = property_long.groupby(["graph_variant", "gsrp"], as_index=False)["value"].agg(
        mean="mean", std="std", median="median", q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75), n="count"
    )
    performance_summary = performance.groupby(["graph_variant", "architecture"], as_index=False)["performance_value"].agg(
        mean_ba="mean", std_ba="std", median_ba="median", n="count"
    )
    return property_summary, performance_summary


def make_figure(performance: pd.DataFrame, gsrp: pd.DataFrame, config: Config) -> None:
    pair_keys = ["embedding_model", "seed", "fold", "num_layers"]
    gnn = performance[~performance["architecture"].eq("mlp")].copy()
    mlp = performance[performance["architecture"].eq("mlp")][
        pair_keys + ["performance_value"]
    ].rename(columns={"performance_value": "mlp_bacc"})
    paired = gnn.merge(mlp, on=pair_keys, how="inner", validate="many_to_one")
    paired["delta_bacc"] = paired["performance_value"] - paired["mlp_bacc"]
    plot_data = (
        paired.groupby(["graph_variant", "architecture", "num_layers"], as_index=False)
        .agg(delta_bacc=("delta_bacc", "mean"))
        .merge(variant_metadata(paired["graph_variant"]), on="graph_variant", how="left")
        .merge(
            gsrp[["graph_variant"] + [f"mean_{metric}" for metric in PRIMARY_GSRPS]],
            on="graph_variant",
            how="inner",
        )
    )
    missing_metrics = [
        metric for metric in PRIMARY_GSRPS
        if f"mean_{metric}" not in plot_data.columns
    ]
    if missing_metrics:
        raise ValueError(f"Missing GSRPs for plotting: {missing_metrics}")

    plot_data["_variant_order"] = plot_data["graph_variant"].map(graph_variant_sort_key)
    plot_data = plot_data.sort_values(
        ["architecture", "num_layers", "graph_family", "_variant_order"]
    )

    architectures = sorted(plot_data["architecture"].unique())
    architecture_colors = dict(zip(
        architectures,
        sns.color_palette("colorblind", n_colors=len(architectures)),
    ))
    family_order = ["grid", "knn", "random"]
    family_markers = {"grid": "s", "knn": "^", "random": "o"}
    family_colors = {"grid": "#0072B2", "knn": "#D55E00", "random": "#009E73"}
    family_titles = {"grid": "Grid", "knn": "kNN", "random": "Random"}
    layer_order = [1, 2, 3, 4, 5]
    layer_styles = {
        1: "-",
        2: "--",
        3: "-.",
        4: ":",
        5: (0, (5, 1, 1, 1)),
    }
    layer_sizes = {1: 38, 2: 50, 3: 62, 4: 74, 5: 86}
    family_handles = [
        Line2D(
            [0], [0], color=family_colors[family], marker=family_markers[family],
            linewidth=1.5, markersize=7, label=family_titles[family],
        )
        for family in family_order
    ]
    layer_handles = [
        Line2D(
            [0], [0], color="black", linestyle=layer_styles[layer],
            marker="o", markersize=np.sqrt(layer_sizes[layer]), linewidth=1.4,
            label=f"{layer} layer{'s' if layer != 1 else ''}",
        )
        for layer in layer_order
    ]

    for metric in PRIMARY_GSRPS:
        property_column = f"mean_{metric}"
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharey=True)

        for ax, architecture in zip(axes.flat, architectures):
            architecture_data = plot_data[plot_data["architecture"] == architecture]
            for family in family_order:
                family_data = architecture_data[
                    architecture_data["graph_family"] == family
                ]
                for layer in layer_order:
                    subset = family_data[family_data["num_layers"] == layer]
                    if subset.empty:
                        continue
                    subset = subset.sort_values("_variant_order")
                    ax.plot(
                        subset[property_column],
                        subset["delta_bacc"],
                        color=family_colors[family],
                        marker=family_markers[family],
                        markersize=np.sqrt(layer_sizes[layer]),
                        linestyle=layer_styles[layer],
                        linewidth=1.2 if config.show_trend_lines else 0,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                        alpha=0.9,
                    )
            ax.set_title(architecture, fontweight="bold", pad=8)
            ax.set_xlabel(GSRP_SHORT_LABELS.get(metric, metric), fontsize=9, labelpad=8)
            ax.set_ylabel("Delta balanced accuracy")
            ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.55)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=9)

        fig.legend(
            handles=family_handles + layer_handles,
            # title="Graph family (color/marker) | Layers (line style/size)",
            frameon=False,
            fontsize=9,
            title_fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            ncol=4,
            columnspacing=1.2,
            handletextpad=0.5,
        )
        fig.suptitle(
            f"{GSRP_LABELS.get(metric, metric)} and GNN advantage over MLP",
            fontsize=17,
            fontweight="bold",
            y=0.995,
        )
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.78, wspace=0.18, hspace=0.3)
        output_stem = config.output_dir / f"figure3_{metric}"
        fig.savefig(f"{output_stem}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def write_outputs(config: Config, performance: pd.DataFrame, gsrp: pd.DataFrame, correlations: pd.DataFrame, inspection: list[dict[str, Any]]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    correlations.to_csv(config.output_dir / "association_results.csv", index=False)

def run(config: Config) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    performance, gsrp, inspection = prepare_analysis_data(config)
    correlations = correlation_analysis(performance, gsrp, config)
    make_figure(performance, gsrp, config)
    make_correlation_heatmap(correlations, config)
    write_outputs(config, performance, gsrp, correlations, inspection)
    print(f"Saved GSRP analysis to {config.output_dir}")

def make_config() -> Config:
    return Config(
        graph_root=GRAPH_ROOT,
        performance_csv=PERFORMANCE_CSV,
        model_name=MODEL_NAME,
        split=SPLIT,
        output_dir=OUTPUT_DIR,
        bootstrap_iterations=BOOTSTRAP_ITERATIONS,
        show_trend_lines=SHOW_TREND_LINES,
    )

# %%
GRAPH_ROOT = Path("graph_outputs")
PERFORMANCE_CSV = Path("gnn_results/_3master_results.csv")
MODEL_NAME = "a9d7feb3402a4670bbcfa73f534acab7"
SPLIT = "test"
OUTPUT_DIR = Path("figures/gsrp_analysis")
BOOTSTRAP_ITERATIONS = 2000
SHOW_TREND_LINES = True

run(make_config())

