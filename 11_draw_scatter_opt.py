import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path


ARCHITECTURE_ORDER = [
    "gin",
    "graphsage",
    "mlp",
    "fagcn",
    "gcnii",
    "gcn",
    "transformer",
    "gatv2",
    "gat",
]


def validate_columns(df: pd.DataFrame) -> None:
    required = {"architecture", "trial_id"}
    has_mean = "val_bacc_mean" in df.columns
    has_raw = "val_bacc" in df.columns
    has_std = "val_bacc_std" in df.columns

    if not has_std:
        raise ValueError("Missing required column: val_bacc_std")
    if not (has_mean or has_raw):
        raise ValueError("Missing required column: val_bacc_mean or val_bacc")
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the input data and keep only the columns needed for plotting."""
    data = df.copy()
    data["architecture"] = data["architecture"].astype(str)

    if "val_bacc_mean" not in data.columns and "val_bacc" in data.columns:
        data["val_bacc_mean"] = pd.to_numeric(data["val_bacc"], errors="coerce")
    else:
        data["val_bacc_mean"] = pd.to_numeric(data["val_bacc_mean"], errors="coerce")

    data["val_bacc_std"] = pd.to_numeric(data["val_bacc_std"], errors="coerce")

    data = data.dropna(subset=["architecture", "val_bacc_mean"]).copy()

    data["architecture"] = pd.Categorical(data["architecture"], categories=ARCHITECTURE_ORDER, ordered=True)
    data = data.sort_values(["architecture", "val_bacc_mean"], ascending=[True, False], na_position="last")
    return data


def best_trial_per_architecture(data: pd.DataFrame) -> pd.DataFrame:
    """Select the single best trial for each architecture, defined by highest val_bacc_mean."""
    best = (
        data.sort_values(["architecture", "val_bacc_mean"], ascending=[True, False])
        .groupby("architecture", as_index=False)
        .head(1)
        .copy()
    )
    return best[["architecture", "trial_id", "val_bacc_mean", "val_bacc_std"]].reset_index(drop=True)


def top10_trials_per_architecture(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Select the top-N trials per architecture for the scatter plot."""
    top10 = (
        data.sort_values(["architecture", "val_bacc_mean"], ascending=[True, False])
        .groupby("architecture", group_keys=False)
        .head(top_n)
        .copy()
    )
    top10["x_pos"] = top10["architecture"].cat.codes
    return top10.reset_index(drop=True)


def configure_style() -> None:
    """Apply a clean scientific publication style to Matplotlib."""
    matplotlib.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "legend.frameon": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
    })


def plot_best_by_architecture(best: pd.DataFrame, output_dir: Path) -> tuple[plt.Figure, plt.Axes]:
    """Create a dot plot with error bars showing the best trial for each architecture."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x_positions = np.arange(len(best))
    y = best["val_bacc_mean"].to_numpy()
    yerr = best["val_bacc_std"].fillna(0.0).to_numpy()

    ax.errorbar(
        x_positions,
        y,
        yerr=yerr,
        fmt="o",
        ms=7,
        mfc="tab:blue",
        mec="black",
        mew=0.5,
        ecolor="tab:blue",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        linestyle="none",
        zorder=3,
    )

    for i, value in enumerate(y):
        ax.text(
            i,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(best["architecture"], rotation=45, ha="right")
    ax.set_ylabel("Best validation BACC")
    ax.set_xlabel("Architecture")
    ax.set_title("Best-performing configuration per architecture")

    y_min = y.min() - 0.05
    y_max = y.max() + 0.05
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "best_by_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "best_by_architecture.pdf", bbox_inches="tight")
    return fig, ax


def plot_top10_per_architecture(top10: pd.DataFrame, best: pd.DataFrame, output_dir: Path) -> tuple[plt.Figure, plt.Axes]:
    """Create the top-10 scatter plot per architecture with deterministic jitter."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x_center = np.arange(len(ARCHITECTURE_ORDER))
    architecture_to_pos = {arch: pos for pos, arch in enumerate(ARCHITECTURE_ORDER)}

    for arch in ARCHITECTURE_ORDER:
        subset = top10[top10["architecture"] == arch].copy().reset_index(drop=True)
        if subset.empty:
            continue

        x_positions = architecture_to_pos[arch] + np.linspace(-0.18, 0.18, len(subset))
        ax.scatter(
            x_positions,
            subset["val_bacc_mean"],
            s=16,
            alpha=0.7,
            color="tab:blue",
            edgecolors="none",
            zorder=2,
        )

        best_trial = best[best["architecture"] == arch].iloc[0]
        best_pos = np.flatnonzero(subset["trial_id"].to_numpy() == best_trial["trial_id"])
        if len(best_pos) > 0:
            pos = int(best_pos[0])
            ax.scatter(
                x_positions[pos],
                subset.iloc[pos]["val_bacc_mean"],
                s=35,
                facecolor="none",
                edgecolor="black",
                linewidth=1.0,
                zorder=4,
            )

    ax.set_xticks(x_center)
    ax.set_xticklabels(ARCHITECTURE_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Validation BACC")
    ax.set_xlabel("Architecture")
    ax.set_title("Top 10 configurations per architecture")

    ymin = min(top10["val_bacc_mean"].min(), best["val_bacc_mean"].min()) - 0.05
    ymax = max(top10["val_bacc_mean"].max(), best["val_bacc_mean"].max()) + 0.05
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "top10_by_architecture.png", dpi=300, bbox_inches="tight")
    return fig, ax


def generate_figures(df_trials: pd.DataFrame, output_dir: str = ".") -> tuple[plt.Figure, plt.Figure]:
    """Create both publication-quality figures from the raw trial-level results."""
    validate_columns(df_trials)
    data = prepare_data(df_trials)

    best = best_trial_per_architecture(data)
    top10 = top10_trials_per_architecture(data, top_n=10)

    output_path = Path(output_dir)
    # fig_best, ax_best = plot_best_by_architecture(best, output_path)
    fig_top10, ax_top10 = plot_top10_per_architecture(top10, best, output_path)
    return _, fig_top10



if __name__ == "__main__":
    configure_style()

    csv_path = Path("gnn_results_raytune/hparam_search_global_top10.csv")
    df_trials = pd.read_csv(csv_path)
    # fig_best, fig_top10 = generate_figures(df_trials, output_dir="figures/ray_tune_perf")
    _, fig_top10 = generate_figures(df_trials, output_dir="figures/ray_tune_perf")
    plt.show()
