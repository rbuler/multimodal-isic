import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


def validate_columns(df: pd.DataFrame, mean_column: str, std_column: str) -> None:
    required = {"architecture", "trial_id"}
    has_mean = mean_column in df.columns
    has_raw = mean_column.removesuffix("_mean") in df.columns
    has_std = std_column in df.columns

    if not has_std:
        raise ValueError(f"Missing required column: {std_column}")
    if not (has_mean or has_raw):
        raise ValueError(f"Missing required column: {mean_column} or {mean_column.removesuffix('_mean')}")
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing required columns: {missing}")


def prepare_data(df: pd.DataFrame, mean_column: str, std_column: str) -> pd.DataFrame:
    """Normalize the input data and keep only the columns needed for plotting."""
    data = df.copy()
    data["architecture"] = data["architecture"].astype(str)
    data["trial_id"] = data["trial_id"].astype(str)

    raw_column = mean_column.removesuffix("_mean")
    if mean_column not in data.columns and raw_column in data.columns:
        data["score_mean"] = pd.to_numeric(data[raw_column], errors="coerce")
    else:
        data["score_mean"] = pd.to_numeric(data[mean_column], errors="coerce")

    data["score_std"] = pd.to_numeric(data[std_column], errors="coerce")

    data = data.dropna(subset=["architecture", "score_mean"]).copy()

    data["architecture"] = pd.Categorical(data["architecture"], categories=ARCHITECTURE_ORDER, ordered=True)
    data = data.dropna(subset=["architecture"])
    data = data.sort_values(["architecture", "score_mean"], ascending=[True, False], na_position="last")
    return data


def best_trial_per_architecture(data: pd.DataFrame) -> pd.DataFrame:
    """Select the single best trial for each architecture."""
    best = (
        data.sort_values(["architecture", "score_mean"], ascending=[True, False])
        .groupby("architecture", as_index=False)
        .head(1)
        .copy()
    )
    return best[["architecture", "trial_id", "score_mean", "score_std"]].reset_index(drop=True)


def selected_test_per_architecture(
        validation_best: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Find each validation winner's corresponding held-out test result."""
    selected = validation_best[["architecture", "trial_id"]].merge(
        test,
        on=["architecture", "trial_id"],
        how="left",
    )
    return selected[["architecture", "trial_id", "score_mean", "score_std"]]


def top10_trials_per_architecture(data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Select the top-N trials per architecture for the scatter plot."""
    top10 = (
        data.sort_values(["architecture", "score_mean"], ascending=[True, False])
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


def plot_top10_per_architecture(
        validation: pd.DataFrame,
        validation_best: pd.DataFrame,
    test: pd.DataFrame,
    test_best: pd.DataFrame,
    output_dir: Path,
) -> tuple[plt.Figure, plt.Axes]:
    """Compare validation and held-out test top-10 scores per architecture."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    x_center = np.arange(len(ARCHITECTURE_ORDER))
    architecture_to_pos = {arch: pos for pos, arch in enumerate(ARCHITECTURE_ORDER)}

    for arch in ARCHITECTURE_ORDER:
        for data, best, color, offset, label in (
            (validation, validation_best, "tab:blue", -0.10, "Validation"),
            (test, test_best, "tab:red", 0.10, "Held-out test"),
        ):
            subset = data[data["architecture"] == arch].copy().reset_index(drop=True)
            if subset.empty:
                continue

            x_positions = architecture_to_pos[arch] + offset + np.linspace(-0.07, 0.07, len(subset))
            ax.scatter(
                x_positions,
                subset["score_mean"],
                s=18,
                alpha=0.7,
                color=color,
                edgecolors="none",
                label=label if arch == ARCHITECTURE_ORDER[0] else "_nolegend_",
                zorder=2,
            )

            best_rows = best[best["architecture"] == arch]
            if best_rows.empty:
                continue
            best_trial = best_rows.iloc[0]
            best_pos = np.flatnonzero(subset["trial_id"].astype(str).to_numpy() == str(best_trial["trial_id"]))
            if len(best_pos) > 0:
                pos = int(best_pos[0])
                ax.scatter(
                    x_positions[pos],
                    subset.iloc[pos]["score_mean"],
                    s=75,
                    marker="*",
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=4,
                )

    ax.set_xticks(x_center)
    ax.set_xticklabels(ARCHITECTURE_ORDER, rotation=45, ha="right")
    ax.set_ylabel("Balanced accuracy (BACC)")
    ax.set_xlabel("Architecture")
    ax.set_title("Validation and held-out test performance")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=6, label="Validation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=6, label="Held-out test"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="white", markeredgecolor="black", markersize=9, label="Best per set"),
    ], loc="best")

    all_scores = pd.concat([validation["score_mean"], test["score_mean"]])
    ymin = all_scores.min() - 0.05
    ymax = all_scores.max() + 0.05
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "top10_by_architecture.png", dpi=300, bbox_inches="tight")
    return fig, ax


def generate_figures(
        validation_trials: pd.DataFrame,
    test_trials: pd.DataFrame,
    output_dir: str = ".",
) -> tuple[plt.Figure, plt.Figure]:
    """Create publication-quality validation/test figures from trial results."""
    validate_columns(validation_trials, "val_bacc_mean", "val_bacc_std")
    validate_columns(test_trials, "test_bacc_mean", "test_bacc_std")
    validation = prepare_data(validation_trials, "val_bacc_mean", "val_bacc_std")
    test = prepare_data(test_trials, "test_bacc_mean", "test_bacc_std")

    validation_best = best_trial_per_architecture(validation)
    validation_top10 = top10_trials_per_architecture(validation, top_n=10)
    test_best = selected_test_per_architecture(validation_best, test)
    test_top10 = top10_trials_per_architecture(test, top_n=10)

    output_path = Path(output_dir)
    # fig_best, ax_best = plot_best_by_architecture(best, output_path)
    fig_top10, ax_top10 = plot_top10_per_architecture(
        validation_top10, validation_best, test_top10, test_best, output_path,
    )
    return None, fig_top10



if __name__ == "__main__":
    configure_style()

    validation_path = Path("gnn_results_raytune/hparam_search_global_top10.csv")
    test_path = Path("gnn_results_raytune/test_hparam_search_global_top10.csv")
    validation_trials = pd.read_csv(validation_path)
    test_trials = pd.read_csv(test_path)
    # fig_best, fig_top10 = generate_figures(df_trials, output_dir="figures/ray_tune_perf")
    _, fig_top10 = generate_figures(
        validation_trials, test_trials, output_dir="figures/ray_tune_perf",
    )
    plt.show()
