import glob
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def merge_worker_results(results_dir: str = "results") -> None:
    results_path = Path(results_dir)
    
    summary_files = glob.glob(str(results_path / "results_job_*.csv"))
    if summary_files:
        df_summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
        summary_keys = [
            "embedding_model", "graph_variant", "graph_model", 
            "seed", "hidden_dim", "dropout", "num_layers"
        ]
        summary_keys = [col for col in summary_keys if col in df_summary.columns]
        df_summary = df_summary.drop_duplicates(subset=summary_keys, keep="last")
        output_summary = results_path / "master_results.csv"
        df_summary.sort_values(summary_keys).to_csv(output_summary, index=False)
        print(f"[SUMMARY] Scalono {len(summary_files)} plików do {output_summary} (Łącznie: {len(df_summary)} eksperymentów)")

    detailed_files = glob.glob(str(results_path / "detailed_fold_results_job_*.csv"))
    if detailed_files:
        df_detailed = pd.concat([pd.read_csv(f) for f in detailed_files], ignore_index=True)
        detailed_keys = [
            "embedding_model", "graph_variant", "graph_model", 
            "seed", "hidden_dim", "dropout", "num_layers", "fold"
        ]
        detailed_keys = [col for col in detailed_keys if col in df_detailed.columns]
        df_detailed = df_detailed.drop_duplicates(subset=detailed_keys, keep="last")
        output_detailed = results_path / "master_detailed_fold_results.csv"
        df_detailed.sort_values(detailed_keys).to_csv(output_detailed, index=False)
        print(f"[DETAILED] Scalono {len(detailed_files)} plików foldów do {output_detailed} (Łącznie: {len(df_detailed)} wierszy)")




def generate_plots(
    results_dir: str = "results",
    target_metric: str = "test_bacc_mean",
    whiskers: bool = True
) -> None:

    results_path = Path(results_dir)
    master_summary_path = results_path / "master_results.csv"

    if not master_summary_path.exists():
        print("[ERROR] Brak pliku master_results.csv.")
        return

    df = pd.read_csv(master_summary_path)

    if target_metric not in df.columns:
        print(f"[ERROR] Brak kolumny '{target_metric}'.")
        return

    VARIANTS_ORDER = (
        ["grid4", "grid8"]
        + [f"knn{k}" for k in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]]
        + [f"random{r}" for r in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]]
    )

    MODELS = [
        "mlp",
        "gcn",
        "gat",
        "gatv2",
        "gin",
        "graphsage",
        "transformer",
        "fagcn",
        "gcnii",
    ]

    LAYERS = [1, 2, 3, 4, 5]

    std_metric = target_metric.replace("_mean", "_std")

    if whiskers and std_metric not in df.columns:
        print(
            f"[WARNING] Brak kolumny '{std_metric}'. "
            "Whiskery zostaną wyłączone."
        )
        whiskers = False

    df = df.copy()
    df["num_layers"] = pd.to_numeric(df["num_layers"], errors="coerce")
    df[target_metric] = pd.to_numeric(df[target_metric], errors="coerce")
    
    if std_metric in df.columns:
        df[std_metric] = pd.to_numeric(df[std_metric], errors="coerce")
    
    df = df.dropna(subset=["graph_model", "graph_variant", "num_layers", target_metric])
    df = df[
        (
            (df["graph_model"] == "mlp")
            & (df["graph_variant"] == "none")
            & df["num_layers"].isin(LAYERS)
        )
        |
        (
            (df["graph_model"] != "mlp")
            & df["graph_variant"].isin(VARIANTS_ORDER)
            & df["num_layers"].isin(LAYERS)
        )
    ]

    df = df[df["graph_model"].isin(MODELS)]

    if df.empty:
        print("[ERROR] Brak danych do narysowania.")
        return

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 10

    sns.set_theme(style="ticks", palette="colorblind")
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(16, 11),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()
    layer_colors = sns.color_palette("viridis", len(LAYERS))

    for i, model in enumerate(MODELS):
        ax = axes[i]
        model_df = df[df["graph_model"] == model]

        if model == "mlp":
            for layer_idx, layer in enumerate(LAYERS):
                layer_data = model_df[
                    (model_df["graph_variant"] == "none")
                    & (model_df["num_layers"] == layer)
                ]

                if layer_data.empty:
                    continue
                y = layer_data[target_metric].mean()

                if not whiskers:
                    ax.axhline(
                        y,
                        color=layer_colors[layer_idx],
                        linestyle="-",
                        linewidth=1.5,
                        alpha=0.7
                    )

                else:
                    if std_metric in layer_data.columns:
                        std = layer_data[std_metric].mean()
                        ax.axhline(
                            y,
                            color=layer_colors[layer_idx],
                            linestyle="-",
                            linewidth=1.5,
                            alpha=0.7
                        )

            ax.text(
                0.5,
                0.08,
                "No graph structure!",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                style="italic"
            )

        else:
            for layer_idx, layer in enumerate(LAYERS):
                layer_data = model_df[model_df["num_layers"] == layer]

                for family_prefix in ["grid", "knn", "random"]:
                    family_variants = [v for v in VARIANTS_ORDER if v.startswith(family_prefix)]
                    fam_data = layer_data[layer_data["graph_variant"].str.startswith(family_prefix)]

                    if fam_data.empty:
                        continue

                    fam_data = (
                        fam_data
                        .groupby(
                            "graph_variant",
                            as_index=True
                        )
                        .agg(
                            mean=(target_metric, "mean"),
                            std=(
                                std_metric,
                                "mean"
                            )
                            if std_metric in fam_data.columns
                            else (
                                target_metric,
                                "std"
                            )
                        )
                        .reindex(family_variants)
                    )

                    fam_data = fam_data.dropna(
                        subset=["mean"]
                    )

                    if fam_data.empty:
                        continue

                    x = [
                        VARIANTS_ORDER.index(v)
                        for v in fam_data.index
                    ]

                    y = fam_data["mean"].values

                    if not whiskers:
                        ax.plot(
                            x,
                            y,
                            marker="o",
                            markersize=4,
                            linewidth=1.5,
                            alpha=0.7,
                            color=layer_colors[layer_idx],
                        )

                    else:
                        yerr = fam_data["std"].values
                        ax.errorbar(
                            x,
                            y,
                            yerr=yerr,
                            fmt="-o",
                            markersize=4,
                            linewidth=1.5,
                            alpha=0.7,
                            elinewidth=1.0,
                            capsize=3,
                            capthick=1.0,
                            color=layer_colors[layer_idx],
                        )

        ax.axvline(
            1.5,
            color="gray",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6
        )

        ax.axvline(
            11.5,
            color="gray",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6
        )

        ax.axvspan(
            1.5,
            11.5,
            color="gray",
            alpha=0.05
        )

        ax.set_title(
            model.upper(),
            fontsize=12,
            fontweight="bold",
            pad=8
        )

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.7,
            alpha=0.6
        )

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=9
        )

        sns.despine(
            ax=ax,
            top=True,
            right=True
        )

    for ax in axes:

        ax.set_xticks(
            range(len(VARIANTS_ORDER))
        )

        ax.set_xticklabels(
            VARIANTS_ORDER,
            rotation=90,
            fontsize=8
        )

    fig.supxlabel(
        "Graph Variant (Grid | kNN | Random)",
        fontsize=13,
        fontweight="bold",
        y=0.02
    )

    fig.supylabel(
        "Balanced Test Accuracy",
        fontsize=13,
        fontweight="bold",
        x=0.02
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=layer_colors[i],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=5,
            label=str(layer)
        )
        for i, layer in enumerate(LAYERS)
    ]

    fig.legend(
        handles=legend_handles,
        title="Layers",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(LAYERS),
        frameon=True,
        fontsize=10,
        title_fontsize=10,
        handlelength=2.0,
        columnspacing=1.5
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.18,
        top=0.90,
        wspace=0.08,
        hspace=0.28
    )

    output_path = (
        results_path
        / "_gnn_variants_layers_benchmark.pdf"
    )

    fig.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)
    print(f"[PLOT] Zapisano: {output_path}")

# %%

def main():
    RESULTS_DIR = "/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/gnn_results"
    # merge_worker_results(results_dir=RESULTS_DIR)
    generate_plots(results_dir=RESULTS_DIR, target_metric="test_bacc_mean", whiskers=False)

if __name__ == "__main__":
    main()

# TODO think of showing plot per layer count, and show all models in one plot, with different colors for each model.