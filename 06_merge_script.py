# %%
import glob
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# %%

def merge_worker_results(results_dir: str = "results") -> None:
    results_path = Path(results_dir)
    
    summary_files = glob.glob(str(results_path / "results_job_*.csv"))
    if summary_files:
        df_summary = pd.concat([pd.read_csv(f) for f in summary_files], ignore_index=True)
        summary_keys = ["embedding_model", "graph_variant", "graph_model", 
                        "seed", "hidden_dim", "dropout", "num_layers"]
        summary_keys = [col for col in summary_keys if col in df_summary.columns]
        df_summary = df_summary.drop_duplicates(subset=summary_keys, keep="last")
        output_summary = results_path / "_1master_results.csv"
        df_summary.sort_values(summary_keys).to_csv(output_summary, index=False)
        print(f"[SUMMARY] Scalono {len(summary_files)} plików do {output_summary} (Łącznie: {len(df_summary)} eksperymentów)")

    detailed_files = glob.glob(str(results_path / "detailed_fold_results_job_*.csv"))
    if detailed_files:
        df_detailed = pd.concat([pd.read_csv(f) for f in detailed_files], ignore_index=True)
        detailed_keys = ["embedding_model", "graph_variant", "graph_model", 
                         "seed", "hidden_dim", "dropout", "num_layers", "fold"]
        detailed_keys = [col for col in detailed_keys if col in df_detailed.columns]
        df_detailed = df_detailed.drop_duplicates(subset=detailed_keys, keep="last")
        output_detailed = results_path / "_1master_detailed_fold_results.csv"
        df_detailed.sort_values(detailed_keys).to_csv(output_detailed, index=False)
        print(f"[DETAILED] Scalono {len(detailed_files)} plików foldów do {output_detailed} (Łącznie: {len(df_detailed)} wierszy)")


def generate_plots(
    results_dir: str = "results",
    target_metric: str = "test_bacc_mean",
    whiskers: bool = True,
    embedding_mode: str = "average",
    embedding_model: str = None,
    plot_style: str = "layers",
) -> None:

    results_path = Path(results_dir)
    master_summary_path = results_path / "_1master_results.csv"

    if not master_summary_path.exists():
        print("[ERROR] Brak pliku _1master_results.csv.")
        return

    df = pd.read_csv(master_summary_path)

    if embedding_mode not in {"average", "per_model"}:
        raise ValueError("embedding_mode must be 'average' or 'per_model'")
    if plot_style not in {"layers", "summary"}:
        raise ValueError("plot_style must be 'layers' or 'summary'")

    if "embedding_model" not in df.columns:
        print("[ERROR] Brak kolumny 'embedding_model'.")
        return

    if embedding_mode == "per_model" and embedding_model is None:
        embedding_models = sorted(df["embedding_model"].dropna().unique())
        if not embedding_models:
            print("[ERROR] Brak embedding models do narysowania.")
            return
        for model_name in embedding_models:
            generate_plots(
                results_dir=results_dir,
                target_metric=target_metric,
                whiskers=whiskers,
                embedding_mode="per_model",
                embedding_model=str(model_name),
                plot_style=plot_style,
            )
        return

    if embedding_mode == "per_model":
        df = df[df["embedding_model"].astype(str) == str(embedding_model)].copy()
        if df.empty:
            print(f"[ERROR] Brak danych dla embedding modelu '{embedding_model}'.")
            return

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
        "gcnii",
        "gat",
        "gatv2",
        "transformer",
        "gin",
        "graphsage",
        "fagcn",
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

        if plot_style == "summary":
            if model == "mlp":
                summary_values = model_df[target_metric].dropna()
                if not summary_values.empty:
                    mean_value = summary_values.mean()
                    min_value = summary_values.min()
                    max_value = summary_values.max()
                    ax.axhspan(min_value, max_value, color="tab:blue", alpha=0.18)
                    ax.axhline(mean_value, color="tab:blue", linewidth=2.0)
                    ax.text(
                        0.5,
                        0.08,
                        f"mean={mean_value:.3f}\nrange={min_value:.3f}-{max_value:.3f}",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=9,
                    )
                ax.text(
                    0.5,
                    0.92,
                    "No graph structure",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    style="italic",
                )
            else:
                for family_prefix, family_color in zip(
                    ["grid", "knn", "random"],
                    ["tab:blue", "tab:orange", "tab:green"],
                ):
                    family_variants = [
                        v for v in VARIANTS_ORDER if v.startswith(family_prefix)
                    ]
                    family_data = model_df[
                        model_df["graph_variant"].str.startswith(family_prefix)
                    ]
                    if family_data.empty:
                        continue

                    summary_data = (
                        family_data.groupby("graph_variant")[target_metric]
                        .agg(mean="mean", minimum="min", maximum="max")
                        .reindex(family_variants)
                        .dropna(subset=["mean"])
                    )
                    if summary_data.empty:
                        continue

                    x = [VARIANTS_ORDER.index(v) for v in summary_data.index]
                    ax.plot(
                        x,
                        summary_data["mean"].values,
                        marker="o",
                        markersize=4,
                        linewidth=1.8,
                        color=family_color,
                    )
                    ax.fill_between(
                        x,
                        summary_data["minimum"].values,
                        summary_data["maximum"].values,
                        color=family_color,
                        alpha=0.18,
                    )

        elif model == "mlp":
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

    if plot_style == "summary":
        legend_handles = [
            Line2D([0], [0], color="black", linewidth=1.8, label="Mean across layers"),
            Patch(facecolor="gray", alpha=0.22, label="Min-max across layers"),
            Line2D([0], [0], color="tab:blue", linewidth=1.8, label="Grid"),
            Line2D([0], [0], color="tab:orange", linewidth=1.8, label="kNN"),
            Line2D([0], [0], color="tab:green", linewidth=1.8, label="Random"),
        ]
        legend_title = "Summary"
    else:
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
        legend_title = "Layers"

    fig.legend(
        handles=legend_handles,
        title=legend_title,
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

    if embedding_mode == "average":
        output_name = f"_gnn_variants_layers_benchmark_{plot_style}.png"
    else:
        safe_model_name = str(embedding_model).replace("/", "_").replace("\\", "_")
        output_name = f"_gnn_variants_layers_benchmark_{safe_model_name}_{plot_style}.png"
    output_path = results_path / output_name

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
    merge_worker_results(results_dir=RESULTS_DIR)
    generate_plots(
        results_dir=RESULTS_DIR,
        target_metric="test_bacc_mean",
        whiskers=False,
        embedding_mode="average",        # average or per_model
        plot_style="summary",             # layers  or summary
    )

if __name__ == "__main__":
    main()

# TODO think of showing plot per layer count, and show all models in one plot, with different colors for each model.