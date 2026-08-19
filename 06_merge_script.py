import glob
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def merge_worker_results(results_dir: str = "results") -> None:
    results_path = Path(results_dir)
    
    # 1. Scalanie podsumowania (Summary)
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

    # 2. Scalanie wyników foldów (Detailed)
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

def generate_plots_and_stats(results_dir: str = "results", target_metric: str = "val_auc") -> None:
    results_path = Path(results_dir)
    master_summary_path = results_path / "master_results.csv"

    if not master_summary_path.exists():
        print("[ERROR] Brak pliku master_results.csv. Najpierw scal wyniki.")
        return

    df = pd.read_csv(master_summary_path)

    # Autodetekcja metryki, jeśli podana nie istnieje w pliku
    if target_metric not in df.columns:
        possible = [m for m in ["val_auc", "test_auc", "auc", "val_f1", "val_loss"] if m in df.columns]
        if possible:
            target_metric = possible[0]
        else:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            target_metric = numeric_cols[-1] if numeric_cols else None

    if not target_metric or target_metric not in df.columns:
        print("[ERROR] Nie znaleziono odpowiedniej metryki numerycznej do analizy.")
        return

    plots_dir = results_path / "plots_and_stats"
    plots_dir.mkdir(exist_ok=True)
    
    sns.set_theme(style="whitegrid", palette="muted")
    print(f"\n=== Generowanie wykresów i statystyk dla metryki: {target_metric} ===")

    # --- WYKRES 1: Boxplot porównujący architektury GNN ---
    plt.figure(figsize=(10, 6))
    order = df.groupby("graph_model")[target_metric].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="graph_model", y=target_metric, order=order, palette="Set2")
    sns.stripplot(data=df, x="graph_model", y=target_metric, order=order, color="black", alpha=0.3, jitter=0.2)
    plt.title(f"Rozkład {target_metric} wg architektury GNN", fontsize=14, fontweight='bold')
    plt.xlabel("Model GNN")
    plt.ylabel(target_metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / f"boxplot_models_{target_metric}.png", dpi=300)
    plt.close()

    # --- WYKRES 2: Heatmapa wpływu hiperparametrów (Hidden Dim vs Dropout) ---
    if "hidden_dim" in df.columns and "dropout" in df.columns:
        plt.figure(figsize=(8, 6))
        pivot_df = df.pivot_table(values=target_metric, index="hidden_dim", columns="dropout", aggfunc="mean")
        sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title(f"Średnia {target_metric} (Hidden Dim vs Dropout)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / f"heatmap_hyperparams_{target_metric}.png", dpi=300)
        plt.close()

    # --- WYKRES 3: Średnie z odchyleniem standardowym ---
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="graph_model", y=target_metric, order=order, capsize=0.1, err_kws={'linewidth': 1.5})
    plt.title(f"Średnia wartość {target_metric} (± Standard Error)", fontsize=14, fontweight='bold')
    plt.xlabel("Model GNN")
    plt.ylabel(target_metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / f"barplot_means_{target_metric}.png", dpi=300)
    plt.close()

    # --- TESTY STATYSTYCZNE ---
    groups = [group[target_metric].dropna().values for _, group in df.groupby("graph_model")]
    model_names = [name for name, _ in df.groupby("graph_model")]

    if len(groups) > 1:
        # 1. Test Kruskala-Wallisa (Nieparametryczna ANOVA)
        kw_stat, kw_p = stats.kruskal(*groups)
        
        # Wyznaczenie najlepszego modelu
        mean_scores = df.groupby("graph_model")[target_metric].mean().sort_values(ascending=False)
        best_model = mean_scores.index[0]
        best_scores = df[df["graph_model"] == best_model][target_metric]

        # 2. Porównania parami z najlepszym modelem (Mann-Whitney U Test)
        stat_results = []
        for model in model_names:
            if model == best_model:
                continue
            comp_scores = df[df["graph_model"] == model][target_metric]
            u_stat, p_val = stats.mannwhitneyu(best_scores, comp_scores, alternative='greater')
            
            stat_results.append({
                "Best_Model": best_model,
                "Compared_Model": model,
                "Best_Mean": round(best_scores.mean(), 5),
                "Comp_Mean": round(comp_scores.mean(), 5),
                "Diff": round(best_scores.mean() - comp_scores.mean(), 5),
                "U_Stat": u_stat,
                "p_value": round(p_val, 6),
                "Significant_alpha_0.05": p_val < 0.05
            })

        df_stats = pd.DataFrame(stat_results).sort_values("p_value")
        df_stats.to_csv(plots_dir / "statistical_tests_results.csv", index=False)

        # Zapis podsumowania tekstowego
        with open(plots_dir / "stats_summary.txt", "w") as f:
            f.write(f"=== TESTY STATYSTYCZNE DLA METRYKI: {target_metric} ===\n\n")
            f.write(f"1. Test Kruskala-Wallisa (czy architektura ma znaczenie?):\n")
            f.write(f"   H-Statistic: {kw_stat:.4f}, p-value: {kw_p:.4e}\n")
            f.write(f"   Wniosek: {'Różnice są istotne statystycznie (p < 0.05)' if kw_p < 0.05 else 'Brak istotnych różnic między modelami'}\n\n")
            f.write(f"2. Najlepszy model pod względem średniej: {best_model} ({mean_scores.iloc[0]:.4f})\n\n")
            f.write("3. Testy Mann-Whitneya U (czy najlepszy model jest istotnie lepszy od pozostałych):\n")
            f.write(df_stats.to_string(index=False))

        print(f"[STATS] Wyniki testów zapisano w: {plots_dir / 'stats_summary.txt'}")
        print(f"[PLOTS] Wykresy zapisano w katalogu: {plots_dir}")

def main():
    RESULTS_DIR = "/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/gnn_results"
    merge_worker_results(results_dir=RESULTS_DIR)
    generate_plots_and_stats(results_dir=RESULTS_DIR, target_metric="test_bacc")

if __name__ == "__main__":
    main()