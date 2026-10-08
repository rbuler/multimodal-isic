# %%
import glob
import pandas as pd
from pathlib import Path
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
        output_summary = results_path / "_2master_results.csv"
        df_summary.sort_values(summary_keys).to_csv(output_summary, index=False)
        print(f"[SUMMARY] Scalono {len(summary_files)} plików do {output_summary} (Łącznie: {len(df_summary)} eksperymentów)")

    detailed_files = glob.glob(str(results_path / "detailed_fold_results_job_*.csv"))
    if detailed_files:
        df_detailed = pd.concat([pd.read_csv(f) for f in detailed_files], ignore_index=True)
        detailed_keys = ["embedding_model", "graph_variant", "graph_model", 
                         "seed", "hidden_dim", "dropout", "num_layers", "fold"]
        detailed_keys = [col for col in detailed_keys if col in df_detailed.columns]
        df_detailed = df_detailed.drop_duplicates(subset=detailed_keys, keep="last")
        output_detailed = results_path / "_2master_detailed_fold_results.csv"
        df_detailed.sort_values(detailed_keys).to_csv(output_detailed, index=False)
        print(f"[DETAILED] Scalono {len(detailed_files)} plików foldów do {output_detailed} (Łącznie: {len(df_detailed)} wierszy)")


# %%

def main():
    RESULTS_DIR = "/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/gnn_results"
    merge_worker_results(results_dir=RESULTS_DIR)

if __name__ == "__main__":
    main()

# TODO think of showing plot per layer count, and show all models in one plot, with different colors for each model.