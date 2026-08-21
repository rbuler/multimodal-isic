import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("gnn_results")
DEFAULT_NEIGHBORS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]

GRAPH_VARIANTS = ["grid4", "grid8"] + [f"knn{k}" for k in DEFAULT_NEIGHBORS] + [f"random{r}" for r in DEFAULT_NEIGHBORS]
GNN_MODELS = ["mlp", "gcn", "gat", "gatv2", "gin", "graphsage", "transformer", "fagcn", "gcnii"]
SEEDS = [42, 123, 999]
HIDDEN_DIMS = [128, 256, 384]
DROPOUTS = [0.25, 0.50, 0.75]
LAYERS = [1, 2, 3, 4, 5]

csv_files = list(RESULTS_DIR.glob("results*.csv"))
if not csv_files:
    print(f"Nie znaleziono plików CSV w {RESULTS_DIR}")

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

embedding_models = sorted(df["embedding_model"].dropna().unique())
print(f"Wykryto {len(embedding_models)} modeli embeddingowych: {embedding_models}")

KEY_COLS = ["embedding_model", "graph_variant", "graph_model", "num_layers", "seed", "hidden_dim", "dropout"]
num_duplicates = df.duplicated(subset=KEY_COLS).sum()

print("\n" + "=" * 45)
print("  ANALIZA DUPLIKATÓW")
print("=" * 45)
if num_duplicates > 0:
    print(f"⚠️ ZNALEZIONO DUPLIKATY! Nadmiarowe/zduplikowane wpisy: {num_duplicates}")
    dup_summary = df[df.duplicated(subset=KEY_COLS, keep=False)].groupby(KEY_COLS).size().reset_index(name='powtórzenia')
    print(dup_summary.to_string(index=False))
else:
    print("✅ Brak duplikatów – żaden worker nie przeliczył tego samego zadania.")

df = df.drop_duplicates(subset=KEY_COLS)


expected_combos = set()
for emb in embedding_models:
    for layer in LAYERS:
        for seed in SEEDS:
            for hidden in HIDDEN_DIMS:
                for drop in DROPOUTS:
                    for gnn in GNN_MODELS:
                        if gnn.lower() == "mlp":
                            expected_combos.add((emb, "none", gnn, layer, seed, hidden, drop))
                        else:
                            for var in GRAPH_VARIANTS:
                                expected_combos.add((emb, var, gnn, layer, seed, hidden, drop))

actual_combos = set(
    zip(
        df["embedding_model"],
        df["graph_variant"],
        df["graph_model"],
        df["num_layers"],
        df["seed"],
        df["hidden_dim"],
        df["dropout"]
    )
)

completed = expected_combos.intersection(actual_combos)
missing = expected_combos - actual_combos

total_expected = len(expected_combos)
total_completed = len(completed)
pct = (total_completed / total_expected * 100) if total_expected > 0 else 0.0

print("\n" + "=" * 45)
print(f"  POSTĘP EXPERYMENTU: {pct:.2f}%")
print("=" * 45)
print(f"Ukończono: {total_completed} / {total_expected}")
print(f"Brakujące: {len(missing)}")

if missing:
    missing_df = pd.DataFrame(list(missing), columns=["emb", "variant", "gnn", "layer", "seed", "hidden", "drop"])
    
    print("\n--- Brakujące wg WARSTW (num_layers) ---")
    print(missing_df.groupby("layer").size().to_string())
    
    print("\n--- Brakujące wg ARCHITEKTUR GNN ---")
    print(missing_df.groupby("gnn").size().to_string())
