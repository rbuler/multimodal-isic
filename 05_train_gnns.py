"""Cross-validated graph classification from saved patch embeddings.

This script intentionally has no dependency on torch-geometric.  The default
``GraphClassifier`` is a small, working GCN-style model and is the single
class to replace when comparing other graph architectures.

Input artifacts are produced by 01--03:
  patch_stats/<embedding_model>/patch_stats_fold_<fold>_<split>.pkl
  graph_outputs/<embedding_model>/graph_dataset.pkl

Each result row is one embedding-model / graph-variant / graph-model
experiment, aggregated over the five saved folds.  The external test split is
evaluated once with each fold's validation-selected checkpoint.
"""

import argparse
import copy
import os
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch_geometric import nn as pyg_nn
import yaml
from utils import get_args_parser



DEFAULT_NEIGHBORS = tuple(range(1, 9)) + (12, 16)
GNN_TYPES = ("mlp", "gcn", "gat", "gatv2", "gin", "graphsage", "transformer", "fagcn", "gcnii")
parser = get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphMIL(nn.Module):
    """
    Enhanced Graph-based Multiple Instance Learning model.
    
    Recommended configurations for 196 patches  x 768 features:
    - gnn_type: 'mlp', 'gat', 'gcn', 'gin' (most expressive), 
                'graphsage' (most efficient), 'transformer' (attention-based)
    - gnn_hidden: 256-512 (balance between capacity and efficiency)
    - gnn_layers: 2-3 (deeper can oversmooth)
    - use_residual: True (helps training deeper networks)
    - use_layer_norm: True (stabilizes training)
    """
    def __init__(self, input_dim=768, gnn_type='gat', gnn_hidden=256, 
                 gnn_layers=2, gnn_dropout=0.1, k_neighbors=8,
                 gnn_heads=4, gnn_concat=True, gcnii_alpha=0.1, gcnii_theta=0.5,
                 att_dim=128, att_heads=4, pool_dropout=0.2, 
                 classifier_dim=128, classifier_light=False, num_classes=7,
                 use_residual=True, use_layer_norm=True):
        super().__init__()
        
        self.gnn_type = gnn_type.lower()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.k_neighbors = k_neighbors
        self.classifier_light = classifier_light
        self.gnn_heads = gnn_heads
        self.gnn_concat = gnn_concat
        

        if (use_residual or self.gnn_type in {"fagcn", "gcnii"}) and input_dim != gnn_hidden:
            self.input_proj = nn.Linear(input_dim, gnn_hidden)
        else:
            self.input_proj = None
        
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        in_dim = input_dim if self.input_proj is None else gnn_hidden
        
        for i in range(gnn_layers):
            out_dim = gnn_hidden
            
            if self.gnn_type == 'gcn':
                layer = pyg_nn.GCNConv(in_dim, out_dim)
            elif self.gnn_type == 'gat':
                layer = pyg_nn.GATConv(in_dim, out_dim, heads=self.gnn_heads,
                                       concat=self.gnn_concat, dropout=gnn_dropout)
                out_dim *= self.gnn_heads if self.gnn_concat else 1
            elif self.gnn_type == 'graphsage':
                layer = pyg_nn.SAGEConv(in_dim, out_dim, aggr='mean', normalize=True)
            elif self.gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
                )
                layer = pyg_nn.GINConv(mlp, train_eps=True)
            elif self.gnn_type == 'transformer':
                layer = pyg_nn.TransformerConv(in_dim, out_dim, heads=self.gnn_heads,
                                               concat=self.gnn_concat, dropout=gnn_dropout,
                                               beta=True)
                out_dim *= self.gnn_heads if self.gnn_concat else 1
            elif self.gnn_type == 'gatv2':
                layer = pyg_nn.GATv2Conv(in_dim, out_dim, heads=self.gnn_heads,
                                       concat=self.gnn_concat, dropout=gnn_dropout)
            elif self.gnn_type == 'fagcn':
                if in_dim != out_dim:
                    raise ValueError("FAGCN requires a constant hidden dimension")
                layer = pyg_nn.FAConv(out_dim, eps=0.1, dropout=gnn_dropout)
            elif self.gnn_type == 'gcnii':
                if in_dim != out_dim:
                    raise ValueError("GCNII requires a constant hidden dimension across layers")
                layer = pyg_nn.GCN2Conv(out_dim, alpha=gcnii_alpha, theta=gcnii_theta, layer=i + 1)
            elif self.gnn_type == 'mlp':
                layer = nn.Sequential(nn.Linear(in_dim, out_dim))
            
            else:
                raise ValueError(f"Unsupported gnn_type: {self.gnn_type}")
            
            self.gnn_layers.append(layer)
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(out_dim))
            
            in_dim = out_dim
        
        self.gnn_dropout = nn.Dropout(gnn_dropout)
        self.final_gnn_dim = in_dim
        
        # Multi-head attention pooling (better than single attention)
        self.att_heads = att_heads
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.final_gnn_dim, att_dim),
                nn.Tanh(),
                nn.Linear(att_dim, 1)
            ) for _ in range(att_heads)
        ])
        

        if classifier_light:
            self.classifier = nn.Sequential(
                nn.Linear(self.final_gnn_dim, classifier_dim),
                nn.ReLU(),
                nn.Dropout(pool_dropout),
                nn.Linear(classifier_dim, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.final_gnn_dim, classifier_dim),
                nn.LayerNorm(classifier_dim),
                nn.ReLU(),
                nn.Dropout(pool_dropout),
                nn.Linear(classifier_dim, classifier_dim // 2),
                nn.LayerNorm(classifier_dim // 2),
                nn.ReLU(),
                nn.Dropout(pool_dropout / 2),
                nn.Linear(classifier_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index=None, edge_weight=None):
        """
        Args:
            x: Node features [N, F] where N=196, F=768
            adj: Optional dense adjacency matrix [N, N]
            edge_index: Optional edge index [2, E]
            
        Returns:
            probs: Class probabilities [num_classes]
            attention_weights: Attention weights [N, att_heads]
        """
        # Input projection for residual
        if self.input_proj is not None:
            x_input = self.input_proj(x)
        else:
            x_input = x
        
        h = x_input
        x_0 = x_input  # For GCNII

        # GNN layers with residual connections
        for i, layer in enumerate(self.gnn_layers):
            h_prev = h

            if self.gnn_type == 'mlp':
                h = layer(h)
            elif self.gnn_type == 'gcnii':
                h = layer(h, x_0, edge_index, edge_weight)
            elif self.gnn_type in {'gcn', 'fagcn'}:
                h = layer(h, edge_index, edge_weight)
            else:
                h = layer(h, edge_index)
            
            # Layer normalization
            if self.use_layer_norm:
                h = self.layer_norms[i](h)
            
            # Activation and dropout
            h = F.relu(h)
            h = self.gnn_dropout(h)
            
            # Residual connection
            if self.use_residual and h_prev.shape == h.shape:
                h = h + h_prev
        
        # Multi-head attention pooling
        attention_weights = []
        pooled_features = []
        
        for att_layer in self.attention_layers:
            a = F.softmax(att_layer(h), dim=0)  # [N, 1]
            attention_weights.append(a)
            z = torch.sum(a * h, dim=0)  # [hidden]
            pooled_features.append(z)
        
        # Aggregate multi-head outputs (mean pooling)
        z_agg = torch.stack(pooled_features, dim=0).mean(dim=0)
        attention_weights = torch.cat(attention_weights, dim=1)  # [N, att_heads]
        
        # Classification
        logits = self.classifier(z_agg)
        probs = F.softmax(logits, dim=0)
        
        return probs, attention_weights


def graph_variants() -> List[str]:
    return ["grid4", "grid8"] + [f"knn{k}" for k in DEFAULT_NEIGHBORS] + [
        f"random{r}" for r in DEFAULT_NEIGHBORS
    ]


def edge_index_for_variant(row: pd.Series, variant: str) -> np.ndarray:
    if variant == "grid4":
        return np.asarray(row["grid4_edge_index"], dtype=np.int64)
    if variant == "grid8":
        return np.asarray(row["grid8_edge_index"], dtype=np.int64)
    if variant.startswith("knn"):
        return np.asarray(row["knn_edge_indices"][int(variant[3:])], dtype=np.int64)
    if variant.startswith("random"):
        return np.asarray(row["random_edge_indices"][int(variant[6:])], dtype=np.int64)
    raise ValueError(f"Unknown graph variant: {variant}")


def load_pickle_dataframe(path: Path) -> pd.DataFrame:
    with path.open("rb") as handle:
        frame = pickle.load(handle)
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)


def load_fold_records(root: Path, embedding_model: str, fold: int, split: str,
                      variant: str) -> List[Dict]:
    graph_path = root / "graph_outputs" / embedding_model / "graph_dataset.pkl"
    stats_path = root / "patch_stats" / embedding_model / f"patch_stats_fold_{fold}_{split}.pkl"
    if not graph_path.exists() or not stats_path.exists():
        raise FileNotFoundError(f"Missing graph or patch stats for {embedding_model}, fold {fold}, {split}")

    graphs = load_pickle_dataframe(graph_path)
    graphs = graphs[(graphs["fold"] == fold) & (graphs["split"] == split)].copy()
    stats = load_pickle_dataframe(stats_path)[["image_id", "label", "patch_embeddings"]].copy()
    merged = graphs.merge(stats, on="image_id", how="inner", validate="one_to_one")
    if len(merged) != len(graphs):
        raise ValueError(f"{embedding_model}, fold {fold}, {split}: graph rows without patch statistics")

    records = []
    for _, row in merged.iterrows():
        x = np.asarray(row["patch_embeddings"], dtype=np.float32)
        edge_index = edge_index_for_variant(row, variant)
        if x.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Invalid graph record for image {row['image_id']}")
        records.append({"x": x, "edge_index": edge_index, "y": int(row["label"]),
                        "image_id": str(row["image_id"])})
    return records


def evaluate(model: nn.Module, records: Iterable[Dict], criterion: nn.Module,
             device: torch.device, num_classes: int) -> Dict[str, float]:
    model.eval()
    labels, scores, total_loss = [], [], 0.0
    with torch.no_grad():
        for record in records:
            probs, att = model(torch.from_numpy(record["x"]).to(device),
                           torch.from_numpy(record["edge_index"]).to(device))
            target = torch.tensor([record["y"]], device=device)
            total_loss += float(criterion(torch.log(probs + 1e-9).unsqueeze(0), target).item())
            labels.append(record["y"])
            scores.append(probs.cpu().numpy())
    if not labels:
        return {name: float("nan") for name in ("loss", "accuracy", "bacc", "auc", "macro_f1")}
    labels_array, score_array = np.asarray(labels), np.vstack(scores)
    predictions = score_array.argmax(axis=1)
    try:
        auc = roc_auc_score(labels_array, score_array, multi_class="ovr", labels=np.arange(num_classes))
    except ValueError:
        auc = float("nan")
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels_array, predictions, average="macro", zero_division=0
    )
    return {
        "loss": total_loss / len(labels),
        "accuracy": accuracy_score(labels_array, predictions),
        "bacc": balanced_accuracy_score(labels_array, predictions),
        "auc": auc,
        "macro_f1": macro_f1,
    }


def train_one_fold(train_records: List[Dict], val_records: List[Dict], test_records: List[Dict],
                   args: argparse.Namespace, fold: int, num_classes: int,
                   input_dim: int, device: torch.device) -> Tuple[Dict, Dict, int]:
    set_seed(args.seed + fold)

    model = GraphMIL(input_dim=input_dim,
                    gnn_type=args.gnn,
                    gnn_hidden=int(config.get('gnn_hidden', 128)),
                    gnn_layers=int(config.get('gnn_layers', 2)),
                    gnn_dropout=float(config.get('gnn_dropout', 0.0)),
                    gnn_heads=int(config.get('gnn_heads', 4)),
                    gnn_concat=bool(config.get('gnn_concat', True)),
                    att_dim=int(config.get('att_dim', 64)),
                    pool_dropout=float(config.get('pool_dropout', 0.0)),
                    classifier_dim=int(config.get('classifier_dim', 64)),
                    classifier_light=bool(config.get('classifier_light', False)),
                    num_classes=num_classes).to(device)

    counts = Counter(record["y"] for record in train_records)
    weights = torch.tensor([len(train_records) / (num_classes * counts.get(c, 1))
                            for c in range(num_classes)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    best_state, best_bacc, no_improvement, best_epoch = None, -np.inf, 0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(train_records))
        for index in order:
            record = train_records[int(index)]
            optimizer.zero_grad()
            probs, _ = model(torch.from_numpy(record["x"]).to(device),
                           torch.from_numpy(record["edge_index"]).to(device))
            loss = criterion(torch.log(probs + 1e-9).unsqueeze(0), torch.tensor([record["y"]], device=device))
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(model, val_records, criterion, device, num_classes)
        if val_metrics["bacc"] > best_bacc + args.min_delta:
            best_bacc, no_improvement, best_epoch = val_metrics["bacc"], 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement += 1
        if no_improvement >= args.patience:
            break

    model.load_state_dict(best_state)
    return (evaluate(model, val_records, criterion, device, num_classes),
            evaluate(model, test_records, criterion, device, num_classes), best_epoch)


def aggregate(metric_rows: List[Dict], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{metric}_{stat}": float(np.nanmean([row[metric] for row in metric_rows]))
            if stat == "mean" else float(np.nanstd([row[metric] for row in metric_rows], ddof=0))
            for metric in ("accuracy", "bacc", "auc", "macro_f1") for stat in ("mean", "std")}


RESULT_KEY = ["embedding_model", "graph_variant", "graph_model", "seed", "hidden_dim",
              "num_layers", "dropout", "learning_rate", "weight_decay"]


def export_detailed_fold_results(fold_records: List[Dict], output_path: Path) -> None:
    """Saves unaggregated per-fold test metrics for statistical testing (Friedman/Wilcoxon)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(fold_records)
    
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Drop duplicates in case an experiment was rerun
        df_combined = df_combined.drop_duplicates(
            subset=["embedding_model", "graph_variant", "graph_model", "fold", "seed"], 
            keep="last"
        )
    else:
        df_combined = df_new
        
    df_combined.to_csv(output_path, index=False)


def save_results(results: pd.DataFrame, output_path: Path) -> None:
    """Atomically replace the CSV so completed variants survive job timeouts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    results.sort_values(["embedding_model", "graph_variant", "graph_model"]).to_csv(
        temporary_path, index=False
    )
    os.replace(temporary_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", nargs="*", help="Embedding checkpoint basenames; defaults to all graph outputs.")
    parser.add_argument("--variants", nargs="*", default=graph_variants())
    parser.add_argument("--folds", nargs="*", type=int, default=list(range(5)))
    parser.add_argument("--gnn", nargs="+", choices=GNN_TYPES, default=list(GNN_TYPES),
                        help="GNN architectures to run; defaults to all supported architectures.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=16)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results-csv", type=Path, default=Path("gnn_results/common_results.csv"))
    parser.add_argument("--job-id", type=str, default="0", help="Unique identifier for parallel jobs")
    return parser.parse_args()


def run_gnn_experiments(args: argparse.Namespace) -> None:
    root, device = args.root.resolve(), torch.device(args.device)
    available_models = sorted(path.name for path in (root / "graph_outputs").iterdir() if path.is_dir())
    models = args.models or available_models

    unknown = sorted(set(args.variants) - set(graph_variants()))
    if unknown:
        raise ValueError(f"Unsupported variants: {unknown}")

    output_dir = args.results_csv.parent if args.results_csv.is_absolute() else root / args.results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"results_job_{args.job_id}.csv"
    detailed_csv_path = output_dir / f"detailed_fold_results_job_{args.job_id}.csv"

    completed = set()
    for existing_csv in output_dir.glob("results_job_*.csv"):
        try:
            df_existing = pd.read_csv(existing_csv)
            if set(RESULT_KEY).issubset(df_existing.columns):
                completed.update(set(map(tuple, df_existing[RESULT_KEY].itertuples(index=False, name=None))))
        except Exception:
            pass

    if output_path.exists():
        results = pd.read_csv(output_path)
    else:
        results = pd.DataFrame()

    for embedding_model in models:
        if embedding_model not in available_models:
            raise FileNotFoundError(f"No graph artifacts for embedding model {embedding_model}")
        for variant in args.variants:
            experiment_key = (embedding_model, variant, args.gnn, args.seed, args.hidden_dim,
                              args.num_layers, args.dropout, args.learning_rate, args.weight_decay)
            if experiment_key in completed:
                print(f"Skipping completed experiment: {embedding_model} | {variant} | {args.gnn}")
                continue
            fold_validation, fold_test, epochs = [], [], []
            dedicated_fold_results = []
            for fold in args.folds:
                train = load_fold_records(root, embedding_model, fold, "train", variant)
                validation = load_fold_records(root, embedding_model, fold, "val", variant)
                test = load_fold_records(root, embedding_model, fold, "test", variant)
                all_labels = [record["y"] for record in train + validation + test]
                num_classes = max(all_labels) + 1
                val_metrics, test_metrics, best_epoch = train_one_fold(
                    train, validation, test, args, fold, num_classes, train[0]["x"].shape[1], device
                )
                fold_validation.append(val_metrics)
                fold_test.append(test_metrics)
                epochs.append(best_epoch)
                dedicated_fold_results.append({
                    "embedding_model": embedding_model,
                    "graph_variant": variant,
                    "graph_model": args.gnn,
                    "fold": fold,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })
                print(f"{embedding_model} | {variant} | fold {fold}: "
                      f"val BAcc={val_metrics['bacc']:.4f}, test BAcc={test_metrics['bacc']:.4f}")
            result = {
                "embedding_model": embedding_model,
                "graph_variant": variant,
                "graph_model": args.gnn,
                "num_folds": len(args.folds),
                "seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "best_epoch_mean": float(np.mean(epochs)),
                "best_epoch_std": float(np.std(epochs, ddof=0)),
                **aggregate(fold_validation, "val"),
                **aggregate(fold_test, "test"),
            }
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            results = results.drop_duplicates(RESULT_KEY, keep="last")
            save_results(results, output_path)
            export_detailed_fold_results(dedicated_fold_results, detailed_csv_path)
            completed.add(experiment_key)
            print(f"Saved {len(results)} completed experiment rows to {output_path}")


def main() -> None:
    base_args = parse_args()
    for gnn_type in base_args.gnn:
        args = copy.copy(base_args)
        args.gnn = gnn_type
        run_gnn_experiments(args)


if __name__ == "__main__":
    main()
