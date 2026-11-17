"""Evaluate the global top-ten Ray Tune candidates on held-out test folds.

Each Ray trial owns one row from ``hparam_search_global_top10.csv``.  It trains
that configuration independently on folds 0--4, selects the best epoch using
the corresponding validation split, and evaluates the selected checkpoint on
the held-out test split.  The resulting test mean and standard deviation are
written to ``test_hparam_search_global_top10.csv``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from ray import tune
from ray.tune import CLIReporter


ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "a9d7feb3402a4670bbcfa73f534acab7"
RESULTS_DIR = ROOT / "gnn_results_raytune"
DEFAULT_CANDIDATES = RESULTS_DIR / "hparam_search_global_top10.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "test_hparam_search_global_top10.csv"


def load_training_helpers():
	path = ROOT / "05_train_gnns.py"
	spec = importlib.util.spec_from_file_location("train_gnns_for_test", path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load training helpers from {path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def parser() -> argparse.Namespace:
	arg_parser = argparse.ArgumentParser(description=__doc__)
	arg_parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
	arg_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
	arg_parser.add_argument("--root", type=Path, default=ROOT)
	arg_parser.add_argument("--folds", nargs="*", type=int, default=list(range(5)))
	arg_parser.add_argument("--seed", type=int, default=42)
	arg_parser.add_argument("--epochs", type=int, default=200)
	arg_parser.add_argument("--patience", type=int, default=16)
	arg_parser.add_argument("--max-concurrent", type=int, default=None)
	arg_parser.add_argument("--gpu-per-trial", type=float, default=0.25)
	arg_parser.add_argument("--cpu-per-trial", type=float, default=8.0)
	arg_parser.add_argument("--max-cpus", type=int, default=None)
	return arg_parser.parse_args()


def train_and_test_candidate(
	config: dict[str, Any], *, root: str, embedding_model: str,
	folds: list[int], seed: int, max_epochs: int, patience: int,
) -> None:
	"""Train one candidate across all folds and report held-out test metrics."""
	params = config["candidate"]["params_json"]
	helpers = load_training_helpers()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	fold_metrics: list[dict[str, float]] = []
	start = time.perf_counter()

	for fold in folds:
		architecture = params["architecture"]
		variant = "grid4" if architecture == "mlp" else params["graph_variant"]
		train_records = helpers.load_fold_records(Path(root), embedding_model, fold, "train", variant)
		val_records = helpers.load_fold_records(Path(root), embedding_model, fold, "val", variant)
		test_records = helpers.load_fold_records(Path(root), embedding_model, fold, "test", variant)
		input_dim = train_records[0]["x"].shape[1]
		num_classes = max(record["y"] for record in train_records + val_records + test_records) + 1
		helpers.set_seed(seed + fold)

		model = helpers.GraphMIL(
			input_dim=input_dim, gnn_type=architecture,
			gnn_hidden=int(params["hidden_dim"]), gnn_layers=int(params["num_layers"]),
			gnn_dropout=float(params["dropout"]), gnn_heads=int(params["gnn_heads"]),
			gnn_concat=bool(params["gnn_concat"]), att_dim=int(params["att_dim"]),
			gcnii_alpha=float(params.get("gcnii_alpha", 0.1)),
			gcnii_theta=float(params.get("gcnii_theta", 0.5)),
			fagcn_eps=float(params.get("fagcn_eps", 0.1)),
			gin_train_eps=bool(params.get("gin_train_eps", True)),
			att_heads=int(params["att_heads"]), pool_dropout=float(params["pool_dropout"]),
			classifier_dim=int(params["classifier_dim"]),
			classifier_light=bool(params["classifier_light"]), num_classes=num_classes,
			use_residual=bool(params["use_residual"]),
			use_layer_norm=bool(params["use_layer_norm"]),
		).to(device)

		optimizer_class = torch.optim.Adam if params["optimizer"] == "adam" else torch.optim.AdamW
		optimizer = optimizer_class(
			model.parameters(), lr=float(params["lr"]),
			weight_decay=float(params["weight_decay"]),
		)
		criterion = torch.nn.CrossEntropyLoss()
		labels = np.array([record["y"] for record in train_records])
		class_counts = Counter(labels)
		sample_weights = np.array(
			[1.0 / class_counts[int(label)] for label in labels],
			dtype=np.float64,
		)
		sampler = WeightedRandomSampler(
			weights=torch.from_numpy(sample_weights),
			num_samples=len(sample_weights),
			replacement=True,
		)
		best_state = None
		best_loss = np.inf
		no_improvement = 0

		for _epoch in range(1, max_epochs + 1):
			model.train()
			for index in sampler:
				record = train_records[int(index)]
				optimizer.zero_grad()
				edge_index = record["edge_index"]
				edge_index = None if edge_index is None else torch.from_numpy(edge_index).to(device)
				probs, _ = model(torch.from_numpy(record["x"]).to(device), edge_index=edge_index)
				loss = criterion(
					torch.log(probs + 1e-9).unsqueeze(0),
					torch.tensor([record["y"]], device=device),
				)
				loss.backward()
				optimizer.step()

			val_metrics = helpers.evaluate(model, val_records, criterion, device, num_classes)
			if val_metrics["loss"] < best_loss - 1e-6:
				best_loss = val_metrics["loss"]
				best_state = {
					key: value.detach().cpu().clone()
					for key, value in model.state_dict().items()
				}
				no_improvement = 0
			else:
				no_improvement += 1
			if no_improvement >= patience:
				break

		if best_state is not None:
			model.load_state_dict(best_state)
		test_metrics = helpers.evaluate(model, test_records, criterion, device, num_classes)
		fold_metrics.append(test_metrics)
		tune.report({
			"fold": fold,
			"test_bacc": float(test_metrics["bacc"]),
			"test_accuracy": float(test_metrics["accuracy"]),
		})

		del model, optimizer
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def mean(name: str) -> float:
		return float(np.nanmean([item[name] for item in fold_metrics]))

	def std(name: str) -> float:
		return float(np.nanstd([item[name] for item in fold_metrics], ddof=0))

	tune.report({
		"test_accuracy_mean": mean("accuracy"),
		"test_accuracy_std": std("accuracy"),
		"test_bacc_mean": mean("bacc"),
		"test_bacc_std": std("bacc"),
		"test_auc_mean": mean("auc"),
		"test_auc_std": std("auc"),
		"test_macro_f1_mean": mean("macro_f1"),
		"test_macro_f1_std": std("macro_f1"),
		"test_seconds": time.perf_counter() - start,
	})


def resource_settings(args: argparse.Namespace) -> tuple[int, int, int]:
	available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
	available_cpus = args.max_cpus or max(1, (os.cpu_count() or 1) - 4)
	if args.gpu_per_trial > 0:
		max_by_gpu = int(available_gpus / args.gpu_per_trial)
	else:
		max_by_gpu = 10**9
	max_by_cpu = int(available_cpus / max(args.cpu_per_trial, 1e-9))
	capacity = max(1, min(max_by_gpu, max_by_cpu))
	max_concurrent = capacity if args.max_concurrent is None else min(args.max_concurrent, capacity)
	print(
		f"Automatic concurrency: {max_concurrent} trials "
		f"({available_gpus} GPUs at {args.gpu_per_trial} GPU/trial, "
		f"{available_cpus} CPUs at {args.cpu_per_trial} CPU/trial)."
	)
	return available_gpus, available_cpus, max_concurrent


def main() -> None:
	args = parser()
	candidates = pd.read_csv(args.candidates)
	required = {"trial_id", "architecture", "params_json"}
	missing = sorted(required - set(candidates.columns))
	if missing:
		raise ValueError(f"Missing required candidate columns: {missing}")
	if not args.folds:
		raise ValueError("At least one fold is required")

	configs = []
	for _, candidate in candidates.iterrows():
		params = json.loads(candidate["params_json"])
		configs.append({
			"trial_id": str(candidate["trial_id"]),
			"candidate_id": str(candidate.get("candidate_id", candidate["trial_id"])),
			"architecture": str(candidate["architecture"]),
			"source_val_bacc": float(candidate.get("val_bacc", np.nan)),
			"params_json": params,
		})
	if not configs:
		raise ValueError(f"No candidates found in {args.candidates}")

	_, _, max_concurrent = resource_settings(args)
	storage_path = args.root / "tune_mil_outputs" / "ray_test_results"
	storage_path.mkdir(parents=True, exist_ok=True)
	reporter = CLIReporter(metric_columns=["test_bacc_mean", "test_bacc_std", "test_accuracy_mean"])
	analysis = tune.run(
		tune.with_parameters(
			train_and_test_candidate,
			root=str(args.root), embedding_model=EMBEDDING_MODEL,
			folds=args.folds, seed=args.seed, max_epochs=args.epochs, patience=args.patience,
		),
		config={"candidate": tune.grid_search(configs)},
		resources_per_trial={"cpu": args.cpu_per_trial, "gpu": args.gpu_per_trial},
		max_concurrent_trials=max_concurrent,
		verbose=1,
		progress_reporter=reporter,
		storage_path=str(storage_path),
		name="heldout_test_global_top10",
	)

	rows = []
	for trial in analysis.trials:
		result = trial.last_result or {}
		candidate = trial.config["candidate"]
		rows.append({
			"trial_id": candidate["trial_id"],
			"candidate_id": candidate["candidate_id"],
			"architecture": candidate["architecture"],
			"graph_variant": candidate["params_json"].get("graph_variant"),
			"source_val_bacc": candidate["source_val_bacc"],
			"test_accuracy_mean": result.get("test_accuracy_mean", np.nan),
			"test_accuracy_std": result.get("test_accuracy_std", np.nan),
			"test_bacc_mean": result.get("test_bacc_mean", np.nan),
			"test_bacc_std": result.get("test_bacc_std", np.nan),
			"test_auc_mean": result.get("test_auc_mean", np.nan),
			"test_auc_std": result.get("test_auc_std", np.nan),
			"test_macro_f1_mean": result.get("test_macro_f1_mean", np.nan),
			"test_macro_f1_std": result.get("test_macro_f1_std", np.nan),
			"n_folds": len(args.folds),
			"folds": json.dumps(args.folds),
			"params_json": json.dumps(candidate["params_json"], sort_keys=True),
			"trial_status": trial.status,
		})

	result_frame = pd.DataFrame(rows).sort_values(["architecture", "test_bacc_mean"], ascending=[True, False])
	args.output.parent.mkdir(parents=True, exist_ok=True)
	result_frame.to_csv(args.output, index=False)
	print(f"Saved {len(result_frame)} held-out test results to {args.output}")


if __name__ == "__main__":
	main()