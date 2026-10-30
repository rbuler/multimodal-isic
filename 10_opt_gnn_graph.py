"""Ray Tune search for graph-MIL hyperparameters.

Each invocation adds new train/validation-only trials to a shared history and
refreshes the best ten configurations for every architecture. The Tune stage
uses epoch-level ASHA pruning on one screening fold. Test data is never loaded
by either stage; confirmation evaluates retained candidates across all validation folds.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL = "a9d7feb3402a4670bbcfa73f534acab7"
ARCHITECTURES = (
	"mlp", "gcn", "gat", "gatv2", "gin", "graphsage", "transformer", "fagcn", "gcnii"
)
GRAPH_VARIANTS = (
	["grid4", "grid8"]
	+ [f"knn{k}" for k in range(1, 9)] + ["knn12", "knn16"]
	+ [f"random{k}" for k in range(1, 9)] + ["random12", "random16"]
)
RESULTS_DIR = ROOT / "gnn_results_raytune"
TRIALS_PATH = RESULTS_DIR / "hparam_search_trials.csv"
TOP10_PATH = RESULTS_DIR / "hparam_search_top10_per_execution.csv"
GLOBAL_TOP10_PATH = RESULTS_DIR / "hparam_search_global_top10.csv"
CONFIRMATION_PATH = RESULTS_DIR / "hparam_search_five_fold_confirmation.csv"


def load_training_helpers():
	"""Load the existing model and graph-record implementation without copying it."""
	path = ROOT / "05_train_gnns.py"
	spec = importlib.util.spec_from_file_location("train_gnns_for_search", path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load training helpers from {path}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


def parser() -> argparse.Namespace:
	arg_parser = argparse.ArgumentParser(description=__doc__)
	arg_parser.add_argument("--mode", choices=("search", "confirm"), default="search")
	arg_parser.add_argument("--architecture", choices=("all",) + ARCHITECTURES, default="all")
	arg_parser.add_argument("--num-samples", type=int, default=100)
	arg_parser.add_argument("--max-concurrent", type=int, default=None)
	arg_parser.add_argument("--gpu-per-trial", type=float, default=.25)
	arg_parser.add_argument("--cpu-per-trial", type=float, default=8.0)
	arg_parser.add_argument("--search-fold", type=int, default=0)
	arg_parser.add_argument("--seed", type=int, default=42)
	arg_parser.add_argument("--epochs", type=int, default=200)
	arg_parser.add_argument("--patience", type=int, default=16)
	arg_parser.add_argument("--grace-period", type=int, default=10)
	arg_parser.add_argument("--root", type=Path, default=ROOT)
	arg_parser.add_argument("--candidates", type=Path, default=GLOBAL_TOP10_PATH)
	arg_parser.add_argument("--output", type=Path, default=CONFIRMATION_PATH)
	return arg_parser.parse_args()


def search_space(architecture: str) -> dict[str, Any]:
	space: dict[str, Any] = {
		"architecture": tune.choice([architecture]),
		"hidden_dim": tune.choice([64, 128, 256, 384, 512]),
		"num_layers": tune.choice([1, 2, 3, 4, 5]),
		"dropout": tune.choice([0.3, 0.4, 0.5, 0.6, 0.7]),
		"att_dim": tune.choice([64, 128, 256, 384, 512]),
		"att_heads": tune.choice([1, 2, 4, 8]),
		"pool_dropout": tune.choice([0.3, 0.5, 0.6, 0.7]),
		"classifier_dim": tune.choice([64, 128, 256, 384, 512]),
		"classifier_light": tune.choice([True, False]),
		"use_residual": tune.choice([True, False]),
		"use_layer_norm": tune.choice([True, False]),
		"optimizer": tune.choice(["adam", "adamw"]),
		"lr": tune.loguniform(1e-6, 1e-3),
		"weight_decay": tune.loguniform(1e-8, 1e-3),
	}
	if architecture != "mlp":
		space.update({
			"graph_variant": tune.choice(GRAPH_VARIANTS),
			"gnn_heads": tune.choice([1, 2, 4, 8]),
			"gnn_concat": tune.choice([True, False]),
		})
	else:
		space["graph_variant"] = tune.choice(["none"])
		space["gnn_heads"] = tune.choice([1])
		space["gnn_concat"] = tune.choice([False])
	return space


def train_validation_trial(
	config: dict[str, Any], *, root: str, embedding_model: str, folds: list[int],
	seed: int, max_epochs: int, patience: int,
) -> None:
	"""Train one configuration across folds using validation data only."""
	helpers = load_training_helpers()
	start = time.perf_counter()
	architecture = config["architecture"]
	variant = config["graph_variant"]
	load_variant = "grid4" if architecture == "mlp" else variant
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	fold_metrics: list[dict[str, float]] = []
	
	for fold in folds:
		train_records = helpers.load_fold_records(Path(root), embedding_model, fold, "train", load_variant)
		val_records = helpers.load_fold_records(Path(root), embedding_model, fold, "val", load_variant)
		input_dim = train_records[0]["x"].shape[1]
		num_classes = max(record["y"] for record in train_records + val_records) + 1
		helpers.set_seed(seed + fold)
		
		model = helpers.GraphMIL(
			input_dim=input_dim, gnn_type=architecture,
			gnn_hidden=config["hidden_dim"], gnn_layers=config["num_layers"],
			gnn_dropout=config["dropout"], gnn_heads=config["gnn_heads"],
			gnn_concat=config["gnn_concat"], att_dim=config["att_dim"],
			att_heads=config["att_heads"], pool_dropout=config["pool_dropout"],
			classifier_dim=config["classifier_dim"], classifier_light=config["classifier_light"],
			num_classes=num_classes, use_residual=config["use_residual"],
			use_layer_norm=config["use_layer_norm"],
		).to(device)
		
		optimizer_class = torch.optim.Adam if config["optimizer"] == "adam" else torch.optim.AdamW
		optimizer = optimizer_class(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
		criterion = torch.nn.CrossEntropyLoss()
		rng = np.random.default_rng(seed + fold)
		best_state = None
		best_fold_bacc = -np.inf
		no_improvement = 0
		
		for epoch in range(1, max_epochs + 1):
			model.train()
			
			for index in rng.integers(0, len(train_records), len(train_records)):
				record = train_records[int(index)]
				optimizer.zero_grad()
				edge_index = None if record["edge_index"] is None else torch.from_numpy(record["edge_index"]).to(device)
				probs, _ = model(torch.from_numpy(record["x"]).to(device), edge_index=edge_index)
				loss = criterion(torch.log(probs + 1e-9).unsqueeze(0), torch.tensor([record["y"]], device=device))
				loss.backward()
				optimizer.step()
				
			metrics = helpers.evaluate(model, val_records, criterion, device, num_classes)
			
			if metrics["bacc"] > best_fold_bacc + 1e-6:
				best_fold_bacc = metrics["bacc"]
				best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
				no_improvement = 0
			else:
				no_improvement += 1
			tune.report({
				"val_bacc": float(metrics["bacc"]),
				"training_iteration": epoch,
				"fold": fold,
			})
			if no_improvement >= patience:
				break
			
		if best_state is not None:
			model.load_state_dict(best_state)

		fold_metrics.append(helpers.evaluate(model, val_records, criterion, device, num_classes))

	def mean_metric(name: str) -> float:
		return float(np.mean([item[name] for item in fold_metrics]))

	tune.report({
		"val_bacc": mean_metric("bacc"),
		"val_bacc_std": float(np.std([item["bacc"] for item in fold_metrics], ddof=0)),
		"val_accuracy": mean_metric("accuracy"),
		"val_auc": mean_metric("auc"),
		"val_macro_f1": mean_metric("macro_f1"),
		"search_epochs": epoch,
		"search_seconds": time.perf_counter() - start,
		"search_folds": json.dumps(folds),
	})


def json_config(config: dict[str, Any]) -> str:
	return json.dumps(config, sort_keys=True, separators=(",", ":"))


def append_and_refresh_top10(rows: list[dict[str, Any]], execution_id: str) -> None:
	RESULTS_DIR.mkdir(parents=True, exist_ok=True)
	new_frame = pd.DataFrame(rows)
	if new_frame.empty:
		return
	new_frame["execution_id"] = execution_id
	new_frame["candidate_id"] = new_frame["execution_id"] + "_" + new_frame["trial_id"].astype(str)
	if TRIALS_PATH.exists():
		old_frame = pd.read_csv(TRIALS_PATH)
		if "execution_id" not in old_frame:
			old_frame["execution_id"] = "legacy"
		if "candidate_id" not in old_frame:
			old_frame["candidate_id"] = "legacy_" + old_frame["trial_id"].astype(str)
		frame = pd.concat([old_frame, new_frame], ignore_index=True)
		frame = frame.drop_duplicates(subset=["candidate_id"], keep="last")
	else:
		frame = new_frame
	frame = frame.sort_values(["architecture", "val_bacc"], ascending=[True, False])
	frame.to_csv(TRIALS_PATH, index=False)

	top10_per_execution = (
		frame.sort_values(
			["execution_id", "architecture", "val_bacc"],
			ascending=[True, True, False],
		)
		.groupby(["execution_id", "architecture"], as_index=False, group_keys=False)
		.head(10)
		.copy()
	)
	top10_per_execution["search_rank"] = top10_per_execution.groupby(
		["execution_id", "architecture"]
	)["val_bacc"].rank(method="first", ascending=False).astype(int)
	top10_per_execution.to_csv(TOP10_PATH, index=False)

	global_top10 = (
		frame.sort_values(["architecture", "val_bacc"], ascending=[True, False])
		.groupby("architecture", as_index=False, group_keys=False)
		.head(10)
		.reset_index(drop=True)
	)
	global_top10.to_csv(GLOBAL_TOP10_PATH, index=False)
	print(f"Saved {len(frame)} trial rows to {TRIALS_PATH}")
	print(f"Saved {len(top10_per_execution)} per-execution candidates to {TOP10_PATH}")
	print(f"Saved {len(global_top10)} global top-ten rows to {GLOBAL_TOP10_PATH}")


def run_architecture(architecture: str, args: argparse.Namespace, execution_id: str) -> list[dict[str, Any]]:
	storage_path = args.root / "tune_mil_outputs" / "ray_results"
	storage_path.mkdir(parents=True, exist_ok=True)
	available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
	available_cpus = max(1, (os.cpu_count() or 1) - 4)
	
	if args.gpu_per_trial > 0:
		max_by_gpu = int(available_gpus / args.gpu_per_trial)
	else:
		max_by_gpu = 10**9
		
	max_by_cpu = int(available_cpus / max(args.cpu_per_trial, 1e-9))
	resource_capacity = max(1, min(max_by_gpu, max_by_cpu))
	max_concurrent = resource_capacity if args.max_concurrent is None else min(args.max_concurrent, resource_capacity)
	
	if args.max_concurrent is None:
		print(
			f"Automatic concurrency: {max_concurrent} trials "
			f"({available_gpus} GPUs at {args.gpu_per_trial} GPU/trial, "
			f"{available_cpus} CPUs at {args.cpu_per_trial} CPU/trial)."
		)
		
	scheduler = ASHAScheduler(
		metric="val_bacc", mode="max", max_t=args.epochs,
		grace_period=min(args.grace_period, args.epochs), reduction_factor=2,
	)
	
	reporter = CLIReporter(metric_columns=["val_bacc", "val_accuracy", "val_auc", "val_macro_f1", "training_iteration"])
	analysis = tune.run(
		tune.with_parameters(
			train_validation_trial,
			root=str(args.root), embedding_model=EMBEDDING_MODEL, folds=[args.search_fold],
			seed=args.seed, max_epochs=args.epochs, patience=args.patience,
		),
		config=search_space(architecture),
		num_samples=args.num_samples,
		resources_per_trial={"cpu": args.cpu_per_trial, "gpu": args.gpu_per_trial},
		max_concurrent_trials=max_concurrent,
		scheduler=scheduler,
		progress_reporter=reporter,
		storage_path=str(storage_path),
		name=f"hparam_search_{architecture}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
		verbose=1,
	)
	rows = []
	
	for trial in analysis.trials:
		result = trial.last_result or {}
		row = {
			"trial_id": trial.trial_id,
			"execution_id": execution_id,
			"architecture": architecture,
			"embedding_model": EMBEDDING_MODEL,
			"val_bacc": result.get("val_bacc", np.nan),
			"val_bacc_std": result.get("val_bacc_std", np.nan),
			"val_accuracy": result.get("val_accuracy", np.nan),
			"val_auc": result.get("val_auc", np.nan),
			"val_macro_f1": result.get("val_macro_f1", np.nan),
			"search_epochs": result.get("search_epochs", np.nan),
			"search_seconds": result.get("search_seconds", np.nan),
			"search_folds": json.dumps([args.search_fold]),
			"timestamp_utc": datetime.now(timezone.utc).isoformat(),
			"params_json": json_config(trial.config),
		}
		rows.append(row)
	return rows


def confirm_candidate(
	params: dict[str, Any], helpers: Any, root: Path, fold: int,
	seed: int, max_epochs: int, patience: int,
) -> dict[str, float]:
	architecture = params["architecture"]
	variant = "grid4" if architecture == "mlp" else params["graph_variant"]
	train_records = helpers.load_fold_records(root, EMBEDDING_MODEL, fold, "train", variant)
	val_records = helpers.load_fold_records(root, EMBEDDING_MODEL, fold, "val", variant)
	input_dim = train_records[0]["x"].shape[1]
	num_classes = max(record["y"] for record in train_records + val_records) + 1
	helpers.set_seed(seed + fold)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = helpers.GraphMIL(
		input_dim=input_dim, gnn_type=architecture,
		gnn_hidden=int(params["hidden_dim"]), gnn_layers=int(params["num_layers"]),
		gnn_dropout=float(params["dropout"]), gnn_heads=int(params["gnn_heads"]),
		gnn_concat=bool(params["gnn_concat"]), att_dim=int(params["att_dim"]),
		att_heads=int(params["att_heads"]), pool_dropout=float(params["pool_dropout"]),
		classifier_dim=int(params["classifier_dim"]), classifier_light=bool(params["classifier_light"]),
		num_classes=num_classes, use_residual=bool(params["use_residual"]),
		use_layer_norm=bool(params["use_layer_norm"]),
	).to(device)
	optimizer_class = torch.optim.Adam if params["optimizer"] == "adam" else torch.optim.AdamW
	optimizer = optimizer_class(model.parameters(), lr=float(params["lr"]), weight_decay=float(params["weight_decay"]))
	criterion = torch.nn.CrossEntropyLoss()
	rng = np.random.default_rng(seed + fold)
	best_state = None
	best_bacc = -np.inf
	no_improvement = 0
	for _epoch in range(1, max_epochs + 1):
		model.train()
		for index in rng.integers(0, len(train_records), len(train_records)):
			record = train_records[int(index)]
			optimizer.zero_grad()
			edge_index = None if record["edge_index"] is None else torch.from_numpy(record["edge_index"]).to(device)
			probs, _ = model(torch.from_numpy(record["x"]).to(device), edge_index=edge_index)
			loss = criterion(torch.log(probs + 1e-9).unsqueeze(0), torch.tensor([record["y"]], device=device))
			loss.backward()
			optimizer.step()
		metrics = helpers.evaluate(model, val_records, criterion, device, num_classes)
		if metrics["bacc"] > best_bacc + 1e-6:
			best_bacc = metrics["bacc"]
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
			no_improvement = 0
		else:
			no_improvement += 1
		if no_improvement >= patience:
			break
	if best_state is not None:
		model.load_state_dict(best_state)
	return helpers.evaluate(model, val_records, criterion, device, num_classes)


def confirm_candidates(args: argparse.Namespace) -> None:
	candidates = pd.read_csv(args.candidates)
	helpers = load_training_helpers()
	rows = []
	for candidate_index, candidate in candidates.iterrows():
		params = json.loads(candidate["params_json"])
		fold_metrics = []
		for fold in args.folds:
			metrics = confirm_candidate(params, helpers, args.root, fold, args.seed, args.epochs, args.patience)
			fold_metrics.append(metrics)
			print(f"candidate {candidate_index + 1}/{len(candidates)}, fold {fold}: BAcc={metrics['bacc']:.4f}")
		rows.append({
			"candidate_id": candidate.get("candidate_id", candidate.get("trial_id", candidate_index)),
			"architecture": params["architecture"],
			"graph_variant": params["graph_variant"],
			"source_val_bacc": candidate.get("val_bacc", np.nan),
			"confirmation_val_bacc_mean": np.mean([item["bacc"] for item in fold_metrics]),
			"confirmation_val_bacc_std": np.std([item["bacc"] for item in fold_metrics], ddof=1),
			"confirmation_accuracy_mean": np.mean([item["accuracy"] for item in fold_metrics]),
			"confirmation_auc_mean": np.mean([item["auc"] for item in fold_metrics]),
			"confirmation_macro_f1_mean": np.mean([item["macro_f1"] for item in fold_metrics]),
			"n_folds": len(args.folds), "folds": json.dumps(args.folds),
			"params_json": json.dumps(params, sort_keys=True),
		})
	result = pd.DataFrame(rows).sort_values(
		["architecture", "confirmation_val_bacc_mean", "confirmation_val_bacc_std"],
		ascending=[True, False, True],
	)
	result["confirmation_rank"] = result.groupby("architecture")["confirmation_val_bacc_mean"].rank(
		method="first", ascending=False,
	).astype(int)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(args.output, index=False)
	print(f"Saved five-fold confirmation results to {args.output}")


def main() -> None:
	args = parser()
	if args.mode == "confirm":
		args.folds = list(range(5))
		confirm_candidates(args)
		return
	architectures = ARCHITECTURES if args.architecture == "all" else (args.architecture,)
	execution_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	all_rows = []
	
	for architecture in architectures:
		print(f"\n=== Searching {architecture} ({args.num_samples} new trials) ===")
		rows = run_architecture(architecture, args, execution_id)
		append_and_refresh_top10(rows, execution_id)
		all_rows.extend(rows)
	print(f"Completed {len(all_rows)} new trials across {len(architectures)} architectures.")


if __name__ == "__main__":
	main()