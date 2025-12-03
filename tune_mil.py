# %%
import os
import numpy as np
import torch
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import argparse
from argparse import Namespace
from utils import get_args_parser
import yaml
import pickle
from utils import get_args_parser
from utils_g_mil import train_mil, train_graph_mil
import cloudpickle, traceback
from datetime import datetime
from save_latent import extract_latents

def main():

    # Enable Ray's verbose pickle debug to surface non-serializable captured variables
    os.environ.setdefault("RAY_PICKLE_VERBOSE_DEBUG", "1")
    print(f"RAY_PICKLE_VERBOSE_DEBUG={os.environ['RAY_PICKLE_VERBOSE_DEBUG']} (verbose Ray pickling debug enabled)")
    print("If you see pickling errors, inspect the Ray output for names/types of non-serializable objects captured in scope.")
    args = Namespace(
        config_path="config.yml",
        # model_name="a9d7feb3402a4670bbcfa73f534acab7.pth",  # <-- the AE model basename to use
        model_name="e6b29aa3b47145ec935e675a13c4b71d.pth",
        num_samples=2000,
        # Allow a large default so the script can cap based on available resources
        max_concurrent=999,
        # Reduce per-trial CPU by default so we can run more trials concurrently
        cpus_per_trial=8.0,
        # Allow fractional GPU allocation so multiple trials can share a single GPU
        # (needs per-process memory fraction enforcement in the training code)
        gpus_per_trial=(0.25 if torch.cuda.is_available() else 0.0),
        num_workers=min(8, max(1, (os.cpu_count() or 1) // 2)),
        pin_memory=False,
        torch_threads=min(8, max(1, (os.cpu_count() or 1) // 2)),
        num_epochs=200,
        patience=16,
        seed=42,
        max_failures=5,        
    )

    def _parse_known_args_override(self, *override_args, **override_kwargs):
        return args, []

    argparse.ArgumentParser.parse_known_args = _parse_known_args_override

    parser = get_args_parser('config.yml')
    args, unknown = parser.parse_known_args()
    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    load = True
    if load:
        patch_train_df = '/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/dataframes_latents/patch_level_latents_train_df.pkl'
        patch_test_df = '/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/dataframes_latents/patch_level_latents_test_df.pkl'
        with open(patch_train_df, 'rb') as f:
            patch_train_df = pickle.load(f)
        with open(patch_test_df, 'rb') as f:
            patch_test_df = pickle.load(f)
    else:
        patch_level_train_df, patch_level_test_df, latent_pooled_train, latent_pooled_test, latent_raw_train, latent_raw_test = extract_latents(config, args.model_name, remove_background=False)
        patch_train_df = patch_level_train_df
        patch_test_df = patch_level_test_df
# %%
    patch_train_df['patient_id'] = patch_train_df['image_path'].apply(
        lambda x: os.path.basename(x).split('_')[1].split('.')[0]
    )
    patch_test_df['patient_id'] = patch_test_df['image_path'].apply(
        lambda x: os.path.basename(x).split('_')[1].split('.')[0]
    )

    def _sort_group_patches(g):
        """Return group DataFrame sorted by numeric patch id.

        Strategy:
        - If column `patch_id` exists, sort by it.
        - Otherwise try to parse a trailing integer from `image_path` basename
          (last underscore-separated token before extension).
        - If parsing fails, preserve original order.
        """
        if 'patch_id' in g.columns:
            try:
                return g.sort_values('patch_id')
            except Exception:
                return g

        def _extract_from_path(x):
            try:
                b = os.path.basename(x)
                name = os.path.splitext(b)[0]
                tok = name.split('_')[-1]
                return int(tok)
            except Exception:
                return None

        g = g.copy()
        g['_patch_num'] = g['image_path'].apply(_extract_from_path)
        if g['_patch_num'].notnull().all():
            return g.sort_values('_patch_num')
        # if we couldn't parse numeric ids for all rows, drop helper column and return original
        g = g.drop(columns=['_patch_num'], errors='ignore')
        return g

    train_patient_feats = [
        np.vstack(_sort_group_patches(g)['patch_latent'].values)
        for pid, g in patch_train_df.groupby('patient_id')
    ]
    train_patient_labels = [
        int(_sort_group_patches(g)['target'].mode().iat[0])
        for pid, g in patch_train_df.groupby('patient_id')
    ]
    test_patient_feats = [
        np.vstack(_sort_group_patches(g)['patch_latent'].values)
        for pid, g in patch_test_df.groupby('patient_id')
    ]
    test_patient_labels = [
        int(_sort_group_patches(g)['target'].mode().iat[0])
        for pid, g in patch_test_df.groupby('patient_id')
    ]

    input_dim = train_patient_feats[0].shape[1] if len(train_patient_feats) > 0 else 76

    tune_data = {
        'train_feats': train_patient_feats,
        'train_labels': train_patient_labels,
        'test_feats': test_patient_feats,
        'test_labels': test_patient_labels,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'torch_threads': args.torch_threads,
        'input_dim': input_dim,
    }

    total_cpus = os.cpu_count()-4 or 1
    total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Detected resources: {total_cpus} CPUs, {total_gpus} GPUs")
    # Tell training processes how much of a single GPU they are expected to use.
    # e.g. gpus_per_trial=0.25 -> each trial should limit itself to ~25% of GPU memory.
    os.environ.setdefault('PER_PROC_GPU_MEM_FRACTION', str(args.gpus_per_trial))
    
    ray.init(ignore_reinit_error=True, num_cpus=total_cpus, num_gpus=total_gpus)

    scheduler = ASHAScheduler(
        metric="val_bacc",
        mode="max",
        max_t=args.num_epochs,
        grace_period=10,
        reduction_factor=2)

    reporter = CLIReporter(metric_columns=["val_bacc", "val_acc", "val_auc", "training_iteration"])

    # Define both search spaces and select based on args.tune_type
    search_space_mil = {
        "hidden_dim": tune.randint(32, 1025),
        "att_dim": tune.randint(32, 1025),
        "dropout": tune.uniform(0.0, 0.75),
        "optimizer": tune.choice(["adam", "adamw"]),
        "lr": tune.loguniform(1e-7, 1e-3),
        "weight_decay": tune.uniform(0, 1e-3),
    }

    # search_space_graph = {
    #     # GNN architecture choices
    #     "gnn_type": tune.choice(["gcn", "gat"]),
    #     "gnn_hidden": tune.randint(32, 513),
    #     "gnn_layers": tune.randint(1, 8),
    #     "gnn_dropout": tune.uniform(0.0, 0.75),
    #     "connect_diagonals": tune.choice([False, True]),

    #     # MIL pooling / classifier
    #     "att_dim": tune.randint(16, 512),
    #     "pool_dropout": tune.uniform(0.0, 0.75),
    #     "classifier_dim": tune.randint(16, 512),

    #     # optimization
    #     "optimizer": tune.choice(["adam", "adamw", "sgd"]),
    #     "lr": tune.loguniform(1e-6, 1e-3),
    #     "weight_decay": tune.loguniform(1e-8, 1e-3),
    # }

    search_space_graph = {
        # GNN architecture choices (updated)
        "gnn_type": tune.choice(["gcn", "gat", "gin", "graphsage", "transformer"]),
        "gnn_hidden": tune.choice([64, 128, 256, 384, 512]),
        "gnn_layers": tune.choice([2, 3, 4, 5, 6, 7]),
        "gnn_dropout": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]),

        # Additional graph construction parameter
        # "k_neighbors": tune.choice([4, 8, 12, 16]),
        'connect_diagonals': tune.choice([False, True]),

        # MIL pooling / classifier (updated)
        "att_dim": tune.choice([64, 128, 256, 512]),
        "att_heads": tune.choice([1, 2, 4, 8]),
        "pool_dropout": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75]),
        "classifier_dim": tune.choice([64, 128, 256, 384, 512]),

        # Architectural flags
        "use_residual": tune.choice([True, False]),
        "use_layer_norm": tune.choice([True, False]),

        # Optimization (updated)
        "optimizer": tune.choice(["adam", "adamw"]),
        "lr": tune.loguniform(1e-6, 1e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-3),
    }


    # choose search space and train function
    tune_type = 'graph_mil' # 'mil' or 'graph_mil'
    
    if tune_type == 'graph_mil':
        search_space = search_space_graph
        train_fn = train_graph_mil
    else:
        search_space = search_space_mil
        train_fn = train_mil

    resources = {"cpu": float(args.cpus_per_trial), "gpu": float(args.gpus_per_trial)}

    max_by_cpu = int(total_cpus // max(1, int(max(1, args.cpus_per_trial))))
    # Support fractional gpus_per_trial (e.g. 0.25 -> 4 trials per GPU)
    if total_gpus > 0 and float(args.gpus_per_trial) > 0:
        try:
            max_by_gpu = int(total_gpus / float(args.gpus_per_trial))
        except Exception:
            max_by_gpu = int(total_gpus)
    else:
        max_by_gpu = int(1e9)
    cap_concurrency = max(1, min(max_by_cpu, max_by_gpu))
    if args.max_concurrent > cap_concurrency:
        print(f"Reducing max_concurrent from {args.max_concurrent} to {cap_concurrency} to match available resources.")
        args.max_concurrent = cap_concurrency



    def check_pickle(obj, name):
        try:
            cloudpickle.dumps(obj)
            print(f"[OK] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    for k,v in tune_data.items():
        check_pickle(v, f"tune_data['{k}']")
    check_pickle(train_fn, f"{train_fn.__name__} (function)")

    analysis = tune.run(
        tune.with_parameters(train_fn, data=tune_data, seed=args.seed, num_classes=int(config.get('num_classes', 7)),
                             device_str=('cuda' if torch.cuda.is_available() else 'cpu'),
                             patience=args.patience, max_epochs=args.num_epochs),
        resources_per_trial=resources,
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        max_concurrent_trials=args.max_concurrent,
        fail_fast=False,
    )

    ts = datetime.now().strftime("%y%m%d%H%M%S")

    best_trial = analysis.get_best_trial("val_bacc", mode="max", scope="all")
    best_config = analysis.get_best_config(metric="val_bacc", mode="max")
    best_df = analysis.results_df
    out_dir = "./tune_mil_outputs"
    os.makedirs(out_dir, exist_ok=True)

    results_path = os.path.join(out_dir, f"ray_tune_results_{ts}.csv")
    config_path = os.path.join(out_dir, f"best_config_{ts}.yaml")

    best_df.to_csv(results_path, index=False)
    with open(config_path, "w") as wf:
        yaml.dump(best_config, wf)

    print(f"Best trial: {best_trial}, results saved to {results_path}, best_config saved to {config_path}")
    ray.shutdown()

if __name__ == "__main__":
    main()

# %%

# %%