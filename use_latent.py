# %% IMPORTS
import os
import hashlib
import yaml
import pandas as pd
import uuid
import torch
import random
import numpy as np
import torch.nn as nn
from datetime import datetime
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score, accuracy_score
from save_latent import extract_latents
from fetch_experiments import fetch_experiment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from utils import get_args_parser
from utils_g_mil import AttentionMIL, PatientDataset, GraphMIL, build_graph


def _evaluate_model(state_dict):
            if state_dict is not None:
                model.load_state_dict(state_dict)
            model.eval()
            y_true = []
            y_score = []
            with torch.no_grad():
                for x, y_batch in test_loader:
                    x = x[0].to(device)
                    y_long = y_batch.to(device).long()
                    if mil_type == 'graph':
                        adj_norm, adj_mask, edge_index, edge_weight = build_graph(
                            x, graph_type=graph_type, k=k_neighbors,
                            connect_diagonals=connect_diagonals, device=device)
                        probs, att = model(x, adj=adj_norm, adj_mask=adj_mask,
                                         edge_index=edge_index, edge_weight=edge_weight)
                    else:
                        probs, att = model(x)
                    y_true.append(int(y_long.item()))
                    y_score.append(probs.cpu().numpy())

            if len(y_true) == 0:
                return {k: np.nan for k in ['micro','macro_p','macro_r','macro_f1','weighted_p','weighted_r','weighted_f1']}

            y_true_arr = np.array(y_true)
            y_score_arr = np.vstack(y_score)
            y_pred_arr = np.argmax(y_score_arr, axis=1)

            micro_acc = accuracy_score(y_true_arr, y_pred_arr)
            macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='macro', zero_division=0)
            weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='weighted', zero_division=0)

            return {'micro': micro_acc,
                    'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
                    'weighted_p': weighted_p, 'weighted_r': weighted_r, 'weighted_f1': weighted_f1}



parser = get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
# %%
# experiment_ids = list(range(798, 814)) + list(range(726, 732))
experiment_ids = [726, 799, 802, 803, 804, 805]


runs_df = fetch_experiment(experiment_ids=experiment_ids)
runs_df = runs_df[['sys/id',
                   'config/training_plan/parameters/norm_pix_loss',
                   'config/training_plan/parameters/include_lesion_mask',
                   'best_model_path']].copy()
runs_df['best_model_path'] = runs_df['best_model_path'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else x)

runs_df['micro_accuracy'] = 0.0
runs_df['macro_precision'] = 0.0
runs_df['macro_recall'] = 0.0
runs_df['macro_f1'] = 0.0
runs_df['weighted_precision'] = 0.0
runs_df['weighted_recall'] = 0.0
runs_df['weighted_f1'] = 0.0

results_rows = []

SEED = config['seed']

output_dir = os.path.join(os.getcwd(), "mil_results")
os.makedirs(output_dir, exist_ok=True)
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_id = uuid.uuid4().hex[:6]
out_csv = os.path.join(output_dir, f"0runs_df_mil_results_{date_str}_{unique_id}.csv")
config_out = os.path.join(output_dir, f"0config_{date_str}_{unique_id}.yml")

def _persist_results(df):
    # Save/overwrite the aggregated results CSV
    df.to_csv(out_csv, index=False)
    # Save the exact config used (once) for reproducibility
    try:
        if not os.path.exists(config_out):
            # Optionally compute and print a short hash of the config for traceability
            cfg_bytes = yaml.dump(config, sort_keys=False).encode("utf-8")
            cfg_hash = hashlib.sha1(cfg_bytes).hexdigest()[:8]
            with open(config_out, 'w') as f:
                f.write(f"# config_hash: {cfg_hash}\n")
                f.write(yaml.dump(config, sort_keys=False))
    except Exception as e:
        print(f"Warning: failed to persist config to {config_out}: {e}")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for idx, row in runs_df.iterrows():
    model_name = row['best_model_path']
    if not isinstance(model_name, str) or model_name == 'nan':
        print(f"Skipping row {idx} because best_model_path is missing")
        continue

    print(f"\n=== Processing run {idx} - model: {model_name} ===")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    try:
        (patch_level_latents_df_train,
         patch_level_latents_df_test,
         latent_pooled_train,
         latent_pooled_test,
         latent_raw_train,
         latent_raw_test) = extract_latents(config, path=model_name, remove_background=False)
    except Exception as e:
        print(f"  Error extracting latents for {model_name}: {e}")
        runs_df.loc[idx, ['micro_accuracy','macro_precision','macro_recall','macro_f1',
                          'weighted_precision','weighted_recall','weighted_f1']] = [np.nan]*7
        _persist_results(runs_df)
        
        continue

    patch_level_latents_df_train['patient_id'] = patch_level_latents_df_train['image_path'].apply(
        lambda x: os.path.basename(x).split('_')[1].split('.')[0]
    )
    patch_level_latents_df_test['patient_id'] = patch_level_latents_df_test['image_path'].apply(
        lambda x: os.path.basename(x).split('_')[1].split('.')[0]
    )

    drop_background = False
    if drop_background:
        patch_level_latents_df_train = patch_level_latents_df_train[patch_level_latents_df_train['patch_in_mask'] == 1].reset_index(drop=True)
        patch_level_latents_df_test = patch_level_latents_df_test[patch_level_latents_df_test['patch_in_mask'] == 1].reset_index(drop=True)

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

    # --- Prepare patient-level features/labels ---
    train_patient_features = [
        torch.tensor(np.vstack(_sort_group_patches(g)['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_level_latents_df_train.groupby('patient_id')
    ]
    train_patient_labels = [
        int(_sort_group_patches(g)['target'].mode().iat[0])
        for pid, g in patch_level_latents_df_train.groupby('patient_id')
    ]

    test_patient_features = [
        torch.tensor(np.vstack(_sort_group_patches(g)['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_level_latents_df_test.groupby('patient_id')
    ]
    test_patient_labels = [
        int(_sort_group_patches(g)['target'].mode().iat[0])
        for pid, g in patch_level_latents_df_test.groupby('patient_id')
    ]

    # --- 5-fold cross-validation on patient-level training data ---

    SPLITS = 5
    skf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)
    y = np.array(train_patient_labels)

    fold_test_metrics_bacc = []
    fold_test_metrics_loss = []

    test_dataset = PatientDataset(test_patient_features, test_patient_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_epochs = 200
    criterion = nn.CrossEntropyLoss()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        print(f"  Fold {fold_idx+1}/{SPLITS}: train {len(train_idx)} patients, val {len(val_idx)} patients")

        fold_train_feats = [train_patient_features[i] for i in train_idx]
        fold_train_labels = [int(train_patient_labels[i]) for i in train_idx]
        fold_val_feats = [train_patient_features[i] for i in val_idx]
        fold_val_labels = [int(train_patient_labels[i]) for i in val_idx]

        patient_labels_fold = np.array(fold_train_labels)
        class_counts = Counter(patient_labels_fold)
        weights = np.array([1.0 / class_counts[int(lbl)] for lbl in patient_labels_fold], dtype=np.float64)
        sample_weights = torch.from_numpy(weights)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_dataset = PatientDataset(fold_train_feats, fold_train_labels)
        val_dataset = PatientDataset(fold_val_feats, fold_val_labels)

        train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        torch.manual_seed(SEED + fold_idx)
        np.random.seed(SEED + fold_idx)
        random.seed(SEED + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + fold_idx)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## ------------------------------------------------ tune PARAMS ------------------------------------------------ ##
        mil_type = 'graph'  # 'classic' or 'graph'
        
        input_dim = train_dataset[0][0].shape[1]
        if mil_type == 'classic':
            if config['best_params']['use'] is False:
                model = AttentionMIL(input_dim=input_dim, hidden_dim=256, att_dim=128, dropout=0.5).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            else:
                best_params = config['best_params']
                model = AttentionMIL(input_dim=input_dim,
                                     hidden_dim=best_params['hidden_dim'],
                                     att_dim=best_params['att_dim'],
                                     dropout=best_params['dropout']).to(device)
                if best_params['optimizer'] == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(),
                                                 lr=best_params['learning_rate'],
                                                 weight_decay=best_params['weight_decay'])
                elif best_params['optimizer'] == 'adamW':
                    optimizer = torch.optim.AdamW(model.parameters(),
                                                  lr=best_params['learning_rate'],
                                                  weight_decay=best_params['weight_decay'])
                else:
                    raise ValueError(f"Unknown optimizer: {best_params['optimizer']}")
        elif mil_type == 'graph':
            if config.get('best_params_graph-mil', {}).get('use', False):
                # Use best hyperparameters from config
                best_params = config['best_params_graph-mil']
                graph_type = best_params.get('graph_type', 'grid')
                k_neighbors = best_params.get('k_neighbors', 8)
                connect_diagonals = best_params.get('connect_diagonals', False)
                
                model = GraphMIL(input_dim=input_dim,
                                 gnn_type=best_params.get('gnn_type', 'gcn'),
                                 gnn_hidden=best_params.get('gnn_hidden', 128),
                                 gnn_layers=best_params.get('gnn_layers', 2),
                                 gnn_dropout=best_params.get('gnn_dropout', 0.5),
                                 gnn_heads=best_params.get('gnn_heads', 4),
                                 gnn_concat=best_params.get('gnn_concat', True),
                                 att_dim=best_params.get('att_dim', 64),
                                 att_heads=best_params.get('att_heads', 4),
                                 pool_dropout=best_params.get('pool_dropout', 0.3),
                                 classifier_dim=best_params.get('classifier_dim', 128),
                                 classifier_light=best_params.get('classifier_light', False),
                                 num_classes=7,
                                 use_residual=best_params.get('use_residual', True),
                                 use_layer_norm=best_params.get('use_layer_norm', True)).to(device)
                
                opt_name = best_params.get('optimizer', 'adam')
                lr = best_params.get('lr', 1e-4)
                weight_decay = best_params.get('weight_decay', 1e-5)
                
                if opt_name == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                elif opt_name == 'adamw':
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                elif opt_name == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
                else:
                    raise ValueError(f"Unknown optimizer: {opt_name}")
            else:
                # Use default hyperparameters
                graph_type = 'grid'
                k_neighbors = 8
                connect_diagonals = False
                
                model = GraphMIL(input_dim=input_dim,
                                 gnn_type='gcn',
                                 gnn_hidden=128,
                                 gnn_layers=2,
                                 gnn_dropout=0.75,
                                 gnn_heads=4,
                                 gnn_concat=True,
                                 att_dim=64,
                                 att_heads=4,
                                 pool_dropout=0.0,
                                 classifier_dim=64,
                                 classifier_light=False,
                                 num_classes=7,
                                 use_residual=True,
                                 use_layer_norm=True).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            
        else:
            raise ValueError(f"Unsupported mil_type: {mil_type}; choose 'classic' or 'graph'")
        ## ------------------------------------------------------------------------------------------------------------- ##
        patience = config.get('training_plan', {}).get('parameters', {}).get('patience', 8)
        epochs_no_improve = 0
        best_val_bacc = -np.inf
        best_state_bacc = None
        best_val_loss = float('inf')
        best_state_loss = None

        # add option to freeze gnn layers and only train attention + classifier
        freeze_gnn = config.get('best_params_graph-mil', {}).get('freeze_gnn', False)
        freeze_gnn = False # override to never freeze for this experiment   # TEMP OVERRIDE
        if mil_type == 'graph' and freeze_gnn:
            print("    Freezing GNN layers, only training attention and classifier")
            for name, param in model.named_parameters():
                # Freeze: gnn_layers, layer_norms, input_proj, gnn_dropout
                # Keep unfrozen: attention_layers, classifier
                if any(x in name for x in ['gnn_layers', 'layer_norms', 'input_proj']):
                    param.requires_grad = False
                    # print what was frozen
                    print(f"      Frozen parameter: {name}")

        for epoch in range(1, num_epochs + 1):
            model.train()
            for x, y_batch in train_loader:
                x = x[0].to(device)
                y_long = y_batch.to(device).long()
                optimizer.zero_grad()
                if mil_type == 'graph':
                    # build graph for current bag according to graph_type
                    adj_norm, adj_mask, edge_index, edge_weight = build_graph(
                        x, graph_type=graph_type, k=k_neighbors,
                        connect_diagonals=connect_diagonals, device=device)
                    probs, att = model(x, adj=adj_norm, adj_mask=adj_mask, 
                                     edge_index=edge_index, edge_weight=edge_weight)
                else:
                    probs, att = model(x)
                loss = criterion(torch.log(probs + 1e-9).unsqueeze(0), y_long)
                loss.backward()
                optimizer.step()

            model.eval()
            y_true = []
            y_score = []
            val_loss_sum = 0.0
            with torch.no_grad():
                for x, y_batch in val_loader:
                    x = x[0].to(device)
                    y_long = y_batch.to(device).long()
                    if mil_type == 'graph':
                        adj_norm, adj_mask, edge_index, edge_weight = build_graph(
                            x, graph_type=graph_type, k=k_neighbors,
                            connect_diagonals=connect_diagonals, device=device)
                        probs, att = model(x, adj=adj_norm, adj_mask=adj_mask,
                                         edge_index=edge_index, edge_weight=edge_weight)
                    else:
                        probs, att = model(x)
                    val_loss_sum += criterion(torch.log(probs + 1e-9).unsqueeze(0), y_long).item()
                    y_true.append(int(y_long.item()))
                    y_score.append(probs.cpu().numpy())

            if len(y_true) == 0:
                print("    No validation samples for this fold, skipping")
                break

            y_true = np.array(y_true)
            y_score = np.vstack(y_score)
            y_pred = np.argmax(y_score, axis=1)
            try:
                _ = roc_auc_score(y_true, y_score, multi_class='ovr')
            except Exception:
                pass
            val_bacc = balanced_accuracy_score(y_true, y_pred)
            val_loss = val_loss_sum / len(val_loader)

            if val_bacc > best_val_bacc + 1e-6:
                best_val_bacc = val_bacc
                best_state_bacc = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if val_loss < best_val_loss - 1e-6:
                epochs_no_improve = 0
                best_val_loss = val_loss
                best_state_loss = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            print(f"    Epoch {epoch:03d}: Val BAcc: {val_bacc:.4f} (best: {best_val_bacc:.4f})  | Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f} ) | Epochs no improve: {epochs_no_improve}/{patience}")
            
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        # ---------------- Test evaluations for both checkpoints ----------------
        
        metrics_best_bacc = _evaluate_model(best_state_bacc)
        metrics_best_loss = _evaluate_model(best_state_loss if best_state_loss is not None else best_state_bacc)
        fold_test_metrics_bacc.append(metrics_best_bacc)
        fold_test_metrics_loss.append(metrics_best_loss)

    # print model, optimizer, graph params
    print("Model and training parameters:")
    print(f"  MIL type: {mil_type}")
    if mil_type == 'graph':
        print(f"  Graph type: {graph_type}, k_neighbors: {k_neighbors}, connect_diagonals: {connect_diagonals}")
    print(f"  Num epochs: {num_epochs}, Patience: {patience}")
    print(f"  Criterion: CrossEntropyLoss")
    print(f"  Optimizer: {optimizer.__class__.__name__}, LR: {optimizer.param_groups[0]['lr']}, Weight Decay: {optimizer.param_groups[0]['weight_decay']}")
    print(f"Model Graph-MIL with hyperparameters:", model)
        
    metrics_keys = ['micro','macro_p','macro_r','macro_f1','weighted_p','weighted_r','weighted_f1']
    agg_mean_bacc, agg_std_bacc = {}, {}
    agg_mean_loss, agg_std_loss = {}, {}
    for k in metrics_keys:
        vals_bacc = np.array([m[k] for m in fold_test_metrics_bacc], dtype=np.float64)
        vals_loss = np.array([m[k] for m in fold_test_metrics_loss], dtype=np.float64)

        if np.all(np.isnan(vals_bacc)):
            agg_mean_bacc[k] = np.nan
            agg_std_bacc[k] = np.nan
        else:
            agg_mean_bacc[k] = float(np.nanmean(vals_bacc))
            agg_std_bacc[k] = float(np.nanstd(vals_bacc, ddof=0))

        if np.all(np.isnan(vals_loss)):
            agg_mean_loss[k] = np.nan
            agg_std_loss[k] = np.nan
        else:
            agg_mean_loss[k] = float(np.nanmean(vals_loss))
            agg_std_loss[k] = float(np.nanstd(vals_loss, ddof=0))


    row_bacc = {
        'id': row['sys/id'],
        'checkpoint_type': 'best_bacc',
        'micro_accuracy': agg_mean_bacc['micro'],
        'macro_precision': agg_mean_bacc['macro_p'],
        'macro_recall': agg_mean_bacc['macro_r'],
        'macro_f1': agg_mean_bacc['macro_f1'],
        'weighted_precision': agg_mean_bacc['weighted_p'],
        'weighted_recall': agg_mean_bacc['weighted_r'],
        'weighted_f1': agg_mean_bacc['weighted_f1'],
        'micro_accuracy_std': agg_std_bacc['micro'],
        'macro_precision_std': agg_std_bacc['macro_p'],
        'macro_recall_std': agg_std_bacc['macro_r'],
        'macro_f1_std': agg_std_bacc['macro_f1'],
        'weighted_precision_std': agg_std_bacc['weighted_p'],
        'weighted_recall_std': agg_std_bacc['weighted_r'],
        'weighted_f1_std': agg_std_bacc['weighted_f1'],
    }

    row_loss = {
        'id': row['sys/id'],
        'checkpoint_type': 'best_loss',
        'micro_accuracy': agg_mean_loss['micro'],
        'macro_precision': agg_mean_loss['macro_p'],
        'macro_recall': agg_mean_loss['macro_r'],
        'macro_f1': agg_mean_loss['macro_f1'],
        'weighted_precision': agg_mean_loss['weighted_p'],
        'weighted_recall': agg_mean_loss['weighted_r'],
        'weighted_f1': agg_mean_loss['weighted_f1'],
        'micro_accuracy_std': agg_std_loss['micro'],
        'macro_precision_std': agg_std_loss['macro_p'],
        'macro_recall_std': agg_std_loss['macro_r'],
        'macro_f1_std': agg_std_loss['macro_f1'],
        'weighted_precision_std': agg_std_loss['weighted_p'],
        'weighted_recall_std': agg_std_loss['weighted_r'],
        'weighted_f1_std': agg_std_loss['weighted_f1'],
    }

    results_rows.append(row_bacc)
    results_rows.append(row_loss)

    # Persist accumulated results after each model to avoid losing progress mid-run
    _persist_results(pd.DataFrame(results_rows))

print(f"\nSaved runs results to {out_csv}")
# %%