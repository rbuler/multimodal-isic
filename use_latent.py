# %% IMPORTS
import os
import yaml
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


parser = get_args_parser('config.yml')
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
# %%
experiment_ids = list(range(798, 814)) + list(range(726, 732))
experiment_ids = [805]


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

SEED = config['seed']

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

    # --- Prepare patient-level features/labels ---
    train_patient_features = [
        torch.tensor(np.vstack(g['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_level_latents_df_train.groupby('patient_id')
    ]
    train_patient_labels = [
        int(g['target'].mode().iat[0])
        for pid, g in patch_level_latents_df_train.groupby('patient_id')
    ]

    test_patient_features = [
        torch.tensor(np.vstack(g['patch_latent'].values), dtype=torch.float32)
        for pid, g in patch_level_latents_df_test.groupby('patient_id')
    ]
    test_patient_labels = [
        int(g['target'].mode().iat[0])
        for pid, g in patch_level_latents_df_test.groupby('patient_id')
    ]

    # --- 5-fold cross-validation on patient-level training data ---

    SPLITS = 5
    skf = StratifiedKFold(n_splits=SPLITS, shuffle=True, random_state=SEED)
    y = np.array(train_patient_labels)

    fold_test_metrics = []

    test_dataset = PatientDataset(test_patient_features, test_patient_labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    num_epochs = 100
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
        graph_type = 'grid'  # 'grid' or 'knn'
        k_neighbors = 8  # only used if graph_type='knn'
        connect_diagonals = False  # only used if graph_type='grid'

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
            model = GraphMIL(input_dim=input_dim,
                             gnn_type='gcn',
                             gnn_hidden=128,
                             gnn_layers=2,
                             gnn_dropout=0.75,
                             att_dim=64,
                             pool_dropout=0.0,
                             classifier_dim=64,
                             num_classes=7).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
            
        else:
            raise ValueError(f"Unsupported mil_type: {mil_type}; choose 'classic' or 'graph'")
        ## ------------------------------------------------------------------------------------------------------------- ##

        best_val_bacc = -np.inf
        best_state = None

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

            if val_bacc > best_val_bacc + 1e-6:
                best_val_bacc = val_bacc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

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
            print("  No test samples available, recording NaNs for this fold")
            fold_test_metrics.append({k: np.nan for k in ['micro','macro_p','macro_r','macro_f1','weighted_p','weighted_r','weighted_f1']})
            continue

        y_true = np.array(y_true)
        y_score = np.vstack(y_score)
        y_pred = np.argmax(y_score, axis=1)

        micro_acc = accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        fold_test_metrics.append({'micro': micro_acc,
                                  'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
                                  'weighted_p': weighted_p, 'weighted_r': weighted_r, 'weighted_f1': weighted_f1})

    metrics_keys = ['micro','macro_p','macro_r','macro_f1','weighted_p','weighted_r','weighted_f1']
    agg_mean = {}
    agg_std = {}
    for k in metrics_keys:
        vals = np.array([m[k] for m in fold_test_metrics], dtype=np.float64)
        if np.all(np.isnan(vals)):
            agg_mean[k] = np.nan
            agg_std[k] = np.nan
        else:
            agg_mean[k] = float(np.nanmean(vals))
            agg_std[k] = float(np.nanstd(vals, ddof=0))

    runs_df.loc[idx, 'micro_accuracy'] = agg_mean['micro']
    runs_df.loc[idx, 'macro_precision'] = agg_mean['macro_p']
    runs_df.loc[idx, 'macro_recall'] = agg_mean['macro_r']
    runs_df.loc[idx, 'macro_f1'] = agg_mean['macro_f1']
    runs_df.loc[idx, 'weighted_precision'] = agg_mean['weighted_p']
    runs_df.loc[idx, 'weighted_recall'] = agg_mean['weighted_r']
    runs_df.loc[idx, 'weighted_f1'] = agg_mean['weighted_f1']

    runs_df.loc[idx, 'micro_accuracy_std'] = agg_std['micro']
    runs_df.loc[idx, 'macro_precision_std'] = agg_std['macro_p']
    runs_df.loc[idx, 'macro_recall_std'] = agg_std['macro_r']
    runs_df.loc[idx, 'macro_f1_std'] = agg_std['macro_f1']
    runs_df.loc[idx, 'weighted_precision_std'] = agg_std['weighted_p']
    runs_df.loc[idx, 'weighted_recall_std'] = agg_std['weighted_r']
    runs_df.loc[idx, 'weighted_f1_std'] = agg_std['weighted_f1']

output_dir = os.path.join(os.getcwd(), "mil_results")
os.makedirs(output_dir, exist_ok=True)

date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
out_pickle = os.path.join(output_dir, f"runs_df_mil_results_{date_str}.pkl")
out_csv = os.path.join(output_dir, f"runs_df_mil_results_{date_str}.csv")

runs_df.to_pickle(out_pickle)
runs_df.to_csv(out_csv, index=False)

print(f"\nSaved runs results to {output_dir}: {out_pickle}, {out_csv}")
# %%