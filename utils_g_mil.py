import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from ray import tune
from torch_geometric import nn as pyg_nn    # cant import torch.sparse, torch.cluster etc. TODO fix environment

class AttentionMIL(nn.Module):
    def __init__(self, input_dim=76, hidden_dim=128, att_dim=64, dropout=0.5, num_classes=7):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, att_dim),
            nn.Tanh(),
            nn.Linear(att_dim, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.feature_extractor(x)                # [N, hidden_dim]
        a = torch.softmax(self.attention(h), dim=0)  # [N, 1]
        z = torch.sum(a * h, dim=0)                  # [hidden_dim]
        logits = self.classifier(z)                  # [num_classes]
        probs = torch.softmax(logits, dim=0)
        return probs, a

class PatientDataset(Dataset):
    def __init__(self, patient_features, patient_labels):
        self.features = patient_features
        self.labels = patient_labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(train_feats, train_labels, test_feats, test_labels, num_workers=0, pin_memory=False):
    train_dataset = PatientDataset(train_feats, train_labels)
    val_dataset = PatientDataset(test_feats, test_labels)
    patient_labels = np.array(train_labels)
    patient_class_counts = Counter(patient_labels)
    weights = np.array([1.0 / patient_class_counts[int(lbl)] for lbl in patient_labels], dtype=np.float64)
    sample_weights = torch.from_numpy(weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, drop_last=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=max(0, num_workers//2), pin_memory=pin_memory)
    return train_loader, val_loader

def train_mil(config, data=None, seed=42, num_classes=7, device_str=None, patience=8, max_epochs=50):
    set_seed(seed)
    torch_threads = int(data.get('torch_threads', 1))
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(max(1, torch_threads//2))
    device = torch.device(device_str if device_str is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # If we're on CUDA and a per-process GPU memory fraction is provided,
    # apply it so multiple trials can safely share a single physical GPU.
    if device.type == 'cuda':
        frac_str = os.environ.get('PER_PROC_GPU_MEM_FRACTION')
        if frac_str is not None:
            try:
                frac = float(frac_str)
                frac = max(0.01, min(1.0, frac))
                try:
                    torch.cuda.set_per_process_memory_fraction(frac, device=device)
                except Exception:
                    # Older PyTorch versions may not have set_per_process_memory_fraction
                    print(f"Warning: could not set per-process GPU mem fraction (PyTorch may be too old)")
            except Exception:
                print(f"Warning: invalid PER_PROC_GPU_MEM_FRACTION='{frac_str}'")

    train_feats = [torch.tensor(arr, dtype=torch.float32) for arr in data['train_feats']]
    train_labels = [int(l) for l in data['train_labels']]
    test_feats = [torch.tensor(arr, dtype=torch.float32) for arr in data.get('test_feats', [])]
    test_labels = [int(l) for l in data.get('test_labels', [])]

    train_loader, val_loader = get_data_loaders(train_feats, train_labels, test_feats, test_labels,
                                                num_workers=int(data.get('num_workers', 0)),
                                                pin_memory=bool(data.get('pin_memory', False)))

    input_dim = train_feats[0].shape[1] if len(train_feats) > 0 else data.get('input_dim', 76)

    model = AttentionMIL(input_dim=input_dim,
                         hidden_dim=int(config['hidden_dim']),
                         att_dim=int(config['att_dim']),
                         dropout=float(config['dropout']),
                         num_classes=num_classes).to(device)

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=config.get('weight_decay', 1e-5))
    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=config.get('weight_decay', 1e-5))
    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config['lr']), momentum=0.9, weight_decay=config.get('weight_decay', 1e-5))
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    criterion = nn.CrossEntropyLoss()
    best_val_bacc = -np.inf
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x[0].to(device)
            y_long = y.to(device).long()
            optimizer.zero_grad()
            probs, _ = model(x)
            loss = criterion(torch.log(probs.unsqueeze(0) + 1e-9), y_long)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x[0].to(device)
                y_long = y.to(device).long()
                probs, _ = model(x)
                y_true.append(int(y_long.item()))
                y_score.append(probs.cpu().numpy())

        if len(y_true) == 0:
            tune.report({'val_bacc': float('nan')})
            return

        y_true = np.array(y_true)
        y_score = np.vstack(y_score)
        y_pred = np.argmax(y_score, axis=1)
        try:
            val_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception:
            val_auc = float('nan')
        val_acc = accuracy_score(y_true, y_pred)
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        tune.report({
            'val_bacc': val_bacc, 'val_acc': val_acc, 'val_auc': val_auc,
            'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1,
            'weighted_precision': weighted_p, 'weighted_recall': weighted_r, 'weighted_f1': weighted_f1
        })

        if val_bacc > best_val_bacc + 1e-6:
            best_val_bacc = val_bacc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break


######################### Graph-MIL components for GNN tuning #########################
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.conv = pyg_nn.GINConv(self.nn, eps=eps, train_eps=True)
    
    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index)


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='mean', normalize=True):
        super().__init__()

        self.conv = pyg_nn.SAGEConv(in_dim, out_dim, aggr=aggr, normalize=normalize)
    
    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index)


class TransformerConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.0):
        super().__init__()

        self.conv = pyg_nn.TransformerConv(
            in_dim, out_dim, heads=heads, concat=concat, 
            dropout=dropout, edge_dim=None, beta=True
        )
        self.out_dim = out_dim * heads if concat else out_dim
    
    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index)


class GraphMIL(nn.Module):
    """
    Enhanced Graph-based Multiple Instance Learning model.
    
    Recommended configurations for 196 patches  x 768 features:
    - gnn_type: 'gat', 'gcn', 'gin' (most expressive), 
                'graphsage' (most efficient), 'transformer' (attention-based)
    - gnn_hidden: 256-512 (balance between capacity and efficiency)
    - gnn_layers: 2-3 (deeper can oversmooth)
    - use_residual: True (helps training deeper networks)
    - use_layer_norm: True (stabilizes training)
    """
    def __init__(self, input_dim=768, gnn_type='gat', gnn_hidden=256, 
                 gnn_layers=2, gnn_dropout=0.1, k_neighbors=8,
                 att_dim=128, att_heads=4, pool_dropout=0.2, 
                 classifier_dim=128, classifier_light=False, num_classes=7,
                 use_residual=True, use_layer_norm=True):
        super().__init__()
        
        self.gnn_type = gnn_type.lower()
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.k_neighbors = k_neighbors
        self.classifier_light = classifier_light
        
        # Input projection if using residual connections
        if use_residual and input_dim != gnn_hidden:
            self.input_proj = nn.Linear(input_dim, gnn_hidden)
        else:
            self.input_proj = None
        
        # Build GNN layers
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        in_dim = input_dim if self.input_proj is None else gnn_hidden
        
        for i in range(gnn_layers):
            out_dim = gnn_hidden
            
            if self.gnn_type == 'gin':
                layer = GINLayer(in_dim, out_dim, eps=0.0)
            elif self.gnn_type == 'graphsage':
                layer = GraphSAGELayer(in_dim, out_dim, aggr='mean', normalize=True)
            elif self.gnn_type == 'transformer':
                layer = TransformerConvLayer(in_dim, out_dim, heads=4, 
                                            concat=True, dropout=gnn_dropout)
                out_dim = layer.out_dim  # Adjust for multi-head concat
            elif self.gnn_type == 'gat':
                layer = pyg_nn.GATConv(in_dim, out_dim, heads=4, concat=True, dropout=gnn_dropout)
                out_dim = out_dim * 4
            elif self.gnn_type == 'gcn':
                layer = pyg_nn.GCNConv(in_dim, out_dim)
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
        

        # TODO simpler classifier like in traditional MIL?
        # TODO should dropout be applied for gin and graphsage too?
        # Classifier with residual connection
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
    
    # Note: edge/index construction moved out of the model. See module helpers
    # `build_knn_edge_index` and `build_graph` below. The model expects either
    # a dense `adj` (normalized) or an `edge_index` to be provided by the caller.
    
    def forward(self, x, adj=None, adj_mask=None, edge_index=None, edge_weight=None):
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

        # If caller provided a dense adjacency, convert to integer edge_index
        if edge_index is None and adj is not None:
            try:
                mask = (adj > 0)
                edge_index = mask.nonzero(as_tuple=False).t().to(x.device).long()
                try:
                    edge_weight = adj.to(x.device)[mask]
                except Exception:
                    edge_weight = edge_weight
            except Exception:
                edge_index = None

        # Validation / coercion: ensure edge_index is a long tensor of shape [2, E]
        if edge_index is not None:
            if not torch.is_tensor(edge_index):
                edge_index = torch.tensor(edge_index, device=x.device)
            edge_index = edge_index.to(x.device).long()
            # If user accidentally passed shape [E,2], transpose to [2,E]
            if edge_index.dim() == 2 and edge_index.shape[0] == 2:
                edge_index = edge_index.contiguous()
            elif edge_index.dim() == 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.t().contiguous()
            else:
                raise ValueError(f"edge_index must be shape [2, E]; got {tuple(edge_index.shape)}")

        if edge_weight is not None:
            if not torch.is_tensor(edge_weight):
                edge_weight = torch.tensor(edge_weight, device=x.device)
            edge_weight = edge_weight.to(x.device).flatten()

        # GNN layers with residual connections
        for i, layer in enumerate(self.gnn_layers):
            h_prev = h

            h = layer(h, edge_index, edge_weight)
            
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


def build_grid_adj(num_nodes, connect_diagonals=False, device=None):
    s = int(np.sqrt(num_nodes))
    if s * s != num_nodes:
        raise ValueError('num_nodes must be a perfect square to build grid adjacency')
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for r in range(s):
        for c in range(s):
            i = r * s + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < s and 0 <= cc < s:
                    j = rr * s + cc
                    adj[i, j] = 1.0
            if connect_diagonals:
                for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < s and 0 <= cc < s:
                        j = rr * s + cc
                        adj[i, j] = 1.0
    adj = adj + torch.eye(num_nodes)
    deg = adj.sum(dim=1)
    deg_inv = torch.diag(1.0 / deg)
    adj_norm = deg_inv @ adj
    if device is not None:
        adj_norm = adj_norm.to(device)
    return adj_norm, (adj > 0).float()


# cache for grid adjacency to avoid rebuilding identical adj matrices repeatedly
_GRID_ADJ_CACHE = {}


def build_knn_edge_index(x, k=8, device=None):
    """Build a k-NN edge_index for PyG from node features `x`.

    Returns a tensor of shape [2, E] suitable for passing to a PyG conv.
    """
    try:
        # pyg_nn.knn_graph returns edge_index on the same device as x
        edge_index = pyg_nn.knn_graph(x, k=k, batch=None, loop=False)
        return edge_index.long().to(x.device)
    except Exception:
        # Fallback: fully-connected edge_index
        num_nodes = x.size(0)
        idx = torch.arange(num_nodes, device=device, dtype=torch.long)
        ii, jj = torch.meshgrid(idx, idx, indexing='ij')
        edge_index = torch.vstack([ii.reshape(-1), jj.reshape(-1)]).long().to(device)
        return edge_index


def build_graph(x, graph_type='grid', k=None, connect_diagonals=False, device=None):
    """Return (adj_norm, adj_mask, edge_index, edge_weight) for given node features `x`.

    - For `graph_type='grid'`: returns a normalized dense adjacency (`adj_norm`),
      binary `adj_mask`, and also `edge_index`/`edge_weight` derived from the mask.
    - For `graph_type='knn'`: returns `edge_index` built with `k` and `edge_weight=None`.
    - For unknown `graph_type`, returns a fully-connected normalized adjacency and
      corresponding edge_index/edge_weight.
    """
    num_nodes = x.size(0)
    if graph_type == 'grid':
        key = (num_nodes, bool(connect_diagonals), str(device))
        if key in _GRID_ADJ_CACHE:
            adj_norm, adj_mask = _GRID_ADJ_CACHE[key]
        else:
            adj_norm, adj_mask = build_grid_adj(num_nodes, connect_diagonals=connect_diagonals, device=device)
            _GRID_ADJ_CACHE[key] = (adj_norm, adj_mask)

        # derive edge_index and edge_weight from adj_mask/adj_norm
        mask = adj_mask.to(x.device).bool()
        edge_index = mask.nonzero(as_tuple=False).t().to(x.device).long()
        # gather edge weights in the same ordering as edge_index (if possible)
        try:
            edge_weight = adj_norm.to(x.device)[mask]
        except Exception:
            edge_weight = None
        return adj_norm, adj_mask, edge_index, edge_weight

    elif graph_type == 'knn':
        k_val = 8 if k is None else int(k)
        edge_index = build_knn_edge_index(x, k=k_val, device=device)
        return None, None, edge_index, None

    else:
        raise ValueError(f"Unsupported graph_type='{graph_type}'. Supported types: 'grid', 'knn'.")


def train_graph_mil(config, data=None, seed=42, num_classes=7, device_str=None, patience=8, max_epochs=50):
    """
    Train wrapper for graph-based MIL models. Accepts the same style `data` dict as `train_mil`.
    Expected config keys (examples):
      - gnn_type: 'gcn' | 'gat'
      - gnn_hidden: int
      - gnn_layers: int
      - gnn_dropout: float
      - att_dim: int
      - classifier_dim: int
      - lr, optimizer, weight_decay
    """
    set_seed(seed)
    torch_threads = int(data.get('torch_threads', 1))
    torch.set_num_threads(torch_threads)
    torch.set_num_interop_threads(max(1, torch_threads//2))
    device = torch.device(device_str if device_str is not None else ('cuda' if torch.cuda.is_available() else 'cpu'))

    # Apply per-process GPU memory fraction if requested (see tune wrapper)
    if device.type == 'cuda':
        frac_str = os.environ.get('PER_PROC_GPU_MEM_FRACTION')
        if frac_str is not None:
            try:
                frac = float(frac_str)
                frac = max(0.01, min(1.0, frac))
                try:
                    torch.cuda.set_per_process_memory_fraction(frac, device=device)
                except Exception:
                    print(f"Warning: could not set per-process GPU mem fraction (PyTorch may be too old)")
            except Exception:
                print(f"Warning: invalid PER_PROC_GPU_MEM_FRACTION='{frac_str}'")

    train_feats = [torch.tensor(arr, dtype=torch.float32) for arr in data['train_feats']]
    train_labels = [int(l) for l in data['train_labels']]
    test_feats = [torch.tensor(arr, dtype=torch.float32) for arr in data.get('test_feats', [])]
    test_labels = [int(l) for l in data.get('test_labels', [])]

    train_loader, val_loader = get_data_loaders(train_feats, train_labels, test_feats, test_labels,
                                                num_workers=int(data.get('num_workers', 0)),
                                                pin_memory=bool(data.get('pin_memory', False)))

    input_dim = train_feats[0].shape[1] if len(train_feats) > 0 else data.get('input_dim', 76)

    model = GraphMIL(input_dim=input_dim,
                     gnn_type=config.get('gnn_type', 'gcn'),
                     gnn_hidden=int(config.get('gnn_hidden', 128)),
                     gnn_layers=int(config.get('gnn_layers', 2)),
                     gnn_dropout=float(config.get('gnn_dropout', 0.0)),
                     att_dim=int(config.get('att_dim', 64)),
                     pool_dropout=float(config.get('pool_dropout', 0.0)),
                     classifier_dim=int(config.get('classifier_dim', 64)),
                     classifier_light=bool(config.get('classifier_light', False)),
                     num_classes=num_classes).to(device)

    opt_name = config.get('optimizer', 'adam')
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get('lr', 1e-4)), weight_decay=float(config.get('weight_decay', 1e-5)))
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('lr', 1e-4)), weight_decay=float(config.get('weight_decay', 1e-5)))
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get('lr', 1e-3)), momentum=0.9, weight_decay=float(config.get('weight_decay', 1e-5)))
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    criterion = nn.CrossEntropyLoss()
    best_val_bacc = -np.inf
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x[0].to(device)  # x: [N_nodes, feat]
            y_long = y.to(device).long()
            num_nodes = x.shape[0]

            # Build graph according to configuration. `build_graph` returns a tuple
            # (adj_norm, adj_mask, edge_index). For grid graphs we pass adj_norm,
            # for knn graphs we pass edge_index.
            graph_type = config.get('graph_type', 'grid')
            connect_diagonals = bool(config.get('connect_diagonals', False))
            k_neighbors = config.get('k_neighbors', None)
            adj_norm, adj_mask, edge_index, edge_weight = build_graph(
                x, graph_type=graph_type, k=k_neighbors,
                connect_diagonals=connect_diagonals, device=device)

            optimizer.zero_grad()
            probs, _ = model(x, adj=adj_norm, adj_mask=adj_mask, edge_index=edge_index, edge_weight=edge_weight)
            loss = criterion(torch.log(probs.unsqueeze(0) + 1e-9), y_long)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x[0].to(device)
                y_long = y.to(device).long()
                num_nodes = x.shape[0]
                graph_type = config.get('graph_type', 'grid')
                connect_diagonals = bool(config.get('connect_diagonals', False))
                k_neighbors = config.get('k_neighbors', None)
                adj_norm, adj_mask, edge_index, edge_weight = build_graph(
                    x, graph_type=graph_type, k=k_neighbors,
                    connect_diagonals=connect_diagonals, device=device)
                probs, _ = model(x, adj=adj_norm, adj_mask=adj_mask, edge_index=edge_index, edge_weight=edge_weight)
                y_true.append(int(y_long.item()))
                y_score.append(probs.cpu().numpy())

        if len(y_true) == 0:
            tune.report({'val_bacc': float('nan')})
            return

        y_true = np.array(y_true)
        y_score = np.vstack(y_score)
        y_pred = np.argmax(y_score, axis=1)
        try:
            val_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        except Exception:
            val_auc = float('nan')
        val_acc = accuracy_score(y_true, y_pred)
        val_bacc = balanced_accuracy_score(y_true, y_pred)
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        tune.report({
            'val_bacc': val_bacc, 'val_acc': val_acc, 'val_auc': val_auc,
            'macro_precision': macro_p, 'macro_recall': macro_r, 'macro_f1': macro_f1,
            'weighted_precision': weighted_p, 'weighted_recall': weighted_r, 'weighted_f1': weighted_f1
        })

        if val_bacc > best_val_bacc + 1e-6:
            best_val_bacc = val_bacc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

