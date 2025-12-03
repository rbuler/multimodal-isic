# %%
import traceback
import torch
import numpy as np

LAYERS_TO_TEST = [
    'AttentionMIL',
    # 'EdgeConvLayer',
    'GINLayer',
    'GraphSAGELayer',
    'TransformerConvLayer',
    'GraphMIL'
]

results = {}

try:
    import utils_g_mil
except Exception as e:
    print('Failed to import utils_g_mil.py:', e)
    traceback.print_exc()
    raise SystemExit(1)

print('utils_g_mil.py imported, testing layers...')
# small synthetic graph: N nodes, F features
N = 16
F = 8
x = torch.randn(N, F)
# simple fully connected edge_index (no self loops)
rows = []
cols = []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        rows.append(i)
        cols.append(j)
edge_index = torch.tensor([rows, cols], dtype=torch.long)

# For models that pool per-graph, provide a batch vector of zeros (single graph)
batch = torch.zeros(N, dtype=torch.long)

for name in LAYERS_TO_TEST:
    try:
        cls = getattr(utils_g_mil, name)
    except AttributeError:
        results[name] = ('missing', 'Not defined in temp.py')
        continue

    try:
        if name == 'AttentionMIL':
            m = cls(input_dim=F, hidden_dim=16, att_dim=8, dropout=0.1, num_classes=3)
            out, att = m(x)
            results[name] = ('ok', f'out_shape={out.shape}, att_shape={att.shape}')
        elif name == 'EdgeConvLayer':
            m = cls(in_dim=F, out_dim=16, k=4)
            out = m(x, edge_index=None)  # EdgeConv may build kNN internally
            results[name] = ('ok', f'out_shape={out.shape}')
        elif name == 'GINLayer':
            m = cls(in_dim=F, out_dim=16)
            out = m(x, edge_index)
            results[name] = ('ok', f'out_shape={out.shape}')
        elif name == 'GraphSAGELayer':
            m = cls(in_dim=F, out_dim=16)
            out = m(x, edge_index)
            results[name] = ('ok', f'out_shape={out.shape}')
        elif name == 'TransformerConvLayer':
            m = cls(in_dim=F, out_dim=8, heads=2)
            out = m(x, edge_index)
            results[name] = ('ok', f'out_shape={out.shape}')
        elif name == 'GraphMIL':
            m = cls(input_dim=F, gnn_type='gcn', gnn_hidden=16, gnn_layers=2, gnn_dropout=0.0, att_dim=8, att_heads=2, pool_dropout=0.0, classifier_dim=16, num_classes=3)
            # GraphMIL.forward expects x [N,F], optional adj or edge_index
            out_probs, att = m(x, edge_index=edge_index)
            results[name] = ('ok', f'probs_shape={out_probs.shape}, att_shape={att.shape}')
        else:
            results[name] = ('skip', 'No test implemented')
    except Exception as e:
        results[name] = ('error', f'{type(e).__name__}: {e}\n' + traceback.format_exc())

print('\nTest results:')
for k, v in results.items():
    status, info = v
    print(f' - {k}: {status} -- {info}')

# exit code 0 if at least one layer succeeded, else 2
if any(v[0] == 'ok' for v in results.values()):
    print('\nAt least one layer ran successfully.')
    raise SystemExit(0)
else:
    print('\nNo layers ran successfully; check imports and torch_geometric availability.')
    raise SystemExit(2)


# %%
