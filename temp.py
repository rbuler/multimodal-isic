import re
import neptune
import pandas as pd
from neptune.exceptions import NeptuneException
# %%
def fetch_experiments(
    tags: list = None,
    group_tags: list = None,
    experiment_ids: list = None,
):
    try:
        project = neptune.init_project(project="ProjektMMG/multimodal-isic",)
        runs_table = project.fetch_runs_table(sort_by='sys/creation_time')
        if tags:
            runs_table = runs_table[runs_table["tags"].apply(
                lambda run_tags: all(tag in run_tags for tag in tags)
            )]
        if group_tags:
            runs_table = runs_table[runs_table["groups"].apply(
                lambda run_groups: any(gtag in run_groups for gtag in group_tags)
            )]
        if experiment_ids:
            runs_table = runs_table[runs_table["sys/id"].isin(experiment_ids)]

        return runs_table

    except NeptuneException as e:
        print("Error querying Neptune:", str(e))
        return None


def parse_classification_report(report: str):
    metrics = {
        'accuracy': None,
        'macro_precision': None,
        'macro_recall': None,
        'macro_f1': None,
        'weighted_precision': None,
        'weighted_recall': None,
        'weighted_f1': None
    }

    for line in report.strip().split('\n'):
        line = line.strip()

        # Accuracy line
        if line.startswith("accuracy"):
            match = re.search(r'accuracy\s+([0-9.]+)', line)
            if match:
                metrics['accuracy'] = float(match.group(1))

        # Macro avg line
        elif line.startswith("macro avg"):
            match = re.search(r'macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', line)
            if match:
                metrics['macro_precision'] = float(match.group(1))
                metrics['macro_recall'] = float(match.group(2))
                metrics['macro_f1'] = float(match.group(3))

        # Weighted avg line
        elif line.startswith("weighted avg"):
            match = re.search(r'weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', line)
            if match:
                metrics['weighted_precision'] = float(match.group(1))
                metrics['weighted_recall'] = float(match.group(2))
                metrics['weighted_f1'] = float(match.group(3))

    return metrics




    
results_df = fetch_experiments(
    # tags=["baseline", "v1.0"],
    # group_tags=["resnet", "experiments"],
    # experiment_ids=[],
)
results_df = results_df.to_pandas()
results_df = results_df.dropna(subset=['test/classification_report'])
results_df['parsed_report'] = results_df['test/classification_report'].apply(parse_classification_report)
# assign df as id col and parser rep
df = results_df[['sys/id', 'sys/group_tags', 'parsed_report']].copy()

metrics_df = df['parsed_report'].apply(pd.Series)

# %%
df_expanded = df.drop(columns=['parsed_report']).join(metrics_df)

cols = ['sys/id', 'sys/group_tags', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
        'weighted_precision', 'weighted_recall', 'weighted_f1']
df_expanded = df_expanded[cols]

# id is str: text and number, like "MUL-1"
# i want to drop from df where number is less thant 275
df_expanded['sys/id'] = df_expanded['sys/id'].astype(str)
df_expanded['number'] = df_expanded['sys/id'].str.extract('(\d+)').astype(int)
df_expanded = df_expanded[df_expanded['number'] >= 275].drop(columns=['number'])




target_tags = {'image', 'radiomics', 'artifacts', 'attention', 'late'}




df_expanded = df_expanded[df_expanded['sys/group_tags'].apply(
    lambda x: set(x.split(',')) == target_tags
)]

metric_cols = [
    'accuracy',
    'macro_precision',
    'macro_recall',
    'macro_f1',
    'weighted_precision',
    'weighted_recall',
    'weighted_f1'
]

formatted_metrics = []
for col in metric_cols:
    mean = df_expanded[col].mean() * 100
    std = df_expanded[col].std() * 100
    formatted = f"{mean:.2f} ± {std:.2f}"
    formatted_metrics.append(formatted)

latex_row = ' & '.join(formatted_metrics) + ' \\\\'
print(len(df_expanded))
print(latex_row)


# %%