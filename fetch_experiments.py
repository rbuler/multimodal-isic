import re
import neptune
import pandas as pd
from neptune.exceptions import NeptuneException
# %%
def fetch_experiment(
    tags: list = None,
    group_tags: list = None,
    experiment_ids: list = None,
):
    try:
        project = neptune.init_project(project="ProjektMMG/multimodal-isic",)
        runs_table = project.fetch_runs_table(sort_by='sys/creation_time')

        runs_table = runs_table.to_pandas()

        if tags:
            def tag_contains_all(run_tags):
                if pd.isna(run_tags):
                    return False
                if isinstance(run_tags, str):
                    items = [t.strip() for t in run_tags.split(',') if t.strip()]
                else:
                    items = list(run_tags)
                return all(tag in items for tag in tags)

            if "tags" in runs_table.columns:
                runs_table = runs_table[runs_table["tags"].apply(tag_contains_all)]

        if group_tags:
            def group_contains_any(run_groups):
                if pd.isna(run_groups):
                    return False
                if isinstance(run_groups, str):
                    items = [g.strip() for g in run_groups.split(',') if g.strip()]
                else:
                    items = list(run_groups)
                return any(gtag in items for gtag in group_tags)

            if "groups" in runs_table.columns:
                runs_table = runs_table[runs_table["groups"].apply(group_contains_any)]

        if experiment_ids:
            ids_set = set()
            for i in experiment_ids:
                try:
                    ids_set.add(int(i))
                except (ValueError, TypeError):
                    continue

            if "sys/id" in runs_table.columns:
                def id_matches(run_id):
                    if pd.isna(run_id):
                        return False
                    m = re.search(r'(\d+)', str(run_id))
                    return int(m.group(1)) in ids_set if m else False

            runs_table = runs_table[runs_table["sys/id"].apply(id_matches)]

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



def main():
    results_df = fetch_experiment(
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
if __name__ == "__main__":
    main()


# %%