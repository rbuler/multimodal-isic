import pickle
from pathlib import Path
import numpy as np
import pandas as pd
# %%

def process_teacher_file(input_path):
    input_path = Path(input_path)

    with open(input_path, "rb") as f:
        df = pickle.load(f)

    stats = []

    for _, row in df.iterrows():

        dominant_class = np.argmax(row["patch_probs"], axis=1)

        stats.append({
            "image_id": row["image_id"],
            "label": row["label"],

            "patch_embeddings": row["patch_embeddings"],
            "patch_probs": row["patch_probs"],
            "dominant_class": dominant_class
        })

    stats_df = pd.DataFrame(stats)
    patch_stats_root = Path('/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/patch_stats')
    teacher_outputs_root = Path('/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/teacher_outputs')
    relative_path = input_path.relative_to(teacher_outputs_root)
    output_path = patch_stats_root / relative_path
    output_path = Path(str(output_path).replace("teacher_outputs_f", "patch_stats_f"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(stats_df, f)

    input_path.unlink()
    print(f"Saved: {output_path}")


root = Path('/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/teacher_outputs')

for input_path in sorted(root.rglob('*.pkl')):
    if input_path.name.startswith('patch_stats_'):
        continue
    process_teacher_file(str(input_path))

# %%