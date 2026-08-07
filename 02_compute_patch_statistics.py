import pickle
from pathlib import Path
import numpy as np
import pandas as pd
# %%

def compute_patch_statistics(patch_probs):
    """
    patch_probs: (196, 7)
    """

    eps = 1e-8

    # entropy per patch
    entropy = -np.sum(
        patch_probs * np.log(patch_probs + eps),
        axis=1
    )

    # confidence per patch
    confidence = np.max(
        patch_probs,
        axis=1
    )

    # most likely class
    dominant_class = np.argmax(
        patch_probs,
        axis=1
    )

    return {
        "entropy": entropy,                    # (196,)
        "confidence": confidence,              # (196,)
        "dominant_class": dominant_class       # (196,)
    }

def process_teacher_file(input_path):
    input_path = Path(input_path)

    with open(input_path, "rb") as f:
        df = pickle.load(f)

    stats = []

    for _, row in df.iterrows():

        stat = compute_patch_statistics(row["patch_probs"])

        entropy = stat["entropy"]
        confidence = stat["confidence"]
        dominant_class = stat["dominant_class"]

        stats.append({
            "image_id": row["image_id"],
            "label": row["label"],

            "patch_embeddings": row["patch_embeddings"],
            "patch_probs": row["patch_probs"],
            "attention": row["attention"],

            "entropy": entropy,
            "confidence": confidence,
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