# %%
import json
import pandas as pd
from pathlib import Path

# Path to the ray results directory
# ray_results_path = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/tune_mil_outputs/ray_results/ray_tune_graph_mil_260402_164517")
ray_results_path = Path("/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/tune_mil_outputs/ray_results/ray_tune_graph_mil_260402_163851")

# Find all result.json files recursively
json_files = list(ray_results_path.rglob("result.json"))

print(f"Found {len(json_files)} result.json files\n")

results_list = []

for json_file in sorted(json_files):
    try:
        with open(json_file, 'r') as f:
            # Read all lines (each line is a JSON record)
            lines = f.readlines()
            
        if len(lines) > 0:
            # Get the last line (best iteration)
            last_line = lines[-1].strip()
            if last_line:
                data = json.loads(last_line)
                # Add the file path for reference
                data['trial_dir'] = str(json_file.parent.name)
                results_list.append(data)
                print(f"✓ {json_file.parent.name}")
            else:
                print(f"✗ {json_file.parent.name} - empty last line")
        else:
            print(f"✗ {json_file.parent.name} - no content")
            
    except Exception as e:
        print(f"✗ {json_file.parent.name} - Error: {e}")

# Create DataFrame
df = pd.DataFrame(results_list)

print(f"\n{'='*80}")
print(f"EXTRACTED {len(df)} TRIAL RESULTS")
print(f"{'='*80}\n")

print(df.to_string())

print(f"\n{'='*80}")
print(f"COLUMNS: {list(df.columns)}")
print(f"{'='*80}\n")

# Optionally, save to CSV
# output_csv = "/users/project1/pt01191/MMODAL_ISIC/Code/multimodal-isic/ray_results_summary.csv"
# df.to_csv(output_csv, index=False)
# print(f"Saved to: {output_csv}\n")
# %%