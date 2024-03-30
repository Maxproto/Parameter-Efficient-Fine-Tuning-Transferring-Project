import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define your folder paths
folders = ["lora_transfer", "pt_transfer"]

# Initialize data structures for accuracy and F1 scores
dataset_names_mapping = {
    'agnews': 'AG News',
    'amazon': 'Amazon Polarity',
    'dbpedia': 'DBpedia',
    'imdb': 'IMDb'
}
datasets = list(dataset_names_mapping.keys())
metrics = ["accuracy", "f1"]
methods = ["LoRA", "Prompt Tuning"]

# Initialize results and original_scores dictionaries
results, original_scores = {}, {}
for method in methods:
    results[method], original_scores[method] = {}, {}
    for metric in metrics:
        df_template = pd.DataFrame(index=datasets, columns=datasets)
        results[method][metric] = df_template.copy()
        original_scores[method][metric] = df_template.copy()

def process_files(folder):
    tmp_method = folder.split('_')[0]
    if tmp_method == 'lora':
        method = 'LoRA'
    else:
        method = 'Prompt Tuning'
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder, file_name), 'r') as file:
                lines = file.readlines()
                final_line = lines[-1]
                match = re.search(r"'eval_accuracy': ([\d.]+), 'eval_f1': ([\d.]+)", final_line)
                if match:
                    accuracy, f1 = map(float, match.groups())
                    # Attempt to match transfer pattern
                    transfer_match = re.findall(r'transfer_(\w+?)2(\w+?)_', file_name)
                    # Check if file is for transfer or training from scratch
                    if transfer_match:
                        source, target = transfer_match[0]
                    else:
                        # Attempt to extract source for training from scratch
                        scratch_match = re.search(r'(\w+?)_log_', file_name)
                        if scratch_match:
                            source = scratch_match.group(1)
                            target = source
                        else:
                            continue  # Skip file if neither pattern matches
                    for metric, value in zip(metrics, [accuracy, f1]):
                        results[method][metric].loc[source, target] = value


# Process all files
for folder in folders:
    process_files(folder)

# Map dataset codes to full names
for method in methods:
    for metric in metrics:
        results[method][metric].index = results[method][metric].index.map(dataset_names_mapping)
        results[method][metric].columns = results[method][metric].columns.map(dataset_names_mapping)
        original_scores[method][metric].index = original_scores[method][metric].index.map(dataset_names_mapping)
        original_scores[method][metric].columns = original_scores[method][metric].columns.map(dataset_names_mapping)

new_order = ['AG News', 'DBpedia', 'Amazon Polarity', 'IMDb']

# Calculate ratios and update original scores
for method in methods:
    for metric in metrics:
        results[method][metric] = results[method][metric].reindex(index=new_order, columns=new_order)
        original_scores[method][metric] = results[method][metric].copy()
        diagonal_values = results[method][metric].values.diagonal()
        for i, col in enumerate(results[method][metric].columns):
            results[method][metric][col] = results[method][metric][col] / diagonal_values[i]
        results[method][metric] = results[method][metric].astype(float)


# Generate and save heatmaps
for method in methods:
    for metric in metrics:
        # Find the maximum and minimum ratio values for the current method and metric
        max_ratio = results[method][metric].max().max()
        min_ratio = results[method][metric].min().min()
        
        # Determine the range to set the color scale such that 1 is the center
        range_max = max(max_ratio - 1, 1 - min_ratio)
        vmin, vmax = 1 - range_max, 1 + range_max

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(results[method][metric], annot=True, fmt=".4f", cmap='vlag', cbar_kws={'label': 'Ratio'},
                         annot_kws={"size": 20, "color": "black"}, vmin=0.97, vmax=1.03, center=1)
        cbar = ax.collections[0].colorbar
        cbar.set_label('Ratio', size=18) 
        cbar.ax.tick_params(labelsize=18) 
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(f"{method} {metric.capitalize()} Heatmap", fontsize=25)
        plt.ylabel('Source Dataset', fontsize=22)
        plt.xlabel('Target Dataset', fontsize=22)

        # Manually add original score annotations
        # for y in range(len(datasets)):
        #     for x in range(len(datasets)):
        #         original_score = original_scores[method][metric].iloc[y, x]
        #         if not pd.isna(original_score):
        #             plt.text(x + 0.5, y + 0.7, f"({original_score:.4f})", ha='center', va='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{method}_{metric}_heatmap_with_annotations.png", dpi=300)
