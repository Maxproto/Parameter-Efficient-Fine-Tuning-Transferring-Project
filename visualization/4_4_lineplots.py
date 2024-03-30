import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns

# Define your folder paths
folders = ["lora_transfer", "pt_transfer"]
datasets = ["agnews", "amazon", "dbpedia", "imdb"]
metrics = ["accuracy", "f1"]
methods = ["LoRA", "PT"]

# Dataset names mapping
dataset_names_mapping = {
    'agnews': 'AG News',
    'amazon': 'Amazon Polarity',
    'dbpedia': 'DBpedia',
    'imdb': 'IMDb'
}

# Data structure to store the steps and scores for each experiment
exp_data = {method: {dataset_names_mapping[dataset]: {metric: [] for metric in metrics} for dataset in datasets} for method in methods}

# Read and process files
for folder in folders:
    method = "LoRA" if "lora" in folder else "PT"
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            if 'transfer' in file_name:
                src_tgt_match = re.findall(r'transfer_(\w+?)2(\w+?)_', file_name)
                if src_tgt_match:
                    src, tgt = src_tgt_match[0]
                    src = dataset_names_mapping[src]  # Map source dataset name
                    tgt = dataset_names_mapping[tgt]  # Map target dataset name
                    src_label = f"Transfer from {src}"  # Correctly label as transfer
            else:
                scratch_match = re.search(r'(\w+?)_log_', file_name)
                if scratch_match:
                    src = scratch_match.group(1)
                    tgt = src  # For training from scratch
                    src = dataset_names_mapping[src]  # Map source (same as target here) dataset name
                    tgt = dataset_names_mapping[tgt]  # No change needed, but for consistency
                    src_label = "Training from scratch"  # Label as training from scratch
            
            with open(os.path.join(folder, file_name), 'r') as file:
                lines = file.readlines()[:-1]  # Exclude the last line
                for line in lines:
                    step_match = re.search(r'Step: (\d+)', line)
                    results_match = re.search(r"'eval_accuracy': ([\d.]+), 'eval_f1': ([\d.]+)", line)
                    if step_match and results_match:
                        step, accuracy, f1 = int(step_match.group(1)), float(results_match.group(1)), float(results_match.group(2))
                        # Append data with clear source labeling
                        exp_data[method][tgt]["accuracy"].append((src_label, step, accuracy))
                        exp_data[method][tgt]["f1"].append((src_label, step, f1))


# Convert lists to DataFrames for plotting
for method in methods:
    for dataset in dataset_names_mapping.values():
        for metric in metrics:
            exp_data[method][dataset][metric] = pd.DataFrame(exp_data[method][dataset][metric], columns=['Source', 'Step', metric.capitalize()]).sort_values('Step')

# Adjusting the layout for the subplot and legend
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(24, 24))
fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust top margin to make space for the legend

axes_flat = axes.flatten()

# Counter to keep track of the subplot position
counter = 0

# Dictionary to track unique legends
legend_labels = {}

for method in methods:
    for dataset in dataset_names_mapping.values():
        for metric in metrics:
            ax = axes_flat[counter]
            sns_plot = sns.lineplot(
                x='Step', 
                y=metric.capitalize(), 
                hue='Source', 
                style='Source', 
                markers=True, 
                dashes=False, 
                data=exp_data[method][dataset][metric], 
                ax=ax
            )
            ax.set_title(f"{dataset} - {metric.capitalize()} ({method})", fontsize=25)  # Increased font size
            ax.set_xlabel("Step", fontsize=24)  # Increased font size
            ax.set_ylabel(metric.capitalize(), fontsize=24)  # Increased font size
            ax.tick_params(axis='x', labelsize=20)  # Increased font size
            ax.tick_params(axis='y', labelsize=20)  # Increased font size
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.grid(True, which="both", linestyle='--', linewidth=0.5, color='grey')
            counter += 1
            # Update unique legend labels
            for line in ax.get_lines():
                label = line.get_label()
                if label not in legend_labels:
                    legend_labels[label] = line

# Assuming `legend_labels` is your dict with labels and line properties
proxy_artists = []
new_labels = []
for label, line in legend_labels.items():
    # Create a proxy artist for the line
    proxy_artists.append(mlines.Line2D([], [], color=line.get_color(), marker=line.get_marker(), linestyle=line.get_linestyle()))
    new_labels.append(label)

# Use `proxy_artists` and `new_labels` for the legend
fig.legend(proxy_artists, new_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=20)

# Remove individual legends from subplots
for ax in axes_flat:
    ax.get_legend().remove()

# Adjust layout and save figure
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rectangle in which to fit the subplots
plt.savefig("combined_performance_plots.png", dpi=300, bbox_inches='tight')
plt.close()