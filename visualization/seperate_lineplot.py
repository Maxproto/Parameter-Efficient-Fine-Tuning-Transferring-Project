import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Define your folder paths
folders = ["lora_transfer", "pt_transfer"]
datasets = ["agnews", "amazon", "dbpedia", "imdb"]
metrics = ["accuracy", "f1"]
methods = ["LoRA", "Prompt Tuning"]

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
    method = "LoRA" if "lora" in folder else "Prompt Tuning"
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

# Generate plots
for method in methods:
    for dataset in dataset_names_mapping.values():
        for metric in metrics:
            plt.figure(figsize=(12, 8))
            ax = sns.lineplot(x='Step', y=metric.capitalize(), hue='Source', style='Source', markers=True, dashes=False, data=exp_data[method][dataset][metric])
            ax.set_title(f"{dataset} - {metric.capitalize()} over Steps ({method})", fontsize=25)
            ax.set_xlabel("Step", fontsize=23)
            ax.set_ylabel(metric.capitalize(), fontsize=23)
            ax.tick_params(axis='x', labelsize=20) 
            ax.tick_params(axis='y', labelsize=20)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            ax.legend(title='Experiment', loc='lower right', fontsize=19)
            ax.grid(True, which="both", linestyle='--', linewidth=0.5, color='grey')

            plt.tight_layout()
            plt.savefig(f"{dataset.replace(' ', '_')}_{metric}_{method.replace(' ', '_')}_lineplot.png", dpi=300)
            plt.close()
