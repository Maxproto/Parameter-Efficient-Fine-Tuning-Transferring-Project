import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mapping from dataset identifiers to more readable names
dataset_names_mapping = {
    'imdb': 'IMDb',
    'amazon': 'Amazon Polarity',
    'dbpedia': 'DBpedia',
    'agnews': 'AG News'
}

# Function to read and parse files
def read_results(folder_path):
    results = {}
    pattern = re.compile(r"Step: \d+, Evaluation results:.*?'eval_accuracy': (\d+\.\d+),.*?'epoch': (\d+\.\d+|\d+)")

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt') and not file_name.startswith('transfer_'):
            # Use the part before "_log" as a key to look up in the dataset_names_mapping
            dataset_key = file_name.split('_log')[0]
            dataset_name = dataset_names_mapping.get(dataset_key, dataset_key)  # Fallback to the key if not found in mapping
            with open(os.path.join(folder_path, file_name), 'r') as file:
                if dataset_name not in results:
                    results[dataset_name] = []
                
                for line in file:
                    match = pattern.search(line)
                    if match:
                        accuracy, epoch = match.groups()
                        results[dataset_name].append((float(epoch), float(accuracy)))

    return results

# Function to create line plots
def plot_results(lora_results, pt_results, readable_dataset_name):
    plt.figure(figsize=(10, 6))
    df_lora = pd.DataFrame(lora_results[readable_dataset_name], columns=['Epoch', 'Accuracy'])
    df_pt = pd.DataFrame(pt_results[readable_dataset_name], columns=['Epoch', 'Accuracy'])
    sns.lineplot(data=df_lora, x='Epoch', y='Accuracy', label='LoRA', marker='o', linestyle='-', linewidth=2)
    sns.lineplot(data=df_pt, x='Epoch', y='Accuracy', label='Prompt Tuning', marker='o', linestyle='--', linewidth=2)
    
    # Adding grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')
    
    # Enhancing the plot
    # plt.title(f'Comparison of LoRA and Prompt Tuning on {readable_dataset_name}', fontsize=16)
    plt.xlabel('Epoch', fontsize=25)
    plt.ylabel('Accuracy Score', fontsize=25)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24, loc="lower right")
    
    plt.tight_layout()  # Adjusts plot parameters to give some padding
    plt.savefig(f'{readable_dataset_name}_comparison.png', dpi=300)
    plt.show()

# Function to create heatmap
def create_heatmap(data, title, file_name):
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap='viridis')
    plt.title(title)
    plt.savefig(f'{file_name}.pdf', dpi=300)
    plt.show()

# Main code
lora_folder = 'lora_transfer'
pt_folder = 'pt_transfer'

lora_results = read_results(lora_folder)
pt_results = read_results(pt_folder)

# Plotting line plots for each dataset
for dataset in lora_results.keys():
    plot_results(lora_results, pt_results, dataset)
