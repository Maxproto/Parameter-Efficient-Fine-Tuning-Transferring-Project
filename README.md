# Parameter-Efficient-Fine-Tuning-Transferring-Project
## Overview
By leveraging data from source tasks, PEFT parameters are trained from scratch and subsequently applied as initializations for target tasks. The primary goal is to evaluate the improvements in model performance, with a focus on convergence speed and final accuracy.

## Methodology
1. **Training PEFT Parameters**: Initially, PEFT parameters are trained from scratch using a dataset compiled from various source tasks. This phase is crucial for capturing a broad spectrum of knowledge that can be beneficial for downstream tasks.
2. **Application to Target Tasks**: The trained PEFT parameters are then applied as initializations for models performing on target tasks. This step is designed to assess the effectiveness of PEFT parameters in enhancing model performance across different domains.

## Evaluation Metrics
The impact of PEFT parameters on model performance is evaluated using the following metrics:
1. **Accuracy**: This metric represents the proportion of instances correctly predicted by the model out of the total number of instances. A higher accuracy indicates better performance in making correct predictions.
2. **Macro-F1 Score**: The Macro-F1 Score is calculated as the F1 Score across all classes, which balances the precision and recall of the model. It provides a more nuanced view of model performance across different categories.
3. **Speed of Convergence**: This metric measures the number of epochs required for the model to reach the convergence accuracy compared to training from scratch. A lower number of epochs signifies faster learning and adaptation by the model, indicating a positive impact of PEFT parameters on learning efficiency.
