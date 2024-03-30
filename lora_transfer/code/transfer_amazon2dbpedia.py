import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import accuracy_score, f1_score
import itertools
from transformers import TrainerCallback
import random

class LoggingCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        with open(self.log_file, 'a') as file:
            file.write(f"Step: {state.global_step}, Evaluation results: {metrics}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a BERT model with PEFT")
    parser.add_argument('--base_model', type=str, default='../bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--source_model', type=str, default='lora-amazon_model__epoch5_lr0.0001_data25000', help='Pre-trained model name')
    parser.add_argument('--dataset_name', type=str, default='fancyzhx/dbpedia_14', help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='bert_peft_trainer', help='Output directory for the trained model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--bz', type=int, default=256, help='Batch size')
    parser.add_argument('--epoch', type=int, default=3, help='Number of epoch')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=25000, help='Number of training samples')
    parser.add_argument('--num_eval', type=int, default=5000, help='Number of test samples')

    return parser.parse_args()

def load_and_tokenize_data(model_name, dataset_name, seed, train_sample_size, eval_sample_size):
    # Set the random seed for reproducibility
    random.seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        if dataset_name == "sst2":
            text_key = "sentence"
        elif dataset_name == "amazon_polarity" or dataset_name == "fancyzhx/dbpedia_14":
            text_key = "content"
        else:
            text_key = "text"
        return tokenizer(examples[text_key], max_length=512, padding="max_length", truncation=True)

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Shuffle and sample the training dataset
    shuffled_train_dataset = dataset["train"].shuffle(seed=seed)
    if train_sample_size == -1:
        sampled_train_dataset = shuffled_train_dataset
    else:
        sampled_train_dataset = shuffled_train_dataset.select(range(train_sample_size))
    tokenized_train_dataset = sampled_train_dataset.map(tokenize_function, batched=True)

    # Shuffle and sample the evaluation dataset
    if dataset_name == "sst2":
        shuffled_eval_dataset = dataset["validation"].shuffle(seed=seed)
    else:
        shuffled_eval_dataset = dataset["test"].shuffle(seed=seed)
    if eval_sample_size == -1:
        sampled_eval_dataset = shuffled_eval_dataset
    else:
        sampled_eval_dataset = shuffled_eval_dataset.select(range(eval_sample_size))
    tokenized_eval_dataset = sampled_eval_dataset.map(tokenize_function, batched=True)

    return tokenized_train_dataset, tokenized_eval_dataset

'''def create_model(base_model, source_model, num_label = 2):
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=4)
    model = PeftModel.from_pretrained(model = base, model_id = source_model)
    original_classifier = model.base_model.classifier.original_module
    in_features = original_classifier.in_features
    new_classifier = torch.nn.Linear(in_features, num_label)
    model.base_model.classifier.original_module = new_classifier

    if 'default' in model.base_model.classifier.modules_to_save:
        model.base_model.classifier.modules_to_save['default'] = new_classifier

    print(model)
    model.print_trainable_parameters()
    return model'''

def create_model(base_model, source_model, num_label = 2):
    base = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    model_1 = PeftModel.from_pretrained(model = base, model_id = source_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_label)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS, # this is necessary
        inference_mode=True
    )
    model_2 = get_peft_model(model, lora_config)
    model_2.base_model.model.bert = model_1.base_model.model.bert
    model_2.print_trainable_parameters()
    return model_2


def main():
    # Hyperparameter ranges
    epochs = [30]
    learning_rates = [1e-4]

    # Log file setup
    log_file = 'transfer_amazon2dbpedia_log_epoch30_lr1e-4.txt'

    for epoch, lr in itertools.product(epochs, learning_rates):
        args = parse_args()

        # Override args with current hyperparameters
        args.epoch = epoch
        args.lr = lr

        # Rest of your setup code
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}, Epoch: {epoch}, Learning rate: {lr}")
        if args.dataset_name == "ag_news":
            num_label = 4
        elif args.dataset_name == "fancyzhx/dbpedia_14":
            num_label = 14
        else:
            num_label = 2
        model = create_model(args.base_model, args.source_model, num_label=num_label).to(device)
        tokenized_train_dataset, tokenized_eval_dataset = load_and_tokenize_data(args.base_model, args.dataset_name, args.seed, args.num_train, args.num_eval)
        
        

        # Define compute_metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='macro')
            return {"accuracy": accuracy, "f1": f1}

        # Update TrainingArguments with the new hyperparameters
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            evaluation_strategy="steps",
            eval_steps=int((len(tokenized_train_dataset) * args.epoch / 10 / args.bz)),
            save_strategy="no",
            learning_rate=lr,
            per_device_train_batch_size=args.bz,
            per_device_eval_batch_size=args.bz,
            num_train_epochs=epoch,
            weight_decay=0.01,
            report_to=[]
        )

        # Initialize Trainer with the custom callback
        logging_callback = LoggingCallback(log_file)
        bert_peft_trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[logging_callback]
        )

        # Train the model
        bert_peft_trainer.train()

        # Final evaluation results
        eval_results = bert_peft_trainer.evaluate()
        with open(log_file, 'a') as file:
            file.write(f"Final Results -  Epoch: {epoch}, LR: {lr}, Results: {eval_results}\n")

        # Optional: Save model for each configuration
        model.save_pretrained(f"lora-transfer_amazon2dbpedia__epoch{epoch}_lr{lr}_data{args.num_train}")

if __name__ == "__main__":
    main()
