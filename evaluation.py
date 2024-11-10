from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from preprocess import preprocess_data
import pandas as pd
import torch
from torch.utils.data import Dataset
import os

# Custom Dataset Class
class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load data for evaluation
def load_data(path):
    df = pd.read_csv(path)
    df = preprocess_data(df)
    df['queue_label'], _ = pd.factorize(df['queue'])
    return df

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Evaluation function
def evaluate_model(data_path='./data/helpdesk_customer_tickets.csv', model_path='./saved_model'):
    # Load and preprocess data
    df = load_data(data_path)
    eval_texts = df['cleaned_body'].tolist()
    eval_labels = df['queue_label'].tolist()
    
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(df['queue_label'].unique()))
    
    # Create dataset for evaluation
    eval_dataset = TicketDataset(eval_texts, eval_labels, tokenizer)

    # Define training arguments for evaluation
    training_args = TrainingArguments(
        output_dir=model_path,
        per_device_eval_batch_size=8,
        logging_dir='./logs',
        logging_steps=10
    )

    # Initialize Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Evaluate the model
    eval_result = trainer.evaluate()
    print("Evaluation results:", eval_result)
    
    # Ensure the save directory exists
    os.makedirs(model_path, exist_ok=True)
    
    # Save model and tokenizer
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pth'))  # Save model without safetensors
    tokenizer.save_pretrained(model_path)  # Tokenizer can still use save_pretrained
    print(f"Model and tokenizer saved to {model_path}")

# Run evaluation
if __name__ == "__main__":
    evaluate_model()

