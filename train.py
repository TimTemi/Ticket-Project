from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from preprocess import preprocess_data
import pandas as pd
import torch
from torch.utils.data import Dataset

# Load and preprocess data
def load_data(path):
    df = pd.read_csv(path)
    df = preprocess_data(df)  # Clean the text data
    df['queue_label'], _ = pd.factorize(df['queue'])  # Convert queue to numeric labels
    return df

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
        
        # Tokenize the text and return a dictionary
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Return the correct format for Trainer
        return {
            'input_ids': inputs['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)  # Convert label to tensor
        }

# Training function
def train_model(data_path='./data/helpdesk_customer_tickets.csv'):
    # Load and preprocess data
    df = load_data(data_path)
    
    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(df['queue_label'].unique()))
    
    # Prepare training data
    train_texts = df['cleaned_body'].tolist()
    train_labels = df['queue_label'].tolist()
    train_dataset = TicketDataset(train_texts, train_labels, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        logging_steps=10
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

# Run training
if __name__ == "__main__":
    train_model()
