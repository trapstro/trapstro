import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from collections import Counter

# Load the datasets
class PTBDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = pd.read_csv(file_path, sep='\t', header=None)
        self.tokenizer = tokenizer  # Store the tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]  # Assuming text is in the first column
        # Encode the text
        encoded = self.tokenizer.encode(text, return_tensors='pt', truncation=True, padding='max_length',
                                        max_length=512)
        return encoded.squeeze(0)  # Remove the extra dimension


class WikiText2Dataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = pd.read_csv(file_path, sep='\t', header=None)
        self.tokenizer = tokenizer  # Store the tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]  # Assuming text is in the first column
        # Encode the text
        encoded = self.tokenizer.encode(text, return_tensors='pt', truncation=True, padding='max_length',
                                        max_length=512)
        return encoded.squeeze(0)  # Remove the extra dimension


# Define the paths to your dataset files
ptb_file_path = r"C:\Users\USER\Downloads\test.txt\test.txt"  # Adjusted to correct path
wikitext_train_file_path = r"C:\Users\USER\Downloads\archive\ptbdataset\ptb.train.txt"
wikitext_valid_file_path = r"C:\Users\USER\Downloads\archive\ptbdataset\ptb.valid.txt"

# Load the tokenizer
model_name = "gpt2"  # Change this to your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if the tokenizer has a padding token
if tokenizer.pad_token is None:
    # Use the EOS token as the padding token
    tokenizer.pad_token = tokenizer.eos_token

# Load datasets
ptb_data = PTBDataset(ptb_file_path, tokenizer)
wikitext_train_data = WikiText2Dataset(wikitext_train_file_path, tokenizer)
wikitext_valid_data = WikiText2Dataset(wikitext_valid_file_path, tokenizer)

# Create DataLoaders
train_loader = DataLoader(wikitext_train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(wikitext_valid_data, batch_size=32, shuffle=False)

# Load a pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize lists to track loss
train_losses = []
valid_losses = []


# Training function
def train_model(model, train_loader, valid_loader, epochs=3):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Ensure the batch is of type Long
            batch = batch.long()  # Convert to Long tensor
            outputs = model(input_ids=batch, labels=batch)  # Pass labels to compute loss

            if outputs is None or outputs.loss is None:
                print("Output is None. Check the model's forward method.")
                continue  # Skip this batch if outputs are not valid

            loss = outputs.loss  # Get the loss from the outputs
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Record average training loss
        train_losses.append(epoch_train_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_train_loss / len(train_loader)}")

        # Validate the model
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch.long()  # Convert to Long tensor
                outputs = model(input_ids=batch, labels=batch)  # Pass labels to compute loss
                if outputs is None or outputs.loss is None:
                    print("Output is None during validation. Check the model's forward method.")
                    continue  # Skip this batch if outputs are not valid

                total_loss += outputs.loss.item()

        # Record average validation loss
        valid_losses.append(total_loss / len(valid_loader))
        print(f"Validation Loss: {total_loss / len(valid_loader)}")


# Train your model on the combined datasets
train_model(model, train_loader, valid_loader)

# Save the model
model.save_pretrained("trained_model")
tokenizer.save_pretrained("trained_model")


# Analysis and Visualization
def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(valid_losses, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()  # Ensure show is called to display the plot


def plot_text_length_distribution(dataset, title):
    lengths = [len(tokenizer.encode(text, truncation=True)) for text in dataset.data[0]]
    plt.figure(figsize=(10, 6))
    sns.histplot(lengths, bins=30, kde=True)
    plt.title(f'Text Length Distribution: {title}')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()  # Ensure show is called to display the plot


# Plot losses
plot_losses(train_losses, valid_losses)

# Plot text length distribution for training and validation datasets
plot_text_length_distribution(wikitext_train_data, "WikiText Train Data")
plot_text_length_distribution(wikitext_valid_data, "WikiText Validation Data")


# Token frequency analysis
def plot_token_frequency(dataset, num_tokens=10):
    all_text = ' '.join(dataset.data[0])
    tokens = word_tokenize(all_text.lower())
    token_counts = Counter(tokens)
    most_common_tokens = token_counts.most_common(num_tokens)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=[token for token, _ in most_common_tokens], y=[count for _, count in most_common_tokens])
    plt.title(f'Top {num_tokens} Most Common Tokens')
    plt.xlabel('Tokens')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()  # Ensure show is called to display the plot


# Plot token frequency for training data
plot_token_frequency(wikitext_train_data, num_tokens=10)
