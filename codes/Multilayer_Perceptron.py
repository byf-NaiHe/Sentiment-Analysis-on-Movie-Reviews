# Import necessary libraries
import numpy as np
import pandas as pd
import os
import torch
import random
import gc
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset, random_split

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load training and testing data
train = pd.read_csv('E:/shuju/train.tsv', sep='\t')
test = pd.read_csv('E:/shuju/test.tsv', sep='\t')

# Function to perform TF-IDF vectorization on text data
def tfidf_vectorization(train_df, test_df):
    """
    Convert text data into TF-IDF vectors.
    
    Args:
        train_df (pd.DataFrame): Training data containing 'Phrase' column.
        test_df (pd.DataFrame): Testing data containing 'Phrase' column.
    
    Returns:
        X_train (np.array): TF-IDF vectors for training data.
        X_test (np.array): TF-IDF vectors for testing data.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
    X_train = vectorizer.fit_transform(train_df['Phrase'].astype(str)).toarray()  # Convert training text to TF-IDF
    X_test = vectorizer.transform(test_df['Phrase'].astype(str)).toarray()  # Convert testing text to TF-IDF
    
    return X_train, X_test, vectorizer

# Perform TF-IDF vectorization
X_train, X_test, vectorizer = tfidf_vectorization(train, test)

# Convert TF-IDF data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# Define a custom Dataset class
class PhraseDataset(Dataset):
    """
    Custom PyTorch Dataset for handling phrase data.
    
    Args:
        features (np.array or torch.Tensor): Feature data (TF-IDF vectors).
        labels (np.array or torch.Tensor, optional): Label data (sentiment).
    """
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]  # Return features and labels
        else:
            return self.features[idx]  # Return only features

# Prepare the dataset
train_labels = train['Sentiment'].values  # Extract training labels
train_dataset = PhraseDataset(X_train_tensor, train_labels)  # Create training dataset

# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Create DataLoader instances
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)  # Training DataLoader
val_loader = DataLoader(val_subset, batch_size=32)  # Validation DataLoader

# Define the sentiment classification model
class SentimentNN(nn.Module):
    """
    Neural network model for sentiment classification.
    
    Args:
        input_dim (int): Input dimension (size of TF-IDF vectors).
        output_size (int): Number of output classes.
    """
    def __init__(self, input_dim, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 32)  # Second fully connected layer
        self.fc3 = nn.Linear(32, output_size)  # Output layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)  # Output layer
        return x

# Set model parameters
input_dim = X_train.shape[1]  # Input dimension (size of TF-IDF vectors)
output_size = 5  # Number of sentiment classes
net = SentimentNN(input_dim, output_size).to(device)  # Initialize model and move to device
net.train()  # Set model to training mode

# Set hyperparameters
epochs = 100  # Number of training epochs
lr = 0.001  # Learning rate

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Early stopping setup
best_val_acc = 0  # Track best validation accuracy
patience = 10  # Number of epochs to wait for improvement
counter = 0  # Counter for early stopping

# Training loop
for epoch in range(epochs):
    net.train()  # Set model to training mode
    running_loss = 0.0  # Track training loss
    running_acc = 0.0  # Track training accuracy

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device

        optimizer.zero_grad()  # Clear gradients
        
        output = net(inputs)  # Forward pass
        loss = criterion(output, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss
        running_acc += (output.argmax(dim=1) == labels).float().mean()  # Accumulate accuracy

    # Print training metrics
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.6f}, Acc: {running_acc / len(train_loader):.6f}")

    # Validation phase
    net.eval()  # Set model to evaluation mode
    val_loss = 0.0  # Track validation loss
    val_acc = 0.0  # Track validation accuracy
    with torch.no_grad():  # Disable gradient computation
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)  # Move data to device
            val_output = net(val_inputs)  # Forward pass
            val_loss += criterion(val_output, val_labels).item()  # Accumulate loss
            val_acc += (val_output.argmax(dim=1) == val_labels).float().mean().item()  # Accumulate accuracy

    # Print validation metrics
    val_acc /= len(val_loader)  # Compute average validation accuracy
    print(f"Validation Loss: {val_loss / len(val_loader):.6f}, Validation Accuracy: {val_acc:.6f}")

    # Early stopping check
    if val_acc > best_val_acc:
        best_val_acc = val_acc  # Update best validation accuracy
        counter = 0  # Reset counter
    else:
        counter += 1  # Increment counter
        if counter >= patience:
            print("Early stopping triggered.")  # Stop training if no improvement
            break

# Test set predictions
net.eval()  # Set model to evaluation mode
test_predictions = []  # Store predictions
with torch.no_grad():  # Disable gradient computation
    test_loader = DataLoader(PhraseDataset(X_test_tensor), batch_size=32)  # Create test DataLoader
    for test_inputs in test_loader:
        test_inputs = test_inputs.to(device)  # Move data to device
        test_output = net(test_inputs)  # Forward pass
        test_predictions.extend(test_output.argmax(dim=1).cpu().numpy())  # Store predictions

# Create output DataFrame
output_df = pd.DataFrame({
    'PhraseId': test['PhraseId'],  # Ensure 'PhraseId' matches the test data
    'Sentiment': test_predictions
})

# Save predictions to CSV
output_path = './predictions.csv'
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create output directory if it doesn't exist
output_df.to_csv(output_path, index=False)  # Save predictions

print(f"Predictions saved to {output_path}")
