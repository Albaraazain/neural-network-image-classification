import os

import cv2
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def compute_loss(Y, A4, weights, l2_lambda):
    """
    Compute the loss function with L2 regularization.

    Args:
        Y (np.array): True labels.
        A4 (np.array): Predicted probabilities.
        weights (dict): Dictionary containing weights of the model.
        l2_lambda (float): Regularization parameter.

    Returns:
        float: Computed loss.
    """
    m = Y.shape[0]
    logprobs = -np.log(A4[range(m), Y])
    loss = np.sum(logprobs) / m
    l2_loss = (l2_lambda / (2 * m)) * (
            np.sum(np.square(weights['W1'])) +
            np.sum(np.square(weights['W2'])) +
            np.sum(np.square(weights['W3'])) +
            np.sum(np.square(weights['W4'])))
    loss += l2_loss
    return loss


def load_dataset(base_dir):
    """
    Load the dataset from the given directory and create a DataFrame.

    Args:
        base_dir (str): Base directory containing image files.

    Returns:
        pd.DataFrame: DataFrame with image paths and labels.
    """
    data = []  # Initialize an empty list to hold the image paths and labels.
    for root, dirs, files in os.walk(base_dir):  # Walk through the directory structure.
        for file in files:  # Iterate over the files in the current directory.
            if file.endswith(".jpg") or file.endswith(".png"):  # Check if the file is an image.
                label = os.path.basename(root)  # Use the directory name as the label.
                data.append((os.path.join(root, file), label))  # Append the file path and label to the data list.
    df = pd.DataFrame(data, columns=['image_path', 'label'])  # Convert the data list to a DataFrame.
    return df  # Return the DataFrame containing image paths and labels.


def split_train_validation(df, validation_size=0.25):
    """
    Split the dataset into training and validation sets.

    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        validation_size (float): Fraction of data to be used for validation.

    Returns:
        tuple: Training and validation DataFrames.
    """
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame and reset index.
    grouped = df.groupby('label')  # Group the DataFrame by label.
    train_df = pd.DataFrame()  # Initialize an empty DataFrame for training data.
    validation_df = pd.DataFrame()  # Initialize an empty DataFrame for validation data.

    for _, group in grouped:  # Iterate over each group of images with the same label.
        n_validation = int(len(group) * validation_size)  # Calculate the number of validation samples.
        validation_group = group[:n_validation]  # Select the validation samples.
        train_group = group[n_validation:]  # Select the training samples.
        validation_df = pd.concat([validation_df, validation_group], axis=0)  # Add validation samples to validation DataFrame.
        train_df = pd.concat([train_df, train_group], axis=0)  # Add training samples to training DataFrame.

    train_df = train_df.sample(frac=1).reset_index(drop=True)  # Shuffle the training DataFrame and reset index.
    validation_df = validation_df.sample(frac=1).reset_index(drop=True)  # Shuffle the validation DataFrame and reset index.

    return train_df, validation_df  # Return the training and validation DataFrames.


def imageLoaderAndFeatureExtractor(df, image_size=(32, 32)):
    """
    Load images from paths in the DataFrame, resize, normalize, and extract features.

    Args:
        df (pd.DataFrame): DataFrame with image paths and labels.
        image_size (tuple): Size to which images are resized.

    Returns:
        tuple: Features and labels as numpy arrays.
    """
    features = []  # Initialize an empty list to hold image features.
    labels = []  # Initialize an empty list to hold labels.

    for _, row in df.iterrows():  # Iterate over each row in the DataFrame.
        image_path = row['image_path']  # Get the image path.
        label = row['label']  # Get the label.
        image = cv2.imread(image_path)  # Read the image from the path.

        if image is None:  # Check if the image was successfully loaded.
            continue  # Skip this iteration if the image could not be loaded.

        image = cv2.resize(image, image_size)  # Resize the image to the specified size.
        image = image / 255.0  # Normalize the image to the range [0, 1].
        feature = image.flatten()  # Flatten the image to a 1D array of pixel values.

        features.append(feature)  # Add the feature array to the features list.
        labels.append(label)  # Add the label to the labels list.

    features = np.array(features)  # Convert the features list to a numpy array.
    labels = np.array(labels)  # Convert the labels list to a numpy array.

    return features, labels  # Return the features and labels as numpy arrays.


def setFormer(train_dir, test_dir, validation_split=0.25):
    """
    Load datasets from directories, split the training set into training and validation sets, and encode labels.

    Args:
        train_dir (str): Directory containing training data.
        test_dir (str): Directory containing testing data.
        validation_split (float): Fraction of training data to be used for validation.

    Returns:
        tuple: Training, validation, and test DataFrames and their respective labels.
    """
    train_df = load_dataset(train_dir)  # Load the training dataset.
    test_df = load_dataset(test_dir)  # Load the testing dataset.

    grouped = train_df.groupby('label')  # Group the training DataFrame by label.
    train_set = pd.DataFrame()  # Initialize an empty DataFrame for training data.
    validation_set = pd.DataFrame()  # Initialize an empty DataFrame for validation data.

    for label, group in grouped:  # Iterate over each group of images with the same label.
        n_validation = int(len(group) * validation_split)  # Calculate the number of validation samples.
        validation_samples = group[:n_validation]  # Select the validation samples.
        training_samples = group[n_validation:]  # Select the training samples.
        validation_set = pd.concat([validation_set, validation_samples], axis=0)  # Add validation samples to validation DataFrame.
        train_set = pd.concat([train_set, training_samples], axis=0)  # Add training samples to training DataFrame.

    train_set = train_set.sample(frac=1).reset_index(drop=True)  # Shuffle the training DataFrame and reset index.
    validation_set = validation_set.sample(frac=1).reset_index(drop=True)  # Shuffle the validation DataFrame and reset index.

    label_encoder = LabelEncoder()  # Initialize the label encoder.
    train_labels = label_encoder.fit_transform(train_set['label'].values)  # Encode the training labels.
    validation_labels = label_encoder.transform(validation_set['label'].values)  # Encode the validation labels.
    test_labels = label_encoder.transform(test_df['label'].values)  # Encode the testing labels.

    return train_set, train_labels, validation_set, validation_labels, test_df, test_labels  # Return the DataFrames and their respective labels.


class NeuralNetwork2HL(nn.Module):
    """
    Neural Network with 2 hidden layers.
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(NeuralNetwork2HL, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden2_size, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = self.fc3(x)  # Output layer (Softmax will be applied in the loss function)
        return x  # Return the final output


class NeuralNetwork3HL(nn.Module):
    """
    Neural Network with 3 hidden layers.
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super(NeuralNetwork3HL, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)  # Second fully connected layer
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)  # Third fully connected layer
        self.fc4 = nn.Linear(hidden3_size, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = F.relu(self.fc3(x))  # Apply ReLU activation after the third layer
        x = self.fc4(x)  # Output layer (Softmax will be applied in the loss function)
        return x  # Return the final output


def init_weights(m):
    """
    Initialize the weights of the neural network.

    Args:
        m (nn.Module): Layer of the neural network.
    """
    if isinstance(m, nn.Linear):  # Check if the layer is a fully connected layer
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Initialize weights using Kaiming uniform distribution
        nn.init.zeros_(m.bias)  # Initialize biases to zero


def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)  # Set seed for CPU operations
    torch.cuda.manual_seed(seed)  # Set seed for GPU operations
    np.random.seed(seed)  # Set seed for numpy operations
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN backend


set_seed(42)  # Seed value


def train_model(model, train_features, train_labels, val_features, val_labels, num_epochs=50, learning_rate=0.01,
                l2_lambda=0.001, save_interval=5):
    """
    Train the neural network model.

    Args:
        model (nn.Module): Neural network model.
        train_features (np.array): Training feature data.
        train_labels (np.array): Training labels.
        val_features (np.array): Validation feature data.
        val_labels (np.array): Validation labels.
        num_epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
        l2_lambda (float): L2 regularization parameter.
        save_interval (int): Interval for saving model checkpoints.
    """
    # Convert numpy arrays to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_features = torch.tensor(val_features, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.long)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # Lists to store training and validation history
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Zero the gradient buffers
        outputs = model(train_features)  # Forward pass
        loss = criterion(outputs, train_labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Compute training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_accuracy = (predicted == train_labels).sum().item() / train_labels.size(0)

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            val_outputs = model(val_features)  # Forward pass for validation
            val_loss = criterion(val_outputs, val_labels)  # Compute validation loss
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == val_labels).sum().item() / val_labels.size(0)

        # Record loss and accuracy
        train_loss_history.append(loss.item())
        val_loss_history.append(val_loss.item())
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')  # Save model checkpoint
            torch.save(optimizer.state_dict(), f'optimizer_epoch_{epoch + 1}.pth')  # Save optimizer checkpoint

    # Plot training and validation loss
    plt.figure()
    plt.plot(range(num_epochs), train_loss_history, label='Training Loss')
    plt.plot(range(num_epochs), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(range(num_epochs), train_acc_history, label='Training Accuracy')
    plt.plot(range(num_epochs), val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def evaluate_model(model, test_features, test_labels):
    """
    Evaluate the trained model on the test dataset.

    Args:
        model (nn.Module): Trained neural network model.
        test_features (np.array): Test feature data.
        test_labels (np.array): Test labels.
    """
    # Convert numpy arrays to PyTorch tensors
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        outputs = model(test_features)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class with the highest score
        accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)  # Compute accuracy
        print(f'Test Accuracy: {accuracy:.4f}')  # Print the test accuracy

def main():
    """
    Main function to execute the complete workflow: loading data, training models, and evaluating them.
    """
    # Load the training and testing datasets
    train_df = load_dataset('./Database/TrainingSet')
    test_df = load_dataset('./Database/TestingSet')

    # Split the training dataset into training and validation sets
    train_df, validation_df = split_train_validation(train_df)

    # Initialize the label encoder and transform the labels
    label_encoder = LabelEncoder()
    train_df['label'] = label_encoder.fit_transform(train_df['label'].values)
    validation_df['label'] = label_encoder.transform(validation_df['label'].values)
    test_df['label'] = label_encoder.transform(test_df['label'].values)

    # Extract features and labels from the datasets
    train_features, train_labels = imageLoaderAndFeatureExtractor(train_df)
    validation_features, validation_labels = imageLoaderAndFeatureExtractor(validation_df)
    test_features, test_labels = imageLoaderAndFeatureExtractor(test_df)

    # Define the input size and the sizes of the hidden layers and output layer
    input_size = 32 * 32 * 3
    hidden1_size = 256
    hidden2_size = 128
    hidden3_size = 128
    output_size = 3

    # Initialize models with 2 and 3 hidden layers
    model_2hl = NeuralNetwork2HL(input_size, hidden1_size, hidden2_size, output_size)
    model_3hl = NeuralNetwork3HL(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)

    # Apply weight initialization to the models
    model_2hl.apply(init_weights)
    model_3hl.apply(init_weights)

    # Define hyperparameters
    l2_lambda = 0.0001
    learning_rate = 0.004
    num_epochs = 150

    # Train the model with 2 hidden layers
    print("Training 2 hidden layers model")
    train_model(model_2hl, train_features, train_labels, validation_features, validation_labels, num_epochs=num_epochs,
                learning_rate=learning_rate, l2_lambda=l2_lambda)

    # Train the model with 3 hidden layers
    print("Training 3 hidden layers model")
    train_model(model_3hl, train_features, train_labels, validation_features, validation_labels, num_epochs=num_epochs,
                learning_rate=learning_rate, l2_lambda=l2_lambda)

    # Evaluate the model with 3 hidden layers on the test set
    print("Evaluating on test set")
    evaluate_model(model_3hl, test_features, test_labels)

if __name__ == "__main__":
    main()
