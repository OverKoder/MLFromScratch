# Implementation of a Neural Network

# PyTorch
import torch
from torch.utils.data import DataLoader, random_split

# Torchvision
import torchvision
import torchvision.datasets as datasets

# Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Other
import numpy as np
import math
from tqdm import tqdm

# -------- Global variables ------------
# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def ReLU(x: torch.tensor) -> torch.tensor:
    """
    Calculates the ReLU function element-wise

    Args:
        x (torch.tensor): The value of the input x

    Returns:
        ReLU: The ReLU value
    """    

    # Calculate ReLU
    ReLU = torch.where(x > 0, x, 0)

    return ReLU

def ReLU_derivative(x):
    """
    Calculates the derivative of the ReLU function

    Args:
        x (torch.tensor): The value of the input x

    Returns:
        ReLU_derivative: The derivative of the ReLU function
    """    

    # Calculate ReLU derivative
    ReLU_derivative = torch.where(x > 0, 1, 0)

    return ReLU_derivative

def softmax(x):
    """
    Calculates the softmax function

    Args:
        x (torch.tensor): The value of the input x

    Returns:
        softmax: The softmax value
    """    

    # Numerically stable with large exponentials
    softmax = torch.exp(x - x.max())
    return softmax / torch.sum(softmax, dim=0)

def cross_entropy_loss(pred_label_prob: float) -> float:
    """
    Calculates the cross-entropy loss 

    Args:
        pred_label_prob (float): The predicted label probability

    Returns:
        loss: The loss
    """      
    # Calculate loss (some numerical stability adjustments are made)
    if pred_label_prob < 1e-15:
        loss = -math.log(1e-15)

    elif pred_label_prob > 1 - 1e-15:
        loss = -math.log(1 - 1e-15)

    else:
        loss = -math.log(pred_label_prob)

    return loss


def cross_entropy_loss_derivative(softmax_vector: torch.tensor, true_label: int) -> torch.tensor:
    """
    Calculates the cross-entropy loss 

    Args:
        softmax_vector (torch.tensor): The softmax vector

    Returns:
        loss: The loss
    """      
    # One hot encoded vector of the true label
    one_hot_encoded_true_label = torch.zeros(softmax_vector.shape).to(device)
    one_hot_encoded_true_label[true_label] = 1

    # Calculate loss
    loss = softmax_vector - one_hot_encoded_true_label
    
    return loss



class NeuralNetwork():

    def __init__(self, input_size: int, hidden_sizes: list, output_sizes: int, learning_rate: float, device: str):
        """
        Initialize Neural Network
        Args:
            input_size (int): The size of the input layer.
            hidden_sizes (list): The list of sizes for the hidden layers. The length of the list is the number of hidden layers.
            output_sizes (int): The size of the input layer. Usually the number of classes
            learning_rate (float): The learning rate used for training.
        """

        # Attributes
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_sizes
        self.learning_rate = learning_rate
        self.device = device

        # Initialize weights and biases
        self.weights = []
        self.biases = []

        # Input layer to first hidden layer
        self.weights.append(torch.rand((self.input_size, self.hidden_sizes[0]), device = self.device))
        self.biases.append(torch.rand((1, self.hidden_sizes[0]), device = self.device))

        # Hidden layers
        for i in range(len(self.hidden_sizes) - 1):
            self.weights.append(torch.rand((self.hidden_sizes[i], self.hidden_sizes[i + 1]), device = self.device))
            self.biases.append(torch.rand((1, self.hidden_sizes[i + 1]), device = self.device))

        # Last hidden layer to output layer
        self.weights.append(torch.rand((self.hidden_sizes[-1], self.output_size), device = self.device))
        self.biases.append(torch.rand((1, self.output_size), device = self.device))



    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the neural network

        Args:
            x (torch.tensor): The input data (shape: (input_size, 1))

        Returns:
            output: The output of the neural network
        """    

        # Output without applying activation function
        phi_list = []

        # Output after applying activation function
        g_phi_list = [x]


        # Forward pass
        for layer in range(len(self.weights)):

            # Multiply weights with input and add bias
            x = torch.matmul(x, self.weights[layer]) + self.biases[layer]
            phi_list.append(x)

            # Apply activation function
            if layer < len(self.weights) - 1:

                # ReLU
                x = ReLU(x)
                g_phi_list.append(x)

        return phi_list, g_phi_list

    def backward(self, phi_list: list, g_phi_list: list, true_label: int) -> None:
        """
        Calculates the backward pass.

        Args:
            phi_list (list): The outputs of the hidden layers without applying activation function
            g_phi_list (list): The outputs of the hidden layers with activation function 
            true_label (int): True label (to calculate loss)

        Returns:
            loss: The loss
        """    

        # From finish to start
        for layer in range(len(self.weights) - 1, -1, -1):

            if layer == len(self.weights) - 1:
                
                # Since it is the last layer, we use the cross-entropy loss as the error
                # Get the last output of the network
                softmax_vector = softmax(phi_list[layer] [0])

                # Calculate loss
                loss =  cross_entropy_loss(softmax_vector [true_label])

                # Output layer
                delta = cross_entropy_loss_derivative(softmax_vector, true_label) * loss
                delta = delta.reshape((1, delta.shape[0]))

                # Update weights and biases
                self.weights[layer] -= self.learning_rate * torch.matmul(g_phi_list[layer].T, delta)
                self.biases[layer] -= self.learning_rate * delta
            else:
                    
                # Hidden layer
                delta = ReLU_derivative(phi_list[layer]) * torch.matmul(delta, self.weights[layer + 1].T)

                # Update weights and biases
                self.weights[layer] -= self.learning_rate * torch.matmul(g_phi_list[layer].T, delta)
                self.biases[layer] -= self.learning_rate * delta

        return loss
        

    def train(self, train_loader, validation_loader, epochs=30):
        """
        Trains the neural network

        Args:
            target_word (str): The target word
            context_index (int): The index of the context word
            output (torch.tensor): The output of the neural network
        """
        best_accuracy = 0.0
        for epoch in range(1, epochs + 1):

            total_loss = 0
            # Mini batch gradient descent
            for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc = "Training...")):

                # Inputs and labels
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = torch.flatten(inputs, 1)

                # Forward pass
                phi_list, g_phi_list = self.forward(inputs)

                # Backward pass
                loss = self.backward(phi_list, g_phi_list, labels.item())
                total_loss += loss

            print(f"Epoch: {epoch}, Loss: {loss/len(train_loader)}")

            # Predict all the data
            y_true_list = []
            y_pred_list = []

            for batch_idx, (inputs, labels) in enumerate(tqdm(validation_loader, desc = "Calculating accuracy...")):

                # Inputs and labels
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = torch.flatten(inputs, 1)

                y_true_list.append(labels.item())
                y_pred_list.append(self.predict(inputs).item())

            # Calculate accuracy
            accuracy = accuracy_score(y_true_list, y_pred_list) * 100

            if accuracy > best_accuracy:
                best_accuracy = accuracy

            print(f"Epoch: {epoch}, Accuracy: {accuracy}")

        return best_accuracy

    def predict(self, x: torch.tensor) -> torch.tensor:
        """
        Predicts the output of the neural network

        Args:
            x (torch.tensor): The input data (shape: (input_size, 1))

        Returns:
            output: The output of the neural network
        """    

        # Forward pass
        phi_list, g_phi_list = self.forward(x)

        # Output layer
        output = softmax(phi_list[-1] [0])

        # Return label
        return torch.argmax(output)

print("Loading data...")

# Dataset
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Split into training and validation
train_data, validation_data = random_split(dataset = train_data, lengths = [0.8, 0.2])

# DataLoader
train_loader = DataLoader(dataset = train_data, batch_size = 1, shuffle = True)
validation_loader = DataLoader(dataset = validation_data, batch_size = 1, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = 1, shuffle = False)
print("Done.")
# Model
model = NeuralNetwork(input_size=784, hidden_sizes=[100], output_sizes=10, learning_rate=0.000001, device = device)

# Train model
acc = model.train(train_loader, validation_loader, epochs=10)
print("Finished training, best accuracy:",acc)

y_true_list = []
y_pred_list = []
# Test on test data
for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader, desc = "Calculating accuracy...")):

    # Inputs and labels
    inputs, labels = inputs.to(device), labels.to(device)
    inputs = torch.flatten(inputs, 1)

    y_true_list.append(labels.item())
    y_pred_list.append(model.predict(inputs).item())

# Calculate accuracy
accuracy = accuracy_score(y_true_list, y_pred_list) * 100

print("Accuracy on test data:", accuracy)