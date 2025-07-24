import flwr as fl
from flwr.client import NumPyClient
# IMPORTANT: Remove or comment out 'from flwr.common import Context' if you're sticking
# to `start_simulation` with `cid: str`. The Context object is for different setups.
import pandas as pd
import numpy as np
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time
import random
import collections
from typing import Dict, Tuple, Optional # Import for type hints in evaluate_fn

# -------------------------------
# Configuration
# -------------------------------
NUM_CLIENTS = 10
BATCH_SIZE = 32
EPOCHS = 5 # Increased epochs for better learning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIENT_PROFILES = [
    {"speed": random.uniform(0.5, 2.0), "latency": random.uniform(0.1, 1.0)}
    for _ in range(NUM_CLIENTS)
]

# -------------------------------
# Model Definition
# -------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# Data Partitioning
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # <--- Global testset for server

def partition_dataset(dataset, num_clients):
    partition_size = len(dataset) // num_clients
    return [list(range(i * partition_size, (i + 1) * partition_size)) for i in range(num_clients)]

train_partitions = partition_dataset(trainset, NUM_CLIENTS)

# -------------------------------
# Flower Client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    # 'cid_str' is the string ID passed directly from client_fn
    def __init__(self, cid_str: str, train_loader, model, profile):
        self.cid = int(cid_str) # Convert to int for internal use (e.g., indexing)
        self.model = model
        self.train_loader = train_loader
        self.profile = profile
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        time.sleep(self.profile['latency'])
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        time.sleep(self.profile['speed'])

        for _ in range(EPOCHS):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

        correct, total = 0, 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.train_loader: # Evaluating on TRAIN loader
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0 # Handle division by zero
        #individual local accuracy
        data1 = pd.DataFrame([[accuracy]],columns=['Accuracy'])
        data1.to_csv('random_Fedavg2.csv', mode='a', index=False, header=False)
        return self.get_parameters({}), len(self.train_loader.dataset), {"accuracy": accuracy}
        

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.train_loader: # Evaluating on TRAIN loader
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / total if total > 0 else 0.0 # Calculate average loss
        accuracy = correct / total if total > 0 else 0.0
        #Calculation of each Node accuracy and average loss at each round
        data1 = pd.DataFrame([[accuracy,avg_loss]],columns=['Accuracy','Average Loss'])
        data1.to_csv('random_Fedavg1.csv', mode='a', index=False, header=False)
        return avg_loss, total, {"accuracy": accuracy}
       

# -------------------------------
# Client Function (Corrected for `start_simulation`)
# -------------------------------
def client_fn(cid: str) -> fl.client.Client: # <--- CORRECTED: Takes 'cid' directly
    
    client_id_as_int = int(cid) # Convert cid string to int for indexing

    indices = train_partitions[client_id_as_int] 
    client_data = Subset(trainset, indices)
    train_loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
    model = CNN().to(DEVICE)
    
    profile = CLIENT_PROFILES[client_id_as_int]
    
    return FlowerClient(cid, train_loader, model, profile).to_client()

# -------------------------------
# Metric Aggregation Functions (Robustified)
# -------------------------------
def fit_metrics_aggregation_fn(results: list[Tuple[fl.common.Parameters, int, Dict[str, fl.common.Scalar]]]):
    try:
        # results is a list of (parameters, num_examples, metrics)
        accuracies = [
            res[2]["accuracy"]
            for res in results
            if len(res) > 2 and isinstance(res[2], dict) and "accuracy" in res[2]
        ]
        
        return {"avg_accuracy": float(np.mean(accuracies)) if accuracies else 0.0}
    except Exception as e:
        print(f"Aggregation error in fit_metrics_aggregation_fn: {e}")
        return {"avg_accuracy": 0.0}


def evaluate_metrics_aggregation_fn(results: list[Tuple[float, int, Dict[str, fl.common.Scalar]]]):
    try:
        # results is a list of (loss, num_examples, metrics)
        accuracies = [
            res[2]["accuracy"]
            for res in results
            if len(res) > 2 and isinstance(res[2], dict) and "accuracy" in res[2]
        ]
        return {"accuracy": float(np.mean(accuracies)) if accuracies else 0.0}
    except Exception as e:
        print(f"Aggregation error in evaluate_metrics_aggregation_fn: {e}")
        return {"accuracy": 0.0}

# -------------------------------
# Centralized Evaluation Function for Server
# -------------------------------
def get_evaluate_fn(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    """Returns an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Load model parameters to the server's global model
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(state_dict)
        model.eval()

        correct, total, loss = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
#Centralized data
        print(f"Server-side evaluation: Round {server_round}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        data = pd.DataFrame([[server_round,avg_loss,accuracy]],columns=['Server Round','Average Loss','Accuracy'])
        data.to_csv('random_Fedavg.csv', mode='a', index=False, header=False)
        return avg_loss, {"accuracy": accuracy}

    return evaluate


# -------------------------------
# Strategy (UPDATED with evaluate_fn and initial_parameters)
# -------------------------------
# Instantiate a model for the server to use for centralized evaluation
global_model = CNN().to(DEVICE)
global_test_loader = DataLoader(testset, batch_size=BATCH_SIZE) # Use the globally loaded testset

strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5, # Clients also perform evaluation and report
    min_fit_clients=5,
    min_available_clients=NUM_CLIENTS,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    evaluate_fn=get_evaluate_fn(global_model, global_test_loader, DEVICE), # <--- Centralized evaluation
    # Provide initial parameters for the server's strategy
    initial_parameters=fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in global_model.state_dict().values()]
    ),
)


# -------------------------------
# Simulation
# -------------------------------
def main():
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=2),
        client_resources= {"num_cpus": 1, "num_gpus": 0.25},
        strategy=strategy,
    )

    print("\n--- Final Training Summary ---")

    # Always check if the attribute exists before accessing, especially for distributed metrics
    # which can vary based on strategy and what clients report.

    # Centralized Metrics (these are usually reliable and exist)
    print("History (loss, centralized):", history.losses_centralized)
    if 'accuracy' in history.metrics_centralized:
        print("History (metrics, centralized, accuracy):", history.metrics_centralized['accuracy'])
        df_centralized_acc = pd.DataFrame(history.metrics_centralized['accuracy'], columns=['Round', 'Accuracy'])
        df_centralized_acc.to_csv('centralized_model_accuracy.csv', index=False)
        print("Centralized model accuracy saved to centralized_model_accuracy.csv")


    # Distributed Losses (this one should exist)
    print("History (loss, distributed):", history.losses_distributed)
    df_distributed_loss = pd.DataFrame(history.losses_distributed, columns=['Round', 'Loss'])
    df_distributed_loss.to_csv('distributed_evaluation_loss.csv', index=False)
    print("Distributed evaluation loss saved to distributed_evaluation_loss.csv")

    # Distributed Fit Metrics (access by key, as you saw 'avg_accuracy' in your INFO logs)
    # Check for the *key* in the dictionary, not an attribute name
    if 'avg_accuracy' in history.metrics_distributed_fit:
        print("History (metrics, distributed, fit, avg_accuracy):", history.metrics_distributed_fit['avg_accuracy'])
        df_dist_fit_acc = pd.DataFrame(history.metrics_distributed_fit['avg_accuracy'], columns=['Round', 'Accuracy'])
        df_dist_fit_acc.to_csv('distributed_fit_accuracy.csv', index=False)
        print("Distributed fit accuracy saved to distributed_fit_accuracy.csv")


if __name__ == "__main__":
    main()