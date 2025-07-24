import flwr as fl
import numpy as np
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
from flwr.common import Context

# -------------------------------
# Configuration
# -------------------------------
NUM_CLIENTS = 10
BATCH_SIZE = 32
EPOCHS = 1
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

def partition_dataset(dataset, num_clients):
    partition_size = len(dataset) // num_clients
    return [list(range(i * partition_size, (i + 1) * partition_size)) for i in range(num_clients)]

partitions = partition_dataset(trainset, NUM_CLIENTS)

# -------------------------------
# Flower Client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, model, profile):
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.profile = profile
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        time.sleep(self.profile['latency'])
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

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
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return self.get_parameters({}), len(self.train_loader.dataset), {"accuracy": correct / total}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return loss / total, total, {"accuracy": correct / total}

# -------------------------------
# Client Function
# -------------------------------
def client_fn(client_id: str):
    cid = int(client_id)
    indices = partitions[cid]
    client_data = Subset(trainset, indices)
    train_loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
    model = CNN().to(DEVICE)
    profile = CLIENT_PROFILES[cid]
    return FlowerClient(cid, train_loader, model, profile).to_client()

# -------------------------------
# Metric Aggregation
# -------------------------------
def fit_metrics_aggregation_fn(results):
    try:
        accuracies = [fit_res.metrics["accuracy"] for fit_res in results if "accuracy" in fit_res.metrics]
        return {"avg_accuracy": float(np.mean(accuracies))}
    except Exception as e:
        print(f"Aggregation error: {e}")
        return {"avg_accuracy": 0.0}


def evaluate_metrics_aggregation_fn(results):
    accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
    return {"accuracy": float(np.mean(accuracies))} if accuracies else {}

# -------------------------------
# Strategy
# -------------------------------
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=5,
    min_available_clients=NUM_CLIENTS,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
)


# -------------------------------
# Simulation
# -------------------------------
def main():
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    print("\n--- Training Summary ---")
    for rnd, metrics in enumerate(history.metrics_centralized.get("fit", []), start=1):
        print(f"Round {rnd} - Fit Accuracy: {metrics['accuracy']:.4f}")
    for rnd, metrics in enumerate(history.metrics_centralized.get("evaluate", []), start=1):
        print(f"Round {rnd} - Eval Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main()
