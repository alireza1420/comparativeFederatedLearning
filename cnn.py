import flwr as fl
from flwr.client import NumPyClient
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
from typing import Dict, Tuple, Optional, List, Union # Import all necessary types

# For CPU Usage monitoring (client-side, limited scope)
import psutil

# -------------------------------
# Configuration
# -------------------------------
NUM_CLIENTS = 10
BATCH_SIZE = 32
EPOCHS = 5 # Increased epochs for better learning
DEVICE = torch.device("cuda")

# CPU/GPU resources per client actor for Ray simulation backend
# Adjust based on your system's capabilities
# For GPU, 1.0 means one client gets full GPU, 0.0 means CPU only.
# Fractional (e.g., 0.25) can be used for scheduling but doesn't mean true concurrent sharing on one GPU.
CLIENT_RESOURCES = {"num_cpus": 2, "num_gpus": 1.0} # Start with 0.0 GPU if unsure

CLIENT_PROFILES = [
    {"speed": random.uniform(0.5, 2.0), "latency": random.uniform(0.1, 1.0)}
    for _ in range(NUM_CLIENTS)
]

# Global storage for round timings
round_times_data = collections.defaultdict(dict) # Stores {round_num: {"start": timestamp, "end": timestamp}}
# Global storage for aggregated client-side CPU usage metrics
client_cpu_usage_metrics = collections.defaultdict(list) # Stores {round_num: [cpu_percent_client1, cpu_percent_client2, ...]}

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
    # Ensure all data is distributed, handle remainder
    indices_per_client = [list(range(i * partition_size, (i + 1) * partition_size)) for i in range(num_clients)]
    # Distribute any remaining samples to the last client
    if len(dataset) % num_clients != 0:
        indices_per_client[-1].extend(range(NUM_CLIENTS * partition_size, len(dataset)))
    return indices_per_client

train_partitions = partition_dataset(trainset, NUM_CLIENTS)

# -------------------------------
# Flower Client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid_str: str, train_loader, model, profile):
        self.cid = int(cid_str)
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

        # Start CPU usage monitoring
        process = psutil.Process()
        cpu_before = process.cpu_percent(interval=None) # Non-blocking

        for _ in range(EPOCHS):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = self.criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
        
        # Calculate CPU usage AFTER training
        cpu_after = process.cpu_percent(interval=None) # Non-blocking
        cpu_diff = cpu_after - cpu_before if cpu_after >= cpu_before else cpu_before - cpu_after # Simple diff

        correct, total = 0, 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.train_loader: # Evaluating on TRAIN loader
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Return CPU usage as a metric
        return self.get_parameters({}), len(self.train_loader.dataset), {"accuracy": accuracy, "cpu_usage_percent": cpu_diff}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        
        # Start CPU usage monitoring
        process = psutil.Process()
        cpu_before = process.cpu_percent(interval=None) # Non-blocking

        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.train_loader: # Evaluating on TRAIN loader
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate CPU usage AFTER evaluation
        cpu_after = process.cpu_percent(interval=None) # Non-blocking
        cpu_diff = cpu_after - cpu_before if cpu_after >= cpu_before else cpu_before - cpu_after # Simple diff

        avg_loss = loss / total if total > 0 else 0.0 # Calculate average loss
        accuracy = correct / total if total > 0 else 0.0
        
        # Return CPU usage as a metric
        return avg_loss, total, {"accuracy": accuracy, "cpu_usage_percent": cpu_diff}

# -------------------------------
# Client Function
# -------------------------------
def client_fn(cid: str) -> fl.client.Client:
    client_id_as_int = int(cid)

    indices = train_partitions[client_id_as_int]
    client_data = Subset(trainset, indices)
    train_loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=True)
    model = CNN().to(DEVICE)
    
    profile = CLIENT_PROFILES[client_id_as_int]
    
    return FlowerClient(cid, train_loader, model, profile).to_client()

# -------------------------------
# Metric Aggregation Functions
# -------------------------------
def fit_metrics_aggregation_fn(results: List[Tuple[fl.common.Parameters, int, Dict[str, fl.common.Scalar]]]):
    accuracies = []
    cpu_usages = []
    for res in results:
        if len(res) > 2 and isinstance(res[2], dict):
            if "accuracy" in res[2]:
                accuracies.append(res[2]["accuracy"])
            if "cpu_usage_percent" in res[2]:
                cpu_usages.append(res[2]["cpu_usage_percent"])
    
    aggregated_metrics = {}
    if accuracies:
        aggregated_metrics["avg_accuracy"] = float(np.mean(accuracies))
    else:
        aggregated_metrics["avg_accuracy"] = 0.0 # Still debug this to get non-zero
        
    if cpu_usages:
        aggregated_metrics["avg_cpu_usage_percent"] = float(np.mean(cpu_usages))
    else:
        aggregated_metrics["avg_cpu_usage_percent"] = 0.0

    return aggregated_metrics

def evaluate_metrics_aggregation_fn(results: List[Tuple[float, int, Dict[str, fl.common.Scalar]]]):
    accuracies = []
    cpu_usages = []
    for res in results:
        if len(res) > 2 and isinstance(res[2], dict):
            if "accuracy" in res[2]:
                accuracies.append(res[2]["accuracy"])
            if "cpu_usage_percent" in res[2]:
                cpu_usages.append(res[2]["cpu_usage_percent"])
    
    aggregated_metrics = {}
    if accuracies:
        aggregated_metrics["accuracy"] = float(np.mean(accuracies))
    else:
        aggregated_metrics["accuracy"] = 0.0
        
    if cpu_usages:
        aggregated_metrics["avg_cpu_usage_percent"] = float(np.mean(cpu_usages))
    else:
        aggregated_metrics["avg_cpu_usage_percent"] = 0.0

    return aggregated_metrics

# -------------------------------
# Centralized Evaluation Function for Server
# -------------------------------
def get_evaluate_fn(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
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
        
        print(f"Server-side evaluation: Round {server_round}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        # Remove CSV writing from here as well, collect from history
        return avg_loss, {"accuracy": accuracy}

    return evaluate


# -------------------------------
# Custom Strategy for Round Timing
# -------------------------------
class TimedFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0

    def configure_fit(
        self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        self.current_round = server_round
        round_times_data[self.current_round]["start"] = time.time()
        print(f"Server Round {self.current_round} started at {round_times_data[self.current_round]['start']:.2f}s")
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            round_times_data[server_round]["end"] = time.time()
            duration = round_times_data[server_round]["end"] - round_times_data[server_round]["start"]
            print(f"Server Round {server_round} completed in {duration:.2f} seconds.")
            # Add duration to metrics if you want it in the History object
            aggregated_metrics["round_duration_s"] = duration
        
        return aggregated_parameters, aggregated_metrics

    # If you also need to time evaluation rounds separately (not just fit rounds that include eval)
    # def configure_evaluate(
    #     self, server_round: int, parameters: fl.common.Parameters, client_manager: fl.server.client_manager.ClientManager
    # ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
    #     # Not typically needed if evaluation is part of the fit cycle or server-side
    #     return super().configure_evaluate(server_round, parameters, client_manager)

    # def aggregate_evaluate(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
    #     failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    # ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
    #     # Only if clients perform dedicated evaluation rounds and report metrics
    #     loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
    #     # Add timing logic here if needed for pure evaluate rounds
    #     return loss_aggregated, metrics_aggregated


# -------------------------------
# Strategy Instantiation
# -------------------------------
global_model = CNN().to(DEVICE)
global_test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

strategy = TimedFedAvg( # Use your custom strategy here
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=5,
    min_available_clients=NUM_CLIENTS,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    evaluate_fn=get_evaluate_fn(global_model, global_test_loader, DEVICE),
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
        config=fl.server.ServerConfig(num_rounds=3), # Set a reasonable number of rounds for testing
        client_resources= CLIENT_RESOURCES, # Use the defined CLIENT_RESOURCES
        strategy=strategy,
    )

    print("\n--- Final Training Summary ---")

    # Centralized Metrics
    print("History (loss, centralized):", history.losses_centralized)
    if 'accuracy' in history.metrics_centralized:
        print("History (metrics, centralized, accuracy):", history.metrics_centralized['accuracy'])
        df_centralized_acc = pd.DataFrame(history.metrics_centralized['accuracy'], columns=['Round', 'Accuracy'])
        df_centralized_acc.to_csv('centralized_model_accuracy.csv', index=False)
        print("Centralized model accuracy saved to centralized_model_accuracy.csv")

    # Distributed Losses
    print("History (loss, distributed):", history.losses_distributed)
    df_distributed_loss = pd.DataFrame(history.losses_distributed, columns=['Round', 'Loss'])
    df_distributed_loss.to_csv('distributed_evaluation_loss.csv', index=False)
    print("Distributed evaluation loss saved to distributed_evaluation_loss.csv")

    # Distributed Fit Metrics (including average CPU usage)
    if 'avg_accuracy' in history.metrics_distributed_fit: # Check for a known key
        print("History (metrics, distributed, fit, avg_accuracy):", history.metrics_distributed_fit['avg_accuracy'])
        df_dist_fit_acc = pd.DataFrame(history.metrics_distributed_fit['avg_accuracy'], columns=['Round', 'Accuracy'])
        df_dist_fit_acc.to_csv('distributed_fit_accuracy.csv', index=False)
        print("Distributed fit accuracy saved to distributed_fit_accuracy.csv")
    
    if 'avg_cpu_usage_percent' in history.metrics_distributed_fit:
        print("History (metrics, distributed, fit, avg_cpu_usage_percent):", history.metrics_distributed_fit['avg_cpu_usage_percent'])
        df_dist_fit_cpu = pd.DataFrame(history.metrics_distributed_fit['avg_cpu_usage_percent'], columns=['Round', 'Avg_CPU_Usage_Percent'])
        df_dist_fit_cpu.to_csv('distributed_fit_cpu_usage.csv', index=False)
        print("Distributed fit CPU usage saved to distributed_fit_cpu_usage.csv")


    # Distributed Evaluate Metrics (including average CPU usage)
    # The `evaluate_metrics_aggregation_fn` aggregates metrics reported by clients during their `evaluate` call.
    if 'accuracy' in history.metrics_distributed_evaluate:
        print("History (metrics, distributed, evaluate, accuracy):", history.metrics_distributed_evaluate['accuracy'])
        df_dist_eval_acc = pd.DataFrame(history.metrics_distributed_evaluate['accuracy'], columns=['Round', 'Accuracy'])
        df_dist_eval_acc.to_csv('distributed_eval_accuracy.csv', index=False)
        print("Distributed evaluate accuracy saved to distributed_eval_accuracy.csv")

    if 'avg_cpu_usage_percent' in history.metrics_distributed:
         print("History (metrics, distributed, evaluate, avg_cpu_usage_percent):", history.metrics_distributed['avg_cpu_usage_percent'])
         df_dist_eval_cpu = pd.DataFrame(history.metrics_distributed['avg_cpu_usage_percent'], columns=['Round', 'Avg_CPU_Usage_Percent'])
         df_dist_eval_cpu.to_csv('distributed_eval_cpu_usage.csv', index=False)
         print("Distributed evaluate CPU usage saved to distributed_eval_cpu_usage.csv")
    else:
         print("Warning: 'avg_cpu_usage_percent' not found in history.metrics_distributed from evaluation.")


    # Round Durations
    print("\n--- Round Durations ---")
    round_duration_list = []
    for round_num, times in sorted(round_times_data.items()):
        if "start" in times and "end" in times:
            duration = times["end"] - times["start"]
            round_duration_list.append({'Round': round_num, 'Duration_s': duration})
            print(f"Round {round_num}: {duration:.2f} seconds")
    
    if round_duration_list:
        df_round_durations = pd.DataFrame(round_duration_list)
        df_round_durations.to_csv('round_durations.csv', index=False)
        print("Round durations saved to round_durations.csv")


if __name__ == "__main__":
    main()