import flwr as fl
from flwr.client import NumPyClient
import pandas as pd
import numpy as np
import csv
import torch
from collections import defaultdict
import random
import pynvml
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
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
# For CPU Usage monitoring (client-side, limited scope)
import psutil
from typing import Callable, Optional, Union

import numpy as np

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


# -------------------------------
# Configuration
# -------------------------------
NUM_CLIENTS = 100
BATCH_SIZE = 32
EPOCHS = 5 # Increased epochs for better learning
DEVICE = torch.device("cuda")


CLIENT_RESOURCES = {"num_cpus": 1, "num_gpus": 0.1} # Start with 0.0 GPU if unsure


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
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10) # 10 classes for MNIST digits (0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Match the input size of fc1
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# -------------------------------
# Data Partitioning
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform) # <--- Global testset for server



def partition_dataset_noniid(dataset, num_clients, num_classes_per_client=2):

    num_classes = len(dataset.classes)
    
    # 1. Group all indices by their class label
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    class_assignments = []
    assignments_per_class = (num_clients * num_classes_per_client) // num_classes
    for label in range(num_classes):
        class_assignments.extend([label] * assignments_per_class)
    
    # Fill remaining assignments if division wasn't perfect
    remaining = (num_clients * num_classes_per_client) - len(class_assignments)
    class_assignments.extend(np.random.choice(range(num_classes), remaining, replace=False))
    
    random.shuffle(class_assignments)

    # Assign `num_classes_per_client` to each client from the shuffled list
    client_to_labels = defaultdict(list)
    for client_id in range(num_clients):
        start = client_id * num_classes_per_client
        end = start + num_classes_per_client
        client_to_labels[client_id] = class_assignments[start:end]

    # 3. Create a map from a label to the clients that have it
    label_to_clients = defaultdict(list)
    for client_id, labels in client_to_labels.items():
        for label in labels:
            label_to_clients[label].append(client_id)

    # 4. Partition the data indices for each label among its assigned clients
    client_indices = [[] for _ in range(num_clients)]
    for label, indices in label_to_indices.items():
        assigned_clients = label_to_clients[label]
        
        # If a label has no assigned clients (can happen in some configs), continue
        if not assigned_clients:
            continue
            
        # Shuffle indices for randomness
        random.shuffle(indices)
        
        # Split indices among the assigned clients
        num_chunks = len(assigned_clients)
        chunks = np.array_split(indices, num_chunks)
        
        for i, client_id in enumerate(assigned_clients):
            client_indices[client_id].extend(chunks[i])

    return client_indices


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
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError_AlreadyInitialized:
            pass  # Safe to ignore
        except pynvml.NVMLError as e:
            print(f"NVML init failed: {e}")

    def get_parameters(self, config):
        time.sleep(self.profile['latency'])
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        straggler_rate=random.uniform(0.1, 0.9)
        print(f"we should see straggler rate at {straggler_rate}")
        if random.random() < straggler_rate:
            print(f"client{self.cid} is a straggler this round.")
            time.sleep(5)
            raise Exception(f"client{self.cid} is a straggler this round.")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        time.sleep(self.profile['speed'])
        process = psutil.Process()


        for _ in range(EPOCHS):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()

        util_after = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = util_after.gpu
        cpu_usage = process.cpu_percent(interval=1.0)

        correct, total = 0, 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0.0

        return self.get_parameters({}), len(self.train_loader.dataset), {
            "accuracy": accuracy,
            "cpu_usage_percent": cpu_usage,
            "gpu_usage_percent": gpu_usage
        }


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        # GPU usage before is optional; can ignore
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) 

        # Start CPU usage monitoring (will re-sample after evaluation)
        process = psutil.Process()

        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Sample GPU usage *after* evaluation as a proxy
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = util.gpu

        # Get CPU usage with a short sampling interval
        cpu_usage = process.cpu_percent(interval=1.0)

        avg_loss = loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        print("GPU usage (%):", gpu_usage, "CPU usage (%):", cpu_usage)

        return avg_loss, total, {
            "accuracy": accuracy,
            "cpu_usage_percent": cpu_usage,
            "gpu_usage_percent": gpu_usage,
        }


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
        all_preds = []
        all_labels = []


        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                correct += (predicted == labels).sum().item()
        
        avg_loss = loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)


        print(f"Server-side evaluation: Round {server_round}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
}
    return evaluate


# -------------------------------
# Custom Strategy for Round Timing
# -------------------------------
class FedYogi(fl.server.strategy.FedOpt):
    """FedYogi [Reddi et al., 2020] strategy."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                Optional[tuple[float, dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedYogi(accept_failures={self.accept_failures})"
        return rep



    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        # Yogi
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated


# -------------------------------
# Strategy Instantiation
# -------------------------------
global_model = CNN().to(DEVICE)
global_test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

strategy = FedYogi( # Use your custom strategy here
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(global_model, global_test_loader, DEVICE),
    initial_parameters=fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for val in global_model.state_dict().values()]
    ),
    accept_failures=True,

)

# -------------------------------
# Simulation
# -------------------------------
def main():
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10), # Set a reasonable number of rounds for testing
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

    print("History (precision, centralized):", history.losses_centralized)
    if "precision" in history.metrics_centralized:
        print("History (metrics, centralized, precision):", history.metrics_centralized['precision'])
        df_centralized_precision=pd.DataFrame(history.metrics_centralized['precision'])
        df_centralized_precision.to_csv('centralized_model_precision.csv', index=False)
        print("Centralized model precision saved to centralized_model_precision.csv")
        
    print("History (f1, centralized):", history.losses_centralized)
    if "f1" in history.metrics_centralized:
        print("History (metrics, centralized, f1):", history.metrics_centralized['f1'])
        df_centralized_f1=pd.DataFrame(history.metrics_centralized['f1'])
        df_centralized_f1.to_csv('centralized_model_f1.csv', index=False)
        print("Centralized model precision saved to centralized_model_f1.csv")

    print("History (f1, recall):", history.losses_centralized)
    if "recall" in history.metrics_centralized:
        print("History (metrics, centralized, recall):", history.metrics_centralized['recall'])
        df_centralized_recall=pd.DataFrame(history.metrics_centralized['recall'])
        df_centralized_recall.to_csv('centralized_model_recall.csv', index=False)
        print("Centralized model precision saved to centralized_model_recall.csv")   
        
    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]

    plt.figure(figsize=(10, 6)) # Optional: Make the plot a bit larger
    plt.plot(round, acc)
    plt.grid(True) # Use True for clarity, though grid() works too
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("Centralized Model Accuracy Over Rounds") # Good practice to add a title
    plt.show() # This line is crucial to display the plot when run as a script

        

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



    if 'avg_cpu_usage_percent' in history.metrics_distributed:
         print("History (metrics, distributed, evaluate, avg_cpu_usage_percent):", history.metrics_distributed['avg_cpu_usage_percent'])
         df_dist_eval_cpu = pd.DataFrame(history.metrics_distributed['avg_cpu_usage_percent'], columns=['Round', 'Avg_CPU_Usage_Percent'])
         df_dist_eval_cpu.to_csv('distributed_evaluation_cpu_usage.csv', index=False)
         print("Distributed evaluate CPU usage saved to distributed_evaluation_cpu_usage.csv")

    if 'avg_gpu_usage_percent' in history.metrics_distributed_fit:
        print("History (metrics, distributed, fit, avg_gpu_usage_percent):", history.metrics_distributed_fit['avg_gpu_usage_percent'])
        df_dist_fit_gpu = pd.DataFrame(history.metrics_distributed_fit['avg_gpu_usage_percent'], columns=['Round', 'Avg_GPU_Usage_Percent'])
        df_dist_fit_gpu.to_csv('distributed_fit_gpu_usage.csv', index=False)
        print("Distributed fit GPU usage saved to distributed_fit_gpu_usage.csv")

       

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