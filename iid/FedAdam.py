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

# CPU/GPU resources per client actor for Ray simulation backend
# Adjust based on your system's capabilities
# For GPU, 1.0 means one client gets full GPU, 0.0 means CPU only.
# Fractional (e.g., 0.25) can be used for scheduling but doesn't mean true concurrent sharing on one GPU.
CLIENT_RESOURCES = {"num_cpus": 5, "num_gpus": 1.0} # Start with 0.0 GPU if unsure


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


def partition_dataset(dataset, num_clients):
    partition_size = len(dataset) // num_clients
    # Ensure all data is distributed, handle remainder
    indices_per_client = [list(range(i * partition_size, (i + 1) * partition_size)) for i in range(num_clients)]
    # Distribute any remaining samples to the last client
    if len(dataset) % num_clients != 0:
        indices_per_client[-1].extend(range(NUM_CLIENTS * partition_size, len(dataset)))
    return indices_per_client

train_partitions = partition_dataset(trainset, NUM_CLIENTS)



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
            pass
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
        straggler_rate = random.uniform(0.1, 0.9)
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
    gpu_usages = []
    print("we need to debug this fffffffffiiiiiiiiitttttttt",results)

    for res in results:
        print(res[1]["accuracy"])
           
        accuracies.append(res[1]["accuracy"])
            
        cpu_usages.append(res[1]["cpu_usage_percent"])
            
        gpu_usages.append(res[1]["gpu_usage_percent"])

    print("this is acaaaaaacuracies in evaluate metric aggregation",accuracies)
    print("this is cpu_usages in evaluate metric aggregation",cpu_usages)
    print("this is gpu_usages in evaluate metric aggregation",gpu_usages)
    
    aggregated_metrics = {}
    if accuracies:
        aggregated_metrics["avg_accuracy"] = float(np.mean(accuracies))
    else:
        aggregated_metrics["avg_accuracy"] = 0.0 # Still debug this to get non-zero
        
    if cpu_usages:
        aggregated_metrics["avg_cpu_usage_percent"] = float(np.mean(cpu_usages))
    else:
        aggregated_metrics["avg_cpu_usage_percent"] = 0.0
    if gpu_usages:
        aggregated_metrics["avg_gpu_usage_percent"] = float(np.mean(gpu_usages))
    else:
        aggregated_metrics["avg_gpu_usage_percent"] = 1.55

    return aggregated_metrics

def evaluate_metrics_aggregation_fn(results: List[Tuple[float, int, Dict[str, fl.common.Scalar]]]):
    accuracies = []
    cpu_usages = []
    gpu_usages = []
    print("evaluateeeeeeeeeeee")
    for res in results:
        print(res[1]["accuracy"])
           
        accuracies.append(res[1]["accuracy"])
            
        cpu_usages.append(res[1]["cpu_usage_percent"])
            
        gpu_usages.append(res[1]["gpu_usage_percent"])
    print("this is acaaaaaacuracies in evaluate metric aggregation",accuracies)
    print("this is cpu_usages in evaluate metric aggregation",cpu_usages)
    print("this is gpu_usages in evaluate metric aggregation",gpu_usages)
    
    aggregated_metrics = {}
    if accuracies:
        aggregated_metrics["accuracy"] = float(np.mean(accuracies))
    else:
        aggregated_metrics["accuracy"] = 0.0
        
    if cpu_usages:
        aggregated_metrics["avg_cpu_usage_percent"] = float(np.mean(cpu_usages))
    else:
        aggregated_metrics["avg_cpu_usage_percent"] = 0.0
    if gpu_usages:
        aggregated_metrics["avg_gpu_usage_percent"] = float(np.mean(gpu_usages))
    else:
        aggregated_metrics["avg_gpu_usage_percent"] = 1.55
    

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
# FedAdam
# -------------------------------
class FedAdam(fl.server.strategy.FedOpt):
    """FedAdam - Adaptive Federated Optimization using Adam.   """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
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
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
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
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep



def aggregate_fit(
    self,
    server_round: int,
    results: list[tuple[ClientProxy, FitRes]],
    failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
) -> tuple[Optional[Parameters], dict[str, Scalar]]:
    """Aggregate fit results using FedAdam."""
    # This part remains the same
    if not results:
        return None, {}
    
    # This call to super().aggregate_fit() will return the FedAvg aggregated model
    # which is what we need to compute the pseudo-gradient.
    aggregated_params, metrics_aggregated = super().aggregate_fit(
        server_round=server_round, results=results, failures=failures
    )
    if aggregated_params is None:
        return None, {}

    aggregated_weights = parameters_to_ndarrays(aggregated_params)

    # Calculate pseudo-gradient
    # Note: current_weights are the weights from the previous round
    delta_t = [
        x - y for x, y in zip(aggregated_weights, self.current_weights)
    ]
    
    # Adam moment updates (your implementation was correct)
    # m_t
    if not self.m_t:
        self.m_t = [np.zeros_like(x) for x in delta_t]
    self.m_t = [
        np.multiply(self.beta_1, m) + (1 - self.beta_1) * d
        for m, d in zip(self.m_t, delta_t)
    ]

    # v_t
    if not self.v_t:
        self.v_t = [np.zeros_like(x) for x in delta_t]
    self.v_t = [
        np.multiply(self.beta_2, v) + (1 - self.beta_2) * np.multiply(d, d)
        for v, d in zip(self.v_t, delta_t)
    ]


    t = server_round 
    m_hat = [
        m / (1.0 - self.beta_1**t) for m in self.m_t
    ]
    v_hat = [
        v / (1.0 - self.beta_2**t) for v in self.v_t
    ]

    # Server-side update
    new_weights = [
        w + self.eta * m / (np.sqrt(v) + self.tau)
        for w, m, v in zip(self.current_weights, m_hat, v_hat)
    ]

    self.current_weights = new_weights
    return ndarrays_to_parameters(self.current_weights), metrics_aggregated

  


# -------------------------------
# Strategy Instantiation
# -------------------------------
global_model = CNN().to(DEVICE)
global_test_loader = DataLoader(testset, batch_size=BATCH_SIZE)

strategy = FedAdam( 
    eta=0.001,  # A much smaller server-side learning rate
    eta_l=0.01, # This is the client-side learning rate FedAdam suggests, but it's often better to set this in your client code
    beta_1=0.9,
    beta_2=0.99,
    tau=1e-9,
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_available_clients=NUM_CLIENTS,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
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


    # # Distributed Evaluate Metrics (including average CPU usage)
    # # The `evaluate_metrics_aggregation_fn` aggregates metrics reported by clients during their `evaluate` call.
    # if 'accuracy' in history.metrics_distributed_evaluate:
    #     print("History (metrics, distributed, evaluate, accuracy):", history.metrics_distributed_evaluate['accuracy'])
    #     df_dist_eval_acc = pd.DataFrame(history.metrics_distributed_evaluate['accuracy'], columns=['Round', 'Accuracy'])
    #     df_dist_eval_acc.to_csv('distributed_eval_accuracy.csv', index=False)
    #     print("Distributed evaluate accuracy saved to distributed_eval_accuracy.csv")

    # if 'avg_gpu_usage_percent' in history.metrics_distributed_evaluate:
    #     print("History (metrics, distributed, fit, avg_gpu_usage_percent):", history.metrics_distributed_evaluate['avg_gpu_usage_percent'])
    #     df_dist_fit_gpu = pd.DataFrame(history.metrics_distributed_evaluate['avg_gpu_usage_percent'], columns=['Round', 'Avg_GPU_Usage_Percent'])
    #     df_dist_fit_gpu.to_csv('distributed_evaluate_gpu_usage.csv', index=False)
    #     print("Distributed fit GPU usage saved to distributed_evaluate_gpu_usage.csv")

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