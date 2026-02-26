#!/usr/bin/env python3
"""
PyTorch Profiling Basics - Learning Exercise 1

This example teaches the fundamentals of PyTorch profiling, including:
- Using torch.profiler for performance analysis
- Identifying bottlenecks in model training
- Understanding GPU utilization patterns
- Memory profiling during training
- Exporting and analyzing profiling results

Run this to understand why your model is slow and where to optimize.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

# Import our learning modules
import sys
sys.path.append('..')
from pytorch_learning.core.memory import memory_profile, MemoryManager
from pytorch_learning.core.profiling import Profiler, BottleneckAnalyzer


class SimpleModel(nn.Module):
    """A simple model for profiling demonstrations."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


class InefficientModel(nn.Module):
    """An intentionally inefficient model to demonstrate profiling benefits."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, num_classes: int = 10):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, num_classes)
        
        # Add some inefficient operations
        self.useless_tensor = torch.randn(hidden_size, hidden_size)
        
    def forward(self, x):
        # Inefficient: multiple separate operations instead of sequential
        x = self.layer1(x)
        x = torch.relu(x)
        
        # Inefficient: unnecessary tensor operations
        x = x + 0  # Useless addition
        x = x * 1  # Useless multiplication
        
        x = self.layer2(x)
        x = torch.relu(x)
        
        # Inefficient: creating new tensors in forward pass
        noise = torch.randn_like(x) * 0.01
        x = x + noise
        
        x = self.layer3(x)
        x = torch.relu(x)
        
        # Inefficient: matrix multiplication with useless tensor
        useless_op = torch.mm(x, self.useless_tensor[:x.size(0)])
        x = x + useless_op * 0.001
        
        x = self.layer4(x)
        return x


def create_dummy_data(batch_size: int = 32, input_size: int = 784, num_batches: int = 10):
    """Create dummy training data."""
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_size)
        y = torch.randint(0, 10, (batch_size,))
        yield x, y


def basic_profiling_demo():
    """Demonstrate basic PyTorch profiling."""
    print("=== Basic PyTorch Profiling Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create profiler
    profiler = Profiler(device)
    
    # Profile training step
    with profiler.profile_training_step(model, optimizer, criterion, "basic_step"):
        # Simulate one training step
        x, y = next(create_dummy_data())
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    # Analyze results
    analysis = profiler.analyze_performance()
    print("Performance Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")


def memory_profiling_demo():
    """Demonstrate memory profiling during training."""
    print("\n=== Memory Profiling Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Profile memory usage
    with memory_profile("training_step"):
        for batch_idx, (x, y) in enumerate(create_dummy_data(num_batches=5)):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if batch_idx == 2:
                # Create some temporary tensors to show memory spikes
                temp_tensors = [torch.randn(1000, 1000, device=device) for _ in range(3)]
                del temp_tensors


def bottleneck_analysis_demo():
    """Demonstrate bottleneck identification."""
    print("\n=== Bottleneck Analysis Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Compare efficient vs inefficient models
    efficient_model = SimpleModel().to(device)
    inefficient_model = InefficientModel().to(device)
    
    analyzer = BottleneckAnalyzer(device)
    
    # Profile both models
    results = {}
    
    for name, model in [("efficient", efficient_model), ("inefficient", inefficient_model)]:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\nProfiling {name} model...")
        
        # Profile multiple steps
        start_time = time.time()
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for batch_idx, (x, y) in enumerate(create_dummy_data(num_batches=10)):
                x, y = x.to(device), y.to(device)
                
                with record_function(f"batch_{batch_idx}"):
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
        
        end_time = time.time()
        
        # Collect results
        results[name] = {
            "total_time": end_time - start_time,
            "profiler": prof,
            "key_averages": prof.key_averages(),
            "summary": prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=10)
        }
        
        print(f"Total time: {end_time - start_time:.3f}s")
        print(results[name]["summary"])
    
    # Compare results
    print("\n=== Comparison ===")
    efficient_time = results["efficient"]["total_time"]
    inefficient_time = results["inefficient"]["total_time"]
    speedup = inefficient_time / efficient_time
    
    print(f"Efficient model: {efficient_time:.3f}s")
    print(f"Inefficient model: {inefficient_time:.3f}s")
    print(f"Speedup: {speedup:.2f}x")
    
    # Find bottlenecks in inefficient model
    print("\nTop bottlenecks in inefficient model:")
    inefficient_averages = results["inefficient"]["key_averages"]
    top_ops = sorted(inefficient_averages, key=lambda x: x.cuda_time_total if device.type == "cuda" else x.cpu_time_total, reverse=True)[:5]
    
    for i, op in enumerate(top_ops, 1):
        time_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(f"  {i}. {op.key}: {getattr(op, time_key):.3f}ms")


def advanced_profiling_demo():
    """Demonstrate advanced profiling techniques."""
    print("\n=== Advanced Profiling Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Advanced profiling with custom schedule
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=1,      # Skip first step
            warmup=1,    # Warmup for 1 step
            active=3,    # Profile 3 steps
            repeat=2     # Repeat 2 times
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        
        for batch_idx, (x, y) in enumerate(create_dummy_data(num_batches=10)):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            prof.step()
    
    # Export results for analysis
    prof.export_chrome_trace("advanced_profile.json")
    
    # Memory timeline analysis
    if device.type == "cuda":
        memory_summary = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)
        print("\nMemory Usage Summary:")
        print(memory_summary)


def optimization_suggestions_demo():
    """Demonstrate how profiling leads to optimization suggestions."""
    print("\n=== Optimization Suggestions Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a model with specific inefficiencies
    model = InefficientModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Profile and analyze
    analyzer = BottleneckAnalyzer(device)
    suggestions = analyzer.analyze_and_suggest(model, optimizer, criterion)
    
    print("Optimization Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")


def plot_profiling_results():
    """Create visualizations from profiling data."""
    print("\n=== Profiling Visualization ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    
    # Collect timing data
    timings = []
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        
        for batch_idx, (x, y) in enumerate(create_dummy_data(num_batches=20)):
            x, y = x.to(device), y.to(device)
            
            start = time.time()
            outputs = model(x)
            end = time.time()
            
            timings.append(end - start)
    
    # Create simple plot
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(timings)
        plt.title("Forward Pass Timing per Batch")
        plt.xlabel("Batch")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.savefig("timing_analysis.png")
        print("Timing analysis plot saved as 'timing_analysis.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")


def main():
    """Run all profiling demonstrations."""
    print("PyTorch Profiling Learning Module")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\nStarting profiling demonstrations...\n")
    
    # Run all demos
    basic_profiling_demo()
    memory_profiling_demo()
    bottleneck_analysis_demo()
    advanced_profiling_demo()
    optimization_suggestions_demo()
    plot_profiling_results()
    
    print("\n" + "=" * 50)
    print("Profiling demonstrations complete!")
    print("\nKey takeaways:")
    print("1. Use torch.profiler to identify performance bottlenecks")
    print("2. Profile memory usage to catch OOM issues early")
    print("3. Compare efficient vs inefficient implementations")
    print("4. Use profiling results to guide optimization efforts")
    print("5. Export traces for deeper analysis with Chrome trace viewer")


if __name__ == "__main__":
    main()
