#!/usr/bin/env python3
"""
GPU Memory Management - Learning Exercise 2

This example teaches critical GPU memory management concepts:
- Understanding CUDA memory allocation patterns
- Detecting and preventing OOM errors
- Memory optimization techniques
- Peak memory usage tracking
- Emergency cleanup strategies
- Mixed precision memory savings

Run this to understand why your model OOMs at 3am and how to prevent it.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gc
import time
import psutil
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import numpy as np

# Import our learning modules
import sys
sys.path.append('..')
from pytorch_learning.core.memory import (
    MemoryManager, GPUMonitor, OOMDebugger, memory_profile
)


class MemoryHungryModel(nn.Module):
    """A model that uses significant GPU memory for demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [1024, 512, 256]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, 10)
        
    def forward(self, x):
        x = self.layers(x)
        return self.classifier(x)


class GradientAccumulationTrainer:
    """Demonstrates gradient accumulation to reduce memory usage."""
    
    def __init__(self, model: nn.Module, accumulation_steps: int = 4):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def train_step(self, batch, optimizer, criterion):
        """Perform one training step with gradient accumulation."""
        x, y = batch
        outputs = self.model(x)
        loss = criterion(outputs, y)
        
        # Normalize loss to account for accumulation
        loss = loss / self.accumulation_steps
        
        loss.backward()
        
        self.current_step += 1
        
        if self.current_step % self.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            self.current_step = 0
            
        return loss.item() * self.accumulation_steps


def memory_allocation_patterns_demo():
    """Demonstrate different memory allocation patterns."""
    print("=== Memory Allocation Patterns Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping demo")
        return
    
    monitor = GPUMonitor(device)
    initial_stats = monitor.get_current_stats()
    
    print(f"Initial memory: {initial_stats.allocated:.2f}GB allocated, {initial_stats.free:.2f}GB free")
    
    # Pattern 1: Sequential allocation
    print("\n1. Sequential allocation:")
    tensors = []
    for i in range(5):
        tensor = torch.randn(1000, 1000, device=device)
        tensors.append(tensor)
        stats = monitor.get_current_stats()
        print(f"   Tensor {i+1}: {stats.allocated:.2f}GB allocated")
    
    # Pattern 2: Batch allocation
    print("\n2. Batch allocation:")
    del tensors  # Clear previous
    torch.cuda.empty_cache()
    
    batch_tensors = [torch.randn(1000, 1000, device=device) for _ in range(5)]
    stats = monitor.get_current_stats()
    print(f"   Batch allocation: {stats.allocated:.2f}GB allocated")
    
    # Pattern 3: In-place operations
    print("\n3. In-place vs out-of-place operations:")
    
    # Out-of-place (creates new tensor)
    a = torch.randn(1000, 1000, device=device)
    stats_before = monitor.get_current_stats()
    b = a + 1  # Out-of-place
    stats_after = monitor.get_current_stats()
    print(f"   Out-of-place: +{stats_after.allocated - stats_before.allocated:.3f}GB")
    
    # In-place (reuses memory)
    a = torch.randn(1000, 1000, device=device)
    stats_before = monitor.get_current_stats()
    a.add_(1)  # In-place
    stats_after = monitor.get_current_stats()
    print(f"   In-place: +{stats_after.allocated - stats_before.allocated:.3f}GB")
    
    # Cleanup
    del batch_tensors, a, b
    torch.cuda.empty_cache()


def oom_simulation_demo():
    """Simulate and debug OOM errors."""
    print("\n=== OOM Simulation & Debugging Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping demo")
        return
    
    debugger = OOMDebugger(device)
    monitor = GPUMonitor(device)
    
    print("Simulating OOM scenarios...")
    
    # Scenario 1: Large tensor allocation
    try:
        stats = monitor.get_current_stats()
        available_gb = stats.free
        
        print(f"Available memory: {available_gb:.2f}GB")
        
        # Try to allocate more than available
        oversize_tensor = torch.randn(
            int(available_gb * 1.5 * 1024**3 // 4),  # 1.5x available memory
            device=device
        )
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            context = debugger.capture_oom_context(e)
            print(f"OOM captured at allocation: {context['memory_stats'].utilization:.1f}% utilization")
    
    # Scenario 2: Model training OOM
    try:
        # Create a very large model
        large_model = nn.Sequential(
            *[nn.Linear(1000, 1000) for _ in range(50)],  # 50 layers
            nn.ReLU()
        ).to(device)
        
        # Try to forward pass large batch
        large_input = torch.randn(32, 1000, device=device)
        output = large_model(large_input)
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            context = debugger.capture_oom_context(e)
            print(f"OOM captured during forward pass: {context['memory_stats'].utilization:.1f}% utilization")
    
    # Analyze OOM patterns
    analysis = debugger.analyze_oom_patterns()
    print(f"\nOOM Analysis: {analysis}")


def memory_optimization_techniques():
    """Demonstrate various memory optimization techniques."""
    print("\n=== Memory Optimization Techniques Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping demo")
        return
    
    monitor = GPUMonitor(device)
    manager = MemoryManager(device)
    
    # Technique 1: Gradient checkpointing
    print("1. Gradient Checkpointing:")
    
    # Create a model
    model = MemoryHungryModel(hidden_sizes=[1024, 512, 256]).to(device)
    
    # Without checkpointing
    x = torch.randn(64, 784, device=device)
    stats_before = monitor.get_current_stats()
    
    # Simulate forward pass
    with torch.no_grad():
        output = model(x)
    
    stats_after = monitor.get_current_stats()
    memory_without_checkpoint = stats_after.allocated - stats_before.allocated
    
    # With checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
        stats_before = monitor.get_current_stats()
        with torch.no_grad():
            output = model(x)
        stats_after = monitor.get_current_stats()
        
        memory_with_checkpoint = stats_after.allocated - stats_before.allocated
        
        savings = (memory_without_checkpoint - memory_with_checkpoint) / memory_without_checkpoint * 100
        print(f"   Memory savings with checkpointing: {savings:.1f}%")
    
    # Technique 2: Mixed precision
    print("\n2. Mixed Precision Training:")
    
    # FP32 baseline
    model_fp32 = MemoryHungryModel().to(device)
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)
    
    with memory_profile("fp32_training"):
        for _ in range(3):
            x = torch.randn(32, 784, device=device)
            y = torch.randint(0, 10, (32,), device=device)
            
            optimizer_fp32.zero_grad()
            output = model_fp32(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optimizer_fp32.step()
    
    # FP16 with automatic mixed precision
    from torch.cuda.amp import autocast, GradScaler
    
    model_fp16 = MemoryHungryModel().to(device)
    optimizer_fp16 = optim.Adam(model_fp16.parameters(), lr=0.001)
    scaler = GradScaler()
    
    with memory_profile("fp16_training"):
        for _ in range(3):
            x = torch.randn(32, 784, device=device)
            y = torch.randint(0, 10, (32,), device=device)
            
            optimizer_fp16.zero_grad()
            
            with autocast():
                output = model_fp16(x)
                loss = nn.CrossEntropyLoss()(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer_fp16)
            scaler.update()
    
    # Technique 3: Gradient accumulation
    print("\n3. Gradient Accumulation:")
    
    # Large batch without accumulation
    model = MemoryHungryModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    large_batch_size = 64
    x_large = torch.randn(large_batch_size, 784, device=device)
    y_large = torch.randint(0, 10, (large_batch_size,), device=device)
    
    with memory_profile("large_batch"):
        optimizer.zero_grad()
        output = model(x_large)
        loss = nn.CrossEntropyLoss()(output, y_large)
        loss.backward()
        optimizer.step()
    
    # Small batch with accumulation
    accumulation_steps = 4
    small_batch_size = large_batch_size // accumulation_steps
    trainer = GradientAccumulationTrainer(model, accumulation_steps)
    
    with memory_profile("gradient_accumulation"):
        for step in range(accumulation_steps):
            x_small = torch.randn(small_batch_size, 784, device=device)
            y_small = torch.randint(0, 10, (small_batch_size,), device=device)
            
            loss = trainer.train_step((x_small, y_small), optimizer, nn.CrossEntropyLoss())


def memory_leak_detection():
    """Demonstrate memory leak detection and prevention."""
    print("\n=== Memory Leak Detection Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping demo")
        return
    
    monitor = GPUMonitor(device)
    monitor.start_monitoring(interval=0.5)
    
    # Simulate memory leak
    print("Simulating memory leak...")
    leaked_tensors = []
    
    for i in range(10):
        # Create tensors but don't delete them (leak)
        tensor = torch.randn(1000, 1000, device=device)
        leaked_tensors.append(tensor)
        
        time.sleep(0.2)
        
        stats = monitor.get_current_stats()
        print(f"  Step {i+1}: {stats.allocated:.2f}GB allocated")
    
    # Detect leak
    timeline = monitor.get_memory_timeline()
    if len(timeline) > 1:
        initial_memory = timeline[0].allocated
        final_memory = timeline[-1].allocated
        memory_growth = final_memory - initial_memory
        
        if memory_growth > 0.5:  # More than 500MB growth
            print(f"⚠️  Memory leak detected: {memory_growth:.2f}GB growth")
    
    # Fix the leak
    print("\nFixing memory leak...")
    del leaked_tensors
    torch.cuda.empty_cache()
    gc.collect()
    
    final_stats = monitor.get_current_stats()
    print(f"After cleanup: {final_stats.allocated:.2f}GB allocated")
    
    monitor.stop_monitoring()


def emergency_cleanup_demo():
    """Demonstrate emergency cleanup procedures."""
    print("\n=== Emergency Cleanup Demo ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping demo")
        return
    
    manager = MemoryManager(device)
    monitor = GPUMonitor(device)
    
    # Fill up memory
    print("Filling GPU memory...")
    tensors = []
    for i in range(20):
        tensor = torch.randn(1000, 1000, device=device)
        tensors.append(tensor)
    
    stats = monitor.get_current_stats()
    print(f"Memory filled: {stats.allocated:.2f}GB allocated, {stats.free:.2f}GB free")
    
    # Emergency cleanup
    print("Performing emergency cleanup...")
    manager.emergency_cleanup()
    
    stats_after = monitor.get_current_stats()
    print(f"After cleanup: {stats_after.allocated:.2f}GB allocated, {stats_after.free:.2f}GB free")
    
    # Memory optimization suggestions
    suggestions = manager.suggest_optimizations()
    print("\nOptimization suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")


def memory_efficiency_comparison():
    """Compare memory efficiency of different approaches."""
    print("\n=== Memory Efficiency Comparison ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping demo")
        return
    
    monitor = GPUMonitor(device)
    
    approaches = {
        "baseline": lambda: train_baseline(device),
        "mixed_precision": lambda: train_mixed_precision(device),
        "gradient_accumulation": lambda: train_with_accumulation(device),
        "checkpointing": lambda: train_with_checkpointing(device),
    }
    
    results = {}
    
    for approach_name, approach_func in approaches.items():
        print(f"\nTesting {approach_name}...")
        
        # Clear memory before each test
        torch.cuda.empty_cache()
        gc.collect()
        
        initial_stats = monitor.get_current_stats()
        
        try:
            approach_func()
            final_stats = monitor.get_current_stats()
            memory_used = final_stats.max_allocated - initial_stats.max_allocated
            results[approach_name] = memory_used
            print(f"  Peak memory usage: {memory_used:.2f}GB")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[approach_name] = float('inf')
    
    # Compare results
    print("\n=== Memory Efficiency Comparison ===")
    baseline = results.get("baseline", float('inf'))
    
    for approach, memory_used in results.items():
        if memory_used != float('inf') and baseline != float('inf'):
            savings = (baseline - memory_used) / baseline * 100
            print(f"{approach:20}: {memory_used:.2f}GB ({savings:+.1f}% vs baseline)")


def train_baseline(device):
    """Baseline training approach."""
    model = MemoryHungryModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for _ in range(5):
        x = torch.randn(32, 784, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()


def train_mixed_precision(device):
    """Mixed precision training approach."""
    from torch.cuda.amp import autocast, GradScaler
    
    model = MemoryHungryModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    for _ in range(5):
        x = torch.randn(32, 784, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def train_with_accumulation(device):
    """Gradient accumulation approach."""
    model = MemoryHungryModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = GradientAccumulationTrainer(model, accumulation_steps=4)
    
    for _ in range(20):  # 4x more steps to get same total batches
        x = torch.randn(8, 784, device=device)  # Smaller batch
        y = torch.randint(0, 10, (8,), device=device)
        
        loss = trainer.train_step((x, y), optimizer, nn.CrossEntropyLoss())


def train_with_checkpointing(device):
    """Gradient checkpointing approach."""
    model = MemoryHungryModel().to(device)
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for _ in range(5):
        x = torch.randn(32, 784, device=device)
        y = torch.randint(0, 10, (32,), device=device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()


def main():
    """Run all memory management demonstrations."""
    print("GPU Memory Management Learning Module")
    print("=" * 50)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  CUDA not available. Some demos will be skipped.")
    
    print("\nStarting memory management demonstrations...\n")
    
    # Run all demos
    memory_allocation_patterns_demo()
    oom_simulation_demo()
    memory_optimization_techniques()
    memory_leak_detection()
    emergency_cleanup_demo()
    memory_efficiency_comparison()
    
    print("\n" + "=" * 50)
    print("Memory management demonstrations complete!")
    print("\nKey takeaways:")
    print("1. Monitor memory usage continuously during development")
    print("2. Use mixed precision training to reduce memory by ~50%")
    print("3. Implement gradient accumulation for large effective batch sizes")
    print("4. Enable gradient checkpointing for memory-intensive models")
    print("5. Have emergency cleanup procedures ready for production")
    print("6. Profile memory usage to identify leaks and optimization opportunities")
    print("7. Test memory limits during development, not in production")


if __name__ == "__main__":
    main()
