"""
PyTorch profiling utilities for performance analysis and optimization.

This module provides comprehensive profiling tools for identifying bottlenecks,
analyzing performance, and optimizing PyTorch models.
"""

import torch
import time
import contextlib
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import warnings


class Profiler:
    """Advanced PyTorch profiler with customizable tracing and analysis."""
    
    def __init__(self, 
                 use_cuda: bool = True,
                 record_shapes: bool = True,
                 profile_memory: bool = True,
                 with_stack: bool = False,
                 with_flops: bool = True):
        """
        Initialize profiler.
        
        Args:
            use_cuda: Enable CUDA profiling
            record_shapes: Record tensor shapes
            profile_memory: Profile memory usage
            with_stack: Include stack traces
            with_flops: Count FLOPs
        """
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self._profiler = None
        
    def __enter__(self):
        """Start profiling context."""
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            
        self._profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            on_trace_ready=self._trace_handler
        )
        self._profiler.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling context."""
        if self._profiler:
            self._profiler.__exit__(exc_type, exc_val, exc_tb)
            
    def _trace_handler(self, prof):
        """Handle profiler trace output."""
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    def export_chrome_trace(self, path: str):
        """Export profiling results to Chrome trace format."""
        if self._profiler:
            self._profiler.export_chrome_trace(path)


class BottleneckAnalyzer:
    """Analyze and identify performance bottlenecks in PyTorch models."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def analyze_model(self, model: torch.nn.Module, 
                     input_shape: tuple,
                     num_runs: int = 100) -> Dict[str, Any]:
        """
        Analyze model performance and identify bottlenecks.
        
        Args:
            model: PyTorch model to analyze
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with performance metrics and bottleneck analysis
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
                
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
                
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = num_runs / (end_time - start_time)
        
        # Memory analysis
        if device.type == 'cuda':
            memory_before = torch.cuda.memory_allocated()
            with torch.no_grad():
                _ = model(dummy_input)
            memory_after = torch.cuda.memory_allocated()
            memory_usage = memory_after - memory_before
        else:
            memory_usage = 0
            
        return {
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'memory_usage_mb': memory_usage / (1024**2),
            'device': str(device),
            'input_shape': input_shape
        }
        
    def profile_layers(self, model: torch.nn.Module, 
                      input_shape: tuple) -> Dict[str, Dict[str, float]]:
        """
        Profile individual layers to identify slow operations.
        
        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape
            
        Returns:
            Dictionary mapping layer names to their timing information
        """
        layer_times = {}
        dummy_input = torch.randn(input_shape, device=next(model.parameters()).device)
        
        def create_hook(name):
            def hook(module, input, output):
                start = time.time()
                with torch.no_grad():
                    _ = module(input[0])
                end = time.time()
                layer_times[name] = end - start
            return hook
            
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
                
        # Forward pass
        with torch.no_grad():
            _ = model(dummy_input)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return layer_times


class PerformanceTracker:
    """Track and compare performance metrics over time."""
    
    def __init__(self):
        self.history = []
        
    def record_metrics(self, metrics: Dict[str, Any], label: str = None):
        """
        Record performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            label: Optional label for this measurement
        """
        entry = {
            'timestamp': time.time(),
            'label': label,
            'metrics': metrics
        }
        self.history.append(entry)
        
    def get_history(self, metric_name: str = None) -> List[Dict[str, Any]]:
        """
        Get performance history.
        
        Args:
            metric_name: Optional specific metric to retrieve
            
        Returns:
            List of historical measurements
        """
        if metric_name:
            return [
                {**entry, 'metrics': {metric_name: entry['metrics'].get(metric_name)}}
                for entry in self.history
                if metric_name in entry['metrics']
            ]
        return self.history
        
    def compare_runs(self, run1_label: str, run2_label: str) -> Dict[str, Dict[str, float]]:
        """
        Compare two performance runs.
        
        Returns:
            Dictionary comparing metrics between runs
        """
        run1 = next((entry for entry in self.history if entry['label'] == run1_label), None)
        run2 = next((entry for entry in self.history if entry['label'] == run2_label), None)
        
        if not run1 or not run2:
            raise ValueError("Both runs must exist in history")
            
        comparison = {}
        metrics1, metrics2 = run1['metrics'], run2['metrics']
        
        for key in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 != 0:
                    change_pct = ((val2 - val1) / val1) * 100
                else:
                    change_pct = float('inf') if val2 != 0 else 0
                    
                comparison[key] = {
                    'run1': val1,
                    'run2': val2,
                    'change_pct': change_pct,
                    'improvement': change_pct < 0 if 'time' in key.lower() else change_pct > 0
                }
                
        return comparison
        
    def export_report(self, filepath: str):
        """Export performance report to file."""
        import json
        
        report = {
            'summary': {
                'total_runs': len(self.history),
                'time_range': {
                    'start': min(entry['timestamp'] for entry in self.history),
                    'end': max(entry['timestamp'] for entry in self.history)
                }
            },
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


@contextlib.contextmanager
def profile_function(func: Callable, *args, **kwargs):
    """
    Context manager to profile a function execution.
    
    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Yields:
        Dictionary with timing information
    """
    start_time = time.time()
    start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    try:
        result = func(*args, **kwargs)
        yield result
    finally:
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        profiling_info = {
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'function_name': func.__name__ if hasattr(func, '__name__') else str(func)
        }
        
        print(f"Function {profiling_info['function_name']}:")
        print(f"  Execution time: {profiling_info['execution_time']:.4f}s")
        if torch.cuda.is_available():
            print(f"  Memory delta: {profiling_info['memory_delta'] / (1024**2):.2f} MB")
