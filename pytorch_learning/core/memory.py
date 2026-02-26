"""
GPU Memory Management for PyTorch

This module teaches the critical aspects of GPU memory management that every
ML engineer needs to understand to avoid OOM errors and optimize performance.

Key concepts covered:
- CUDA memory allocation patterns
- Memory fragmentation and cleanup
- Peak memory usage tracking
- OOM debugging strategies
- Memory-efficient training techniques
"""

import torch
import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics for a single GPU."""
    device_id: int
    allocated: float  # GB
    cached: float     # GB
    max_allocated: float  # GB
    free: float       # GB
    total: float      # GB
    utilization: float  # percentage


class GPUMonitor:
    """Real-time GPU memory monitoring with historical tracking."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_id = self.device.index if self.device.type == "cuda" else 0
        self.history: List[MemoryStats] = []
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics for the GPU."""
        if self.device.type != "cuda":
            raise RuntimeError("GPU monitoring only available on CUDA devices")
            
        allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(self.device_id) / 1024**3      # GB
        max_allocated = torch.cuda.max_memory_allocated(self.device_id) / 1024**3  # GB
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3  # GB
        free_memory = total_memory - cached
        
        return MemoryStats(
            device_id=self.device_id,
            allocated=allocated,
            cached=cached,
            max_allocated=max_allocated,
            free=free_memory,
            total=total_memory,
            utilization=(cached / total_memory) * 100
        )
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started GPU memory monitoring on device {self.device_id}")
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped GPU memory monitoring")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                stats = self.get_current_stats()
                with self.lock:
                    self.history.append(stats)
                    # Keep only last 1000 entries to prevent memory bloat
                    if len(self.history) > 1000:
                        self.history.pop(0)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                break
    
    def get_memory_timeline(self) -> List[MemoryStats]:
        """Get historical memory usage timeline."""
        with self.lock:
            return self.history.copy()
    
    def find_peak_usage(self) -> MemoryStats:
        """Find the peak memory usage in the monitoring history."""
        with self.lock:
            if not self.history:
                return self.get_current_stats()
            return max(self.history, key=lambda x: x.max_allocated)


class MemoryManager:
    """Advanced GPU memory management with intelligent cleanup and optimization."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = GPUMonitor(self.device)
        self.oom_count = 0
        self.cleanup_strategies = [
            self._clear_cache,
            self._force_garbage_collection,
            self._reset_peak_memory_stats
        ]
        
    @contextmanager
    def memory_limit(self, limit_gb: float):
        """Context manager to limit memory usage during execution."""
        if self.device.type != "cuda":
            yield
            return
            
        initial_stats = self.monitor.get_current_stats()
        try:
            yield
        finally:
            current_stats = self.monitor.get_current_stats()
            if current_stats.allocated > limit_gb:
                logger.warning(f"Memory limit exceeded: {current_stats.allocated:.2f}GB > {limit_gb:.2f}GB")
                self.emergency_cleanup()
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup to prevent OOM."""
        logger.warning("Performing emergency memory cleanup")
        
        for strategy in self.cleanup_strategies:
            try:
                strategy()
                # Check if cleanup helped
                stats = self.monitor.get_current_stats()
                if stats.free > 0.1:  # At least 100MB free
                    break
            except Exception as e:
                logger.error(f"Cleanup strategy failed: {e}")
    
    def _clear_cache(self):
        """Clear PyTorch CUDA cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
    
    def _force_garbage_collection(self):
        """Force Python garbage collection."""
        gc.collect()
        logger.debug("Forced garbage collection")
    
    def _reset_peak_memory_stats(self):
        """Reset peak memory tracking."""
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device.index if self.device.index else 0)
            logger.debug("Reset peak memory stats")
    
    def optimize_memory_efficiency(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply memory optimization techniques to a model."""
        # Enable gradient checkpointing if supported
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        
        # Use memory-efficient attention if available
        if hasattr(model, "set_use_memory_efficient_attention_xformers"):
            try:
                model.set_use_memory_efficient_attention_xformers(True)
                logger.info("Enabled memory-efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable memory-efficient attention: {e}")
        
        return model
    
    def get_memory_breakdown(self) -> Dict[str, float]:
        """Get detailed memory breakdown by component."""
        stats = self.monitor.get_current_stats()
        
        breakdown = {
            "allocated_gb": stats.allocated,
            "cached_gb": stats.cached,
            "free_gb": stats.free,
            "total_gb": stats.total,
            "utilization_percent": stats.utilization,
            "peak_allocated_gb": stats.max_allocated
        }
        
        return breakdown
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest memory optimizations based on current usage."""
        stats = self.monitor.get_current_stats()
        suggestions = []
        
        if stats.utilization > 90:
            suggestions.append("Consider gradient accumulation to reduce batch size")
            suggestions.append("Enable mixed precision training")
            suggestions.append("Use gradient checkpointing")
        
        if stats.cached > stats.allocated * 2:
            suggestions.append("Excessive cache usage - consider manual cache clearing")
        
        if stats.max_allocated > stats.total * 0.8:
            suggestions.append("Peak memory usage is high - consider model parallelism")
        
        return suggestions


class OOMDebugger:
    """Advanced Out-Of-Memory error debugging and analysis."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monitor = GPUMonitor(self.device)
        self.oom_history: List[Dict[str, Any]] = []
    
    def capture_oom_context(self, error: RuntimeError) -> Dict[str, Any]:
        """Capture full context when OOM occurs."""
        if "out of memory" not in str(error).lower():
            return {}
        
        context = {
            "timestamp": time.time(),
            "error_message": str(error),
            "memory_stats": self.monitor.get_current_stats(),
            "python_memory": psutil.Process().memory_info().rss / 1024**3,  # GB
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
        
        # Add stack trace information
        import traceback
        context["stack_trace"] = traceback.format_stack()
        
        self.oom_history.append(context)
        logger.error(f"OOM captured: {context['memory_stats']}")
        
        return context
    
    def analyze_oom_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in OOM occurrences."""
        if not self.oom_history:
            return {"message": "No OOM events recorded"}
        
        analysis = {
            "total_ooms": len(self.oom_history),
            "average_memory_usage": sum(h["memory_stats"].utilization for h in self.oom_history) / len(self.oom_history),
            "common_stack_traces": self._find_common_patterns(),
            "memory_trend": self._get_memory_trend(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _find_common_patterns(self) -> List[str]:
        """Find common patterns in OOM stack traces."""
        # Simple pattern matching on stack traces
        patterns = {}
        for oom in self.oom_history:
            for line in oom["stack_trace"]:
                if "forward" in line or "backward" in line:
                    patterns[line] = patterns.get(line, 0) + 1
        
        # Return top patterns
        return sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_memory_trend(self) -> str:
        """Analyze memory usage trend leading to OOMs."""
        if len(self.oom_history) < 2:
            return "Insufficient data for trend analysis"
        
        recent_utilizations = [h["memory_stats"].utilization for h in self.oom_history[-5:]]
        if all(u > 90 for u in recent_utilizations):
            return "Consistently high memory usage before OOM"
        elif recent_utilizations[-1] > recent_utilizations[0]:
            return "Memory usage trending upward before OOM"
        else:
            return "No clear trend in memory usage"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on OOM analysis."""
        recommendations = []
        
        if len(self.oom_history) > 5:
            recommendations.append("Frequent OOMs - consider architectural changes")
        
        avg_util = sum(h["memory_stats"].utilization for h in self.oom_history) / len(self.oom_history)
        if avg_util > 95:
            recommendations.append("Reduce batch size or use gradient accumulation")
            recommendations.append("Enable mixed precision training")
        
        return recommendations


@contextmanager
def memory_profile(operation_name: str = "operation"):
    """Context manager for profiling memory usage of an operation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = GPUMonitor(device)
    
    # Start monitoring
    monitor.start_monitoring(interval=0.1)
    
    # Record initial state
    initial_stats = monitor.get_current_stats()
    logger.info(f"Starting memory profiling for '{operation_name}'")
    logger.info(f"Initial memory: {initial_stats.allocated:.2f}GB allocated, {initial_stats.utilization:.1f}% utilization")
    
    try:
        yield monitor
    finally:
        # Stop monitoring and analyze
        monitor.stop_monitoring()
        final_stats = monitor.get_current_stats()
        peak_stats = monitor.find_peak_usage()
        
        logger.info(f"Memory profiling for '{operation_name}' complete:")
        logger.info(f"  Final: {final_stats.allocated:.2f}GB allocated, {final_stats.utilization:.1f}% utilization")
        logger.info(f"  Peak: {peak_stats.max_allocated:.2f}GB allocated")
        logger.info(f"  Delta: {final_stats.allocated - initial_stats.allocated:.2f}GB")


# Example usage and learning exercises
def demonstrate_memory_management():
    """Demonstrate key memory management concepts."""
    
    # 1. Basic memory monitoring
    print("=== Basic Memory Monitoring ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monitor = GPUMonitor(device)
    
    print(f"Current memory stats: {monitor.get_current_stats()}")
    
    # 2. Memory profiling with large tensor
    if device.type == "cuda":
        with memory_profile("large_tensor_creation"):
            # Create a large tensor that uses significant memory
            large_tensor = torch.randn(10000, 10000, device=device)
            result = torch.mm(large_tensor, large_tensor.T)
            del large_tensor, result  # Explicit cleanup
    
    # 3. OOM simulation and debugging
    print("\n=== OOM Debugging Demo ===")
    debugger = OOMDebugger(device)
    
    try:
        if device.type == "cuda":
            # Try to allocate more memory than available
            stats = monitor.get_current_stats()
            available_gb = stats.free
            
            # Allocate 90% of available memory
            big_tensor = torch.randn(int(available_gb * 0.9 * 1024**3 // 4), device=device)
            del big_tensor
            
            # This might cause OOM
            huge_tensor = torch.randn(int(available_gb * 2 * 1024**3 // 4), device=device)
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            context = debugger.capture_oom_context(e)
            print(f"OOM captured: {context['memory_stats']}")
            analysis = debugger.analyze_oom_patterns()
            print(f"OOM analysis: {analysis}")
    
    # 4. Memory optimization suggestions
    print("\n=== Memory Optimization Suggestions ===")
    manager = MemoryManager(device)
    suggestions = manager.suggest_optimizations()
    print("Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_memory_management()
