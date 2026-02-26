# PyTorch Memory Optimization: Best Practices Guide

This guide covers comprehensive memory optimization techniques in PyTorch for efficient training and deployment.

## Table of Contents
1. [Mixed Precision Training](#mixed-precision-training)
2. [Gradient Checkpointing](#gradient-checkpointing)
3. [Optimizer Zero Grad](#optimizer-zero-grad)
4. [Skipping Gradients Entirely](#skipping-gradients-entirely)
5. [Data Loading Optimization](#data-loading-optimization)
6. [Model Quantization](#model-quantization)
7. [Reduce Redundant Variables](#reduce-redundant-variables)
8. [Dynamic Batch Size Adjustment](#dynamic-batch-size-adjustment)
9. [Parameter Pruning](#parameter-pruning)
10. [Distributed Training with Memory Splitting](#distributed-training-with-memory-splitting)
11. [Sparse Matrices](#sparse-matrices)
12. [Knowledge Distillation](#knowledge-distillation)
13. [Dynamic Layer Offloading](#dynamic-layer-offloading)
14. [Gradient Accumulation](#gradient-accumulation)
15. [Memory Monitoring](#memory-monitoring)
16. [Profile First Approach](#profile-first-approach)

---

## Mixed Precision Training

**Description**: Uses FP16/BF16 instead of FP32 to reduce memory usage by ~50% and speed up training on compatible GPUs.

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def mixed_precision_training_example():
    model = torch.nn.Linear(1000, 1000).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()
    
    # Training loop
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(batch['input'])
            loss = criterion(outputs, batch['target'])
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

**Benefits**: 50% memory reduction, 2-3x speedup on modern GPUs
**Trade-offs**: Potential numerical instability, requires gradient scaling

---

## Gradient Checkpointing

**Description**: Trades computation for memory by recomputing intermediate activations during backward pass.

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1000, 1000)
        self.layer2 = torch.nn.Linear(1000, 1000)
        self.layer3 = torch.nn.Linear(1000, 1000)
    
    def forward(self, x):
        # Checkpoint the middle layers
        x = self.layer1(x)
        x = checkpoint(self.layer2, x)  # Recomputed during backward
        x = self.layer3(x)
        return x

# Usage
model = CheckpointedModel().cuda()
```

**Benefits**: Significant memory savings for deep models
**Trade-offs**: ~20-30% slower training due to recomputation

---

## Optimizer Zero Grad

**Description**: Explicitly zero gradients to prevent memory accumulation and use efficient zero_grad variants.

```python
import torch

# Standard zero_grad
optimizer.zero_grad()

# More efficient: set_to_none=True
optimizer.zero_grad(set_to_none=True)  # Faster, uses less memory

# Example training loop
def efficient_training_loop():
    model = torch.nn.Linear(1000, 1000).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

**Benefits**: Prevents gradient memory leaks, `set_to_none=True` is faster
**Trade-offs**: None, this is always recommended

---

## Skipping Gradients Entirely

**Description**: Disable gradient computation for inference or specific layers to save memory.

```python
import torch

# Method 1: Using torch.no_grad() for inference
@torch.no_grad()
def inference_mode(model, input_data):
    return model(input_data)

# Method 2: Disable gradients for specific parameters
def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only specific layers
    for param in model.classifier.parameters():
        param.requires_grad = True

# Method 3: Context manager for selective gradient computation
def selective_gradients():
    with torch.no_grad():
        # Preprocessing without gradients
        processed_data = preprocess(data)
    
    # Enable gradients only for training
    with torch.enable_grad():
        output = model(processed_data)
        loss = criterion(output, target)
        loss.backward()
```

**Benefits**: Major memory savings during inference, faster execution
**Trade-offs**: Cannot train with disabled gradients

---

## Data Loading Optimization

**Description**: Optimize data loading pipeline to reduce memory footprint and improve efficiency.

```python
import torch
from torch.utils.data import DataLoader, Dataset

class OptimizedDataset(Dataset):
    def __init__(self, data_path):
        # Load data lazily instead of storing everything in memory
        self.data_path = data_path
        self.index = self._build_index()
    
    def _build_index(self):
        # Store only file paths/metadata, not actual data
        return [f"sample_{i}.pt" for i in range(num_samples)]
    
    def __getitem__(self, idx):
        # Load data on-demand
        return torch.load(f"{self.data_path}/{self.index[idx]}")
    
    def __len__(self):
        return len(self.index)

# Optimized DataLoader configuration
def create_optimized_dataloader(dataset, batch_size=32):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,  # Parallel loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive
        prefetch_factor=2,  # Prefetch batches
        drop_last=True  # Consistent batch sizes
    )
```

**Benefits**: Reduced memory usage, faster data loading, better GPU utilization
**Trade-offs**: More complex setup, requires careful tuning of workers

---

## Model Quantization

**Description**: Reduce model precision and size using quantization techniques.

```python
import torch
import torch.quantization as quant

# Post-training quantization
def quantize_model(model):
    # Prepare model for quantization
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)
    
    # Calibrate with representative data
    with torch.no_grad():
        for data in calibration_data:
            model(data)
    
    # Convert to quantized model
    quantized_model = quant.convert(model, inplace=False)
    return quantized_model

# Dynamic quantization (better for RNNs, LSTMs)
def dynamic_quantization(model):
    return quant.quantize_dynamic(
        model, 
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU}, 
        dtype=torch.qint8
    )

# Usage example
original_model = torch.nn.Linear(1000, 1000)
quantized_model = dynamic_quantization(original_model)

# Memory comparison
print(f"Original size: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

**Benefits**: 4x smaller models, faster inference on CPU
**Trade-offs**: Slight accuracy degradation, not all operations supported

---

## Reduce Redundant Variables

**Description**: Eliminate unnecessary variables and optimize memory usage patterns.

```python
import torch

def memory_efficient_processing():
    # Bad: Creating multiple intermediate tensors
    def inefficient():
        x = torch.randn(1000, 1000).cuda()
        y = x + 1
        z = y * 2
        w = z - 1
        result = w.sum()
        return result
    
    # Good: In-place operations and variable reuse
    def efficient():
        x = torch.randn(1000, 1000).cuda()
        x.add_(1)  # In-place addition
        x.mul_(2)  # In-place multiplication
        x.sub_(1)  # In-place subtraction
        return x.sum()
    
    # Good: Use context managers for cleanup
    def with_cleanup():
        with torch.cuda.device(0):
            temp_tensors = []
            for i in range(10):
                temp = torch.randn(1000, 1000).cuda()
                # Process temp
                temp_tensors.append(temp)
                del temp  # Explicit cleanup
            
            result = torch.stack(temp_tensors).sum()
            del temp_tensors  # Clear list
            torch.cuda.empty_cache()  # Clear cache
            return result
```

**Benefits**: Reduced memory fragmentation, lower peak memory usage
**Trade-offs**: Code may be less readable, requires careful management

---

## Dynamic Batch Size Adjustment

**Description**: Automatically adjust batch size based on available memory.

```python
import torch
import gc

class DynamicBatchSizer:
    def __init__(self, initial_batch_size=32, min_batch_size=1, max_batch_size=256):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.oom_count = 0
        
    def get_batch_size(self):
        return self.current_batch_size
    
    def handle_oom(self):
        """Handle out-of-memory error by reducing batch size"""
        self.oom_count += 1
        self.current_batch_size = max(
            self.min_batch_size, 
            self.current_batch_size // 2
        )
        gc.collect()
        torch.cuda.empty_cache()
        print(f"OOM detected. Reduced batch size to {self.current_batch_size}")
    
    def try_increase_batch_size(self):
        """Gradually increase batch size if memory is available"""
        if self.oom_count == 0 and self.current_batch_size < self.max_batch_size:
            self.current_batch_size = min(
                self.max_batch_size,
                self.current_batch_size + 8
            )

# Usage
def train_with_dynamic_batching():
    batch_sizer = DynamicBatchSizer()
    model = torch.nn.Linear(1000, 1000).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(epochs):
        try:
            batch_size = batch_sizer.get_batch_size()
            # Create dataloader with current batch size
            dataloader = create_dataloader(batch_size)
            
            for batch in dataloader:
                optimizer.zero_grad(set_to_none=True)
                
                outputs = model(batch)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                batch_sizer.try_increase_batch_size()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_sizer.handle_oom()
                continue
            else:
                raise e
```

**Benefits**: Adapts to available memory, maximizes throughput
**Trade-offs**: Variable batch sizes may affect training dynamics

---

## Parameter Pruning

**Description**: Remove unnecessary weights to reduce model size and memory usage.

```python
import torch
import torch.nn.utils.prune as prune

def prune_model(model, pruning_ratio=0.2):
    """Prune model parameters to reduce memory usage"""
    
    # Global structured pruning
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_ratio,
    )
    
    # Remove pruning masks to make it permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    return model

# Fine-grained unstructured pruning
def fine_grained_pruning(model, layers_to_prune):
    for name, module in model.named_modules():
        if name in layers_to_prune:
            prune.l1_unstructured(module, name='weight', amount=0.3)
            prune.remove(module, 'weight')  # Make permanent

# Usage
model = torch.nn.Sequential(
    torch.nn.Linear(1000, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 100)
)

pruned_model = prune_model(model, pruning_ratio=0.3)
print(f"Original parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Pruned parameters: {sum(p.numel() for p in pruned_model.parameters())}")
```

**Benefits**: Reduced model size, potential speedup, less memory usage
**Trade-offs**: Potential accuracy degradation, requires careful tuning

---

## Distributed Training with Memory Splitting

**Description**: Use multiple GPUs to split memory requirements across devices.

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)
    
    # Create model and move to GPU
    model = torch.nn.Linear(1000, 1000).cuda(rank)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for batch in dataloader:
        # Move batch to GPU
        batch = batch.cuda(rank)
        
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    cleanup_distributed()

# Launch distributed training
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
```

**Benefits**: Scales to larger models, reduces memory per GPU
**Trade-offs**: Requires multiple GPUs, communication overhead

---

## Sparse Matrices

**Description**: Use sparse matrix representations for memory efficiency.

```python
import torch
from torch.sparse import mm

def sparse_matrix_operations():
    # Create sparse matrices
    dense_size = 10000
    sparsity = 0.95
    
    # Create sparse tensor
    indices = torch.randint(0, dense_size, (2, 100))
    values = torch.randn(100)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (dense_size, dense_size))
    
    # Sparse matrix multiplication
    dense_vector = torch.randn(dense_size)
    result = torch.sparse.mm(sparse_tensor, dense_vector.unsqueeze(1)).squeeze()
    
    # Convert between sparse and dense
    dense_tensor = sparse_tensor.to_dense()
    sparse_again = dense_tensor.to_sparse()
    
    return result

# Sparse embedding layers (useful for NLP)
class SparseEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, sparse=True)
    
    def forward(self, indices):
        return self.embedding(indices)

# Usage
sparse_embed = SparseEmbedding(vocab_size=100000, embed_dim=300)
indices = torch.randint(0, 100000, (1000,))
embeddings = sparse_embed(indices)  # Memory efficient for large vocab
```

**Benefits**: Significant memory savings for sparse data, faster operations
**Trade-offs**: Limited operation support, conversion overhead for dense operations

---

## Knowledge Distillation

**Description**: Use smaller student models trained with guidance from larger teacher models.

```python
import torch
import torch.nn.functional as F

class KnowledgeDistillation:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha
        
    def distillation_loss(self, student_outputs, teacher_outputs, labels, criterion):
        """Combined loss for knowledge distillation"""
        
        # Standard cross-entropy loss
        ce_loss = criterion(student_outputs, labels)
        
        # Knowledge distillation loss
        soft_teacher = F.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / self.temperature, dim=1)
        kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        
        # Combine losses
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss * (self.temperature ** 2)
        return total_loss
    
    def train_step(self, inputs, labels, optimizer, criterion):
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        
        # Student forward pass
        student_outputs = self.student(inputs)
        
        # Compute distillation loss
        loss = self.distillation_loss(student_outputs, teacher_outputs, labels, criterion)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        return loss.item()

# Usage
teacher_model = torch.nn.Sequential(
    torch.nn.Linear(1000, 2000),
    torch.nn.ReLU(),
    torch.nn.Linear(2000, 1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 10)
).cuda()

student_model = torch.nn.Sequential(
    torch.nn.Linear(1000, 500),
    torch.nn.ReLU(),
    torch.nn.Linear(500, 10)
).cuda()

distillation = KnowledgeDistillation(teacher_model, student_model)
```

**Benefits**: Smaller models use less memory, maintain most of teacher performance
**Trade-offs**: Training complexity, requires pre-trained teacher model

---

## Dynamic Layer Offloading

**Description**: Move layers between CPU and GPU dynamically based on memory needs.

```python
import torch
from contextlib import contextmanager

class LayerOffloader:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.cpu_device = 'cpu'
        
    @contextmanager
    def offload_layer(self, layer_name):
        """Context manager for temporary layer offloading"""
        layer = dict(self.model.named_modules())[layer_name]
        
        # Move to CPU if on GPU
        original_device = next(layer.parameters()).device
        if original_device == torch.device(self.device):
            layer.cpu()
            torch.cuda.empty_cache()
        
        try:
            yield layer
        finally:
            # Move back to original device
            if original_device == torch.device(self.device):
                layer.to(self.device)
    
    def forward_with_offloading(self, x):
        """Forward pass with dynamic layer offloading"""
        
        # Example: Offload intermediate layers
        x = self.model.layer1(x)
        
        with self.offload_layer('layer2'):
            x = self.model.layer2(x)
        
        with self.offload_layer('layer3'):
            x = self.model.layer3(x)
        
        x = self.model.layer4(x)
        return x

# Gradient checkpointing with offloading
def memory_efficient_forward(model, x):
    """Combine checkpointing with offloading"""
    
    def create_custom_forward(module):
        def custom_forward(*inputs):
            with torch.cuda.device(x.device.index):
                return module(*inputs)
        return custom_forward
    
    # Checkpoint with offloading
    x = torch.utils.checkpoint.checkpoint(
        create_custom_forward(model.layer1),
        x
    )
    
    # Offload layer2 during computation
    model.layer2.cpu()
    torch.cuda.empty_cache()
    
    x = torch.utils.checkpoint.checkpoint(
        create_custom_forward(model.layer2),
        x
    )
    
    model.layer2.cuda()
    return x
```

**Benefits**: Enables training very large models on limited GPU memory
**Trade-offs**: Significant overhead, complex implementation

---

## Gradient Accumulation

**Description**: Simulate larger batch sizes by accumulating gradients over multiple steps.

```python
import torch

class GradientAccumulator:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def accumulate_gradients(self, loss):
        """Accumulate gradients over multiple steps"""
        
        # Scale loss for accumulation
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        self.current_step += 1
        
        # Update weights after accumulation steps
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
    
    def step(self, batch):
        """Single training step with accumulation"""
        inputs, targets = batch
        
        # Forward pass
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        
        # Accumulate gradients
        self.accumulate_gradients(loss)
        
        return loss.item() * self.accumulation_steps

# Usage
model = torch.nn.Linear(1000, 1000).cuda()
optimizer = torch.optim.Adam(model.parameters())
accumulator = GradientAccumulator(model, optimizer, accumulation_steps=8)

# Effective batch size = actual_batch_size * accumulation_steps
for batch in dataloader:
    loss = accumulator.step(batch)
    
    # Print loss only after accumulation
    if accumulator.current_step % accumulator.accumulation_steps == 0:
        print(f"Accumulated loss: {loss:.4f}")
```

**Benefits**: Simulate large batch sizes with limited memory, better gradient estimates
**Trade-offs**: Slower training, requires careful learning rate adjustment

---

## Memory Monitoring

**Description**: Continuously monitor memory usage to detect issues and optimize performance.

```python
import torch
import psutil
import time
from contextlib import contextmanager

class MemoryMonitor:
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        self.peak_memory = 0
        self.memory_history = []
    
    def get_memory_info(self):
        """Get current memory usage information"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            return {
                'allocated': allocated,
                'reserved': reserved,
                'max_allocated': max_allocated,
                'free': (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_reserved()) / 1024**3
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0, 'free': 0}
    
    def update_peak_memory(self):
        """Update peak memory tracking"""
        current_memory = self.get_memory_info()['allocated']
        self.peak_memory = max(self.peak_memory, current_memory)
        self.memory_history.append(current_memory)
    
    @contextmanager
    def monitor_context(self, operation_name=""):
        """Context manager for monitoring memory during operations"""
        initial_memory = self.get_memory_info()
        start_time = time.time()
        
        try:
            yield
        finally:
            final_memory = self.get_memory_info()
            duration = time.time() - start_time
            
            print(f"\n=== Memory Report: {operation_name} ===")
            print(f"Duration: {duration:.2f}s")
            print(f"Initial memory: {initial_memory['allocated']:.2f} GB")
            print(f"Final memory: {final_memory['allocated']:.2f} GB")
            print(f"Peak memory: {final_memory['max_allocated']:.2f} GB")
            print(f"Memory change: {final_memory['allocated'] - initial_memory['allocated']:.2f} GB")
    
    def print_summary(self):
        """Print memory usage summary"""
        info = self.get_memory_info()
        print(f"\n=== Memory Summary ===")
        print(f"Current allocated: {info['allocated']:.2f} GB")
        print(f"Current reserved: {info['reserved']:.2f} GB")
        print(f"Peak allocated: {info['max_allocated']:.2f} GB")
        print(f"Free memory: {info['free']:.2f} GB")

# Usage
monitor = MemoryMonitor()

def training_with_monitoring():
    model = torch.nn.Linear(10000, 10000).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    with monitor.monitor_context("Training Loop"):
        for i, batch in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(batch)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            monitor.update_peak_memory()
            
            if i % 10 == 0:
                monitor.print_summary()

# Automatic memory cleanup
def auto_cleanup():
    """Automatic memory cleanup when memory is low"""
    memory_info = monitor.get_memory_info()
    
    if memory_info['allocated'] > memory_info['free'] * 0.8:
        print("Memory usage high, performing cleanup...")
        torch.cuda.empty_cache()
        import gc
        gc.collect()
```

**Benefits**: Early detection of memory issues, performance optimization insights
**Trade-offs**: Minimal overhead, requires integration into training code

---

## Profile First Approach

**Description**: Always profile your code before applying optimizations to identify actual bottlenecks.

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd

def profile_model(model, dataloader, num_batches=10):
    """Comprehensive profiling of model performance"""
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1, warmup=1, active=num_batches, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches + 2:  # wait + warmup + active
                break
                
            with record_function("forward_pass"):
                outputs = model(batch)
                loss = outputs.mean()
            
            with record_function("backward_pass"):
                loss.backward()
            
            prof.step()
    
    return prof

def analyze_profiling_results(prof):
    """Analyze and display profiling results"""
    
    print("=== Top 10 CPU Operations ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    if torch.cuda.is_available():
        print("\n=== Top 10 CUDA Operations ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        print("\n=== Top 10 Memory Operations ===")
        print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
    
    # Export to CSV for further analysis
    df = pd.DataFrame([
        {
            'name': item.key,
            'cpu_time': item.cpu_time_total,
            'cuda_time': item.cuda_time_total if torch.cuda.is_available() else 0,
            'cpu_memory': item.cpu_memory_usage,
            'cuda_memory': item.cuda_memory_usage if torch.cuda.is_available() else 0,
            'call_count': item.count
        }
        for item in prof.key_averages()
    ])
    
    df.to_csv('profiling_results.csv', index=False)
    return df

# Memory profiling specifically
def profile_memory_usage(model, input_size=(1000, 1000)):
    """Profile memory usage patterns"""
    
    if not torch.cuda.is_available():
        print("CUDA not available for memory profiling")
        return
    
    model.cuda()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        
        # Forward pass
        x = torch.randn(*input_size).cuda()
        with record_function("forward"):
            output = model(x)
        
        # Backward pass
        with record_function("backward"):
            loss = output.mean()
            loss.backward()
    
    peak_memory = torch.cuda.max_memory_allocated()
    
    print(f"\n=== Memory Profile ===")
    print(f"Initial memory: {initial_memory / 1024**2:.2f} MB")
    print(f"Peak memory: {peak_memory / 1024**2:.2f} MB")
    print(f"Memory increase: {(peak_memory - initial_memory) / 1024**2:.2f} MB")
    
    # Memory timeline
    print("\n=== Memory Timeline ===")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))

# Usage example
def profile_first_approach():
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 2000),
        torch.nn.ReLU(),
        torch.nn.Linear(2000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 10)
    )
    
    # Create dummy data
    dataloader = [torch.randn(32, 1000) for _ in range(20)]
    
    # Profile the model
    prof = profile_model(model, dataloader)
    results_df = analyze_profiling_results(prof)
    
    # Profile memory usage
    profile_memory_usage(model)
    
    return results_df

# Optimization decisions based on profiling
def optimize_based_on_profiling(profiling_results):
    """Make optimization decisions based on profiling data"""
    
    # Find top memory consumers
    top_memory_ops = profiling_results.nlargest(5, 'cpu_memory')
    print("Top memory operations:")
    for _, row in top_memory_ops.iterrows():
        print(f"  {row['name']}: {row['cpu_memory'] / 1024**2:.2f} MB")
    
    # Find slow operations
    top_time_ops = profiling_results.nlargest(5, 'cpu_time')
    print("\nSlowest operations:")
    for _, row in top_time_ops.iterrows():
        print(f"  {row['name']}: {row['cpu_time']:.2f} μs")
    
    # Suggest optimizations
    suggestions = []
    
    if any('matmul' in op.lower() or 'linear' in op.lower() for op in top_memory_ops['name']):
        suggestions.append("Consider mixed precision training for matrix operations")
    
    if any('backward' in op.lower() for op in top_time_ops['name']):
        suggestions.append("Consider gradient checkpointing to reduce backward pass memory")
    
    if profiling_results['cpu_memory'].sum() > 1024**3:  # > 1GB
        suggestions.append("Consider reducing batch size or using gradient accumulation")
    
    print("\nOptimization suggestions:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")
```

**Benefits**: Data-driven optimization, avoids premature optimization, identifies real bottlenecks
**Trade-offs**: Profiling overhead, requires analysis time

---

## Summary and Best Practices

### Quick Reference Checklist

1. **Before Training**
   - [ ] Profile your model first
   - [ ] Set up memory monitoring
   - [ ] Choose appropriate batch size
   - [ ] Enable mixed precision if available

2. **During Training**
   - [ ] Use `optimizer.zero_grad(set_to_none=True)`
   - [ ] Monitor memory usage continuously
   - [ ] Clear cache periodically if needed
   - [ ] Use gradient accumulation for large effective batches

3. **For Large Models**
   - [ ] Consider gradient checkpointing
   - [ ] Use distributed training if possible
   - [ ] Implement layer offloading if necessary
   - [ ] Try model pruning and quantization

4. **Memory Optimization Techniques**
   - [ ] Remove redundant variables
   - [ ] Use in-place operations where safe
   - [ ] Implement sparse matrices for sparse data
   - [ ] Consider knowledge distillation for deployment

### Memory Optimization Priority

1. **High Impact, Low Effort**
   - Mixed precision training
   - `set_to_none=True` in zero_grad
   - Proper data loading configuration

2. **High Impact, Medium Effort**
   - Gradient checkpointing
   - Gradient accumulation
   - Memory monitoring

3. **High Impact, High Effort**
   - Distributed training
   - Dynamic layer offloading
   - Custom memory management

4. **Specialized Use Cases**
   - Sparse matrices
   - Knowledge distillation
   - Model quantization

Remember: **Always profile first, then optimize based on actual bottlenecks rather than assumptions!**
