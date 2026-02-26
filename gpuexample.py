import torch
from torch import nn

# Check if CUDA is available
if not torch.cuda.is_available():
    print("CUDA is not available. Running on CPU.")
    device = "cpu"
else:
    print("CUDA is available. Running on GPU.")
    # Start recording memory snapshot history (only if CUDA is available)
    try:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        print("Memory history recording started.")
    except AttributeError as e:
        print(f"Memory history recording not available: {e}")
    device = "cuda"

model = nn.Linear(10_000, 50_000, device=device)
for _ in range(3):
    inputs = torch.randn(5_000, 10_000, device=device)
    outputs = model(inputs)

# Dump memory snapshot history to file and stop recording (only if CUDA is available)
if torch.cuda.is_available():
    try:
        torch.cuda.memory._dump_snapshot("profile.pkl")
        torch.cuda.memory._record_memory_history(enabled=None)
        print("Memory snapshot saved to profile.pkl")
    except AttributeError as e:
        print(f"Memory snapshot functionality not available: {e}")

print(f"Model device: {next(model.parameters()).device}")
print(f"Final output shape: {outputs.shape}")
