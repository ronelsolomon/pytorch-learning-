"""
Mixed precision training utilities for PyTorch.

This module provides tools for training with reduced precision (FP16/BF16)
to improve performance and reduce memory usage.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import warnings


class MixedPrecisionTrainer:
    """Trainer with mixed precision support using GradScaler."""
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: torch.device,
                 precision: str = 'fp16'):
        """
        Initialize mixed precision trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer for training
            device: Device to train on
            precision: Precision mode ('fp16', 'bf16', or 'fp32')
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.precision = precision
        
        if precision == 'fp16':
            self.scaler = torch.cuda.amp.GradScaler()
        elif precision == 'bf16':
            if not torch.cuda.is_bf16_supported():
                warnings.warn("BF16 not supported, falling back to FP32")
                self.precision = 'fp32'
                self.scaler = None
            else:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
    def train_step(self, 
                   data: torch.Tensor, 
                   target: torch.Tensor,
                   criterion: nn.Module) -> Dict[str, float]:
        """
        Perform one training step with mixed precision.
        
        Args:
            data: Input data
            target: Target labels
            criterion: Loss function
            
        Returns:
            Dictionary with loss and other metrics
        """
        data, target = data.to(self.device), target.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.precision in ['fp16', 'bf16']:
            with torch.cuda.amp.autocast(dtype=torch.float16 if self.precision == 'fp16' else torch.bfloat16):
                output = self.model(data)
                loss = criterion(output, target)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
        return {'loss': loss.item()}


class LossScaler:
    """Custom loss scaling for mixed precision training."""
    
    def __init__(self, 
                 initial_scale: float = 2**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """
        Initialize loss scaler.
        
        Args:
            initial_scale: Initial scaling factor
            growth_factor: Factor to increase scale when no overflow
            backoff_factor: Factor to decrease scale when overflow occurs
            growth_interval: Steps between scale growth attempts
        """
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.step_count = 0
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss by current scale factor."""
        return loss * self.scale
        
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients in optimizer."""
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                param.grad.div_(self.scale)
                
    def update_scale(self, overflow_detected: bool):
        """Update scale based on whether overflow was detected."""
        if overflow_detected:
            self.scale *= self.backoff_factor
            self.step_count = 0
        else:
            self.step_count += 1
            if self.step_count >= self.growth_interval:
                self.scale *= self.growth_factor
                self.step_count = 0
                
    def check_overflow(self, param: torch.Tensor) -> bool:
        """Check if parameter contains overflow."""
        return not torch.isfinite(param).all()


class PrecisionAnalyzer:
    """Analyze numerical precision and stability of models."""
    
    def __init__(self):
        self.metrics = {}
        
    def analyze_model_precision(self, 
                                model: nn.Module,
                                sample_input: torch.Tensor,
                                precision_modes: list = ['fp32', 'fp16']) -> Dict[str, Any]:
        """
        Analyze model behavior across different precision modes.
        
        Args:
            model: Model to analyze
            sample_input: Sample input for testing
            precision_modes: List of precision modes to test
            
        Returns:
            Analysis results comparing precision modes
        """
        model.eval()
        results = {}
        
        # FP32 baseline
        with torch.no_grad():
            fp32_output = model(sample_input.float())
            results['fp32'] = {
                'output': fp32_output,
                'mean': fp32_output.mean().item(),
                'std': fp32_output.std().item(),
                'max': fp32_output.max().item(),
                'min': fp32_output.min().item()
            }
            
        # Test other precisions
        for precision in precision_modes:
            if precision == 'fp32':
                continue
                
            with torch.no_grad():
                if precision == 'fp16':
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        output = model(sample_input.float())
                elif precision == 'bf16':
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        output = model(sample_input.float())
                else:
                    continue
                    
                # Compare with FP32
                diff = torch.abs(output - fp32_output)
                rel_error = diff / (torch.abs(fp32_output) + 1e-8)
                
                results[precision] = {
                    'output': output,
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'max': output.max().item(),
                    'min': output.min().item(),
                    'abs_error_mean': diff.mean().item(),
                    'rel_error_mean': rel_error.mean().item(),
                    'max_abs_error': diff.max().item(),
                    'max_rel_error': rel_error.max().item()
                }
                
        return results
        
    def check_numerical_stability(self, 
                                 model: nn.Module,
                                 sample_input: torch.Tensor,
                                 num_runs: int = 100) -> Dict[str, Any]:
        """
        Check numerical stability over multiple forward passes.
        
        Args:
            model: Model to test
            sample_input: Sample input for testing
            num_runs: Number of forward passes to run
            
        Returns:
            Stability analysis results
        """
        model.eval()
        outputs = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(sample_input)
                outputs.append(output.clone())
                
        # Stack outputs and compute statistics
        outputs_tensor = torch.stack(outputs)
        mean_output = outputs_tensor.mean(dim=0)
        std_output = outputs_tensor.std(dim=0)
        
        # Compute coefficient of variation
        cv = std_output / (torch.abs(mean_output) + 1e-8)
        
        return {
            'mean_std': std_output.mean().item(),
            'max_std': std_output.max().item(),
            'mean_cv': cv.mean().item(),
            'max_cv': cv.max().item(),
            'is_stable': cv.max().item() < 0.01,  # Threshold for stability
            'num_runs': num_runs
        }
