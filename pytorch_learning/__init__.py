"""
PyTorch Learning System - Comprehensive ML Engineering Curriculum

A production-ready learning system covering the full ML stack from GPU kernels to deployment.
"""

__version__ = "0.1.0"
__author__ = "ML Engineer"
__email__ = "ml@example.com"

# Core modules
from pytorch_learning.core import *
from pytorch_learning.data import *
from pytorch_learning.stats import *
from pytorch_learning.training import *
from pytorch_learning.llm import *
from pytorch_learning.inference import *
from pytorch_learning.retrieval import *
from pytorch_learning.monitoring import *
from pytorch_learning.deployment import *

__all__ = [
    # Core PyTorch
    "MemoryManager",
    "Profiler", 
    "MixedPrecisionTrainer",
    "CustomKernel",
    
    # Data Engineering
    "DataDriftDetector",
    "LeakageDetector", 
    "LabelValidator",
    "FeatureStore",
    
    # Statistics
    "BiasVarianceAnalyzer",
    "ConfidenceInterval",
    "CalibrationMetric",
    "DistributionShiftDetector",
    
    # Training
    "DistributedTrainer",
    "CheckpointManager",
    "ReproducibilityManager",
    "GradientAccumulator",
    
    # LLM Engineering
    "TokenizerManager",
    "AttentionProfiler",
    "LoRATrainer",
    "RAGSystem",
    
    # Inference
    "BatchProcessor",
    "QuantizationEngine",
    "ServingOptimizer",
    "CacheManager",
    
    # Retrieval
    "EmbeddingTrainer",
    "DocumentChunker",
    "HybridSearch",
    "Reranker",
    
    # Monitoring
    "DriftMonitor",
    "PerformanceTracker",
    "QualityAssurance",
    "MetricsCollector",
    
    # Deployment
    "ModelVersioning",
    "CanaryDeployer",
    "RollbackManager",
    "DocumentationGenerator",
]
