"""
Data Engineering Modules for ML Systems

This module covers the critical data engineering aspects that cause 80% of model failures:
- Data drift detection and monitoring
- Data leakage identification and prevention
- Label quality validation and cleaning
- Feature engineering and validation
- Data contracts and schema evolution
- Statistical validation and testing
"""

from .drift import DataDriftDetector, DistributionShiftDetector, CovariateShiftDetector
from .leakage import LeakageDetector, TargetLeakageDetector, TemporalLeakageDetector
from .quality import DataQualityValidator, LabelValidator, OutlierDetector
from .features import FeatureEngineer, FeatureValidator, FeatureStore
from .contracts import DataContract, SchemaEvolution, BackfillManager
from .validation import StatisticalValidator, DataProfiler, QualityMetrics

__all__ = [
    # Drift Detection
    "DataDriftDetector",
    "DistributionShiftDetector", 
    "CovariateShiftDetector",
    
    # Leakage Detection
    "LeakageDetector",
    "TargetLeakageDetector",
    "TemporalLeakageDetector",
    
    # Data Quality
    "DataQualityValidator",
    "LabelValidator",
    "OutlierDetector",
    
    # Feature Engineering
    "FeatureEngineer",
    "FeatureValidator",
    "FeatureStore",
    
    # Data Contracts
    "DataContract",
    "SchemaEvolution",
    "BackfillManager",
    
    # Validation
    "StatisticalValidator",
    "DataProfiler",
    "QualityMetrics",
]
