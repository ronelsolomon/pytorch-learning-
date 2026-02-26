"""
Data Drift Detection and Monitoring

This module teaches the critical skill of detecting when your data distribution changes,
which is one of the most common causes of model failure in production.

Key concepts covered:
- Covariate shift detection (input distribution changes)
- Prior probability shift (label distribution changes)
- Concept drift (relationship between inputs and labels changes)
- Statistical tests for drift detection
- Monitoring drift over time
- Handling drift in production systems

Understanding data drift is crucial because 80% of model failures are upstream data issues.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    feature_name: str
    drift_detected: bool
    drift_score: float
    p_value: Optional[float]
    test_statistic: Optional[float]
    method: str
    threshold: float
    interpretation: str


@dataclass
class DriftReport:
    """Comprehensive drift detection report."""
    timestamp: str
    overall_drift_detected: bool
    overall_drift_score: float
    feature_drifts: List[DriftResult]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]


class DataDriftDetector:
    """
    Comprehensive data drift detection using multiple statistical methods.
    
    This class implements various statistical tests and methods to detect
    when your data distribution changes over time, which is critical for
    maintaining model performance in production.
    """
    
    def __init__(self, 
                 alpha: float = 0.05,
                 methods: List[str] = None,
                 threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            alpha: Significance level for statistical tests
            methods: List of drift detection methods to use
            threshold: Threshold for drift score classification
        """
        self.alpha = alpha
        self.threshold = threshold
        self.methods = methods or ['ks_test', 'wasserstein', 'psi', 'classifier']
        self.reference_stats = {}
        self.feature_importance = {}
        
    def fit_reference(self, reference_data: pd.DataFrame, target_col: Optional[str] = None):
        """
        Fit reference statistics on baseline data.
        
        Args:
            reference_data: Baseline dataset
            target_col: Target column name (if supervised)
        """
        logger.info(f"Fitting reference statistics on {len(reference_data)} samples")
        
        # Store reference statistics for each feature
        for col in reference_data.columns:
            if col == target_col:
                continue
                
            feature_data = reference_data[col].dropna()
            
            self.reference_stats[col] = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max(),
                'q25': feature_data.quantile(0.25),
                'q75': feature_data.quantile(0.75),
                'histogram': np.histogram(feature_data, bins=50, density=True),
                'distribution_type': self._detect_distribution_type(feature_data)
            }
        
        # Calculate feature importance if target is provided
        if target_col and target_col in reference_data.columns:
            self._calculate_feature_importance(reference_data, target_col)
        
        logger.info(f"Reference statistics fitted for {len(self.reference_stats)} features")
    
    def detect_drift(self, 
                    current_data: pd.DataFrame,
                    target_col: Optional[str] = None) -> DriftReport:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current dataset to compare against reference
            target_col: Target column name (if supervised)
            
        Returns:
            Comprehensive drift report
        """
        logger.info(f"Detecting drift on {len(current_data)} samples")
        
        feature_drifts = []
        
        for col in self.reference_stats.keys():
            if col not in current_data.columns:
                logger.warning(f"Feature {col} not found in current data")
                continue
            
            # Detect drift for this feature
            drift_result = self._detect_feature_drift(col, current_data[col])
            feature_drifts.append(drift_result)
        
        # Calculate overall drift score
        overall_drift_score = self._calculate_overall_drift_score(feature_drifts)
        overall_drift_detected = overall_drift_score > self.threshold
        
        # Generate recommendations
        recommendations = self._generate_recommendations(feature_drifts, overall_drift_detected)
        
        # Create summary statistics
        summary_stats = self._create_summary_statistics(feature_drifts, current_data)
        
        return DriftReport(
            timestamp=pd.Timestamp.now().isoformat(),
            overall_drift_detected=overall_drift_detected,
            overall_drift_score=overall_drift_score,
            feature_drifts=feature_drifts,
            summary_statistics=summary_stats,
            recommendations=recommendations
        )
    
    def _detect_feature_drift(self, feature_name: str, current_data: pd.Series) -> DriftResult:
        """Detect drift for a single feature using multiple methods."""
        reference_stats = self.reference_stats[feature_name]
        current_clean = current_data.dropna()
        
        # Use the best method based on data characteristics
        best_method = self._select_best_method(reference_stats, current_clean)
        
        if best_method == 'ks_test':
            return self._ks_test_drift(feature_name, reference_stats, current_clean)
        elif best_method == 'wasserstein':
            return self._wasserstein_drift(feature_name, reference_stats, current_clean)
        elif best_method == 'psi':
            return self._psi_drift(feature_name, reference_stats, current_clean)
        elif best_method == 'classifier':
            return self._classifier_drift(feature_name, current_clean)
        else:
            # Fallback to KS test
            return self._ks_test_drift(feature_name, reference_stats, current_clean)
    
    def _ks_test_drift(self, feature_name: str, reference_stats: Dict, current_data: pd.Series) -> DriftResult:
        """Kolmogorov-Smirnov test for drift detection."""
        # Generate reference sample from stored statistics
        reference_sample = np.random.normal(
            reference_stats['mean'], 
            reference_stats['std'], 
            len(current_data)
        )
        
        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(reference_sample, current_data)
        
        drift_detected = p_value < self.alpha
        drift_score = 1 - p_value  # Higher score means more likely drift
        
        interpretation = self._interpret_ks_result(ks_statistic, p_value)
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=p_value,
            test_statistic=ks_statistic,
            method="Kolmogorov-Smirnov Test",
            threshold=self.alpha,
            interpretation=interpretation
        )
    
    def _wasserstein_drift(self, feature_name: str, reference_stats: Dict, current_data: pd.Series) -> DriftResult:
        """Wasserstein distance for drift detection."""
        # Generate reference sample
        reference_sample = np.random.normal(
            reference_stats['mean'], 
            reference_stats['std'], 
            len(current_data)
        )
        
        # Calculate Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(reference_sample, current_data)
        
        # Normalize distance (approximate normalization)
        normalized_dist = wasserstein_dist / (reference_stats['std'] + 1e-8)
        
        drift_detected = normalized_dist > self.threshold
        drift_score = min(normalized_dist, 1.0)
        
        interpretation = self._interpret_wasserstein_result(normalized_dist)
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=None,
            test_statistic=wasserstein_dist,
            method="Wasserstein Distance",
            threshold=self.threshold,
            interpretation=interpretation
        )
    
    def _psi_drift(self, feature_name: str, reference_stats: Dict, current_data: pd.Series) -> DriftResult:
        """Population Stability Index (PSI) for drift detection."""
        # Create bins based on reference distribution
        ref_hist, bin_edges = reference_stats['histogram']
        
        # Calculate current histogram with same bins
        current_hist, _ = np.histogram(current_data, bins=bin_edges, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_hist = ref_hist + epsilon
        current_hist = current_hist + epsilon
        
        # Calculate PSI
        psi = np.sum((current_hist - ref_hist) * np.log(current_hist / ref_hist))
        
        drift_detected = psi > 0.25  # Common PSI threshold
        drift_score = min(psi / 0.25, 1.0)  # Normalize to [0, 1]
        
        interpretation = self._interpret_psi_result(psi)
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=None,
            test_statistic=psi,
            method="Population Stability Index",
            threshold=0.25,
            interpretation=interpretation
        )
    
    def _classifier_drift(self, feature_name: str, current_data: pd.Series) -> DriftResult:
        """Classifier-based drift detection."""
        # Generate reference sample
        reference_stats = self.reference_stats[feature_name]
        reference_sample = np.random.normal(
            reference_stats['mean'], 
            reference_stats['std'], 
            len(current_data)
        )
        
        # Create labels (0 for reference, 1 for current)
        X = np.concatenate([reference_sample.reshape(-1, 1), current_data.values.reshape(-1, 1)])
        y = np.concatenate([np.zeros(len(reference_sample)), np.ones(len(current_data))])
        
        # Train simple classifier
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Use cross-validation to get robust score
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
        
        auc_score = cv_scores.mean()
        
        # High AUC means classifier can distinguish distributions -> drift detected
        drift_detected = auc_score > 0.7
        drift_score = (auc_score - 0.5) * 2  # Scale to [0, 1]
        
        interpretation = self._interpret_classifier_result(auc_score)
        
        return DriftResult(
            feature_name=feature_name,
            drift_detected=drift_detected,
            drift_score=drift_score,
            p_value=None,
            test_statistic=auc_score,
            method="Classifier-based Detection",
            threshold=0.7,
            interpretation=interpretation
        )
    
    def _select_best_method(self, reference_stats: Dict, current_data: pd.Series) -> str:
        """Select the best drift detection method based on data characteristics."""
        sample_size = len(current_data)
        
        # For small samples, use KS test
        if sample_size < 30:
            return 'ks_test'
        
        # For large samples, use classifier-based method
        if sample_size > 1000:
            return 'classifier'
        
        # For medium samples, use Wasserstein
        return 'wasserstein'
    
    def _detect_distribution_type(self, data: pd.Series) -> str:
        """Detect the type of distribution for the data."""
        # Simple heuristic based on skewness and kurtosis
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 3:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        else:
            return 'other'
    
    def _calculate_feature_importance(self, data: pd.DataFrame, target_col: str):
        """Calculate feature importance for weighted drift scoring."""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Use Random Forest for importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Store importance scores
        for i, feature in enumerate(X.columns):
            self.feature_importance[feature] = rf.feature_importances_[i]
    
    def _calculate_overall_drift_score(self, feature_drifts: List[DriftResult]) -> float:
        """Calculate overall drift score weighted by feature importance."""
        if not feature_drifts:
            return 0.0
        
        # Weight by feature importance if available
        if self.feature_importance:
            total_importance = 0.0
            weighted_score = 0.0
            
            for drift in feature_drifts:
                importance = self.feature_importance.get(drift.feature_name, 1.0)
                weighted_score += drift.drift_score * importance
                total_importance += importance
            
            return weighted_score / total_importance if total_importance > 0 else 0.0
        else:
            # Simple average
            return np.mean([d.drift_score for d in feature_drifts])
    
    def _generate_recommendations(self, feature_drifts: List[DriftResult], overall_drift: bool) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        if overall_drift:
            recommendations.append("Overall data drift detected - consider model retraining")
        
        # Find most drifted features
        drifted_features = [d for d in feature_drifts if d.drift_detected]
        
        if len(drifted_features) > 0:
            # Sort by drift score
            drifted_features.sort(key=lambda x: x.drift_score, reverse=True)
            
            top_drifted = drifted_features[:3]
            recommendations.append(f"Top drifted features: {[f.feature_name for f in top_drifted]}")
            
            # Specific recommendations based on drift type
            for drift in top_drifted:
                if drift.method == "Kolmogorov-Smirnov Test" and drift.p_value < 0.01:
                    recommendations.append(f"Strong distribution shift in {drift.feature_name} - investigate data pipeline")
                elif drift.method == "Population Stability Index" and drift.test_statistic > 0.5:
                    recommendations.append(f"Severe instability in {drift.feature_name} - check data sources")
        
        # General recommendations
        if len(drifted_features) > len(feature_drifts) * 0.5:
            recommendations.append("Widespread drift detected - review entire data pipeline")
            recommendations.append("Consider implementing automated drift monitoring")
        
        if not recommendations:
            recommendations.append("No significant drift detected - continue monitoring")
        
        return recommendations
    
    def _create_summary_statistics(self, feature_drifts: List[DriftResult], current_data: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for the drift report."""
        drifted_features = [d for d in feature_drifts if d.drift_detected]
        
        return {
            'total_features': len(feature_drifts),
            'drifted_features': len(drifted_features),
            'drift_percentage': len(drifted_features) / len(feature_drifts) * 100,
            'average_drift_score': np.mean([d.drift_score for d in feature_drifts]),
            'max_drift_score': max([d.drift_score for d in feature_drifts]) if feature_drifts else 0.0,
            'current_sample_size': len(current_data),
            'methods_used': list(set([d.method for d in feature_drifts]))
        }
    
    def _interpret_ks_result(self, ks_statistic: float, p_value: float) -> str:
        """Interpret KS test result."""
        if p_value < 0.001:
            return f"Very strong evidence of distribution shift (KS={ks_statistic:.3f}, p<0.001)"
        elif p_value < 0.01:
            return f"Strong evidence of distribution shift (KS={ks_statistic:.3f}, p<0.01)"
        elif p_value < 0.05:
            return f"Moderate evidence of distribution shift (KS={ks_statistic:.3f}, p<0.05)"
        else:
            return f"No significant distribution shift detected (KS={ks_statistic:.3f}, p={p_value:.3f})"
    
    def _interpret_wasserstein_result(self, normalized_dist: float) -> str:
        """Interpret Wasserstein distance result."""
        if normalized_dist > 1.0:
            return f"Very large distributional difference (normalized distance={normalized_dist:.2f})"
        elif normalized_dist > 0.5:
            return f"Large distributional difference (normalized distance={normalized_dist:.2f})"
        elif normalized_dist > 0.2:
            return f"Moderate distributional difference (normalized distance={normalized_dist:.2f})"
        else:
            return f"Small distributional difference (normalized distance={normalized_dist:.2f})"
    
    def _interpret_psi_result(self, psi: float) -> str:
        """Interpret PSI result."""
        if psi > 0.5:
            return f"Severe population instability (PSI={psi:.3f})"
        elif psi > 0.25:
            return f"Moderate population instability (PSI={psi:.3f})"
        elif psi > 0.1:
            return f"Mild population instability (PSI={psi:.3f})"
        else:
            return f"Stable population (PSI={psi:.3f})"
    
    def _interpret_classifier_result(self, auc_score: float) -> str:
        """Interpret classifier-based drift result."""
        if auc_score > 0.9:
            return f"Very strong drift detected (AUC={auc_score:.3f})"
        elif auc_score > 0.8:
            return f"Strong drift detected (AUC={auc_score:.3f})"
        elif auc_score > 0.7:
            return f"Moderate drift detected (AUC={auc_score:.3f})"
        elif auc_score > 0.6:
            return f"Mild drift detected (AUC={auc_score:.3f})"
        else:
            return f"No significant drift detected (AUC={auc_score:.3f})"


class CovariateShiftDetector:
    """
    Specialized detector for covariate shift (input distribution changes).
    
    Covariate shift occurs when P(X) changes but P(Y|X) remains the same.
    This is one of the most common types of drift in production.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.reference_classifier = None
        self.reference_features = None
    
    def fit(self, reference_data: pd.DataFrame):
        """Fit detector on reference data."""
        self.reference_features = reference_data.columns.tolist()
        
        # For covariate shift, we'll use a simple approach:
        # Train a classifier to distinguish reference from current data
        # If it can distinguish them well, there's covariate shift
        
        # Store reference statistics
        self.reference_stats = {}
        for col in reference_data.columns:
            self.reference_stats[col] = {
                'mean': reference_data[col].mean(),
                'std': reference_data[col].std()
            }
    
    def detect_shift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect covariate shift."""
        if self.reference_stats is None:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Create mixed dataset
        reference_sample = self._generate_reference_sample(len(current_data))
        
        X = pd.concat([reference_sample, current_data], ignore_index=True)
        y = np.concatenate([np.zeros(len(reference_sample)), np.ones(len(current_data))])
        
        # Train classifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
        
        auc_score = cv_scores.mean()
        
        # High AUC indicates covariate shift
        shift_detected = auc_score > 0.7
        
        return {
            'shift_detected': shift_detected,
            'auc_score': auc_score,
            'interpretation': self._interpret_covariate_shift(auc_score)
        }
    
    def _generate_reference_sample(self, n_samples: int) -> pd.DataFrame:
        """Generate sample from reference distribution."""
        data = {}
        for col, stats in self.reference_stats.items():
            data[col] = np.random.normal(stats['mean'], stats['std'], n_samples)
        
        return pd.DataFrame(data)
    
    def _interpret_covariate_shift(self, auc_score: float) -> str:
        """Interpret covariate shift result."""
        if auc_score > 0.9:
            return f"Severe covariate shift detected (AUC={auc_score:.3f})"
        elif auc_score > 0.8:
            return f"Strong covariate shift detected (AUC={auc_score:.3f})"
        elif auc_score > 0.7:
            return f"Moderate covariate shift detected (AUC={auc_score:.3f})"
        elif auc_score > 0.6:
            return f"Mild covariate shift detected (AUC={auc_score:.3f})"
        else:
            return f"No significant covariate shift (AUC={auc_score:.3f})"


def demonstrate_drift_detection():
    """Demonstrate drift detection with synthetic data."""
    print("=== Data Drift Detection Demonstration ===")
    
    # Create reference data
    np.random.seed(42)
    n_samples = 1000
    
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(1, n_samples),
        'feature3': np.random.uniform(-1, 1, n_samples),
        'target': np.random.binomial(1, 0.5, n_samples)
    })
    
    print(f"Reference data shape: {reference_data.shape}")
    print(f"Reference data summary:\n{reference_data.describe()}")
    
    # Initialize drift detector
    detector = DataDriftDetector(alpha=0.05)
    
    # Fit reference statistics
    detector.fit_reference(reference_data, target_col='target')
    
    # Test 1: No drift (similar data)
    print("\n--- Test 1: No Drift ---")
    current_data_no_drift = pd.DataFrame({
        'feature1': np.random.normal(0.1, 1.1, n_samples),  # Slight shift
        'feature2': np.random.exponential(1.1, n_samples),   # Slight shift
        'feature3': np.random.uniform(-1.1, 1.1, n_samples),  # Slight shift
        'target': np.random.binomial(1, 0.52, n_samples)
    })
    
    report_no_drift = detector.detect_drift(current_data_no_drift, target_col='target')
    print(f"Overall drift detected: {report_no_drift.overall_drift_detected}")
    print(f"Overall drift score: {report_no_drift.overall_drift_score:.3f}")
    print(f"Recommendations: {report_no_drift.recommendations}")
    
    # Test 2: Moderate drift
    print("\n--- Test 2: Moderate Drift ---")
    current_data_moderate = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.5, n_samples),  # Moderate shift
        'feature2': np.random.exponential(2.0, n_samples),  # Large shift
        'feature3': np.random.uniform(-2, 2, n_samples),     # Large shift
        'target': np.random.binomial(1, 0.7, n_samples)
    })
    
    report_moderate = detector.detect_drift(current_data_moderate, target_col='target')
    print(f"Overall drift detected: {report_moderate.overall_drift_detected}")
    print(f"Overall drift score: {report_moderate.overall_drift_score:.3f}")
    
    # Show drifted features
    drifted_features = [d for d in report_moderate.feature_drifts if d.drift_detected]
    print(f"Drifted features: {[d.feature_name for d in drifted_features]}")
    
    # Test 3: Severe drift
    print("\n--- Test 3: Severe Drift ---")
    current_data_severe = pd.DataFrame({
        'feature1': np.random.normal(2.0, 3.0, n_samples),  # Large shift
        'feature2': np.random.exponential(5.0, n_samples),  # Very large shift
        'feature3': np.random.uniform(-5, 5, n_samples),     # Very large shift
        'target': np.random.binomial(1, 0.9, n_samples)
    })
    
    report_severe = detector.detect_drift(current_data_severe, target_col='target')
    print(f"Overall drift detected: {report_severe.overall_drift_detected}")
    print(f"Overall drift score: {report_severe.overall_drift_score:.3f}")
    print(f"Recommendations: {report_severe.recommendations}")
    
    # Covariate shift specific test
    print("\n--- Covariate Shift Detection ---")
    covariate_detector = CovariateShiftDetector()
    covariate_detector.fit(reference_data.drop('target', axis=1))
    
    shift_result = covariate_detector.detect_shift(current_data_severe.drop('target', axis=1))
    print(f"Covariate shift result: {shift_result}")
    
    print("\n=== Key Takeaways ===")
    print("1. Multiple methods provide robust drift detection")
    print("2. Different features can drift at different rates")
    print("3. Early detection prevents model performance degradation")
    print("4. Statistical significance vs practical significance matter")
    print("5. Monitoring should be continuous in production")


if __name__ == "__main__":
    demonstrate_drift_detection()
