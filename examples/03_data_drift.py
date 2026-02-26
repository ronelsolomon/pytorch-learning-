#!/usr/bin/env python3
"""
Data Drift Detection - Learning Exercise 3

This example teaches you how to detect and handle data drift, which causes
80% of model failures in production. You'll learn:

- Why data drift happens and why it's dangerous
- Different types of drift (covariate, concept, prior probability)
- Statistical methods for drift detection
- Real-world drift scenarios and patterns
- How to set up drift monitoring in production
- Strategies for handling detected drift

Run this to understand why your model performance suddenly degrades
and how to prevent it from happening silently.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our learning modules
import sys
sys.path.append('..')
from pytorch_learning.data.drift import DataDriftDetector, CovariateShiftDetector


def create_synthetic_data_stream():
    """Create a realistic data stream that experiences drift over time."""
    
    print("=== Creating Synthetic Data Stream ===")
    
    # Base parameters
    n_days = 30
    samples_per_day = 1000
    
    data_stream = []
    timestamps = []
    
    for day in range(n_days):
        # Simulate gradual drift and sudden changes
        
        # Base distribution parameters
        base_mean = 0.0
        base_std = 1.0
        
        # Introduce different types of drift
        if day < 10:
            # Stable period
            mean1, std1 = base_mean, base_std
            mean2, std2 = 1.0, 1.0
            target_prob = 0.5
            
        elif day < 15:
            # Gradual drift (concept drift)
            drift_factor = (day - 10) / 5.0  # 0 to 1
            mean1 = base_mean + drift_factor * 0.5
            std1 = base_std * (1 + drift_factor * 0.3)
            mean2 = 1.0 + drift_factor * 0.3
            std2 = 1.0 * (1 + drift_factor * 0.2)
            target_prob = 0.5 + drift_factor * 0.2
            
        elif day < 20:
            # Sudden drift (data pipeline change)
            mean1, std1 = 0.8, 1.4
            mean2, std2 = 1.5, 1.3
            target_prob = 0.7
            
        else:
            # Recovery period (partial)
            recovery_factor = (day - 20) / 10.0  # 0 to 1
            mean1 = 0.8 - recovery_factor * 0.4
            std1 = 1.4 - recovery_factor * 0.2
            mean2 = 1.5 - recovery_factor * 0.3
            std2 = 1.3 - recovery_factor * 0.1
            target_prob = 0.7 - recovery_factor * 0.1
        
        # Generate daily data
        daily_data = pd.DataFrame({
            'feature1': np.random.normal(mean1, std1, samples_per_day),
            'feature2': np.random.exponential(std2, samples_per_day),
            'feature3': np.random.uniform(-abs(mean2), abs(mean2), samples_per_day),
            'feature4': np.random.normal(mean1 * 0.5, std1 * 0.8, samples_per_day),
            'target': np.random.binomial(1, target_prob, samples_per_day),
            'day': day,
            'timestamp': [datetime(2024, 1, day + 1) + timedelta(hours=np.random.randint(0, 24)) 
                         for _ in range(samples_per_day)]
        })
        
        data_stream.append(daily_data)
    
    full_data = pd.concat(data_stream, ignore_index=True)
    
    print(f"Generated {len(full_data)} samples over {n_days} days")
    print(f"Features: {[col for col in full_data.columns if col.startswith('feature')]}")
    
    return full_data


def visualize_drift_over_time(data_stream: pd.DataFrame):
    """Visualize how data distributions change over time."""
    
    print("\n=== Visualizing Drift Over Time ===")
    
    # Create plots for each feature
    features = [col for col in data_stream.columns if col.startswith('feature')]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(features):
        if i >= 4:
            break
            
        ax = axes[i]
        
        # Plot daily statistics
        daily_stats = data_stream.groupby('day')[feature].agg(['mean', 'std']).reset_index()
        
        ax.plot(daily_stats['day'], daily_stats['mean'], 'b-', label='Mean', linewidth=2)
        ax.fill_between(daily_stats['day'], 
                        daily_stats['mean'] - daily_stats['std'],
                        daily_stats['mean'] + daily_stats['std'], 
                        alpha=0.3, label='±1 Std')
        
        ax.set_title(f'{feature} Over Time')
        ax.set_xlabel('Day')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drift_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Plot target distribution over time
    plt.figure(figsize=(10, 4))
    daily_target_rate = data_stream.groupby('day')['target'].mean().reset_index()
    
    plt.plot(daily_target_rate['day'], daily_target_rate['target'], 'r-', linewidth=2)
    plt.title('Target Rate Over Time (Concept Drift)')
    plt.xlabel('Day')
    plt.ylabel('Target Rate')
    plt.grid(True, alpha=0.3)
    plt.savefig('target_drift.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Drift visualizations saved as 'drift_visualization.png' and 'target_drift.png'")


def continuous_drift_monitoring_demo():
    """Demonstrate continuous drift monitoring on a data stream."""
    
    print("\n=== Continuous Drift Monitoring Demo ===")
    
    # Generate data stream
    data_stream = create_synthetic_data_stream()
    
    # Visualize the drift
    visualize_drift_over_time(data_stream)
    
    # Initialize drift detector
    detector = DataDriftDetector(alpha=0.05)
    
    # Use first week as reference
    reference_data = data_stream[data_stream['day'] < 7].drop(['day', 'timestamp'], axis=1)
    detector.fit_reference(reference_data, target_col='target')
    
    print(f"Reference period: Days 0-6 ({len(reference_data)} samples)")
    
    # Monitor drift over time
    monitoring_results = []
    
    for day in range(7, 30):
        current_day_data = data_stream[data_stream['day'] == day].drop(['day', 'timestamp'], axis=1)
        
        if len(current_day_data) == 0:
            continue
        
        # Detect drift
        report = detector.detect_drift(current_day_data, target_col='target')
        
        monitoring_results.append({
            'day': day,
            'drift_detected': report.overall_drift_detected,
            'drift_score': report.overall_drift_score,
            'drifted_features': len([d for d in report.feature_drifts if d.drift_detected]),
            'recommendations': report.recommendations
        })
        
        # Print alerts for significant drift
        if report.overall_drift_detected:
            print(f"⚠️  Day {day}: DRIFT DETECTED (score: {report.overall_drift_score:.3f})")
            drifted_features = [d.feature_name for d in report.feature_drifts if d.drift_detected]
            print(f"   Drifted features: {drifted_features}")
        else:
            print(f"✓ Day {day}: No significant drift (score: {report.overall_drift_score:.3f})")
    
    # Analyze monitoring results
    monitoring_df = pd.DataFrame(monitoring_results)
    
    print(f"\n=== Monitoring Summary ===")
    print(f"Days monitored: {len(monitoring_df)}")
    print(f"Days with drift: {monitoring_df['drift_detected'].sum()}")
    print(f"Drift detection rate: {monitoring_df['drift_detected'].mean():.1%}")
    
    # Plot drift scores over time
    plt.figure(figsize=(12, 6))
    plt.plot(monitoring_df['day'], monitoring_df['drift_score'], 'b-', linewidth=2)
    plt.axhline(y=0.05, color='r', linestyle='--', label='Drift Threshold')
    plt.fill_between(monitoring_df['day'], 0, monitoring_df['drift_score'], 
                     where=monitoring_df['drift_detected'], alpha=0.3, color='red', label='Drift Periods')
    
    plt.title('Drift Detection Over Time')
    plt.xlabel('Day')
    plt.ylabel('Drift Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('drift_monitoring_timeline.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Drift monitoring timeline saved as 'drift_monitoring_timeline.png'")


def drift_detection_methods_comparison():
    """Compare different drift detection methods."""
    
    print("\n=== Drift Detection Methods Comparison ===")
    
    # Create test scenarios
    scenarios = {
        'no_drift': {
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.exponential(1, 1000),
            'target': np.random.binomial(1, 0.5, 1000)
        },
        'mean_shift': {
            'feature1': np.random.normal(0.5, 1, 1000),  # Mean shift
            'feature2': np.random.exponential(1, 1000),
            'target': np.random.binomial(1, 0.5, 1000)
        },
        'variance_shift': {
            'feature1': np.random.normal(0, 2, 1000),  # Variance shift
            'feature2': np.random.exponential(1, 1000),
            'target': np.random.binomial(1, 0.5, 1000)
        },
        'distribution_shift': {
            'feature1': np.random.exponential(1, 1000),  # Different distribution
            'feature2': np.random.exponential(1, 1000),
            'target': np.random.binomial(1, 0.5, 1000)
        }
    }
    
    # Reference data
    reference_data = pd.DataFrame(scenarios['no_drift'])
    
    # Initialize detector
    detector = DataDriftDetector(alpha=0.05)
    detector.fit_reference(reference_data, target_col='target')
    
    # Test each scenario
    results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        current_data = pd.DataFrame(scenario_data)
        report = detector.detect_drift(current_data, target_col='target')
        
        results[scenario_name] = {
            'overall_drift': report.overall_drift_detected,
            'overall_score': report.overall_drift_score,
            'feature_drifts': {d.feature_name: d.drift_detected for d in report.feature_drifts}
        }
        
        print(f"\n{scenario_name.upper()}:")
        print(f"  Overall drift: {report.overall_drift_detected} (score: {report.overall_drift_score:.3f})")
        
        for drift in report.feature_drifts:
            print(f"  {drift.feature_name}: {drift.drift_detected} ({drift.method}, score: {drift.drift_score:.3f})")
    
    # Create comparison table
    comparison_df = pd.DataFrame([
        {
            'Scenario': scenario,
            'Overall Drift': results[scenario]['overall_drift'],
            'Overall Score': results[scenario]['overall_score'],
            'Feature1 Drift': results[scenario]['feature_drifts']['feature1'],
            'Feature2 Drift': results[scenario]['feature_drifts']['feature2']
        }
        for scenario in results.keys()
    ])
    
    print(f"\n=== Method Comparison Summary ===")
    print(comparison_df.to_string(index=False))


def real_world_drift_scenarios():
    """Demonstrate real-world drift scenarios."""
    
    print("\n=== Real-World Drift Scenarios ===")
    
    scenarios = {
        'seasonal_pattern': seasonal_drift_scenario,
        'data_pipeline_change': pipeline_change_scenario,
        'user_behavior_shift': user_behavior_scenario,
        'model_degradation': model_degradation_scenario
    }
    
    for scenario_name, scenario_func in scenarios.items():
        print(f"\n--- {scenario_name.replace('_', ' ').title()} ---")
        scenario_func()


def seasonal_drift_scenario():
    """Simulate seasonal drift patterns."""
    
    # Simulate monthly data over a year
    months = []
    for month in range(1, 13):
        # Seasonal pattern
        seasonal_factor = np.sin(2 * np.pi * month / 12)
        
        # Base parameters with seasonal variation
        mean = seasonal_factor * 2
        target_prob = 0.5 + seasonal_factor * 0.2
        
        month_data = pd.DataFrame({
            'sales': np.random.normal(mean + 10, 2, 500),
            'visitors': np.random.poisson(100 + seasonal_factor * 50, 500),
            'conversion_rate': np.random.beta(target_prob * 10, (1-target_prob) * 10, 500),
            'month': month
        })
        
        months.append(month_data)
    
    yearly_data = pd.concat(months, ignore_index=True)
    
    # Detect seasonal drift
    detector = DataDriftDetector(alpha=0.05)
    
    # Use winter months as reference
    winter_data = yearly_data[yearly_data['month'].isin([12, 1, 2])].drop('month', axis=1)
    detector.fit_reference(winter_data)
    
    # Test summer months
    summer_data = yearly_data[yearly_data['month'].isin([6, 7, 8])].drop('month', axis=1)
    report = detector.detect_drift(summer_data)
    
    print(f"Seasonal drift detected: {report.overall_drift_detected}")
    print(f"Drift score: {report.overall_drift_score:.3f}")
    print("This is expected seasonal drift - not necessarily problematic")


def pipeline_change_scenario():
    """Simulate data pipeline changes."""
    
    # Before pipeline change
    before_data = pd.DataFrame({
        'user_age': np.random.normal(35, 10, 1000),
        'purchase_amount': np.random.exponential(50, 1000),
        'session_duration': np.random.gamma(2, 30, 1000)
    })
    
    # After pipeline change (e.g., new tracking, different preprocessing)
    after_data = pd.DataFrame({
        'user_age': np.random.normal(32, 12, 1000),  # Age calculation changed
        'purchase_amount': np.random.exponential(55, 1000),  # Currency conversion
        'session_duration': np.random.gamma(2.2, 28, 1000)  # Different time tracking
    })
    
    # Detect drift
    detector = DataDriftDetector(alpha=0.05)
    detector.fit_reference(before_data)
    
    report = detector.detect_drift(after_data)
    
    print(f"Pipeline change drift detected: {report.overall_drift_detected}")
    print(f"Drift score: {report.overall_drift_score:.3f}")
    
    drifted_features = [d.feature_name for d in report.feature_drifts if d.drift_detected]
    print(f"Affected features: {drifted_features}")
    print("This requires investigation - may need model retraining")


def user_behavior_scenario():
    """Simulate user behavior changes (e.g., COVID-19 impact)."""
    
    # Pre-pandemic behavior
    pre_data = pd.DataFrame({
        'online_orders': np.random.poisson(2, 1000),
        'store_visits': np.random.poisson(5, 1000),
        'avg_basket_size': np.random.normal(50, 20, 1000)
    })
    
    # Post-pandemic behavior
    post_data = pd.DataFrame({
        'online_orders': np.random.poisson(8, 1000),  # More online
        'store_visits': np.random.poisson(1, 1000),   # Fewer store visits
        'avg_basket_size': np.random.normal(65, 25, 1000)  # Larger baskets
    })
    
    # Detect drift
    detector = DataDriftDetector(alpha=0.05)
    detector.fit_reference(pre_data)
    
    report = detector.detect_drift(post_data)
    
    print(f"User behavior drift detected: {report.overall_drift_detected}")
    print(f"Drift score: {report.overall_drift_score:.3f}")
    print("This represents a fundamental behavior change - model likely needs retraining")


def model_degradation_scenario():
    """Simulate model performance degradation due to drift."""
    
    # Simulate model predictions over time with drift
    time_periods = 10
    performance_data = []
    
    for period in range(time_periods):
        # Gradual drift in feature distribution
        drift_factor = period / time_periods
        
        # Features that drift
        feature1 = np.random.normal(drift_factor, 1, 200)
        feature2 = np.random.exponential(1 + drift_factor, 200)
        
        # True relationship changes (concept drift)
        true_prob = 1 / (1 + np.exp(-(feature1 - feature2 + drift_factor)))
        
        # Model predictions (stale model)
        model_prob = 1 / (1 + np.exp(-(feature1 - feature2)))  # No drift factor
        
        # Calculate accuracy
        predictions = (model_prob > 0.5).astype(int)
        true_labels = (np.random.random(200) < true_prob).astype(int)
        accuracy = (predictions == true_labels).mean()
        
        performance_data.append({
            'period': period,
            'accuracy': accuracy,
            'drift_factor': drift_factor,
            'feature1_mean': feature1.mean(),
            'feature2_mean': feature2.mean()
        })
    
    performance_df = pd.DataFrame(performance_data)
    
    # Plot performance degradation
    plt.figure(figsize=(10, 6))
    plt.plot(performance_df['period'], performance_df['accuracy'], 'b-', linewidth=2, label='Model Accuracy')
    plt.plot(performance_df['period'], performance_df['drift_factor'], 'r--', linewidth=2, label='Drift Factor')
    
    plt.title('Model Performance Degradation Due to Drift')
    plt.xlabel('Time Period')
    plt.ylabel('Accuracy / Drift Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('model_degradation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Initial accuracy: {performance_df.iloc[0]['accuracy']:.3f}")
    print(f"Final accuracy: {performance_df.iloc[-1]['accuracy']:.3f}")
    print(f"Performance drop: {performance_df.iloc[0]['accuracy'] - performance_df.iloc[-1]['accuracy']:.3f}")
    print("Model saved as 'model_degradation.png'")


def drift_response_strategies():
    """Demonstrate different strategies for responding to drift."""
    
    print("\n=== Drift Response Strategies ===")
    
    strategies = {
        'ignore': "Continue using current model (risky)",
        'retrain': "Retrain model on recent data",
        'ensemble': "Use ensemble of old and new models",
        'adaptive': "Use online learning or adaptive models",
        'monitor': "Increase monitoring frequency"
    }
    
    # Simulate drift severity levels and appropriate responses
    drift_scenarios = [
        {'severity': 0.02, 'type': 'mild', 'recommended': 'monitor'},
        {'severity': 0.08, 'type': 'moderate', 'recommended': 'retrain'},
        {'severity': 0.15, 'type': 'severe', 'recommended': 'ensemble'},
        {'severity': 0.25, 'type': 'critical', 'recommended': 'adaptive'}
    ]
    
    for scenario in drift_scenarios:
        print(f"\n{scenario['type'].title()} Drift (score: {scenario['severity']:.3f}):")
        print(f"  Recommended action: {scenario['recommended']}")
        print(f"  Description: {strategies[scenario['recommended']]}")
        
        # Add specific guidance
        if scenario['recommended'] == 'retrain':
            print("  - Collect recent data (last 1-2 weeks)")
            print("  - Validate model performance on holdout set")
            print("  - Deploy with A/B testing")
        elif scenario['recommended'] == 'ensemble':
            print("  - Keep old model as baseline")
            print("  - Train new model on recent data")
            print("  - Blend predictions based on confidence")
        elif scenario['recommended'] == 'adaptive':
            print("  - Implement online learning")
            print("  - Use incremental updates")
            print("  - Monitor for concept drift")


def main():
    """Run all drift detection demonstrations."""
    print("Data Drift Detection Learning Module")
    print("=" * 60)
    print("This module teaches you why 80% of model failures are data-related")
    print("and how to detect and handle data drift in production systems.\n")
    
    # Run all demonstrations
    continuous_drift_monitoring_demo()
    drift_detection_methods_comparison()
    real_world_drift_scenarios()
    drift_response_strategies()
    
    print("\n" + "=" * 60)
    print("Data Drift Detection Learning Complete!")
    print("\nKey Takeaways:")
    print("1. Data drift is inevitable in production systems")
    print("2. Multiple detection methods provide robust monitoring")
    print("3. Different types of drift require different responses")
    print("4. Early detection prevents catastrophic model failures")
    print("5. Continuous monitoring is essential for production ML")
    print("6. Not all drift is bad (e.g., seasonal patterns)")
    print("7. Have clear response strategies for different drift levels")
    
    print("\nNext Steps:")
    print("1. Implement drift monitoring in your production systems")
    print("2. Set up automated alerts for significant drift")
    print("3. Create drift response playbooks")
    print("4. Regularly review and update drift detection thresholds")
    print("5. Consider drift in model retraining schedules")


if __name__ == "__main__":
    main()
