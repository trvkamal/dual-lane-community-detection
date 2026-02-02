"""
Metrics for Evaluating Streaming Community Detection Algorithms

This module provides functions for computing:
- Quality metrics (modularity, NMI)
- Stability metrics (latency ratios)
- Statistical significance tests
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import community as community_louvain
import networkx as nx


@dataclass
class LatencyMetrics:
    """Latency statistics for an algorithm run."""
    normal_mean: float
    normal_std: float
    normal_ci: Tuple[float, float]
    
    burst_mean: float
    burst_std: float
    burst_ci: Tuple[float, float]
    
    stability_ratio: float  # burst_mean / normal_mean
    
    all_latencies: List[float]
    normal_latencies: List[float]
    burst_latencies: List[float]


@dataclass
class QualityMetrics:
    """Quality metrics for community detection."""
    modularity: float
    relative_quality: float  # As percentage of batch Louvain
    num_communities: int


@dataclass
class StatisticalTests:
    """Results of statistical significance tests."""
    t_statistic: float
    p_value: float
    cohens_d: float
    
    significant: bool  # p < 0.05
    effect_size: str  # small/medium/large/very_large


def compute_latency_metrics(
    latencies: List[float],
    is_burst: List[bool],
    confidence: float = 0.95
) -> LatencyMetrics:
    """
    Compute latency metrics from a list of operation latencies.
    
    Parameters
    ----------
    latencies : list of float
        Latency values in milliseconds
    is_burst : list of bool
        Whether each operation was during a burst period
    confidence : float
        Confidence level for intervals (default: 0.95)
    
    Returns
    -------
    LatencyMetrics
        Computed latency statistics
    """
    latencies = np.array(latencies)
    is_burst = np.array(is_burst)
    
    normal_mask = ~is_burst
    burst_mask = is_burst
    
    normal_latencies = latencies[normal_mask]
    burst_latencies = latencies[burst_mask]
    
    def compute_ci(data: np.ndarray, conf: float) -> Tuple[float, float]:
        """Compute confidence interval."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        sem = stats.sem(data)
        
        if sem == 0:
            return (mean, mean)
        
        t_val = stats.t.ppf((1 + conf) / 2, len(data) - 1)
        margin = t_val * sem
        return (mean - margin, mean + margin)
    
    normal_mean = np.mean(normal_latencies) if len(normal_latencies) > 0 else 0
    normal_std = np.std(normal_latencies) if len(normal_latencies) > 0 else 0
    normal_ci = compute_ci(normal_latencies, confidence)
    
    burst_mean = np.mean(burst_latencies) if len(burst_latencies) > 0 else 0
    burst_std = np.std(burst_latencies) if len(burst_latencies) > 0 else 0
    burst_ci = compute_ci(burst_latencies, confidence)
    
    stability_ratio = burst_mean / normal_mean if normal_mean > 0 else float('inf')
    
    return LatencyMetrics(
        normal_mean=normal_mean,
        normal_std=normal_std,
        normal_ci=normal_ci,
        burst_mean=burst_mean,
        burst_std=burst_std,
        burst_ci=burst_ci,
        stability_ratio=stability_ratio,
        all_latencies=latencies.tolist(),
        normal_latencies=normal_latencies.tolist(),
        burst_latencies=burst_latencies.tolist()
    )


def compute_quality_metrics(
    algorithm_modularity: float,
    reference_modularity: float
) -> QualityMetrics:
    """
    Compute quality metrics relative to a reference.
    
    Parameters
    ----------
    algorithm_modularity : float
        Modularity achieved by the algorithm
    reference_modularity : float
        Reference modularity (typically batch Louvain)
    
    Returns
    -------
    QualityMetrics
        Computed quality metrics
    """
    relative = (
        algorithm_modularity / reference_modularity * 100
        if reference_modularity > 0 else 0.0
    )
    
    return QualityMetrics(
        modularity=algorithm_modularity,
        relative_quality=relative,
        num_communities=0  # To be filled by caller
    )


def compute_statistical_tests(
    sample1: List[float],
    sample2: List[float],
    alternative: str = 'two-sided'
) -> StatisticalTests:
    """
    Compute statistical significance tests between two samples.
    
    Performs independent samples t-test and computes Cohen's d effect size.
    
    Parameters
    ----------
    sample1 : list of float
        First sample (e.g., baseline latencies)
    sample2 : list of float
        Second sample (e.g., Dual-Lane latencies)
    alternative : str
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
    
    Returns
    -------
    StatisticalTests
        Test results including t-statistic, p-value, and effect size
    """
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    # Independent samples t-test
    t_stat, p_value = stats.ttest_ind(sample1, sample2, alternative=alternative)
    
    # Cohen's d effect size
    n1, n2 = len(sample1), len(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    )
    
    cohens_d = (np.mean(sample1) - np.mean(sample2)) / pooled_std if pooled_std > 0 else 0
    
    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_size = "negligible"
    elif abs_d < 0.5:
        effect_size = "small"
    elif abs_d < 0.8:
        effect_size = "medium"
    elif abs_d < 1.2:
        effect_size = "large"
    else:
        effect_size = "very_large"
    
    return StatisticalTests(
        t_statistic=t_stat,
        p_value=p_value,
        cohens_d=cohens_d,
        significant=p_value < 0.05,
        effect_size=effect_size
    )


def compute_normalized_mutual_information(
    communities1: Dict[int, int],
    communities2: Dict[int, int]
) -> float:
    """
    Compute Normalized Mutual Information between two community structures.
    
    Parameters
    ----------
    communities1 : dict
        First community assignment (node -> community_id)
    communities2 : dict
        Second community assignment (node -> community_id)
    
    Returns
    -------
    float
        NMI score in range [0, 1]
    """
    # Get common nodes
    common_nodes = set(communities1.keys()) & set(communities2.keys())
    
    if not common_nodes:
        return 0.0
    
    # Create label arrays
    labels1 = [communities1[n] for n in sorted(common_nodes)]
    labels2 = [communities2[n] for n in sorted(common_nodes)]
    
    # Build contingency table
    from collections import Counter
    
    # Map to consecutive integers
    unique1 = list(set(labels1))
    unique2 = list(set(labels2))
    
    map1 = {v: i for i, v in enumerate(unique1)}
    map2 = {v: i for i, v in enumerate(unique2)}
    
    labels1 = [map1[l] for l in labels1]
    labels2 = [map2[l] for l in labels2]
    
    n = len(labels1)
    
    # Compute entropies and mutual information
    counter1 = Counter(labels1)
    counter2 = Counter(labels2)
    joint_counter = Counter(zip(labels1, labels2))
    
    # H(X)
    h1 = -sum(
        (c / n) * np.log(c / n)
        for c in counter1.values()
    )
    
    # H(Y)
    h2 = -sum(
        (c / n) * np.log(c / n)
        for c in counter2.values()
    )
    
    # Mutual Information
    mi = 0.0
    for (l1, l2), joint_count in joint_counter.items():
        p_joint = joint_count / n
        p1 = counter1[l1] / n
        p2 = counter2[l2] / n
        
        if p_joint > 0:
            mi += p_joint * np.log(p_joint / (p1 * p2))
    
    # Normalized MI
    if h1 + h2 > 0:
        nmi = 2 * mi / (h1 + h2)
    else:
        nmi = 0.0
    
    return nmi


def compute_volatility_rate(
    stats: Dict,
    total_deletions: int
) -> float:
    """
    Compute volatility rate (fraction of deletions from fast lane).
    
    Parameters
    ----------
    stats : dict
        Algorithm statistics containing 'fast_lane_deletions'
    total_deletions : int
        Total number of deletion operations
    
    Returns
    -------
    float
        Volatility rate in range [0, 1]
    """
    if total_deletions == 0:
        return 0.0
    
    fast_lane_deletions = stats.get('fast_lane_deletions', 0)
    return fast_lane_deletions / total_deletions


class MetricsCollector:
    """
    Collects and aggregates metrics during algorithm evaluation.
    
    Example
    -------
    >>> collector = MetricsCollector()
    >>> for op in stream:
    ...     latency = algorithm.process(op)
    ...     collector.record_latency(latency, is_burst=op.is_burst)
    >>> metrics = collector.get_summary()
    """
    
    def __init__(self):
        self.latencies: List[float] = []
        self.is_burst: List[bool] = []
        self.modularities: List[float] = []
        self.timestamps: List[float] = []
    
    def record_latency(self, latency: float, is_burst: bool = False, timestamp: float = None):
        """Record a latency measurement."""
        self.latencies.append(latency)
        self.is_burst.append(is_burst)
        if timestamp is not None:
            self.timestamps.append(timestamp)
    
    def record_modularity(self, modularity: float):
        """Record a modularity measurement."""
        self.modularities.append(modularity)
    
    def get_latency_metrics(self) -> LatencyMetrics:
        """Get computed latency metrics."""
        return compute_latency_metrics(self.latencies, self.is_burst)
    
    def get_summary(self) -> Dict:
        """Get summary of all collected metrics."""
        latency_metrics = self.get_latency_metrics()
        
        return {
            'latency': {
                'normal_mean_ms': latency_metrics.normal_mean,
                'normal_std_ms': latency_metrics.normal_std,
                'normal_ci': latency_metrics.normal_ci,
                'burst_mean_ms': latency_metrics.burst_mean,
                'burst_std_ms': latency_metrics.burst_std,
                'burst_ci': latency_metrics.burst_ci,
                'stability_ratio': latency_metrics.stability_ratio
            },
            'quality': {
                'final_modularity': self.modularities[-1] if self.modularities else 0,
                'mean_modularity': np.mean(self.modularities) if self.modularities else 0
            },
            'counts': {
                'total_operations': len(self.latencies),
                'normal_operations': sum(1 for b in self.is_burst if not b),
                'burst_operations': sum(1 for b in self.is_burst if b)
            }
        }
    
    def reset(self):
        """Reset collected data."""
        self.latencies.clear()
        self.is_burst.clear()
        self.modularities.clear()
        self.timestamps.clear()


def format_results_table(
    results: Dict[str, Dict],
    reference_modularity: float
) -> str:
    """
    Format results as a text table.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm names to their metrics
    reference_modularity : float
        Reference modularity for quality percentage
    
    Returns
    -------
    str
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Algorithm':<20} {'Normal (ms)':<12} {'Burst (ms)':<12} "
                 f"{'Ratio':<8} {'Quality':<10} {'Q (%)':<8}")
    lines.append("-" * 80)
    
    for name, metrics in results.items():
        latency = metrics['latency']
        quality = metrics['quality']
        
        q_pct = (quality['final_modularity'] / reference_modularity * 100
                 if reference_modularity > 0 else 0)
        
        lines.append(
            f"{name:<20} "
            f"{latency['normal_mean_ms']:<12.3f} "
            f"{latency['burst_mean_ms']:<12.3f} "
            f"{latency['stability_ratio']:<8.2f} "
            f"{quality['final_modularity']:<10.4f} "
            f"{q_pct:<8.1f}%"
        )
    
    lines.append("=" * 80)
    return "\n".join(lines)


def demo():
    """Demonstrate metrics computation."""
    print("Metrics Demo")
    print("=" * 50)
    
    # Simulate latency data
    np.random.seed(42)
    
    normal_latencies = np.random.exponential(0.01, 1000)  # ~10μs average
    burst_latencies = np.random.exponential(0.05, 200)    # ~50μs average
    
    latencies = list(normal_latencies) + list(burst_latencies)
    is_burst = [False] * 1000 + [True] * 200
    
    metrics = compute_latency_metrics(latencies, is_burst)
    
    print(f"Normal latency: {metrics.normal_mean:.4f}ms ± {metrics.normal_std:.4f}ms")
    print(f"Burst latency:  {metrics.burst_mean:.4f}ms ± {metrics.burst_std:.4f}ms")
    print(f"Stability ratio: {metrics.stability_ratio:.2f}")
    
    # Statistical test
    tests = compute_statistical_tests(
        list(burst_latencies),
        list(normal_latencies)
    )
    
    print(f"\nStatistical Test:")
    print(f"  t-statistic: {tests.t_statistic:.2f}")
    print(f"  p-value: {tests.p_value:.2e}")
    print(f"  Cohen's d: {tests.cohens_d:.2f} ({tests.effect_size})")
    print(f"  Significant: {tests.significant}")


if __name__ == "__main__":
    demo()
