"""
Dual-Lane: Bounded-Complexity Community Detection for Streaming Graphs

This package provides the Dual-Lane algorithm and baselines for
streaming community detection under burst churn conditions.

Modules:
    dual_lane: Main Dual-Lane algorithm implementation
    dual_lane_v2: Research implementation with proper community tracking
    baselines: Incremental LPA, Windowed Louvain, Batch Louvain
    stream_generator: Synthetic stream generation with burst patterns
    metrics: Evaluation metrics (latency, quality, statistical tests)
    run_experiments: Experiment runner script
"""

from .dual_lane import DualLane
from .dual_lane_v2 import DualLaneV2, IncrementalLPAV2, WindowedLouvainV2
from .baselines import IncrementalLPA, WindowedLouvain, BatchLouvain
from .stream_generator import StreamGenerator, StreamConfig, OperationType
from .metrics import MetricsCollector, compute_latency_metrics, compute_statistical_tests

__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[your.email@example.com]"

__all__ = [
    "DualLane",
    "DualLaneV2",
    "IncrementalLPA",
    "IncrementalLPAV2",
    "WindowedLouvain", 
    "WindowedLouvainV2",
    "BatchLouvain",
    "StreamGenerator",
    "StreamConfig",
    "OperationType",
    "MetricsCollector",
    "compute_latency_metrics",
    "compute_statistical_tests"
]
