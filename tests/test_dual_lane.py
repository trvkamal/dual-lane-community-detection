"""
Unit Tests for Dual-Lane Algorithm

Run with: pytest tests/test_dual_lane.py -v
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dual_lane import DualLane
from baselines import IncrementalLPA, WindowedLouvain, BatchLouvain
from stream_generator import StreamGenerator, StreamConfig, OperationType
from metrics import compute_latency_metrics, compute_statistical_tests


class TestDualLane:
    """Tests for the Dual-Lane algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization with default parameters."""
        dl = DualLane()
        assert dl.T == 5.0
        assert dl.K == 10
        assert dl.tau_d == 0.7
        assert dl.fast_lane.number_of_nodes() == 0
        assert dl.slow_lane.number_of_nodes() == 0
    
    def test_custom_parameters(self):
        """Test algorithm initialization with custom parameters."""
        dl = DualLane(T=10.0, K=20, tau_d=0.5)
        assert dl.T == 10.0
        assert dl.K == 20
        assert dl.tau_d == 0.5
    
    def test_add_edge(self):
        """Test adding edges."""
        dl = DualLane()
        
        latency = dl.add_edge(1, 2)
        assert latency >= 0
        assert dl.fast_lane.has_edge(1, 2) or dl.slow_lane.has_edge(1, 2)
        
        stats = dl.get_statistics()
        assert stats['additions'] == 1
    
    def test_delete_edge(self):
        """Test deleting edges."""
        dl = DualLane()
        
        dl.add_edge(1, 2)
        latency = dl.delete_edge(1, 2)
        
        assert latency >= 0
        assert not dl.fast_lane.has_edge(1, 2)
        assert not dl.slow_lane.has_edge(1, 2)
        
        stats = dl.get_statistics()
        assert stats['deletions'] == 1
    
    def test_delete_nonexistent_edge(self):
        """Test deleting an edge that doesn't exist."""
        dl = DualLane()
        
        latency = dl.delete_edge(1, 2)
        assert latency >= 0  # Should not raise exception
    
    def test_get_communities(self):
        """Test community detection."""
        dl = DualLane()
        
        # Create a simple graph with two communities
        # Community 1: 1-2-3 triangle
        dl.add_edge(1, 2)
        dl.add_edge(2, 3)
        dl.add_edge(3, 1)
        
        # Community 2: 4-5-6 triangle
        dl.add_edge(4, 5)
        dl.add_edge(5, 6)
        dl.add_edge(6, 4)
        
        communities = dl.get_communities()
        
        assert len(communities) == 6
        assert all(node in communities for node in [1, 2, 3, 4, 5, 6])
    
    def test_get_modularity(self):
        """Test modularity calculation."""
        dl = DualLane()
        
        dl.add_edge(1, 2)
        dl.add_edge(2, 3)
        dl.add_edge(3, 1)
        
        modularity = dl.get_modularity()
        
        assert -0.5 <= modularity <= 1.0
    
    def test_reset(self):
        """Test algorithm reset."""
        dl = DualLane()
        
        dl.add_edge(1, 2)
        dl.add_edge(2, 3)
        dl.reset()
        
        assert dl.fast_lane.number_of_nodes() == 0
        assert dl.slow_lane.number_of_nodes() == 0
        assert dl.get_statistics()['additions'] == 0
    
    def test_bounded_complexity(self):
        """Test that deletion complexity is bounded."""
        dl = DualLane(K=10)
        
        # Add many edges
        for i in range(100):
            dl.add_edge(i, i + 1)
        
        # All deletions should complete quickly
        for i in range(50):
            latency = dl.delete_edge(i, i + 1)
            # Latency should be reasonable (not exponential)
            assert latency < 100  # 100ms is generous bound


class TestBaselines:
    """Tests for baseline algorithms."""
    
    def test_incremental_lpa(self):
        """Test Incremental LPA."""
        lpa = IncrementalLPA()
        
        lpa.add_edge(1, 2)
        lpa.add_edge(2, 3)
        
        communities = lpa.get_communities()
        assert len(communities) >= 2
    
    def test_windowed_louvain(self):
        """Test Windowed Louvain."""
        wl = WindowedLouvain(window_size=100, recompute_interval=10)
        
        wl.add_edge(1, 2)
        wl.add_edge(2, 3)
        
        modularity = wl.get_modularity()
        assert -0.5 <= modularity <= 1.0
    
    def test_batch_louvain(self):
        """Test Batch Louvain."""
        bl = BatchLouvain()
        
        bl.add_edge(1, 2)
        bl.add_edge(2, 3)
        bl.add_edge(3, 1)
        
        communities = bl.get_communities()
        modularity = bl.get_modularity()
        
        assert len(communities) == 3
        assert modularity >= 0


class TestStreamGenerator:
    """Tests for stream generation."""
    
    def test_basic_generation(self):
        """Test basic stream generation."""
        config = StreamConfig(
            num_nodes=100,
            initial_edges=200,
            total_operations=1000,
            seed=42
        )
        
        gen = StreamGenerator(config)
        operations = list(gen.generate())
        
        assert len(operations) == 1000
    
    def test_burst_periods(self):
        """Test that burst periods occur."""
        config = StreamConfig(
            num_nodes=100,
            initial_edges=200,
            total_operations=10000,
            burst_interval=5000,
            burst_duration=500,
            seed=42
        )
        
        gen = StreamGenerator(config)
        burst_count = sum(1 for op in gen.generate() if op.is_burst)
        
        # Should have ~1000 burst operations (2 bursts × 500 duration)
        assert burst_count > 500
    
    def test_operation_types(self):
        """Test that both add and delete operations occur."""
        config = StreamConfig(
            num_nodes=100,
            initial_edges=500,
            total_operations=1000,
            normal_delete_rate=0.3,
            seed=42
        )
        
        gen = StreamGenerator(config)
        operations = list(gen.generate())
        
        adds = sum(1 for op in operations if op.op_type == OperationType.ADD)
        deletes = sum(1 for op in operations if op.op_type == OperationType.DELETE)
        
        assert adds > 0
        assert deletes > 0


class TestMetrics:
    """Tests for metrics computation."""
    
    def test_latency_metrics(self):
        """Test latency metrics computation."""
        latencies = [0.1, 0.2, 0.15, 0.5, 0.6, 0.55]
        is_burst = [False, False, False, True, True, True]
        
        metrics = compute_latency_metrics(latencies, is_burst)
        
        assert metrics.normal_mean > 0
        assert metrics.burst_mean > 0
        assert metrics.stability_ratio > 0
    
    def test_statistical_tests(self):
        """Test statistical significance tests."""
        sample1 = [1.0, 1.1, 1.2, 0.9, 1.0]
        sample2 = [0.1, 0.2, 0.15, 0.1, 0.12]
        
        tests = compute_statistical_tests(sample1, sample2)
        
        assert tests.p_value < 0.05  # Should be significant
        assert tests.significant
        assert tests.cohens_d > 0  # sample1 > sample2


class TestIntegration:
    """Integration tests."""
    
    def test_full_pipeline(self):
        """Test complete experiment pipeline."""
        config = StreamConfig(
            num_nodes=100,
            initial_edges=200,
            total_operations=500,
            seed=42
        )
        
        dl = DualLane()
        bl = BatchLouvain()
        gen = StreamGenerator(config)
        
        latencies = []
        is_burst = []
        
        for op in gen.generate():
            if op.op_type == OperationType.ADD:
                latency = dl.add_edge(op.u, op.v)
                bl.add_edge(op.u, op.v)
            else:
                latency = dl.delete_edge(op.u, op.v)
                bl.delete_edge(op.u, op.v)
            
            latencies.append(latency)
            is_burst.append(op.is_burst)
        
        # Compute metrics
        metrics = compute_latency_metrics(latencies, is_burst)
        dl_modularity = dl.get_modularity()
        bl_modularity = bl.get_modularity()
        
        # Verify reasonable results
        assert metrics.normal_mean > 0
        assert dl_modularity >= 0 or bl_modularity == 0
        
        if bl_modularity > 0:
            quality_ratio = dl_modularity / bl_modularity
            # Should maintain reasonable quality
            assert quality_ratio > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
