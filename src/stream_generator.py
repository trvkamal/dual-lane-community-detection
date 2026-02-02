"""
Stream Generator for Streaming Graph Experiments

This module generates synthetic edge streams with configurable burst churn
patterns for evaluating streaming community detection algorithms.

The burst model simulates real-world scenarios where deletion rates
spike periodically (e.g., spam cleanup, user departures, system failures).
"""

import random
import time
from typing import Generator, Tuple, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx


class OperationType(Enum):
    """Types of stream operations."""
    ADD = "add"
    DELETE = "delete"


@dataclass
class StreamOperation:
    """A single operation in the edge stream."""
    op_type: OperationType
    u: int
    v: int
    timestamp: float
    is_burst: bool = False


@dataclass
class StreamConfig:
    """Configuration for stream generation."""
    # Graph parameters
    num_nodes: int = 5000
    initial_edges: int = 10000
    
    # Stream parameters
    total_operations: int = 100000
    window_size: int = 20000  # Sliding window for deletions
    
    # Normal operation parameters
    normal_delete_rate: float = 0.15  # 15% deletions normally
    
    # Burst parameters
    burst_interval: int = 50000  # Operations between bursts
    burst_delete_rate: float = 0.50  # 50% deletions during burst
    burst_duration: int = 5000  # Operations per burst
    
    # Reproducibility
    seed: Optional[int] = 42


class StreamGenerator:
    """
    Generates edge streams with burst churn patterns.
    
    The generator creates a stream of edge additions and deletions
    that simulates realistic streaming graph workloads with periodic
    burst events where deletion rates spike dramatically.
    
    Parameters
    ----------
    config : StreamConfig
        Configuration for stream generation
    
    Example
    -------
    >>> config = StreamConfig(num_nodes=1000, total_operations=10000)
    >>> gen = StreamGenerator(config)
    >>> for op in gen.generate():
    ...     if op.op_type == OperationType.ADD:
    ...         algorithm.add_edge(op.u, op.v)
    ...     else:
    ...         algorithm.delete_edge(op.u, op.v)
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        
        if config.seed is not None:
            random.seed(config.seed)
        
        # Graph state
        self.graph = nx.Graph()
        self.edge_list: List[Tuple[int, int]] = []  # For deletion sampling
        
        # Statistics
        self.stats = {
            'total_additions': 0,
            'total_deletions': 0,
            'burst_additions': 0,
            'burst_deletions': 0,
            'normal_additions': 0,
            'normal_deletions': 0
        }
    
    def _normalize_edge(self, u: int, v: int) -> Tuple[int, int]:
        """Normalize edge representation."""
        return (min(u, v), max(u, v))
    
    def _initialize_graph(self):
        """Initialize graph with random edges."""
        # Add initial edges
        edges_added = 0
        attempts = 0
        max_attempts = self.config.initial_edges * 10
        
        while edges_added < self.config.initial_edges and attempts < max_attempts:
            u = random.randint(0, self.config.num_nodes - 1)
            v = random.randint(0, self.config.num_nodes - 1)
            
            if u != v and not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v)
                self.edge_list.append(self._normalize_edge(u, v))
                edges_added += 1
            
            attempts += 1
        
        print(f"Initialized graph with {self.graph.number_of_edges()} edges")
    
    def _is_burst_period(self, operation_idx: int) -> bool:
        """Check if current operation is during a burst period."""
        # Burst starts every burst_interval operations
        position_in_cycle = operation_idx % self.config.burst_interval
        return position_in_cycle < self.config.burst_duration
    
    def _sample_new_edge(self) -> Tuple[int, int]:
        """Sample a new edge to add."""
        attempts = 0
        max_attempts = 1000
        
        while attempts < max_attempts:
            u = random.randint(0, self.config.num_nodes - 1)
            v = random.randint(0, self.config.num_nodes - 1)
            
            if u != v and not self.graph.has_edge(u, v):
                return u, v
            
            attempts += 1
        
        # Fallback: add new node
        new_node = self.graph.number_of_nodes()
        existing = random.choice(list(self.graph.nodes())) if self.graph.nodes() else 0
        return new_node, existing
    
    def _sample_edge_to_delete(self) -> Optional[Tuple[int, int]]:
        """Sample an edge from the sliding window for deletion."""
        if not self.edge_list:
            return None
        
        # Sample from recent edges (sliding window)
        window_start = max(0, len(self.edge_list) - self.config.window_size)
        window = self.edge_list[window_start:]
        
        if not window:
            return None
        
        # Random sampling from window
        edge = random.choice(window)
        return edge
    
    def generate(self) -> Generator[StreamOperation, None, None]:
        """
        Generate the edge stream.
        
        Yields
        ------
        StreamOperation
            Each operation in the stream
        """
        # Initialize graph
        self._initialize_graph()
        
        # Yield initial edges as additions (optional)
        start_time = time.time()
        
        for i in range(self.config.total_operations):
            is_burst = self._is_burst_period(i)
            delete_rate = (
                self.config.burst_delete_rate if is_burst 
                else self.config.normal_delete_rate
            )
            
            timestamp = start_time + i * 0.001  # Simulated timestamp
            
            # Decide operation type
            if random.random() < delete_rate and self.edge_list:
                # Delete operation
                edge = self._sample_edge_to_delete()
                
                if edge and self.graph.has_edge(*edge):
                    u, v = edge
                    self.graph.remove_edge(u, v)
                    
                    # Remove from edge list
                    if edge in self.edge_list:
                        self.edge_list.remove(edge)
                    
                    # Update stats
                    self.stats['total_deletions'] += 1
                    if is_burst:
                        self.stats['burst_deletions'] += 1
                    else:
                        self.stats['normal_deletions'] += 1
                    
                    yield StreamOperation(
                        op_type=OperationType.DELETE,
                        u=u, v=v,
                        timestamp=timestamp,
                        is_burst=is_burst
                    )
                else:
                    # Edge already deleted, do addition instead
                    u, v = self._sample_new_edge()
                    self.graph.add_edge(u, v)
                    self.edge_list.append(self._normalize_edge(u, v))
                    
                    self.stats['total_additions'] += 1
                    if is_burst:
                        self.stats['burst_additions'] += 1
                    else:
                        self.stats['normal_additions'] += 1
                    
                    yield StreamOperation(
                        op_type=OperationType.ADD,
                        u=u, v=v,
                        timestamp=timestamp,
                        is_burst=is_burst
                    )
            else:
                # Add operation
                u, v = self._sample_new_edge()
                self.graph.add_edge(u, v)
                self.edge_list.append(self._normalize_edge(u, v))
                
                self.stats['total_additions'] += 1
                if is_burst:
                    self.stats['burst_additions'] += 1
                else:
                    self.stats['normal_additions'] += 1
                
                yield StreamOperation(
                    op_type=OperationType.ADD,
                    u=u, v=v,
                    timestamp=timestamp,
                    is_burst=is_burst
                )
    
    def get_statistics(self) -> dict:
        """Get generation statistics."""
        total = self.stats['total_additions'] + self.stats['total_deletions']
        return {
            **self.stats,
            'total_operations': total,
            'deletion_rate': self.stats['total_deletions'] / total if total > 0 else 0,
            'final_nodes': self.graph.number_of_nodes(),
            'final_edges': self.graph.number_of_edges()
        }


class RealDatasetStream:
    """
    Load and stream edges from real dataset files.
    
    Supports common formats like edge lists from SNAP datasets.
    
    Parameters
    ----------
    filepath : str
        Path to edge list file
    config : StreamConfig
        Configuration for burst simulation
    """
    
    def __init__(self, filepath: str, config: StreamConfig):
        self.filepath = filepath
        self.config = config
        
        if config.seed is not None:
            random.seed(config.seed)
        
        self.edges: List[Tuple[int, int]] = []
        self.stats = {
            'total_additions': 0,
            'total_deletions': 0,
            'burst_additions': 0,
            'burst_deletions': 0
        }
    
    def _load_edges(self):
        """Load edges from file."""
        print(f"Loading edges from {self.filepath}...")
        
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#') or line.startswith('%'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        u, v = int(parts[0]), int(parts[1])
                        if u != v:
                            self.edges.append((min(u, v), max(u, v)))
                    except ValueError:
                        continue
        
        # Remove duplicates
        self.edges = list(set(self.edges))
        random.shuffle(self.edges)
        
        print(f"Loaded {len(self.edges)} unique edges")
    
    def _is_burst_period(self, operation_idx: int) -> bool:
        """Check if current operation is during a burst period."""
        position_in_cycle = operation_idx % self.config.burst_interval
        return position_in_cycle < self.config.burst_duration
    
    def generate(self) -> Generator[StreamOperation, None, None]:
        """Generate stream from real dataset."""
        self._load_edges()
        
        # Track active edges for deletion
        active_edges: List[Tuple[int, int]] = []
        start_time = time.time()
        operation_idx = 0
        edge_idx = 0
        
        while operation_idx < self.config.total_operations and edge_idx < len(self.edges):
            is_burst = self._is_burst_period(operation_idx)
            delete_rate = (
                self.config.burst_delete_rate if is_burst 
                else self.config.normal_delete_rate
            )
            
            timestamp = start_time + operation_idx * 0.001
            
            # Decide operation
            if random.random() < delete_rate and len(active_edges) > self.config.window_size // 10:
                # Delete from recent edges
                window_start = max(0, len(active_edges) - self.config.window_size)
                window = active_edges[window_start:]
                
                if window:
                    edge = random.choice(window)
                    u, v = edge
                    active_edges.remove(edge)
                    
                    self.stats['total_deletions'] += 1
                    if is_burst:
                        self.stats['burst_deletions'] += 1
                    
                    yield StreamOperation(
                        op_type=OperationType.DELETE,
                        u=u, v=v,
                        timestamp=timestamp,
                        is_burst=is_burst
                    )
                    operation_idx += 1
                    continue
            
            # Add next edge from dataset
            u, v = self.edges[edge_idx]
            active_edges.append((u, v))
            edge_idx += 1
            
            self.stats['total_additions'] += 1
            if is_burst:
                self.stats['burst_additions'] += 1
            
            yield StreamOperation(
                op_type=OperationType.ADD,
                u=u, v=v,
                timestamp=timestamp,
                is_burst=is_burst
            )
            operation_idx += 1
    
    def get_statistics(self) -> dict:
        """Get stream statistics."""
        total = self.stats['total_additions'] + self.stats['total_deletions']
        return {
            **self.stats,
            'total_operations': total,
            'deletion_rate': self.stats['total_deletions'] / total if total > 0 else 0
        }


def demo():
    """Demonstrate stream generation."""
    print("Stream Generator Demo")
    print("=" * 50)
    
    config = StreamConfig(
        num_nodes=1000,
        initial_edges=5000,
        total_operations=10000,
        burst_interval=5000,
        burst_duration=500,
        seed=42
    )
    
    gen = StreamGenerator(config)
    
    # Count operations
    add_count = 0
    del_count = 0
    burst_add = 0
    burst_del = 0
    
    for i, op in enumerate(gen.generate()):
        if op.op_type == OperationType.ADD:
            add_count += 1
            if op.is_burst:
                burst_add += 1
        else:
            del_count += 1
            if op.is_burst:
                burst_del += 1
        
        # Progress
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1} operations...")
    
    print(f"\nStream Statistics:")
    print(f"  Total additions: {add_count}")
    print(f"  Total deletions: {del_count}")
    print(f"  Burst additions: {burst_add}")
    print(f"  Burst deletions: {burst_del}")
    print(f"  Overall delete rate: {del_count / (add_count + del_count):.2%}")
    
    stats = gen.get_statistics()
    print(f"\nFinal graph: {stats['final_nodes']} nodes, {stats['final_edges']} edges")


if __name__ == "__main__":
    demo()
