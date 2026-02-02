"""
Baseline Algorithms for Streaming Community Detection

This module implements baseline algorithms for comparison with Dual-Lane:
- Incremental Label Propagation Algorithm (LPA)
- Windowed Louvain
- Batch Louvain (for quality reference)

These implementations are designed to match the experimental methodology
described in the research paper.
"""

import time
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import community as community_louvain
import random


class IncrementalLPA:
    """
    Incremental Label Propagation Algorithm for streaming graphs.
    
    LPA assigns community labels to nodes by iteratively adopting
    the most frequent label among neighbors. This incremental version
    updates labels only for affected nodes after each graph modification.
    
    Key limitation: O(degree) deletion complexity for hub nodes.
    
    Reference:
        Raghavan, U. N., Albert, R., & Kumara, S. (2007).
        Near linear time algorithm to detect community structures in large-scale networks.
        Physical Review E, 76(3), 036106.
    
    Parameters
    ----------
    max_iterations : int
        Maximum iterations for label convergence. Default: 100
    convergence_threshold : float
        Fraction of nodes that must be stable for convergence. Default: 0.95
    """
    
    def __init__(self, max_iterations: int = 100, convergence_threshold: float = 0.95):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        self.graph = nx.Graph()
        self.labels: Dict[int, int] = {}
        self.next_label = 0
        
        self.stats = {
            'additions': 0,
            'deletions': 0,
            'label_updates': 0,
            'convergence_failures': 0
        }
    
    def _get_dominant_label(self, node: int) -> int:
        """Get the most frequent label among node's neighbors."""
        if self.graph.degree(node) == 0:
            return self.labels.get(node, self._new_label())
        
        label_counts = defaultdict(int)
        for neighbor in self.graph.neighbors(node):
            if neighbor in self.labels:
                label_counts[self.labels[neighbor]] += 1
        
        if not label_counts:
            return self.labels.get(node, self._new_label())
        
        # Random tie-breaking for stability
        max_count = max(label_counts.values())
        candidates = [l for l, c in label_counts.items() if c == max_count]
        return random.choice(candidates)
    
    def _new_label(self) -> int:
        """Generate a new unique label."""
        label = self.next_label
        self.next_label += 1
        return label
    
    def _propagate_labels(self, affected_nodes: Optional[List[int]] = None):
        """Run label propagation on affected nodes."""
        if affected_nodes is None:
            affected_nodes = list(self.graph.nodes())
        
        if not affected_nodes:
            return
        
        # Expand to include neighbors for more stable propagation
        nodes_to_update = set(affected_nodes)
        for node in affected_nodes:
            if node in self.graph:
                nodes_to_update.update(self.graph.neighbors(node))
        
        nodes_to_update = [n for n in nodes_to_update if n in self.graph]
        
        for _ in range(self.max_iterations):
            changes = 0
            random.shuffle(nodes_to_update)
            
            for node in nodes_to_update:
                old_label = self.labels.get(node)
                new_label = self._get_dominant_label(node)
                
                if new_label != old_label:
                    self.labels[node] = new_label
                    changes += 1
                    self.stats['label_updates'] += 1
            
            # Check convergence
            if changes == 0 or changes / len(nodes_to_update) < (1 - self.convergence_threshold):
                break
        else:
            self.stats['convergence_failures'] += 1
    
    def add_edge(self, u: int, v: int) -> float:
        """
        Add an edge to the graph and update labels.
        
        Returns latency in milliseconds.
        """
        start_time = time.perf_counter()
        
        if self.graph.has_edge(u, v):
            return (time.perf_counter() - start_time) * 1000
        
        # Initialize labels for new nodes
        if u not in self.labels:
            self.labels[u] = self._new_label()
        if v not in self.labels:
            self.labels[v] = self._new_label()
        
        self.graph.add_edge(u, v)
        self.stats['additions'] += 1
        
        # Propagate labels from affected nodes
        self._propagate_labels([u, v])
        
        return (time.perf_counter() - start_time) * 1000
    
    def delete_edge(self, u: int, v: int) -> float:
        """
        Delete an edge from the graph and update labels.
        
        Note: This has O(degree) complexity for high-degree nodes,
        which causes latency spikes during burst churn.
        
        Returns latency in milliseconds.
        """
        start_time = time.perf_counter()
        
        if not self.graph.has_edge(u, v):
            return (time.perf_counter() - start_time) * 1000
        
        self.graph.remove_edge(u, v)
        self.stats['deletions'] += 1
        
        # Handle isolated nodes
        nodes_to_remove = []
        if u in self.graph and self.graph.degree(u) == 0:
            nodes_to_remove.append(u)
        if v in self.graph and self.graph.degree(v) == 0:
            nodes_to_remove.append(v)
        
        for node in nodes_to_remove:
            self.graph.remove_node(node)
            if node in self.labels:
                del self.labels[node]
        
        # Re-propagate labels (expensive for high-degree nodes)
        remaining = [n for n in [u, v] if n in self.graph]
        if remaining:
            self._propagate_labels(remaining)
        
        return (time.perf_counter() - start_time) * 1000
    
    def get_communities(self) -> Dict[int, int]:
        """Get current community assignments."""
        return dict(self.labels)
    
    def get_modularity(self) -> float:
        """Calculate modularity of current labeling."""
        if self.graph.number_of_edges() == 0:
            return 0.0
        return community_louvain.modularity(self.labels, self.graph)
    
    def get_statistics(self) -> Dict:
        """Get algorithm statistics."""
        return {
            **self.stats,
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_communities': len(set(self.labels.values()))
        }
    
    def reset(self):
        """Reset algorithm state."""
        self.graph.clear()
        self.labels.clear()
        self.next_label = 0
        self.stats = {k: 0 for k in self.stats}


class WindowedLouvain:
    """
    Windowed Louvain algorithm for streaming graphs.
    
    Maintains a sliding window of recent edges and periodically
    recomputes communities using the Louvain algorithm.
    
    Note: While this achieves high quality, it provides batch-style
    semantics rather than true streaming, and recomputation can be
    expensive for large windows.
    
    Parameters
    ----------
    window_size : int
        Maximum number of edges to maintain. Default: 20000
    recompute_interval : int
        Number of operations between Louvain recomputations. Default: 100
    """
    
    def __init__(self, window_size: int = 20000, recompute_interval: int = 100):
        self.window_size = window_size
        self.recompute_interval = recompute_interval
        
        self.graph = nx.Graph()
        self.edge_order: List[Tuple[int, int]] = []  # FIFO queue
        self.communities: Dict[int, int] = {}
        self.operation_count = 0
        self._needs_recompute = True
        
        self.stats = {
            'additions': 0,
            'deletions': 0,
            'window_evictions': 0,
            'recomputations': 0
        }
    
    def _normalize_edge(self, u: int, v: int) -> Tuple[int, int]:
        """Normalize edge representation."""
        return (min(u, v), max(u, v))
    
    def _maybe_recompute(self):
        """Recompute communities if needed."""
        self.operation_count += 1
        
        if self._needs_recompute or self.operation_count >= self.recompute_interval:
            self._recompute_communities()
            self.operation_count = 0
            self._needs_recompute = False
    
    def _recompute_communities(self):
        """Run Louvain algorithm on current window."""
        if self.graph.number_of_nodes() == 0:
            self.communities = {}
        else:
            try:
                self.communities = community_louvain.best_partition(self.graph)
            except Exception:
                self.communities = {n: i for i, n in enumerate(self.graph.nodes())}
        
        self.stats['recomputations'] += 1
    
    def _evict_oldest(self):
        """Remove oldest edge when window is full."""
        while len(self.edge_order) > self.window_size:
            u, v = self.edge_order.pop(0)
            if self.graph.has_edge(u, v):
                self.graph.remove_edge(u, v)
                
                # Clean isolated nodes
                if self.graph.degree(u) == 0:
                    self.graph.remove_node(u)
                if v in self.graph and self.graph.degree(v) == 0:
                    self.graph.remove_node(v)
            
            self.stats['window_evictions'] += 1
        
        self._needs_recompute = True
    
    def add_edge(self, u: int, v: int) -> float:
        """Add an edge to the window."""
        start_time = time.perf_counter()
        
        edge = self._normalize_edge(u, v)
        
        if not self.graph.has_edge(u, v):
            self.graph.add_edge(u, v)
            self.edge_order.append(edge)
            self.stats['additions'] += 1
            self._needs_recompute = True
        
        # Evict if window is full
        self._evict_oldest()
        
        # Maybe recompute communities
        self._maybe_recompute()
        
        return (time.perf_counter() - start_time) * 1000
    
    def delete_edge(self, u: int, v: int) -> float:
        """Delete an edge from the window."""
        start_time = time.perf_counter()
        
        edge = self._normalize_edge(u, v)
        
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            
            # Remove from order list
            if edge in self.edge_order:
                self.edge_order.remove(edge)
            
            # Clean isolated nodes
            if self.graph.degree(u) == 0:
                self.graph.remove_node(u)
            if v in self.graph and self.graph.degree(v) == 0:
                self.graph.remove_node(v)
            
            self.stats['deletions'] += 1
            self._needs_recompute = True
        
        # Maybe recompute communities
        self._maybe_recompute()
        
        return (time.perf_counter() - start_time) * 1000
    
    def get_communities(self) -> Dict[int, int]:
        """Get current community assignments."""
        if self._needs_recompute:
            self._recompute_communities()
        return dict(self.communities)
    
    def get_modularity(self) -> float:
        """Calculate modularity of current communities."""
        if self.graph.number_of_edges() == 0:
            return 0.0
        communities = self.get_communities()
        return community_louvain.modularity(communities, self.graph)
    
    def get_statistics(self) -> Dict:
        """Get algorithm statistics."""
        return {
            **self.stats,
            'window_fill': len(self.edge_order) / self.window_size,
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_communities': len(set(self.communities.values())) if self.communities else 0
        }
    
    def reset(self):
        """Reset algorithm state."""
        self.graph.clear()
        self.edge_order.clear()
        self.communities.clear()
        self.operation_count = 0
        self._needs_recompute = True
        self.stats = {k: 0 for k in self.stats}


class BatchLouvain:
    """
    Batch Louvain algorithm for quality reference.
    
    Runs full Louvain algorithm on the complete graph snapshot.
    Used as the quality baseline (100% reference) in experiments.
    
    Note: This is not suitable for streaming as it requires
    full recomputation after each modification.
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.communities: Dict[int, int] = {}
        self._dirty = True
        
        self.stats = {
            'additions': 0,
            'deletions': 0,
            'recomputations': 0
        }
    
    def add_edge(self, u: int, v: int) -> float:
        """Add an edge."""
        start_time = time.perf_counter()
        
        if not self.graph.has_edge(u, v):
            self.graph.add_edge(u, v)
            self.stats['additions'] += 1
            self._dirty = True
        
        return (time.perf_counter() - start_time) * 1000
    
    def delete_edge(self, u: int, v: int) -> float:
        """Delete an edge."""
        start_time = time.perf_counter()
        
        if self.graph.has_edge(u, v):
            self.graph.remove_edge(u, v)
            
            if self.graph.degree(u) == 0:
                self.graph.remove_node(u)
            if v in self.graph and self.graph.degree(v) == 0:
                self.graph.remove_node(v)
            
            self.stats['deletions'] += 1
            self._dirty = True
        
        return (time.perf_counter() - start_time) * 1000
    
    def get_communities(self) -> Dict[int, int]:
        """Get community assignments via full Louvain."""
        if self._dirty or not self.communities:
            if self.graph.number_of_nodes() > 0:
                self.communities = community_louvain.best_partition(self.graph)
                self.stats['recomputations'] += 1
            else:
                self.communities = {}
            self._dirty = False
        return dict(self.communities)
    
    def get_modularity(self) -> float:
        """Calculate modularity."""
        if self.graph.number_of_edges() == 0:
            return 0.0
        communities = self.get_communities()
        return community_louvain.modularity(communities, self.graph)
    
    def get_statistics(self) -> Dict:
        """Get algorithm statistics."""
        return {
            **self.stats,
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges()
        }
    
    def reset(self):
        """Reset algorithm state."""
        self.graph.clear()
        self.communities.clear()
        self._dirty = True
        self.stats = {k: 0 for k in self.stats}


def demo():
    """Demonstrate baseline algorithms."""
    print("Baseline Algorithms Demo")
    print("=" * 50)
    
    # Create test edges
    edges = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)]
    
    algorithms = [
        ("Incremental LPA", IncrementalLPA()),
        ("Windowed Louvain", WindowedLouvain(window_size=100, recompute_interval=10)),
        ("Batch Louvain", BatchLouvain())
    ]
    
    for name, algo in algorithms:
        print(f"\n{name}:")
        
        # Add edges
        total_add_latency = 0
        for u, v in edges:
            total_add_latency += algo.add_edge(u, v)
        
        print(f"  Added {len(edges)} edges, total latency: {total_add_latency:.3f}ms")
        print(f"  Modularity: {algo.get_modularity():.4f}")
        print(f"  Communities: {len(set(algo.get_communities().values()))}")
        
        # Delete edge
        del_latency = algo.delete_edge(3, 4)
        print(f"  Delete (3,4) latency: {del_latency:.3f}ms")
        print(f"  Modularity after delete: {algo.get_modularity():.4f}")


if __name__ == "__main__":
    demo()
