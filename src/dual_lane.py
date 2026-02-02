"""
Dual-Lane: Bounded-Complexity Community Detection for Streaming Graphs Under Burst Churn

This module implements the Dual-Lane algorithm, a two-tier buffering architecture that achieves
bounded O(K) deletion complexity while maintaining near-optimal community detection quality.

Paper: "Dual-Lane: Bounded-Complexity Community Detection for Streaming Graphs Under Burst Churn"


Author: [Your Name]
"""

import time
import networkx as nx
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, List
import community as community_louvain  # python-louvain package


class DualLane:
    """
    Dual-Lane streaming community detection algorithm.
    
    The algorithm uses a two-tier architecture:
    - Fast Lane: Bounded buffer (K-node components) for absorbing volatile edges
    - Slow Lane: Global graph with Louvain community detection
    
    Parameters
    ----------
    T : float
        Age threshold in seconds. Edges older than T are promoted to slow lane.
        Default: 5.0
    K : int
        Maximum component size in fast lane before forced promotion.
        Default: 10
    tau_d : float
        Density threshold. Components below tau_d/2 are candidates for promotion.
        Default: 0.7
    
    Attributes
    ----------
    fast_lane : nx.Graph
        Buffer graph for recent/volatile edges
    slow_lane : nx.Graph
        Global graph for stable community structure
    edge_timestamps : dict
        Mapping of edges to their addition timestamps
    communities : dict
        Current community assignments (node -> community_id)
    
    Example
    -------
    >>> dl = DualLane(T=5.0, K=10, tau_d=0.7)
    >>> dl.add_edge(1, 2)
    >>> dl.add_edge(2, 3)
    >>> communities = dl.get_communities()
    >>> dl.delete_edge(1, 2)
    """
    
    def __init__(self, T: float = 5.0, K: int = 10, tau_d: float = 0.7):
        self.T = T
        self.K = K
        self.tau_d = tau_d
        
        # Two-tier graph structure
        self.fast_lane = nx.Graph()
        self.slow_lane = nx.Graph()
        
        # Edge metadata
        self.edge_timestamps: Dict[Tuple[int, int], float] = {}
        
        # Component tracking in fast lane
        self.node_to_component: Dict[int, int] = {}
        self.component_nodes: Dict[int, Set[int]] = defaultdict(set)
        self.next_component_id = 0
        
        # Community assignments
        self.communities: Dict[int, int] = {}
        self._communities_dirty = True
        
        # Metrics tracking
        self.stats = {
            'additions': 0,
            'deletions': 0,
            'promotions': 0,
            'fast_lane_deletions': 0,
            'slow_lane_deletions': 0,
            'volatility_filtered': 0
        }
    
    def _normalize_edge(self, u: int, v: int) -> Tuple[int, int]:
        """Normalize edge representation for consistent lookup."""
        return (min(u, v), max(u, v))
    
    def _get_component_size(self, component_id: int) -> int:
        """Get the number of nodes in a component."""
        return len(self.component_nodes.get(component_id, set()))
    
    def _get_component_density(self, component_id: int) -> float:
        """Calculate density of a component in fast lane."""
        nodes = self.component_nodes.get(component_id, set())
        n = len(nodes)
        if n < 2:
            return 0.0
        
        # Count edges within component
        edge_count = sum(
            1 for u in nodes for v in self.fast_lane.neighbors(u)
            if v in nodes and u < v
        )
        max_edges = n * (n - 1) / 2
        return edge_count / max_edges if max_edges > 0 else 0.0
    
    def _merge_components(self, comp1: int, comp2: int) -> int:
        """Merge two components, returning the surviving component id."""
        if comp1 == comp2:
            return comp1
        
        # Merge smaller into larger
        if len(self.component_nodes[comp1]) < len(self.component_nodes[comp2]):
            comp1, comp2 = comp2, comp1
        
        # Move nodes from comp2 to comp1
        for node in self.component_nodes[comp2]:
            self.node_to_component[node] = comp1
            self.component_nodes[comp1].add(node)
        
        del self.component_nodes[comp2]
        return comp1
    
    def _create_new_component(self, nodes: Set[int]) -> int:
        """Create a new component with the given nodes."""
        comp_id = self.next_component_id
        self.next_component_id += 1
        
        for node in nodes:
            self.node_to_component[node] = comp_id
            self.component_nodes[comp_id].add(node)
        
        return comp_id
    
    def _promote_component(self, component_id: int):
        """Promote a component from fast lane to slow lane."""
        nodes = self.component_nodes.get(component_id, set())
        if not nodes:
            return
        
        # Get all edges in this component
        edges_to_promote = []
        for u in nodes:
            for v in self.fast_lane.neighbors(u):
                if v in nodes and u < v:
                    edge = self._normalize_edge(u, v)
                    edges_to_promote.append((u, v))
        
        # Add to slow lane
        for u, v in edges_to_promote:
            self.slow_lane.add_edge(u, v)
            edge = self._normalize_edge(u, v)
            if edge in self.edge_timestamps:
                del self.edge_timestamps[edge]
        
        # Remove from fast lane
        for u, v in edges_to_promote:
            if self.fast_lane.has_edge(u, v):
                self.fast_lane.remove_edge(u, v)
        
        # Clean up isolated nodes
        for node in list(nodes):
            if node in self.fast_lane and self.fast_lane.degree(node) == 0:
                self.fast_lane.remove_node(node)
        
        # Remove component tracking
        for node in nodes:
            if node in self.node_to_component:
                del self.node_to_component[node]
        if component_id in self.component_nodes:
            del self.component_nodes[component_id]
        
        self.stats['promotions'] += 1
        self._communities_dirty = True
    
    def _check_promotion_criteria(self, component_id: int, current_time: float) -> bool:
        """Check if a component should be promoted to slow lane."""
        nodes = self.component_nodes.get(component_id, set())
        if not nodes:
            return False
        
        # Size criterion: component exceeds K
        if len(nodes) > self.K:
            return True
        
        # Age criterion: all edges older than T
        all_old = True
        for u in nodes:
            for v in self.fast_lane.neighbors(u):
                if v in nodes and u < v:
                    edge = self._normalize_edge(u, v)
                    if edge in self.edge_timestamps:
                        age = current_time - self.edge_timestamps[edge]
                        if age < self.T:
                            all_old = False
                            break
            if not all_old:
                break
        
        if all_old and len(nodes) >= 2:
            return True
        
        # Density criterion: sparse component that's been around
        density = self._get_component_density(component_id)
        if density < self.tau_d / 2 and len(nodes) >= self.K // 2:
            return True
        
        return False
    
    def add_edge(self, u: int, v: int, timestamp: Optional[float] = None) -> float:
        """
        Add an edge to the streaming graph.
        
        Algorithm 1: Edge Addition with Bounded Complexity
        
        Parameters
        ----------
        u, v : int
            Node identifiers for the edge endpoints
        timestamp : float, optional
            Edge timestamp. If None, uses current time.
        
        Returns
        -------
        float
            Processing latency in milliseconds
        """
        start_time = time.perf_counter()
        
        if timestamp is None:
            timestamp = time.time()
        
        edge = self._normalize_edge(u, v)
        
        # Check if edge already exists
        if self.fast_lane.has_edge(u, v) or self.slow_lane.has_edge(u, v):
            return (time.perf_counter() - start_time) * 1000
        
        # Add to fast lane
        self.fast_lane.add_edge(u, v)
        self.edge_timestamps[edge] = timestamp
        self.stats['additions'] += 1
        
        # Update component structure
        comp_u = self.node_to_component.get(u)
        comp_v = self.node_to_component.get(v)
        
        if comp_u is None and comp_v is None:
            # Both nodes are new - create new component
            self._create_new_component({u, v})
        elif comp_u is None:
            # u is new, add to v's component
            self.node_to_component[u] = comp_v
            self.component_nodes[comp_v].add(u)
        elif comp_v is None:
            # v is new, add to u's component
            self.node_to_component[v] = comp_u
            self.component_nodes[comp_u].add(v)
        else:
            # Both in components - merge if different
            comp_u = self._merge_components(comp_u, comp_v)
        
        # Check promotion criteria for affected component
        affected_comp = self.node_to_component.get(u)
        if affected_comp is not None:
            if self._check_promotion_criteria(affected_comp, timestamp):
                self._promote_component(affected_comp)
        
        self._communities_dirty = True
        return (time.perf_counter() - start_time) * 1000
    
    def delete_edge(self, u: int, v: int) -> float:
        """
        Delete an edge from the streaming graph.
        
        Algorithm 2: Edge Deletion with Bounded O(K) Complexity
        
        The key insight: we first search the fast lane (bounded by K),
        only falling back to the slow lane if the edge is not found.
        
        Parameters
        ----------
        u, v : int
            Node identifiers for the edge endpoints
        
        Returns
        -------
        float
            Processing latency in milliseconds
        """
        start_time = time.perf_counter()
        
        edge = self._normalize_edge(u, v)
        
        # First, check fast lane (O(K) bounded search)
        if self.fast_lane.has_edge(u, v):
            self.fast_lane.remove_edge(u, v)
            
            if edge in self.edge_timestamps:
                del self.edge_timestamps[edge]
            
            # Update component structure - may need to split
            self._handle_fast_lane_deletion(u, v)
            
            self.stats['fast_lane_deletions'] += 1
            self.stats['volatility_filtered'] += 1
            self.stats['deletions'] += 1
            self._communities_dirty = True
            
            return (time.perf_counter() - start_time) * 1000
        
        # Fall back to slow lane (O(degree) but for stable edges)
        if self.slow_lane.has_edge(u, v):
            self.slow_lane.remove_edge(u, v)
            
            # Clean up isolated nodes
            if self.slow_lane.degree(u) == 0:
                self.slow_lane.remove_node(u)
            if v in self.slow_lane and self.slow_lane.degree(v) == 0:
                self.slow_lane.remove_node(v)
            
            self.stats['slow_lane_deletions'] += 1
            self.stats['deletions'] += 1
            self._communities_dirty = True
        
        return (time.perf_counter() - start_time) * 1000
    
    def _handle_fast_lane_deletion(self, u: int, v: int):
        """Handle component updates after fast lane edge deletion."""
        comp_u = self.node_to_component.get(u)
        comp_v = self.node_to_component.get(v)
        
        if comp_u is None or comp_u != comp_v:
            return
        
        # Check if component is still connected
        component_nodes = self.component_nodes.get(comp_u, set())
        if not component_nodes:
            return
        
        # Use BFS to find connected components within this component
        visited = set()
        components = []
        
        for start_node in component_nodes:
            if start_node in visited:
                continue
            
            # BFS from start_node
            current_component = set()
            queue = [start_node]
            
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                current_component.add(node)
                
                for neighbor in self.fast_lane.neighbors(node):
                    if neighbor in component_nodes and neighbor not in visited:
                        queue.append(neighbor)
            
            components.append(current_component)
        
        # If split occurred, create new components
        if len(components) > 1:
            # Keep first component with original id
            first = components[0]
            self.component_nodes[comp_u] = first
            for node in list(component_nodes):
                if node not in first:
                    del self.node_to_component[node]
            
            # Create new components for splits
            for new_comp_nodes in components[1:]:
                if new_comp_nodes:
                    self._create_new_component(new_comp_nodes)
        
        # Clean up empty/singleton components
        for node in [u, v]:
            if node in self.fast_lane and self.fast_lane.degree(node) == 0:
                comp = self.node_to_component.get(node)
                if comp is not None:
                    self.component_nodes[comp].discard(node)
                    del self.node_to_component[node]
                    if not self.component_nodes[comp]:
                        del self.component_nodes[comp]
                self.fast_lane.remove_node(node)
    
    def get_communities(self) -> Dict[int, int]:
        """
        Get current community assignments using Louvain algorithm.
        
        Returns
        -------
        dict
            Mapping of node -> community_id
        """
        if not self._communities_dirty and self.communities:
            return self.communities
        
        # Combine fast and slow lane for community detection
        combined = nx.Graph()
        combined.add_edges_from(self.fast_lane.edges())
        combined.add_edges_from(self.slow_lane.edges())
        
        if combined.number_of_nodes() == 0:
            self.communities = {}
        else:
            try:
                self.communities = community_louvain.best_partition(combined)
            except Exception:
                # Fallback: each node is its own community
                self.communities = {n: i for i, n in enumerate(combined.nodes())}
        
        self._communities_dirty = False
        return self.communities
    
    def get_modularity(self) -> float:
        """
        Calculate modularity of current community structure.
        
        Returns
        -------
        float
            Modularity score in range [-0.5, 1.0]
        """
        combined = nx.Graph()
        combined.add_edges_from(self.fast_lane.edges())
        combined.add_edges_from(self.slow_lane.edges())
        
        if combined.number_of_edges() == 0:
            return 0.0
        
        communities = self.get_communities()
        return community_louvain.modularity(communities, combined)
    
    def get_statistics(self) -> Dict:
        """
        Get algorithm statistics.
        
        Returns
        -------
        dict
            Statistics including operation counts and graph sizes
        """
        return {
            **self.stats,
            'fast_lane_nodes': self.fast_lane.number_of_nodes(),
            'fast_lane_edges': self.fast_lane.number_of_edges(),
            'slow_lane_nodes': self.slow_lane.number_of_nodes(),
            'slow_lane_edges': self.slow_lane.number_of_edges(),
            'num_components': len(self.component_nodes),
            'volatility_rate': (
                self.stats['volatility_filtered'] / self.stats['deletions']
                if self.stats['deletions'] > 0 else 0.0
            )
        }
    
    def reset(self):
        """Reset algorithm state."""
        self.fast_lane.clear()
        self.slow_lane.clear()
        self.edge_timestamps.clear()
        self.node_to_component.clear()
        self.component_nodes.clear()
        self.next_component_id = 0
        self.communities.clear()
        self._communities_dirty = True
        self.stats = {k: 0 for k in self.stats}


def demo():
    """Demonstration of Dual-Lane algorithm."""
    print("Dual-Lane Community Detection Demo")
    print("=" * 50)
    
    # Initialize algorithm
    dl = DualLane(T=5.0, K=10, tau_d=0.7)
    
    # Simulate stream of edges
    edges = [
        (1, 2), (2, 3), (3, 1),  # Triangle community
        (4, 5), (5, 6), (6, 4),  # Another triangle
        (3, 4),  # Bridge between communities
    ]
    
    print("\nAdding edges:")
    for u, v in edges:
        latency = dl.add_edge(u, v)
        print(f"  Added ({u}, {v}) - latency: {latency:.3f}ms")
    
    print("\nCommunities detected:")
    communities = dl.get_communities()
    for node, comm in sorted(communities.items()):
        print(f"  Node {node} -> Community {comm}")
    
    print(f"\nModularity: {dl.get_modularity():.4f}")
    
    print("\nDeleting volatile edge (3, 4):")
    latency = dl.delete_edge(3, 4)
    print(f"  Latency: {latency:.3f}ms")
    
    print("\nUpdated communities:")
    communities = dl.get_communities()
    for node, comm in sorted(communities.items()):
        print(f"  Node {node} -> Community {comm}")
    
    print(f"\nStatistics: {dl.get_statistics()}")


if __name__ == "__main__":
    demo()
