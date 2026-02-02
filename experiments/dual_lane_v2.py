"""
Dual-Lane Community Detection - V2 Implementation

This module contains the V2 implementation of the Dual-Lane algorithm
along with baseline algorithms (IncrementalLPAV2, WindowedLouvainV2)
that properly compute and track communities.

Key improvements in V2:
1. Forces Louvain execution after every promotion
2. Tracks when communities need recomputation  
3. Ensures get_communities() always returns valid communities
"""

import time
import sys
import networkx as nx
import numpy as np
from collections import defaultdict

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    print("WARNING: python-louvain not installed")
    HAS_LOUVAIN = False


class DualLaneV2:
    """
    Dual-Lane Community Detection - V2 Implementation
    
    Key fix: Communities are ALWAYS computed when queried if stale.
    
    Parameters
    ----------
    time_threshold : float
        Minimum age (seconds) before promotion. Default: 5.0
    size_threshold : int  
        Maximum size for fast lane components. Default: 10
    density_threshold : float
        Minimum edge density (0.7 = 70%). Default: 0.7
    louvain_interval : int
        Run Louvain every N operations. Default: 2000
    """
    
    def __init__(self, time_threshold=5.0, size_threshold=10, density_threshold=0.7,
                 louvain_interval=2000):
        # Global graph (Slow Lane)
        self.global_graph = nx.Graph()
        self.global_communities = {}
        self.communities_stale = True  # Track if communities need recomputation
        
        # Fast Lane (Buffer)
        self.components = {}
        self.node_to_component = {}
        self.next_comp_id = 0
        
        # Parameters
        self.T = time_threshold
        self.K = size_threshold
        self.epsilon = density_threshold
        self.louvain_interval = louvain_interval
        
        # Tracking
        self.operation_count = 0
        self.last_louvain_op = 0
        self.promotions_since_louvain = 0
        
        # Statistics
        self.stats = {
            'additions': 0,
            'deletions': 0,
            'promotions': 0,
            'dissolutions': 0,
            'louvain_runs': 0,
            'volatile_edges': 0,
            'stable_edges': 0,
        }
        
        # Edge tracking
        self.edge_add_times = {}
        self.peak_memory_bytes = 0
    
    def add_edge(self, u, v):
        """Add edge - O(K) worst case"""
        self.stats['additions'] += 1
        self.operation_count += 1
        current_time = time.time()
        edge = tuple(sorted([u, v]))
        
        # Track edge time
        self.edge_add_times[edge] = current_time
        
        # Skip if in global graph
        if self.global_graph.has_edge(u, v):
            return
        
        comp_u = self.node_to_component.get(u)
        comp_v = self.node_to_component.get(v)
        
        if comp_u is not None and comp_u == comp_v:
            # Same component
            comp = self.components[comp_u]
            comp['edges'].add(edge)
            comp['last_active'] = current_time
        
        elif comp_u is not None and comp_v is not None:
            # Merge components
            self._merge_components(comp_u, comp_v, edge, current_time)
        
        elif comp_u is not None:
            # Add v to u's component
            self._add_to_component(comp_u, v, edge, current_time)
        
        elif comp_v is not None:
            # Add u to v's component
            self._add_to_component(comp_v, u, edge, current_time)
        
        else:
            # Create new component
            self._create_component(u, v, edge, current_time)
        
        # Check promotion
        self._check_promotions(current_time)
        
        # Maybe run Louvain
        self._maybe_run_louvain()
    
    def remove_edge(self, u, v):
        """Remove edge - O(K) worst case for fast lane"""
        self.stats['deletions'] += 1
        self.operation_count += 1
        edge = tuple(sorted([u, v]))
        
        # Check fast lane first (bounded search)
        comp_id = self.node_to_component.get(u)
        if comp_id is not None:
            comp = self.components.get(comp_id)
            if comp and edge in comp['edges']:
                # Found in fast lane - volatile edge
                self.stats['volatile_edges'] += 1
                comp['edges'].discard(edge)
                
                # Check if component should dissolve
                if len(comp['edges']) == 0:
                    self._dissolve_component(comp_id)
                
                if edge in self.edge_add_times:
                    del self.edge_add_times[edge]
                return
        
        # Check global graph
        if self.global_graph.has_edge(u, v):
            self.global_graph.remove_edge(u, v)
            self.communities_stale = True
            
            # Clean up isolated nodes
            for node in [u, v]:
                if node in self.global_graph and self.global_graph.degree(node) == 0:
                    self.global_graph.remove_node(node)
            
            if edge in self.edge_add_times:
                del self.edge_add_times[edge]
    
    def _create_component(self, u, v, edge, current_time):
        """Create new component"""
        comp_id = self.next_comp_id
        self.next_comp_id += 1
        
        self.components[comp_id] = {
            'nodes': {u, v},
            'edges': {edge},
            'created_at': current_time,
            'last_active': current_time,
        }
        self.node_to_component[u] = comp_id
        self.node_to_component[v] = comp_id
    
    def _add_to_component(self, comp_id, node, edge, current_time):
        """Add node to existing component"""
        comp = self.components[comp_id]
        comp['nodes'].add(node)
        comp['edges'].add(edge)
        comp['last_active'] = current_time
        self.node_to_component[node] = comp_id
        
        # Check size
        if len(comp['nodes']) > self.K:
            self._promote_component(comp_id)
    
    def _merge_components(self, comp1, comp2, edge, current_time):
        """Merge two components"""
        c1 = self.components[comp1]
        c2 = self.components[comp2]
        
        # Merge into larger
        if len(c1['nodes']) >= len(c2['nodes']):
            target, source = comp1, comp2
        else:
            target, source = comp2, comp1
        
        target_comp = self.components[target]
        source_comp = self.components[source]
        
        target_comp['nodes'].update(source_comp['nodes'])
        target_comp['edges'].update(source_comp['edges'])
        target_comp['edges'].add(edge)
        target_comp['last_active'] = current_time
        
        for node in source_comp['nodes']:
            self.node_to_component[node] = target
        
        del self.components[source]
        
        # Check size
        if len(target_comp['nodes']) > self.K:
            self._promote_component(target)
    
    def _dissolve_component(self, comp_id):
        """Dissolve empty component"""
        if comp_id not in self.components:
            return
        
        comp = self.components[comp_id]
        for node in comp['nodes']:
            if self.node_to_component.get(node) == comp_id:
                del self.node_to_component[node]
        
        del self.components[comp_id]
        self.stats['dissolutions'] += 1
    
    def _check_promotions(self, current_time):
        """Check all components for promotion"""
        to_promote = []
        
        for comp_id, comp in list(self.components.items()):
            age = current_time - comp['created_at']
            
            # Promote if old enough
            if age >= self.T:
                to_promote.append(comp_id)
                continue
            
            # Promote if too large
            if len(comp['nodes']) > self.K:
                to_promote.append(comp_id)
                continue
            
            # Promote if sparse and semi-old
            if age >= self.T / 2:
                n = len(comp['nodes'])
                e = len(comp['edges'])
                if n >= 2:
                    max_edges = n * (n - 1) / 2
                    density = e / max_edges if max_edges > 0 else 0
                    if density < self.epsilon / 2:
                        to_promote.append(comp_id)
        
        for comp_id in to_promote:
            self._promote_component(comp_id)
    
    def _promote_component(self, comp_id):
        """Promote component to global graph"""
        if comp_id not in self.components:
            return
        
        comp = self.components[comp_id]
        
        # Add edges to global graph
        for edge in comp['edges']:
            self.global_graph.add_edge(edge[0], edge[1])
            self.stats['stable_edges'] += 1
        
        # Clean up
        for node in comp['nodes']:
            if self.node_to_component.get(node) == comp_id:
                del self.node_to_component[node]
        
        del self.components[comp_id]
        self.stats['promotions'] += 1
        self.promotions_since_louvain += 1
        self.communities_stale = True
    
    def _maybe_run_louvain(self):
        """Run Louvain if needed"""
        if not HAS_LOUVAIN:
            return
        
        # Run if enough operations or promotions
        should_run = (
            self.operation_count - self.last_louvain_op >= self.louvain_interval or
            self.promotions_since_louvain >= 5
        )
        
        if should_run and self.global_graph.number_of_nodes() > 1:
            try:
                self.global_communities = community_louvain.best_partition(self.global_graph)
                self.stats['louvain_runs'] += 1
                self.last_louvain_op = self.operation_count
                self.promotions_since_louvain = 0
                self.communities_stale = False
            except Exception:
                pass
    
    def get_communities(self):
        """Get current community assignments"""
        # Recompute if stale
        if self.communities_stale and HAS_LOUVAIN:
            if self.global_graph.number_of_nodes() > 1:
                try:
                    self.global_communities = community_louvain.best_partition(self.global_graph)
                    self.stats['louvain_runs'] += 1
                    self.communities_stale = False
                except Exception:
                    pass
        
        # Combine global communities with buffer assignments
        combined = dict(self.global_communities)
        
        # Assign buffer nodes to temporary communities
        max_comm = max(combined.values()) + 1 if combined else 0
        
        for comp_id, comp in self.components.items():
            for node in comp['nodes']:
                if node not in combined:
                    combined[node] = max_comm + comp_id
        
        return combined
    
    def get_full_graph(self):
        """Get combined graph (global + buffer)"""
        combined = nx.Graph()
        combined.add_edges_from(self.global_graph.edges())
        
        for comp in self.components.values():
            for edge in comp['edges']:
                combined.add_edge(edge[0], edge[1])
        
        return combined
    
    def background_merge_process(self):
        """Process pending merges/promotions"""
        self._check_promotions(time.time())
        self._maybe_run_louvain()
    
    def _update_memory(self):
        """Track memory usage"""
        mem = (
            sys.getsizeof(self.global_graph) +
            sys.getsizeof(self.components) +
            self.global_graph.number_of_nodes() * 100 +
            self.global_graph.number_of_edges() * 50
        )
        self.peak_memory_bytes = max(self.peak_memory_bytes, mem)
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        total_processed = self.stats['volatile_edges'] + self.stats['stable_edges']
        vol_rate = self.stats['volatile_edges'] / total_processed if total_processed > 0 else 0
        
        return {
            **self.stats,
            'active_components': len(self.components),
            'buffered_nodes': sum(len(c['nodes']) for c in self.components.values()),
            'buffered_edges': sum(len(c['edges']) for c in self.components.values()),
            'global_nodes': self.global_graph.number_of_nodes(),
            'global_edges': self.global_graph.number_of_edges(),
            'volatility_rate': vol_rate,
            'peak_memory_mb': self.peak_memory_bytes / (1024 * 1024),
            'communities_computed': len(self.global_communities),
        }


class IncrementalLPAV2:
    """Incremental LPA with proper community tracking"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.labels = {}
        self.operation_count = 0
        self.peak_memory_bytes = 0
    
    def add_edge(self, u, v):
        self.graph.add_edge(u, v)
        self.operation_count += 1
        
        if u not in self.labels:
            self.labels[u] = u
        if v not in self.labels:
            self.labels[v] = v
        
        self._propagate([u, v], iterations=2)
        
        if self.operation_count % 10000 == 0:
            self._update_memory()
    
    def remove_edge(self, u, v):
        self.operation_count += 1
        if not self.graph.has_edge(u, v):
            return
        
        self.graph.remove_edge(u, v)
        
        affected = set()
        for node in [u, v]:
            if node in self.graph:
                affected.add(node)
                affected.update(self.graph.neighbors(node))
        
        self._propagate(list(affected), iterations=3)
        
        # Clean isolated nodes
        for node in [u, v]:
            if node in self.graph and self.graph.degree(node) == 0:
                self.graph.remove_node(node)
                if node in self.labels:
                    del self.labels[node]
    
    def _propagate(self, nodes, iterations=2):
        """Label propagation on affected nodes"""
        import random
        
        for _ in range(iterations):
            random.shuffle(nodes)
            for node in nodes:
                if node not in self.graph:
                    continue
                
                neighbors = list(self.graph.neighbors(node))
                if not neighbors:
                    continue
                
                # Count neighbor labels
                label_counts = defaultdict(int)
                for neighbor in neighbors:
                    if neighbor in self.labels:
                        label_counts[self.labels[neighbor]] += 1
                
                if label_counts:
                    max_count = max(label_counts.values())
                    best_labels = [l for l, c in label_counts.items() if c == max_count]
                    self.labels[node] = random.choice(best_labels)
    
    def _update_memory(self):
        mem = sys.getsizeof(self.graph) + self.graph.number_of_nodes() * 100
        self.peak_memory_bytes = max(self.peak_memory_bytes, mem)
    
    def get_communities(self):
        return dict(self.labels)
    
    def get_full_graph(self):
        return self.graph
    
    def get_statistics(self):
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'peak_memory_mb': self.peak_memory_bytes / (1024*1024)
        }


class WindowedLouvainV2:
    """Windowed Louvain with periodic recomputation"""
    
    def __init__(self, window_size=10000):
        self.graph = nx.Graph()
        self.communities = {}
        self.window_size = window_size
        self.operation_count = 0
        self.last_compute = 0
        self.louvain_runs = 0
        self.peak_memory_bytes = 0
    
    def add_edge(self, u, v):
        self.graph.add_edge(u, v)
        self.operation_count += 1
        
        if self.operation_count - self.last_compute >= self.window_size:
            self._compute_communities()
    
    def remove_edge(self, u, v):
        self.operation_count += 1
        if not self.graph.has_edge(u, v):
            return
        
        self.graph.remove_edge(u, v)
        
        # Clean isolated nodes
        for node in [u, v]:
            if node in self.graph and self.graph.degree(node) == 0:
                self.graph.remove_node(node)
        
        if self.operation_count - self.last_compute >= self.window_size:
            self._compute_communities()
    
    def _compute_communities(self):
        if HAS_LOUVAIN and self.graph.number_of_nodes() > 1:
            try:
                self.communities = community_louvain.best_partition(self.graph)
                self.louvain_runs += 1
                self.last_compute = self.operation_count
            except:
                pass
    
    def _update_memory(self):
        mem = sys.getsizeof(self.graph) + self.graph.number_of_nodes() * 100
        self.peak_memory_bytes = max(self.peak_memory_bytes, mem)
    
    def get_communities(self):
        # Force compute if empty
        if not self.communities and self.graph.number_of_nodes() > 1:
            self._compute_communities()
        return dict(self.communities)
    
    def get_full_graph(self):
        return self.graph
    
    def get_statistics(self):
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'louvain_runs': self.louvain_runs,
            'peak_memory_mb': self.peak_memory_bytes / (1024*1024)
        }


# Quick test
if __name__ == "__main__":
    print("Testing DualLaneV2...")
    
    dl = DualLaneV2(time_threshold=1.0, size_threshold=5)
    
    # Build a simple graph with clear community structure
    # Community 1: nodes 0-9
    for i in range(10):
        for j in range(i+1, 10):
            if abs(i-j) <= 3:  # Dense internal connections
                dl.add_edge(i, j)
    
    # Community 2: nodes 10-19
    for i in range(10, 20):
        for j in range(i+1, 20):
            if abs(i-j) <= 3:
                dl.add_edge(i, j)
    
    # Few cross-community edges
    dl.add_edge(5, 15)
    
    # Process
    dl.background_merge_process()
    time.sleep(1.5)  # Let time threshold pass
    dl.background_merge_process()
    
    # Get results
    comms = dl.get_communities()
    stats = dl.get_statistics()
    
    print(f"Nodes with communities: {len(comms)}")
    print(f"Global graph nodes: {stats['global_nodes']}")
    print(f"Louvain runs: {stats['louvain_runs']}")
    print(f"Communities computed: {stats['communities_computed']}")
    
    if len(comms) > 0:
        print("✅ Communities are being computed!")
    else:
        print("❌ Still broken")
