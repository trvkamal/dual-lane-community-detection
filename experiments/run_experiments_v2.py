#!/usr/bin/env python3
"""
Dual-Lane - EXPERIMENT RUNNER V2
Uses fixed algorithms that properly compute communities

RUN: python run_experiments_v2.py --datasets synthetic-planted email-eu facebook --trials 3
"""

import os
import sys
import time
import random
import gzip
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dual_lane_v2 import DualLaneV2, IncrementalLPAV2, WindowedLouvainV2

try:
    import community as community_louvain
    import networkx as nx
except ImportError:
    print("ERROR: pip install networkx python-louvain scipy matplotlib numpy")
    sys.exit(1)


# =============================================================================
# DATASETS
# =============================================================================

def download_dataset(name):
    """Download SNAP dataset"""
    import urllib.request
    
    urls = {
        'email-eu': 'https://snap.stanford.edu/data/email-Eu-core.txt.gz',
        'facebook': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
        'wiki-vote': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
    }
    
    # if name not in urls:
    #     return None
    if name not in urls:
    #     return None
        os.makedirs('data', exist_ok=True)
        path = f'data/{name}.txt.gz'
    
    # if not os.path.exists(path):
    #     print(f"Downloading {name}...")
    #     try:
    #         urllib.request.urlretrieve(urls[name], path)
    #     except Exception as e:
    #         print(f"Download failed: {e}")
    #         return None
    
    return path


def generate_planted_partition(n=5000, k=10, p_in=0.15, p_out=0.005, seed=42):
    """Generate planted partition graph"""
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Generating planted partition: n={n}, k={k} communities")
    
    os.makedirs('data', exist_ok=True)
    path = 'data/synthetic-planted.txt'
    
    community_size = n // k
    edges = []
    communities = {}
    
    for i in range(n):
        communities[i] = i // community_size
    
    for i in range(n):
        for j in range(i + 1, n):
            if communities[i] == communities[j]:
                if random.random() < p_in:
                    edges.append((i, j))
            else:
                if random.random() < p_out:
                    edges.append((i, j))
    
    with open(path, 'w') as f:
        for u, v in edges:
            f.write(f"{u}\t{v}\n")
    
    print(f"Generated: {n} nodes, {len(edges)} edges, {k} communities")
    return path


def generate_lfr(n=5000, mu=0.3, seed=42):
    """Generate LFR benchmark"""
    print(f"Generating LFR: n={n}, mu={mu}")
    
    try:
        from networkx.generators.community import LFR_benchmark_graph
        
        G = LFR_benchmark_graph(
            n, tau1=2.5, tau2=1.5, mu=mu,
            average_degree=15, max_degree=50,
            min_community=20, max_community=100,
            seed=seed
        )
        
        os.makedirs('data', exist_ok=True)
        path = 'data/synthetic-lfr.txt'
        
        with open(path, 'w') as f:
            for u, v in G.edges():
                f.write(f"{u}\t{v}\n")
        
        print(f"Generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return path
        
    except Exception as e:
        print(f"LFR failed: {e}, using planted partition instead")
        return generate_planted_partition(n=n, seed=seed)


# =============================================================================
# STREAM GENERATION
# =============================================================================

def generate_stream(filepath, limit, seed=42, burst_interval=50000, burst_duration=10000,
                   base_churn=0.15, burst_churn=0.50):
    """Generate realistic stream with burst patterns"""
    random.seed(seed)
    np.random.seed(seed)
    
    recent_edges = deque(maxlen=20000)
    edge_set = set()
    count = 0
    
    open_func = gzip.open if filepath.endswith('.gz') else open
    mode = 'rt' if filepath.endswith('.gz') else 'r'
    
    try:
        with open_func(filepath, mode, encoding='utf-8', errors='ignore') as f:
            for line in f:
                if count >= limit:
                    break
                
                if line.startswith('#') or line.startswith('%'):
                    continue
                
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                try:
                    u, v = int(parts[0]), int(parts[1])
                    if u == v:
                        continue
                    
                    edge = tuple(sorted([u, v]))
                    in_burst = (count % burst_interval) < burst_duration
                    
                    # Add edge
                    if edge not in edge_set:
                        yield ('ADD', u, v, in_burst)
                        edge_set.add(edge)
                        recent_edges.append(edge)
                        count += 1
                    
                    # Churn
                    churn_rate = burst_churn if in_burst else base_churn
                    if len(recent_edges) > 500 and random.random() < churn_rate:
                        del_edge = random.choice(recent_edges)
                        if del_edge in edge_set:
                            yield ('DEL', del_edge[0], del_edge[1], in_burst)
                            edge_set.discard(del_edge)
                            count += 1
                
                except (ValueError, IndexError):
                    continue
    
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        # Synthetic fallback
        for i in range(limit):
            u = random.randint(0, 5000)
            v = random.randint(0, 5000)
            if u != v:
                in_burst = (i % burst_interval) < burst_duration
                yield ('ADD', u, v, in_burst)


# =============================================================================
# METRICS
# =============================================================================

def compute_modularity(graph, communities):
    """Compute modularity Q"""
    if not communities or graph.number_of_edges() == 0:
        return 0.0
    
    if isinstance(communities, (list, tuple)):
        comm_dict = {}
        for idx, nodes in enumerate(communities):
            for node in nodes:
                comm_dict[node] = idx
        communities = comm_dict
    
    try:
        m = graph.number_of_edges()
        Q = 0.0
        
        comm_nodes = defaultdict(list)
        for node, comm in communities.items():
            if node in graph:
                comm_nodes[comm].append(node)
        
        for comm, nodes in comm_nodes.items():
            nodes_set = set(nodes)
            Lc = 0
            Dc = 0
            
            for node in nodes:
                Dc += graph.degree(node)
                for neighbor in graph.neighbors(node):
                    if neighbor in nodes_set:
                        Lc += 1
            
            Lc //= 2
            Q += Lc / m - (Dc / (2 * m)) ** 2
        
        return Q
    except:
        return 0.0


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment(dataset_name, dataset_path, limit_ops, seed, 
                  time_threshold=5.0, size_threshold=10, quality_interval=5000):
    """Run single experiment"""
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {dataset_name} (seed={seed})")
    print(f"{'='*70}")
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize algorithms
    algorithms = {
        'Dual-Lane': DualLaneV2(
            time_threshold=time_threshold,
            size_threshold=size_threshold,
            louvain_interval=2000
        ),
        'Incremental LPA': IncrementalLPAV2(),
        'Windowed Louvain': WindowedLouvainV2(window_size=5000)
    }
    
    # Batch reference
    batch_graph = nx.Graph()
    
    # Results
    results = {name: {
        'normal_latencies': [],
        'burst_latencies': [],
        'quality_trace': []
    } for name in algorithms}
    
    # Process stream
    stream = generate_stream(dataset_path, limit_ops, seed)
    op_count = 0
    start_time = time.time()
    
    print("Processing stream...")
    
    for op, u, v, in_burst in stream:
        op_count += 1
        
        # Batch graph
        if op == 'ADD':
            batch_graph.add_edge(u, v)
        elif batch_graph.has_edge(u, v):
            batch_graph.remove_edge(u, v)
        
        # Process each algorithm
        for name, algo in algorithms.items():
            # Sample latency (10%)
            if random.random() < 0.1:
                t0 = time.perf_counter()
            else:
                t0 = None
            
            if op == 'ADD':
                algo.add_edge(u, v)
            else:
                algo.remove_edge(u, v)
            
            if t0 is not None:
                lat = (time.perf_counter() - t0) * 1000
                if in_burst:
                    results[name]['burst_latencies'].append(lat)
                else:
                    results[name]['normal_latencies'].append(lat)
            
            # Background processing
            if op_count % 500 == 0:
                algo.background_merge_process()
        
        # Quality measurement
        if op_count % quality_interval == 0:
            for name, algo in algorithms.items():
                try:
                    comms = algo.get_communities()
                    graph = algo.get_full_graph()
                    Q = compute_modularity(graph, comms) if comms else 0.0
                    results[name]['quality_trace'].append(Q)
                except:
                    results[name]['quality_trace'].append(0.0)
        
        # Progress
        if op_count % 20000 == 0:
            elapsed = time.time() - start_time
            rate = op_count / elapsed
            print(f"  {op_count:>7,}/{limit_ops:,} ({100*op_count/limit_ops:.1f}%) "
                  f"| {rate:.0f} ops/s")
    
    print(f"Completed in {(time.time()-start_time)/60:.1f} min")
    
    # Compute batch quality
    print("Computing batch Louvain reference...")
    try:
        batch_comms = community_louvain.best_partition(batch_graph)
        batch_Q = compute_modularity(batch_graph, batch_comms)
    except:
        batch_Q = 0.0
    
    print(f"Batch Louvain Q = {batch_Q:.3f}")
    
    # Final measurements
    final_results = {'batch_quality': batch_Q}
    
    for name, algo in algorithms.items():
        # Force final community computation
        comms = algo.get_communities()
        graph = algo.get_full_graph()
        final_Q = compute_modularity(graph, comms) if comms else 0.0
        
        normal = np.array(results[name]['normal_latencies'])
        burst = np.array(results[name]['burst_latencies'])
        
        normal_mean = np.mean(normal) if len(normal) > 0 else 0
        burst_mean = np.mean(burst) if len(burst) > 0 else 0
        
        final_results[name] = {
            'normal_mean': normal_mean,
            'normal_std': np.std(normal) if len(normal) > 0 else 0,
            'burst_mean': burst_mean,
            'burst_std': np.std(burst) if len(burst) > 0 else 0,
            'stability_ratio': burst_mean / normal_mean if normal_mean > 0 else 0,
            'final_quality': final_Q,
            'quality_vs_batch': final_Q / batch_Q if batch_Q > 0 else 0,
            'quality_trace': results[name]['quality_trace'],
            'stats': algo.get_statistics(),
            'normal_latencies': normal.tolist(),
            'burst_latencies': burst.tolist(),
        }
        
        print(f"{name}: Q={final_Q:.3f} ({100*final_Q/batch_Q:.1f}% of batch), "
              f"burst={burst_mean:.4f}ms")
    
    return final_results


def run_multi_trial(datasets, limit_per_dataset, num_trials, output_dir):
    """Run multiple trials across datasets"""
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'#'*70}")
        
        # Get dataset
        if dataset_name == 'synthetic-planted':
            path = generate_planted_partition()
        elif dataset_name == 'synthetic-lfr':
            path = generate_lfr()
        else:
            path = download_dataset(dataset_name)
        
        if path is None:
            print(f"Skipping {dataset_name}")
            continue
        
        # Run trials
        trial_results = []
        seeds = [42, 123, 456][:num_trials]
        
        for idx, seed in enumerate(seeds):
            print(f"\n--- Trial {idx+1}/{num_trials} ---")
            result = run_experiment(dataset_name, path, limit_per_dataset, seed)
            trial_results.append(result)
        
        # Aggregate
        all_results[dataset_name] = aggregate_trials(trial_results)
    
    # Generate outputs
    generate_outputs(all_results, output_dir)
    
    return all_results


def aggregate_trials(trials):
    """Aggregate multiple trials"""
    if not trials:
        return {}
    
    algos = [k for k in trials[0].keys() if k != 'batch_quality']
    
    agg = {
        'batch_quality_mean': np.mean([t['batch_quality'] for t in trials]),
    }
    
    for algo in algos:
        normal = [t[algo]['normal_mean'] for t in trials]
        burst = [t[algo]['burst_mean'] for t in trials]
        quality = [t[algo]['final_quality'] for t in trials]
        
        n = len(trials)
        
        agg[algo] = {
            'normal_mean': np.mean(normal),
            'normal_ci': 1.96 * np.std(normal) / np.sqrt(n) if n > 1 else 0,
            'burst_mean': np.mean(burst),
            'burst_ci': 1.96 * np.std(burst) / np.sqrt(n) if n > 1 else 0,
            'stability_ratio': np.mean(burst) / np.mean(normal) if np.mean(normal) > 0 else 0,
            'quality_mean': np.mean(quality),
            'quality_ci': 1.96 * np.std(quality) / np.sqrt(n) if n > 1 else 0,
            'quality_vs_batch': np.mean(quality) / agg['batch_quality_mean'] if agg['batch_quality_mean'] > 0 else 0,
            'burst_samples': burst,
            'quality_samples': quality,
        }
        
        # Dual-Lane specific
        if algo == 'Dual-Lane':
            vol_rates = [t[algo]['stats'].get('volatility_rate', 0) for t in trials]
            agg[algo]['volatility_rate'] = np.mean(vol_rates)
    
    return agg


def generate_outputs(results, output_dir):
    """Generate all outputs"""
    
    print(f"\n{'='*70}")
    print("GENERATING OUTPUTS")
    print(f"{'='*70}")
    
    # Text summary
    with open(f"{output_dir}/results_summary.txt", 'w') as f:
        f.write("="*70 + "\n")
        f.write("Dual-Lane - RESULTS SUMMARY (V2 FIXED)\n")
        f.write("="*70 + "\n\n")
        
        for dataset, data in results.items():
            f.write(f"\nDATASET: {dataset}\n")
            f.write("-"*50 + "\n")
            f.write(f"Batch Louvain Q = {data['batch_quality_mean']:.3f}\n\n")
            
            for algo in ['Incremental LPA', 'Windowed Louvain', 'Dual-Lane']:
                if algo not in data:
                    continue
                d = data[algo]
                f.write(f"{algo}:\n")
                f.write(f"  Normal: {d['normal_mean']:.4f} ± {d['normal_ci']:.4f} ms\n")
                f.write(f"  Burst:  {d['burst_mean']:.4f} ± {d['burst_ci']:.4f} ms\n")
                f.write(f"  Stability: {d['stability_ratio']:.2f}x\n")
                f.write(f"  Quality: {d['quality_mean']:.3f} ({100*d['quality_vs_batch']:.1f}% of batch)\n")
                if algo == 'Dual-Lane':
                    f.write(f"  Volatility: {100*d.get('volatility_rate',0):.1f}%\n")
                f.write("\n")
    
    print(f"Saved: {output_dir}/results_summary.txt")
    
    # Main figure
    if results:
        generate_figure(results, output_dir)


def generate_figure(results, output_dir):
    """Generate main results figure"""
    
    # Use first dataset for figure
    dataset = list(results.keys())[0]
    data = results[dataset]
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plt.rcParams.update({'font.size': 11})
    
    algos = ['Incremental LPA', 'Windowed Louvain', 'Dual-Lane']
    colors = {'Incremental LPA': '#D55E00', 'Windowed Louvain': '#009E73', 'Dual-Lane': '#CC79A7'}
    
    # (a) Latency
    ax = axes[0, 0]
    x = np.arange(len(algos))
    width = 0.35
    
    normal = [data[a]['normal_mean'] for a in algos]
    normal_err = [data[a]['normal_ci'] for a in algos]
    burst = [data[a]['burst_mean'] for a in algos]
    burst_err = [data[a]['burst_ci'] for a in algos]
    
    ax.bar(x - width/2, normal, width, yerr=normal_err, label='Normal', color='#4A90E2', capsize=4)
    ax.bar(x + width/2, burst, width, yerr=burst_err, label='Burst', color='#E85D4A', capsize=4)
    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('(a) Latency Comparison', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # (b) Stability
    ax = axes[0, 1]
    stability = [data[a]['stability_ratio'] for a in algos]
    bars = ax.barh(range(len(algos)), stability, color=[colors[a] for a in algos])
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Ideal')
    ax.set_xlabel('Burst/Normal Ratio', fontweight='bold')
    ax.set_title('(b) Stability (lower = better)', fontweight='bold')
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # (c) Quality
    ax = axes[1, 0]
    quality = [data[a]['quality_mean'] for a in algos]
    quality_err = [data[a]['quality_ci'] for a in algos]
    
    ax.bar(x, quality, yerr=quality_err, color=[colors[a] for a in algos], capsize=4)
    ax.axhline(y=data['batch_quality_mean'], color='black', linestyle='--', label='Batch')
    ax.set_ylabel('Modularity Q', fontweight='bold')
    ax.set_title('(c) Community Quality', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # (d) Tradeoff
    ax = axes[1, 1]
    for i, algo in enumerate(algos):
        ax.scatter(data[algo]['burst_mean'], data[algo]['quality_mean'],
                  s=150, c=colors[algo], label=algo, marker='o', edgecolors='black')
    
    ax.set_xlabel('Burst Latency (ms)', fontweight='bold')
    ax.set_ylabel('Modularity Q', fontweight='bold')
    ax.set_title('(d) Quality-Latency Tradeoff', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/main_results.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/main_results.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_dir}/main_results.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual-Lane V2')
    parser.add_argument('--datasets', nargs='+', default=['synthetic-planted'])
    parser.add_argument('--limit', type=int, default=100000)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--output', type=str, default='results_v2')
    
    args = parser.parse_args()
    
    print("\n" + "🎯"*35)
    print("Dual-Lane - V2 FIXED EXPERIMENTS")
    print("🎯"*35 + "\n")
    
    results = run_multi_trial(args.datasets, args.limit, args.trials, args.output)
    
    print("\n" + "="*70)
    print("✅ COMPLETE")
    print("="*70)
    print(f"Results: {args.output}/")


if __name__ == "__main__":
    main()
