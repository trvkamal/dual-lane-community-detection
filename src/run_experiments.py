#!/usr/bin/env python3
"""
Experiment Runner for Dual-Lane Evaluation

This script runs the complete experimental evaluation as described in the
research paper. It compares Dual-Lane against baselines on
synthetic and real-world datasets with burst churn patterns.

Usage:
    python run_experiments.py --dataset synthetic --trials 3 --seed 42
    python run_experiments.py --dataset facebook --filepath data/facebook.txt
    python run_experiments.py --all --output results/
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dual_lane import DualLane
from baselines import IncrementalLPA, WindowedLouvain, BatchLouvain
from stream_generator import StreamGenerator, StreamConfig, OperationType, RealDatasetStream
from metrics import MetricsCollector, compute_statistical_tests, format_results_table


def run_single_trial(
    algorithm,
    stream_generator,
    batch_louvain: BatchLouvain,
    verbose: bool = True
) -> Dict:
    """
    Run a single experimental trial.
    
    Parameters
    ----------
    algorithm : DualLane or baseline algorithm
    stream_generator : StreamGenerator or RealDatasetStream
    batch_louvain : BatchLouvain for quality reference
    verbose : bool
    
    Returns
    -------
    dict
        Trial results including latency and quality metrics
    """
    collector = MetricsCollector()
    algorithm.reset() if hasattr(algorithm, 'reset') else None
    batch_louvain.reset()
    
    operation_count = 0
    last_report = 0
    
    for op in stream_generator.generate():
        # Process operation
        if op.op_type == OperationType.ADD:
            latency = algorithm.add_edge(op.u, op.v)
            batch_louvain.add_edge(op.u, op.v)
        else:
            latency = algorithm.delete_edge(op.u, op.v)
            batch_louvain.delete_edge(op.u, op.v)
        
        collector.record_latency(latency, is_burst=op.is_burst, timestamp=op.timestamp)
        operation_count += 1
        
        # Progress reporting
        if verbose and operation_count - last_report >= 10000:
            print(f"    Processed {operation_count} operations...")
            last_report = operation_count
    
    # Final quality measurement
    final_modularity = algorithm.get_modularity()
    reference_modularity = batch_louvain.get_modularity()
    
    collector.record_modularity(final_modularity)
    
    # Get summary
    summary = collector.get_summary()
    summary['quality']['reference_modularity'] = reference_modularity
    summary['quality']['relative_quality'] = (
        final_modularity / reference_modularity * 100
        if reference_modularity > 0 else 0
    )
    
    # Add algorithm-specific stats
    if hasattr(algorithm, 'get_statistics'):
        summary['algorithm_stats'] = algorithm.get_statistics()
    
    return summary


def run_experiment(
    dataset_name: str,
    config: StreamConfig,
    algorithms: Dict,
    num_trials: int = 3,
    seeds: List[int] = [42, 123, 456],
    verbose: bool = True
) -> Dict:
    """
    Run complete experiment with multiple trials.
    
    Parameters
    ----------
    dataset_name : str
        Name for the dataset
    config : StreamConfig
        Stream configuration
    algorithms : dict
        Dictionary mapping names to algorithm instances
    num_trials : int
        Number of independent trials
    seeds : list of int
        Random seeds for each trial
    verbose : bool
    
    Returns
    -------
    dict
        Complete experiment results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment: {dataset_name}")
        print(f"{'='*60}")
    
    results = {name: [] for name in algorithms.keys()}
    batch_louvain = BatchLouvain()
    
    for trial_idx in range(num_trials):
        seed = seeds[trial_idx % len(seeds)]
        config.seed = seed
        
        if verbose:
            print(f"\n  Trial {trial_idx + 1}/{num_trials} (seed={seed})")
        
        for name, algo_class in algorithms.items():
            if verbose:
                print(f"    Running {name}...")
            
            # Create fresh instances
            if name == "Dual-Lane":
                algorithm = DualLane(T=5.0, K=10, tau_d=0.7)
            elif name == "Incremental LPA":
                algorithm = IncrementalLPA()
            elif name == "Windowed Louvain":
                algorithm = WindowedLouvain(window_size=20000, recompute_interval=100)
            else:
                algorithm = algo_class()
            
            # Reset batch Louvain
            batch_louvain.reset()
            
            # Create stream generator
            stream_gen = StreamGenerator(config)
            
            # Run trial
            start_time = time.time()
            trial_result = run_single_trial(
                algorithm, stream_gen, batch_louvain, verbose=False
            )
            elapsed = time.time() - start_time
            
            trial_result['elapsed_seconds'] = elapsed
            results[name].append(trial_result)
            
            if verbose:
                latency = trial_result['latency']
                quality = trial_result['quality']
                print(f"      Normal: {latency['normal_mean_ms']:.3f}ms, "
                      f"Burst: {latency['burst_mean_ms']:.3f}ms, "
                      f"Quality: {quality['relative_quality']:.1f}%")
    
    return aggregate_results(results, dataset_name)


def aggregate_results(results: Dict[str, List[Dict]], dataset_name: str) -> Dict:
    """Aggregate results across trials."""
    aggregated = {'dataset': dataset_name, 'algorithms': {}}
    
    for name, trials in results.items():
        if not trials:
            continue
        
        # Aggregate latency
        normal_means = [t['latency']['normal_mean_ms'] for t in trials]
        burst_means = [t['latency']['burst_mean_ms'] for t in trials]
        ratios = [t['latency']['stability_ratio'] for t in trials]
        qualities = [t['quality']['relative_quality'] for t in trials]
        modularities = [t['quality']['final_modularity'] for t in trials]
        
        aggregated['algorithms'][name] = {
            'latency': {
                'normal_mean': np.mean(normal_means),
                'normal_std': np.std(normal_means),
                'normal_ci': (
                    np.mean(normal_means) - 1.96 * np.std(normal_means) / np.sqrt(len(normal_means)),
                    np.mean(normal_means) + 1.96 * np.std(normal_means) / np.sqrt(len(normal_means))
                ),
                'burst_mean': np.mean(burst_means),
                'burst_std': np.std(burst_means),
                'burst_ci': (
                    np.mean(burst_means) - 1.96 * np.std(burst_means) / np.sqrt(len(burst_means)),
                    np.mean(burst_means) + 1.96 * np.std(burst_means) / np.sqrt(len(burst_means))
                ),
                'stability_ratio_mean': np.mean(ratios),
                'stability_ratio_std': np.std(ratios)
            },
            'quality': {
                'modularity_mean': np.mean(modularities),
                'modularity_std': np.std(modularities),
                'relative_quality_mean': np.mean(qualities),
                'relative_quality_std': np.std(qualities)
            },
            'num_trials': len(trials),
            'raw_trials': trials
        }
    
    # Statistical tests (Dual-Lane vs LPA)
    if 'Dual-Lane' in results and 'Incremental LPA' in results:
        dl_burst = [t['latency']['burst_mean_ms'] for t in results['Dual-Lane']]
        lpa_burst = [t['latency']['burst_mean_ms'] for t in results['Incremental LPA']]
        
        if len(dl_burst) >= 2 and len(lpa_burst) >= 2:
            tests = compute_statistical_tests(lpa_burst, dl_burst)
            aggregated['statistical_tests'] = {
                't_statistic': tests.t_statistic,
                'p_value': tests.p_value,
                'cohens_d': tests.cohens_d,
                'effect_size': tests.effect_size
            }
    
    return aggregated


def print_results_summary(results: Dict):
    """Print formatted results summary."""
    print(f"\n{'='*70}")
    print(f"Results Summary: {results['dataset']}")
    print(f"{'='*70}")
    
    # Header
    print(f"{'Algorithm':<20} {'Normal (ms)':<14} {'Burst (ms)':<14} "
          f"{'Ratio':<10} {'Quality (%)':<12}")
    print("-" * 70)
    
    for name, data in results['algorithms'].items():
        lat = data['latency']
        qual = data['quality']
        
        normal_str = f"{lat['normal_mean']:.3f}±{lat['normal_std']:.3f}"
        burst_str = f"{lat['burst_mean']:.3f}±{lat['burst_std']:.3f}"
        ratio_str = f"{lat['stability_ratio_mean']:.2f}"
        quality_str = f"{qual['relative_quality_mean']:.1f}±{qual['relative_quality_std']:.1f}"
        
        print(f"{name:<20} {normal_str:<14} {burst_str:<14} "
              f"{ratio_str:<10} {quality_str:<12}")
    
    # Statistical tests
    if 'statistical_tests' in results:
        tests = results['statistical_tests']
        print(f"\n{'Statistical Tests (Dual-Lane vs LPA):'}")
        print(f"  t-statistic: {tests['t_statistic']:.2f}")
        print(f"  p-value: {tests['p_value']:.2e}")
        print(f"  Cohen's d: {tests['cohens_d']:.2f} ({tests['effect_size']})")


def save_results(results: Dict, output_path: str):
    """Save results to JSON file."""
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(v) for v in obj)
        return obj
    
    results_clean = convert(results)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Dual-Lane experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --dataset synthetic
  python run_experiments.py --dataset facebook --filepath data/facebook.txt
  python run_experiments.py --all --output results/
        """
    )
    
    parser.add_argument(
        '--dataset', type=str, default='synthetic',
        choices=['synthetic', 'email-eu', 'facebook', 'twitter', 'custom'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--filepath', type=str, default=None,
        help='Path to edge list file for real datasets'
    )
    parser.add_argument(
        '--trials', type=int, default=3,
        help='Number of independent trials'
    )
    parser.add_argument(
        '--operations', type=int, default=100000,
        help='Number of streaming operations'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Base random seed'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run experiments on all synthetic configurations'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Setup algorithms
    algorithms = {
        'Dual-Lane': DualLane,
        'Incremental LPA': IncrementalLPA,
        'Windowed Louvain': WindowedLouvain
    }
    
    # Setup seeds
    seeds = [args.seed, args.seed + 81, args.seed + 414]
    
    # Dataset configurations
    configs = {
        'synthetic': StreamConfig(
            num_nodes=5000,
            initial_edges=10000,
            total_operations=args.operations,
            window_size=20000,
            normal_delete_rate=0.15,
            burst_interval=50000,
            burst_delete_rate=0.50,
            burst_duration=5000
        ),
        'email-eu': StreamConfig(
            num_nodes=1005,
            initial_edges=25571,
            total_operations=args.operations,
            window_size=20000,
            normal_delete_rate=0.15,
            burst_interval=50000,
            burst_delete_rate=0.50,
            burst_duration=5000
        ),
        'facebook': StreamConfig(
            num_nodes=4039,
            initial_edges=88234,
            total_operations=args.operations,
            window_size=20000,
            normal_delete_rate=0.15,
            burst_interval=50000,
            burst_delete_rate=0.50,
            burst_duration=5000
        )
    }
    
    print("=" * 60)
    print("Dual-Lane Experimental Evaluation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Trials: {args.trials}")
    print(f"Operations: {args.operations}")
    print(f"Seeds: {seeds[:args.trials]}")
    
    all_results = []
    
    if args.all:
        # Run on all synthetic configurations
        for name, config in configs.items():
            results = run_experiment(
                name, config, algorithms,
                num_trials=args.trials,
                seeds=seeds,
                verbose=not args.quiet
            )
            print_results_summary(results)
            all_results.append(results)
    else:
        # Single dataset
        config = configs.get(args.dataset, configs['synthetic'])
        
        results = run_experiment(
            args.dataset, config, algorithms,
            num_trials=args.trials,
            seeds=seeds,
            verbose=not args.quiet
        )
        print_results_summary(results)
        all_results.append(results)
    
    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output, f"results_{timestamp}.json")
        
        save_results({'experiments': all_results}, output_path)
    
    print("\n" + "=" * 60)
    print("Experiments completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
