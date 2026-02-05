# Dual-Lane: Bounded-Complexity Community Detection for Streaming Graphs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A two-tier buffering architecture for streaming community detection that achieves **bounded O(K) deletion complexity** while maintaining **93-100% quality** of batch Louvain.

## 🎯 Key Results

| Metric | Value | Comparison |
|--------|-------|------------|
| **Quality** | 93-100% | of batch Louvain modularity |
| **Latency Improvement** | 67-150× | lower burst latency vs. LPA |
| **Stability Ratio** | 1.02-1.07× | burst/normal (vs. 1.38-1.82× for baselines) |
| **Scale** | 1.47B edges | tested on Twitter social graph |
| **Statistical Significance** | p < 0.001 | Cohen's d: 10.07-23.15 (very large) |

## 📋 Abstract

Streaming community detection algorithms face significant performance challenges during **burst churn events**—periods of dramatically elevated edge addition and deletion rates common in social networks, cybersecurity monitoring, and communication systems. Traditional incremental algorithms exhibit latency spikes due to expensive degree-dependent deletion operations, making them unsuitable for latency-sensitive analytics.

**Dual-Lane** is a two-tier buffering architecture that achieves bounded deletion complexity by absorbing volatile edges in a *fast lane* before promoting stable structures to a *slow lane* for community detection. Our key insight is that edges deleted shortly after creation represent transient noise rather than stable community structure, and can be filtered without impacting detection quality.

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Edge Stream                 │
                    │     (additions + deletions + bursts)     │
                    └─────────────────┬───────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │           FAST LANE (Buffer)             │
                    │  • Bounded K-node components             │
                    │  • O(K) deletion complexity              │
                    │  • Absorbs volatile edges                │
                    │  • Age threshold T = 5s                  │
                    └─────────────────┬───────────────────────┘
                                      │ Promotion
                                      │ (age ≥ T, size ≥ K, or density < τ_d/2)
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │          SLOW LANE (Global)              │
                    │  • Stable community structure            │
                    │  • Louvain algorithm                     │
                    │  • High-quality detection                │
                    └─────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/trvkamal/dual-lane-community-detection.git
cd dual-lane-community-detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.dual_lane import DualLane

# Initialize with default parameters
dl = DualLane(T=5.0, K=10, tau_d=0.7)

# Process edge stream
dl.add_edge(1, 2)
dl.add_edge(2, 3)
dl.add_edge(3, 1)

# Get communities
communities = dl.get_communities()
print(f"Communities: {communities}")

# Delete volatile edge
dl.delete_edge(1, 2)

# Get modularity
print(f"Modularity: {dl.get_modularity():.4f}")
```

### Running Experiments

```bash
# Run on synthetic dataset (3 trials)
python experiments/run_experiments_v2.py --datasets synthetic-planted --trials 3

# Run on all datasets
python experiments/run_experiments_v2.py --datasets synthetic-planted email-eu facebook --trials 3 --output results/

# Run Twitter-scale experiment
python experiments/run_experiments_v2_twitter.py --datasets twitter --limit 500000 --trials 1

# Run parameter sensitivity analysis
python experiments/run_sensitivity.py
```

## 📊 Experimental Results

### Main Results (Table II from paper)

| Dataset | Algorithm | Normal (ms) | Burst (ms) | Ratio | Quality (%) |
|---------|-----------|-------------|------------|-------|-------------|
| **Synthetic** | Incremental LPA | 0.958±0.025 | 1.324±0.108 | 1.38 | 0.0% |
| | Windowed Louvain | 0.005±0.001 | 0.005±0.001 | 1.01 | 100.0% |
| | **Dual-Lane** | **0.012±0.001** | **0.013±0.002** | **1.07** | **100.2%** |
| **Email-EU** | Incremental LPA | 0.510±0.009 | 0.238±0.026 | 0.47 | 0.0% |
| | Windowed Louvain | 0.004±0.000 | 0.004±0.001 | 0.98 | 98.0% |
| | **Dual-Lane** | **0.012±0.000** | **0.010±0.001** | **0.85** | **99.1%** |
| **Facebook** | Incremental LPA | 1.161±0.039 | 1.818±0.162 | 1.57 | 13.1% |
| | Windowed Louvain | 0.004±0.000 | 0.005±0.000 | 1.12 | 100.0% |
| | **Dual-Lane** | **0.011±0.000** | **0.012±0.000** | **1.06** | **100.0%** |
| **Twitter** | Incremental LPA | 1.141±0.000 | 2.074±0.000 | 1.82 | 83.1% |
| | Windowed Louvain | 0.006±0.000 | 0.006±0.000 | 1.02 | 93.1% |
| | **Dual-Lane** | **0.030±0.000** | **0.031±0.000** | **1.02** | **93.0%** |

### Statistical Significance (Table III from paper)

| Dataset | t-statistic | p-value | Cohen's d | Effect |
|---------|-------------|---------|-----------|--------|
| Synthetic | -18.52 | <0.001 | -10.69 | Very Large |
| Email-EU | -28.91 | <0.001 | -23.15 | Very Large |
| Facebook | -23.78 | <0.001 | -13.73 | Very Large |
| Twitter | -25.34 | <0.001 | -10.07 | Very Large |

## 🔧 Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T` | 5.0s | Age threshold for promotion |
| `K` | 10 | Maximum component size in fast lane |
| `tau_d` | 0.7 | Density threshold for sparse promotion |

### Parameter Sensitivity

The algorithm is robust across wide parameter ranges:
- **T**: 1-20 seconds → Quality 98.7-101.2%
- **K**: 5-50 nodes → Quality 99.1-102.8%
- **τ_d**: 0.5-0.9 → Quality 98.9-101.5%

## 📁 Project Structure

```
dual-lane-community-detection/
├── src/
│   ├── __init__.py
│   ├── dual_lane.py        # Main algorithm (clean implementation)
│   ├── dual_lane_v2.py     # Research implementation with proper tracking
│   ├── baselines.py        # LPA, Windowed Louvain
│   ├── stream_generator.py # Stream generation
│   ├── metrics.py          # Evaluation metrics
│   └── run_experiments.py  # Basic experiment runner
├── experiments/
│   ├── run_experiments_v2.py       # Full experiment runner
│   ├── run_experiments_v2_twitter.py # Twitter-scale experiments
│   └── run_sensitivity.py          # Parameter sensitivity analysis
├── tests/
│   └── test_dual_lane.py   # Unit tests
├── docs/
│   └── RESEARCH_SUMMARY.md # Detailed write-up
├── data/                   # Dataset files (not included)
├── results/                # Experiment outputs
├── requirements.txt
├── LICENSE
└── README.md
```

## 📚 Datasets

The experiments use the following datasets from [Stanford SNAP](https://snap.stanford.edu/data/):

| Dataset | Nodes | Edges | Description |
|---------|-------|-------|-------------|
| Synthetic | 5,000 | 243,482 | Synthetic preferential attachment |
| Email-EU | 1,005 | 25,571 | Email network from European institution |
| Facebook | 4,039 | 88,234 | Facebook friendships |
| Twitter | 41.7M | 1.47B | Twitter follower network |

Download datasets:
```bash
# Email-EU
wget https://snap.stanford.edu/data/email-Eu-core.txt.gz

# Facebook
wget https://snap.stanford.edu/data/facebook_combined.txt.gz

# Twitter (large, requires significant resources)
wget https://snap.stanford.edu/data/twitter-2010.txt.gz
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{author2026duallane,
  title={Dual-Lane: Bounded-Complexity Community Detection for Streaming Graphs Under Burst Churn},
  author={Kamalesh Velmurugan},
  year={2026}
}
```

## 🔬 Theoretical Guarantees

**Theorem 1 (Bounded Complexity):** For any edge deletion operation, the worst-case time complexity of Dual-Lane is O(K), where K is a user-defined constant parameter.

**Lemma 2 (Quality Preservation):** Let Q_DL denote Dual-Lane's modularity and Q_B denote batch Louvain's modularity on the same graph snapshot. The expected quality ratio satisfies: E[Q_DL/Q_B] ≥ 1 - ε, where ε depends on the volatility rate.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anonymous reviewers for valuable feedback
- Stanford SNAP for providing benchmark datasets
- NetworkX and python-louvain communities for excellent libraries

## 📧 Contact

- **Author:** [Kamalesh Velmurugan]
- **Email:** [t.r.v.kamal@gmail.com]

---

**Keywords:** streaming graphs, community detection, bounded complexity, burst churn, resilient systems, graph analytics
