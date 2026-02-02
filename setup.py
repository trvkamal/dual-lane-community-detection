"""
Setup script for Dual-Lane Community Detection package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dual-lane",
    version="1.0.0",
    author="[Ragul Ravi]",
    author_email="[r.ragulravi2005@email.com]",
    description="Bounded-complexity community detection for streaming graphs under burst churn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RagulRM/dual-lane-community-detection",
    project_urls={
        "Bug Tracker": "https://github.com/RagulRM/dual-lane-community-detection/issues",
        "Documentation": "https://github.com/RagulRM/dual-lane-community-detection#readme",
        "Source Code": "https://github.com/RagulRM/dual-lane-community-detection",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "networkx>=2.8",
        "python-louvain>=0.16",
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
        ],
    },
    entry_points={
        "console_scripts": [
            "dual-lane-experiment=run_experiments:main",
        ],
    },
    keywords=[
        "community detection",
        "streaming graphs",
        "graph analytics",
        "bounded complexity",
        "burst churn",
        "social networks",
    ],
)
