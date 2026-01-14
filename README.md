# SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2601.04809-b31b1b.svg)](https://arxiv.org/abs/2601.04809)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2601.04809-blue.svg)](https://doi.org/10.48550/arXiv.2601.04809)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-4b5bdc.svg)](LICENSE)

Official codebase for **SCALER** (arXiv 2026): a framework for **synthesizing verifiable, difficulty-controllable reasoning environments** from real-world programming problems, and **training LLMs with adaptive multi-environment RL** to sustain informative learning signals over long horizons.

> Paper: **SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning**  
> arXiv: [2601.04809](https://arxiv.org/abs/2601.04809) (submitted Jan 8, 2026)

---

## Table of Contents

- [SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning](#scaler-synthetic-scalable-adaptive-learning-environment-for-reasoning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Core Ideas](#core-ideas)
    - [Scalable Environment Synthesis](#scalable-environment-synthesis)
    - [Adaptive Multi-Environment RL](#adaptive-multi-environment-rl)
  - [Key Results (arXiv v1)](#key-results-arxiv-v1)
  - [Repository Layout](#repository-layout)
  - [Installation](#installation)
- [(recommended) create env](#recommended-create-env)
- [conda create -n scaler python=3.10 -y](#conda-create--n-scaler-python310--y)
- [conda activate scaler](#conda-activate-scaler)
- [Optional: CUDA / GPU-related dependencies](#optional-cuda--gpu-related-dependencies)
- [Editable install (if the project is packaged)](#editable-install-if-the-project-is-packaged)

---

## Overview

Reinforcement learning (RL) can improve LLM reasoning, but progress often slows when:
1) task difficulty drifts away from the model’s capability frontier (becoming too easy or too hard), or  
2) training is dominated by a narrow set of recurring patterns, reducing distributional diversity.

**SCALER** addresses both via *co-adaptation* between the model and training environments:
- a **synthesis pipeline** that converts real-world programming problems into **verifiable reasoning environments** with **controllable difficulty** and **unbounded instance generation**;
- an **adaptive multi-environment RL strategy** that dynamically adjusts instance difficulty and curates the active environment set to maintain informative rewards and sustain improvement.

---

## Core Ideas

### Scalable Environment Synthesis

Given a programming problem (statement + reference solution), SCALER synthesizes a reasoning environment with:

- **Verifiability**: deterministic oracle / unit tests provide correctness signals.
- **Difficulty control**: explicit scale parameters (e.g., array length, graph size), discretized into levels.
- **Unbounded instance generation**: randomized testcase generation yields unlimited training instances.

The pipeline in the paper includes (high level):
- **Meta-information extraction** (I/O formats, constraints, scale parameters, etc.).
- **Testcase generator validation** (breadth/depth checks).
- **Heuristic difficulty calibration** under context-length and time-limit constraints (e.g., binary search for feasible max scale; arithmetic/geometric discretization depending on range span).

### Adaptive Multi-Environment RL

SCALER maintains effective signals at two levels:

**(1) In-environment difficulty controller**  
An online controller updates per-environment difficulty based on on-policy accuracy to keep samples near a target success rate (avoiding “all-correct” or “all-wrong” regimes).

**(2) Environment curation mechanism**  
Training uses an **active set** of environments. Environments whose learning signal saturates are retired and replaced (e.g., difficulty stops increasing; becomes consistently trivial or unlearnable).  
Paper setting (v1): `K_slope = 10`, `K_zero = K_sat = 5`.

---

## Key Results (arXiv v1)

Performance on five reasoning benchmarks: **MATH-500**, **AMC23**, **AIME24**, **MMLU-Pro**, **BBEH**.  
Numbers below are taken from Table 1 in the paper (AVG = unweighted mean):

| Base Model | Method | MATH-500 | AMC23 | AIME24 | MMLU-Pro | BBEH | AVG |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B-base | Base | 59.6 | 29.21 | 3.33 | 33.30 | 3.26 | 25.74 |
|  | + SCALER | 75.8 | 49.53 | 12.91 | 50.89 | 11.74 | **40.18** |
| Qwen3-4B-base | Base | 66.4 | 44.70 | 8.75 | 51.60 | 8.10 | 35.91 |
|  | + SCALER | 84.4 | 75.00 | 27.29 | 70.00 | 14.56 | **54.25** |

Environment pool statistics (paper v1): **4973** programming problems → **2739** synthesized SCALER environments.

---

## Repository Layout

This repository is built on top of **[verl](https://github.com/volcengine/verl)** (Volcano Engine Reinforcement Learning for LLMs) and includes SCALER-specific code and assets.

Typical top-level directories:
- `SCALER/` — SCALER core (environment synthesis, controllers, curation, integration code).
- `SCALER-data/` — environment pool metadata / examples / (optional) released assets.
- `verl/` — upstream RL framework codebase.
- `recipe/` — runnable training / evaluation recipes and configs.
- `examples/`, `scripts/`, `docs/` — auxiliary examples, utilities, and documentation.

> If you are looking for “the exact command lines used in the paper”, start from `recipe/` and Appendix 5 of the paper, then adapt paths/checkpoints for your cluster.

---

## Installation

> The codebase follows the dependency style of `verl`. GPU + distributed training are recommended for full reproduction.

```bash
git clone https://github.com/molumolua/SCALER.git
cd SCALER

# (recommended) create env
# conda create -n scaler python=3.10 -y
# conda activate scaler

pip install -U pip
pip install -r requirements.txt

# Optional: CUDA / GPU-related dependencies
pip install -r requirements-cuda.txt

# Editable install (if the project is packaged)
pip install -e .
