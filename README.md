# SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning

<p align="center">
  <a href="https://arxiv.org/abs/2601.04809"><img src="https://img.shields.io/badge/arXiv-2601.04809-b31b1b.svg" /></a>
  <a href="https://doi.org/10.48550/arXiv.2601.04809"><img src="https://img.shields.io/badge/DOI-10.48550%2FarXiv.2601.04809-blue.svg" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-4b5bdc.svg" /></a>
</p>

Official codebase for **SCALER** (arXiv 2026): a framework for synthesizing **verifiable, difficulty-controllable reasoning environments** from real-world programming problems, and training LLMs with **adaptive multi-environment RL** to sustain informative learning signals over long horizons.

- Paper: **SCALER: Synthetic sCalable Adaptive Learning Environment for Reasoning** — https://arxiv.org/abs/2601.04809
- This repository is built on top of [**verl**](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLMs) and follows its environment/runtime conventions.

<p align="center">
  <img src="figures/rl_example.png" width="900" />
</p>



---

## Table of Contents
- [SCALER: Synthetic Scalable Adaptive Learning Environment for Reasoning](#scaler-synthetic-scalable-adaptive-learning-environment-for-reasoning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Core Ideas](#core-ideas)
    - [Scalable Environment Synthesis](#scalable-environment-synthesis)
    - [Adaptive Multi-Environment RL](#adaptive-multi-environment-rl)
  - [Quickstart](#quickstart)
    - [1) Setup](#1-setup)
    - [2) Construct environments](#2-construct-environments)
    - [3) Train](#3-train)
  - [Repository Layout](#repository-layout)
  - [Key Results](#key-results)
  - [Citation](#citation)
  - [License \& Acknowledgements](#license--acknowledgements)
  - [Contact](#contact)
  - [中文简介](#中文简介)

---

## Overview

Reinforcement learning (RL) can enhance LLM reasoning, but progress often slows when:
1) task difficulty drifts away from the model’s capability frontier (too easy / too hard), or  
2) training is dominated by a narrow set of recurring patterns, reducing distributional diversity.

**SCALER** addresses both via *co-adaptation* between the model and training environments:
- a scalable synthesis pipeline that converts real-world programming problems into **verifiable environments** with **controllable difficulty** and **unbounded instance generation**;
- an adaptive multi-environment RL strategy that dynamically adjusts instance difficulty and curates the active set of environments to maintain informative rewards and sustain improvement.

---

## Core Ideas

### Scalable Environment Synthesis
Given a programming problem (statement + reference solution), SCALER synthesizes a reasoning environment with:
- **Verifiability**: deterministic oracle / unit tests provide correctness signals.
- **Difficulty control**: explicit scale parameters discretized into difficulty levels.
- **Unbounded instance generation**: randomized testcase generation yields unlimited training instances.

### Adaptive Multi-Environment RL
SCALER sustains learning signals at two levels:
- **In-environment difficulty controller**: keeps sampling near a target success regime.
- **Environment curation**: maintains an active set and replaces saturated/uninformative environments to preserve diversity and long-horizon improvements.

---

## Quickstart

### 1) Setup

This repo follows **verl** for environment setup (CUDA / PyTorch / distributed runtime / Docker, etc.).
Please refer to:
- verl documentation: https://verl.readthedocs.io/en/latest/index.html
- verl repo: https://github.com/volcengine/verl

> Tip: If you already have verl working on your machine/cluster, SCALER should be a minimal delta.

### 2) Construct environments

SCALER’s environment synthesis pipeline entry:
- `SCALER/environment_construct.sh`

Run:
```bash
bash SCALER/environment_construct.sh
````

Notes:

* The script is intended as the **one-click pipeline** entry. Customize dataset paths / output dirs / parallelism in the script as needed.
* Synthesized environments and metadata are typically managed under `SCALER-data/` (see repo layout).

### 3) Train

Paper-style training runs are organized under `recipe/`.
A concrete entry (Qwen3-1.7B, 2739 envs):

* `recipe/environment/qwen3-1.7b-2739-envs.sh`

Run:

```bash
bash recipe/environment/qwen3-1.7b-2739-envs.sh
```

Notes:

* This script is the **actual training entry**. It typically sets model, env pool, runtime (GPU / distributed), and logging.
* If you modify environment pool / difficulty scheduling / curation knobs, do it by editing the recipe script (and/or its referenced config files).

---

## Repository Layout

High-level structure (major directories):

* `SCALER/` — SCALER core code (synthesis, controllers, curation, integration).
* `SCALER-data/` — environment pools / metadata / released artifacts (if any).
* `recipe/` — runnable training / evaluation recipes (paper entry points).
* `verl/` — upstream training infrastructure (forked / vendored).
* `docs/` — documentation + figures + examples (recommended place for README images).
* `scripts/`, `examples/` — utilities and demos.

---

## Key Results

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

## Citation

If you use SCALER in your research, please cite:

```bibtex
@article{xu2026scaler,
  title   = {SCALER: Synthetic sCalable Adaptive Learning Environment for Reasoning},
  author  = {Xu, Caijun and Xiao, Changyi and Peng, Zhongyuan and Wang, Xinrun and Cao, Yixin},
  journal = {arXiv preprint arXiv:2601.04809},
  year    = {2026},
  doi     = {10.48550/arXiv.2601.04809}
}
```

---

## License & Acknowledgements

* Released under **Apache License 2.0** (see `LICENSE`).
* Built on top of **verl** and reuses its training infrastructure. Please also check upstream verl license/notice files when redistributing.

---

## Contact

Correspondence (paper): **[cjxu25@m.fudan.edu.cn](mailto:cjxu25@m.fudan.edu.cn)**

---

## 中文简介

SCALER（Synthetic sCalable Adaptive Learning Environment for Reasoning）旨在让推理类 RL 训练长期保持“有效监督信号”。核心包括：

1. **环境合成管道**：将真实编程题系统化转化为“可验证、可控难度、可无限生成实例”的推理环境（testcase generator + oracle/unit tests）。
2. **多环境自适应 RL**：在单环境内做在线难度控制，在跨环境层面做环境集合的动态维护（curation），从而追踪能力边界并保持训练分布多样性。

入口脚本：

* 合成环境：`SCALER/environment_construct.sh`
* 训练入口：`recipe/environment`

