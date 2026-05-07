# DRL-DPKI

Research codebase for Certificate Authority (CA) selection in a DPKI-style setting using Deep Reinforcement Learning, with baseline strategy comparisons, malicious-behavior simulations, and cost analysis utilities.

The core RL implementation uses Double DQN (DDQN) with experience replay and a target network to learn a CA-selection policy. The scripts under `src/` are runnable experiment entry points (training plus analysis/plots).

## Repository Layout

- `src/`
  - `run_full_experiment.py`: full DDQN training and analysis pipeline
  - `run_ablation_no_hunger.py`: ablation where hunger features/updates are disabled
  - `run_ablation_no_slidewindow.py`: ablation where sliding-window statistics are disabled
- `baseline_methods/`: baseline strategies and an experiment manager for comparisons
- `malicious_behavior_experiment/`: scripts for generating/testing custom scenarios and visualizing saved results
- `cost_analysis/`: off-chain and on-chain cost analysis helpers (Python, Solidity, and Node.js)

## Requirements

This repo does not include a pinned `requirements.txt` or `pyproject.toml`. The Python scripts commonly rely on:

- Python 3.x
- `torch`
- `numpy`
- `pandas`
- `matplotlib`

Optional, depending on what you run:

- Node.js and `web3` (for parts of `cost_analysis/`)

Example installation (adjust versions to your environment):

```bash
pip install torch numpy pandas matplotlib
```

## Quick Start

From the repository root:

```bash
python src/run_full_experiment.py
```

## Experiments (src)

These scripts are intended as executable experiment entry points (not unit tests). They typically include training, reporting, and figure generation.

Run the full experiment:

```bash
python src/run_full_experiment.py
```

Run ablations:

```bash
python src/run_ablation_no_hunger.py
python src/run_ablation_no_slidewindow.py
```

### GPU Selection

The scripts under `src/` set `CUDA_VISIBLE_DEVICES` at the top of the file (commonly to `1`). Change that value or remove it if it does not match your machine.

## Baseline Methods

The baseline framework lives under `baseline_methods/`. The main entry point is:

```bash
python baseline_methods/main.py --help
```

Example usage patterns are shown in the docstring of `baseline_methods/main.py`.

## Malicious Behavior Experiments

Scripts under `malicious_behavior_experiment/` support running custom scenarios and visualizing saved outputs:

```bash
python malicious_behavior_experiment/run_custom_dataset_experiment.py
python malicious_behavior_experiment/visualize_results.py
```

## Outputs and Paths

Outputs (plots, checkpoints, CSVs) are written under an output root directory:

- If `DRL_DPKI_OUTPUT_DIR` is set, scripts use that as the output root.
- Otherwise, if `/mnt/data/wy2024` exists, scripts keep using it for backward compatibility.
- Otherwise, scripts write to `./outputs/` under the repository root.

## Notes

- This repository is organized as runnable research scripts rather than a packaged Python library.
- Smart-contract code in `cost_analysis/onchain_cost/` is provided for experimentation only and is not security-audited.

## Partial Open Source Notice

This repository is partially open-sourced to share the overall framework, design ideas, and reproducible components of the project.

Some core modules and implementation details are currently withheld because they are still under active research and development. These components may involve:
- ongoing experiments
- unpublished results
- directions that are not yet stable enough for public release

We plan to progressively release more parts of the project once the research reaches a more mature stage.

Thank you for your understanding and interest in this work.  
If you have questions or would like to collaborate, feel free to open an issue or get in touch.
