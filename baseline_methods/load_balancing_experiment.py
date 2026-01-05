#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run load balancing experiments for a subset of baseline strategies.

This script measures how often each strategy selects every CA (node agent)
within the simulated environment. Results are written to JSON and a heatmap
figure is generated to visualize selection frequencies.

Usage:
    python load_balancing_experiment.py --steps 100 \
        --output results/load_balancing/selection_summary.json \
        --figure results/load_balancing/selection_heatmap.png

The script relies on existing environment and strategy implementations found
in environment.py and models.py.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from environment import NAConfig, create_baseline_environment
from models import (
    DDQNModelStrategy,
    EpsilonGreedyStrategy,
    GeneticAlgorithmStrategy,
    PartialSemiGreedyStrategy,
    RandomSelectionStrategy,
    ReputationBasedStrategy,
    RoundRobinStrategy,
    WeightedScoreStrategy,
)

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PROJECT_ROOT = BASE_DIR.parent

DRL_MODEL_BASELINE_BALANCE = PROJECT_ROOT / "models_baseline_balance" / "policy_net_state_dict.pth"
DRL_MODEL_ABLATION_NO_HUNGER = PROJECT_ROOT / "models_ablation_no_hunger" / "policy_net_state_dict.pth"

StrategyInfo = Tuple[str, Callable[[], object]]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_strategies() -> List[StrategyInfo]:
    """Instantiate the strategies requested for the experiment."""
    return [
        ("random", RandomSelectionStrategy),
        ("weighted", WeightedScoreStrategy),
        ("reputation", ReputationBasedStrategy),
        ("round_robin", RoundRobinStrategy),
        ("psg", PartialSemiGreedyStrategy),
        ("ga", GeneticAlgorithmStrategy),
        ("ddqn", DDQNModelStrategy),
    ]


def run_strategy(
    label: str,
    strategy_cls,
    steps: int,
    config: NAConfig,
    seed: int,
) -> Dict[str, object]:
    """Execute a single strategy and collect NA selection counts."""
    set_global_seed(seed)
    setattr(config, "random_seed", seed)
    env = create_baseline_environment(config)
    strategy = strategy_cls()

    # Track how often each NA is selected.
    selection_counts = np.zeros(config.n_na, dtype=int)

    state = env.reset()
    env.current_strategy = strategy

    steps_executed = 0
    for step in range(steps):
        selected = strategy.select(state, env)
        for idx in selected:
            selection_counts[idx] += 1
        state, _, done, _ = env.step(selected)
        steps_executed += 1
        if done:
            break

    env.current_strategy = None

    return {
        "strategy_name": strategy.name if hasattr(strategy, "name") else label,
        "label": label,
        "seed": seed,
        "counts": selection_counts,
        "steps_executed": steps_executed,
        "malicious_nas": sorted(env.malicious_nas) if hasattr(env, "malicious_nas") else [],
    }


def _run_baseline_experiment_summary(
    steps: int,
    seeds: List[int],
    n_na: int,
    strategies: Optional[List[StrategyInfo]] = None,
) -> Dict[str, object]:
    config = NAConfig(
        n_na=n_na,
        max_steps=steps,
        transactions_per_step=10,
        malicious_attack_enabled=True,
        malicious_attack_indices=[1, 11],
    )
    if strategies is None:
        strategies = build_strategies()

    summary: Dict[str, object] = {
        "steps_requested": steps,
        "n_na": config.n_na,
        "selected_na_per_step": config.selected_na_count,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "results": {},
    }

    malicious_indices: List[int] = [1, 11]

    for label, strategy_cls in strategies:
        print(f"\nRunning strategy: {label}")

        per_seed_counts: List[np.ndarray] = []
        per_seed_steps: List[int] = []
        per_seed_totals: List[int] = []
        per_seed_freqs: List[np.ndarray] = []
        strategy_name: str = label

        for seed in seeds:
            try:
                result = run_strategy(label, strategy_cls, steps, config, seed=seed)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Strategy {label} failed (seed={seed}): {exc}")
                continue

            strategy_name = result["strategy_name"]
            counts = result["counts"].astype(int)
            total_selections = int(counts.sum())
            denom = max(total_selections, 1)
            freqs = counts.astype(float) / denom

            per_seed_counts.append(counts)
            per_seed_steps.append(int(result["steps_executed"]))
            per_seed_totals.append(total_selections)
            per_seed_freqs.append(freqs)

            if not malicious_indices and result.get("malicious_nas"):
                malicious_indices = result["malicious_nas"]

        if not per_seed_counts:
            summary["results"][label] = {"error": "all seeds failed"}
            continue

        counts_stack = np.stack(per_seed_counts, axis=0)
        freqs_stack = np.stack(per_seed_freqs, axis=0)

        counts_sum = counts_stack.sum(axis=0).astype(int)
        counts_mean = counts_stack.mean(axis=0)
        freqs_mean = freqs_stack.mean(axis=0)

        ddof = 1 if len(per_seed_counts) > 1 else 0
        counts_std = counts_stack.std(axis=0, ddof=ddof)
        freqs_std = freqs_stack.std(axis=0, ddof=ddof)

        avg_selected_per_step_per_seed = []
        for total, steps_executed in zip(per_seed_totals, per_seed_steps):
            avg_selected_per_step_per_seed.append(
                total / max(steps_executed, 1) / config.selected_na_count
            )

        summary["results"][label] = {
            "strategy_name": strategy_name,
            "seeds_executed": len(per_seed_counts),
            "steps_executed_per_seed": per_seed_steps,
            "total_selections_per_seed": per_seed_totals,
            "total_selections": int(counts_sum.sum()),
            "selection_counts": counts_sum.tolist(),
            "selection_counts_mean": counts_mean.tolist(),
            "selection_counts_std": counts_std.tolist(),
            "selection_frequencies": freqs_mean.tolist(),
            "selection_frequencies_std": freqs_std.tolist(),
            "average_selected_per_step_mean": float(np.mean(avg_selected_per_step_per_seed)),
            "average_selected_per_step_std": float(np.std(avg_selected_per_step_per_seed, ddof=ddof)),
        }

    summary["malicious_nas"] = malicious_indices

    return summary


def run_experiment(steps: int, output_path: Path, seeds: List[int]) -> Dict[str, object]:
    """Run the load balancing experiment across all target strategies."""
    summary = _run_baseline_experiment_summary(steps=steps, seeds=seeds, n_na=20)

    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return summary


def run_ca_scale_sensitivity_experiment(
    steps: int,
    output_path: Path,
    seeds: List[int],
) -> Dict[str, object]:
    ca_counts = [400, 1000, 2500, 5200, 6000]

    if not DRL_MODEL_BASELINE_BALANCE.is_file():
        raise FileNotFoundError(f"Model not found: {DRL_MODEL_BASELINE_BALANCE}")

    strategies: List[StrategyInfo] = [
        ("drl_baseline_balance", lambda: DDQNModelStrategy(model_path=str(DRL_MODEL_BASELINE_BALANCE))),
    ]

    summary: Dict[str, object] = {
        "mode": "ca_scale_sensitivity",
        "steps_requested": steps,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ca_counts": ca_counts,
        "results_by_ca_count": {},
    }

    for n_na in ca_counts:
        print(f"\n=== CA scale sensitivity: M={n_na} ===")
        per_summary = _run_baseline_experiment_summary(
            steps=steps,
            seeds=seeds,
            n_na=n_na,
            strategies=strategies,
        )
        summary["results_by_ca_count"][str(n_na)] = per_summary

    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return summary


def run_drl_model_comparison(steps: int, output_path: Path, seeds: List[int]) -> Dict[str, object]:
    config = NAConfig(
        n_na=20,
        max_steps=steps,
        transactions_per_step=10,
        malicious_attack_enabled=True,
        malicious_attack_indices=[1, 11],
    )

    if not DRL_MODEL_BASELINE_BALANCE.is_file():
        raise FileNotFoundError(f"Model not found: {DRL_MODEL_BASELINE_BALANCE}")
    if not DRL_MODEL_ABLATION_NO_HUNGER.is_file():
        raise FileNotFoundError(f"Model not found: {DRL_MODEL_ABLATION_NO_HUNGER}")

    strategies: List[StrategyInfo] = [
        ("drl_baseline_balance", lambda: DDQNModelStrategy(model_path=str(DRL_MODEL_BASELINE_BALANCE))),
        ("drl_ablation_no_hunger", lambda: DDQNModelStrategy(model_path=str(DRL_MODEL_ABLATION_NO_HUNGER))),
    ]

    summary: Dict[str, object] = {
        "mode": "drl_model_comparison",
        "steps_requested": steps,
        "n_na": config.n_na,
        "selected_na_per_step": config.selected_na_count,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "models": {
            "drl_baseline_balance": str(DRL_MODEL_BASELINE_BALANCE),
            "drl_ablation_no_hunger": str(DRL_MODEL_ABLATION_NO_HUNGER),
        },
        "results": {},
    }

    malicious_indices: List[int] = [1, 11]

    for label, strategy_cls in strategies:
        print(f"\nRunning DRL-DPKI model: {label}")

        per_seed_counts: List[np.ndarray] = []
        per_seed_steps: List[int] = []
        per_seed_totals: List[int] = []
        per_seed_freqs: List[np.ndarray] = []
        strategy_name: str = label

        for seed in seeds:
            try:
                result = run_strategy(label, strategy_cls, steps, config, seed=seed)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"DRL-DPKI model {label} failed (seed={seed}): {exc}")
                continue

            strategy_name = result["strategy_name"]
            counts = result["counts"].astype(int)
            total_selections = int(counts.sum())
            denom = max(total_selections, 1)
            freqs = counts.astype(float) / denom

            per_seed_counts.append(counts)
            per_seed_steps.append(int(result["steps_executed"]))
            per_seed_totals.append(total_selections)
            per_seed_freqs.append(freqs)

            if not malicious_indices and result.get("malicious_nas"):
                malicious_indices = result["malicious_nas"]

        if not per_seed_counts:
            summary["results"][label] = {"error": "all seeds failed"}
            continue

        counts_stack = np.stack(per_seed_counts, axis=0)
        freqs_stack = np.stack(per_seed_freqs, axis=0)

        counts_sum = counts_stack.sum(axis=0).astype(int)
        counts_mean = counts_stack.mean(axis=0)
        freqs_mean = freqs_stack.mean(axis=0)

        ddof = 1 if len(per_seed_counts) > 1 else 0
        counts_std = counts_stack.std(axis=0, ddof=ddof)
        freqs_std = freqs_stack.std(axis=0, ddof=ddof)

        avg_selected_per_step_per_seed = []
        for total, steps_executed in zip(per_seed_totals, per_seed_steps):
            avg_selected_per_step_per_seed.append(
                total / max(steps_executed, 1) / config.selected_na_count
            )

        summary["results"][label] = {
            "strategy_name": strategy_name,
            "seeds_executed": len(per_seed_counts),
            "steps_executed_per_seed": per_seed_steps,
            "total_selections_per_seed": per_seed_totals,
            "total_selections": int(counts_sum.sum()),
            "selection_counts": counts_sum.tolist(),
            "selection_counts_mean": counts_mean.tolist(),
            "selection_counts_std": counts_std.tolist(),
            "selection_frequencies": freqs_mean.tolist(),
            "selection_frequencies_std": freqs_std.tolist(),
            "average_selected_per_step_mean": float(np.mean(avg_selected_per_step_per_seed)),
            "average_selected_per_step_std": float(np.std(avg_selected_per_step_per_seed, ddof=ddof)),
        }

    summary["malicious_nas"] = malicious_indices

    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_file:
        json.dump(summary, out_file, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return summary


def _prompt_choice(prompt: str, default: str = "1") -> str:
    try:
        value = input(prompt).strip()
    except EOFError:
        value = ""
    return value or default


def choose_run_mode() -> str:
    print("\n=== Load Balancing Experiment ===")
    print("1) Baseline experiment (existing logic)")
    print("2) DRL-DPKI model comparison: models_baseline_balance vs models_ablation_no_hunger")
    print("3) CA scale sensitivity: M ∈ {400, 1000, 2500, 5200, 6000}")
    return _prompt_choice("Select mode [1/2/3] (default=1): ", default="1")


def parse_args() -> argparse.Namespace:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Load balancing experiment for CA strategies")
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of environment steps to run for each strategy",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds for independent repeats",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed (actual seeds are seed..seed+seeds-1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "results" / "load_balancing" / "selection_summary.json",
        help="Path to the JSON output file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    seeds = list(range(args.seed, args.seed + max(int(args.seeds), 1)))

    mode = choose_run_mode()
    if mode == "2":
        output_path = args.output
        default_output = BASE_DIR / "results" / "load_balancing" / "selection_summary.json"
        if output_path == default_output:
            output_path = BASE_DIR / "results" / "load_balancing" / "drl_model_comparison.json"
        run_drl_model_comparison(steps=args.steps, output_path=output_path, seeds=seeds)
    elif mode == "3":
        output_path = args.output
        default_output = BASE_DIR / "results" / "load_balancing" / "selection_summary.json"
        if output_path == default_output:
            output_path = BASE_DIR / "results" / "load_balancing" / "ca_scale_sensitivity.json"
        run_ca_scale_sensitivity_experiment(steps=args.steps, output_path=output_path, seeds=seeds)
    else:
        run_experiment(steps=args.steps, output_path=args.output, seeds=seeds)
