#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Baseline Strategy Test Platform - NA Selection Optimization
============================================================

Features:
- Baseline strategy tests: supports multiple NA selection baselines
- Environment configuration: flexible experiment settings
- Performance evaluation: full analysis and visualization
- Result saving: automatically saves results and figures

Supported baseline strategies:
- Random Selection
- Reputation-based
- Balanced Selection
- Hunger-based
- Weighted Selection

Examples:
    python main.py --test_strategy --strategy random --nas 10 --time_points 150
    python main.py --test_strategy --strategy reputation --nas 10 --time_points 100
"""

import os
import sys
import argparse
import time
import json
import traceback
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import matplotlib.font_manager as fm

# Configure matplotlib backend and style
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Prefer loading project-local custom fonts to avoid missing-font warnings
font_dir = Path(__file__).resolve().parent.parent / 'font'
loaded_custom_font = False
if font_dir.is_dir():
    font_files = sorted(font_dir.glob('*.ttf'))
    for font_path in font_files:
        try:
            fm.fontManager.addfont(str(font_path))
            loaded_custom_font = True
        except Exception as exc:  # pragma: no cover - best-effort font loading
            print(f"Font load failed {font_path.name}: {exc}")

preferred_fonts = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'] if loaded_custom_font else ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
matplotlib.rcParams['font.sans-serif'] = preferred_fonts
matplotlib.rcParams['axes.unicode_minus'] = False

# Add project path
sys.path.append('/mnt/data/wy2024/baseline_methods')

# Import baseline strategies and environment
try:
    from parameter import GlobalConfig
    from models import (
        RandomSelectionStrategy, ReputationBasedStrategy, BalancedReputationStrategy,
        HungerBasedStrategy, WeightedScoreStrategy, RoundRobinStrategy, DDQNModelStrategy,
        AdaptiveStrategy, EpsilonGreedyStrategy, MultiCriteriaStrategy, PartialSemiGreedyStrategy,
        GeneticAlgorithmStrategy
    )
    from environment import NAEnvironment, NAConfig, create_baseline_environment
        
except ImportError as e:
    print(f"Module import failed: {e}")
    print("Ensure parameter.py, models.py, and environment.py exist")
    sys.exit(1)

# Initialize global configuration
global_config = GlobalConfig()

# Configure matplotlib
plt.rcParams['font.family'] = preferred_fonts
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = global_config.visualization.FONT_SIZE


def set_all_seeds(seed: int):
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

class ExperimentManager:
    """
    Baseline strategy test manager focused on testing and evaluation.
    """
    
    def __init__(self, config_name='STANDARD', save_dir=None):
        """
        Initialize the experiment manager.
        
        Args:
            config_name: Configuration name
            save_dir: Result output directory
        """
        self.config = GlobalConfig()
        self.save_dir = save_dir or f"/mnt/data/wy2024/baseline_methods/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir = Path(self.save_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment data storage
        self.experiment_data = {
            'start_time': datetime.now().isoformat(),
            'config_name': config_name,
            'save_dir': str(self.save_dir)
        }
        
        print("Experiment manager initialized")
        print(f"Results directory: {self.results_dir.absolute()}")
    
    def initialize_environment(self, **kwargs):
        """
        Initialize the baseline test environment.
        
        Args:
            **kwargs: Environment parameter overrides
        """
        print("Initializing baseline test environment...")
        
        # Load environment parameters
        env_params = self.config.get_environment_params()
        env_params.update(kwargs)
        
        # Initialize with the two-group mode
        na_config = NAConfig(
            na_init_mode='two_groups',
            osa_attack_enabled=True,
            **env_params
        )
        
        # Create environment
        self.env = create_baseline_environment(na_config)
        
        # Save environment information
        self.experiment_data['environment_info'] = env_params
        self.experiment_data['na_config'] = na_config.__dict__
        
        return self.env
        
    def initialize_environment(self, **kwargs):
        """
        Initialize the experiment environment.
        
        Args:
            **kwargs: Environment parameter overrides
        """
        print("Initializing experiment environment...")
        
        # Load environment parameters
        env_params = self.config.get_environment_params()
        env_params.update(kwargs)
        
        # Convert parameters to the NAConfig format
        na_config_params = {
            'n_na': env_params['n_na'],
            'transactions_per_step': env_params.get('transactions_per_step', 1),
            'malicious_ratio': env_params.get('malicious_ratio', self.config.na_state.MALICIOUS_RATIO)  # From parameter.py
        }
        
        # Create NAEnvironment instance
        try:
            self.env = create_baseline_environment(NAConfig(**na_config_params))
            print(f"NAEnvironment initialized: {na_config_params['n_na']} NAs")
            print(f"  - Malicious NA ratio: {na_config_params.get('malicious_ratio', self.config.na_state.MALICIOUS_RATIO):.1%}")
        except Exception as e:
            print(f"Environment initialization failed: {e}")
            raise
            
        # Save environment information
        self.experiment_data['environment_info'] = env_params
        self.experiment_data['na_config'] = na_config_params
        
        return self.env
    
    def run_strategy_test(self, strategy_name='random', n_time_points=None, n_na=None,
                          na_config: Optional[NAConfig] = None, save_visuals: bool = True,
                          warmup_first_round: bool = False,
                          ddqn_model_path: Optional[str] = None,
                          save_name: Optional[str] = None,
                          env_override: Optional[NAEnvironment] = None):
        """
        Run a strategy test at specified time points.
        
        Args:
            strategy_name: Strategy name ('random', 'reputation', 'balanced', 'hunger', 'weighted', 'ddqn')
            n_time_points: Number of time points (defaults to config)
            n_na: Number of NAs (defaults to config)
            na_config: Optional environment configuration
            save_visuals: Whether to generate and save figures (disable for batch runs)
        """
        # print(f" Start strategy test: {strategy_name}")
        # print("=" * 60)
        
        # Load test parameters
        if n_time_points is None:
            n_time_points = self.config.training.N_TIME_POINTS
        if n_na is None:
            n_na = self.config.training.N_NA
        
        # Load selection and execution parameters
        selected_na_count = self.config.training.SELECTED_NA_COUNT
        execution_cycles = self.config.training.EXECUTION_CYCLES
            
        # Create environment configuration
        if na_config is None:
            config = NAConfig(n_na=n_na, transactions_per_step=self.config.environment.TRANSACTIONS_PER_STEP)
        else:
            config = na_config
            n_na = config.n_na
        env = env_override if env_override is not None else create_baseline_environment(config)
        
        # Create strategy instance
        strategy_map = {
            'random': RandomSelectionStrategy(),
            'reputation': ReputationBasedStrategy(),
            'balanced': BalancedReputationStrategy(),
            'hunger': HungerBasedStrategy(),
            'weighted': WeightedScoreStrategy(),
            'round_robin': RoundRobinStrategy(seed=42),
            'ddqn': DDQNModelStrategy(model_path=ddqn_model_path) if ddqn_model_path else DDQNModelStrategy(),
            'adaptive': AdaptiveStrategy(),
            'epsilon_greedy': EpsilonGreedyStrategy(),
            'multi_criteria': MultiCriteriaStrategy(),
            'psg': PartialSemiGreedyStrategy(),
            'ga': GeneticAlgorithmStrategy()
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(f"Unsupported strategy: {strategy_name}. Supported: {list(strategy_map.keys())}")
            
        strategy = strategy_map[strategy_name]
        
        # print("Test configuration:")
        # print(f"  Strategy: {strategy.name}")
        # print(f"  NA count: {config.n_na}")
        # print(f"  Total time points: {n_time_points}")
        # print(f"  Selected NAs per round: {selected_na_count}")
        # print(f"  Execution cycles per selection: {execution_cycles}")
        # print(f"  Selection rounds: {n_time_points // execution_cycles}")
        # print()
        
        # Initialize record
        history = {
            'selection_rounds': [],
            'time_points': [],
            'na_reputations': [],
            'na_success_rates': [],
            'na_weighted_success_rates': [],
            'na_latencies': [],
            'selected_nas_history': [],
            'round_rewards': [],
            'round_success_rates': [],
            'cycle_details': []
        }
        
        # Record initial state
        state = env.reset()
        initial_reputation_sum = np.sum(env.na_reputations)
        # print(f"Initial reputation sum: {initial_reputation_sum:.2f}")
        
        # Compute total selection rounds
        total_rounds = n_time_points // execution_cycles
        remaining_cycles = n_time_points % execution_cycles
        
        current_time_point = 0
        
        # Run selection rounds
        for selection_round in range(total_rounds):
            # print(f"\nSelection round {selection_round + 1}/{total_rounds}:")

            warmup_mode = warmup_first_round and selection_round == 0
            if warmup_mode:
                warmup_order = list(range(env.config.n_na))
                selected_nas = warmup_order[:selected_na_count]
                # print("  Warm-up: rotate to cover all NAs (select 5 per cycle)")
                # print(f"  Warm-up first-cycle selection: {selected_nas}")
            else:
                selected_nas = strategy.select(state, env)
                # print(f"  Selected NAs: {selected_nas}")
            
            # Execute the configured number of cycles
            round_rewards = []
            round_success_rates = []
            
            for cycle in range(execution_cycles):
                current_time_point += 1

                if warmup_mode:
                    start_idx = (cycle * selected_na_count) % len(warmup_order)
                    selected_nas = [warmup_order[(start_idx + i) % len(warmup_order)] for i in range(selected_na_count)]
                
                # Execute one step (only selected NAs run transactions)
                next_state, reward, done, info = env.step(selected_nas)
                
                # Record per-cycle metrics
                round_rewards.append(reward)
                round_success_rates.append(info['success_rate'])
                
                # Record detailed time series
                history['time_points'].append(current_time_point)
                history['na_reputations'].append(env.na_reputations.copy())
                
                # Record dynamic success rates (including malicious attacks)
                # current_time_point starts from 1, but OSA uses 0-based steps
                current_success_rates = np.zeros(env.config.n_na)
                for na_id in range(env.config.n_na):
                    # Temporarily set the correct step to query success rate at this time point
                    temp_step = env.current_step
                    env.current_step = current_time_point - 1  # Time point 1 => step 0
                    current_success_rates[na_id] = env._get_malicious_attack_success_rate(na_id)
                    env.current_step = temp_step  # Restore original step
                history['na_success_rates'].append(current_success_rates.copy())
                
                # Record weighted success rates and delay levels
                weighted_success_rates, weighted_latencies = env._calculate_weighted_metrics()
                history['na_weighted_success_rates'].append(weighted_success_rates.copy())
                history['na_latencies'].append(weighted_latencies.copy())
                
                history['cycle_details'].append({
                    'round': selection_round + 1,
                    'cycle': cycle + 1,
                    'time_point': current_time_point,
                    'selected_nas': selected_nas.copy(),
                    'reward': reward,
                    'success_rate': info['success_rate']
                })
                
                state = next_state
            
            # Record round summary
            round_avg_reward = np.mean(round_rewards)
            round_avg_success = np.mean(round_success_rates)
            
            history['selection_rounds'].append(selection_round + 1)
            history['selected_nas_history'].append(selected_nas.copy())
            history['round_rewards'].append(round_avg_reward)
            history['round_success_rates'].append(round_avg_success)
            
            current_reputation_sum = np.sum(env.na_reputations)
            # print(f"  After executing {execution_cycles} cycles:")
            # print(f"  Avg reward: {round_avg_reward:.1f}")
            # print(f"  Avg success rate: {round_avg_success:.1%}")
            # print(f"  Current reputation sum: {current_reputation_sum:.2f}")
        
        # Handle remaining cycles
        if remaining_cycles > 0:
            # print(f"\nFinal {remaining_cycles} cycles:")
            selected_nas = strategy.select(state, env)
            # print(f"  Selected NAs: {selected_nas}")
            
            round_rewards = []
            round_success_rates = []
            
            for cycle in range(remaining_cycles):
                current_time_point += 1
                
                next_state, reward, done, info = env.step(selected_nas)
                
                round_rewards.append(reward)
                round_success_rates.append(info['success_rate'])
                
                # Record detailed time series
                history['time_points'].append(current_time_point)
                history['na_reputations'].append(env.na_reputations.copy())
                
                # Record dynamic success rates (including malicious attacks)
                # current_time_point starts from 1, but OSA uses 0-based steps
                current_success_rates = np.zeros(env.config.n_na)
                for na_id in range(env.config.n_na):
                    # Temporarily set the correct step to query success rate at this time point
                    temp_step = env.current_step
                    env.current_step = current_time_point - 1  # Time point 1 => step 0
                    current_success_rates[na_id] = env._get_malicious_attack_success_rate(na_id)
                    env.current_step = temp_step  # Restore original step
                history['na_success_rates'].append(current_success_rates.copy())
                
                weighted_success_rates, weighted_latencies = env._calculate_weighted_metrics()
                history['na_weighted_success_rates'].append(weighted_success_rates.copy())
                history['na_latencies'].append(weighted_latencies.copy())
                
                history['cycle_details'].append({
                    'round': total_rounds + 1,
                    'cycle': cycle + 1,
                    'time_point': current_time_point,
                    'selected_nas': selected_nas.copy(),
                    'reward': reward,
                    'success_rate': info['success_rate']
                })
                
                state = next_state
            
            # Record the final round
            round_avg_reward = np.mean(round_rewards)
            round_avg_success = np.mean(round_success_rates)
            
            history['selection_rounds'].append(total_rounds + 1)
            history['selected_nas_history'].append(selected_nas.copy())
            history['round_rewards'].append(round_avg_reward)
            history['round_success_rates'].append(round_avg_success)
            
            # print(f"  Avg reward: {round_avg_reward:.1f}")
            # print(f"  Avg success rate: {round_avg_success:.1%}")
        
        # Final statistics (if warm-up is enabled, start comparison after warm-up ends)
        final_reputation_sum = np.sum(env.na_reputations)

        if warmup_first_round and history['cycle_details']:
            warmup_end_time_point = 0
            for cd in history['cycle_details']:
                if cd.get('round') == 1:
                    warmup_end_time_point = max(warmup_end_time_point, int(cd.get('time_point', 0)))

            if warmup_end_time_point > 0 and warmup_end_time_point <= len(history['na_reputations']):
                initial_reputation_sum = float(np.sum(history['na_reputations'][warmup_end_time_point - 1]))

        reputation_change = final_reputation_sum - initial_reputation_sum
        
        # print()
        if save_visuals:
            print("Test summary:")
            print("-" * 40)
            print(f"Initial reputation sum: {initial_reputation_sum:.2f}")
            print(f"Final reputation sum: {final_reputation_sum:.2f}")
            print(f"Reputation change: {reputation_change:+.2f}")
            print(f"Change rate: {(reputation_change/initial_reputation_sum)*100:+.2f}%")
            print(f"Total selection rounds: {len(history['selection_rounds'])}")
            print(f"Average round reward: {np.mean(history['round_rewards']):.2f}")
            print(f"Average round success rate: {np.mean(history['round_success_rates']):.1%}")
        
        result_strategy_name = save_name or strategy_name

        # Save data and generate figures
        self._save_strategy_results(result_strategy_name, history, env)
        if save_visuals:
            self._plot_strategy_performance(result_strategy_name, history)
        
        return history, reputation_change

    def _create_ddqn_ablation_instant_state_env(self, na_config: NAConfig) -> NAEnvironment:
        env = create_baseline_environment(na_config)

        def instant_weighted_metrics():
            n = int(env.config.n_na)
            succ = np.array([float(env._get_malicious_attack_success_rate(i)) for i in range(n)], dtype=float)
            delay = np.array([float(env._get_current_delay_level(i)) for i in range(n)], dtype=float)
            return succ, delay

        def instant_state() -> np.ndarray:
            succ, delay = instant_weighted_metrics()
            denom = float(env.config.reputation_max - env.config.reputation_min)
            denom = denom if denom != 0.0 else 1.0
            norm_rep = (env.na_reputations - float(env.config.reputation_min)) / denom
            return np.column_stack([succ, delay, norm_rep, env.na_hunger])

        env._calculate_weighted_metrics = instant_weighted_metrics
        env.get_state = instant_state
        env._pack_all_na_windows = lambda: None
        return env

    def _run_ddqn_ablation_instant_state(self, n_time_points: int, na_config: NAConfig, ddqn_model_path: str,
                                        warmup_first_round: bool = True,
                                        save_name: Optional[str] = None):
        env = self._create_ddqn_ablation_instant_state_env(na_config)
        return self.run_strategy_test(
            strategy_name='ddqn',
            n_time_points=int(n_time_points),
            n_na=int(na_config.n_na),
            na_config=na_config,
            save_visuals=False,
            warmup_first_round=bool(warmup_first_round),
            ddqn_model_path=str(ddqn_model_path),
            save_name=save_name,
            env_override=env
        )
    
    def _save_strategy_results(self, strategy_name, history, env):
        """Save strategy test results."""
        results_dir = self.results_dir / 'strategy_results'
        results_dir.mkdir(exist_ok=True)
        
        # Recursively convert NumPy types to Python native types
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to JSON-serializable Python types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Save history data
        data = {
            'strategy_name': strategy_name,
            'selection_rounds': convert_numpy_types(history['selection_rounds']),
            'time_points': convert_numpy_types(history['time_points']),
            'round_rewards': convert_numpy_types(history['round_rewards']),
            'round_success_rates': convert_numpy_types(history['round_success_rates']),
            'selected_nas_history': convert_numpy_types(history['selected_nas_history']),
            'cycle_details': convert_numpy_types(history['cycle_details']),
            'na_success_rates_history': convert_numpy_types(history['na_success_rates']),
            'na_weighted_success_rates_history': convert_numpy_types(history['na_weighted_success_rates']),
            'na_reputations_history': convert_numpy_types(history['na_reputations']),
            'na_latencies_history': convert_numpy_types(history['na_latencies']),
            'final_na_reputations': convert_numpy_types(env.na_reputations),
            'final_na_success_rates': convert_numpy_types(env.na_success_rates)
        }
        
        with open(results_dir / f'{strategy_name}_results.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {results_dir}")
    
    def _plot_strategy_performance(self, strategy_name, history):
        """Plot strategy performance figures (removed; handled by external scripts)."""
        pass
    
    def _plot_na_with_selection_style(self, ax, time_points, data, selection_status, execution_cycles, label, color):
        """Plot NA curves with selection style (removed)."""
        pass
        
    def _plot_performance_summary(self, strategy_name, history):
        """Plot performance summary table (removed; handled by external scripts)."""
        pass
        
    def _compute_malicious_order_by_mode(self, env: NAEnvironment, scenario_seed: int, mode: str) -> List[int]:
        if mode == 'random':
            rng = np.random.RandomState(scenario_seed)
            return [int(x) for x in rng.permutation(env.config.n_na).tolist()]
        return self._compute_malicious_order(env)

    def run_all_strategies_comparison_ci(self, n_time_points=None, n_na=None, seed=42, repeats: int = 10):
        print("Starting full-strategy repeated comparison (CI)")
        print("=" * 60)

        if n_time_points is None:
            n_time_points = self.config.training.N_TIME_POINTS
        if n_na is None:
            n_na = self.config.training.N_NA

        all_strategies = ['random', 'weighted', 'round_robin',
                          'reputation', 'psg', 'ga', 'ddqn']

        strategy_results = {
            strategy_name: {
                'metrics': [],
                'reputation_changes': [],
                'metrics_std': [],
                'metrics_ci95': []
            }
            for strategy_name in all_strategies
        }

        scenario_records = []

        malicious_counts = []
        for ratio in range(0, 55, 5):
            count = int(n_na * ratio / 100)
            malicious_counts.append(count)
        malicious_counts = sorted(list(set(malicious_counts)))

        base_env_params = self.config.get_environment_params()
        base_env_params['n_na'] = n_na

        numeric_keys = [
            'malicious_na_selection_count',
            'malicious_na_selection_rate',
            'avg_delay_level',
            'reputation_change',
            'avg_success_rate',
            'total_selections',
            'transaction_count'
        ]

        for malicious_count in malicious_counts:
            print(f"\nMalicious NA count: {malicious_count}")
            print("-" * 40)

            scenario_repeat_records = []

            per_strategy_repeat_metrics = {name: [] for name in all_strategies}

            for repeat_id in range(int(repeats)):
                scenario_seed = int(seed + malicious_count * 9973 + repeat_id * 1000003)

                set_all_seeds(scenario_seed)
                probe_params = base_env_params.copy()
                probe_params['malicious_attack_enabled'] = False
                probe_params['malicious_attack_indices'] = []
                probe_config = NAConfig(**probe_params)
                probe_env = create_baseline_environment(probe_config)

                attack_order = self._compute_malicious_order_by_mode(probe_env, scenario_seed, 'random')
                malicious_indices = attack_order[:malicious_count]

                scenario_repeat_records.append({
                    'repeat_id': repeat_id,
                    'scenario_seed': scenario_seed,
                    'malicious_indices': malicious_indices,
                    'attack_order': attack_order
                })

                for strategy_name in all_strategies:
                    print(f"\r  Testing strategy: {strategy_name} | repeat {repeat_id + 1}/{repeats}", end='')

                    set_all_seeds(scenario_seed)
                    scenario_params = base_env_params.copy()
                    scenario_params['malicious_attack_enabled'] = malicious_count > 0
                    scenario_params['malicious_attack_indices'] = malicious_indices.copy()

                    na_config = NAConfig(**scenario_params)

                    try:
                        history, _ = self.run_strategy_test(
                            strategy_name=strategy_name,
                            n_time_points=n_time_points,
                            n_na=n_na,
                            na_config=na_config,
                            save_visuals=False,
                            warmup_first_round=True
                        )

                        metrics = self._calculate_strategy_metrics(
                            history,
                            strategy_name,
                            malicious_indices,
                            scenario_seed,
                            scenario_params
                        )
                        per_strategy_repeat_metrics[strategy_name].append(metrics)

                    except Exception as e:
                        per_strategy_repeat_metrics[strategy_name].append(None)

            print()

            scenario_records.append({
                'malicious_count': malicious_count,
                'repeats': scenario_repeat_records
            })

            for strategy_name in all_strategies:
                valid_metrics = [m for m in per_strategy_repeat_metrics[strategy_name] if m is not None]
                if not valid_metrics:
                    strategy_results[strategy_name]['metrics'].append(None)
                    strategy_results[strategy_name]['reputation_changes'].append(None)
                    strategy_results[strategy_name]['metrics_std'].append(None)
                    strategy_results[strategy_name]['metrics_ci95'].append(None)
                    continue

                n_valid = len(valid_metrics)

                mean_metrics = {'strategy_name': strategy_name}
                std_metrics = {'strategy_name': strategy_name}
                ci95_metrics = {'strategy_name': strategy_name}

                for key in numeric_keys:
                    values = np.array([float(m[key]) for m in valid_metrics], dtype=float)
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values, ddof=1)) if n_valid > 1 else 0.0
                    ci_val = float(1.96 * std_val / np.sqrt(n_valid)) if n_valid > 1 else 0.0

                    if key in ('total_selections', 'transaction_count'):
                        mean_metrics[key] = int(round(mean_val))
                    else:
                        mean_metrics[key] = mean_val

                    std_metrics[key] = std_val
                    ci95_metrics[key] = ci_val

                strategy_results[strategy_name]['metrics'].append(mean_metrics)
                strategy_results[strategy_name]['reputation_changes'].append(mean_metrics['reputation_change'])
                strategy_results[strategy_name]['metrics_std'].append(std_metrics)
                strategy_results[strategy_name]['metrics_ci95'].append(ci95_metrics)

        self._generate_comparison_report(
            strategy_results,
            malicious_counts,
            scenario_records,
            n_time_points,
            n_na,
            seed
        )

        return strategy_results

    def run_all_strategies_comparison(self, n_time_points=None, n_na=None, seed=42, malicious_selection_mode: str = 'reputation'):
        """
        Run all strategy comparison tests
        
        Args:
            n_time_points: Number of time points
            n_na: Number of NAs
            seed: Random seed
        """
        print(" Start full strategy comparison test")
        print("=" * 60)
        
        # Load test parameters
        if n_time_points is None:
            n_time_points = self.config.training.N_TIME_POINTS
        if n_na is None:
            n_na = self.config.training.N_NA

        # Strategies used in the global comparison
        all_strategies = ['random', 'weighted', 'round_robin',
                          'reputation', 'psg', 'ga', 'ddqn']

        strategy_results = {
            strategy_name: {
                'metrics': [],
                'reputation_changes': []
            }
            for strategy_name in all_strategies
        }

        scenario_records = []
        
        # Use a 5% step for malicious ratio, from 0% to 50%
        malicious_counts = []
        for ratio in range(0, 55, 5):  # 0, 5, 10, ..., 50
            count = int(n_na * ratio / 100)
            malicious_counts.append(count)
        # Deduplicate and sort
        malicious_counts = sorted(list(set(malicious_counts)))
        
        base_env_params = self.config.get_environment_params()
        base_env_params['n_na'] = n_na

        for malicious_count in malicious_counts:
            print(f"\n Malicious NA count: {malicious_count}")
            print("-" * 40)

            scenario_seed = seed + malicious_count * 9973

            # Build a probe environment to determine the malicious-NA activation order
            set_all_seeds(scenario_seed)
            probe_params = base_env_params.copy()
            probe_params['malicious_attack_enabled'] = False
            probe_params['malicious_attack_indices'] = []
            probe_config = NAConfig(**probe_params)
            probe_env = create_baseline_environment(probe_config)
            attack_order = self._compute_malicious_order_by_mode(probe_env, scenario_seed, malicious_selection_mode)
            malicious_indices = attack_order[:malicious_count]

            scenario_records.append({
                'malicious_count': malicious_count,
                'malicious_indices': malicious_indices,
                'attack_order': attack_order
            })

            for strategy_name in all_strategies:
                print(f"\n Testing strategy: {strategy_name} (malicious NAs: {malicious_indices})")
                print("~" * 40)

                set_all_seeds(scenario_seed)
                scenario_params = base_env_params.copy()
                scenario_params['malicious_attack_enabled'] = malicious_count > 0
                scenario_params['malicious_attack_indices'] = malicious_indices.copy()

                na_config = NAConfig(**scenario_params)

                try:
                    history, reputation_change = self.run_strategy_test(
                        strategy_name=strategy_name,
                        n_time_points=n_time_points,
                        n_na=n_na,
                        na_config=na_config,
                        save_visuals=False,
                        warmup_first_round=True
                    )

                    metrics = self._calculate_strategy_metrics(
                        history,
                        strategy_name,
                        malicious_indices,
                        scenario_seed,
                        scenario_params
                    )
                    strategy_results[strategy_name]['metrics'].append(metrics)
                    strategy_results[strategy_name]['reputation_changes'].append(reputation_change)

                    print(f" {strategy_name} test completed (malicious NA count: {malicious_count})")

                except Exception as e:
                    print(f" {strategy_name} test failed: {e}")
                    strategy_results[strategy_name]['metrics'].append(None)
                    strategy_results[strategy_name]['reputation_changes'].append(None)

        # Generate comparison report
        self._generate_comparison_report(
            strategy_results,
            malicious_counts,
            scenario_records,
            n_time_points,
            n_na,
            seed
        )

        return {
            'strategies': strategy_results,
            'malicious_counts': malicious_counts,
            'scenario_records': scenario_records
        }

    def run_drl_cold_warm_comparison(self, n_time_points=None, n_na=None, seed=42, malicious_selection_mode: str = 'reputation'):
        """
        Compare DRL-DPKI cold start vs warm start (first-round warm-up) across four metrics under different malicious ratios.
        """
        print(" Start DRL-DPKI cold/warm start comparison test")
        print("=" * 60)

        if n_time_points is None:
            n_time_points = self.config.training.N_TIME_POINTS
        if n_na is None:
            n_na = self.config.training.N_NA

        comparison_dir = Path('/mnt/data/wy2024/baseline_methods/strategy_comparison')
        comparison_dir.mkdir(exist_ok=True)

        # Use a 5% step for malicious ratio, from 0% to 50%
        malicious_counts = []
        for ratio in range(0, 55, 5):  # 0, 5, 10, ..., 50
            count = int(n_na * ratio / 100)
            malicious_counts.append(count)
        # Deduplicate and sort
        malicious_counts = sorted(list(set(malicious_counts)))

        base_env_params = self.config.get_environment_params()
        base_env_params['n_na'] = n_na

        results = {
            'cold': {'metrics': [], 'label': 'Cold Start'},
            'warm': {'metrics': [], 'label': 'Warm Start'}
        }

        for malicious_count in malicious_counts:
            print(f"\n Malicious NA count: {malicious_count}")
            print("-" * 40)

            scenario_seed = seed + malicious_count * 9973

            # Compute malicious index order
            set_all_seeds(scenario_seed)
            probe_params = base_env_params.copy()
            probe_params['malicious_attack_enabled'] = False
            probe_params['malicious_attack_indices'] = []
            probe_config = NAConfig(**probe_params)
            probe_env = create_baseline_environment(probe_config)
            attack_order = self._compute_malicious_order_by_mode(probe_env, scenario_seed, malicious_selection_mode)
            malicious_indices = attack_order[:malicious_count]

            scenario_params = base_env_params.copy()
            scenario_params['malicious_attack_enabled'] = malicious_count > 0
            scenario_params['malicious_attack_indices'] = malicious_indices.copy()
            na_config = NAConfig(**scenario_params)

            # Cold start
            try:
                set_all_seeds(scenario_seed)
                history_cold, rep_change_cold = self.run_strategy_test(
                    strategy_name='ddqn',
                    n_time_points=n_time_points,
                    n_na=n_na,
                    na_config=na_config,
                    save_visuals=False,
                    warmup_first_round=False
                )
                metrics_cold = self._calculate_strategy_metrics(
                    history_cold, 'ddqn', malicious_indices, scenario_seed, scenario_params
                )
                results['cold']['metrics'].append(metrics_cold)
            except Exception as e:
                print(f" Cold start test failed: {e}")
                results['cold']['metrics'].append(None)

            # Warm start (first-round warm-up)
            try:
                set_all_seeds(scenario_seed)
                history_warm, rep_change_warm = self.run_strategy_test(
                    strategy_name='ddqn',
                    n_time_points=n_time_points,
                    n_na=n_na,
                    na_config=na_config,
                    save_visuals=False,
                    warmup_first_round=True
                )
                metrics_warm = self._calculate_strategy_metrics(
                    history_warm, 'ddqn', malicious_indices, scenario_seed, scenario_params
                )
                results['warm']['metrics'].append(metrics_warm)
            except Exception as e:
                print(f" Warm start test failed: {e}")
                results['warm']['metrics'].append(None)

        self._save_drl_cold_warm_results(results, malicious_counts, comparison_dir, n_na, seed)

        return {
            'results': results,
            'malicious_counts': malicious_counts,
            'save_dir': comparison_dir
        }

    def run_drl_slidewindow_ablation_comparison(self, n_time_points=None, n_na=None, seed=42, repeats: int = 10, malicious_selection_mode: str = 'random'):
        print(" Start sliding-window ablation comparison (DRL-DPKI)")
        print("=" * 60)

        if n_time_points is None:
            n_time_points = self.config.training.N_TIME_POINTS
        if n_na is None:
            n_na = self.config.training.N_NA

        full_model_path = Path('/mnt/data/wy2024/models/policy_net_state_dict.pth')
        ablation_model_path = Path('/mnt/data/wy2024/models_ablation_no_slidewindow/policy_net_state_dict.pth')

        if not full_model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {full_model_path}")
        if not ablation_model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {ablation_model_path}")

        base_env_params = self.config.get_environment_params()
        base_env_params['n_na'] = n_na

        malicious_counts = []
        for ratio in range(0, 55, 5):
            count = int(n_na * ratio / 100)
            malicious_counts.append(count)
        malicious_counts = sorted(list(set(malicious_counts)))

        strategies = {
            'drl_full': {
                'model_path': str(full_model_path),
                'window_queue_size': None
            },
            'drl_ablation_no_slidewindow': {
                'model_path': str(ablation_model_path),
                'window_queue_size': 0
            }
        }

        numeric_keys = [
            'malicious_na_selection_count',
            'malicious_na_selection_rate',
            'avg_delay_level',
            'reputation_change',
            'avg_success_rate',
            'total_selections',
            'transaction_count'
        ]

        strategy_results = {
            name: {
                'metrics': [],
                'reputation_changes': [],
                'metrics_std': [],
                'metrics_ci95': []
            }
            for name in strategies.keys()
        }

        scenario_records = []

        for malicious_count in malicious_counts:
            print(f"\n Malicious NA count: {malicious_count}")
            print("-" * 40)

            scenario_repeat_records = []
            per_strategy_repeat_metrics = {name: [] for name in strategies.keys()}

            for repeat_id in range(int(repeats)):
                scenario_seed = int(seed + malicious_count * 9973 + repeat_id * 1000003)

                set_all_seeds(scenario_seed)
                probe_params = base_env_params.copy()
                probe_params['malicious_attack_enabled'] = False
                probe_params['malicious_attack_indices'] = []
                probe_config = NAConfig(**probe_params)
                probe_env = create_baseline_environment(probe_config)
                attack_order = self._compute_malicious_order_by_mode(probe_env, scenario_seed, malicious_selection_mode)
                malicious_indices = attack_order[:malicious_count]

                scenario_repeat_records.append({
                    'repeat_id': repeat_id,
                    'scenario_seed': scenario_seed,
                    'malicious_indices': malicious_indices,
                    'attack_order': attack_order
                })

                for run_name, run_cfg in strategies.items():
                    print(f"\r  Testing model: {run_name} | repeat {repeat_id + 1}/{int(repeats)}", end='')

                    set_all_seeds(scenario_seed)

                    scenario_params = base_env_params.copy()
                    scenario_params['malicious_attack_enabled'] = malicious_count > 0
                    scenario_params['malicious_attack_indices'] = malicious_indices.copy()

                    window_queue_size = run_cfg.get('window_queue_size')
                    if window_queue_size is not None:
                        scenario_params['window_queue_size'] = int(window_queue_size)

                    na_config = NAConfig(**scenario_params)

                    try:
                        if run_name == 'drl_ablation_no_slidewindow':
                            history, _ = self._run_ddqn_ablation_instant_state(
                                n_time_points=int(n_time_points),
                                na_config=na_config,
                                ddqn_model_path=run_cfg['model_path'],
                                warmup_first_round=True,
                                save_name=f"{run_name}_m{malicious_count}_r{repeat_id}"
                            )
                        else:
                            history, _ = self.run_strategy_test(
                                strategy_name='ddqn',
                                n_time_points=n_time_points,
                                n_na=n_na,
                                na_config=na_config,
                                save_visuals=False,
                                warmup_first_round=True,
                                ddqn_model_path=run_cfg['model_path'],
                                save_name=f"{run_name}_m{malicious_count}_r{repeat_id}"
                            )

                        metrics = self._calculate_strategy_metrics(
                            history,
                            run_name,
                            malicious_indices,
                            scenario_seed,
                            scenario_params
                        )
                        per_strategy_repeat_metrics[run_name].append(metrics)

                    except Exception as e:
                        print(f"\n {run_name} test failed (repeat {repeat_id}): {e}")
                        per_strategy_repeat_metrics[run_name].append(None)

            print()

            scenario_records.append({
                'malicious_count': malicious_count,
                'repeats': scenario_repeat_records
            })

            for run_name in strategies.keys():
                valid_metrics = [m for m in per_strategy_repeat_metrics[run_name] if m is not None]
                if not valid_metrics:
                    strategy_results[run_name]['metrics'].append(None)
                    strategy_results[run_name]['reputation_changes'].append(None)
                    strategy_results[run_name]['metrics_std'].append(None)
                    strategy_results[run_name]['metrics_ci95'].append(None)
                    continue

                n_valid = len(valid_metrics)

                mean_metrics = {'strategy_name': run_name}
                std_metrics = {'strategy_name': run_name}
                ci95_metrics = {'strategy_name': run_name}

                for key in numeric_keys:
                    values = np.array([float(m[key]) for m in valid_metrics], dtype=float)
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values, ddof=1)) if n_valid > 1 else 0.0
                    ci_val = float(1.96 * std_val / np.sqrt(n_valid)) if n_valid > 1 else 0.0

                    if key in ('total_selections', 'transaction_count'):
                        mean_metrics[key] = int(round(mean_val))
                    else:
                        mean_metrics[key] = mean_val

                    std_metrics[key] = std_val
                    ci95_metrics[key] = ci_val

                strategy_results[run_name]['metrics'].append(mean_metrics)
                strategy_results[run_name]['reputation_changes'].append(mean_metrics['reputation_change'])
                strategy_results[run_name]['metrics_std'].append(std_metrics)
                strategy_results[run_name]['metrics_ci95'].append(ci95_metrics)

        self._generate_comparison_report(
            strategy_results,
            malicious_counts,
            scenario_records,
            n_time_points,
            n_na,
            seed
        )

        return {
            'strategies': strategy_results,
            'malicious_counts': malicious_counts,
            'scenario_records': scenario_records
        }

    def run_ca_scale_sensitivity(self, n_time_points=None, scales=None, seed=42, malicious_selection_mode: str = 'reputation', warmup_first_round: bool = True):
        startup_label = "Warm Start" if warmup_first_round else "Cold Start"
        print(f" Start CA-scale sensitivity experiment (DRL-DPKI, {startup_label})")
        print("=" * 60)

        if n_time_points is None:
            n_time_points = self.config.training.N_TIME_POINTS
        if scales is None:
            scales = [400, 1000, 2500, 5200, 6000]

        malicious_ratios = list(range(0, 55, 5))
        results = {}

        for n_na in scales:
            print(f"\n=== CA scale: {n_na} ===")
            base_env_params = self.config.get_environment_params()
            base_env_params['n_na'] = n_na

            malicious_counts = sorted(list(set(int(n_na * ratio / 100) for ratio in malicious_ratios)))

            scale_records = []
            scale_metrics = []

            for malicious_count in malicious_counts:
                print(f"\n Malicious NA count: {malicious_count}")
                print("-" * 40)

                scenario_seed = seed + n_na * 100000 + malicious_count * 9973

                set_all_seeds(scenario_seed)
                probe_params = base_env_params.copy()
                probe_params['malicious_attack_enabled'] = False
                probe_params['malicious_attack_indices'] = []
                probe_config = NAConfig(**probe_params)
                probe_env = create_baseline_environment(probe_config)
                attack_order = self._compute_malicious_order_by_mode(probe_env, scenario_seed, malicious_selection_mode)
                malicious_indices = attack_order[:malicious_count]

                scale_records.append({
                    'malicious_count': malicious_count,
                    'malicious_indices': malicious_indices,
                    'attack_order': attack_order
                })

                set_all_seeds(scenario_seed)
                scenario_params = base_env_params.copy()
                scenario_params['malicious_attack_enabled'] = malicious_count > 0
                scenario_params['malicious_attack_indices'] = malicious_indices.copy()
                na_config = NAConfig(**scenario_params)

                try:
                    history, _ = self.run_strategy_test(
                        strategy_name='ddqn',
                        n_time_points=n_time_points,
                        n_na=n_na,
                        na_config=na_config,
                        save_visuals=False,
                        warmup_first_round=warmup_first_round
                    )

                    metrics = self._calculate_strategy_metrics(
                        history,
                        'ddqn',
                        malicious_indices,
                        scenario_seed,
                        scenario_params
                    )
                    scale_metrics.append(metrics)
                    print(f" ddqn test completed (malicious NA count: {malicious_count})")
                except Exception as e:
                    print(f" ddqn test failed: {e}")
                    scale_metrics.append(None)

            results[str(n_na)] = {
                'n_na': n_na,
                'malicious_counts': malicious_counts,
                'malicious_ratios': malicious_ratios,
                'strategy': 'ddqn',
                'scenarios': scale_records,
                'metrics': scale_metrics
            }

        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        save_data = {
            'experiment_type': 'ca_scale_sensitivity',
            'test_info': {
                'test_date': datetime.now().isoformat(),
                'n_time_points': n_time_points,
                'seed': seed,
                'warmup_first_round': warmup_first_round,
                'malicious_selection_mode': malicious_selection_mode,
                'scales': scales,
                'malicious_ratios': malicious_ratios
            },
            'results': convert_numpy_types(results)
        }

        self.results_dir.mkdir(parents=True, exist_ok=True)
        json_file = self.results_dir / 'ca_scale_sensitivity_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n CA-scale sensitivity experiment completed!")
        print(f" Results saved to: {json_file}")

        return {
            'result_file': json_file,
            'results': results
        }
    
    def _save_drl_cold_warm_results(self, results, malicious_counts, comparison_dir, n_na, seed):
        """Save DRL cold/warm comparison results"""
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        save_data = {
            'test_info': {
                'test_date': datetime.now().isoformat(),
                'n_na': n_na,
                'seed': seed
            },
            'malicious_counts': malicious_counts,
            'results': convert_numpy_types(results)
        }
        
        json_file = comparison_dir / 'drl_cold_warm_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
            
        print(f" DRL cold/warm comparison results saved: {json_file}")
    
    def _calculate_strategy_metrics(self, history, strategy_name, malicious_indices,
                                    scenario_seed, scenario_params):
        """
        Compute key metrics for a strategy
        
        Args:
            history: Strategy test history data
            strategy_name: Strategy name
            
        Returns:
            dict: A dict containing key metrics
        """
        # Count malicious-NA selections
        set_all_seeds(scenario_seed)
        env_config = scenario_params.copy()
        env_config['max_steps'] = len(history['cycle_details'])
        env_config['malicious_attack_enabled'] = bool(malicious_indices)
        env_config['malicious_attack_indices'] = malicious_indices.copy()
        env = create_baseline_environment(NAConfig(**env_config))
        malicious_set = set(env.malicious_nas)

        malicious_na_selection_count = 0
        total_selections = 0
        
        for cycle_detail in history['cycle_details']:
            # Skip the first decision because the strategy has no prior behavior data to assess NAs
            if cycle_detail['round'] == 1:
                continue
                
            selected_nas = cycle_detail['selected_nas']
            for na_id in selected_nas:
                if na_id in malicious_set:
                    malicious_na_selection_count += 1
                total_selections += 1
        
        warmup_end_time_point = 0
        for cd in history['cycle_details']:
            if cd.get('round') == 1:
                warmup_end_time_point = max(warmup_end_time_point, int(cd.get('time_point', 0)))

        # Compute average delay level (excluding warm-up round: round==1)
        total_delay_level = 0.0
        transaction_count = 0

        for cycle_detail in history['cycle_details']:
            if cycle_detail.get('round') == 1:
                continue

            selected_nas = cycle_detail['selected_nas']
            time_point = cycle_detail['time_point'] - 1

            if 0 <= time_point < len(history['na_latencies']):
                na_latencies_at_time = history['na_latencies'][time_point]
                for na_id in selected_nas:
                    total_delay_level += float(na_latencies_at_time[na_id])
                    transaction_count += 1

        avg_delay_level = total_delay_level / transaction_count if transaction_count > 0 else 0.0

        # Compute overall reputation change (using the end of warm-up as the initial state)
        if history['na_reputations']:
            if warmup_end_time_point > 0 and warmup_end_time_point <= len(history['na_reputations']):
                initial_reputation_sum = float(np.sum(history['na_reputations'][warmup_end_time_point - 1]))
            else:
                initial_reputation_sum = float(np.sum(history['na_reputations'][0]))
            final_reputation_sum = float(np.sum(history['na_reputations'][-1]))
        else:
            initial_reputation_sum = 0.0
            final_reputation_sum = 0.0

        reputation_change = final_reputation_sum - initial_reputation_sum

        # Compute average success rate (excluding warm-up round: the first entry in round_success_rates)
        if len(history['round_success_rates']) > 1:
            avg_success_rate = float(np.mean(history['round_success_rates'][1:]))
        else:
            avg_success_rate = float(np.mean(history['round_success_rates'])) if history['round_success_rates'] else 0.0
        
        return {
            'strategy_name': strategy_name,
            'malicious_na_selection_count': malicious_na_selection_count,
            'malicious_na_selection_rate': malicious_na_selection_count / total_selections if total_selections > 0 else 0,
            'avg_delay_level': avg_delay_level,
            'reputation_change': reputation_change,
            'avg_success_rate': avg_success_rate,
            'total_selections': total_selections,
            'transaction_count': transaction_count
        }

    def _compute_malicious_order(self, env: NAEnvironment) -> List[int]:
        """Generate the malicious-NA activation order based on current reputation distribution"""
        reputations = env.na_reputations
        threshold = env.config.reputation_threshold

        high_indices = [idx for idx, rep in enumerate(reputations) if rep >= threshold]
        low_indices = [idx for idx, rep in enumerate(reputations) if rep < threshold]

        high_sorted = sorted(high_indices, key=lambda idx: reputations[idx], reverse=True)
        low_sorted = sorted(low_indices, key=lambda idx: reputations[idx], reverse=True)

        attack_order: List[int] = []
        max_len = max(len(high_sorted), len(low_sorted))
        for i in range(max_len):
            if i < len(high_sorted):
                attack_order.append(int(high_sorted[i]))
            if i < len(low_sorted):
                attack_order.append(int(low_sorted[i]))

        remaining = [idx for idx in range(env.config.n_na) if idx not in attack_order]
        attack_order.extend(remaining)

        return attack_order
    
    def _generate_comparison_report(self, strategy_results, malicious_counts, scenario_records,
                                    n_time_points, n_na, seed):
        """
        Generate a strategy comparison report and charts
        
        Args:
            comparison_results: Test results of all strategies
            n_time_points: Number of time points
            n_na: Number of NAs
            seed: Random seed
        """
        # Create comparison results directory
        comparison_dir = self.results_dir
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n Generating strategy comparison report...")
        print(f" Report output directory: {comparison_dir.absolute()}")
        
        # Filter successful test results
        available_strategies = {
            name: data for name, data in strategy_results.items()
            if any(metric is not None for metric in data['metrics'])
        }
        
        if not available_strategies:
            print(" No successful strategy test results")
            return

        # Build the full comparison table data (one row per malicious count)
        csv_rows = []
        for strategy_name, data in available_strategies.items():
            for malicious_count, metrics in zip(malicious_counts, data['metrics']):
                if metrics is None:
                    continue
                csv_rows.append({
                    'Strategy Name': strategy_name.capitalize(),
                    'Malicious Count': malicious_count,
                    'Malicious NA Selections': metrics['malicious_na_selection_count'],
                    'Malicious NA Rate': metrics['malicious_na_selection_rate'],
                    'Average Delay Level': metrics['avg_delay_level'],
                    'Reputation Change': metrics['reputation_change'],
                    'Average Success Rate': metrics['avg_success_rate']
                })

        csv_df = pd.DataFrame(csv_rows)
        csv_file = comparison_dir / 'strategy_comparison.csv'
        csv_generated = False
        if not csv_df.empty:
            csv_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            csv_generated = True
        else:
            print(" No data available to generate CSV")

        # Comparison charts have been removed; use plot_results.py instead
        # chart_generated = self._plot_comparison_charts(available_strategies, malicious_counts, comparison_dir, n_na)
        print(" Tip: run plot_results.py to generate comparison charts")

        # Save detailed results JSON
        json_file = self._save_comparison_json(strategy_results, malicious_counts, scenario_records,
                                               comparison_dir, n_time_points, n_na, seed)
        
        print(" Strategy comparison report generated!")
        if csv_generated:
            print(f"    Comparison table: {csv_file}")
        
        print(f"    Detailed data: {json_file}")
    

    
    def _save_comparison_json(self, strategy_results, malicious_counts, scenario_records,
                               comparison_dir, n_time_points, n_na, seed):
        """Save detailed comparison results to JSON"""
        
        # Recursively convert numpy types
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Prepare payload
        save_data = {
            'test_info': {
                'test_date': datetime.now().isoformat(),
                'n_time_points': n_time_points,
                'n_na': n_na,
                'seed': seed
            },
            'malicious_counts': malicious_counts,
            'scenarios': scenario_records,
            'strategies': {}
        }
        
        # Save results for each strategy
        for strategy_name, result in strategy_results.items():
            if result is None:
                save_data['strategies'][strategy_name] = None
                continue

            metrics_list = []
            for metrics in result.get('metrics', []):
                if metrics is None:
                    metrics_list.append(None)
                else:
                    metrics_list.append(convert_numpy_types(metrics))

            strategy_payload = {
                'metrics': metrics_list,
                'reputation_changes': convert_numpy_types(result.get('reputation_changes', []))
            }

            if 'metrics_std' in result:
                std_list = []
                for std_metrics in result.get('metrics_std', []):
                    std_list.append(None if std_metrics is None else convert_numpy_types(std_metrics))
                strategy_payload['metrics_std'] = std_list

            if 'metrics_ci95' in result:
                ci_list = []
                for ci_metrics in result.get('metrics_ci95', []):
                    ci_list.append(None if ci_metrics is None else convert_numpy_types(ci_metrics))
                strategy_payload['metrics_ci95'] = ci_list

            save_data['strategies'][strategy_name] = strategy_payload
        
        json_file = comparison_dir / 'comparison_results.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        return json_file
        
    def run_training(self, episodes=None, steps_per_episode=None, **kwargs):
        """
        Run the training process
        
        Args:
            episodes: Number of episodes
            steps_per_episode: Steps per episode
            **kwargs: Other training parameters
        """
        print(" Start training...")
        
        # Load training parameters
        train_params = self.config.get_train_dqn_params()
        if episodes is not None:
            train_params['n_episodes'] = episodes
        if steps_per_episode is not None:
            train_params['steps_per_episode'] = steps_per_episode
        train_params.update(kwargs)
        
        # Check environment and agent
        if not hasattr(self, 'env') or not hasattr(self, 'agent'):
            raise RuntimeError("Initialize the environment and agent first")
            
        # Run training
        if hasattr(self.agent, 'train') and callable(self.agent.train):
            # Use the agent's training method
            training_results = self.agent.train(
                env=self.env,
                **train_params
            )
        else:
            # Use a generic training loop
            training_results = self._generic_training_loop(**train_params)
            
        # Save training history
        self.experiment_data['training_history'] = training_results
        
        print(f" Training completed: {train_params['n_episodes']} episodes")
        return training_results
        
    def _generic_training_loop(self, n_episodes, steps_per_episode, **kwargs):
        """
        Generic training loop
        """
        training_data = []
        
        for episode in range(n_episodes):
            episode_start_time = time.time()
            
            # Reset environment
            if self.env is not None:
                state = self.env.reset()
            else:
                # Simulated state - generate a [n_na, n_features] shaped state
                n_na = self.config.get_environment_params()['n_na']
                n_features = 4  # reputation, success rate, delay level, hunger
                state = np.random.randn(n_na, n_features)
                
            episode_reward = 0
            episode_steps = 0
            
            for step in range(steps_per_episode):
                # Select action
                if hasattr(self.agent, 'select_action'):
                    action = self.agent.select_action(state)
                else:
                    action = np.random.randint(0, self.config.get_environment_params()['n_na'])
                    
                # Execute action
                if self.env is not None:
                    # Convert a single action to list format for NAEnvironment.step()
                    if isinstance(action, (int, np.integer)):
                        action_list = [action]
                    elif isinstance(action, (list, np.ndarray)):
                        action_list = list(action)
                    else:
                        action_list = [action]
                    
                    next_state, reward, done, info = self.env.step(action_list)
                else:
                    # Simulated environment feedback
                    n_na = self.config.get_environment_params()['n_na']
                    n_features = 4
                    next_state = np.random.randn(n_na, n_features)
                    reward = np.random.randn()
                    done = False
                    info = {}
                    
                episode_reward += reward
                episode_steps += 1
                
                # Store experience
                if hasattr(self.agent, 'store'):
                    self.agent.store(state, action, reward, next_state, done)
                    
                # Update agent
                if hasattr(self.agent, 'maybe_update'):
                    self.agent.maybe_update()
                    
                state = next_state
                
                if done:
                    break
                    
            # Record episode data
            episode_data = {
                'episode': episode,
                'reward': episode_reward,
                'steps': episode_steps,
                'avg_step_reward': episode_reward / episode_steps if episode_steps > 0 else 0,
                'duration': time.time() - episode_start_time
            }
            
            training_data.append(episode_data)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean([d['reward'] for d in training_data[-10:]])
                print(f"Episode {episode + 1}/{n_episodes}, average reward: {avg_reward:.2f}")
                
        return training_data
        
    def collect_performance_metrics(self):
        """
        Collect and compute performance metrics
        """
        print(" Collecting performance metrics...")
        
        if not self.experiment_data['training_history']:
            print(" No training history data")
            return {}
            
        df = pd.DataFrame(self.experiment_data['training_history'])
        
        # Basic statistics
        metrics = {
            'total_episodes': len(df),
            'mean_reward': df['reward'].mean(),
            'std_reward': df['reward'].std(),
            'min_reward': df['reward'].min(),
            'max_reward': df['reward'].max(),
            'final_reward': df['reward'].iloc[-1] if len(df) > 0 else 0,
        }
        
        # Success rate (assume reward>0 means success)
        if 'reward' in df.columns:
            metrics['success_rate'] = (df['reward'] > 0).mean()
            
        # Learning curve analysis
        if len(df) >= 20:
            early_rewards = df['reward'].iloc[:len(df)//3].mean()
            late_rewards = df['reward'].iloc[-len(df)//3:].mean()
            metrics['improvement_rate'] = (late_rewards - early_rewards) / abs(early_rewards) if early_rewards != 0 else 0
            
        # Convergence analysis
        if len(df) >= 10:
            recent_std = df['reward'].iloc[-10:].std()
            metrics['convergence_stability'] = 1.0 / (1.0 + recent_std)  # stability metric
            
        self.experiment_data['performance_metrics'] = metrics
        
        print(" Performance metrics collection completed")
        print(f"   Mean reward: {metrics['mean_reward']:.3f}")
        print(f"   Success rate: {metrics.get('success_rate', 0):.3f}")
        
        return metrics
        
    def generate_visualization(self, save_plots=True):
        """
        Generate visualization charts for the training process
        
        Args:
            save_plots: Whether to save plots
        """
        print(" Generating visualization charts...")
        
        if not self.experiment_data['training_history']:
            print(" No training history data, skip visualization")
            return
            
        df = pd.DataFrame(self.experiment_data['training_history'])
        
        # Create summary chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Performance Analysis', fontsize=16)
        
        # 1. Reward curve
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.6, label='Episode Reward')
        if len(df) >= 10:
            # Moving average
            window = min(10, len(df) // 4)
            smoothed = df['reward'].rolling(window=window, center=True).mean()
            axes[0, 0].plot(df['episode'], smoothed, label=f'Moving Average ({window})', linewidth=2)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Average step reward
        if 'avg_step_reward' in df.columns:
            axes[0, 1].plot(df['episode'], df['avg_step_reward'], color='orange')
            axes[0, 1].set_title('Average Step Reward')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Avg Step Reward')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No Step Reward Data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Average Step Reward')
            
        # 3. Reward distribution
        axes[1, 0].hist(df['reward'], bins=20, alpha=0.7, color='green')
        axes[1, 0].axvline(df['reward'].mean(), color='red', linestyle='--', label=f'Mean: {df["reward"].mean():.2f}')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training duration
        if 'duration' in df.columns:
            axes[1, 1].plot(df['episode'], df['duration'], color='purple', alpha=0.7)
            axes[1, 1].set_title('Episode Duration')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Duration (seconds)')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Duration Data', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Episode Duration')
            
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.results_dir / 'training_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f" Chart saved: {plot_path}")
            
        plt.show()
        
        # Generate performance metrics chart
        self._generate_metrics_plot(save_plots)
        
    def _generate_metrics_plot(self, save_plots=True):
        """
        Generate performance metrics chart
        """
        metrics = self.experiment_data.get('performance_metrics', {})
        if not metrics:
            return
            
        # Create chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Select key metrics
        key_metrics = {
            'Mean Reward': metrics.get('mean_reward', 0),
            'Success Rate': metrics.get('success_rate', 0),
            'Improvement Rate': metrics.get('improvement_rate', 0),
            'Stability': metrics.get('convergence_stability', 0)
        }
        
        # Plot bar chart
        names = list(key_metrics.keys())
        values = list(key_metrics.values())
        colors = ['blue', 'green', 'orange', 'purple']
        
        bars = ax.bar(names, values, color=colors, alpha=0.7)
        ax.set_title('Performance Metrics Summary')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
                   
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.results_dir / 'performance_metrics.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f" Metrics chart saved: {plot_path}")
            
        plt.show()
        
    def save_results(self):
        """
        Save experiment results
        """
        print(" Saving experiment results...")
        
        # Save configuration
        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data['config'], f, indent=2, ensure_ascii=False)
            
        # Save training history
        if self.experiment_data['training_history']:
            df = pd.DataFrame(self.experiment_data['training_history'])
            csv_path = self.results_dir / 'training_history.csv'
            df.to_csv(csv_path, index=False)
            print(f" Training history saved: {csv_path}")
            
        # Save performance metrics
        metrics_path = self.results_dir / 'performance_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_data['performance_metrics'], f, indent=2, ensure_ascii=False)
            
        # Save model (if DDQN)
        if hasattr(self, 'agent') and hasattr(self.agent, 'save_model'):
            model_path = self.results_dir / 'model.pth'
            self.agent.save_model(str(model_path))
            print(f" Model saved: {model_path}")
            
        # Save experiment summary
        summary = self._generate_experiment_summary()
        summary_path = self.results_dir / 'experiment_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        print(f" Experiment results saved: {self.save_dir}")
        
    def _generate_experiment_summary(self):
        """
        Generate experiment summary
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        summary = f"""
 Experiment Summary
{'='*50}

 Time:
   Start: {self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}
   End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   Duration: {duration:.2f}s

 Configuration:
   Algorithm: {self.experiment_data['model_info'].get('algorithm', 'Unknown')}
   Environment: {self.experiment_data['environment_info'].get('n_na', 'Unknown')} NAs
   Episodes: {self.experiment_data['performance_metrics'].get('total_episodes', 'Unknown')}

 Metrics:
   Mean reward: {self.experiment_data['performance_metrics'].get('mean_reward', 0):.3f}
   Reward std: {self.experiment_data['performance_metrics'].get('std_reward', 0):.3f}
   Success rate: {self.experiment_data['performance_metrics'].get('success_rate', 0):.3f}
   Improvement rate: {self.experiment_data['performance_metrics'].get('improvement_rate', 0):.3f}

 Saved files:
   - config.json: configuration
   - training_history.csv: training history
   - performance_metrics.json: performance metrics
   - training_analysis.png: training analysis chart
   - performance_metrics.png: performance metrics chart
   - model.pth: trained model (if applicable)

{'='*50}
Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return summary


class RandomAgent:
    """
    Random selection agent - baseline comparison
    """
    
    def __init__(self, n_na):
        self.n_na = n_na
        
    def select_action(self, state):
        return np.random.randint(0, self.n_na)
        
    def store(self, *args):
        pass  # Random agent does not store experience
        
    def maybe_update(self):
        pass  # Random agent does not update
        

def interactive_mode_selection():
    """
    Interactive mode selection
    """
    print(" CA selection strategy test platform")
    print("=" * 50)
    print()
    
    print(" Available test modes:")
    print("   1. Single Strategy     - Detailed evaluation of one strategy")
    print("   2. All Strategies      - Compare key metrics across strategies")
    print("   3. Cold/Warm Compare   - Compare DRL cold start vs warm-up")
    print("   4. Scale Sensitivity   - DRL-DPKI across different CA scales")
    print("   5. Ablation (Window)   - DRL full vs without sliding window")
    print()
    
    while True:
        try:
            choice = input("Select test mode (1-5) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print(" Goodbye!")
                return None
                
            if choice == '1':
                print(" Selected: single strategy test")
                return 'single'
            elif choice == '2':
                print(" Selected: all-strategies comparison")
                return 'compare'
            elif choice == '3':
                print(" Selected: DRL cold/warm comparison")
                return 'compare_drl_cold_warm'
            elif choice == '4':
                print(" Selected: CA-scale sensitivity experiment")
                return 'ca_scale_sensitivity'
            elif choice == '5':
                print(" Selected: sliding-window ablation comparison")
                return 'drl_slidewindow_ablation'
            else:
                print(" Invalid choice. Enter 1-5 or 'q'")
                print()
                
        except KeyboardInterrupt:
            print("\n Goodbye!")
            return None
        except Exception as e:
            print(f" Input error: {e}")
            print()


def interactive_strategy_selection():
    """
    Interactive strategy selection
    """
    print(" NA selection strategy test platform")
    print("=" * 50)
    print()
    
    # Show available strategies
    strategies = {
        '1': 'random',
        '2': 'reputation', 
        '3': 'balanced',
        '4': 'hunger',
        '5': 'weighted',
        '6': 'round_robin',
        '7': 'ddqn',
        '8': 'adaptive',
        '9': 'epsilon_greedy',
        '10': 'multi_criteria',
        '11': 'psg',
        '12': 'ga'
    }
    
    print(" Available NA selection strategies:")
    print("   1. Random Selection      - Random baseline")
    print("   2. Reputation-based      - Select by reputation")
    print("   3. Balanced Selection    - Balanced (3 high + 2 low)")
    print("   4. Hunger-based          - Select by hunger")
    print("   5. Weighted Selection    - Weighted composite score")
    print("   6. Round Robin           - Fixed random order rotation")
    print("   7. DDQN Model            - Deep Q-Network model policy")
    print("   8. Historical Learning   - Multi-factor + historical adaptation")
    print("   9. Epsilon Greedy        - ε-greedy (exploration/exploitation)")
    print("   10. Multi-Criteria       - TOPSIS-based decision policy")
    print("   11. Partial Semi-Greedy  - Semi-greedy (reward threshold=3)")
    print("   12. Genetic Algorithm     - Genetic algorithm selection")
    print()
    
    # User selection
    while True:
        try:
            choice = input("Select strategy (1-12) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print(" Goodbye!")
                return None
                
            if choice in strategies:
                selected_strategy = strategies[choice]
                strategy_names = {
                    'random': 'Random Selection',
                    'reputation': 'Reputation-based', 
                    'balanced': 'Balanced Selection',
                    'hunger': 'Hunger-based',
                    'weighted': 'Weighted Selection',
                    'round_robin': 'Round Robin',
                    'ddqn': 'DDQN Model',
                    'adaptive': 'Historical Learning Strategy',
                    'epsilon_greedy': 'Epsilon Greedy',
                    'multi_criteria': 'Multi-Criteria',
                    'psg': 'Partial Semi-Greedy',
                    'ga': 'Genetic Algorithm'
                }
                
                print(f" Selected strategy: {strategy_names[selected_strategy]}")
                print()
                print(" Default configuration:")
                print("   • NAs: 20")
                print("   • Time points: 150")
                print("   • Random seed: 42")
                print("   • Attack type: OSA")
                print()
                
                confirm = input("Start test with this configuration? (Y/n): ").strip().lower()
                if confirm in ['', 'y', 'yes']:
                    return selected_strategy
                else:
                    print(" Cancelled. Please select again")
                    print()
                    continue
            else:
                print(" Invalid choice. Enter 1-12 or 'q'")
                print()
                
        except KeyboardInterrupt:
            print("\n Goodbye!")
            return None
        except Exception as e:
            print(f" Input error: {e}")
            print()


def parse_arguments():
    """
    Parse command-line arguments (simplified)
    """
    parser = argparse.ArgumentParser(description='NA selection optimization experiment platform')
    
    # Keep some advanced options with reasonable defaults
    parser.add_argument('--strategy', type=str, default=None,
                       choices=['random', 'reputation', 'balanced', 'hunger', 'weighted', 'round_robin', 'ddqn', 'adaptive', 'epsilon_greedy', 'multi_criteria', 'psg', 'ga'],
                       help='Specify a strategy type directly (skip interactive selection)')
    
    parser.add_argument('--compare-all', action='store_true',
                       help='Run all-strategies comparison and generate a report')

    parser.add_argument('--compare-all-ci', action='store_true',
                       help='Run repeated comparison (fixed random malicious order, repeats=10, output mean and 95% CI)')

    parser.add_argument('--compare-drl-cold-warm', action='store_true',
                       help='Run DRL-DPKI cold/warm comparison (generate 4 metric charts)')

    parser.add_argument('--ca-scale-sensitivity', action='store_true',
                       help='Run CA-scale sensitivity (DRL-DPKI only, cold/warm, random/reputation malicious order, scales: 200/400/1000/1600/2500)')
    
    parser.add_argument('--attack_type', type=str, default='OSA',
                       choices=['ME', 'OSA', 'OOA'],
                       help='Attack type (default: OSA)')

    parser.add_argument('--malicious_selection', type=str, default='reputation',
                       choices=['reputation', 'random'],
                       help='Malicious selection mode (default: reputation)')

    parser.add_argument('--ca_scale_startup', type=str, default='warm',
                       choices=['cold', 'warm'],
                       help='CA-scale sensitivity startup (default: warm)')
    
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Results output directory (optional)')
    
    parser.add_argument('--advanced', action='store_true',
                       help='Enable advanced configuration mode')
    
    # Advanced options (only when --advanced is present)
    if '--advanced' in sys.argv:
        parser.add_argument('--nas', type=int, default=10,
                           help='Number of NAs (default: 10)')
        parser.add_argument('--time_points', type=int, default=150,
                           help='Number of time points (default: 150)')
        parser.add_argument('--seed', type=int, default=42,
                           help='Random seed (default: 42)')
    
    # Backward compatibility with older versions
    parser.add_argument('--test_strategy', action='store_true',
                       help='Run strategy test mode (deprecated; now default)')
    
    return parser.parse_args()


def main():
    """
    Main entry (simplified startup)
    """
    print(" Baseline NA selection strategy test platform")
    print("=" * 60)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Read defaults from config
    config = GlobalConfig()
    default_config = {
        'nas': config.training.N_NA,
        'time_points': config.training.N_TIME_POINTS, 
        'seed': config.training.RANDOM_SEED,
        'attack_type': getattr(args, 'attack_type', 'OSA'),
        'malicious_selection': getattr(args, 'malicious_selection', 'reputation'),
        'ca_scale_startup': getattr(args, 'ca_scale_startup', 'warm')
    }

    # Fixed NA scale used in all-strategies comparison
    COMPARE_N_NA = 20
    
    # If advanced mode is enabled, override defaults via CLI args
    if args.advanced:
        if hasattr(args, 'nas'):
            default_config['nas'] = args.nas
        if hasattr(args, 'time_points'):
            default_config['time_points'] = args.time_points
        if hasattr(args, 'seed'):
            default_config['seed'] = args.seed
    
    # Set random seed
    set_all_seeds(default_config['seed'])
    
    # Repeated all-strategies comparison mode (CI)
    if getattr(args, 'compare_all_ci', False):
        print(" Start repeated all-strategies comparison (CI, repeats=10)")
        print(f"   NAs: {COMPARE_N_NA}")
        print(f"   Time points: {default_config['time_points']}")
        print(f"   Base seed: {default_config['seed']}")
        print("   Malicious selection: random (fixed)")
        print()

        try:
            manager = ExperimentManager(
                config_name='STRATEGY_COMPARISON_CI',
                save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison/ci'
            )

            manager.run_all_strategies_comparison_ci(
                n_time_points=default_config['time_points'],
                n_na=COMPARE_N_NA,
                seed=default_config['seed'],
                repeats=10
            )

            print("\n Repeated all-strategies comparison completed!")
            print(" Results saved to: /mnt/data/wy2024/baseline_methods/strategy_comparison/ci/")

            return 0

        except KeyboardInterrupt:
            print("\n Repeated all-strategies comparison interrupted by user")
            return 1

    # All-strategies comparison mode
    if args.compare_all:
        print(" Start all-strategies comparison")
        print(f"   NAs: {COMPARE_N_NA}")
        print(f"   Time points: {default_config['time_points']}")
        print(f"   Random seed: {default_config['seed']}")
        print()
        
        try:
            # Create experiment manager
            manager = ExperimentManager(
                config_name='STRATEGY_COMPARISON',
                save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison'
            )
            
            # Run full comparison
            comparison_results = manager.run_all_strategies_comparison(
                n_time_points=default_config['time_points'],
                n_na=COMPARE_N_NA,
                seed=default_config['seed'],
                malicious_selection_mode='random'
            )
            
            print("\n All-strategies comparison completed!")
            print(" Results saved to: /mnt/data/wy2024/baseline_methods/strategy_comparison/")
            
            return 0
            
        except KeyboardInterrupt:
            print("\n All-strategies comparison interrupted by user")
            return 1

    if getattr(args, 'ca_scale_sensitivity', False):
        warmup_first_round = default_config['ca_scale_startup'] == 'warm'
        startup_label = "Warm Start (warm-up first round)" if warmup_first_round else "Cold Start"
        malicious_label = "Random" if default_config['malicious_selection'] == 'random' else "Reputation"

        print(f" Start CA-scale sensitivity (DRL-DPKI, {startup_label})")
        print("   CA scales: 200, 400, 1000, 1600, 2500")
        print(f"   Time points: {default_config['time_points']}")
        print(f"   Random seed: {default_config['seed']}")
        print("   Startup options: cold / warm (warm-up first round)")
        print("   Malicious selection options: random / reputation")
        print(f"   Startup: {startup_label}")
        print(f"   Malicious selection: {malicious_label}")
        print()

        try:
            manager = ExperimentManager(
                config_name='CA_SCALE_SENSITIVITY',
                save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison/ca_scale_sensitivity'
            )

            manager.run_ca_scale_sensitivity(
                n_time_points=default_config['time_points'],
                scales=[400, 1000, 2500, 5200, 6000],
                seed=default_config['seed'],
                malicious_selection_mode=default_config['malicious_selection'],
                warmup_first_round=warmup_first_round
            )

            print("\n CA-scale sensitivity completed!")
            print(" Tip: run plot_results.py to generate cross-scale summary charts")
            return 0

        except KeyboardInterrupt:
            print("\n CA-scale sensitivity interrupted by user")
            return 1
        except Exception as e:
            print(f"\n CA-scale sensitivity failed: {e}")
            traceback.print_exc()
            return 1

    # DRL cold/warm comparison mode
    if getattr(args, 'compare_drl_cold_warm', False):
        print(" Start DRL-DPKI cold/warm comparison")
        print(f"   NAs: {default_config['nas']}")
        print(f"   Time points: {default_config['time_points']}")
        print(f"   Random seed: {default_config['seed']}")
        print()

        try:
            manager = ExperimentManager(
                config_name='DRL_COLD_WARM',
                save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison'
            )

            manager.run_drl_cold_warm_comparison(
                n_time_points=default_config['time_points'],
                n_na=default_config['nas'],
                seed=default_config['seed'],
                malicious_selection_mode=default_config['malicious_selection']
            )

            print("\n DRL cold/warm comparison completed!")
            print(" Results saved to: /mnt/data/wy2024/baseline_methods/strategy_comparison/")
            return 0

        except KeyboardInterrupt:
            print("\n DRL cold/warm comparison interrupted by user")
            return 1
        except Exception as e:
            print(f"\n DRL cold/warm comparison failed: {e}")
            traceback.print_exc()
            return 1
        except Exception as e:
            print(f"\n All-strategies comparison failed: {e}")
            traceback.print_exc()
            return 1
    
    # Determine test mode and strategy
    if args.strategy:
        # Strategy specified via CLI (single-strategy mode)
        test_mode = 'single'
        selected_strategy = args.strategy
        print(f" Strategy specified via CLI: {selected_strategy}")
    else:
        # Interactive selection mode
        if sys.stdin.isatty():
            test_mode = interactive_mode_selection()
            if test_mode is None:
                return 0
            
            if test_mode == 'compare':
                # All-strategies comparison mode
                try:
                    manager = ExperimentManager(
                        config_name='STRATEGY_COMPARISON',
                        save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison'
                    )
                    
                    # Use a fixed NA scale of 20 for all-strategies comparison
                    comparison_results = manager.run_all_strategies_comparison(
                        n_time_points=default_config['time_points'],
                        n_na=COMPARE_N_NA,
                        seed=default_config['seed'],
                        malicious_selection_mode='random'
                    )
                    
                    print("\n All-strategies comparison completed!")
                    print(" Results saved to: /mnt/data/wy2024/baseline_methods/strategy_comparison/")
                    return 0
                    
                except Exception as e:
                    print(f"\n All-strategies comparison failed: {e}")
                    traceback.print_exc()
                    return 1
            elif test_mode == 'ca_scale_sensitivity':
                try:
                    warmup_first_round = default_config['ca_scale_startup'] == 'warm'
                    manager = ExperimentManager(
                        config_name='CA_SCALE_SENSITIVITY',
                        save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison/ca_scale_sensitivity'
                    )

                    manager.run_ca_scale_sensitivity(
                        n_time_points=default_config['time_points'],
                        scales=[400, 1000, 2500, 5200, 6000],
                        seed=default_config['seed'],
                        malicious_selection_mode=default_config['malicious_selection'],
                        warmup_first_round=warmup_first_round
                    )

                    print("\n CA-scale sensitivity completed!")
                    print(" Tip: run plot_results.py to generate cross-scale summary charts")
                    return 0

                except Exception as e:
                    print(f"\n CA-scale sensitivity failed: {e}")
                    traceback.print_exc()
                    return 1
            elif test_mode == 'compare_drl_cold_warm':
                # DRL cold/warm comparison mode
                try:
                    manager = ExperimentManager(
                        config_name='DRL_COLD_WARM',
                        save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison'
                    )

                    manager.run_drl_cold_warm_comparison(
                        n_time_points=default_config['time_points'],
                        n_na=default_config['nas'],
                        seed=default_config['seed'],
                        malicious_selection_mode=default_config['malicious_selection']
                    )

                    print("\n DRL cold/warm comparison completed!")
                    print(" Results saved to: /mnt/data/wy2024/baseline_methods/strategy_comparison/")
                    return 0

                except Exception as e:
                    print(f"\n DRL cold/warm comparison failed: {e}")
                    traceback.print_exc()
                    return 1

            elif test_mode == 'drl_slidewindow_ablation':
                try:
                    manager = ExperimentManager(
                        config_name='DRL_SLIDEWINDOW_ABLATION',
                        save_dir='/mnt/data/wy2024/baseline_methods/strategy_comparison/slidewindow_ablation'
                    )

                    manager.run_drl_slidewindow_ablation_comparison(
                        n_time_points=default_config['time_points'],
                        n_na=COMPARE_N_NA,
                        seed=default_config['seed'],
                        malicious_selection_mode='random'
                    )

                    print("\n Sliding-window ablation comparison completed!")
                    print(" Results saved to: /mnt/data/wy2024/baseline_methods/strategy_comparison/slidewindow_ablation/")
                    return 0

                except Exception as e:
                    print(f"\n Sliding-window ablation comparison failed: {e}")
                    traceback.print_exc()
                    return 1
                    
            else:  # test_mode == 'single'
                # Single-strategy mode
                selected_strategy = interactive_strategy_selection()
                if selected_strategy is None:
                    return 0
        else:
            # Non-interactive environment: use default strategy
            test_mode = 'single'
            selected_strategy = 'random'
            print(" Non-interactive environment detected; using default strategy: random")
    
    # Run single-strategy test
    print(" Start single-strategy test...")
    print(f"   Strategy: {selected_strategy}")
    print(f"   NAs: {default_config['nas']}")
    print(f"   Time points: {default_config['time_points']}")
    print(f"   Attack type: {default_config['attack_type']}")
    print(f"   Random seed: {default_config['seed']}")
    print()
    
    try:
        # Create experiment manager
        manager = ExperimentManager(
            config_name='STANDARD',
            save_dir=args.save_dir
        )
        
        # Run strategy test
        history, reputation_change = manager.run_strategy_test(
            strategy_name=selected_strategy,
            n_time_points=default_config['time_points'],
            n_na=default_config['nas']
        )
        
        print("\n Strategy test completed!")
        print(f" Reputation change: {reputation_change:+.2f}")
        print(f" Results saved to: {manager.save_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n Strategy test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n Strategy test failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
