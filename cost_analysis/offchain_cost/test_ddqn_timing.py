#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDQN Strategy Timing Measurement Script

Measure the time spent on (1) sliding-window filling per NA and (2) decision making in the DDQN strategy.
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
baseline_methods_dir = os.path.join(project_root, "baseline_methods")

if os.path.isdir(baseline_methods_dir) and baseline_methods_dir not in sys.path:
    sys.path.insert(0, baseline_methods_dir)

if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import numpy as np
import time
import json
import csv
import contextlib
import io
from datetime import datetime
from models import DDQNModelStrategy
from environment import create_baseline_environment, NAConfig
from parameter import get_standard_config

def filter_outliers(data, method='iqr', threshold=3.0):
    """
    Filter outliers.
    
    Args:
        data: Data list
        method: Filtering method ('iqr' or 'zscore')
        threshold: Threshold (multiplier for IQR, stddev multiplier for Z-score)
    
    Returns:
        tuple: (filtered_data, outliers, stats)
    """
    if not data or len(data) < 3:
        return data, [], {'total': len(data), 'outliers': 0, 'filtered': len(data)}
    
    data_array = np.array(data)
    outliers = []
    
    if method == 'iqr':
        # Interquartile range method
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        mask = (data_array >= lower_bound) & (data_array <= upper_bound)
        filtered_data = data_array[mask].tolist()
        outliers = data_array[~mask].tolist()
        
    elif method == 'zscore':
        # Z-score method
        mean = np.mean(data_array)
        std = np.std(data_array)
        
        if std == 0:  # Avoid division-by-zero
            return data, [], {'total': len(data), 'outliers': 0, 'filtered': len(data)}
        
        z_scores = np.abs((data_array - mean) / std)
        mask = z_scores <= threshold
        filtered_data = data_array[mask].tolist()
        outliers = data_array[~mask].tolist()
    
    else:
        raise ValueError(f"Unsupported filtering method: {method}")
    
    stats = {
        'total': len(data),
        'outliers': len(outliers),
        'filtered': len(filtered_data),
        'outlier_ratio': len(outliers) / len(data) * 100 if data else 0
    }
    
    return filtered_data, outliers, stats

def test_ddqn_timing():
    """Test DDQN strategy timing performance."""
    print("Starting DDQN timing measurement test")
    print("="*60)
    
    # Create environment and strategy
    global_config = get_standard_config()
    
    # Create NAConfig
    na_config = NAConfig(
        n_na=global_config.training.N_NA,
        selected_na_count=global_config.training.SELECTED_NA_COUNT,
        max_steps=global_config.environment.MAX_STEPS,
        window_size=global_config.window.WINDOW_SIZE,
        na_init_mode=global_config.environment.NA_INIT_MODE,
        uniform_success_rate=global_config.na_state.UNIFORM_SUCCESS_RATE,
        delay_level_range=global_config.na_state.DELAY_LEVEL_RANGE,
        malicious_attack_enabled=global_config.na_state.MALICIOUS_ATTACK_ENABLED,
        malicious_attack_type=global_config.na_state.MALICIOUS_ATTACK_TYPE,
        malicious_attack_indices=global_config.na_state.MALICIOUS_ATTACK_INDICES,
        malicious_ratio=global_config.na_state.MALICIOUS_RATIO
    )
    
    env = create_baseline_environment(na_config)
    strategy = DDQNModelStrategy()
    
    print("Test configuration:")
    print(f"   NA count: {na_config.n_na}")
    print(f"   Selected NA count: {na_config.selected_na_count}")
    print("   Test steps: 50")
    print(f"   Device: {strategy.device}")
    print(f"   NA init mode: {na_config.na_init_mode}")
    print()
    
    # Reset environment to initial state
    print("Resetting environment to initial state")
    env.reset()
    
    # Reset timing statistics
    strategy.reset_timing_statistics()
    
    # Run test
    n_steps = 50
    step_times = []
    
    for step in range(n_steps):
        step_start_time = time.time()
        
        # Get current state
        state = env.get_state()
        
        # Select NAs with DDQN strategy (includes timing)
        selected_nas = strategy.select(state, env)
        
        # Step the environment
        state, reward, done, info = env.step(selected_nas)
        
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:2d}/50 complete | "
                  f"Selected NAs: {selected_nas} | "
                  f"Reward: {reward:.3f} | "
                  f"Step time: {step_time*1000:.2f} ms")
    
    print("\nTest completed.")
    
    # Print detailed timing summary
    strategy.print_timing_summary()
    
    # Step-level statistics
    print("\n" + "="*60)
    print("Step-level timing statistics")
    print("="*60)
    print(f"Total steps: {len(step_times)}")
    print(f"Average step time: {np.mean(step_times)*1000:.2f} ms")
    print(f"Fastest step: {min(step_times)*1000:.2f} ms")
    print(f"Slowest step: {max(step_times)*1000:.2f} ms")
    print(f"Step time stddev: {np.std(step_times)*1000:.2f} ms")
    
    # Analyze time distribution
    timing_stats = strategy.get_timing_statistics()
    if timing_stats['decision_count'] > 0:
        window_ratio = (timing_stats['total_window_fill_time_ms'] / 
                       (timing_stats['total_window_fill_time_ms'] + timing_stats['total_decision_time_ms'])) * 100
        decision_ratio = 100 - window_ratio
        
        print("\n" + "="*60)
        print("Time distribution analysis")
        print("="*60)
        print(f"Sliding-window filling time ratio: {window_ratio:.1f}%")
        print(f"Decision computation time ratio: {decision_ratio:.1f}%")
        
        # Show detailed per-step time (first 10 and last 10)
        print("\nDetailed time records (first 10):")
        for i in range(min(10, len(timing_stats['window_fill_times_ms']))):
            print(f"   Step {i+1:2d}: window fill {timing_stats['window_fill_times_ms'][i]:.2f} ms, "
                  f"decision {timing_stats['decision_times_ms'][i]:.2f} ms")
        
        if len(timing_stats['window_fill_times_ms']) > 10:
            print("\nDetailed time records (last 10):")
            start_idx = len(timing_stats['window_fill_times_ms']) - 10
            for i in range(start_idx, len(timing_stats['window_fill_times_ms'])):
                print(f"   Step {i+1:2d}: window fill {timing_stats['window_fill_times_ms'][i]:.2f} ms, "
                      f"decision {timing_stats['decision_times_ms'][i]:.2f} ms")
    
    print("\n" + "="*60)
    print("DDQN timing measurement test finished")
    print("="*60)

def test_frequency_group(frequency, na_config, strategy, selections_count=30, warmup_count=3):
    """
    Test performance for a specific selection frequency group.
    
    Args:
        frequency: Frequency (select once every N steps)
        na_config: NA configuration
        strategy: DDQN strategy
        selections_count: Number of selections
        warmup_count: Warm-up selections
    
    Returns:
        dict: Result statistics
    """
    print(f"\nTesting frequency group: select once every {frequency} step(s)")
    print(f"   Target selections: {selections_count}")
    print(f"   Expected total transactions: {selections_count * frequency}")
    print(f"   Warm-up selections: {warmup_count}")
    
    # Reset environment
    env = create_baseline_environment(na_config)
    env.reset()
    
    # Warm-up run: exclude model initialization overhead
    if warmup_count > 0:
        print(f"   Starting warm-up run ({warmup_count} rounds)...")
        warmup_start_time = time.time()
        
        for warmup_round in range(warmup_count):
            round_start_time = time.time()
            state = env.get_state()
            
            # Run warm-up decision (excluded from statistics)
            selected_nas = strategy.select(state, env)
            
            # Run transaction steps
            for _ in range(frequency):
                state, reward, done, info = env.step(selected_nas)
            
            round_time = (time.time() - round_start_time) * 1000  # Convert to milliseconds
            print(f"      Warm-up {warmup_round + 1}/{warmup_count} completed (time: {round_time:.2f} ms)")
        
        total_warmup_time = (time.time() - warmup_start_time) * 1000
        print(f"   Warm-up completed (total time: {total_warmup_time:.2f} ms). Starting main test.")
    
    # Reset timing statistics after warm-up to keep metrics accurate
    strategy.reset_timing_statistics()
    
    # Run main test
    selection_times = []
    total_transactions = 0
    
    for selection_round in range(selections_count):
        selection_start_time = time.time()
        
        # Get current state
        state = env.get_state()
        
        # Select NAs using DDQN (includes timing measurement)
        selected_nas = strategy.select(state, env)
        
        # Run multiple transactions (based on frequency)
        for transaction_step in range(frequency):
            state, reward, done, info = env.step(selected_nas)
            total_transactions += len(selected_nas)
        
        selection_time = time.time() - selection_start_time
        selection_times.append(selection_time)
        
        # Print progress every 10 selections
        if (selection_round + 1) % 10 == 0:
            print(
                f"   Selection {selection_round + 1:2d}/{selections_count} completed | "
                f"Selected NAs: {selected_nas} | "
                f"Round time: {selection_time*1000:.2f} ms"
            )
    
    # Collect timing statistics
    timing_stats = strategy.get_timing_statistics()
    
    # Apply outlier filtering
    print("   Applying outlier filtering...")
    
    # Filter outliers in selection times
    selection_times_ms = [t * 1000 for t in selection_times]  # Convert to milliseconds
    filtered_selection_times, selection_outliers, selection_outlier_stats = filter_outliers(
        selection_times_ms, method='zscore', threshold=3.0
    )
    
    # Filter outliers in window-fill times
    filtered_window_times, window_outliers, window_outlier_stats = filter_outliers(
        timing_stats['window_fill_times_ms'], method='zscore', threshold=3.0
    )
    
    # Filter outliers in decision times
    filtered_decision_times, decision_outliers, decision_outlier_stats = filter_outliers(
        timing_stats['decision_times_ms'], method='zscore', threshold=3.0
    )
    
    # Print outlier filtering results
    if selection_outlier_stats['outliers'] > 0:
        print(
            f"   Warning: selection time outliers: {selection_outlier_stats['outliers']}/{selection_outlier_stats['total']} "
            f"({selection_outlier_stats['outlier_ratio']:.1f}%) - values: {[round(x, 2) for x in selection_outliers]}"
        )
    
    if window_outlier_stats['outliers'] > 0:
        print(
            f"   Warning: window-fill time outliers: {window_outlier_stats['outliers']}/{window_outlier_stats['total']} "
            f"({window_outlier_stats['outlier_ratio']:.1f}%) - values: {[round(x, 2) for x in window_outliers]}"
        )
    
    if decision_outlier_stats['outliers'] > 0:
        print(
            f"   Warning: decision time outliers: {decision_outlier_stats['outliers']}/{decision_outlier_stats['total']} "
            f"({decision_outlier_stats['outlier_ratio']:.1f}%) - values: {[round(x, 2) for x in decision_outliers]}"
        )
    
    # Compute statistics using filtered data
    avg_selection_time = np.mean(filtered_selection_times) if filtered_selection_times else 0
    min_selection_time = min(filtered_selection_times) if filtered_selection_times else 0
    max_selection_time = max(filtered_selection_times) if filtered_selection_times else 0
    std_selection_time = np.std(filtered_selection_times) if filtered_selection_times else 0
    
    avg_window_fill_time = np.mean(filtered_window_times) if filtered_window_times else 0
    avg_decision_time = np.mean(filtered_decision_times) if filtered_decision_times else 0
    
    # Recompute totals using filtered data
    total_window_time = sum(filtered_window_times) if filtered_window_times else 0
    total_decision_time = sum(filtered_decision_times) if filtered_decision_times else 0
    total_core_time = total_window_time + total_decision_time
    
    window_ratio = (total_window_time / total_core_time * 100) if total_core_time > 0 else 0
    decision_ratio = (total_decision_time / total_core_time * 100) if total_core_time > 0 else 0
    
    result = {
        'frequency': frequency,
        'selections_count': selections_count,
        'total_transactions': total_transactions,
        'selection_times': {
            'avg_ms': round(avg_selection_time, 2),
            'min_ms': round(min_selection_time, 2),
            'max_ms': round(max_selection_time, 2),
            'std_ms': round(std_selection_time, 2)
        },
        'window_fill_times': {
            'avg_ms': round(avg_window_fill_time, 2),
            'total_ms': round(total_window_time, 2),
            'ratio_percent': round(window_ratio, 1)
        },
        'decision_times': {
            'avg_ms': round(avg_decision_time, 2),
            'total_ms': round(total_decision_time, 2),
            'ratio_percent': round(decision_ratio, 1)
        },
        'outlier_filtering': {
            'selection_times': {
                'outliers_detected': selection_outlier_stats['outliers'],
                'total_samples': selection_outlier_stats['total'],
                'outlier_ratio_percent': round(selection_outlier_stats['outlier_ratio'], 1),
                'outlier_values': [round(x, 2) for x in selection_outliers]
            },
            'window_fill_times': {
                'outliers_detected': window_outlier_stats['outliers'],
                'total_samples': window_outlier_stats['total'],
                'outlier_ratio_percent': round(window_outlier_stats['outlier_ratio'], 1),
                'outlier_values': [round(x, 2) for x in window_outliers]
            },
            'decision_times': {
                'outliers_detected': decision_outlier_stats['outliers'],
                'total_samples': decision_outlier_stats['total'],
                'outlier_ratio_percent': round(decision_outlier_stats['outlier_ratio'], 1),
                'outlier_values': [round(x, 2) for x in decision_outliers]
            }
        },
        'detailed_times': {
            'original_data': {
                'selection_times_ms': [round(t, 2) for t in selection_times_ms],
                'window_fill_times_ms': [round(t, 2) for t in timing_stats['window_fill_times_ms']],
                'decision_times_ms': [round(t, 2) for t in timing_stats['decision_times_ms']]
            },
            'filtered_data': {
                'selection_times_ms': [round(t, 2) for t in filtered_selection_times],
                'window_fill_times_ms': [round(t, 2) for t in filtered_window_times],
                'decision_times_ms': [round(t, 2) for t in filtered_decision_times]
            }
        }
    }
    
    print(f"   Frequency group {frequency} completed:")
    print(f"      Selections: {selections_count}, Total transactions: {total_transactions}")
    print(f"      Avg selection time: {avg_selection_time:.2f} ms")
    print(f"      Avg window fill: {avg_window_fill_time:.2f} ms ({window_ratio:.1f}%)")
    print(f"      Avg decision time: {avg_decision_time:.2f} ms ({decision_ratio:.1f}%)")
    
    return result

def test_multiple_frequency_groups():
    """Test multiple frequency groups and generate a comparison report."""
    print("Starting multi-frequency comparison test")
    print("="*80)
    
    # Create environment configuration
    global_config = get_standard_config()
    na_config = NAConfig(
        n_na=global_config.training.N_NA,
        selected_na_count=global_config.training.SELECTED_NA_COUNT,
        max_steps=global_config.environment.MAX_STEPS,
        window_size=global_config.window.WINDOW_SIZE,
        na_init_mode=global_config.environment.NA_INIT_MODE,
        uniform_success_rate=global_config.na_state.UNIFORM_SUCCESS_RATE,
        delay_level_range=global_config.na_state.DELAY_LEVEL_RANGE,
        malicious_attack_enabled=global_config.na_state.MALICIOUS_ATTACK_ENABLED,
        malicious_attack_type=global_config.na_state.MALICIOUS_ATTACK_TYPE,
        malicious_attack_indices=global_config.na_state.MALICIOUS_ATTACK_INDICES,
        malicious_ratio=global_config.na_state.MALICIOUS_RATIO
    )
    
    # Create DDQN strategy (shared pretrained model across groups)
    strategy = DDQNModelStrategy()
    
    # Test configuration
    selections_per_group = 30
    warmup_count = 3  # Warm-up rounds
    
    print("Test configuration:")
    print(f"   NA count: {na_config.n_na}")
    print(f"   Selected NA count: {na_config.selected_na_count}")
    print(f"   Selections per group: {selections_per_group}")
    print(f"   Warm-up rounds: {warmup_count}")
    print(f"   Device: {strategy.device}")
    print(f"   NA init mode: {na_config.na_init_mode}")
    
    # Define frequency groups
    frequency_groups = [1, 5, 10, 20, 50, 100]
    results = {}
    
    # Run each frequency group
    for frequency in frequency_groups:
        print(f"\n{'='*60}")
        print(f"Testing frequency group: every {frequency} step(s)")
        print(f"{'='*60}")
        
        result = test_frequency_group(frequency, na_config, strategy, 
                                    selections_count=selections_per_group, 
                                    warmup_count=warmup_count)
        results[f"frequency_{frequency}"] = result
    
    # Generate comparison report
    print(f"\n{'='*80}")
    print("Generating comparison report")
    print(f"{'='*80}")
    
    # Build full report payload
    report = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'multi_frequency_comparison',
            'na_count': na_config.n_na,
            'selected_na_count': na_config.selected_na_count,
            'selections_per_group': selections_per_group,
            'warmup_count': warmup_count,
            'device': str(strategy.device),
            'na_init_mode': na_config.na_init_mode
        },
        'frequency_groups': results,
        'summary': {
            'total_groups_tested': len(frequency_groups),
            'frequency_range': f"{min(frequency_groups)}-{max(frequency_groups)}",
            'transaction_counts': {
                f"frequency_{freq}": results[f"frequency_{freq}"]['total_transactions'] 
                for freq in frequency_groups
            }
        }
    }
    
    # Save JSON report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"frequency_comparison_report_{timestamp}.json"
    
    # Ensure the report is saved next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Comparison report saved: {filepath}")
    
    # Print a brief summary table
    print("\nPerformance comparison summary:")
    print("-" * 100)
    print(f"{'Freq':<8} {'Selects':<8} {'TxTotal':<8} {'AvgSelect(ms)':<15} {'WindowFill(ms)':<15} {'Decision(ms)':<15}")
    print("-" * 100)
    
    for frequency in frequency_groups:
        result = results[f"frequency_{frequency}"]
        print(f"{frequency:<8} {result['selections_count']:<8} {result['total_transactions']:<8} "
              f"{result['selection_times']['avg_ms']:<15} "
              f"{result['window_fill_times']['avg_ms']:<15} "
              f"{result['decision_times']['avg_ms']:<15}")
    
    print("-" * 100)
    print(f"Multi-frequency comparison test completed. Report file: {filename}")
    
    return report

def _build_na_config(global_config, n_na: int) -> NAConfig:
    return NAConfig(
        n_na=n_na,
        selected_na_count=global_config.training.SELECTED_NA_COUNT,
        max_steps=global_config.environment.MAX_STEPS,
        window_size=global_config.window.WINDOW_SIZE,
        na_init_mode=global_config.environment.NA_INIT_MODE,
        uniform_success_rate=global_config.na_state.UNIFORM_SUCCESS_RATE,
        delay_level_range=global_config.na_state.DELAY_LEVEL_RANGE,
        malicious_attack_enabled=global_config.na_state.MALICIOUS_ATTACK_ENABLED,
        malicious_attack_type=global_config.na_state.MALICIOUS_ATTACK_TYPE,
        malicious_attack_indices=global_config.na_state.MALICIOUS_ATTACK_INDICES,
        malicious_ratio=global_config.na_state.MALICIOUS_RATIO,
    )


def test_scale_group(
    n_na,
    global_config,
    strategy,
    selections_count=30,
    warmup_count=3,
    frequency=1,
    silence_stdout=True,
):
    print(f"\nStarting scale group test: n_na={n_na}")
    print(f"   Selections: {selections_count}")
    print(f"   Warm-up rounds: {warmup_count}")
    print(f"   frequency: {frequency}")

    na_config = _build_na_config(global_config, n_na)

    env = create_baseline_environment(na_config)
    env.reset()

    if warmup_count > 0:
        with (contextlib.redirect_stdout(io.StringIO()) if silence_stdout else contextlib.nullcontext()):
            for _ in range(warmup_count):
                state = env.get_state()
                selected_nas = strategy.select(state, env)
                for _ in range(frequency):
                    env.step(selected_nas)

    strategy.reset_timing_statistics()

    total_transactions = 0
    with (contextlib.redirect_stdout(io.StringIO()) if silence_stdout else contextlib.nullcontext()):
        for _ in range(selections_count):
            state = env.get_state()
            selected_nas = strategy.select(state, env)
            for _ in range(frequency):
                env.step(selected_nas)
                total_transactions += len(selected_nas)

    timing_stats = strategy.get_timing_statistics()

    filtered_window_times, window_outliers, window_outlier_stats = filter_outliers(
        timing_stats['window_fill_times_ms'], method='zscore', threshold=3.0
    )
    filtered_decision_times, decision_outliers, decision_outlier_stats = filter_outliers(
        timing_stats['decision_times_ms'], method='zscore', threshold=3.0
    )

    avg_window_fill_time = np.mean(filtered_window_times) if filtered_window_times else 0.0
    std_window_fill_time = np.std(filtered_window_times) if filtered_window_times else 0.0
    min_window_fill_time = min(filtered_window_times) if filtered_window_times else 0.0
    max_window_fill_time = max(filtered_window_times) if filtered_window_times else 0.0

    avg_decision_time = np.mean(filtered_decision_times) if filtered_decision_times else 0.0
    std_decision_time = np.std(filtered_decision_times) if filtered_decision_times else 0.0
    min_decision_time = min(filtered_decision_times) if filtered_decision_times else 0.0
    max_decision_time = max(filtered_decision_times) if filtered_decision_times else 0.0

    total_window_time = sum(filtered_window_times) if filtered_window_times else 0.0
    total_decision_time = sum(filtered_decision_times) if filtered_decision_times else 0.0
    total_core_time = total_window_time + total_decision_time

    window_ratio = (total_window_time / total_core_time * 100.0) if total_core_time > 0 else 0.0
    decision_ratio = (total_decision_time / total_core_time * 100.0) if total_core_time > 0 else 0.0

    result = {
        'ca_scale': int(n_na),
        'selections_count': int(selections_count),
        'warmup_count': int(warmup_count),
        'frequency': int(frequency),
        'total_transactions': int(total_transactions),
        'timing': {
            'window_fill_times': {
                'avg_ms': round(float(avg_window_fill_time), 4),
                'std_ms': round(float(std_window_fill_time), 4),
                'min_ms': round(float(min_window_fill_time), 4),
                'max_ms': round(float(max_window_fill_time), 4),
                'total_ms': round(float(total_window_time), 4),
                'ratio_percent': round(float(window_ratio), 2),
            },
            'decision_times': {
                'avg_ms': round(float(avg_decision_time), 4),
                'std_ms': round(float(std_decision_time), 4),
                'min_ms': round(float(min_decision_time), 4),
                'max_ms': round(float(max_decision_time), 4),
                'total_ms': round(float(total_decision_time), 4),
                'ratio_percent': round(float(decision_ratio), 2),
            },
        },
        'outlier_filtering': {
            'window_fill_times': {
                'outliers_detected': int(window_outlier_stats['outliers']),
                'total_samples': int(window_outlier_stats['total']),
                'outlier_ratio_percent': round(float(window_outlier_stats['outlier_ratio']), 2),
                'outlier_values': [round(float(x), 4) for x in window_outliers],
            },
            'decision_times': {
                'outliers_detected': int(decision_outlier_stats['outliers']),
                'total_samples': int(decision_outlier_stats['total']),
                'outlier_ratio_percent': round(float(decision_outlier_stats['outlier_ratio']), 2),
                'outlier_values': [round(float(x), 4) for x in decision_outliers],
            },
        },
        'detailed_times': {
            'original_data': {
                'window_fill_times_ms': [round(float(t), 4) for t in timing_stats['window_fill_times_ms']],
                'decision_times_ms': [round(float(t), 4) for t in timing_stats['decision_times_ms']],
            },
            'filtered_data': {
                'window_fill_times_ms': [round(float(t), 4) for t in filtered_window_times],
                'decision_times_ms': [round(float(t), 4) for t in filtered_decision_times],
            },
        },
    }

    print(f"   Scale group n_na={n_na} completed")
    print(f"      Avg window fill: {avg_window_fill_time:.4f} ms ({window_ratio:.2f}%)")
    print(f"      Avg decision time: {avg_decision_time:.4f} ms ({decision_ratio:.2f}%)")

    return result


def test_multiple_scale_groups():
    print("Starting multi-scale CA decision timing test")
    print("=" * 80)

    scales = [20, 400, 1000, 2500, 5200, 6000]
    selections_per_scale = 30
    warmup_count = 3
    frequency = 1
    silence_stdout = True

    global_config = get_standard_config()
    strategy = DDQNModelStrategy()

    print("Test configuration:")
    print(f"   scales: {scales}")
    print(f"   Selections per scale: {selections_per_scale}")
    print(f"   Warm-up rounds: {warmup_count}")
    print(f"   frequency: {frequency}")
    print(f"   silence_stdout: {int(bool(silence_stdout))}")
    print(f"   Device: {strategy.device}")
    print(f"   NA init mode: {global_config.environment.NA_INIT_MODE}")

    results = {}
    for n_na in scales:
        result = test_scale_group(
            n_na,
            global_config,
            strategy,
            selections_count=selections_per_scale,
            warmup_count=warmup_count,
            frequency=frequency,
            silence_stdout=silence_stdout,
        )
        results[f"scale_{n_na}"] = result

    report = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'multi_scale_comparison',
            'scales': scales,
            'selections_per_scale': selections_per_scale,
            'warmup_count': warmup_count,
            'frequency': frequency,
            'silence_stdout': int(bool(silence_stdout)),
            'device': str(strategy.device),
            'na_init_mode': global_config.environment.NA_INIT_MODE,
        },
        'scale_groups': results,
        'summary': {
            'total_scales_tested': len(scales),
            'scale_range': f"{min(scales)}-{max(scales)}",
        },
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"scale_comparison_report_{timestamp}.json"
    report_path = os.path.join(script_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    csv_filename = "scale_timing_summary.csv"
    csv_path = os.path.join(script_dir, csv_filename)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'ca_scale',
                'selections_count',
                'warmup_count',
                'frequency',
                'decision_count',
                'avg_window_fill_ms',
                'std_window_fill_ms',
                'min_window_fill_ms',
                'max_window_fill_ms',
                'avg_decision_ms',
                'std_decision_ms',
                'min_decision_ms',
                'max_decision_ms',
                'window_ratio_percent',
                'decision_ratio_percent',
                'window_outliers',
                'decision_outliers',
            ],
        )
        writer.writeheader()

        for n_na in scales:
            item = results[f"scale_{n_na}"]
            timing = item['timing']
            window_fill = timing['window_fill_times']
            decision = timing['decision_times']

            writer.writerow(
                {
                    'ca_scale': item['ca_scale'],
                    'selections_count': item['selections_count'],
                    'warmup_count': item['warmup_count'],
                    'frequency': item['frequency'],
                    'decision_count': len(item['detailed_times']['original_data']['decision_times_ms']),
                    'avg_window_fill_ms': window_fill['avg_ms'],
                    'std_window_fill_ms': window_fill['std_ms'],
                    'min_window_fill_ms': window_fill['min_ms'],
                    'max_window_fill_ms': window_fill['max_ms'],
                    'avg_decision_ms': decision['avg_ms'],
                    'std_decision_ms': decision['std_ms'],
                    'min_decision_ms': decision['min_ms'],
                    'max_decision_ms': decision['max_ms'],
                    'window_ratio_percent': window_fill['ratio_percent'],
                    'decision_ratio_percent': decision['ratio_percent'],
                    'window_outliers': item['outlier_filtering']['window_fill_times']['outliers_detected'],
                    'decision_outliers': item['outlier_filtering']['decision_times']['outliers_detected'],
                }
            )

    print("\n" + "=" * 80)
    print("Multi-scale comparison test completed")
    print(f"Report saved: {report_path}")
    print(f"Summary CSV saved: {csv_path}")

    print("\nScale comparison summary:")
    print("-" * 110)
    print(f"{'CA Scale':<10} {'avg_window(ms)':<16} {'avg_decision(ms)':<18} {'window%':<10} {'decision%':<10}")
    print("-" * 110)

    for n_na in scales:
        item = results[f"scale_{n_na}"]
        window_fill = item['timing']['window_fill_times']
        decision = item['timing']['decision_times']
        print(
            f"{n_na:<10} {window_fill['avg_ms']:<16} {decision['avg_ms']:<18} "
            f"{window_fill['ratio_percent']:<10} {decision['ratio_percent']:<10}"
        )

    print("-" * 110)

    return report


def test_scale_frequency_grid():
    print("Starting CA scale x tenure (T_tenure) grid timing test")
    print("=" * 80)

    scales = [20, 400, 1000, 2500, 5200, 6000]
    frequency_groups = [1, 5, 10, 20, 50, 100]
    selections_per_cell = 30
    warmup_count = 3
    silence_stdout = True

    global_config = get_standard_config()
    strategy = DDQNModelStrategy()

    print("Test configuration:")
    print(f"   scales: {scales}")
    print(f"   T_tenure list (frequency): {frequency_groups}")
    print(f"   Selections per cell: {selections_per_cell}")
    print(f"   Warm-up rounds: {warmup_count}")
    print(f"   silence_stdout: {int(bool(silence_stdout))}")
    print(f"   Device: {strategy.device}")
    print(f"   NA init mode: {global_config.environment.NA_INIT_MODE}")

    results = {}
    for n_na in scales:
        for frequency in frequency_groups:
            key = f"scale_{n_na}_freq_{frequency}"
            result = test_scale_group(
                n_na,
                global_config,
                strategy,
                selections_count=selections_per_cell,
                warmup_count=warmup_count,
                frequency=frequency,
                silence_stdout=silence_stdout,
            )
            results[key] = result

    report = {
        'test_info': {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'scale_frequency_grid',
            'scales': scales,
            'frequency_groups': frequency_groups,
            'selections_per_cell': selections_per_cell,
            'warmup_count': warmup_count,
            'silence_stdout': int(bool(silence_stdout)),
            'device': str(strategy.device),
            'na_init_mode': global_config.environment.NA_INIT_MODE,
        },
        'grid_results': results,
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_filename = f"scale_frequency_grid_report_{timestamp}.json"
    report_path = os.path.join(script_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    csv_filename = "scale_frequency_timing_summary.csv"
    csv_path = os.path.join(script_dir, csv_filename)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'ca_scale',
                'frequency',
                'selections_count',
                'warmup_count',
                'decision_count',
                'avg_window_fill_ms',
                'std_window_fill_ms',
                'min_window_fill_ms',
                'max_window_fill_ms',
                'avg_decision_ms',
                'std_decision_ms',
                'min_decision_ms',
                'max_decision_ms',
                'window_ratio_percent',
                'decision_ratio_percent',
                'window_outliers',
                'decision_outliers',
                'total_transactions',
            ],
        )
        writer.writeheader()

        for n_na in scales:
            for frequency in frequency_groups:
                item = results[f"scale_{n_na}_freq_{frequency}"]
                timing = item['timing']
                window_fill = timing['window_fill_times']
                decision = timing['decision_times']

                writer.writerow(
                    {
                        'ca_scale': item['ca_scale'],
                        'frequency': item['frequency'],
                        'selections_count': item['selections_count'],
                        'warmup_count': item['warmup_count'],
                        'decision_count': len(item['detailed_times']['original_data']['decision_times_ms']),
                        'avg_window_fill_ms': window_fill['avg_ms'],
                        'std_window_fill_ms': window_fill['std_ms'],
                        'min_window_fill_ms': window_fill['min_ms'],
                        'max_window_fill_ms': window_fill['max_ms'],
                        'avg_decision_ms': decision['avg_ms'],
                        'std_decision_ms': decision['std_ms'],
                        'min_decision_ms': decision['min_ms'],
                        'max_decision_ms': decision['max_ms'],
                        'window_ratio_percent': window_fill['ratio_percent'],
                        'decision_ratio_percent': decision['ratio_percent'],
                        'window_outliers': item['outlier_filtering']['window_fill_times']['outliers_detected'],
                        'decision_outliers': item['outlier_filtering']['decision_times']['outliers_detected'],
                        'total_transactions': item['total_transactions'],
                    }
                )

    print("\n" + "=" * 80)
    print("Grid test completed")
    print(f"Report saved: {report_path}")
    print(f"Summary CSV saved: {csv_path}")

    return report


def test_timing_methods():
    """Test timing statistics methods."""
    print("\nTesting timing statistics methods")
    print("-"*40)
    
    strategy = DDQNModelStrategy()
    
    # Test empty statistics
    print("Testing empty statistics...")
    strategy.reset_timing_statistics()
    stats = strategy.get_timing_statistics()
    print(f"   Empty statistics: {stats}")
    
    # Test with simulated data
    print("Testing simulated data statistics...")
    strategy.reset_timing_statistics()
    
    # Simulate some timing data
    strategy.window_fill_times_ms = [10.5, 12.3, 9.8, 11.2, 13.1]
    strategy.decision_times_ms = [25.4, 23.8, 26.1, 24.7, 25.9]
    strategy.decision_count = 5
    
    stats = strategy.get_timing_statistics()
    print("   Simulated statistics:")
    print(f"      Decision count: {stats['decision_count']}")
    print(f"      Total window fill time: {stats['total_window_fill_time_ms']:.2f} ms")
    print(f"      Total decision time: {stats['total_decision_time_ms']:.2f} ms")
    print(f"      Avg window fill time: {stats['avg_window_fill_time_ms']:.2f} ms")
    print(f"      Avg decision time: {stats['avg_decision_time_ms']:.2f} ms")
    
    # Test reset behavior
    print("Testing reset behavior...")
    strategy.reset_timing_statistics()
    stats_after_reset = strategy.get_timing_statistics()
    print(f"   After reset: decision_count = {stats_after_reset['decision_count']}")
    print("Timing statistics methods test completed")

if __name__ == "__main__":
    try:
        print("Select test mode:")
        print("1. Multi-frequency comparison (test_multiple_frequency_groups)")
        print("2. Multi-scale CA decision timing (test_multiple_scale_groups)")
        print("3. Run both 1 and 2")
        print("4. CA scale x tenure grid test (test_scale_frequency_grid)")

        try:
            choice = input("Enter choice (default 1): ").strip()
        except EOFError:
            choice = ""

        if not choice:
            choice = "1"

        if choice == "1":
            test_multiple_frequency_groups()
        elif choice == "2":
            test_multiple_scale_groups()
        elif choice == "3":
            test_multiple_frequency_groups()
            test_multiple_scale_groups()
        elif choice == "4":
            test_scale_frequency_grid()
        else:
            print(f"Invalid choice: {choice}")

    except Exception as e:
        print(f"Error during test execution: {e}")
        import traceback
        traceback.print_exc()
