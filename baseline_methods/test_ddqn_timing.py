#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDQN timing benchmark script.

Measures the time spent filling per-NA sliding windows and making decisions in a DDQN strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from models import DDQNModelStrategy
from baseline_methods.environment_aa import create_baseline_environment, NAConfig
from baseline_methods.parameter_aa import get_standard_config

def test_ddqn_timing():
    """Benchmark the DDQN strategy timing performance."""
    print("Starting DDQN timing benchmark")
    print("="*60)
    
    # Create environment and strategy
    global_config = get_standard_config()
    
    # Build NAConfig
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
    print(f"  NA count: {na_config.n_na}")
    print(f"  Selected NA count: {na_config.selected_na_count}")
    print("  Steps: 50")
    print(f"  Device: {strategy.device}")
    print(f"  NA init mode: {na_config.na_init_mode}")
    print()
    
    # Reset timing stats
    strategy.reset_timing_statistics()
    
    # Run benchmark
    n_steps = 50
    step_times = []
    
    for step in range(n_steps):
        step_start_time = time.time()
        
        # Get current state
        state = env.get_state()
        
        # Select NAs using DDQN (includes timing measurement)
        selected_nas = strategy.select(state, env)
        
        # Step the environment
        state, reward, done, info = env.step(selected_nas)
        
        step_time = time.time() - step_start_time
        step_times.append(step_time)
        
        # Print progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:2d}/50 | "
                  f"Selected NAs: {selected_nas} | "
                  f"Reward: {reward:.3f} | "
                  f"Step time: {step_time*1000:.2f} ms")
    
    print("\nBenchmark completed")
    
    # Show detailed timing stats
    strategy.print_timing_summary()
    
    # Step-level statistics
    print("\n" + "="*60)
    print("Step-level timing statistics")
    print("="*60)
    print(f"Total steps: {len(step_times)}")
    print(f"Average step time: {np.mean(step_times)*1000:.2f} ms")
    print(f"Fastest step: {min(step_times)*1000:.2f} ms")
    print(f"Slowest step: {max(step_times)*1000:.2f} ms")
    print(f"Step time std: {np.std(step_times)*1000:.2f} ms")
    
    # Timing breakdown
    timing_stats = strategy.get_timing_statistics()
    if timing_stats['decision_count'] > 0:
        window_ratio = (timing_stats['total_window_fill_time_ms'] / 
                       (timing_stats['total_window_fill_time_ms'] + timing_stats['total_decision_time_ms'])) * 100
        decision_ratio = 100 - window_ratio
        
        print("\n" + "="*60)
        print("Timing breakdown")
        print("="*60)
        print(f"Sliding window fill time ratio: {window_ratio:.1f}%")
        print(f"Decision compute time ratio: {decision_ratio:.1f}%")
        
        # Detailed timings (first 10 and last 10)
        print("\nDetailed timing records (first 10):")
        for i in range(min(10, len(timing_stats['window_fill_times_ms']))):
            print(f"  Step {i+1:2d}: windowFill={timing_stats['window_fill_times_ms'][i]:.2f} ms, "
                  f"decision={timing_stats['decision_times_ms'][i]:.2f} ms")
        
        if len(timing_stats['window_fill_times_ms']) > 10:
            print("\nDetailed timing records (last 10):")
            start_idx = len(timing_stats['window_fill_times_ms']) - 10
            for i in range(start_idx, len(timing_stats['window_fill_times_ms'])):
                print(f"  Step {i+1:2d}: windowFill={timing_stats['window_fill_times_ms'][i]:.2f} ms, "
                      f"decision={timing_stats['decision_times_ms'][i]:.2f} ms")
    
    print("\n" + "="*60)
    print("DDQN timing benchmark finished")
    print("="*60)

def test_timing_methods():
    """Test the timing statistics helpers."""
    print("\nTesting timing statistics helpers")
    print("-"*40)
    
    # Create strategy instance
    strategy = DDQNModelStrategy()
    
    # Empty stats
    print("Empty stats:")
    empty_stats = strategy.get_timing_statistics()
    print(f"  Decision count: {empty_stats['decision_count']}")
    print(f"  Avg window fill time: {empty_stats['avg_window_fill_time_ms']:.2f} ms")
    
    # Simulate some timing data
    strategy.window_fill_times = [0.001, 0.002, 0.0015]
    strategy.decision_times = [0.008, 0.010, 0.009]
    strategy.total_window_fill_time = sum(strategy.window_fill_times)
    strategy.total_decision_time = sum(strategy.decision_times)
    strategy.decision_count = 3
    
    print("\nSimulated data stats:")
    stats = strategy.get_timing_statistics()
    print(f"  Decision count: {stats['decision_count']}")
    print(f"  Avg window fill time: {stats['avg_window_fill_time_ms']:.2f} ms")
    print(f"  Avg decision time: {stats['avg_decision_time_ms']:.2f} ms")
    
    # Reset
    print("\nTesting reset:")
    strategy.reset_timing_statistics()
    reset_stats = strategy.get_timing_statistics()
    print(f"  Decision count after reset: {reset_stats['decision_count']}")
    
    print("Timing statistics helper tests completed")

if __name__ == "__main__":
    try:
        # Run main timing benchmark
        test_ddqn_timing()
        
        # Run helper method tests
        test_timing_methods()
        
    except Exception as e:
        print(f"Error during tests: {e}")
        import traceback
        traceback.print_exc()
