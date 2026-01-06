#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the modified DDQN strategy periodic decision mode.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import DDQNModelStrategy, create_all_baseline_strategies
from baseline_methods.environment_aa import create_baseline_environment, NAConfig
from baseline_methods.parameter_aa import TrainingConfig, get_standard_config
import numpy as np

def test_ddqn_cycle_mode():
    """
    Test the DDQN strategy periodic decision mode.
    """
    print("Testing DDQN periodic decision mode")
    print("=" * 60)
    
    # Load config
    global_config = get_standard_config()
    
    # Create environment config (based on the config file values)
    config = NAConfig(
        n_na=global_config.training.N_NA,
        max_steps=global_config.environment.MAX_STEPS,
        selected_na_count=global_config.training.SELECTED_NA_COUNT,
        window_pack_interval=global_config.window.PACK_INTERVAL,
        window_queue_size=global_config.window.WINDOW_SIZE
    )
    
    # Create environment
    env = create_baseline_environment(config)
    
    # Create DDQN strategy (use the configured execution cycles, but reduce for test)
    try:
        # Use a smaller cycle count for test visibility
        test_cycles = min(10, global_config.training.EXECUTION_CYCLES // 5)  # 1/5 of config or 10, whichever is smaller
        ddqn_strategy = DDQNModelStrategy(execution_cycles=test_cycles, config=global_config)
        print(
            f"DDQN strategy created. execution_cycles={ddqn_strategy.execution_cycles} "
            f"(config default: {global_config.training.EXECUTION_CYCLES})"
        )
    except Exception as e:
        print(f"Failed to create DDQN strategy: {e}")
        print("Possible cause: model file missing. Falling back to random strategy for testing.")
        from models import RandomSelectionStrategy
        ddqn_strategy = RandomSelectionStrategy()
        # Manually add periodic decision attributes for testing
        ddqn_strategy.execution_cycles = test_cycles
        ddqn_strategy.current_cycle = 0
        ddqn_strategy.selected_nas = None
        ddqn_strategy.is_first_decision = True
        ddqn_strategy.cycle_transaction_data = {}
        
        def mock_collect_transaction_data(na_id, success, latency, delay_level, timestamp, reputation_after):
            print(f"  Collected tx data: NA{na_id}, success={success}, latency={latency:.1f}ms")
        
        ddqn_strategy.collect_transaction_data = mock_collect_transaction_data
    
    # Run test
    print("\nStarting test...")
    
    # Attach current strategy to env
    env.current_strategy = ddqn_strategy
    state = env.reset()
    
    total_reward = 0.0
    decision_count = 0
    
    for step in range(50):  # Test 50 steps
        print(f"\n--- Step {step + 1} ---")
        
        # Select NAs
        selected_nas = ddqn_strategy.select(state, env)
        
        # Check whether a new decision was made
        if hasattr(ddqn_strategy, 'current_cycle'):
            if ddqn_strategy.current_cycle == 1:  # start of a new cycle
                decision_count += 1
                print(f"Decision {decision_count}: selected NAs: {selected_nas}")
            else:
                print(f"Cycle step {ddqn_strategy.current_cycle}: reusing NAs: {selected_nas}")
        else:
            print(f"Selected NAs: {selected_nas}")
        
        # Step
        state, reward, done, info = env.step(selected_nas)
        total_reward += reward
        
        print(f"  Reward: {reward:.2f}, Success rate: {info['success_rate']:.3f}")
        
        if done:
            break
    
    # Detach
    env.current_strategy = None
    
    print("\nTest results:")
    print(f"  Total steps: {step + 1}")
    print(f"  Decision count: {decision_count}")
    print(f"  Avg cycle length: {(step + 1) / max(decision_count, 1):.1f} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Avg reward: {total_reward / (step + 1):.3f}")
    
    if hasattr(ddqn_strategy, 'execution_cycles'):
        expected_decisions = (step + 1) // ddqn_strategy.execution_cycles + 1
        print(f"  Expected decision count: {expected_decisions}")
        print(f"  Decision frequency match: {'OK' if abs(decision_count - expected_decisions) <= 1 else 'MISMATCH'}")

def test_comparison_with_normal_strategy():
    """
    Comparison test: DDQN periodic mode vs. normal real-time mode.
    """
    print("\nComparison: periodic mode vs real-time mode")
    print("=" * 60)
    
    # Load config
    global_config = get_standard_config()
    
    # Create environment (use fewer steps for the test)
    test_steps = min(100, global_config.environment.MAX_STEPS // 2)  # half of config or 100, whichever is smaller
    config = NAConfig(
        n_na=global_config.training.N_NA, 
        max_steps=test_steps,
        selected_na_count=global_config.training.SELECTED_NA_COUNT,
        window_pack_interval=global_config.window.PACK_INTERVAL
    )
    
    # Normal strategy (real-time mode)
    print("\nTesting normal strategy (real-time packing):")
    env1 = create_baseline_environment(config)
    from models import RandomSelectionStrategy
    normal_strategy = RandomSelectionStrategy()
    
    test_eval_steps = min(30, test_steps // 3)  # 1/3 of test_steps, capped at 30
    result1 = normal_strategy.evaluate(env1, n_steps=test_eval_steps)
    print(f"  Strategy: {result1['strategy_name']}")
    print(f"  Avg reward: {result1['avg_reward_per_step']:.3f}")
    print(f"  Success rate: {result1['avg_success_rate']:.3f}")
    
    # DDQN strategy (periodic mode)
    print("\nTesting DDQN strategy (periodic packing):")
    env2 = create_baseline_environment(config)
    try:
        # Use the configured execution cycles, but reduce for the test
        test_cycles = min(10, global_config.training.EXECUTION_CYCLES // 5)
        ddqn_strategy = DDQNModelStrategy(execution_cycles=test_cycles, config=global_config)
        result2 = ddqn_strategy.evaluate(env2, n_steps=test_eval_steps)
        print(f"  Strategy: {result2['strategy_name']}")
        print(f"  Avg reward: {result2['avg_reward_per_step']:.3f}")
        print(f"  Success rate: {result2['avg_success_rate']:.3f}")
    except Exception as e:
        print(f"  DDQN test failed: {e}")
        print("  Ensure the model file exists: <output_root>/models/policy_net_state_dict.pth")

if __name__ == "__main__":
    test_ddqn_cycle_mode()
    test_comparison_with_normal_strategy()
    print("\nTests completed")
