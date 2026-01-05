#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline strategy test configuration.

Defines parameters for baseline strategy evaluation, including environment settings,
NA settings, and strategy parameters.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class TrainingConfig:
    """Training and evaluation parameters."""
    N_EPISODES: int = 50
    N_TIME_POINTS: int = 200
    N_NA: int = 20
    SELECTED_NA_COUNT: int = 5
    EXECUTION_CYCLES: int = 50
    
    RANDOM_SEED: int = 42


@dataclass
class NAStateConfig:
    """NA state parameters."""
    REPUTATION_MIN: float = 3300.0
    REPUTATION_MAX: float = 10000.0
    REPUTATION_THRESHOLD: float = 6600.0
    
    SUCCESS_RATE_MIN: float = 0.0
    SUCCESS_RATE_MAX: float = 1.0
    UNIFORM_SUCCESS_RATE: float = 0.85
    
    DELAY_LEVEL_MIN: float = 0.0
    DELAY_LEVEL_MAX: float = 1.0
    DELAY_LEVEL_RANGE: Tuple[float, float] = (0.05, 0.3)
    
    HUNGER_MIN: float = 0.0
    HUNGER_MAX: float = 1.0
    HUNGER_INITIAL_RANGE: Tuple[float, float] = (0.0, 0.5)
    
    MALICIOUS_ATTACK_ENABLED: bool = True
    MALICIOUS_ATTACK_TYPE: str = 'OSA'
    MALICIOUS_ATTACK_INDICES: List[int] = None
    MALICIOUS_RATIO: float = 0.3


@dataclass
class EnvironmentConfig:
    """Environment parameters."""
    MAX_STEPS: int = 200
    TRANSACTIONS_PER_STEP: int = 1
    WINDOW_SIZE: int = 50
    
    NA_INIT_MODE: str = 'two_groups'
    
    HUNGER_GROWTH_SCALE: float = 10.0
    HUNGER_GROWTH_LOG_BASE: float = 11.0


@dataclass
class ReputationConfig:
    """Reputation system parameters."""
    SUCCESS_REWARD_MULTIPLIER: float = 20.0
    FAILURE_PENALTY_MULTIPLIER: float = 100.0
    
    LOW_REP_WEIGHTS: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    HIGH_REP_WEIGHTS: Tuple[float, float, float] = (0.4, 0.2, 0.4)


@dataclass
class WindowConfig:
    """Windowing parameters."""
    WINDOW_SIZE: int = 50
    PACK_INTERVAL: int = 5
    MIN_TRANSACTIONS: int = 10
    
    USE_LINEAR_WEIGHTS: bool = True
    WEIGHT_DECAY: float = 0.9


@dataclass 
class StrategyConfig:
    """Strategy parameters."""
    HIGH_REPUTATION_COUNT: int = 3
    LOW_REPUTATION_COUNT: int = 2
    
    BALANCE_WEIGHT: float = 0.5
    
    REPUTATION_WEIGHT: float = 0.3
    SUCCESS_RATE_WEIGHT: float = 0.3
    HUNGER_WEIGHT: float = 0.2
    LATENCY_WEIGHT: float = 0.2


@dataclass
class VisualizationConfig:
    """Visualization parameters."""
    FONT_SIZE: int = 10
    FIGURE_SIZE: Tuple[int, int] = (16, 12)
    LINE_ALPHA: float = 0.8
    LINE_WIDTH: float = 1.5


class GlobalConfig:
    """
    Global configuration manager.

    Aggregates all config sections and provides a unified access interface.
    """
    
    def __init__(self):
        """Initialize the global config."""
        self.training = TrainingConfig()
        self.na_state = NAStateConfig()
        self.environment = EnvironmentConfig()
        self.reputation = ReputationConfig()
        self.window = WindowConfig()
        self.strategy = StrategyConfig()
        self.visualization = VisualizationConfig()
        
        if self.na_state.MALICIOUS_ATTACK_INDICES is None:
            self.na_state.MALICIOUS_ATTACK_INDICES = []
    
    def to_dict(self) -> Dict:
        """Convert to a dict."""
        return {
            'training': self.training.__dict__,
            'na_state': self.na_state.__dict__,
            'environment': self.environment.__dict__,
            'reputation': self.reputation.__dict__,
            'window': self.window.__dict__,
            'strategy': self.strategy.__dict__
        }
    
    def print_config(self):
        """Print config values."""
        print("Current configuration:")
        print("=" * 50)
        
        configs = [
            ("Training", self.training),
            ("NA State", self.na_state),
            ("Environment", self.environment),
            ("Reputation System", self.reputation),
            ("Time Window", self.window),
            ("Strategy", self.strategy)
        ]
        
        for name, config in configs:
            print(f"\n{name}:")
            for key, value in config.__dict__.items():
                print(f"  {key}: {value}")
    
    def get_environment_params(self) -> Dict:
        """Return parameters needed for environment initialization."""
        return {
            'n_na': self.training.N_NA,
            'max_steps': self.environment.MAX_STEPS,
            'selected_na_count': self.training.SELECTED_NA_COUNT,
            'transactions_per_step': self.environment.TRANSACTIONS_PER_STEP,
            'window_size': self.environment.WINDOW_SIZE,
            'reputation_min': self.na_state.REPUTATION_MIN,
            'reputation_max': self.na_state.REPUTATION_MAX,
            'reputation_threshold': self.na_state.REPUTATION_THRESHOLD,
            'na_init_mode': self.environment.NA_INIT_MODE,
            'uniform_success_rate': self.na_state.UNIFORM_SUCCESS_RATE,
            'delay_level_range': self.na_state.DELAY_LEVEL_RANGE,
            'malicious_attack_enabled': self.na_state.MALICIOUS_ATTACK_ENABLED,
            'malicious_attack_type': self.na_state.MALICIOUS_ATTACK_TYPE,
            'malicious_attack_indices': self.na_state.MALICIOUS_ATTACK_INDICES,
            'hunger_growth_scale': self.environment.HUNGER_GROWTH_SCALE,
            'hunger_growth_log_base': self.environment.HUNGER_GROWTH_LOG_BASE
        }
    
    def get_strategy_params(self) -> Dict:
        """Return parameters needed by strategies."""
        return {
            'high_reputation_count': self.strategy.HIGH_REPUTATION_COUNT,
            'low_reputation_count': self.strategy.LOW_REPUTATION_COUNT,
            'balance_weight': self.strategy.BALANCE_WEIGHT,
            'reputation_weight': self.strategy.REPUTATION_WEIGHT,
            'success_rate_weight': self.strategy.SUCCESS_RATE_WEIGHT,
            'hunger_weight': self.strategy.HUNGER_WEIGHT,
            'latency_weight': self.strategy.LATENCY_WEIGHT
        }
    
    def get_test_params(self) -> Dict:
        """Return parameters needed for tests."""
        return {
            'n_time_points': self.training.N_TIME_POINTS,
            'n_na': self.training.N_NA,
            'selected_na_count': self.training.SELECTED_NA_COUNT,
            'execution_cycles': self.training.EXECUTION_CYCLES,
            'random_seed': self.training.RANDOM_SEED
        }


# ============================================================================
# Predefined configuration scenarios
# ============================================================================

def get_standard_config() -> GlobalConfig:
    """Return the standard config."""
    return GlobalConfig()


def get_high_malicious_config() -> GlobalConfig:
    """Return a high-malicious-ratio config."""
    config = GlobalConfig()
    config.na_state.OSA_ATTACK_INDICES = [5, 6, 7, 8]
    return config


def get_fast_adaptation_config() -> GlobalConfig:
    """Return a fast-adaptation config."""
    config = GlobalConfig()
    config.window.WINDOW_SIZE = 20
    config.environment.WINDOW_SIZE = 20
    return config


CONFIG_SCENARIOS = {
    'STANDARD': get_standard_config,
    'HIGH_MALICIOUS': get_high_malicious_config,
    'FAST_ADAPTATION': get_fast_adaptation_config
}


def get_config(scenario: str = 'STANDARD') -> GlobalConfig:
    """
    Get a config by scenario name.
    
    Args:
        scenario: Scenario name
        
    Returns:
        GlobalConfig: Config object
    """
    if scenario in CONFIG_SCENARIOS:
        return CONFIG_SCENARIOS[scenario]()
    else:
        print(f"Unknown config scenario: {scenario}; using the standard config")
        return get_standard_config()


if __name__ == '__main__':
    config = GlobalConfig()
    config.print_config()
