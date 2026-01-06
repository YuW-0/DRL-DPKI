"""
Model Interface for Baseline NA Selection Environment

This module provides interfaces for different NA selection models to work with
the baseline environment. It includes abstract base classes for selection strategies
and concrete implementations of various baseline algorithms.

Key Features:
- Abstract base class for selection strategies
- Concrete baseline algorithm implementations
- Integration with environment.py
- Performance evaluation and comparison utilities
"""

import numpy as np
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from environment import NAEnvironment, NAConfig, create_baseline_environment
from parameter import GlobalConfig, get_standard_config
import time
import json
import os
import torch
import torch.nn as nn
from pathlib import Path

def get_output_root() -> Path:
    configured = os.environ.get("DRL_DPKI_OUTPUT_DIR")
    if configured:
        return Path(configured)
    legacy = Path("/mnt/data/wy2024")
    if legacy.is_dir():
        return legacy
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "outputs"


class NASelectionStrategy(ABC):
    """
    Abstract base class for NA selection strategies.

    All selection algorithms should inherit from this class and implement select().
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """
        Select NA indices.
        
        Args:
            state: Current environment state (n_na, 4) - [success_rate, latency, reputation, hunger]
            env: Environment instance for accessing additional information
            
        Returns:
            List[int]: Selected NA indices (must be exactly 5)
        """
        pass
    
    def evaluate(self, env: NAEnvironment, n_steps: int = 100) -> Dict[str, Any]:
        """
        Evaluate strategy performance.
        
        Args:
            env: Environment instance
            n_steps: Number of evaluation steps
            
        Returns:
            Dict[str, Any]: Evaluation result
        """
        start_time = time.time()
        
        env.current_strategy = self
        
        state = env.reset()
        
        total_reward = 0.0
        success_rates = []
        latencies = []
        step_rewards = []
        
        for step in range(n_steps):
            selected_nas = self.select(state, env)
            
            state, reward, done, info = env.step(selected_nas)
            
            total_reward += reward
            success_rates.append(info['success_rate'])
            latencies.append(info['avg_latency'])
            step_rewards.append(reward)
            
            if done:
                break
        
        env.current_strategy = None
        
        evaluation_time = time.time() - start_time
        
        return {
            'strategy_name': self.name,
            'total_reward': total_reward,
            'avg_reward_per_step': total_reward / (step + 1),
            'avg_success_rate': np.mean(success_rates),
            'avg_latency': np.mean(latencies),
            'success_rate_std': np.std(success_rates),
            'latency_std': np.std(latencies),
            'reward_std': np.std(step_rewards),
            'steps_completed': step + 1,
            'evaluation_time': evaluation_time,
            'malicious_nas_selected': len([s for step_info in [info] 
                                         for s in step_info.get('malicious_selected', [])])
        }


class QNetwork(nn.Module):
    """DDQN Q-network structure with per-NA encoding and Q-value prediction."""
    
    def __init__(self, n_features, n_na):
        super().__init__()
        self.na_encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        self.q_predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights to diversify outputs."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x):
        batch_size = x.size(0)
        n_na = x.size(1)
        x_flat = x.view(batch_size * n_na, -1)  # [batch*n_na, n_features]
        na_embeddings = self.na_encoder(x_flat)  # [batch*n_na, 16]
        q_values_flat = self.q_predictor(na_embeddings)  # [batch*n_na, 1]
        q_values = q_values_flat.view(batch_size, n_na)  # [batch, n_na]
        return q_values


class DDQNModelStrategy(NASelectionStrategy):
    """Selection strategy based on a trained DDQN model (release name: DRL-DPKI)."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None, 
                 execution_cycles: Optional[int] = None, config: Optional[GlobalConfig] = None):
        super().__init__("DRL-DPKI")
        
        if device is None:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                self.device = torch.device('cuda:1')
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        if model_path is None:
            model_path = str(get_output_root() / "models" / "policy_net_state_dict.pth")
        
        self.model_path = model_path
        self.model = None
        self.model_info = None
        
        if config is None:
            config = get_standard_config()
        
        self.window_fill_times = []
        self.decision_times = []
        self.total_window_fill_time = 0.0
        self.total_decision_time = 0.0
        self.decision_count = 0
        
        self._load_model()
    
    def _load_model(self, allow_fallback: bool = True):
        """Load a trained DDQN model and optionally fall back to CPU."""
        try:
            info_path = Path(self.model_path).parent / 'model_info.pth'
            if info_path.exists():
                self.model_info = torch.load(info_path, map_location='cpu')
                print(
                    f"Model info: trained for {self.model_info['n_episodes']} episodes, "
                    f"final reward: {self.model_info['final_reward']:.1f}"
                )

            n_features = self.model_info.get('n_features', 4) if self.model_info else 4
            n_na = self.model_info.get('n_na', 20) if self.model_info else 20

            self.model = QNetwork(n_features=n_features, n_na=n_na)

            state_dict = torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)

            self.model.eval()
            self.model.to(self.device)

            print(f"DRL-DPKI model loaded: {self.model_path}")
            print(f"   Device: {self.device}")
            print(f"   Network: {n_features} features -> {n_na} NA Q-values")

        except RuntimeError as exc:
            error_message = str(exc).lower()
            if allow_fallback and ('cuda out of memory' in error_message or 'cuda error' in error_message):
                print(f"GPU load failed; DRL-DPKI falling back to CPU: {exc}")
                self.device = torch.device('cpu')
                self.model = None
                self._load_model(allow_fallback=False)
                return
            print(f"Model load failed: {exc}")
            print(f"   Model path: {self.model_path}")
            raise
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Model load failed: {exc}")
            print(f"   Model path: {self.model_path}")
            raise
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Select the top NA by DDQN Q-values and refresh the window before decision."""
        if self.model is None:
            raise RuntimeError("Model is not loaded properly")
        
        window_start_time = time.time()
        print("Forcing window refresh to keep data in sync")
        env._pack_all_na_windows()
        window_fill_time = time.time() - window_start_time
        
        self.window_fill_times.append(window_fill_time)
        self.total_window_fill_time += window_fill_time
        
        updated_state = env.get_state()
        self.selected_nas = self._make_decision_with_current_state(updated_state, env)
        
        return self.selected_nas
    
    def _make_decision_with_current_state(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Make a decision with the provided state (used on the first decision)."""
        decision_start_time = time.time()
        
        try:
            if len(state.shape) == 1:
                n_na = env.config.n_na
                n_features = len(state) // n_na
                state = state.reshape(n_na, n_features)
            
            normalized_state = self._normalize_features(state)
            
            state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.model(state_tensor)
                q_values = q_values.squeeze(0)
            
            n_na_env = env.config.n_na
            n_na_model = q_values.shape[0]
            
            if n_na_model > n_na_env:
                q_values = q_values[:n_na_env]
            elif n_na_model < n_na_env:
                avg_q = q_values.mean()
                padding = torch.full((n_na_env - n_na_model,), avg_q, device=q_values.device)
                q_values = torch.cat([q_values, padding])
            
            selected_nas = self._select_by_reputation_groups(normalized_state, q_values, env)
            
            decision_time = time.time() - decision_start_time
            self.decision_times.append(decision_time)
            self.total_decision_time += decision_time
            self.decision_count += 1
            
            return selected_nas
            
        except Exception as e:
            print(f"DDQN model selection failed: {e}")
            decision_time = time.time() - decision_start_time
            self.decision_times.append(decision_time)
            self.total_decision_time += decision_time
            self.decision_count += 1
            
            return np.random.choice(env.config.n_na, env.config.selected_na_count, replace=False).tolist()
    

    

    
    def _normalize_features(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize features to match training-time preprocessing.
        
        Args:
            state: Raw state array [n_na, n_features] in the format
                [weighted_success_rates, weighted_delay_levels, normalized_reputations, hunger]
        
        Returns:
            normalized_state: Normalized state array
        """
        normalized_state = state.copy()
        
        env_normalized_reputation = state[:, 2]
        normalized_state[:, 0] = env_normalized_reputation
        
        normalized_state[:, 1] = state[:, 0]  # weighted_success_rate
        
        normalized_state[:, 2] = state[:, 1]  # weighted_delay_level
        
        normalized_state[:, 3] = state[:, 3]  # hunger
        
        return normalized_state
    
    def _select_by_reputation_groups(self, normalized_state: np.ndarray, q_values: torch.Tensor, env) -> List[int]:
        """
        Select NA by reputation groups: 3 from high, 2 from low.
        
        Args:
            normalized_state: Normalized state array [n_na, n_features]
            q_values: Q-values tensor [n_na]
            env: Environment instance
        
        Returns:
            selected_nas: Selected NA indices
        """
        reputation_threshold = 0.514
        
        reputations = normalized_state[:, 0]
        
        high_rep_mask = reputations >= reputation_threshold
        low_rep_mask = reputations < reputation_threshold
        
        high_rep_indices = np.where(high_rep_mask)[0]
        low_rep_indices = np.where(low_rep_mask)[0]
        
        print(f"Reputation group split: high {len(high_rep_indices)}, low {len(low_rep_indices)}")
        
        selected_nas = []
        
        target_high = 3
        target_low = 2
        
        if len(high_rep_indices) > 0:
            high_q_values = q_values[high_rep_indices]
            num_to_select = min(target_high, len(high_rep_indices))
            _, top_high_indices = torch.topk(high_q_values, num_to_select, largest=True)
            selected_high = [high_rep_indices[i] for i in top_high_indices.cpu().numpy()]
            selected_nas.extend(selected_high)
            print(f"   Selected {len(selected_high)} from high group: {selected_high}")
        
        if len(low_rep_indices) > 0:
            low_q_values = q_values[low_rep_indices]
            num_to_select = min(target_low, len(low_rep_indices))
            _, top_low_indices = torch.topk(low_q_values, num_to_select, largest=True)
            selected_low = [low_rep_indices[i] for i in top_low_indices.cpu().numpy()]
            selected_nas.extend(selected_low)
            print(f"   Selected {len(selected_low)} from low group: {selected_low}")
        
        total_needed = env.config.selected_na_count
        if len(selected_nas) < total_needed:
            remaining_indices = [i for i in range(len(q_values)) if i not in selected_nas]
            if remaining_indices:
                remaining_q_values = q_values[remaining_indices]
                num_to_add = min(total_needed - len(selected_nas), len(remaining_indices))
                _, top_remaining_indices = torch.topk(remaining_q_values, num_to_add, largest=True)
                additional_selected = [remaining_indices[i] for i in top_remaining_indices.cpu().numpy()]
                selected_nas.extend(additional_selected)
                print(f"   Added {len(additional_selected)} more: {additional_selected}")
        
        if len(selected_nas) > total_needed:
            selected_q_values = q_values[selected_nas]
            _, top_indices = torch.topk(selected_q_values, total_needed, largest=True)
            selected_nas = [selected_nas[i] for i in top_indices.cpu().numpy()]
            print(f"   Adjusted final selection to {len(selected_nas)}: {selected_nas}")
        
        return [int(na_idx) for na_idx in selected_nas]
    
    def _select_by_reputation_groups(self, normalized_state: np.ndarray, q_values: torch.Tensor, env) -> List[int]:
        """
        Select NA by reputation groups: 3 from high, 2 from low.
        
        Args:
            normalized_state: Normalized state [n_na, 4], where column 0 is reputation
            q_values: Model Q-values [n_na]
            env: Environment instance
        
        Returns:
            selected_nas: Selected NA indices
        """
        reputation_threshold = (6600 - 3000) / (10000 - 3000)  # ≈ 0.514
        
        current_reputations = normalized_state[:, 0]
        high_rep_mask = current_reputations >= reputation_threshold
        low_rep_mask = current_reputations < reputation_threshold
        
        high_rep_indices = np.where(high_rep_mask)[0]
        low_rep_indices = np.where(low_rep_mask)[0]
        
        print(
            f"Reputation group split: high {len(high_rep_indices)} NA, "
            f"low {len(low_rep_indices)} NA"
        )
        print(f"   High group NA: {high_rep_indices.tolist()}")
        print(f"   Low group NA: {low_rep_indices.tolist()}")
        
        selected_nas = []
        
        target_high_count = 3
        target_low_count = 2
        total_target = target_high_count + target_low_count
        
        if len(high_rep_indices) > 0:
            high_q_values = q_values[high_rep_indices]
            actual_high_count = min(target_high_count, len(high_rep_indices))
            _, high_top_indices = torch.topk(high_q_values, actual_high_count, largest=True)
            selected_high = [high_rep_indices[i] for i in high_top_indices.cpu().numpy()]
            selected_nas.extend(selected_high)
            print(f"Selected {len(selected_high)} NA from high group: {selected_high}")
        
        if len(low_rep_indices) > 0:
            low_q_values = q_values[low_rep_indices]
            actual_low_count = min(target_low_count, len(low_rep_indices))
            _, low_top_indices = torch.topk(low_q_values, actual_low_count, largest=True)
            selected_low = [low_rep_indices[i] for i in low_top_indices.cpu().numpy()]
            selected_nas.extend(selected_low)
            print(f"Selected {len(selected_low)} NA from low group: {selected_low}")
        
        if len(selected_nas) < total_target:
            remaining_count = total_target - len(selected_nas)
            print(f"Need to add {remaining_count} more NA")
            
            all_indices = set(range(len(q_values)))
            selected_set = set(selected_nas)
            remaining_indices = list(all_indices - selected_set)
            
            if len(remaining_indices) > 0:
                remaining_q_values = q_values[remaining_indices]
                actual_remaining_count = min(remaining_count, len(remaining_indices))
                _, remaining_top_indices = torch.topk(remaining_q_values, actual_remaining_count, largest=True)
                additional_selected = [remaining_indices[i] for i in remaining_top_indices.cpu().numpy()]
                selected_nas.extend(additional_selected)
                print(f"Added {len(additional_selected)} NA: {additional_selected}")
        
        print(f"Final selected NA: {selected_nas} (total: {len(selected_nas)})")
        
        return selected_nas
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        info = {
            'strategy_name': self.name,
            'model_path': self.model_path,
            'device': str(self.device),
            'model_loaded': self.model is not None
        }
        
        if self.model_info:
            info.update(self.model_info)
            
        return info
    
    def get_timing_statistics(self) -> Dict[str, Any]:
        """Return timing statistics."""
        if self.decision_count == 0:
            return {
                'decision_count': 0,
                'avg_window_fill_time_ms': 0.0,
                'avg_decision_time_ms': 0.0,
                'total_window_fill_time_ms': 0.0,
                'total_decision_time_ms': 0.0,
                'window_fill_times_ms': [],
                'decision_times_ms': []
            }
        
        return {
            'decision_count': self.decision_count,
            'avg_window_fill_time_ms': (self.total_window_fill_time / self.decision_count) * 1000,
            'avg_decision_time_ms': (self.total_decision_time / self.decision_count) * 1000,
            'total_window_fill_time_ms': self.total_window_fill_time * 1000,
            'total_decision_time_ms': self.total_decision_time * 1000,
            'window_fill_times_ms': [t * 1000 for t in self.window_fill_times],
            'decision_times_ms': [t * 1000 for t in self.decision_times]
        }
    
    def print_timing_summary(self):
        """Print timing statistics summary."""
        stats = self.get_timing_statistics()
        
        print("\n" + "="*60)
        print("DDQN timing statistics summary")
        print("="*60)
        print(f"Decision count: {stats['decision_count']}")
        print(f"Avg window fill time: {stats['avg_window_fill_time_ms']:.2f} ms")
        print(f"Avg decision time: {stats['avg_decision_time_ms']:.2f} ms")
        print(f"Total window fill time: {stats['total_window_fill_time_ms']:.2f} ms")
        print(f"Total decision time: {stats['total_decision_time_ms']:.2f} ms")
        
        if stats['decision_count'] > 0:
            print(f"\nFastest window fill: {min(stats['window_fill_times_ms']):.2f} ms")
            print(f"Slowest window fill: {max(stats['window_fill_times_ms']):.2f} ms")
            print(f"Fastest decision: {min(stats['decision_times_ms']):.2f} ms")
            print(f"Slowest decision: {max(stats['decision_times_ms']):.2f} ms")
        
        print("="*60)
    
    def reset_timing_statistics(self):
        """Reset timing statistics."""
        self.window_fill_times = []
        self.decision_times = []
        self.total_window_fill_time = 0.0
        self.total_decision_time = 0.0
        self.decision_count = 0
        print("Timing statistics reset.")


class RandomSelectionStrategy(NASelectionStrategy):
    """Random selection strategy."""
    
    def __init__(self):
        super().__init__("Random Selection")
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Randomly select NA."""
        return np.random.choice(env.config.n_na, env.config.selected_na_count, replace=False).tolist()


class RoundRobinStrategy(NASelectionStrategy):
    """Round-robin strategy using a fixed random permutation."""
    
    def __init__(self, seed: int = 42):
        super().__init__("Round Robin (Fixed Random Order)")
        self.seed = seed
        self.current_start = 0
        self.na_order = None
        
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Select consecutive NA by iterating a fixed permutation."""
        if self.na_order is None:
            np.random.seed(self.seed)
            self.na_order = np.random.permutation(env.config.n_na).tolist()
            print(f"Round Robin initialized; fixed order: {self.na_order[:10]}...")
        
        selected_nas = []
        for i in range(env.config.selected_na_count):
            idx = (self.current_start + i) % len(self.na_order)
            selected_nas.append(self.na_order[idx])
        
        self.current_start = (self.current_start + env.config.selected_na_count) % len(self.na_order)
        
        return selected_nas


class ReputationBasedStrategy(NASelectionStrategy):
    """Reputation-based selection strategy."""
    
    def __init__(self):
        super().__init__("Reputation Based")
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Select NA with the highest reputations."""
        reputation_scores = state[:, 2]
        return np.argsort(reputation_scores)[-env.config.selected_na_count:].tolist()


class BalancedReputationStrategy(NASelectionStrategy):
    """Grouped random selection: 3 high-reputation and 2 low-reputation NA."""
    
    def __init__(self, config: Optional[GlobalConfig] = None):
        super().__init__("Grouped Random Selection (3 high + 2 low)")
        if config is None:
            config = GlobalConfig()
        
        self.reputation_threshold = config.na_state.REPUTATION_THRESHOLD
        self.reputation_min = config.na_state.REPUTATION_MIN  
        self.reputation_max = config.na_state.REPUTATION_MAX
        self.high_count = config.strategy.HIGH_REPUTATION_COUNT
        self.low_count = config.strategy.LOW_REPUTATION_COUNT
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Randomly select NA with a 3-high + 2-low rule."""
        normalized_reputation = state[:, 2]
        
        normalized_threshold = (self.reputation_threshold - self.reputation_min) / (self.reputation_max - self.reputation_min)
        
        high_rep_mask = normalized_reputation >= normalized_threshold
        low_rep_mask = normalized_reputation < normalized_threshold
        
        high_rep_indices = np.where(high_rep_mask)[0]
        low_rep_indices = np.where(low_rep_mask)[0]
        
        print(
            f"Reputation group split: high {len(high_rep_indices)} NA, "
            f"low {len(low_rep_indices)} NA"
        )
        print(f"   High group NA: {high_rep_indices.tolist()}")
        print(f"   Low group NA: {low_rep_indices.tolist()}")
        
        selected_nas = []
        
        if len(high_rep_indices) > 0:
            num_high_to_select = min(self.high_count, len(high_rep_indices))
            selected_high = np.random.choice(high_rep_indices, size=num_high_to_select, replace=False)
            selected_nas.extend(selected_high.tolist())
            print(f"Selected {num_high_to_select} NA from high group: {selected_high.tolist()}")
        
        if len(low_rep_indices) > 0:
            num_low_to_select = min(self.low_count, len(low_rep_indices))
            selected_low = np.random.choice(low_rep_indices, size=num_low_to_select, replace=False)
            selected_nas.extend(selected_low.tolist())
            print(f"Selected {num_low_to_select} NA from low group: {selected_low.tolist()}")
        
        total_selected = len(selected_nas)
        if total_selected < env.config.selected_na_count:
            remaining_count = env.config.selected_na_count - total_selected
            all_indices = set(range(env.config.n_na))
            remaining_indices = list(all_indices - set(selected_nas))
            
            if len(remaining_indices) > 0:
                additional = np.random.choice(remaining_indices, 
                                            size=min(remaining_count, len(remaining_indices)), 
                                            replace=False)
                selected_nas.extend(additional.tolist())
                print(f"Added {len(additional)} NA: {additional.tolist()}")
        
        print(f"Final selected NA: {selected_nas} (total: {len(selected_nas)})")
        return selected_nas


class PartialSemiGreedyStrategy(NASelectionStrategy):
    """Partial semi-greedy strategy combining reward threshold and cost ranking."""

    def __init__(self, reward_threshold: float = 3.0, sample_limit: int = 7):
        super().__init__(f"Partial Semi-Greedy (thr:{reward_threshold})")
        self.reward_threshold = reward_threshold
        self.sample_limit = sample_limit

    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        success_rates = state[:, 0]
        delay_levels = state[:, 1]
        hunger_levels = state[:, 3]
        
        rewards = 2.0 * success_rates - 1.0 * delay_levels
        
        costs = 0.7 * delay_levels + 0.3 * hunger_levels

        candidate_indices = np.where(rewards > self.reward_threshold)[0]
        if candidate_indices.size == 0:
            candidate_indices = np.arange(env.config.n_na)

        ordered_candidates = candidate_indices[np.argsort(costs[candidate_indices])]

        selected = []
        primary_count = min(self.sample_limit, ordered_candidates.size)
        if primary_count > 0:
            draw_count = min(env.config.selected_na_count, primary_count)
            selected = np.random.choice(ordered_candidates[:primary_count], size=draw_count, replace=False).tolist()

        if len(selected) < env.config.selected_na_count:
            remaining_needed = env.config.selected_na_count - len(selected)
            remaining_pool = [idx for idx in ordered_candidates if idx not in selected]
            take = min(remaining_needed, len(remaining_pool))
            selected.extend(remaining_pool[:take])

        if len(selected) < env.config.selected_na_count:
            fallback_pool = [idx for idx in range(env.config.n_na) if idx not in selected]
            if fallback_pool:
                extra = np.random.choice(fallback_pool,
                                         size=env.config.selected_na_count - len(selected),
                                         replace=False)
                selected.extend(extra.tolist())

        return [int(idx) for idx in selected]


class HungerBasedStrategy(NASelectionStrategy):
    """Hunger-based selection strategy."""
    
    def __init__(self):
        super().__init__("Hunger Based")
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Select NA with the highest hunger values."""
        hunger_scores = state[:, 3]
        return np.argsort(hunger_scores)[-env.config.selected_na_count:].tolist()


class WeightedScoreStrategy(NASelectionStrategy):
    """Weighted score strategy using configurable weights."""
    
    def __init__(self, config: Optional[GlobalConfig] = None):
        if config is None:
            config = GlobalConfig()
        
        self.success_weight = config.strategy.SUCCESS_RATE_WEIGHT
        self.latency_weight = config.strategy.LATENCY_WEIGHT  
        self.reputation_weight = config.strategy.REPUTATION_WEIGHT
        self.hunger_weight = config.strategy.HUNGER_WEIGHT
        
        super().__init__(f"Weighted Score (sr:{self.success_weight}, lat:{self.latency_weight}, rep:{self.reputation_weight}, hun:{self.hunger_weight})")
        self.weights = np.array([self.success_weight, -self.latency_weight, self.reputation_weight, self.hunger_weight])
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Select NA based on a weighted score."""
        
        actual_features = np.column_stack([
            env.na_success_rates,
            env.na_delay_levels,
            state[:, 2],
            state[:, 3]
        ])
        
        state_normalized = (actual_features - actual_features.mean(axis=0)) / (actual_features.std(axis=0) + 1e-8)
        
        scores = np.dot(state_normalized, self.weights)
        
        return np.argsort(scores)[-env.config.selected_na_count:].tolist()


class GeneticAlgorithmStrategy(NASelectionStrategy):
    """Genetic algorithm strategy that evolves NA combinations."""

    def __init__(self,
                 population_size: int = 30,
                 generations: int = 25,
                 crossover_rate: float = 0.85,
                 mutation_rate: float = 0.25,
                 elite_fraction: float = 0.2,
                 success_weight: float = 0.45,
                 delay_weight: float = 0.30,
                 reputation_weight: float = 0.20,
                 hunger_weight: float = 0.05,
                 malicious_penalty: float = 0.6,
                 seed: Optional[int] = None):
        super().__init__("Genetic Algorithm")
        self.population_size = max(6, population_size)
        self.generations = max(1, generations)
        self.crossover_rate = np.clip(crossover_rate, 0.0, 1.0)
        self.mutation_rate = np.clip(mutation_rate, 0.0, 1.0)
        self.elite_fraction = np.clip(elite_fraction, 0.0, 0.5)
        self.success_weight = success_weight
        self.delay_weight = delay_weight
        self.reputation_weight = reputation_weight
        self.hunger_weight = hunger_weight
        self.risk_penalty = malicious_penalty
        self._rng = np.random.default_rng(seed)

    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        n_na = env.config.n_na
        select_k = env.config.selected_na_count
        candidate_indices = np.arange(n_na)

        success_rates = np.clip(env.na_success_rates, 0.0, 1.0)
        delay_levels = np.clip(env.na_delay_levels, 0.0, 1.0)
        reputations = np.clip(state[:, 2], 0.0, 1.0)
        hunger_levels = np.clip(state[:, 3], 0.0, 1.0)
        population = self._initialize_population(candidate_indices, success_rates, reputations, select_k)
        best_individual = population[0]
        best_score = -np.inf

        elite_count = max(1, int(self.population_size * self.elite_fraction))

        for _ in range(self.generations):
            scores = [self._fitness(ind, success_rates, delay_levels, reputations,
                                    hunger_levels, select_k)
                      for ind in population]
            current_best_idx = int(np.argmax(scores))
            if scores[current_best_idx] > best_score:
                best_score = scores[current_best_idx]
                best_individual = population[current_best_idx].copy()

            elite_indices = np.argsort(scores)[-elite_count:]
            elites = [population[idx].copy() for idx in elite_indices]

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, scores)
                parent2 = self._tournament_select(population, scores)
                child = self._crossover(parent1, parent2, candidate_indices, select_k)
                child = self._mutate(child, candidate_indices, select_k)
                new_population.append(child)

            population = new_population

        return [int(idx) for idx in self._repair_individual(best_individual, candidate_indices, select_k)]

    def _initialize_population(self, candidate_indices: np.ndarray, success_rates: np.ndarray,
                               reputations: np.ndarray, select_k: int) -> List[np.ndarray]:
        population: List[np.ndarray] = []

        if select_k <= len(candidate_indices):
            top_success = candidate_indices[np.argsort(success_rates)[-select_k:]]
            population.append(self._repair_individual(top_success, candidate_indices, select_k))

            top_reputation = candidate_indices[np.argsort(reputations)[-select_k:]]
            population.append(self._repair_individual(top_reputation, candidate_indices, select_k))

            hybrid_scores = 0.6 * success_rates + 0.4 * reputations
            top_hybrid = candidate_indices[np.argsort(hybrid_scores)[-select_k:]]
            population.append(self._repair_individual(top_hybrid, candidate_indices, select_k))

        while len(population) < self.population_size:
            individual = self._rng.choice(candidate_indices, size=select_k, replace=False)
            population.append(np.sort(individual))

        return population

    def _fitness(self, individual: np.ndarray, success_rates: np.ndarray, delay_levels: np.ndarray,
                 reputations: np.ndarray, hunger_levels: np.ndarray, select_k: int) -> float:
        sr = success_rates[individual].mean() if individual.size else 0.0
        delay = delay_levels[individual].mean() if individual.size else 1.0
        rep = reputations[individual].mean() if individual.size else 0.0
        hunger = hunger_levels[individual].mean() if individual.size else 0.0

        score = (
            self.success_weight * sr
            - self.delay_weight * delay
            + self.reputation_weight * rep
            + self.hunger_weight * hunger
        )

        if individual.size:
            avg_success = success_rates.mean()
            avg_delay = delay_levels.mean()
            avg_reputation = reputations.mean()
            avg_hunger = hunger_levels.mean()

            success_risk = max(0.0, avg_success - sr)
            delay_risk = max(0.0, delay - avg_delay)
            reputation_risk = max(0.0, avg_reputation - rep)
            hunger_risk = max(0.0, hunger - avg_hunger)

            behavior_risk = (
                0.5 * success_risk
                + 0.3 * delay_risk
                + 0.1 * reputation_risk
                + 0.1 * hunger_risk
            )

            score -= self.risk_penalty * behavior_risk

        return float(score)

    def _tournament_select(self, population: List[np.ndarray], scores: List[float], tournament_size: int = 3) -> np.ndarray:
        sampled_indices = random.sample(range(len(population)), k=min(tournament_size, len(population)))
        best_idx = max(sampled_indices, key=lambda idx: scores[idx])
        return population[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray, candidate_indices: np.ndarray,
                   select_k: int) -> np.ndarray:
        if self._rng.random() > self.crossover_rate:
            return parent1.copy()

        cut_point = self._rng.integers(1, select_k)
        child = list(parent1[:cut_point])
        for gene in parent2:
            if gene not in child:
                child.append(int(gene))
            if len(child) == select_k:
                break

        return self._repair_individual(np.array(child, dtype=int), candidate_indices, select_k)

    def _mutate(self, individual: np.ndarray, candidate_indices: np.ndarray, select_k: int) -> np.ndarray:
        if self._rng.random() > self.mutation_rate:
            return individual.copy()

        mutated = individual.copy()
        mutation_count = max(1, int(select_k * 0.3))
        positions = self._rng.choice(select_k, size=mutation_count, replace=False)
        available = [idx for idx in candidate_indices if idx not in mutated]

        for pos in positions:
            if not available:
                break
            replacement = self._rng.choice(available)
            available.remove(replacement)
            available.append(mutated[pos])
            mutated[pos] = replacement

        return self._repair_individual(mutated, candidate_indices, select_k)

    def _repair_individual(self, individual: np.ndarray, candidate_indices: np.ndarray, select_k: int) -> np.ndarray:
        unique_values = []
        seen = set()
        for idx in individual:
            value = int(idx)
            if value not in seen and value in candidate_indices:
                unique_values.append(value)
                seen.add(value)
            if len(unique_values) == select_k:
                break

        if len(unique_values) < select_k:
            remaining = [idx for idx in candidate_indices if idx not in seen]
            self._rng.shuffle(remaining)
            unique_values.extend(remaining[:select_k - len(unique_values)])

        return np.array(sorted(unique_values[:select_k]), dtype=int)


class AdaptiveStrategy(NASelectionStrategy):
    """Historical learning strategy based on performance history and multi-factor scoring."""
    
    def __init__(self, learning_rate: float = 0.1):
        super().__init__(f"Historical Learning Strategy (lr:{learning_rate})")
        self.learning_rate = learning_rate
        self.na_performance = None
        self.selection_history = []
        self.reward_history = []
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Adaptively select NA based on history and current state."""
        if self.na_performance is None:
            self.na_performance = np.ones(env.config.n_na)
        
        actual_success_rates = env.na_success_rates
        reputation = state[:, 2]
        delay_levels = state[:, 1]
        hunger = state[:, 3]
        
        inverted_delay = 1 - delay_levels
        
        current_scores = (0.3 * reputation + 
                         0.3 * actual_success_rates + 
                         0.2 * inverted_delay + 
                         0.2 * hunger)
        adaptive_scores = 0.7 * current_scores + 0.3 * self.na_performance
        
        selected = np.argsort(adaptive_scores)[-env.config.selected_na_count:].tolist()
        
        self.selection_history.append(selected)
        
        return selected
    
    def update_performance(self, reward: float):
        """Update NA performance estimates."""
        if len(self.selection_history) > 0:
            last_selected = self.selection_history[-1]
            
            for na_idx in last_selected:
                self.na_performance[na_idx] += self.learning_rate * (reward - self.na_performance[na_idx])
            
            self.reward_history.append(reward)


class EpsilonGreedyStrategy(NASelectionStrategy):
    """Epsilon-greedy strategy."""
    
    def __init__(self, epsilon: float = 0.1):
        super().__init__(f"Epsilon Greedy (ε:{epsilon})")
        self.epsilon = epsilon
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Select NA via epsilon-greedy exploration."""
        if np.random.random() < self.epsilon:
            return np.random.choice(env.config.n_na, env.config.selected_na_count, replace=False).tolist()
        else:
            reputation_scores = state[:, 2]
            return np.argsort(reputation_scores)[-env.config.selected_na_count:].tolist()


class MultiCriteriaStrategy(NASelectionStrategy):
    """Multi-criteria decision strategy using TOPSIS."""
    
    def __init__(self):
        super().__init__("Multi-Criteria (TOPSIS)")
    
    def select(self, state: np.ndarray, env: NAEnvironment) -> List[int]:
        """Select NA using TOPSIS."""
        weights = np.array([0.3, 0.2, 0.3, 0.2])
        
        decision_matrix = state.copy()
        decision_matrix[:, 1] = 1 / (state[:, 1] + 1e-8)
        
        normalized_matrix = decision_matrix / np.sqrt(np.sum(decision_matrix**2, axis=0))
        
        weighted_matrix = normalized_matrix * weights
        
        ideal_solution = np.max(weighted_matrix, axis=0)
        negative_ideal = np.min(weighted_matrix, axis=0)
        
        distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution)**2, axis=1))
        distance_to_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal)**2, axis=1))
        
        topsis_scores = distance_to_negative / (distance_to_ideal + distance_to_negative + 1e-8)
        
        return np.argsort(topsis_scores)[-env.config.selected_na_count:].tolist()


class ModelEvaluator:
    """Evaluator for comparing strategy performance."""
    
    def __init__(self):
        self.results_history = []
    
    def evaluate_strategies(self, strategies: List[NASelectionStrategy], 
                          env: NAEnvironment, n_episodes: int = 10, 
                          n_steps_per_episode: int = 100) -> Dict[str, Any]:
        """
        Evaluate multiple strategies.
        
        Args:
            strategies: List of strategies to evaluate
            env: Environment instance
            n_episodes: Number of evaluation episodes
            n_steps_per_episode: Steps per episode
            
        Returns:
            Dict[str, Any]: Evaluation summary
        """
        results = {}
        
        for strategy in strategies:
            print(f"\nEvaluating strategy: {strategy.name}")
            
            episode_results = []
            
            for episode in range(n_episodes):
                state = env.reset()
                
                result = strategy.evaluate(env, n_steps_per_episode)
                episode_results.append(result)
                
                if (episode + 1) % 5 == 0:
                    avg_reward = np.mean([r['total_reward'] for r in episode_results[-5:]])
                    print(f"  Episode {episode + 1}/{n_episodes}, last-5 mean reward: {avg_reward:.2f}")
            
            strategy_stats = self._compute_strategy_statistics(episode_results)
            strategy_stats['strategy_name'] = strategy.name
            results[strategy.name] = strategy_stats
            
            print(f"{strategy.name} evaluation completed")
            print(
                f"   Mean reward: {strategy_stats['mean_total_reward']:.2f} "
                f"± {strategy_stats['std_total_reward']:.2f}"
            )
            print(f"   Mean success rate: {strategy_stats['mean_success_rate']:.3f}")
        
        evaluation_summary = {
            'timestamp': time.time(),
            'n_episodes': n_episodes,
            'n_steps_per_episode': n_steps_per_episode,
            'strategies_evaluated': len(strategies),
            'results': results
        }
        
        self.results_history.append(evaluation_summary)
        
        return evaluation_summary
    
    def _compute_strategy_statistics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute strategy statistics."""
        metrics = ['total_reward', 'avg_reward_per_step', 'avg_success_rate', 
                  'avg_latency', 'steps_completed', 'evaluation_time']
        
        stats = {}
        for metric in metrics:
            values = [result[metric] for result in episode_results]
            stats[f'mean_{metric}'] = np.mean(values)
            stats[f'std_{metric}'] = np.std(values)
            stats[f'min_{metric}'] = np.min(values)
            stats[f'max_{metric}'] = np.max(values)
        
        return stats


def create_all_baseline_strategies() -> List[NASelectionStrategy]:
    """Create all baseline strategies."""
    strategies = [
        RandomSelectionStrategy(),
        RoundRobinStrategy(),
        ReputationBasedStrategy(),
        BalancedReputationStrategy(),
        PartialSemiGreedyStrategy(),
        HungerBasedStrategy(),
        WeightedScoreStrategy(),
        AdaptiveStrategy(),
        EpsilonGreedyStrategy(),
        MultiCriteriaStrategy(),
        DDQNModelStrategy(),
        GeneticAlgorithmStrategy()
    ]
    
    return strategies


def run_comprehensive_evaluation():
    """Run comprehensive evaluation."""
    print("Starting comprehensive baseline model evaluation")
    
    env = create_baseline_environment()
    
    strategies = create_all_baseline_strategies()
    
    evaluator = ModelEvaluator()
    
    results = evaluator.evaluate_strategies(strategies, env, n_episodes=5, n_steps_per_episode=50)
    
    print("\nEvaluation results summary:")
    for strategy_name, stats in results['results'].items():
        print(f"{strategy_name}: mean reward {stats['mean_total_reward']:.2f}, success rate {stats['mean_success_rate']:.3f}")


def test_single_strategy():
    """Test a single strategy."""
    env = create_baseline_environment()
    
    ddqn_strategy = DDQNModelStrategy()
    print(f"\nTesting strategy: {ddqn_strategy.name}")
    print("Model info:", ddqn_strategy.get_model_info())
    
    result = ddqn_strategy.evaluate(env, n_steps=20)
    print(f"Evaluation result: {result}")


if __name__ == "__main__":
    test_single_strategy()
    
    # run_comprehensive_evaluation()
