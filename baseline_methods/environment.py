"""
Baseline NA Selection Environment

This module implements an independent baseline environment for NA transaction simulation,
designed to test different NA selection strategies. It provides a clean interface for
model.py integration and implements comprehensive transaction execution with weighted
success rate calculations, reputation updates, and hunger management.

Key Features:
- Transaction simulation with 50 parallel transactions per step
- 5 NA selection per step (3 high reputation, 2 low reputation)
- Weighted success rate calculations using sliding window queues
- Reputation-based reward system
- Hunger management with logarithmic growth
- Malicious NA simulation with different behavior patterns

Reference Implementation: src/test.py
"""

import numpy as np
import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
from parameter import GlobalConfig


@dataclass
class NAConfig:
    """NA environment configuration."""
    def __init__(self, n_na: int = 10, max_steps: int = 100, selected_na_count: int = 5,
                 high_reputation_count: int = 3, low_reputation_count: int = 2,
                 transactions_per_step: int = 1, window_size: int = 50,
                 reputation_min: float = 3300.0, reputation_max: float = 10000.0,
                 reputation_threshold: float = 6600.0,
                 # New configuration parameters
                 na_init_mode: str = 'two_groups',  # 'random' or 'two_groups'
                 uniform_success_rate: float = 0.85,  # Uniform initial success rate
                 delay_level_range: Tuple[float, float] = (0.05, 0.3),  # Delay level range
                 malicious_attack_indices: Optional[List[int]] = None,  # Malicious attacker NA indices
                 malicious_attack_enabled: bool = True,  # Whether to enable malicious attackers
                 malicious_attack_type: str = 'OSA',  # Malicious attack type
                 # Hunger-related parameters
                 hunger_growth_scale: float = 10.0,  # Hunger growth scale
                 hunger_growth_log_base: float = 11.0,  # Log base
                 # Sliding window parameters (see test.py)
                 window_pack_interval: int = 5,  # Pack every 5 steps
                 window_queue_size: int = 20,    # Queue capacity: 20 packs
                 # Legacy parameters (backward compatibility)
                 normal_success_rate_range: Tuple[float, float] = (0.7, 0.9),
                 malicious_success_rate_range: Tuple[float, float] = (0.1, 0.3),
                 normal_latency_range: Tuple[float, float] = (0.1, 0.5),
                 malicious_latency_range: Tuple[float, float] = (0.8, 1.0),
                 malicious_ratio: float = 0.3,
                 random_seed: Optional[int] = 42):
        self.n_na = n_na
        self.max_steps = max_steps
        self.selected_na_count = selected_na_count
        self.high_reputation_count = high_reputation_count
        self.low_reputation_count = low_reputation_count
        self.transactions_per_step = transactions_per_step
        self.window_size = window_size
        self.reputation_min = reputation_min
        self.reputation_max = reputation_max
        self.reputation_threshold = reputation_threshold
        # New parameters
        self.na_init_mode = na_init_mode
        self.uniform_success_rate = uniform_success_rate
        self.delay_level_range = delay_level_range
        self.malicious_attack_indices = malicious_attack_indices or []
        self.malicious_attack_enabled = malicious_attack_enabled
        self.malicious_attack_type = malicious_attack_type
        self.hunger_growth_scale = hunger_growth_scale
        self.hunger_growth_log_base = hunger_growth_log_base
        # Sliding window parameters
        self.window_pack_interval = window_pack_interval
        self.window_queue_size = window_queue_size
        # Legacy parameters (backward compatibility)
        self.normal_success_rate_range = normal_success_rate_range
        self.malicious_success_rate_range = malicious_success_rate_range
        self.normal_latency_range = normal_latency_range
        self.malicious_latency_range = malicious_latency_range
        self.malicious_ratio = malicious_ratio
        self.random_seed = random_seed


@dataclass
class TransactionData:
    """Transaction data structure."""
    na_id: int
    success: bool
    latency: float
    timestamp: float
    reputation_before: float
    reputation_after: float
    hunger_level: float
    reward: float


class NAEnvironment:
    """
    Baseline NA selection environment.

    Implements independent baseline methods to test different NA selection strategies.
    Provides a clean interface for model.py integration and implements full transaction
    simulation based on src/test.py.
    """
    
    def __init__(self, config: Optional[NAConfig] = None):
        """
        Initialize the environment.

        Args:
            config: Environment configuration; defaults are used when None
        """
        self.config = config or NAConfig()
        
        seed = getattr(self.config, "random_seed", 42)
        if seed is not None:
            np.random.seed(int(seed))
            random.seed(int(seed))
        
        # Initialize NA attributes
        self._initialize_nas()
        
        # Initialize sliding window queues
        self._initialize_sliding_windows()
        
        # Initialize statistics
        self._initialize_stats()
        
        # Environment state
        self.current_step = 0
        self.start_time = time.time()
        
        # Current strategy (used to determine data collection mode)
        self.current_strategy = None
        
        # print(f"Baseline environment initialized: {self.config.n_na} NAs, "
        #       f"mode: {self.config.na_init_mode}, "
        #       f"malicious attackers: {len(getattr(self, 'malicious_attack_nas', set()))} ({self.config.malicious_attack_type})")
    
    def _initialize_nas(self):
        """Initialize NA attributes."""
        # Choose initialization mode
        if self.config.na_init_mode == 'two_groups':
            self._initialize_nas_two_groups()
        else:
            self._initialize_nas_random()
        
        # Initialize hunger
        self.na_hunger = np.random.uniform(0.0, 0.5, self.config.n_na)
        
        # Record selection time (used for hunger updates)
        self.last_selected_time = np.full(self.config.n_na, -1.0, dtype=float)
    
    def _initialize_nas_two_groups(self):
        """Initialize NAs as two reputation groups (as in test_custom_dataset.py)."""
        # print("Initializing NAs with the two-group mode")
        # Compute group sizes (ceil half for high group; ensure at least the high-reputation requirement)
        high_group_size = max(self.config.high_reputation_count, (self.config.n_na + 1) // 2)
        high_group_size = min(high_group_size, self.config.n_na)
        low_group_size = self.config.n_na - high_group_size

        # Store grouping info for external use
        self.high_group_size = high_group_size
        self.low_group_size = low_group_size

        # Initialize reputations for high/low groups
        high_rep_values = np.random.uniform(6600, 10000, high_group_size)
        low_rep_values = np.random.uniform(3300, 6600, max(low_group_size, 0)) if low_group_size > 0 else np.array([])
        self.na_reputations = np.concatenate([high_rep_values, low_rep_values])

        # Assign success rates (high group 0.95→0.75, low group 0.90→0.70)
        high_success_rates = np.linspace(0.95, 0.75, high_group_size)
        low_success_rates = np.linspace(0.90, 0.70, low_group_size) if low_group_size > 0 else np.array([])
        self.na_success_rates = np.concatenate([high_success_rates, low_success_rates])
        self.na_initial_success_rates = self.na_success_rates.copy()

        # Initialize dynamic success-rate statistics
        self.na_total_transactions = np.zeros(self.config.n_na, dtype=int)
        self.na_successful_transactions = np.zeros(self.config.n_na, dtype=int)

        # Assign delay levels (uniformly across 0.1~0.5 for both groups)
        high_delay_levels = np.linspace(0.1, 0.5, high_group_size)
        low_delay_levels = np.linspace(0.1, 0.5, low_group_size) if low_group_size > 0 else np.array([])
        self.na_delay_levels = np.concatenate([high_delay_levels, low_delay_levels])

        # Malicious attacker setup: pick the highest-reputation NA from each group
        if self.config.malicious_attack_enabled and not self.config.malicious_attack_indices:
            malicious_candidates = []
            if high_group_size > 0:
                high_indices = range(high_group_size)
                high_max_na = max(high_indices, key=lambda idx: self.na_reputations[idx])
                malicious_candidates.append(high_max_na)
            if low_group_size > 0:
                low_indices = range(high_group_size, self.config.n_na)
                low_max_na = max(low_indices, key=lambda idx: self.na_reputations[idx])
                malicious_candidates.append(low_max_na)
            self.malicious_attack_nas = set(malicious_candidates)
        else:
            self.malicious_attack_nas = set(self.config.malicious_attack_indices) if self.config.malicious_attack_enabled else set()

        # Keep malicious_nas for backward compatibility
        self.malicious_nas = self.malicious_attack_nas

        # Initialize transaction counters (used for attack-state tracking)
        self.na_transaction_counters = np.zeros(self.config.n_na, dtype=int)

        # print(f"High-reputation group size: {high_group_size}, range: {high_rep_values.min():.0f}-{high_rep_values.max():.0f}")
        # if low_group_size > 0:
        #     print(f"Low-reputation group size: {low_group_size}, range: {low_rep_values.min():.0f}-{low_rep_values.max():.0f}")
        # print(f"Success rate assignment: high({high_success_rates.min():.2f}-{high_success_rates.max():.2f})", end='')
        # if low_group_size > 0:
        #     print(f", low({low_success_rates.min():.2f}-{low_success_rates.max():.2f})")
        # else:
        #     print()
        # print(f"Delay level assignment: high({high_delay_levels.min():.2f}-{high_delay_levels.max():.2f})", end='')
        # if low_group_size > 0:
        #     print(f", low({low_delay_levels.min():.2f}-{low_delay_levels.max():.2f})")
        # else:
        #     print()
        # print(f"Malicious attackers: {sorted(self.malicious_attack_nas)}")
    
    def _initialize_nas_random(self):
        """Legacy random initialization (backward compatibility)."""
        # print("Initializing NAs with the random mode")
        
        # Determine malicious NAs
        n_malicious = int(self.config.n_na * self.config.malicious_ratio)
        self.malicious_nas = set(np.random.choice(self.config.n_na, n_malicious, replace=False))
        self.osa_attack_nas = self.malicious_nas  # Backward compatibility
        
        # Initialize NA attribute arrays
        self.na_success_rates = np.zeros(self.config.n_na)
        self.na_delay_levels = np.zeros(self.config.n_na)
        self.na_reputations = np.zeros(self.config.n_na)
        
        # Set NA attributes
        for i in range(self.config.n_na):
            if i in self.malicious_nas:
                # Malicious NA: low success rate, high latency
                self.na_success_rates[i] = np.random.uniform(*self.config.malicious_success_rate_range)
                latency = np.random.uniform(*self.config.malicious_latency_range)
                self.na_delay_levels[i] = latency / 2.0  # Normalize to 0-1
            else:
                # Normal NA: high success rate, low latency
                self.na_success_rates[i] = np.random.uniform(*self.config.normal_success_rate_range)
                latency = np.random.uniform(*self.config.normal_latency_range)
                self.na_delay_levels[i] = latency / 2.0  # Normalize to 0-1
            
            # Random initial reputation (3300-10000)
            self.na_reputations[i] = np.random.uniform(self.config.reputation_min, self.config.reputation_max)
    
    def _initialize_sliding_windows(self):
        """Initialize sliding window queues (see test.py)."""
        # Each NA has an independent queue storing up to window_queue_size packs
        self.na_window_queues = {}
        
        # Currently accumulating transaction pack per NA
        self.na_current_pack = {}
        
        # Step counter (used for packing logic)
        self.current_step_count = 0
        
        # Initialize per-NA data structures
        for na_id in range(self.config.n_na):
            self.na_window_queues[na_id] = deque(maxlen=self.config.window_queue_size)
            self.na_current_pack[na_id] = {
                'start_step': 0,
                'end_step': 0,
                'total_count': 0,
                'success_count': 0,
                'transactions': []
            }
        
        # Keep legacy data structures for compatibility
        self.na_transaction_data = [[] for _ in range(self.config.n_na)]
    
    def _initialize_stats(self):
        """Initialize statistics."""
        self.transaction_history = []
        self.reputation_history = []
        self.performance_stats = {
            'step_rewards': [],
            'success_rates': [],
            'avg_latencies': [],
            'hunger_levels': []
        }
    
    def reset(self) -> np.ndarray:
        """Reset the environment to its initial state."""
        # Re-initialize hunger
        for i in range(self.config.n_na):
            self.na_hunger[i] = np.random.uniform(0.0, 0.5)
        
        # Reset step counters and time records
        self.current_step = 0
        self.current_step_count = 0
        self.last_selected_time.fill(-1)
        
        # Reset transaction counters
        self.na_transaction_counters.fill(0)
        
        # Clear history
        self.transaction_history.clear()
        self.reputation_history.clear()
        for key in self.performance_stats:
            self.performance_stats[key].clear()
        
        # Clear sliding windows
        for na_id in range(self.config.n_na):
            self.na_window_queues[na_id].clear()
            self.na_current_pack[na_id] = {
                'start_step': 0,
                'end_step': 0,
                'total_count': 0,
                'success_count': 0,
                'transactions': []
            }
        for data_list in self.na_transaction_data:
            data_list.clear()
        
        return self.get_state()
    
    def _get_malicious_attack_success_rate(self, na_id: int) -> float:
        """
        Get the malicious NA success rate based on current step and attack type.
        
        Args:
            na_id: NA index
            
        Returns:
            float: Current success rate
        """
        if na_id not in self.malicious_attack_nas or not self.config.malicious_attack_enabled:
            # Normal NA: return fixed initial success rate
            return self.na_initial_success_rates[na_id]
        
        # Read initial success rate
        initial_rate = self.na_initial_success_rates[na_id]
        attack_type = self.config.malicious_attack_type
        current_step = self.current_step
        
        if attack_type == 'ME':
            # ME: decreases by 1% each step, floored at 0%
            decline = current_step * 0.01  # 1% per step
            return max(0.0, initial_rate - decline)
            
        elif attack_type == 'OSA':
            # OSA: toggles every 10 steps (low=0, high=initial_rate)
            cycle_position = current_step % 20  # 20 steps per full cycle
            if cycle_position < 10:
                return 0.0  # First 10 steps: low
            else:
                return initial_rate  # Last 10 steps: high
                
        elif attack_type == 'OOA':
            # OOA: toggles between initial_rate and 0 every step
            if current_step % 2 == 0:
                return initial_rate  # Even steps: high
            else:
                return 0.0  # Odd steps: low
                
        else:
            # Unknown attack type: use initial success rate
            return initial_rate

    def _pack_all_na_windows(self):
        """
        Pack current transaction data for all NAs (see test.py).

        Called every 5 steps to store the accumulated transactions into the per-NA queues.
        """
        for na_id in range(self.config.n_na):
            current_pack = self.na_current_pack[na_id]
            
            # Pack if the current pack contains transactions
            if current_pack['total_count'] > 0:
                # Compute pack success rate
                success_rate = current_pack['success_count'] / current_pack['total_count']
                
                # Create pack summary
                pack_summary = {
                    'start_step': current_pack['start_step'],
                    'end_step': current_pack['end_step'],
                    'transaction_count': current_pack['total_count'],
                    'success_count': current_pack['success_count'],
                    'success_rate': success_rate,
                    'transactions': current_pack['transactions'].copy()
                }
                
                # Append to queue (oldest packs are evicted automatically)
                self.na_window_queues[na_id].append(pack_summary)
            
            # Reset current pack
            self.na_current_pack[na_id] = {
                'start_step': self.current_step_count,
                'end_step': self.current_step_count,
                'total_count': 0,
                'success_count': 0,
                'transactions': []
            }

    def _add_transaction_to_pack(self, na_id: int, success: bool, latency: float, 
                                delay_level: float, timestamp: float, reputation_after: float):
        """
        Add a transaction to the current pack (see test.py).
        
        Args:
            na_id: NA index
            success: Whether the transaction succeeded
            latency: Transaction latency
            delay_level: Delay level
            timestamp: Timestamp
            reputation_after: Reputation after the transaction
        """
        current_pack = self.na_current_pack[na_id]
        
        # If this is the first transaction, set the start step
        if current_pack['total_count'] == 0:
            current_pack['start_step'] = self.current_step_count
        
        # Update pack statistics
        current_pack['end_step'] = self.current_step_count
        current_pack['total_count'] += 1
        if success:
            current_pack['success_count'] += 1
        
        # Append transaction record
        transaction_record = {
            'success': success,
            'latency': latency,
            'delay_level': delay_level,
            'timestamp': timestamp,
            'reputation_after': reputation_after
        }
        current_pack['transactions'].append(transaction_record)

    def _apply_malicious_attack(self, na_id: int, transaction_index: int) -> bool:
        """
        Decide whether a transaction succeeds based on the malicious attack type.
        
        Args:
            na_id: NA index
            transaction_index: Index of this transaction among all transactions for the NA
            
        Returns:
            bool: Whether the transaction succeeds
        """
        # Read current success rate (fixed for normal NAs, dynamic for malicious NAs)
        current_success_rate = self._get_malicious_attack_success_rate(na_id)
        
        # Random trial based on success rate
        return np.random.random() < current_success_rate
    
    def _get_current_delay_level(self, na_id: int) -> float:
        """
        Get the current delay level for an NA (keeps the initial value unchanged).
        
        Args:
            na_id: NA index
            
        Returns:
            float: Current delay level (0.0-1.0)
        """
        # Delay level stays fixed for all NAs (including malicious ones)
        return self.na_delay_levels[na_id]

    def get_state(self) -> np.ndarray:
        """
        Get the current environment state.
        
        Returns:
            np.ndarray: State array with shape (n_na, 4): [success_rate, delay_level, reputation, hunger]
        """
        # Calculate weighted success rate and delay level (based on sliding windows)
        weighted_success_rates, weighted_delay_levels = self._calculate_weighted_metrics()
        
        # Normalize reputation to [0, 1]
        normalized_reputations = (self.na_reputations - self.config.reputation_min) / \
                                (self.config.reputation_max - self.config.reputation_min)
        
        # Build state matrix
        state = np.column_stack([
            weighted_success_rates,
            weighted_delay_levels,
            normalized_reputations,
            self.na_hunger
        ])
        
        return state
    
    def calculate_psg_metrics(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the reward and cost arrays required by the PSG strategy."""
        success_rates = np.clip(state[:, 0], 0.0, 1.0)
        delay_levels = np.clip(state[:, 1], 0.0, 1.0)
        normalized_reputations = np.clip(state[:, 2], 0.0, 1.0)

        latency_rate = 1.0 - delay_levels
        reward_coeff = 1.0 + 2.5 * np.exp(1.5 - delay_levels)
        rewards = reward_coeff * latency_rate + normalized_reputations

        cost_penalty = (1.0 - success_rates) * 0.5
        costs = delay_levels + cost_penalty

        return rewards, costs

    def _calculate_weighted_metrics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute weighted success rates and delay levels using sliding windows.

        Follows test.py's get_na_window_summary logic:
        - Pack-level weights: newer packs get higher weights
        - Includes the currently accumulating pack (highest weight)
        - Linearly increasing weights: weight = i + 1
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (weighted success rates, weighted delay levels in 0-1)
        """
        weighted_success_rates = np.zeros(self.config.n_na)
        weighted_delay_levels = np.zeros(self.config.n_na)
        
        for na_id in range(self.config.n_na):
            queue = self.na_window_queues[na_id]
            current_pack = self.na_current_pack[na_id]
            
            # Compute weighted success rate and delay level (newer packs have higher weights)
            weighted_success_rate = 0.0
            weighted_delay_level = 0.0
            total_weight = 0.0
            
            # Assign weights for packs in the queue (newer packs have higher weights)
            if queue:
                for i, pack in enumerate(queue):
                    weight = i + 1  # Weights increase starting from 1
                    
                    # Accumulate weighted success rate
                    weighted_success_rate += pack['success_rate'] * weight
                    
                    # Compute average delay level for the pack
                    if pack['transactions']:
                        pack_delay_levels = [t['delay_level'] for t in pack['transactions']]
                        pack_avg_delay = np.mean(pack_delay_levels)
                        weighted_delay_level += pack_avg_delay * weight
                    
                    total_weight += weight
            
            # Include the currently accumulating pack (highest weight)
            if current_pack['total_count'] > 0:
                current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
                # Current pack has the highest weight
                current_weight = len(queue) + 1
                weighted_success_rate += current_pack_success_rate * current_weight
                
                # Compute average delay level for the current pack
                if current_pack['transactions']:
                    current_delay_levels = [t['delay_level'] for t in current_pack['transactions']]
                    current_avg_delay = np.mean(current_delay_levels)
                    weighted_delay_level += current_avg_delay * current_weight
                
                total_weight += current_weight
            
            # Normalize weighted values
            if total_weight > 0:
                weighted_success_rates[na_id] = weighted_success_rate / total_weight
                weighted_delay_levels[na_id] = weighted_delay_level / total_weight
            else:
                # If there is no history, fall back to initial values
                weighted_success_rates[na_id] = self.na_success_rates[na_id]
                weighted_delay_levels[na_id] = self.na_delay_levels[na_id]
        
        return weighted_success_rates, weighted_delay_levels
    
    def step(self, action_indices: List[int]):
        """
        Execute one step of the environment (full NA transaction simulation).
        
        Args:
            action_indices: Selected NA index list (must be exactly selected_na_count)
            
        Returns:
            tuple: (state, reward, done, info)
        """
        # Validate the number of selected NAs
        if len(action_indices) != self.config.selected_na_count:
            raise ValueError(f"Must select exactly {self.config.selected_na_count} NAs, got {len(action_indices)}")
        
        # Disable reputation-balance validation to allow fully random selection
        # self._validate_reputation_balance(action_indices)
        
        # Execute parallel transactions
        transactions = self._execute_transactions(action_indices)
        
        # Update NA states
        self._update_na_states(transactions)
        
        # Update hunger
        self._update_hunger(action_indices)
        
        # Calculate reward
        total_reward = self._calculate_reward(transactions)
        
        # Update statistics
        self._update_stats(transactions, total_reward)
        
        # Update dynamic success-rate stats
        self._update_dynamic_success_rates()
        
        # Update step counters
        self.current_step += 1
        self.current_step_count += 1
        
        # Decide whether to pack in real time based on the strategy type
        # DDQN uses periodic packing; other strategies use real-time packing
        if not (self.current_strategy and hasattr(self.current_strategy, 'collect_transaction_data')):
            # Non-DDQN strategies: pack once every window_pack_interval steps
            if self.current_step_count % self.config.window_pack_interval == 0:
                self._pack_all_na_windows()
        
        # Check termination
        done = self.current_step >= self.config.max_steps
        
        # Build info dict
        info = {
            'transactions': transactions,
            'selected_nas': action_indices,
            'success_rate': np.mean([t.success for t in transactions]),
            'avg_latency': np.mean([t.latency for t in transactions]),
            'total_reward': total_reward,
            'malicious_selected': [idx for idx in action_indices if idx in self.malicious_nas]
        }
        
        return self.get_state(), total_reward, done, info
    
    def _validate_reputation_balance(self, action_indices: List[int]):
        """Validate the high/low reputation ratio among selected NAs."""
        # The first 50 steps are the data-fill stage; no ratio constraint required
        if self.current_step < 50:
            return
        
        # After step 50, the strategy handles reputation balancing automatically
        return
            
        # The code below is kept for reference but not executed
        high_rep_count = sum(1 for idx in action_indices 
                           if self.na_reputations[idx] >= self.config.reputation_threshold)
        low_rep_count = len(action_indices) - high_rep_count
        
        if high_rep_count != self.config.high_reputation_count or \
           low_rep_count != self.config.low_reputation_count:
            print(f"Warning: Expected {self.config.high_reputation_count} high-rep and "
                  f"{self.config.low_reputation_count} low-rep NAs, "
                  f"got {high_rep_count} high-rep and {low_rep_count} low-rep")
    
    def _execute_transactions(self, selected_na_indices: List[int]) -> List[TransactionData]:
        """
        Execute transaction simulation using the new malicious attack mechanism.
        
        Args:
            selected_na_indices: Selected NA indices
            
        Returns:
            List[TransactionData]: Transaction results
        """
        transactions = []
        current_time = time.time()
        
        for na_id in selected_na_indices:
            # Read current NA attributes
            reputation_before = self.na_reputations[na_id]
            hunger_level = self.na_hunger[na_id]
            
            # Execute the configured number of parallel transactions
            for transaction_idx in range(self.config.transactions_per_step):
                # Read current delay level (keeps initial value unchanged)
                current_delay_level = self._get_current_delay_level(na_id)
                
                # Decide transaction success based on attack type
                success = self._apply_malicious_attack(na_id, self.na_transaction_counters[na_id])
                
                # Convert delay level to actual latency in milliseconds
                actual_latency = current_delay_level * 2000
                
                # Force failure if delay is too high (delay_level >= 1.0 => >= 2000ms)
                if current_delay_level >= 1.0:
                    success = False
                
                # Update transaction counters
                self.na_transaction_counters[na_id] += 1
                
                # Update reputation
                reputation_after = self._update_reputation(na_id, success)
                
                # Create transaction record
                transaction = TransactionData(
                    na_id=na_id,
                    success=success,
                    latency=actual_latency,
                    timestamp=current_time,
                    reputation_before=reputation_before,
                    reputation_after=reputation_after,
                    hunger_level=hunger_level,
                    reward=0.0
                )
                
                transactions.append(transaction)
                
                # Update dynamic success-rate statistics
                self.na_total_transactions[na_id] += 1
                if success:
                    self.na_successful_transactions[na_id] += 1
                
                # Choose data collection mode based on the strategy type
                if self.current_strategy and hasattr(self.current_strategy, 'collect_transaction_data'):
                    # DDQN: collect data within the cycle without real-time packing
                    self.current_strategy.collect_transaction_data(
                        na_id, success, actual_latency, current_delay_level, current_time, reputation_after
                    )
                else:
                    # Other strategies: use real-time packing
                    self._add_transaction_to_pack(na_id, success, actual_latency, current_delay_level, current_time, reputation_after)
                
                # Keep legacy data structures for compatibility
                self.na_transaction_data[na_id].append({
                    'success': success,
                    'latency': actual_latency,
                    'delay_level': current_delay_level,
                    'timestamp': current_time,
                    'reputation_after': reputation_after
                })
                
                # Keep the sliding window size bounded
                if len(self.na_transaction_data[na_id]) > self.config.window_size:
                    self.na_transaction_data[na_id].pop(0)
        
        return transactions
    
    def _update_dynamic_success_rates(self):
        """
        Update dynamic success-rate statistics based on transaction outcomes.

        Note: normal NAs keep a fixed success rate; only malicious NAs vary.
        This method therefore does not update na_success_rates and is kept for statistics only.
        """
        # Dynamic success rate updates are disabled to keep normal NA rates fixed
        pass
    
    def _update_reputation(self, na_id: int, success: bool) -> float:
        """
        Update NA reputation.

        Follows src/test.py's reputation update logic:
        - Compute reputation delta based on computed_reward
        - computed_reward considers success rate, normalized reputation, and hunger
        - On success: reputation += computed_reward * 20
        - On failure: reputation -= computed_reward * 100
        
        Args:
            na_id: NA index
            success: Whether the transaction succeeded
            
        Returns:
            float: Updated reputation
        """
        old_reputation = self.na_reputations[na_id]
        
        # Use the environment's inherent success probability:
        # - If the NA is malicious and attacks are enabled: use the dynamic attack success rate
        # - Otherwise: use the NA's initial/inherent success rate
        if getattr(self, 'malicious_attack_nas', None) and na_id in self.malicious_attack_nas and self.config.malicious_attack_enabled:
            current_success_rate = self._get_malicious_attack_success_rate(na_id)
        else:
            current_success_rate = self.na_success_rates[na_id]
        
        # Normalize reputation
        norm_rep = (old_reputation - 3300.0) / (10000.0 - 3300.0)
        
        # Compute computed_reward (see src/test.py)
        if old_reputation < 6600:
            # Low-reputation NA: emphasize reputation
            computed_reward = (0.4 * current_success_rate +
                              0.4 * norm_rep +
                              0.2 * self.na_hunger[na_id])
        else:
            # High-reputation NA: emphasize hunger
            computed_reward = (0.4 * current_success_rate +
                              0.2 * norm_rep +
                              0.4 * self.na_hunger[na_id])
        
        # Update reputation based on transaction outcome
        if success:
            # Success: increase reputation (computed_reward * 20)
            reputation_increase = computed_reward * 20
            self.na_reputations[na_id] = min(10000, old_reputation + reputation_increase)
        else:
            # Failure: decrease reputation (computed_reward * 100)
            reputation_decrease = computed_reward * 100
            self.na_reputations[na_id] = max(3300, old_reputation - reputation_decrease)
        
        return self.na_reputations[na_id]
    
    def _calculate_transaction_reward(self, *args, **kwargs) -> float:
        """Deprecated: reward is computed at the step level; kept for compatibility."""
        return 0.0
    
    def _calculate_reward(self, transactions: List[TransactionData]) -> float:
        """Compute the total step reward (aligned with src/test.py)."""
        if not transactions:
            return 0.0

        transactions_by_na: Dict[int, List[TransactionData]] = defaultdict(list)
        for trx in transactions:
            transactions_by_na[trx.na_id].append(trx)

        weighted_success_rates, weighted_delay_levels = self._calculate_weighted_metrics()

        total_reward = 0.0

        for na_id, na_transactions in transactions_by_na.items():
            queue = self.na_window_queues.get(na_id, deque())
            current_pack = self.na_current_pack.get(na_id, None)
            has_history = len(queue) > 0
            if current_pack and current_pack.get('total_count', 0) > 0:
                has_history = True

            if has_history:
                effective_success_rate = np.clip(weighted_success_rates[na_id], 0.0, 1.0)
                effective_delay_level = np.clip(weighted_delay_levels[na_id], 0.0, 1.0)
            else:
                effective_success_rate = np.clip(self.na_success_rates[na_id], 0.0, 1.0)
                effective_delay_level = np.clip(self.na_delay_levels[na_id], 0.0, 1.0)

            window_transaction_count = sum(pack.get('transaction_count', 0) for pack in queue)
            if current_pack:
                window_transaction_count += current_pack.get('total_count', 0)

            if window_transaction_count < 5:
                total_transactions = self.na_total_transactions[na_id]
                if total_transactions > 0:
                    overall_success_rate = self.na_successful_transactions[na_id] / total_transactions
                    blended_total = max(window_transaction_count, 0)
                    smoothing_target = 5
                    effective_success_rate = (
                        effective_success_rate * blended_total +
                        overall_success_rate * (smoothing_target - blended_total)
                    ) / smoothing_target

            weighted_success_rate_reward = 4.0 * effective_success_rate - 2.8
            weighted_delay_grade_bonus = 0.8 * (1.0 - 2.0 * effective_delay_level)

            quality_weight = 0.6 * effective_success_rate + 0.4 * (1.0 - effective_delay_level)
            base_hunger_weight = 0.2 + 1.3 * (quality_weight ** 1.5)

            if quality_weight >= 0.8:
                dynamic_hunger_weight = min(1.8, base_hunger_weight * 1.2)
            elif quality_weight >= 0.6:
                dynamic_hunger_weight = base_hunger_weight
            elif quality_weight >= 0.4:
                dynamic_hunger_weight = base_hunger_weight * 0.8
            else:
                dynamic_hunger_weight = base_hunger_weight * 0.5

            effective_hunger = np.clip(self.na_hunger[na_id], 0.0, 1.0)
            hunger_bonus = dynamic_hunger_weight * quality_weight * effective_hunger

            na_reward = weighted_success_rate_reward + weighted_delay_grade_bonus + hunger_bonus

            if na_transactions:
                per_transaction_reward = na_reward / len(na_transactions)
                for trx in na_transactions:
                    trx.reward = per_transaction_reward

            total_reward += na_reward

        return total_reward
    
    def _update_na_states(self, transactions: List[TransactionData]):
        """Update NA states (reputation is updated during transaction execution)."""
        # Record transaction history
        self.transaction_history.extend(transactions)
        self.reputation_history.append(self.na_reputations.copy())
    
    def _update_hunger(self, selected_na_indices: List[int]):
        """
        Update hunger.

        Follows src/test.py's _update_hunger logic:
        - Selected NAs reset hunger to 0
        - Other NAs increase hunger using a logarithmic function
        
        Args:
            selected_na_indices: Selected NA indices
        """
        # Update selection records
        for na_id in selected_na_indices:
            self.last_selected_time[na_id] = self.current_step
        
        # Compute hunger for all NAs
        for i in range(self.config.n_na):
            steps_since_selected = self.current_step - self.last_selected_time[i]
            
            if steps_since_selected <= 0:
                self.na_hunger[i] = 0.0
            else:
                # Log function: hunger = log(1 + steps/scale) / log(base)
                normalized_steps = steps_since_selected / self.config.hunger_growth_scale
                self.na_hunger[i] = min(1.0, np.log(1 + normalized_steps) / np.log(self.config.hunger_growth_log_base))
        
        # Reset hunger to 0 for selected NAs
        for na_id in selected_na_indices:
            self.na_hunger[na_id] = 0.0
    
    def _update_stats(self, transactions: List[TransactionData], total_reward: float):
        """Update statistics."""
        self.performance_stats['step_rewards'].append(total_reward)
        self.performance_stats['success_rates'].append(
            np.mean([t.success for t in transactions])
        )
        self.performance_stats['avg_latencies'].append(
            np.mean([t.latency for t in transactions])
        )
        self.performance_stats['hunger_levels'].append(self.na_hunger.copy())
    
    def get_na_window_summary(self) -> Dict[str, np.ndarray]:
        """
        Get sliding-window summary statistics for all NAs.

        Fully based on test.py's get_na_window_summary, using pack-level weights.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing summary metrics
        """
        weighted_success_rates, weighted_delay_levels = self._calculate_weighted_metrics()
        
        return {
            'success_rates': weighted_success_rates,
            'avg_latencies': weighted_delay_levels * 2000.0,
            'reputations': self.na_reputations.copy(),
            'hunger_levels': self.na_hunger.copy(),
            'weighted_success_rates': weighted_success_rates,
            'weighted_delay_levels': weighted_delay_levels
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_stats['step_rewards']:
            return {}
        
        return {
            'total_steps': self.current_step,
            'avg_step_reward': np.mean(self.performance_stats['step_rewards']),
            'avg_success_rate': np.mean(self.performance_stats['success_rates']),
            'avg_latency': np.mean(self.performance_stats['avg_latencies']),
            'final_reputations': self.na_reputations.copy(),
            'final_hunger_levels': self.na_hunger.copy(),
            'malicious_nas': list(self.malicious_nas),
            'total_transactions': len(self.transaction_history)
        }
    
    # ============================================================================
    # Baseline selection strategy methods have been moved to strategy classes in models.py


# ============================================================================
# Convenience functions
# ============================================================================

def create_baseline_environment(config: Optional[NAConfig] = None) -> NAEnvironment:
    """Create a baseline environment instance."""
    return NAEnvironment(config)


def test_baseline_environment():
    """Test the baseline environment."""
    print("Starting baseline environment test...")
    
    # Create environment
    env = create_baseline_environment()
    
    # Reset environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Test different selection strategies
    strategies = [
        ("Random", env.select_random),
        ("Reputation", env.select_by_reputation),
        ("Balanced Reputation", env.select_balanced_reputation),
        ("Hunger", env.select_by_hunger),
        ("Weighted Score", env.select_by_weighted_score)
    ]
    
    for strategy_name, strategy_func in strategies:
        print(f"\nTesting strategy: {strategy_name}")
        
        # Select NAs
        selected_nas = strategy_func()
        print(f"Selected NAs: {selected_nas}")
        
        # Execute one step
        state, reward, done, info = env.step(selected_nas)
        print(f"Reward: {reward:.2f}, Success rate: {info['success_rate']:.2%}, Avg latency: {info['avg_latency']:.3f}")
        
        if done:
            break
    
    # Get performance summary
    summary = env.get_performance_summary()
    print("\nPerformance summary:")
    print(f"Total steps: {summary['total_steps']}")
    print(f"Average step reward: {summary['avg_step_reward']:.2f}")
    print(f"Average success rate: {summary['avg_success_rate']:.2%}")
    print(f"Average latency: {summary['avg_latency']:.3f}")
    print(f"Malicious NA count: {len(summary['malicious_nas'])}")
    
    print("Baseline environment test completed.")


if __name__ == "__main__":
    test_baseline_environment()
