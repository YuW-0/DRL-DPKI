#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing Custom Dataset with Trained DQN Model
=============================================

This script demonstrates how to:
1. Load trained DQN model
2. Prepare custom test data
3. Use model for prediction and evaluation
4. Analyze model performance

Feature descriptions:
- reputation: Reputation value (3000-10000)
- success_rate: Success rate (0.0-1.0)
- signature_delay: Delay level (0.0-1.0, 0%=fastest, 100%=slowest)
- hunger: Hunger level (0.0-1.0)
"""

import os
import pickle
import random
from visualize_results import visualize_transaction_evolution, visualize_predictions, get_unified_na_colors, reorder_na_legend

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import re

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Load custom fonts
current_dir = os.path.dirname(os.path.abspath(__file__))
font_dir = os.path.join(current_dir, '../font')
if os.path.exists(font_dir):
    font_files = fm.findSystemFonts(fontpaths=[font_dir])
    for font_file in font_files:
        try:
            fm.fontManager.addfont(font_file)
            # print(f"Loaded font: {font_file}")
        except Exception as e:
            print(f"Cannot load font {font_file}: {e}")
else:
    print(f"Warning: Font directory does not exist: {font_dir}")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Source Han Sans CN']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Allow numpy scalar objects when loading checkpoints with weights_only=True (PyTorch >= 2.6)
if hasattr(torch.serialization, "add_safe_globals"):
    safe_globals = []

    try:
        safe_globals.append(np.dtype)
    except AttributeError:
        pass

    for module_name in ("_core", "core"):
        np_module = getattr(np, module_name, None)
        multiarray = getattr(np_module, "multiarray", None)
        scalar_cls = getattr(multiarray, "scalar", None)
        if scalar_cls is not None:
            safe_globals.append(scalar_cls)
            break

    if safe_globals:
        try:
            torch.serialization.add_safe_globals(safe_globals)
        except Exception:
            pass
from collections import deque
import random

# Set random seeds for reproducible results
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)


def torch_load_compat(path, *, map_location=None, weights_only=None):
    """Load checkpoints while remaining compatible with different torch versions."""
    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    if weights_only is not None:
        load_kwargs["weights_only"] = weights_only

    try:
        return torch.load(path, **load_kwargs)
    except TypeError:
        load_kwargs.pop("weights_only", None)
        return torch.load(path, **load_kwargs)


def get_runtime_device(preferred_gpu_index=1):
    """Select runtime device, preferring a specific GPU index when available."""
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        if cuda_count > preferred_gpu_index:
            torch.cuda.set_device(preferred_gpu_index)
            return torch.device(f"cuda:{preferred_gpu_index}"), None
        torch.cuda.set_device(0)
        note = (f"Requested GPU index {preferred_gpu_index} unavailable; "
                f"using cuda:0 out of {cuda_count} visible GPU(s).")
        return torch.device("cuda:0"), note
    return torch.device("cpu"), "CUDA unavailable; using CPU instead."

# Define the network architecture (must match training exactly)
class QNetwork(nn.Module):
    def __init__(self, n_features, n_na):
        super().__init__()
        # NA feature encoder
        self.na_encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(32, 16)
        )
        
        # Q-value predictor
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x):
        batch_size = x.size(0)
        n_na = x.size(1)
        x_flat = x.view(batch_size * n_na, -1)
        na_embeddings = self.na_encoder(x_flat)
        q_values_flat = self.q_predictor(na_embeddings)
        q_values = q_values_flat.view(batch_size, n_na)
        return q_values

def load_trained_model(model_path, model_info_path, device):
    """
    Load a trained model.
    
    Args:
        model_path: Path to model weights
        model_info_path: Path to model configuration
        device: Runtime device
    
    Returns:
        model: Loaded model
        model_info: Model configuration
    """
    print("Loading trained model...")
    
    # Load model configuration
    model_info = torch_load_compat(model_info_path, map_location=device, weights_only=False)
    print(f"Model info: {model_info}")
    
    # Build network
    n_features = model_info['n_features']
    n_na = model_info['n_na']
    model = QNetwork(n_features, n_na).to(device)
    
    # Load weights
    if os.path.exists(model_path):
        if model_path.endswith('_state_dict.pth'):
            # Load state dict
            state_dict = torch_load_compat(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
        else:
            # Load full model
            model = torch_load_compat(model_path, map_location=device, weights_only=False)
        
        model.eval()  # Evaluation mode
        print(f"Model loaded successfully: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return model, model_info

def generate_na_features(n_na, na_type='random', attack_indices=None, attack_type=None):
    """
    Generate NA features.
    
    Args:
        n_na: NA count
        na_type: NA type ('random', 'two_groups': 5 high-rep + 5 low-rep with same initial success rate)
        attack_indices: Attack node indices (kept for compatibility)
        attack_type: Attack type (kept for compatibility)
    
    Returns:
        dict: Generated NA feature data
    """
    if na_type == 'two_groups' and n_na == 10:
        # 5 high-reputation NAs + 5 low-reputation NAs with identical initial success rate
        print("Generating two-group configuration: 5 high-rep NAs + 5 low-rep NAs with identical initial success rate")
        
        # Unified initial success rate
        uniform_success_rate = 0.85  # Unified initial success rate
        
        # High-reputation group (NA 0-4): reputation range 6600-10000
        high_rep_values = np.random.uniform(6600, 10000, 5)
        # Low-reputation group (NA 5-9): reputation range 3300-6600
        low_rep_values = np.random.uniform(3300, 6600, 5)
        all_rep_values = np.concatenate([high_rep_values, low_rep_values])
        
        # Ensure actual success rate is exactly identical for all NAs
        # Use a fixed total_tx to avoid integer-division differences
        total_tx = np.full(n_na, 100)  # Use 100 transactions for all NAs
        success_count = np.full(n_na, int(uniform_success_rate * 100))  # 85 successful transactions for all NAs
        actual_success_rates = np.full(n_na, uniform_success_rate)  # Directly set to unified success rate
        
        # Generate other features
        all_delay_levels = np.random.uniform(0.05, 0.3, n_na)
        hunger_values = np.random.uniform(0.1, 0.5, n_na)
        
        print(f"High-rep reputation range: {high_rep_values.min():.0f}-{high_rep_values.max():.0f}")
        print(f"Low-rep reputation range: {low_rep_values.min():.0f}-{low_rep_values.max():.0f}")
        print(f"Unified initial success rate: {uniform_success_rate:.2f}")
        print(f"Unified transaction count: {total_tx[0]}")
        print(f"Unified successful transaction count: {success_count[0]}")
        
    else:
        # Randomly generate all NA features (original logic)
        all_rep_values = np.random.uniform(6000, 10000, n_na)
        total_tx = np.random.randint(20, 81, n_na)
        initial_success_rates = np.random.uniform(0.7, 1.0, n_na)
        success_count = (initial_success_rates * total_tx).astype(int)
        actual_success_rates = success_count / total_tx
        all_delay_levels = np.random.uniform(0.05, 0.3, n_na)
        hunger_values = np.random.uniform(0.1, 0.5, n_na)
    
    return {
        'reputation': all_rep_values,
        'success_rate': actual_success_rates,
        'delay_level': all_delay_levels,
        'hunger': hunger_values,
        'total_tx': total_tx,
        'success_count': success_count
    }
        


def _get_na_window_summary(na_idx, na_window_queues):
    """
    Get sliding-window summary for a specific NA.
    
    Args:
        na_idx: NA index
        na_window_queues: global sliding window queues
        
    Returns:
        dict: summary containing weighted success rate and weighted delay grade
    """
    if na_idx not in na_window_queues or len(na_window_queues[na_idx]) == 0:
        # Return defaults when no window data exists
        return {
            'weighted_success_rate': 0.5,
            'weighted_delay_grade': 0.5
        }
    
    window_data = list(na_window_queues[na_idx])
    
    # Compute weighted success rate and weighted delay grade
    total_weight = 0
    weighted_success_sum = 0
    weighted_delay_sum = 0
    
    for i, pack_data in enumerate(window_data):
        # Use linear-decay weights; newest pack has highest weight
        weight = (i + 1) / len(window_data)
        total_weight += weight
        
        weighted_success_sum += pack_data['success'] * weight
        weighted_delay_sum += pack_data['delay_grade'] * weight
    
    if total_weight > 0:
        weighted_success_rate = weighted_success_sum / total_weight
        weighted_delay_grade = weighted_delay_sum / total_weight
    else:
        weighted_success_rate = 0.5
        weighted_delay_grade = 0.5
    
    return {
        'weighted_success_rate': weighted_success_rate,
        'weighted_delay_grade': weighted_delay_grade
    }

def normalize_features(data):
    """
    Feature normalization (must match the training-time normalization).
    
    Args:
        data: raw feature data [n_na, n_features]
    
    Returns:
        normalized_data: normalized feature data
    """
    normalized_data = data.copy()
    
    # Normalize each feature (consistent with training)
    normalized_data[:, 0] = (data[:, 0] - 3000) / (10000 - 3000)  # reputation: 3000-10000 -> 0-1
    normalized_data[:, 1] = data[:, 1]  # success_rate (already 0-1)
    normalized_data[:, 2] = data[:, 2]  # delay_level (already 0-1)
    normalized_data[:, 3] = data[:, 3]  # hunger (already 0-1)
    
    return normalized_data

def calculate_computed_reward(success_rate, reputation, hunger):
    """
    Compute computed_reward using the same formula as in test.py.
    
    Args:
        success_rate: success rate [0, 1]
        reputation: reputation value [3000, 10000]
        hunger: hunger level [0, 1]
    
    Returns:
        computed_reward: computed reward factor
    """
    # Normalize reputation
    norm_rep = (reputation - 3300.0) / (10000.0 - 3300.0)
    
    # Choose weights based on reputation
    if reputation < 6600:
        computed_reward = (0.4 * success_rate + 0.4 * norm_rep + 0.2 * hunger)
    else:
        computed_reward = (0.4 * success_rate + 0.2 * norm_rep + 0.4 * hunger)
    
    return computed_reward

def generate_attack_window_data(attack_type, current_features, window_size=5, pack_interval=5, 
                               window_strategy=None, current_time_step=None):
    """
    [DEPRECATED] Generate sliding-window history for attack scenarios following test.py reputation update rules.
    
    Note: this function is no longer used. All window data is maintained by the global sliding window system.
    This function is kept only for backward compatibility; prefer the global sliding window in practice.
    
    Args:
        attack_type: attack type ('ME', 'OOA', 'OSA', 'normal')
        current_features: current features [reputation, success_rate, delay_level, hunger]
        window_size: window size
        pack_interval: pack interval
        window_strategy: window strategy ('A', 'B', None)
                        A: improve during 0-25, degrade during 26-50
                        B: degrade during 0-25, improve during 26-50
                        None: use the original logic
        current_time_step: current time step (0-50) used to determine trend
    
    Returns:
        window_data: list of historical window data
    """
    print("WARN: generate_attack_window_data is deprecated; use the global sliding window system instead.")
    window_data = []
    
    # Base parameters
    reputation, success_rate, delay_level, hunger = current_features
    current_reputation = reputation  # Track current reputation
    
    for i in range(window_size):
        # Generate transaction count for this pack
        transaction_count = np.random.randint(5, 15)
        
        # If a window strategy and time step are provided, use the new strategy logic
        if window_strategy is not None and current_time_step is not None:
            # Progress coefficient
            if current_time_step <= 25:
                # Progress at 0-25 (0.0 -> 1.0)
                progress = current_time_step / 25.0
            else:
                # Progress at 26-50 (0.0 -> 1.0)
                progress = (current_time_step - 25) / 25.0
            
            # Base performance parameters
            base_success_rate = success_rate
            base_delay = delay_level
            
            if window_strategy == 'A':
                # Strategy A: improve at 0-25, degrade at 26-50
                if current_time_step <= 25:
                    # 0-25: gradually improve
                    # Success rate increases from base*0.6 to base*1.2
                    pack_success_rate = base_success_rate * (0.6 + 0.6 * progress)
                    # Delay decreases from base*1.5 to base*0.5
                    pack_delay = base_delay * (1.5 - 1.0 * progress)
                else:
                    # 26-50: gradually degrade
                    # Success rate decreases from base*1.2 to base*0.4
                    pack_success_rate = base_success_rate * (1.2 - 0.8 * progress)
                    # Delay increases from base*0.5 to base*1.8
                    pack_delay = base_delay * (0.5 + 1.3 * progress)
            
            elif window_strategy == 'B':
                # Strategy B: degrade at 0-25, improve at 26-50
                if current_time_step <= 25:
                    # 0-25: gradually degrade
                    # Success rate decreases from base*1.2 to base*0.4
                    pack_success_rate = base_success_rate * (1.2 - 0.8 * progress)
                    # Delay increases from base*0.5 to base*1.8
                    pack_delay = base_delay * (0.5 + 1.3 * progress)
                else:
                    # 26-50: gradually improve
                    # Success rate increases from base*0.4 to base*1.2
                    pack_success_rate = base_success_rate * (0.4 + 0.8 * progress)
                    # Delay decreases from base*1.8 to base*0.5
                    pack_delay = base_delay * (1.8 - 1.3 * progress)
            
            # Add random noise
            pack_success_rate += np.random.uniform(-0.05, 0.05)
            pack_delay += np.random.uniform(-0.02, 0.02)
        
        else:
            # Use the original attack-type logic
            if attack_type == 'ME':  # Malicious With Everyone - continuously malicious
                # Persistently low success rate, high delay
                pack_success_rate = max(0.1, 0.3 - i * 0.02)  # Gradually worsens
                pack_delay = min(1.0, 0.7 + i * 0.05)  # Delay increases
                
            elif attack_type == 'OOA':  # On-Off Attack
                # Periodically switch between honest and malicious behavior
                if i % 2 == 0:  # Honest phase
                    pack_success_rate = min(1.0, 0.8 + np.random.uniform(-0.1, 0.1))
                    pack_delay = max(0.0, 0.2 + np.random.uniform(-0.05, 0.05))
                else:  # Malicious phase
                    pack_success_rate = max(0.0, 0.2 + np.random.uniform(-0.1, 0.1))
                    pack_delay = min(1.0, 0.8 + np.random.uniform(-0.1, 0.1))
                    
            elif attack_type == 'OSA':  # Opportunistic Service Attack
                # Provide high-quality service when reputation is low; degrade after recovery
                current_rep_normalized = (current_reputation - 3000) / (10000 - 3000)
                if current_rep_normalized < 0.5:  # Perform well at low reputation
                    pack_success_rate = min(1.0, 0.9 + np.random.uniform(-0.05, 0.05))
                    pack_delay = max(0.0, 0.1 + np.random.uniform(-0.05, 0.05))
                else:  # Start malicious behavior at higher reputation
                    pack_success_rate = max(0.0, 0.4 + np.random.uniform(-0.1, 0.1))
                    pack_delay = min(1.0, 0.6 + np.random.uniform(-0.1, 0.1))
                    
            else:  # normal
                # Stable good performance with small fluctuations
                pack_success_rate = min(1.0, success_rate + np.random.uniform(-0.1, 0.1))
                pack_delay = max(0.0, delay_level + np.random.uniform(-0.05, 0.05))
        
        # Ensure success rate is within a valid range
        pack_success_rate = max(0.0, min(1.0, pack_success_rate))
        
        # Compute reputation change using the test.py formula
        total_reputation_change = 0
        for _ in range(transaction_count):
            # Compute computed_reward
            computed_reward = calculate_computed_reward(pack_success_rate, current_reputation, hunger)
            
            # Simulate transaction success/failure
            success = np.random.random() < pack_success_rate
            
            if success:
                # On success: reputation increases by computed_reward * 20
                reputation_increase = computed_reward * 20
                current_reputation = min(10000, current_reputation + reputation_increase)
                total_reputation_change += reputation_increase
            else:
                # On failure: reputation decreases by computed_reward * 100
                reputation_decrease = computed_reward * 100
                current_reputation = max(3300, current_reputation - reputation_decrease)
                total_reputation_change -= reputation_decrease
        
        # Create pack data (no weight field; weight is derived from index)
        pack_data = {
            'start_step': i * pack_interval,
            'end_step': (i + 1) * pack_interval - 1,
            'transaction_count': transaction_count,
            'success_rate': pack_success_rate,
            'total_reputation_change': total_reputation_change,
            'avg_delay': max(0.0, min(1.0, pack_delay)) * 2000  # Convert to ms for computation
        }
        
        window_data.append(pack_data)
    
    return window_data

def calculate_weighted_features_from_window(current_features, window_data):
    """
    [DEPRECATED] Compute weighted features from sliding window data (consistent with test.py).
    
    Note: this function is no longer used. All weighted-feature computation is handled by the global sliding window.
    This function is kept only for backward compatibility; prefer the global sliding window in practice.
    
    Args:
        current_features: current features [reputation, success_rate, delay_level, hunger]
        window_data: historical window data
    
    Returns:
        weighted_features: weighted features
    """
    print("WARN: calculate_weighted_features_from_window is deprecated; use the global sliding window system instead.")
    if not window_data:
        return current_features
    
    # Weighted success rate (consistent with test.py: newer packs have higher weight)
    weighted_success_rate = 0.0
    weighted_avg_delay = 0.0
    total_weight = 0.0
    total_reputation_change = 0
    
    # Compute weights for packs in the window (newer = higher weight)
    for i, pack in enumerate(window_data):
        # Linear increasing weights; newest pack has highest weight
        weight = i + 1  # Start at 1
        
        weighted_success_rate += pack['success_rate'] * weight
        weighted_avg_delay += pack['avg_delay'] * weight
        total_weight += weight
        total_reputation_change += pack['total_reputation_change']
    
    # Current pack has the highest weight
    current_weight = len(window_data) + 1
    current_success_rate = current_features[1]
    current_delay_grade = current_features[2]  # Delay grade is already in [0.0, 1.0]
    
    weighted_success_rate += current_success_rate * current_weight
    weighted_avg_delay += current_delay_grade * current_weight
    total_weight += current_weight
    
    # Normalize
    if total_weight > 0:
        weighted_success_rate /= total_weight
        weighted_avg_delay /= total_weight
    
    # Weighted average delay grade is already within [0.0, 1.0]
    weighted_delay_grade = weighted_avg_delay
    
    # Adjust current reputation using historical reputation change
    adjusted_reputation = current_features[0] + total_reputation_change * 0.1  # Scale factor
    adjusted_reputation = max(3000, min(10000, adjusted_reputation))  # Clamp
    
    # Return weighted features
    weighted_features = [
        adjusted_reputation,
        weighted_success_rate,
        weighted_delay_grade,
        current_features[3]  # Keep hunger unchanged
    ]
    
    return weighted_features

def create_custom_dataset(n_na_list, n_samples_per_case=10):
    """
    Create custom test dataset using four-group reputation strategy from test.py
    
    Args:
        n_na_list: List of NA counts to test
        n_samples_per_case: Number of samples per NA count
    
    Returns:
        test_cases: List of test cases
    """
    print("Creating custom test dataset with four-group reputation strategy...")
    
    test_cases = []
    
    for n_na in n_na_list:
        for sample_idx in range(n_samples_per_case):
            # Create random test data
            test_data = np.zeros((n_na, 4))
            
            # Use the unified NA feature generator
            if n_na == 10:
                na_type = 'two_groups'
            elif n_na == 20:
                na_type = 'random'
            else:
                na_type = 'mixed'
            
            # Call the unified NA generator
            na_features = generate_na_features(n_na, na_type)
            
            test_data[:, 0] = na_features['reputation']
            test_data[:, 1] = na_features['success_rate']
            test_data[:, 2] = na_features['delay_level']
            test_data[:, 3] = na_features['hunger']
            
            total_tx = na_features['total_tx']
            success_count = na_features['success_count']
            
            test_cases.append({
                'n_na': n_na,
                'sample_idx': sample_idx,
                'raw_data': test_data,
                'total_tx': total_tx,
                'success_count': success_count,
                'normalized_data': normalize_features(test_data)
            })
    
    print(f"✅ Created {len(test_cases)} test cases with four-group strategy")
    return test_cases

def create_attack_scenario_dataset(attack_scenarios, n_na_per_scenario=10):
    """
    Create an attack-scenario test dataset: each scenario contains one attack node and the rest are normal nodes.
    Note: temporary window data is no longer generated; all window data is maintained by the global sliding window.
    
    Args:
        attack_scenarios: list of attack scenarios ['ME', 'OOA', 'OSA', 'normal']
        n_na_per_scenario: NA count per scenario
    
    Returns:
        test_cases: list of attack-scenario test cases
    """
    print("Creating attack-scenario test dataset (one attack node per scenario)...")
    print("Note: window data is maintained by the global sliding window system (no temporary window data).")
    
    test_cases = []
    
    for attack_type in attack_scenarios:
        if attack_type == 'normal':
            print(f"Generating {attack_type} scenario data (all nodes are normal)...")
        else:
            print(f"Generating {attack_type} attack scenario data (1 attack node + {n_na_per_scenario-1} normal nodes)...")
        
        # ME scenario: pick attack nodes from both the high-rep and low-rep groups
        if attack_type == 'ME':
            # For 10 NAs: fixed selection of NA #2 and #8 as attack nodes
            if n_na_per_scenario == 10:
                # high_rep_attack_idx = np.random.randint(0, 5)  # Pick one from the high-rep group
                # low_rep_attack_idx = np.random.randint(5, 10)  # Pick one from the low-rep group
                attack_node_indices = [2, 8]
            # For 20 NAs: randomly select two nodes as attack nodes
            elif n_na_per_scenario == 20:
                attack_node_indices = np.random.choice(n_na_per_scenario, 2, replace=False).tolist()
            else:
                # Keep the original logic for other cases
                attack_node_indices = [np.random.randint(0, n_na_per_scenario)]
        elif attack_type != 'normal':
            # Keep the original logic for other attack types
            attack_node_indices = [np.random.randint(0, n_na_per_scenario)]
        else:
            attack_node_indices = []  # No attack nodes
        
        # Use the unified NA feature generator
        if n_na_per_scenario == 10:
            na_type = 'two_groups'
        elif n_na_per_scenario == 20:
            na_type = 'random'
        else:
            na_type = 'mixed'
        
        # Generate base features for all NAs (including attack and normal nodes)
        na_features = generate_na_features(n_na_per_scenario, na_type, attack_node_indices, attack_type)
        
        # Create base feature data (no temporary window data)
        test_data = np.zeros((n_na_per_scenario, 4))
        
        test_data[:, 0] = na_features['reputation']
        test_data[:, 1] = na_features['success_rate']
        test_data[:, 2] = na_features['delay_level']
        test_data[:, 3] = na_features['hunger']
        
        test_cases.append({
            'attack_type': attack_type,
            'attack_node_idx': attack_node_indices[0] if len(attack_node_indices) == 1 else None,  # Backward compatibility
            'attack_node_indices': attack_node_indices if attack_type != 'normal' else [],
            'n_na': n_na_per_scenario,
            'raw_data': test_data,
            'total_tx': na_features['total_tx'],
            'success_count': na_features['success_count'],
            'normalized_data': normalize_features(test_data)  # Normalize from raw data directly
        })
    
    print(f"Created {len(test_cases)} attack-scenario test cases (no temporary window data).")
    return test_cases

def load_custom_csv_dataset(csv_path):
    """
    Load custom dataset from CSV file
    
    CSV format requirements:
    - Each row represents one NA
    - Columns: reputation, success_rate, signature_delay, hunger
    
    Args:
        csv_path: CSV file path
    
    Returns:
        test_case: Test case
    """
    print(f"Loading data from CSV file: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    
    # Read CSV data
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_columns = ['reputation', 'success_rate', 'signature_delay', 'hunger']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV file missing required columns: {missing_columns}")
    
    # Convert to numpy array
    raw_data = df[required_columns].values
    n_na = len(raw_data)
    
    print(f"Successfully loaded data for {n_na} NAs")
    print("Data ranges:")
    print(f"   - Reputation: {raw_data[:, 0].min():.0f} ~ {raw_data[:, 0].max():.0f}")
    print(f"   - Success Rate: {raw_data[:, 1].min():.3f} ~ {raw_data[:, 1].max():.3f}")
    print(f"   - Signature Delay: {raw_data[:, 2].min():.3f} ~ {raw_data[:, 2].max():.3f} (delay level)")
    print(f"   - Hunger: {raw_data[:, 3].min():.3f} ~ {raw_data[:, 3].max():.3f}")
    
    return {
        'n_na': n_na,
        'sample_idx': 0,
        'raw_data': raw_data,
        'normalized_data': normalize_features(raw_data)
    }

def export_na_parameters(raw_data, normalized_data, q_values_np, phase_name, attack_type='normal', save_path=None, na_window_queues=None):
    """
    Export parameters of all NAs (including Q-values) to a file.
    
    Args:
        raw_data: raw NA data [n_na, 4] (reputation, success_rate, delay_level, hunger)
        normalized_data: normalized NA data [n_na, 4]
        q_values_np: Q-value array [n_na]
        phase_name: phase name (e.g., 'initial_prediction', 'phase2_selection', 'phase3_selection')
        attack_type: attack type
        save_path: save path; auto-generated if None
        na_window_queues: sliding window queues used to compute real weighted features
    
    Returns:
        save_path: actual saved file path
    """
    import time
    
    # Build save path
    if save_path is None:
        save_path = f"/mnt/data/wy2024/Malicious behavior experiment/experiment_code_data/{attack_type}_{phase_name}_na_parameters.csv"
    
    # Compute weighted success rate and weighted delay grade
    weighted_success_rate = []
    weighted_delay_level = []
    
    # If sliding-window data exists, compute real weighted values
    if na_window_queues is not None:
        for i in range(len(raw_data)):
            # Retrieve NA's sliding-window queue
            queue = na_window_queues.get(i, deque())
            
            if len(queue) == 0:
                # If no window data exists, use current values
                weighted_success_rate.append(raw_data[i, 1])
                weighted_delay_level.append(raw_data[i, 2])
            else:
                # Use the same weighting scheme as the model (linear increasing weight by index)
                weighted_success_rate_sum = 0.0
                weighted_avg_delay_sum = 0.0
                total_weight = 0.0
                
                # Compute weights for packs in the queue (newer packs have higher weight)
                for j, pack in enumerate(queue):
                    # Linear increasing weights; newest pack has the highest weight
                    weight = j + 1
                    
                    weighted_success_rate_sum += pack['success_rate'] * weight
                    weighted_avg_delay_sum += pack['avg_delay'] * weight
                    total_weight += weight
                
                # Normalize weighted values
                w_success = weighted_success_rate_sum / total_weight
                w_avg_delay = weighted_avg_delay_sum / total_weight
                
                # Weighted average delay grade is already within [0.0, 1.0]
                w_delay_grade = w_avg_delay
                
                weighted_success_rate.append(w_success)
                weighted_delay_level.append(w_delay_grade)
    else:
        # Without sliding-window data, directly use raw values
        for i in range(len(raw_data)):
            weighted_success_rate.append(raw_data[i, 1])
            weighted_delay_level.append(raw_data[i, 2])
    
    # Create DataFrame
    data_dict = {
        'NA_Index': range(len(raw_data)),
        'Reputation': np.round(raw_data[:, 0], 2),
        'Success_Rate': np.round(raw_data[:, 1], 2),
        'Delay_Level': np.round(raw_data[:, 2], 2),
        'Hunger': np.round(raw_data[:, 3], 2),
        'Weighted_Success_Rate': np.round(weighted_success_rate, 3),
        'Weighted_Delay_Level': np.round(weighted_delay_level, 3),
        'Q_Value': np.round(q_values_np, 4)
    }
    
    df = pd.DataFrame(data_dict)
    
    # Add ranking
    df['Q_Value_Rank'] = df['Q_Value'].rank(ascending=False, method='min').astype(int)
    
    # Sort by Q-value descending
    df = df.sort_values('Q_Value', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    
    print(f"NA parameters exported to: {save_path}")
    print(f"   Full parameter rows: {len(df)}")
    print(f"   Max Q-value: {df['Q_Value'].max():.4f} (NA_{df.iloc[0]['NA_Index']})")
    print(f"   Min Q-value: {df['Q_Value'].min():.4f} (NA_{df.iloc[-1]['NA_Index']})")
    if na_window_queues is not None:
        print("   Using real sliding-window data for weighted features.")
    else:
        print("   No sliding-window data; using raw feature values.")
    
    return save_path

def predict_with_model(model, test_case, device, export_params=False, phase_name='prediction', attack_type='normal', na_window_queues=None):
    """
    Run inference with the model.
    
    Args:
        model: trained model
        test_case: test case
        device: runtime device
        export_params: whether to export NA parameters
        phase_name: phase name
        attack_type: attack type
        na_window_queues: sliding window queues used to compute real weighted features
    
    Returns:
        predictions: inference result
    """
    # Convert to tensor
    input_tensor = torch.FloatTensor(test_case['normalized_data']).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        q_values = model(input_tensor)
        q_values_np = q_values.cpu().numpy().flatten()
    
    # Export NA parameters if requested
    if export_params:
        export_na_parameters(
            test_case['raw_data'], 
            test_case['normalized_data'], 
            q_values_np, 
            phase_name, 
            attack_type,
            save_path=None,
            na_window_queues=na_window_queues
        )
    
    # Best selection
    best_na_idx = np.argmax(q_values_np)
    
    return {
        'q_values': q_values_np,
        'best_na_idx': best_na_idx,
        'best_q_value': q_values_np[best_na_idx],
        'q_value_range': (q_values_np.min(), q_values_np.max()),
        'q_value_std': q_values_np.std()
    }

def analyze_predictions(test_cases, predictions):
    """
    Analyze prediction results
    
    Args:
        test_cases: List of test cases
        predictions: List of prediction results
    """
    print("\n" + "="*60)
    print("Prediction Results Analysis")
    print("="*60)
    
    # Check whether this is an attack-scenario test
    is_attack_scenario = any('attack_type' in test_case for test_case in test_cases)
    
    if is_attack_scenario:
        # Group by attack type
        attack_types = {}
        for i, test_case in enumerate(test_cases):
            attack_type = test_case['attack_type']
            if attack_type not in attack_types:
                attack_types[attack_type] = []
            attack_types[attack_type].append((test_case, predictions[i]))
        
        for attack_type in ['normal', 'ME', 'OOA', 'OSA']:
            if attack_type in attack_types:
                cases_and_preds = attack_types[attack_type]
                preds = [p for _, p in cases_and_preds]
                
                q_ranges = [p['q_value_range'] for p in preds]
                q_stds = [p['q_value_std'] for p in preds]
                best_q_values = [p['best_q_value'] for p in preds]
                best_indices = [p['best_na_idx'] for p in preds]
                
                print(f"\nAttack Type: {attack_type}")
                print(f"   Sample Count: {len(preds)}")
                print(f"   Q-value Range: {np.mean([r[0] for r in q_ranges]):.4f} ~ {np.mean([r[1] for r in q_ranges]):.4f}")
                print(f"   Q-value Std Dev: {np.mean(q_stds):.4f} ± {np.std(q_stds):.4f}")
                print(f"   Best Q-value: {np.mean(best_q_values):.4f} ± {np.std(best_q_values):.4f}")
                print(f"   Selection Distribution: Average index {np.mean(best_indices):.1f} ± {np.std(best_indices):.1f}")
                
                # Analyze model's ability to identify different attacks
                if attack_type != 'normal':
                    # Calculate tendency to avoid malicious nodes
                    test_case = cases_and_preds[0][0]
                    raw_data = test_case['raw_data']
                    
                    print(f"   Original Feature Ranges:")
                    print(f"     - Reputation: {raw_data[:, 0].min():.0f} ~ {raw_data[:, 0].max():.0f}")
                    print(f"     - Success Rate: {raw_data[:, 1].min():.3f} ~ {raw_data[:, 1].max():.3f}")
                    print("   Note: weighted feature data is computed dynamically by the global sliding window.")
    else:
        # Group analysis by NA count
        na_counts = {}
        for i, test_case in enumerate(test_cases):
            n_na = test_case['n_na']
            if n_na not in na_counts:
                na_counts[n_na] = []
            na_counts[n_na].append(predictions[i])
        
        for n_na in sorted(na_counts.keys()):
            preds = na_counts[n_na]
            q_ranges = [p['q_value_range'] for p in preds]
            q_stds = [p['q_value_std'] for p in preds]
            best_q_values = [p['best_q_value'] for p in preds]
            
            print(f"\nNA Count: {n_na}")
            print(f"   Sample Count: {len(preds)}")
            print(f"   Q-value Range: {np.mean([r[0] for r in q_ranges]):.4f} ~ {np.mean([r[1] for r in q_ranges]):.4f}")
            print(f"   Q-value Std Dev: {np.mean(q_stds):.4f} ± {np.std(q_stds):.4f}")
            print(f"   Best Q-value: {np.mean(best_q_values):.4f} ± {np.std(best_q_values):.4f}")

def analyze_attack_resistance(test_cases, predictions):
    """
    Analyze the model's resistance to attacks.
    
    Args:
        test_cases: attack-scenario test cases
        predictions: prediction results
    """
    print("\n" + "="*60)
    print("Attack Resistance Analysis")
    print("="*60)
    
    attack_results = {}
    
    for i, test_case in enumerate(test_cases):
        attack_type = test_case['attack_type']
        prediction = predictions[i]
        
        if attack_type not in attack_results:
            attack_results[attack_type] = {
                'selections': [],
                'q_values': [],
                'raw_features': []
            }
        
        best_idx = prediction['best_na_idx']
        attack_results[attack_type]['selections'].append(best_idx)
        attack_results[attack_type]['q_values'].append(prediction['q_values'])
        attack_results[attack_type]['raw_features'].append(test_case['raw_data'][best_idx])
    
    # Compare selection quality across attack types
    if 'normal' in attack_results:
        normal_q_mean = np.mean([np.max(q) for q in attack_results['normal']['q_values']])
        print(f"\nBaseline (normal nodes): mean best Q-value = {normal_q_mean:.4f}")
        
        for attack_type in ['ME', 'OOA', 'OSA']:
            if attack_type in attack_results:
                attack_q_mean = np.mean([np.max(q) for q in attack_results[attack_type]['q_values']])
                resistance_ratio = attack_q_mean / normal_q_mean if normal_q_mean > 0 else 0
                
                print(f"\n{attack_type} Attack:")
                print(f"   Average Best Q-value: {attack_q_mean:.4f}")
                print(f"   Resistance Ability: {resistance_ratio:.3f} ({'Strong' if resistance_ratio > 0.8 else 'Medium' if resistance_ratio > 0.6 else 'Weak'})")
                
                # Analyze quality of selected features
                selected_features = np.array(attack_results[attack_type]['raw_features'])
                if len(selected_features) > 0:
                    print(f"   Average Features of Selected Nodes:")
                    print(f"     - Reputation: {selected_features[:, 0].mean():.0f}")
                    print(f"     - Success Rate: {selected_features[:, 1].mean():.3f}")
                    print(f"     - Delay Level: {selected_features[:, 2].mean():.3f}")
                    print("   Note: weighted feature data is computed dynamically by the global sliding window.")



def simulate_transaction_evolution(
    model,
    test_case,
    device,
    n_transactions=50,
    attack_type='normal',
    attack_mode='behavior',
    me_delay_step=0.02,
    low_delay_range=(0.0, 0.3),
    high_delay_range=(0.7, 1.0),
    osa_cycle_length=20,
    force_fail_on_max_delay=True,
):
    """
    Simulate transaction evolution with a complete hunger-update mechanism.
    """
    # Global hunger tracking
    global_last_selected_time = np.full(20, -1)  # Last-selected time for each NA (-1 means never selected)
    global_current_step = 0  # Global step counter
    
    def _update_hunger_for_all_nas(selected_indices, current_step, phase1_data, phase='phase2'):
        """
        Update hunger levels for all 10 NAs.
        
        Args:
            selected_indices: list of selected NA indices
            current_step: current time step
            phase1_data: data array for 10 NAs
            phase: current phase; hunger is not updated during 'phase1'
        """
        # Do not update hunger during Phase 1
        if phase == 'phase1':
            return
            
        # Update selected records
        for idx in selected_indices:
            global_last_selected_time[idx] = current_step
        
        # Compute hunger for all NAs
        for i in range(10):
            steps_since_selected = current_step - global_last_selected_time[i]
            
            if phase == 'phase1':
                pass
            elif steps_since_selected <= 0:
                phase1_data[i, 3] = 0.0
            elif global_last_selected_time[i] == -1:
                # If never selected, use current_step to compute hunger
                normalized_steps = current_step / 20.0
                phase1_data[i, 3] = min(1.0, np.log(1 + normalized_steps) / np.log(11))
            else:
                # Same log function as test.py: hunger = log(1 + steps/20) / log(11)
                normalized_steps = steps_since_selected / 20.0
                phase1_data[i, 3] = min(1.0, np.log(1 + normalized_steps) / np.log(11))
        
        # Reset hunger to 0 for selected NAs
        for idx in selected_indices:
            phase1_data[idx, 3] = 0.0
    """
    Three phases:
    1) Fill sliding-window data for 10 NAs (50 time points)
    2) Model selects 5 NAs for transaction simulation (n_transactions)
    3) Re-select 5 NAs for another simulation (n_transactions)
    """
    print(
        f"[SIM] Start 3-phase simulation [{attack_mode}]: "
        f"Phase 1 (50 time points window fill) + "
        f"Phase 2 ({n_transactions} tx simulation) + "
        f"Phase 3 ({n_transactions} tx simulation)..."
    )
    
    # Sliding window system setup
    window_pack_interval = 5  # Pack every 5 transactions
    window_queue_size = 20    # Max queue length per NA
    current_step_count = 0    # Current step counter
    
    def _init_na_window_system(n_na):
        """Initialize the NA sliding-window system."""
        na_window_queues = {}
        na_current_pack = {}
        for na_id in range(n_na):
            na_window_queues[na_id] = deque(maxlen=window_queue_size)
            na_current_pack[na_id] = {
                'transactions': [],
                'success_count': 0,
                'total_count': 0,
                'reputation_changes': [],
                'start_step': 0,
                'end_step': 0
            }
        return na_window_queues, na_current_pack
    
    # Initialize a global sliding-window system shared across phases (10 NAs)
    na_window_queues_global, na_current_pack_global = _init_na_window_system(10)
    print("[INIT] Global sliding-window system initialized; 10 NAs share one system.")
    
    def _update_na_window_queue(na_idx, transaction_data, na_current_pack, current_step):
        """Update current transaction pack for a specific NA."""
        current_pack = na_current_pack[na_idx]
        
        # If this is the start of a new pack, record start step
        if len(current_pack['transactions']) == 0:
            current_pack['start_step'] = current_step
        
        # Append transaction to current pack
        current_pack['transactions'].append(transaction_data)
        current_pack['total_count'] += 1
        if transaction_data['success']:
            current_pack['success_count'] += 1
        
        # Track reputation change
        reputation_change = transaction_data['reputation_after'] - transaction_data['reputation_before']
        current_pack['reputation_changes'].append(reputation_change)
        
        # Update end step
        current_pack['end_step'] = current_step
    
    def _pack_all_na_windows(na_window_queues, na_current_pack, current_step):
        """Pack all NAs' current pack data into queues."""
        for na_idx in na_current_pack.keys():
            current_pack = na_current_pack[na_idx]
            
            if current_pack['total_count'] > 0:
                # Pack statistics
                pack_success_rate = current_pack['success_count'] / current_pack['total_count']
                total_reputation_change = sum(current_pack['reputation_changes'])
                avg_delay = np.mean([t.get('delay', 0) for t in current_pack['transactions']])
                
                # Pack data
                pack_data = {
                    'start_step': current_pack['start_step'],
                    'end_step': current_pack['end_step'],
                    'transaction_count': current_pack['total_count'],
                    'success_rate': pack_success_rate,
                    'total_reputation_change': total_reputation_change,
                    'avg_delay': avg_delay
                }
                
                # Enqueue
                na_window_queues[na_idx].append(pack_data)
                
                # Reset current pack
                na_current_pack[na_idx] = {
                    'transactions': [],
                    'success_count': 0,
                    'total_count': 0,
                    'reputation_changes': [],
                    'start_step': 0,
                    'end_step': 0
                }
    
    def _get_na_window_summary(na_idx, na_window_queues):
        """Get sliding-window summary stats for an NA."""
        queue = na_window_queues.get(na_idx, deque())
        
        if len(queue) == 0:
            return {
                'weighted_success_rate': 0.0,
                'weighted_avg_delay': 0.0,
                'weighted_delay_grade': 0.0,
                'total_transactions': 0,
                'total_reputation_change': 0.0
            }
        
        # Weighted statistics (consistent with test.py: newer packs have higher weight)
        weighted_success_rate = 0.0
        weighted_avg_delay = 0.0
        total_weight = 0.0
        
        # Compute weights for packs in the queue (newer = higher weight)
        for i, pack in enumerate(queue):
            weight = i + 1
            
            weighted_success_rate += pack['success_rate'] * weight
            weighted_avg_delay += pack['avg_delay'] * weight
            total_weight += weight
        
        # Normalize weighted values
        if total_weight > 0:
            weighted_success_rate /= total_weight
            weighted_avg_delay /= total_weight
        
        # Weighted delay grade is already within [0.0, 1.0]
        weighted_delay_grade = weighted_avg_delay
        
        total_transactions = sum(pack['transaction_count'] for pack in queue)
        total_reputation_change = sum(pack['total_reputation_change'] for pack in queue)
        
        return {
            'weighted_success_rate': weighted_success_rate,
            'weighted_avg_delay': weighted_avg_delay,
            'weighted_delay_grade': weighted_delay_grade,
            'total_transactions': total_transactions,
            'total_reputation_change': total_reputation_change
        }
    
    # Phase 1: fill sliding-window data for 10 NAs
    print("\nPhase 1: fill sliding-window data for 10 NAs...")
    initial_10_na_data = test_case['raw_data']
    phase1_data = initial_10_na_data.copy()
    
    # Set similar but not identical initial hunger values for 10 NAs
    print(f"[DEBUG] Original hunger data: {phase1_data[:, 3]}")
    avg_hunger = np.mean(phase1_data[:, 3])
    print(f"[DEBUG] Original mean hunger: {avg_hunger:.3f}")
    
    # If original hunger values are all 0, set a reasonable baseline
    if avg_hunger == 0.0:
        base_hunger = 0.3
        print(f"[INIT] Original hunger is 0; using baseline: {base_hunger:.3f}")
    else:
        base_hunger = avg_hunger
        print(f"[INIT] Using original mean hunger as baseline: {base_hunger:.3f}")
    
    # Generate 10 distinct hunger values around the baseline (±0.05)
    np.random.seed(42)
    hunger_variations = np.random.uniform(-0.05, 0.05, 10)
    for i in range(10):
        phase1_data[i, 3] = max(0.0, min(1.0, base_hunger + hunger_variations[i]))
    
    print(f"[INIT] Phase 1饥饿度设置为相近但不同的值，基准值: {base_hunger:.3f}")
    print(f"[INIT] 饥饿度范围: {np.min(phase1_data[:, 3]):.3f} - {np.max(phase1_data[:, 3]):.3f}")
    for i in range(10):
        print(f"   NA_{i:02d}: {phase1_data[i, 3]:.4f}")
    
    # 初始化累积事务计数器（用于基于实际事务成功与否计算成功率）
    phase1_total_tx = np.zeros(10, dtype=int)  # 累积总事务数
    phase1_success_count = np.zeros(10, dtype=int)  # 累积成功事务数
    
    print(f"[INIT] 初始化累积事务计数器，将基于实际事务成功与否计算成功率")
    phase1_evolution = {
        'time_points': list(range(51)),  # 0-50，但实际事务生成从1-50
        'reputations': [],
        'na_count': 10,
        'phase': 'window_filling'
    }
    
    # 初始化Phase 1的加权成功率数据收集
    phase1_evolution['weighted_success_rates'] = []
    # 初始化Phase 1的实际成功率数据收集
    phase1_evolution['actual_success_rates'] = []
    
    # 为每个NA分配特定的窗口策略（基于高低信誉分组）
    na_window_strategies = {}
    
    # 高信誉组（NA0-NA4）：5种不同的成功率变化模式
    high_rep_strategies = ['up_down', 'down_up', 'constant', 'always_up', 'always_down']
    for i, na_idx in enumerate(range(5)):
        na_window_strategies[na_idx] = high_rep_strategies[i]
        print(f"[INIT] 高信誉组 NA{na_idx} assigned strategy: {high_rep_strategies[i]}")
    
    # 低信誉组（NA5-NA9）：5种不同的成功率变化模式
    low_rep_strategies = ['up_down', 'down_up', 'constant', 'always_up', 'always_down']
    for i, na_idx in enumerate(range(5, 10)):
        na_window_strategies[na_idx] = low_rep_strategies[i]
        print(f"[INIT] 低信誉组 NA{na_idx} assigned strategy: {low_rep_strategies[i]}")
    
    # 初始化每个NA的事务计数器和当前包数据
    na_transaction_counters = {na_idx: 0 for na_idx in range(10)}
    na_current_transactions = {na_idx: [] for na_idx in range(10)}
    
    # 模拟滑动窗口填充过程 - 修改为每个时间步生成1个事务
    for time_step in range(51):
        # 为每个NA生成1个事务数据并累积
        for na_idx in range(10):
            current_features = phase1_data[na_idx]
            
            # 根据攻击类型生成单个事务数据
            if time_step > 0:  # 跳过初始时间点
                # 生成单个事务的性能参数
                strategy = na_window_strategies[na_idx]
                
                # 基础性能参数
                base_success_rate = current_features[1]
                base_delay = current_features[2]
                
                # 计算时间进度（0到1）
                progress = time_step / 50.0
                
                # 根据策略计算成功概率调节因子
                if strategy == 'up_down':
                    # 成功率先上升后下降（倒U型）
                    if time_step <= 25:
                        # 前半段上升
                        factor = 0.7 + 0.6 * (time_step / 25.0)
                    else:
                        # 后半段下降
                        factor = 1.3 - 0.6 * ((time_step - 25) / 25.0)
                    
                elif strategy == 'down_up':
                    # 成功率先下降后上升（U型）
                    if time_step <= 25:
                        # 前半段下降
                        factor = 1.3 - 0.6 * (time_step / 25.0)
                    else:
                        # 后半段上升
                        factor = 0.7 + 0.6 * ((time_step - 25) / 25.0)
                    
                elif strategy == 'constant':
                    # 成功率保持不变
                    factor = 1.0
                    
                elif strategy == 'always_up':
                    # 成功率一直上升
                    factor = 0.7 + 0.6 * progress
                    
                elif strategy == 'always_down':
                    # 成功率一直下降
                    factor = 1.3 - 0.6 * progress
                    
                else:
                    # 默认：保持基础性能
                    factor = 1.0
                
                # 计算当前时间步的事务成功概率
                transaction_success_probability = base_success_rate * factor
                transaction_success_probability = max(0.1, min(1.0, transaction_success_probability))
                
                # 计算延迟
                transaction_delay = base_delay * (2.0 - factor)
                transaction_delay = max(0.1, min(2.0, transaction_delay))
                
                # 检查延迟度是否达到1.0，如果是则强制事务失败
                if force_fail_on_max_delay and transaction_delay >= 1.0:
                    is_success = False
                    print(f"[DELAY] NA{na_idx} 延迟度达到{transaction_delay:.3f}，事务强制失败")
                else:
                    # 基于策略调控的成功概率决定事务成功与否
                    is_success = np.random.random() < transaction_success_probability
                
                # 更新累积事务计数器
                phase1_total_tx[na_idx] += 1
                if is_success:
                    phase1_success_count[na_idx] += 1
                
                # 计算当前NA的实际成功率（基于累积事务数）
                current_actual_success_rate = phase1_success_count[na_idx] / phase1_total_tx[na_idx] if phase1_total_tx[na_idx] > 0 else 0.0
                
                # 计算信誉变化
                computed_reward = calculate_computed_reward(
                    current_actual_success_rate, current_features[0], current_features[3]
                )
                
                if is_success:
                    reputation_change = computed_reward * 20
                else:
                    reputation_change = -computed_reward * 100
                
                # 更新信誉值
                phase1_data[na_idx, 0] = max(3000, min(10000, 
                    phase1_data[na_idx, 0] + reputation_change * 0.1))
                
                # 将事务添加到当前包中
                na_current_transactions[na_idx].append({
                    'success': is_success,
                    'delay': transaction_delay,  # 直接使用延迟等级（0.0-1.0）
                    'reputation_change': reputation_change * 0.1
                })
                
                na_transaction_counters[na_idx] += 1
                
                # 每10个事务打包一次（与第二、三阶段保持一致）
                if na_transaction_counters[na_idx] % 10 == 0:
                    transactions = na_current_transactions[na_idx]
                    if transactions:
                        # 计算包统计
                        success_count = sum(1 for t in transactions if t['success'])
                        total_count = len(transactions)
                        success_rate = success_count / total_count if total_count > 0 else 0.0
                        avg_delay = np.mean([t['delay'] for t in transactions])
                        total_reputation_change = sum(t['reputation_change'] for t in transactions)
                        
                        # 创建包数据
                        pack_data = {
                            'success_rate': success_rate,
                            'avg_delay': avg_delay,
                            'transaction_count': total_count,
                            'total_reputation_change': total_reputation_change,
                            'weight': total_count,  # 时间权重等于交易数量
                            'start_step': time_step - 9,
                            'end_step': time_step
                        }
                        
                        # 添加到全局滑动窗口队列
                        na_window_queues_global[na_idx].append(pack_data)
                        
                        # 清空当前事务列表
                        na_current_transactions[na_idx] = []
        
        phase1_evolution['reputations'].append(phase1_data[:, 0].copy())
        
        # 收集每个时间步的实际成功率数据（基于累积事务数计算）
        actual_success_rates = []
        for na_idx in range(10):
            # 计算基于累积事务数的实际成功率
            if phase1_total_tx[na_idx] > 0:
                actual_success_rate = phase1_success_count[na_idx] / phase1_total_tx[na_idx]
            else:
                # 如果还没有事务，使用初始成功率
                actual_success_rate = phase1_data[na_idx, 1]
            
            actual_success_rates.append(actual_success_rate)
        
        phase1_evolution['actual_success_rates'].append(actual_success_rates.copy())
        
        # 每5个时间步收集一次加权成功率数据（与Phase 2/3保持一致）
        if time_step % 5 == 0:
            weighted_success_rates = []
            for na_idx in range(10):
                summary = _get_na_window_summary(na_idx, na_window_queues_global)
                weighted_success_rates.append(summary['weighted_success_rate'])
            phase1_evolution['weighted_success_rates'].append(weighted_success_rates.copy())
        
        current_step_count += 1
    
    print(f"✅ 第一阶段完成，10个NA的信誉范围: {phase1_data[:, 0].min():.0f} - {phase1_data[:, 0].max():.0f}")
    
    # 更新phase1_data为第一阶段结束时的实际状态
    # 使用第一阶段最后一个时间步的信誉值
    final_reputations = phase1_evolution['reputations'][-1]  # 最后一个时间步的信誉值
    phase1_data[:, 0] = final_reputations  # 更新信誉值
    
    # 成功率现在完全基于累积事务数计算，不再直接更新phase1_data[:, 1]
    # phase1_data[:, 1]将在需要时通过phase1_success_count/phase1_total_tx动态计算
    
    # 计算并更新延迟值（基于成功率的反向关系）
    for na_idx in range(10):
        strategy = na_window_strategies[na_idx]
        base_delay = test_case['raw_data'][na_idx, 2]  # 原始基础延迟
        
        # 根据策略计算最终延迟
        if strategy == 'up_down':
            # 成功率在最后是下降的，所以延迟应该增加
            factor = 1.3 - 0.6 * ((50 - 25) / 25.0)  # 对应最后时间步的成功率因子
            delay_factor = 2.0 - factor  # 延迟因子与成功率因子相反
        elif strategy == 'down_up':
            # 成功率在最后是上升的，所以延迟应该减少
            factor = 0.7 + 0.6 * ((50 - 25) / 25.0)  # 对应最后时间步的成功率因子
            delay_factor = 2.0 - factor  # 延迟因子与成功率因子相反
        elif strategy == 'constant':
            delay_factor = 1.0  # 保持不变
        elif strategy == 'always_up':
            # 成功率一直上升，延迟一直下降
            factor = 1.0 + 0.3 * 1.0  # 最终成功率因子
            delay_factor = 2.0 - factor  # 延迟因子与成功率因子相反
        elif strategy == 'always_down':
            # 成功率一直下降，延迟一直上升
            factor = 1.0 - 0.3 * 1.0  # 最终成功率因子
            delay_factor = 2.0 - factor  # 延迟因子与成功率因子相反
        else:
            delay_factor = 1.0
        
        final_delay = base_delay * delay_factor
        final_delay = max(0.1, min(2.0, final_delay))  # 限制延迟范围
        phase1_data[na_idx, 2] = final_delay
    
    print(f"📊 更新后的phase1_data状态:")
    print(f"   信誉范围: {phase1_data[:, 0].min():.0f} - {phase1_data[:, 0].max():.0f}")
    print(f"   成功率范围: {phase1_data[:, 1].min():.3f} - {phase1_data[:, 1].max():.3f}")
    print(f"   延迟范围: {phase1_data[:, 2].min():.3f} - {phase1_data[:, 2].max():.3f}")
    
    # 第二阶段：模型选择5个NA
    print("\n🎯 第二阶段：模型从10个NA中选择5个NA...")
    print("📝 使用全局滑动窗口的加权特征数据进行模型预测")
    
    # 🚀 使用全局滑动窗口计算加权特征数据
    weighted_data = np.zeros((10, 4))
    for na_idx in range(10):
        summary = _get_na_window_summary(na_idx, na_window_queues_global)
        weighted_data[na_idx, 0] = phase1_data[na_idx, 0]  # 信誉值使用当前值
        weighted_data[na_idx, 1] = summary['weighted_success_rate']  # 使用加权成功率
        weighted_data[na_idx, 2] = summary['weighted_delay_grade']  # 使用加权延迟等级
        weighted_data[na_idx, 3] = phase1_data[na_idx, 3]  # 饥饿度使用当前值
        print(f"   NA{na_idx}: 加权成功率={summary['weighted_success_rate']:.3f}, 加权延迟等级={summary['weighted_delay_grade']:.3f}")
    
    # 使用加权特征数据进行模型预测
    normalized_data = normalize_features(weighted_data)
    input_tensor = torch.FloatTensor(normalized_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = model(input_tensor)
        q_values_np = q_values.cpu().numpy().flatten()
    
    # 导出第二阶段所有NA的参数（包括Q值），传递全局滑动窗口数据
    export_na_parameters(
        weighted_data,  # 使用加权数据作为原始数据
        normalized_data,  # 归一化数据
        q_values_np,  # Q值
        'phase2_selection',  # 阶段名称
        attack_type,  # 攻击类型
        na_window_queues=na_window_queues_global  # 传递全局滑动窗口数据
    )
    
    # 选择3个高信誉和2个低信誉的NA（严格按照6600分界线）
    reputations = phase1_data[:, 0]
    
    # 分离高信誉（≥6600）和低信誉（<6600）NA
    high_rep_indices = np.where(reputations >= 6600)[0]
    low_rep_indices = np.where(reputations < 6600)[0]
    
    print(f"📊 第一阶段后信誉分布: 高信誉NA(≥6600): {len(high_rep_indices)}个, 低信誉NA(<6600): {len(low_rep_indices)}个")
    
    # 确保有足够的NA进行选择
    if len(high_rep_indices) < 3:
        print(f"⚠️ 警告: 高信誉NA不足3个，仅有{len(high_rep_indices)}个")
        # 如果高信誉NA不足，从所有NA中选择信誉最高的3个
        sorted_indices = np.argsort(reputations)[::-1]
        high_rep_selected = sorted_indices[:3]
    else:
        # 从高信誉NA中选择Q值最高的3个
        high_rep_scores = q_values_np[high_rep_indices]
        high_rep_selected = high_rep_indices[np.argsort(high_rep_scores)[-3:]]
    
    if len(low_rep_indices) < 2:
        print(f"⚠️ 警告: 低信誉NA不足2个，仅有{len(low_rep_indices)}个")
        # 如果低信誉NA不足，从剩余NA中选择信誉最低的2个
        remaining_indices = np.setdiff1d(np.arange(len(reputations)), high_rep_selected)
        if len(remaining_indices) >= 2:
            remaining_reputations = reputations[remaining_indices]
            sorted_remaining = remaining_indices[np.argsort(remaining_reputations)]
            low_rep_selected = sorted_remaining[:2]
        else:
            low_rep_selected = remaining_indices
    else:
        # 从低信誉NA中选择Q值最高的2个
        low_rep_scores = q_values_np[low_rep_indices]
        low_rep_selected = low_rep_indices[np.argsort(low_rep_scores)[-2:]]
    
    selected_indices_phase2 = np.concatenate([high_rep_selected, low_rep_selected])
    selected_data = phase1_data[selected_indices_phase2]
    
    # 确定恶意NA（根据攻击类型）
    if attack_type == 'ME' and 'attack_node_indices' in test_case:
        # ME攻击场景：修改为在高信誉组和低信誉组中各选择一个恶意NA
        malicious_na_indices = []
        
        # 在高信誉组（索引0-2）中选择一个恶意NA
        high_rep_malicious_idx = np.random.choice(range(3))  # 从前3个高信誉NA中选择
        malicious_na_indices.append(high_rep_malicious_idx)
        
        # 在低信誉组中选择一个恶意NA
        if len(low_rep_selected) > 0:
            low_rep_malicious_idx = np.random.choice(range(3, 3 + len(low_rep_selected)))  # 在低信誉NA中选择
            malicious_na_indices.append(low_rep_malicious_idx)
        else:
            # 如果没有低信誉NA，从剩余的高信誉NA中再选择一个
            remaining_high_rep = [i for i in range(3) if i != high_rep_malicious_idx]
            if remaining_high_rep:
                additional_malicious_idx = np.random.choice(remaining_high_rep)
                malicious_na_indices.append(additional_malicious_idx)
        
        malicious_na_idx = malicious_na_indices[0] if len(malicious_na_indices) > 0 else None  # 保持向后兼容
    else:
        # 修改后的逻辑：在高信誉组和低信誉组中各选择一个恶意NA
        malicious_na_indices = []
        
        # 在高信誉组（索引0-2）中选择一个恶意NA
        high_rep_malicious_idx = np.random.choice(range(3))  # 从前3个高信誉NA中选择
        malicious_na_indices.append(high_rep_malicious_idx)
        
        # 在低信誉组中选择一个恶意NA
        if len(low_rep_selected) > 0:
            low_rep_malicious_idx = np.random.choice(range(3, 3 + len(low_rep_selected)))  # 在低信誉NA中选择
            malicious_na_indices.append(low_rep_malicious_idx)
        else:
            # 如果没有低信誉NA，从剩余的高信誉NA中再选择一个
            remaining_high_rep = [i for i in range(3) if i != high_rep_malicious_idx]
            if remaining_high_rep:
                additional_malicious_idx = np.random.choice(remaining_high_rep)
                malicious_na_indices.append(additional_malicious_idx)
        
        # 保持向后兼容性，设置malicious_na_idx为第一个恶意NA的索引
        malicious_na_idx = malicious_na_indices[0] if len(malicious_na_indices) > 0 else None
    
    print(f"📊 选择的5个NA:")
    for i, orig_idx in enumerate(selected_indices_phase2):
        na_type = "恶意" if i in malicious_na_indices else ("高信誉" if i < 3 else "低信誉")
        print(f"   NA{i} (原索引{orig_idx}, {na_type}): 信誉={selected_data[i, 0]:.0f}, Q值={q_values_np[orig_idx]:.4f}")
    
    # 第二阶段：5个选定NA的事务模拟
    print(f"\n🚀 第二阶段：开始{n_transactions}个事务的模拟...")
    current_data = selected_data.copy()
    
    # 初始化选定NA的total_tx和success_count
    if 'total_tx' in test_case and 'success_count' in test_case:
        selected_total_tx = test_case['total_tx'][selected_indices_phase2].copy()
        selected_success_count = test_case['success_count'][selected_indices_phase2].copy()
    else:
        # 如果没有，基于当前成功率生成
        selected_total_tx = np.random.randint(20, 81, 5)
        # 使用动态计算的成功率：如果有累积事务数据则使用，否则使用初始成功率
        current_success_rates = np.where(selected_total_tx > 0, 
                                        selected_success_count / selected_total_tx, 
                                        current_data[:, 1])
        selected_success_count = (current_success_rates * selected_total_tx).astype(int)
    
    # 🚀 使用全局滑动窗口系统（不再重新初始化）
    current_step_count = 50  # 第二阶段从步数50开始
    print("[INFO] 第二阶段使用全局滑动窗口系统，已有窗口数据将被继续使用")
    
    phase2_evolution = {
        'time_points': [50],  # 从50开始，后续动态添加
        'reputations': [],
        'success_rates': [],
        'delay_grades': [],  # 添加延迟等级数据
        'hunger_levels': [],  # 添加饥饿度数据
        'weighted_success_rates': [],  # 添加加权成功率数据
        'weighted_delay_grades': [],  # 添加加权延迟等级数据
        'actual_success_rates': [],  # 添加实际成功率数据
        'na_labels': [],
        'malicious_na_idx': malicious_na_idx,
        'malicious_na_indices': malicious_na_indices,
        'selected_indices': selected_indices_phase2,
        'na_count': 5,
        'phase': 'transaction_simulation_phase2',
        'window_summaries': []  # 添加窗口统计摘要
    }
    
    # Generate labels for selected 5 NAs
    for i in range(5):
        orig_idx = selected_indices_phase2[i]  # 使用原始NA索引
        if i == malicious_na_idx:
            phase2_evolution['na_labels'].append(f'CA{orig_idx} (Malicious)')
        else:
            phase2_evolution['na_labels'].append(f'CA{orig_idx}')
    
    # 记录初始状态
    phase2_evolution['reputations'].append(current_data[:, 0].copy())
    # 使用动态计算的成功率
    current_success_rates = np.where(selected_total_tx > 0, 
                                    selected_success_count / selected_total_tx, 
                                    current_data[:, 1])
    phase2_evolution['success_rates'].append(current_success_rates.copy())
    phase2_evolution['delay_grades'].append(current_data[:, 2].copy())  # 添加延迟等级数据
    phase2_evolution['hunger_levels'].append(current_data[:, 3].copy())  # 添加饥饿度数据
    phase2_evolution['actual_success_rates'].append(current_success_rates.copy())  # 添加实际成功率数据
    
    # 记录初始的加权数据（使用全局窗口系统和原始NA索引）
    initial_weighted_success_rates = []
    initial_weighted_delay_grades = []
    for i in range(5):
        orig_idx = selected_indices_phase2[i]  # 使用原始NA索引
        summary = _get_na_window_summary(orig_idx, na_window_queues_global)
        initial_weighted_success_rates.append(summary['weighted_success_rate'])
        initial_weighted_delay_grades.append(summary['weighted_delay_grade'])
        print(f"   NA{i} (原索引{orig_idx}): 加权成功率={summary['weighted_success_rate']:.3f}, 加权延迟等级={summary['weighted_delay_grade']:.3f}")
    phase2_evolution['weighted_success_rates'].append(initial_weighted_success_rates.copy())
    phase2_evolution['weighted_delay_grades'].append(initial_weighted_delay_grades.copy())
    
    print(f"[INFO] Phase 2 Initial State:")
    for i in range(5):
        print(f"   {phase2_evolution['na_labels'][i]}: Reputation={current_data[i, 0]:.0f}, Success Rate={current_data[i, 1]:.3f}")
    
    # 模拟第二阶段的事务（50个事务，时间点51-100）
    # 记录每个事务的实际成功率和延迟等级
    transaction_actual_success_rates = []
    transaction_actual_delay_grades = []
    
    for transaction in range(1, n_transactions + 1):
        # 记录当前事务所有NA的实际成功率和延迟等级
        current_transaction_success_rates = []
        current_transaction_delay_grades = []
        
        # 第二阶段：所有5个选中的NA都参与每次事务执行
        for na_idx in range(5):
            # 每个NA都执行事务
            if na_idx in malicious_na_indices and attack_mode == 'delay_only':
                success_prob = current_data[na_idx, 1]
                delay_increase = 0.0
                malicious_success = None
            elif na_idx in malicious_na_indices:
                # 根据攻击类型实现不同的恶意行为模式
                if attack_type == 'ME':
                    # ME攻击：事务全部失败
                    malicious_success = False
                    delay_increase = 0.1  # 延迟增加
                    
                elif attack_type == 'OOA':
                    # OOA攻击：事务交替成功失败
                    if transaction % 2 == 0:  # 偶数事务：成功
                        malicious_success = True
                        delay_increase = np.random.uniform(-0.02, 0.02)
                    else:  # 奇数事务：失败
                        malicious_success = False
                        delay_increase = 0.3
                        
                elif attack_type == 'OSA':
                    # OSA攻击：周期性失败成功，每10个事务切换一次
                    cycle_length = 10
                    cycle_position = (transaction - 1) % (cycle_length * 2)  # 20个事务为一个完整周期
                    if cycle_position < cycle_length:  # 前10个事务：失败
                        malicious_success = False
                        delay_increase = 0.25
                    else:  # 后10个事务：成功
                        malicious_success = True
                        delay_increase = np.random.uniform(-0.02, 0.02)
                        
                else:
                    # 默认恶意行为（ME攻击模式）
                    malicious_success = False
                    delay_increase = 0.05
                    
                # 恶意NA不使用success_prob，直接控制事务成功与否
                success_prob = None  # 标记为恶意NA，不使用概率判断
            else:
                # 正常NA的行为
                success_prob = current_data[na_idx, 1]  # 使用固定的当前成功率，不再波动
                delay_increase = np.random.uniform(-0.02, 0.02)
                malicious_success = None  # 标记为正常NA
            
            # 记录当前NA的实际成功率和延迟等级
            # 计算基于累积事务数的实际成功率
            actual_success_rate = selected_success_count[na_idx] / selected_total_tx[na_idx] if selected_total_tx[na_idx] > 0 else 0.0
            current_transaction_success_rates.append(actual_success_rate)
            # 计算当前事务的实际延迟等级
            if na_idx in malicious_na_indices and attack_mode == 'delay_only':
                if attack_type == 'ME':
                    actual_delay_grade = max(0.0, min(1.0, current_data[na_idx, 2] + me_delay_step))
                elif attack_type == 'OOA':
                    if transaction % 2 == 0:
                        actual_delay_grade = np.random.uniform(low_delay_range[0], low_delay_range[1])
                    else:
                        actual_delay_grade = np.random.uniform(high_delay_range[0], high_delay_range[1])
                elif attack_type == 'OSA':
                    cycle_position = (transaction - 1) % osa_cycle_length
                    if cycle_position < (osa_cycle_length // 2):
                        actual_delay_grade = np.random.uniform(low_delay_range[0], low_delay_range[1])
                    else:
                        actual_delay_grade = np.random.uniform(high_delay_range[0], high_delay_range[1])
                else:
                    actual_delay_grade = max(0.0, min(1.0, current_data[na_idx, 2]))
            elif na_idx in malicious_na_indices and attack_type == 'OOA':
                # OOA攻击：延迟等级在诚实和恶意阶段之间周期性变化
                if transaction % 2 == 0:  # 诚实阶段（偶数事务）
                    actual_delay_grade = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # 恶意阶段（奇数事务）
                    actual_delay_grade = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            elif na_idx in malicious_na_indices and attack_type == 'OSA':
                # OSA攻击：延迟等级在正常和恶意阶段之间周期性变化
                cycle_length = 10
                cycle_position = (transaction - 1) % (cycle_length * 2)
                if cycle_position < cycle_length:  # 前10个事务：正常行为
                    actual_delay_grade = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # 后10个事务：恶意行为
                    actual_delay_grade = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            else:
                # 其他情况：累积变化
                actual_delay_grade = max(0.0, min(1.0, current_data[na_idx, 2] + delay_increase))
            current_transaction_delay_grades.append(actual_delay_grade)
            
            # 检查延迟度是否达到1.0，如果是则强制事务失败
            if force_fail_on_max_delay and actual_delay_grade >= 1.0:
                success = False
                print(f"[DELAY] Phase2 NA{na_idx} 延迟度达到{actual_delay_grade:.3f}，事务强制失败")
            else:
                # 模拟事务成功/失败
                if success_prob is None:  # 恶意NA，使用直接控制
                    success = malicious_success
                else:  # 正常NA，使用概率判断
                    success = np.random.random() < success_prob
            
            # 计算信誉变化
            computed_reward = calculate_computed_reward(
                current_data[na_idx, 1], 
                current_data[na_idx, 0], 
                current_data[na_idx, 3]
            )
            
            # 更新total_tx和success_count
            selected_total_tx[na_idx] += 1
            if success:
                selected_success_count[na_idx] += 1
                reputation_change = computed_reward * 20
                current_data[na_idx, 0] = min(10000, current_data[na_idx, 0] + reputation_change)
            else:
                reputation_change = computed_reward * 100
                current_data[na_idx, 0] = max(3000, current_data[na_idx, 0] - reputation_change)
            
            # 成功率现在完全基于累积事务数计算，不再直接更新current_data[:, 1]
            # current_data[:, 1]将在需要时通过selected_success_count/selected_total_tx动态计算
            
            # 更新延迟等级
            if na_idx in malicious_na_indices and attack_mode == 'delay_only':
                current_data[na_idx, 2] = actual_delay_grade
            elif na_idx in malicious_na_indices and attack_type == 'OOA':
                # OOA攻击：延迟等级在诚实和恶意阶段之间周期性变化，而不是累积
                if transaction % 2 == 0:  # 诚实阶段（偶数事务）
                    current_data[na_idx, 2] = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # 恶意阶段（奇数事务）
                    current_data[na_idx, 2] = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            elif na_idx in malicious_na_indices and attack_type == 'OSA':
                # OSA攻击：延迟等级在正常和恶意阶段之间周期性变化，而不是累积
                cycle_length = 10
                cycle_position = (transaction - 1) % (cycle_length * 2)
                if cycle_position < cycle_length:  # 前10个事务：正常行为
                    current_data[na_idx, 2] = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # 后10个事务：恶意行为
                    current_data[na_idx, 2] = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            else:
                # 其他情况：累积变化
                current_data[na_idx, 2] = max(0.0, min(1.0, current_data[na_idx, 2] + delay_increase))
            
            # 饥饿度变化将在事务循环结束后统一更新
            
            # [COLLECT] 收集滑动窗口数据 - 与test.py保持一致：添加饥饿度字段
            transaction_data = {
                'success': success,
                'reputation_before': current_data[na_idx, 0] - reputation_change if success else current_data[na_idx, 0] + abs(reputation_change),
                'reputation_after': current_data[na_idx, 0],
                'delay': current_data[na_idx, 2],  # 直接使用延迟等级（0.0-1.0）
                'hunger': current_data[na_idx, 3],  # 添加饥饿度字段
                'step': current_step_count + transaction
            }
            # 使用原始NA索引和全局窗口系统
            orig_idx = selected_indices_phase2[na_idx]
            _update_na_window_queue(orig_idx, transaction_data, na_current_pack_global, current_step_count + transaction)
        
        # 更新所有10个NA的饥饿度（被选中的5个NA饥饿度重置为0，其他5个NA饥饿度增长）
        _update_hunger_for_all_nas(selected_indices_phase2, global_current_step + transaction, phase1_data, 'phase2')
        global_current_step += 1
        
        # 同步被选中NA的饥饿度到当前数据
        for i, orig_idx in enumerate(selected_indices_phase2):
            current_data[i, 3] = phase1_data[orig_idx, 3]
        
        # [PACK] 定期打包滑动窗口数据（使用全局窗口系统）
        if transaction % window_pack_interval == 0:
            _pack_all_na_windows(na_window_queues_global, na_current_pack_global, current_step_count + transaction)
            print(f"[PACK] Phase 2 - Transaction {transaction}: Window data packed for all NAs")
        
        # 保存当前事务的实际成功率和延迟等级
        transaction_actual_success_rates.append(current_transaction_success_rates.copy())
        transaction_actual_delay_grades.append(current_transaction_delay_grades.copy())
        
        # 记录当前状态
        if transaction % 5 == 0:  # 每5个事务记录一次
            phase2_evolution['time_points'].append(50 + transaction)
            phase2_evolution['reputations'].append(current_data[:, 0].copy())
            # 使用动态计算的成功率
            current_success_rates = np.where(selected_total_tx > 0, 
                                            selected_success_count / selected_total_tx, 
                                            current_data[:, 1])
            phase2_evolution['success_rates'].append(current_success_rates.copy())
            phase2_evolution['delay_grades'].append(current_data[:, 2].copy())  # 添加延迟等级数据
            phase2_evolution['hunger_levels'].append(current_data[:, 3].copy())  # 添加饥饿度数据
            # 使用当前事务的实际成功率而不是基础成功率
            phase2_evolution['actual_success_rates'].append(current_transaction_success_rates.copy())
            
            # [SUMMARY] 记录窗口统计摘要（使用全局窗口系统和原始NA索引）
            window_summary = {}
            weighted_success_rates = []
            weighted_delay_grades = []
            for i in range(5):
                orig_idx = selected_indices_phase2[i]  # 使用原始NA索引
                summary = _get_na_window_summary(orig_idx, na_window_queues_global)
                window_summary[i] = summary
                weighted_success_rates.append(summary['weighted_success_rate'])
                weighted_delay_grades.append(summary['weighted_delay_grade'])
            phase2_evolution['window_summaries'].append(window_summary)
            phase2_evolution['weighted_success_rates'].append(weighted_success_rates.copy())  # 添加加权成功率数据
            phase2_evolution['weighted_delay_grades'].append(weighted_delay_grades.copy())  # 添加加权延迟等级数据
    
    # 保存完整的事务实际成功率和延迟等级数据
    phase2_evolution['transaction_actual_success_rates'] = transaction_actual_success_rates
    phase2_evolution['transaction_actual_delay_grades'] = transaction_actual_delay_grades
    
    print(f"[DONE] Phase 2 Transaction Simulation Completed!")
    print(f"[INFO] Phase 2 Final State:")
    for i in range(5):
        initial_rep = phase2_evolution['reputations'][0][i]
        final_rep = phase2_evolution['reputations'][-1][i]
        change = final_rep - initial_rep
        print(f"   {phase2_evolution['na_labels'][i]}: Reputation {initial_rep:.0f} → {final_rep:.0f} (Change: {change:+.0f})")
    
    # 🔄 将第二阶段更新的5个NA数据回写到原始10个NA数组中
    print(f"\n🔄 Updating original 10 NAs with Phase 2 results...")
    updated_10_na_data = phase1_data.copy()
    for i, orig_idx in enumerate(selected_indices_phase2):
        updated_10_na_data[orig_idx] = current_data[i]
        print(f"   NA{orig_idx}: Updated with Phase 2 results (Reputation: {current_data[i, 0]:.0f})")
    
    # 第三阶段：从更新后的10个NA中选择5个NA
    print(f"\n🎯 第三阶段：从更新后的10个NA中选择5个NA...")
    
    # 使用更新后的数据进行选择
    phase1_final_data = updated_10_na_data.copy()
    normalized_data = normalize_features(phase1_final_data)
    
    # 创建测试用例格式以便使用predict_with_model函数
    phase3_test_case = {
        'raw_data': phase1_final_data,
        'normalized_data': normalized_data
    }
    
    # 第三阶段直接使用实时数据计算Q值（不使用原始预测数据）
    print(f"🔧 第三阶段使用实时数据计算Q值...")
    
    # 使用全局滑动窗口计算加权特征数据
    weighted_data_phase3 = np.zeros((10, 4))
    for na_idx in range(10):
        summary = _get_na_window_summary(na_idx, na_window_queues_global)
        weighted_data_phase3[na_idx, 0] = phase1_final_data[na_idx, 0]  # 信誉值使用当前值
        weighted_data_phase3[na_idx, 1] = summary['weighted_success_rate']  # 使用加权成功率
        weighted_data_phase3[na_idx, 2] = summary['weighted_delay_grade']  # 使用加权延迟等级
        weighted_data_phase3[na_idx, 3] = phase1_final_data[na_idx, 3]  # 饥饿度使用当前值
    
    # 使用实时加权特征数据进行模型预测
    normalized_data_phase3 = normalize_features(weighted_data_phase3)
    input_tensor = torch.FloatTensor(normalized_data_phase3).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = model(input_tensor)
        q_values_np = q_values.cpu().numpy().flatten()
    
    print(f"   实时Q值范围: {q_values_np.min():.4f} ~ {q_values_np.max():.4f}")
    
    # 选择3个高信誉和2个低信誉的NA（严格按照6600分界线）
    reputations = phase1_final_data[:, 0]
    
    # 分离高信誉（≥6600）和低信誉（<6600）NA
    high_rep_indices = np.where(reputations >= 6600)[0]
    low_rep_indices = np.where(reputations < 6600)[0]
    
    print(f"📊 第三阶段选择前信誉分布: 高信誉NA(≥6600): {len(high_rep_indices)}个, 低信誉NA(<6600): {len(low_rep_indices)}个")
    
    # 确保有足够的NA进行选择
    if len(high_rep_indices) < 3:
        print(f"⚠️ 警告: 高信誉NA不足3个，仅有{len(high_rep_indices)}个")
        sorted_indices = np.argsort(reputations)[::-1]
        high_rep_selected_phase3 = sorted_indices[:3]
    else:
        high_rep_scores = q_values_np[high_rep_indices]
        high_rep_selected_phase3 = high_rep_indices[np.argsort(high_rep_scores)[-3:]]
    
    if len(low_rep_indices) < 2:
        print(f"⚠️ 警告: 低信誉NA不足2个，仅有{len(low_rep_indices)}个")
        remaining_indices = np.setdiff1d(np.arange(len(reputations)), high_rep_selected_phase3)
        if len(remaining_indices) >= 2:
            remaining_reputations = reputations[remaining_indices]
            sorted_remaining = remaining_indices[np.argsort(remaining_reputations)]
            low_rep_selected_phase3 = sorted_remaining[:2]
        else:
            low_rep_selected_phase3 = remaining_indices
    else:
        low_rep_scores = q_values_np[low_rep_indices]
        low_rep_selected_phase3 = low_rep_indices[np.argsort(low_rep_scores)[-2:]]
    
    selected_indices_phase3 = np.concatenate([high_rep_selected_phase3, low_rep_selected_phase3])
    selected_data_phase3 = phase1_final_data[selected_indices_phase3]
    
    # 第三阶段取消攻击持续性机制，不再继承第二阶段的恶意NA
    # 第三阶段所有NA都按正常逻辑运行，不设置恶意NA
    malicious_na_indices_phase3 = []
    
    # 保持向后兼容
    malicious_na_idx_phase3 = malicious_na_indices_phase3[0] if len(malicious_na_indices_phase3) > 0 else None
    
    print(f"📊 第三阶段选择的5个NA:")
    for i, orig_idx in enumerate(selected_indices_phase3):
        na_type = "高信誉" if i < 3 else "低信誉"  # 第三阶段取消恶意行为，只显示高信誉和低信誉
        print(f"   NA{i} (原索引{orig_idx}, {na_type}): 信誉={selected_data_phase3[i, 0]:.0f}, Q值={q_values_np[orig_idx]:.4f}")
    
    # 第三阶段：5个新选定NA的事务模拟
    print(f"\n🚀 第三阶段：开始{n_transactions}个事务的模拟...")
    current_data_phase3 = selected_data_phase3.copy()
    
    # 初始化第三阶段的total_tx和success_count
    selected_total_tx_p3 = np.zeros(5, dtype=int)
    selected_success_count_p3 = np.zeros(5, dtype=int)
    
    for i, orig_idx in enumerate(selected_indices_phase3):
        if orig_idx in selected_indices_phase2:
            # 如果在第二阶段被选中过，继承其数据
            old_idx = np.where(selected_indices_phase2 == orig_idx)[0][0]
            selected_total_tx_p3[i] = selected_total_tx[old_idx]
            selected_success_count_p3[i] = selected_success_count[old_idx]
        else:
            # 如果是新选择的NA，基于当前成功率随机生成
            selected_total_tx_p3[i] = np.random.randint(20, 81)
            selected_success_count_p3[i] = int(current_data_phase3[i, 1] * selected_total_tx_p3[i])
        print(f"   NA{i} (原索引{orig_idx}): total_tx={selected_total_tx_p3[i]}, success_count={selected_success_count_p3[i]}")
    
    # 🚀 使用全局滑动窗口系统（不再重新初始化或继承）
    print("[INFO] 第三阶段使用全局滑动窗口系统，所有NA的历史窗口数据将被继续使用")
    for new_idx, orig_idx in enumerate(selected_indices_phase3):
        summary = _get_na_window_summary(orig_idx, na_window_queues_global)
        print(f"   NA{new_idx} (原索引{orig_idx}): 窗口队列长度={len(na_window_queues_global[orig_idx])}, 加权成功率={summary['weighted_success_rate']:.3f}")
    
    current_step_count = 100  # 第三阶段从步数100开始
    
    phase3_evolution = {
        'time_points': [100],  # 从100开始
        'reputations': [],
        'success_rates': [],
        'delay_grades': [],  # 添加延迟等级数据
        'hunger_levels': [],  # 添加饥饿度数据
        'weighted_success_rates': [],  # 添加加权成功率数据
        'actual_success_rates': [],  # 添加实际成功率数据
        'weighted_delay_grades': [],  # 添加加权延迟等级数据
        'na_labels': [],
        'malicious_na_idx': malicious_na_idx_phase3,
        'malicious_na_indices': malicious_na_indices_phase3,  # 添加恶意NA索引列表
        'selected_indices': selected_indices_phase3,
        'na_count': 5,
        'phase': 'transaction_simulation_phase3',
        'window_summaries': []  # 添加窗口统计摘要
    }
    
    # Generate labels for selected 5 NAs in phase 3
    for i in range(5):
        orig_idx = selected_indices_phase3[i]  # 使用原始NA索引
        if i == malicious_na_idx_phase3:
            phase3_evolution['na_labels'].append(f'CA{orig_idx} (Malicious)')
        else:
            phase3_evolution['na_labels'].append(f'CA{orig_idx}')
    
    # 记录初始状态
    phase3_evolution['reputations'].append(current_data_phase3[:, 0].copy())
    # 使用动态计算的成功率
    current_success_rates_p3 = np.where(selected_total_tx_p3 > 0, 
                                       selected_success_count_p3 / selected_total_tx_p3, 
                                       current_data_phase3[:, 1])
    phase3_evolution['success_rates'].append(current_success_rates_p3.copy())
    phase3_evolution['delay_grades'].append(current_data_phase3[:, 2].copy())  # 添加延迟等级数据
    phase3_evolution['hunger_levels'].append(current_data_phase3[:, 3].copy())  # 添加饥饿度数据
    phase3_evolution['actual_success_rates'].append(current_success_rates_p3.copy())  # 添加实际成功率数据
    
    # 记录初始的加权数据（使用全局窗口系统和原始NA索引）
    initial_weighted_success_rates = []
    initial_weighted_delay_grades = []
    for i in range(5):
        orig_idx = selected_indices_phase3[i]  # 使用原始NA索引
        summary = _get_na_window_summary(orig_idx, na_window_queues_global)
        initial_weighted_success_rates.append(summary['weighted_success_rate'])
        initial_weighted_delay_grades.append(summary['weighted_delay_grade'])
        print(f"   NA{i} (原索引{orig_idx}): 初始加权成功率={summary['weighted_success_rate']:.3f}, 加权延迟等级={summary['weighted_delay_grade']:.3f}")
    phase3_evolution['weighted_success_rates'].append(initial_weighted_success_rates.copy())
    phase3_evolution['weighted_delay_grades'].append(initial_weighted_delay_grades.copy())
    
    print(f"[INFO] Phase 3 Initial State:")
    for i in range(5):
        print(f"   {phase3_evolution['na_labels'][i]}: Reputation={current_data_phase3[i, 0]:.0f}, Success Rate={current_data_phase3[i, 1]:.3f}")
    
    # 模拟第三阶段的事务
    for transaction in range(1, n_transactions + 1):
        # 第三阶段：所有5个选中的NA都参与每次事务执行
        for na_idx in range(5):
            # 每个NA都执行事务
            # 第三阶段恶意行为逻辑
            if na_idx in malicious_na_indices_phase3:
                # 恶意NA：降低成功率
                success_prob = current_data_phase3[na_idx, 1] * 0.3  # 恶意行为：成功率降低到30%
                delay_increase = np.random.uniform(0.01, 0.05)  # 恶意NA增加更多延迟
                if transaction % 10 == 1:  # 每10个事务打印一次，避免输出过多
                    print(f"🚨 第三阶段恶意NA{na_idx}执行攻击，成功率从{current_data_phase3[na_idx, 1]:.3f}降低到{success_prob:.3f}")
            else:
                # 正常NA的行为
                success_prob = current_data_phase3[na_idx, 1]  # 使用固定的当前成功率，不再波动
                delay_increase = np.random.uniform(-0.02, 0.02)
            
            # 检查延迟度是否达到1.0，如果是则强制事务失败
            if force_fail_on_max_delay and current_data_phase3[na_idx, 2] >= 1.0:
                success = False
                print(f"[DELAY] Phase3 NA{na_idx} 延迟度达到{current_data_phase3[na_idx, 2]:.3f}，事务强制失败")
            else:
                # 模拟事务成功/失败
                success = np.random.random() < success_prob
            
            # 计算信誉变化
            computed_reward = calculate_computed_reward(
                current_data_phase3[na_idx, 1], 
                current_data_phase3[na_idx, 0], 
                current_data_phase3[na_idx, 3]
            )
            
            # 更新total_tx和success_count
            selected_total_tx_p3[na_idx] += 1
            if success:
                selected_success_count_p3[na_idx] += 1
                reputation_change = computed_reward * 20
                current_data_phase3[na_idx, 0] = min(10000, current_data_phase3[na_idx, 0] + reputation_change)
            else:
                reputation_change = computed_reward * 100
                current_data_phase3[na_idx, 0] = max(3000, current_data_phase3[na_idx, 0] - reputation_change)
            
            # 成功率现在完全基于累积事务数计算，不再直接更新current_data_phase3[:, 1]
            # current_data_phase3[:, 1]将在需要时通过selected_success_count_p3/selected_total_tx_p3动态计算
            
            # 更新延迟等级
            current_data_phase3[na_idx, 2] = max(0.0, min(1.0, current_data_phase3[na_idx, 2] + delay_increase))
            
            # 饥饿度变化将在事务循环结束后统一更新
            
            # 收集滑动窗口数据 - 与test.py保持一致：添加饥饿度字段
            transaction_data = {
                'success': success,
                'reputation_before': current_data_phase3[na_idx, 0] - reputation_change if success else current_data_phase3[na_idx, 0] + abs(reputation_change),
                'reputation_after': current_data_phase3[na_idx, 0],
                'delay': current_data_phase3[na_idx, 2],  # 直接使用延迟等级（0.0-1.0）
                'hunger': current_data_phase3[na_idx, 3],  # 添加饥饿度字段
                'step': current_step_count + transaction
            }
            # 使用原始NA索引更新全局窗口系统
            orig_idx = selected_indices_phase3[na_idx]
            _update_na_window_queue(orig_idx, transaction_data, na_current_pack_global, current_step_count + transaction)
        
        # 更新所有10个NA的饥饿度（被选中的5个NA饥饿度重置为0，其他5个NA饥饿度增长）
        _update_hunger_for_all_nas(selected_indices_phase3, global_current_step + transaction, updated_10_na_data, 'phase3')
        global_current_step += 1
        
        # 同步被选中NA的饥饿度到当前数据
        for i, orig_idx in enumerate(selected_indices_phase3):
            current_data_phase3[i, 3] = updated_10_na_data[orig_idx, 3]
        
        # [PACK] 定期打包滑动窗口数据（使用全局窗口系统）
        if transaction % window_pack_interval == 0:
            _pack_all_na_windows(na_window_queues_global, na_current_pack_global, current_step_count + transaction)
            print(f"[PACK] Phase 3 - Transaction {transaction}: Window data packed for all NAs (global system)")
        
        # 记录当前状态
        if transaction % 5 == 0:  # 每5个事务记录一次
            phase3_evolution['time_points'].append(100 + transaction)
            phase3_evolution['reputations'].append(current_data_phase3[:, 0].copy())
            # 使用动态计算的成功率
            current_success_rates_p3 = np.where(selected_total_tx_p3 > 0, 
                                               selected_success_count_p3 / selected_total_tx_p3, 
                                               current_data_phase3[:, 1])
            phase3_evolution['success_rates'].append(current_success_rates_p3.copy())
            phase3_evolution['delay_grades'].append(current_data_phase3[:, 2].copy())  # 添加延迟等级数据
            phase3_evolution['hunger_levels'].append(current_data_phase3[:, 3].copy())  # 添加饥饿度数据
            phase3_evolution['actual_success_rates'].append(current_success_rates_p3.copy())  # 添加实际成功率数据
            
            # [SUMMARY] 记录窗口统计摘要（使用全局窗口系统和原始NA索引）
            window_summary = {}
            weighted_success_rates = []
            weighted_delay_grades = []
            for i in range(5):
                orig_idx = selected_indices_phase3[i]  # 使用原始NA索引
                summary = _get_na_window_summary(orig_idx, na_window_queues_global)
                window_summary[i] = summary
                weighted_success_rates.append(summary['weighted_success_rate'])
                weighted_delay_grades.append(summary['weighted_delay_grade'])
            phase3_evolution['window_summaries'].append(window_summary)
            phase3_evolution['weighted_success_rates'].append(weighted_success_rates.copy())  # 添加加权成功率数据
            phase3_evolution['weighted_delay_grades'].append(weighted_delay_grades.copy())  # 添加加权延迟等级数据
    
    print(f"[DONE] Phase 3 Transaction Simulation Completed!")
    print(f"[INFO] Phase 3 Final State:")
    for i in range(5):
        initial_rep = phase3_evolution['reputations'][0][i]
        final_rep = phase3_evolution['reputations'][-1][i]
        change = final_rep - initial_rep
        print(f"   {phase3_evolution['na_labels'][i]}: Reputation {initial_rep:.0f} → {final_rep:.0f} (Change: {change:+.0f})")
    
    # 🔄 导出第三阶段参数文件（使用实时数据计算的Q值）
    print(f"\n📊 导出第三阶段参数文件...")
    
    # 直接使用之前计算的实时Q值和数据
    export_na_parameters(
        raw_data=weighted_data_phase3,
        normalized_data=normalized_data_phase3,
        q_values_np=q_values_np,  # 使用实时数据计算的Q值
        phase_name='phase3_selection',
        attack_type=attack_type,
        na_window_queues=na_window_queues_global
    )
    
    print(f"✅ 已导出第三阶段参数文件（基于实时数据）")
    
    # 收集所有10个NA在各阶段的饥饿度和延迟等级数据
    all_hunger_data = []
    all_delay_grade_data = []
    
    # 保存原始的第一阶段饥饿度数据（在被_update_hunger_for_all_nas修改之前）
    original_phase1_hunger = test_case['raw_data'][:, 3].copy()  # 使用原始测试数据的饥饿度
    print(f"[DEBUG] 使用原始饥饿度数据: {original_phase1_hunger}")
    
    for na_idx in range(10):
        na_hunger_timeline = []
        na_delay_grade_timeline = []
        
        # Phase 1: 每个NA使用其原始的初始饥饿度值，延迟等级从phase1_data中获取
        na_initial_hunger = original_phase1_hunger[na_idx]  # 使用原始的初始饥饿度值
        if na_idx == 0:  # 只为第一个NA打印调试信息
            print(f"[DEBUG] NA_{na_idx:02d} 初始饥饿度: {na_initial_hunger:.4f}")
            print(f"[DEBUG] phase1_data[{na_idx}, 3] = {phase1_data[na_idx, 3]:.4f}")
        for t in range(51):
            # 在第一阶段，每个NA保持其初始饥饿度值
            na_hunger_timeline.append(na_initial_hunger)
            # 在第一阶段，所有NA的延迟等级都保持在初始值
            na_delay_grade_timeline.append(phase1_data[na_idx, 2])
        
        # [DEBUG] 打印第一阶段结束时的饥饿度状态
        if na_idx in [10, 15, 16, 18, 19]:  # 之前有问题的NA
            print(f"[DEBUG] NA_{na_idx:02d} Phase1结束饥饿度: {na_initial_hunger:.4f} (应该非零)")
        
        # Phase 2: 检查该NA是否被选中
        if na_idx in selected_indices_phase2:
            # 被选中的NA，饥饿度重置为0，延迟等级使用phase2数据
            selected_idx = np.where(selected_indices_phase2 == na_idx)[0][0]
            for t in range(len(phase2_evolution['time_points'])):
                na_hunger_timeline.append(0.0)  # 被选中的NA饥饿度为0
            for grades in phase2_evolution['delay_grades']:
                na_delay_grade_timeline.append(grades[selected_idx])
            # [DEBUG] 打印被选中NA的状态
            if na_idx in [10, 15, 16, 18, 19]:
                print(f"[DEBUG] NA_{na_idx:02d} 在Phase2被选中，饥饿度重置为0")
        else:
            # 未被选中的NA，饥饿度基于Phase 1结束时的值继续增长，延迟等级保持不变
            phase1_end_hunger = na_initial_hunger  # Phase 1结束时的饥饿值
            for t in range(len(phase2_evolution['time_points'])):
                # 计算从Phase 1结束后经过的步数
                steps_since_phase1_end = (t + 1) * 5  # phase2每5个事务记录一次，从第5步开始
                # 基于Phase 1结束时的饥饿值继续增长
                additional_hunger = np.log(1 + steps_since_phase1_end / 20.0) / np.log(11)
                hunger_value = min(1.0, phase1_end_hunger + additional_hunger * 0.5)  # 适当降低增长速度
                na_hunger_timeline.append(hunger_value)
                # 延迟等级保持初始值
                na_delay_grade_timeline.append(phase1_data[na_idx, 2])
        
        # Phase 3: 检查该NA是否被选中
        if na_idx in selected_indices_phase3:
            # 被选中的NA，饥饿度重置为0，延迟等级使用phase3数据
            selected_idx = np.where(selected_indices_phase3 == na_idx)[0][0]
            for t in range(len(phase3_evolution['time_points'])):
                na_hunger_timeline.append(0.0)  # 被选中的NA饥饿度为0
            for grades in phase3_evolution['delay_grades']:
                na_delay_grade_timeline.append(grades[selected_idx])
            # [DEBUG] 打印被选中NA的状态
            if na_idx in [10, 15, 16, 18, 19]:
                print(f"[DEBUG] NA_{na_idx:02d} 在Phase3被选中，饥饿度重置为0")
        else:
            # 未被选中的NA，饥饿度基于Phase 2结束时的值继续增长，延迟等级保持不变
            # 计算Phase 2结束时的饥饿值
            if na_idx in selected_indices_phase2:
                phase2_end_hunger = 0.0  # 如果在Phase 2被选中过，则从0开始
            else:
                # 如果在Phase 2也未被选中，则基于Phase 1的值计算Phase 2结束时的饥饿值
                phase1_end_hunger = na_initial_hunger
                phase2_steps = len(phase2_evolution['time_points']) * 5  # Phase 2总步数
                additional_hunger_phase2 = np.log(1 + phase2_steps / 20.0) / np.log(11)
                phase2_end_hunger = min(1.0, phase1_end_hunger + additional_hunger_phase2 * 0.5)
            
            for t in range(len(phase3_evolution['time_points'])):
                # 计算从Phase 2结束后经过的步数
                steps_since_phase2_end = (t + 1) * 5  # phase3每5个事务记录一次，从第5步开始
                # 基于Phase 2结束时的饥饿值继续增长
                additional_hunger = np.log(1 + steps_since_phase2_end / 20.0) / np.log(11)
                hunger_value = min(1.0, phase2_end_hunger + additional_hunger * 0.5)  # 适当降低增长速度
                na_hunger_timeline.append(hunger_value)
                # 延迟等级保持初始值
                na_delay_grade_timeline.append(phase1_data[na_idx, 2])
        
        all_hunger_data.append(na_hunger_timeline)
        all_delay_grade_data.append(na_delay_grade_timeline)
    
    # 收集所有20个NA在各阶段的状态数据
    na_states_records = []
    
    # 记录Phase1结束时的状态
    for na_idx in range(10):
        na_states_records.append({
            'NA_Index': na_idx,
            'Phase': 'Phase1_End',
            'Reputation': phase1_data[na_idx, 0],
            'Success_Rate': phase1_data[na_idx, 1],
            'Delay_Grade': phase1_data[na_idx, 2],
            'Hunger_Level': phase1_data[na_idx, 3],
            'Selected': 'No'
        })
    
    # 记录Phase2开始时被选中NA的状态
    for i, na_idx in enumerate(selected_indices_phase2):
        na_states_records.append({
            'NA_Index': na_idx,
            'Phase': 'Phase2_Start',
            'Reputation': phase2_evolution['reputations'][0][i],
            'Success_Rate': phase2_evolution['success_rates'][0][i],
            'Delay_Grade': phase2_evolution['weighted_delay_grades'][0][i],
            'Hunger_Level': phase2_evolution['hunger_levels'][0][i],
            'Selected': 'Yes'
        })
    
    # 记录Phase2结束时被选中NA的状态
    for i, na_idx in enumerate(selected_indices_phase2):
        na_states_records.append({
            'NA_Index': na_idx,
            'Phase': 'Phase2_End',
            'Reputation': phase2_evolution['reputations'][-1][i],
            'Success_Rate': phase2_evolution['success_rates'][-1][i],
            'Delay_Grade': phase2_evolution['weighted_delay_grades'][-1][i],
            'Hunger_Level': phase2_evolution['hunger_levels'][-1][i],
            'Selected': 'Yes'
        })
    
    # 记录Phase3开始时被选中NA的状态
    for i, na_idx in enumerate(selected_indices_phase3):
        na_states_records.append({
            'NA_Index': na_idx,
            'Phase': 'Phase3_Start',
            'Reputation': phase3_evolution['reputations'][0][i],
            'Success_Rate': phase3_evolution['success_rates'][0][i],
            'Delay_Grade': phase3_evolution['weighted_delay_grades'][0][i],
            'Hunger_Level': phase3_evolution['hunger_levels'][0][i],
            'Selected': 'Yes'
        })
    
    # 记录Phase3结束时被选中NA的状态
    for i, na_idx in enumerate(selected_indices_phase3):
        na_states_records.append({
            'NA_Index': na_idx,
            'Phase': 'Phase3_End',
            'Reputation': phase3_evolution['reputations'][-1][i],
            'Success_Rate': phase3_evolution['success_rates'][-1][i],
            'Delay_Grade': phase3_evolution['weighted_delay_grades'][-1][i],
            'Hunger_Level': phase3_evolution['hunger_levels'][-1][i],
            'Selected': 'Yes'
        })
    
    # 记录未被选中NA在各阶段的状态（基于phase1_data和饥饿度计算）
    all_selected = np.unique(np.concatenate([selected_indices_phase2, selected_indices_phase3]))
    unselected_nas = [i for i in range(10) if i not in all_selected]
    
    for na_idx in unselected_nas:
        # Phase2开始时的状态（未被选中）
        phase2_start_hunger = min(1.0, np.log(1 + 50/20.0) / np.log(11))
        na_states_records.append({
            'NA_Index': na_idx,
            'Phase': 'Phase2_Start',
            'Reputation': phase1_data[na_idx, 0],
            'Success_Rate': phase1_data[na_idx, 1],
            'Delay_Grade': phase1_data[na_idx, 2],
            'Hunger_Level': phase2_start_hunger,
            'Selected': 'No'
        })
        
        # Phase2结束时的状态（未被选中）
        phase2_end_hunger = min(1.0, np.log(1 + 100/20.0) / np.log(11))
        na_states_records.append({
            'NA_Index': na_idx,
            'Phase': 'Phase2_End',
            'Reputation': phase1_data[na_idx, 0],
            'Success_Rate': phase1_data[na_idx, 1],
            'Delay_Grade': phase1_data[na_idx, 2],
            'Hunger_Level': phase2_end_hunger,
            'Selected': 'No'
        })
        
        # 如果在Phase3也未被选中
        if na_idx not in selected_indices_phase3:
            # Phase3开始时的状态
            na_states_records.append({
                'NA_Index': na_idx,
                'Phase': 'Phase3_Start',
                'Reputation': phase1_data[na_idx, 0],
                'Success_Rate': phase1_data[na_idx, 1],
                'Delay_Grade': phase1_data[na_idx, 2],
                'Hunger_Level': phase2_end_hunger,
                'Selected': 'No'
            })
            
            # Phase3结束时的状态
            phase3_end_hunger = min(1.0, np.log(1 + 150/20.0) / np.log(11))
            na_states_records.append({
                'NA_Index': na_idx,
                'Phase': 'Phase3_End',
                'Reputation': phase1_data[na_idx, 0],
                'Success_Rate': phase1_data[na_idx, 1],
                'Delay_Grade': phase1_data[na_idx, 2],
                'Hunger_Level': phase3_end_hunger,
                'Selected': 'No'
            })
    
    # CSV export removed
    states_df = pd.DataFrame(na_states_records)
    states_csv_path = f'{attack_type}_na_states_by_phase.csv'
    # states_df.to_csv(states_csv_path, index=False)
    # print(f"[DATA] NA状态记录已保存到: {states_csv_path}")
    print(f"[INFO] 记录了{len(na_states_records)}条状态数据，包含所有10个NA在各阶段的变化")
    
    # 收集所有10个NA的加权成功率数据
    all_weighted_success_rates = []
    
    for na_idx in range(10):
        na_weighted_success_timeline = []
        
        # Phase 1: 从phase1_evolution中获取加权成功率数据
        for rates in phase1_evolution['weighted_success_rates']:
            na_weighted_success_timeline.append(rates[na_idx])
        
        # Phase 2: 检查该NA是否被选中
        if na_idx in selected_indices_phase2:
            # 被选中的NA，使用phase2的加权成功率数据（跳过第一个初始值）
            selected_idx = np.where(selected_indices_phase2 == na_idx)[0][0]
            for rates in phase2_evolution['weighted_success_rates'][1:]:  # 跳过初始值
                na_weighted_success_timeline.append(rates[selected_idx])
        else:
            # 未被选中的NA，使用窗口摘要数据（跳过第一个初始值）
            for t in range(1, len(phase2_evolution['time_points'])):  # 跳过初始值
                summary = _get_na_window_summary(na_idx, na_window_queues_global)
                na_weighted_success_timeline.append(summary['weighted_success_rate'])
        
        # Phase 3: 检查该NA是否被选中
        if na_idx in selected_indices_phase3:
            # 被选中的NA，使用phase3的加权成功率数据（跳过第一个初始值）
            selected_idx = np.where(selected_indices_phase3 == na_idx)[0][0]
            for rates in phase3_evolution['weighted_success_rates'][1:]:  # 跳过初始值
                na_weighted_success_timeline.append(rates[selected_idx])
        else:
            # 未被选中的NA，使用窗口摘要数据（跳过第一个初始值）
            for t in range(1, len(phase3_evolution['time_points'])):  # 跳过初始值
                summary = _get_na_window_summary(na_idx, na_window_queues_global)
                na_weighted_success_timeline.append(summary['weighted_success_rate'])
        
        all_weighted_success_rates.append(na_weighted_success_timeline)
    
    # 合并三阶段数据
    evolution_data = {
        'phase1': phase1_evolution,
        'phase2': phase2_evolution,
        'phase3': phase3_evolution,
        'phase_separator_1': 50,  # 第一个阶段分隔点（第一阶段结束，第二阶段开始）
        'phase_separator_2': 100,  # 第二个阶段分隔点（第二阶段结束，第三阶段开始）
        'attack_type': attack_type,
        'attack_mode': attack_mode,
        'all_hunger_data': all_hunger_data,
        'all_delay_grade_data': all_delay_grade_data,  # 添加所有20个NA的延迟等级数据
        'all_weighted_success_rates': all_weighted_success_rates,  # 添加所有20个NA的加权成功率数据
        'na_states_records': na_states_records  # 添加状态记录
    }
    
    return evolution_data

def export_phase1_reputation_data_to_csv(evolution_data, save_path=None):
    """
    导出第一阶段所有NA的信誉值变化数据到CSV文件
    
    Args:
        evolution_data: 包含phase1_evolution数据的字典
        save_path: 保存路径，如果为None则自动生成
    
    Returns:
        str: 保存的文件路径
    """
    if 'phase1' not in evolution_data:
        print("错误：evolution_data中没有phase1数据")
        return None
    
    phase1_data = evolution_data['phase1']
    
    if save_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'phase1_reputation_evolution_{timestamp}.csv'
    
    # 准备数据
    time_points = phase1_data['time_points']
    reputations = phase1_data['reputations']
    na_count = phase1_data['na_count']
    
    # 创建DataFrame
    data_dict = {'time_step': time_points}
    
    # 为每个NA添加信誉值列
    for na_idx in range(na_count):
        reputation_values = [reputations[t][na_idx] for t in range(len(time_points))]
        data_dict[f'NA{na_idx}_reputation'] = reputation_values
    
    df = pd.DataFrame(data_dict)
    
    # 保存到CSV
    df.to_csv(save_path, index=False, encoding='utf-8')
    
    print(f"第一阶段NA信誉值变化数据已导出到: {save_path}")
    print(f"数据包含 {na_count} 个NA在 {len(time_points)} 个时间步的信誉值变化")
    
    # 显示数据统计信息
    print("\n=== 第一阶段信誉值变化统计 ===")
    for na_idx in range(na_count):
        col_name = f'NA{na_idx}_reputation'
        initial_rep = df[col_name].iloc[0]
        final_rep = df[col_name].iloc[-1]
        change = final_rep - initial_rep
        print(f"NA{na_idx}: 初始={initial_rep:.1f}, 最终={final_rep:.1f}, 变化={change:+.1f}")
    
    return save_path

def export_hunger_data_to_csv(evolution_data, save_path=None):
    """
    导出所有NA的饥饿度变化数据到CSV文件
    
    Args:
        evolution_data: 包含饥饿度数据的演化数据字典
        save_path: 保存路径，如果为None则使用默认路径
    
    Returns:
        str: 保存的文件路径
    """
    if save_path is None:
        save_path = f"{evolution_data['attack_type']}_hunger_evolution.csv"
    
    # 获取饥饿度数据
    all_hunger_data = evolution_data['all_hunger_data']
    
    # 构建时间轴
    phase1_time = list(range(0, 51))  # Phase 1: 0-50
    phase2_time = list(range(51, 101))  # Phase 2: 51-100
    phase3_time = list(range(101, 151))  # Phase 3: 101-150
    all_time_points = phase1_time + phase2_time + phase3_time
    
    # 准备CSV数据
    csv_data = []
    
    # 为每个时间点创建一行数据
    for t_idx, time_point in enumerate(all_time_points):
        row = {'Time_Point': time_point}
        
        # 确定当前阶段
        if time_point <= 50:
            phase = 'Phase1'
        elif time_point <= 100:
            phase = 'Phase2'
        else:
            phase = 'Phase3'
        
        row['Phase'] = phase
        
        # 添加所有20个NA的饥饿度数据
        for na_idx in range(20):
            if t_idx < len(all_hunger_data[na_idx]):
                hunger_value = all_hunger_data[na_idx][t_idx]
            else:
                # 如果数据不足，使用最后一个值
                hunger_value = all_hunger_data[na_idx][-1] if all_hunger_data[na_idx] else 0.0
            
            row[f'NA_{na_idx:02d}_Hunger'] = hunger_value
        
        csv_data.append(row)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(csv_data)
    df.to_csv(save_path, index=False)
    
    print(f"[EXPORT] 饥饿度数据已导出到: {save_path}")
    print(f"[INFO] 导出了{len(csv_data)}个时间点的数据，包含所有20个NA的饥饿度变化")
    print(f"[INFO] 数据范围: 时间点 {all_time_points[0]} 到 {all_time_points[-1]}")
    
    # 显示数据统计信息
    print(f"\n[STATS] 饥饿度数据统计:")
    for na_idx in range(20):
        hunger_values = [row[f'NA_{na_idx:02d}_Hunger'] for row in csv_data]
        min_hunger = min(hunger_values)
        max_hunger = max(hunger_values)
        avg_hunger = sum(hunger_values) / len(hunger_values)
        print(f"   NA_{na_idx:02d}: 最小={min_hunger:.3f}, 最大={max_hunger:.3f}, 平均={avg_hunger:.3f}")
    
    return save_path

def select_nas_for_simulation(test_case, prediction):
    """
    从测试数据中选择5个NA进行模拟（3个高信誉 + 2个低信誉）
    
    Args:
        test_case: 测试用例
        prediction: 预测结果
    
    Returns:
        selected_data: 选中的5个NA数据
        selected_indices: 选中的NA索引
        malicious_na_idx: 恶意NA在选中列表中的索引
    """
    raw_data = test_case['raw_data']
    reputations = raw_data[:, 0]
    
    # 检查是否为攻击场景
    is_attack_scenario = 'attack_type' in test_case and test_case['attack_type'] != 'normal'
    
    # 检查是否有多个恶意节点（ME攻击场景）
    attack_node_indices = test_case.get('attack_node_indices', [])
    if is_attack_scenario and len(attack_node_indices) > 0:
        if test_case['attack_type'] == 'ME' and len(attack_node_indices) == 2:
            # ME攻击场景：两个恶意节点（高信誉组和低信誉组各一个）
            print(f"🎯 ME攻击场景检测到，恶意节点索引: {attack_node_indices}")
            
            # 按信誉排序（排除恶意节点）
            available_indices = np.arange(len(reputations))
            available_indices = np.setdiff1d(available_indices, attack_node_indices)
            available_reputations = reputations[available_indices]
            sorted_available = available_indices[np.argsort(available_reputations)[::-1]]
            
            # 选择3个正常NA（2个高信誉 + 1个低信誉）
            high_rep_pool = sorted_available[:max(2, len(sorted_available)//2)]
            high_rep_selected = np.random.choice(high_rep_pool, size=min(2, len(high_rep_pool)), replace=False)
            
            low_rep_pool = sorted_available[-max(1, len(sorted_available)//3):]
            low_rep_selected = np.random.choice(low_rep_pool, size=min(1, len(low_rep_pool)), replace=False)
            
            # 合并选择的NA（两个恶意节点放在最后）
            selected_indices = np.concatenate([high_rep_selected, low_rep_selected, attack_node_indices])
            malicious_na_idx = [3, 4]  # 两个恶意节点在选中列表的最后两个位置
            
            print(f"🎯 选择的5个NA (ME攻击场景):")
            print(f"   高信誉NA: {high_rep_selected} (信誉: {reputations[high_rep_selected]})")
            print(f"   低信誉NA: {low_rep_selected} (信誉: {reputations[low_rep_selected]})")
            print(f"   恶意NA: {attack_node_indices} (信誉: {reputations[attack_node_indices]}) - ME攻击")
            
        else:
            # 其他攻击场景：单个恶意节点
            attack_node_idx = attack_node_indices[0]
            print(f"🎯 攻击场景检测到，恶意节点索引: {attack_node_idx}")
            
            # 按信誉排序（排除恶意节点）
            available_indices = np.arange(len(reputations))
            available_indices = available_indices[available_indices != attack_node_idx]
            available_reputations = reputations[available_indices]
            sorted_available = available_indices[np.argsort(available_reputations)[::-1]]
            
            # 选择4个正常NA（2个高信誉 + 2个低信誉）
            high_rep_pool = sorted_available[:max(2, len(sorted_available)//3)]
            high_rep_selected = np.random.choice(high_rep_pool, size=min(2, len(high_rep_pool)), replace=False)
            
            low_rep_pool = sorted_available[-max(2, len(sorted_available)//3):]
            low_rep_selected = np.random.choice(low_rep_pool, size=min(2, len(low_rep_pool)), replace=False)
            
            # 合并选择的NA（恶意节点放在最后）
            selected_indices = np.concatenate([high_rep_selected, low_rep_selected, [attack_node_idx]])
            malicious_na_idx = 4  # 恶意节点在选中列表的最后一个位置
            
            print(f"🎯 选择的5个NA (攻击场景):")
            print(f"   高信誉NA: {high_rep_selected} (信誉: {reputations[high_rep_selected]})")
            print(f"   低信誉NA: {low_rep_selected} (信誉: {reputations[low_rep_selected]})")
            print(f"   恶意NA: {attack_node_idx} (信誉: {reputations[attack_node_idx]:.0f}) - {test_case['attack_type']}攻击")
        
    else:
        # 正常场景：严格按照6600信誉值分界线选择
        # 分离高信誉（≥6600）和低信誉（<6600）NA
        high_rep_indices = np.where(reputations >= 6600)[0]
        low_rep_indices = np.where(reputations < 6600)[0]
        
        print(f"📊 信誉分布: 高信誉NA(≥6600): {len(high_rep_indices)}个, 低信誉NA(<6600): {len(low_rep_indices)}个")
        
        # 确保有足够的NA进行选择
        if len(high_rep_indices) < 3:
            print(f"⚠️ 警告: 高信誉NA不足3个，仅有{len(high_rep_indices)}个")
            # 如果高信誉NA不足，从所有NA中选择信誉最高的3个
            sorted_indices = np.argsort(reputations)[::-1]
            high_rep_selected = sorted_indices[:3]
        else:
            # 从高信誉NA中随机选择3个
            high_rep_selected = np.random.choice(high_rep_indices, size=3, replace=False)
        
        if len(low_rep_indices) < 2:
            print(f"⚠️ 警告: 低信誉NA不足2个，仅有{len(low_rep_indices)}个")
            # 如果低信誉NA不足，从剩余NA中选择信誉最低的2个
            remaining_indices = np.setdiff1d(np.arange(len(reputations)), high_rep_selected)
            if len(remaining_indices) >= 2:
                remaining_reputations = reputations[remaining_indices]
                sorted_remaining = remaining_indices[np.argsort(remaining_reputations)]
                low_rep_selected = sorted_remaining[:2]
            else:
                low_rep_selected = remaining_indices
        else:
            # 从低信誉NA中随机选择2个
            low_rep_selected = np.random.choice(low_rep_indices, size=2, replace=False)
        
        # 合并选择的NA
        selected_indices = np.concatenate([high_rep_selected, low_rep_selected])
        
        # 随机选择一个作为恶意NA（优先从低信誉中选择）
        if len(low_rep_selected) > 0:
            malicious_na_idx = np.random.choice(range(3, 3 + len(low_rep_selected)))  # 在低信誉NA中选择
        else:
            malicious_na_idx = np.random.choice(range(len(selected_indices)))  # 如果没有低信誉NA，随机选择
        
        print(f"🎯 选择的5个NA (正常场景):")
        print(f"   高信誉NA: {high_rep_selected} (信誉: {reputations[high_rep_selected]})")
        print(f"   低信誉NA: {low_rep_selected} (信誉: {reputations[low_rep_selected]})")
        print(f"   模拟恶意NA: 选中列表中的第{malicious_na_idx}个 (原索引: {selected_indices[malicious_na_idx]})")
        
        # 验证选择结果
        high_count = np.sum(reputations[selected_indices] >= 6600)
        low_count = np.sum(reputations[selected_indices] < 6600)
        print(f"✅ 验证结果: 高信誉NA({high_count}个), 低信誉NA({low_count}个)")
    
    selected_data = raw_data[selected_indices]
    return selected_data, selected_indices, malicious_na_idx

def main():
    """
    Main function - Demonstrate how to test custom datasets with trained model
    """
    print("🎯 Testing Custom Dataset with Trained DQN Model")
    print("="*60)
    
    # Set device
    device, device_note = get_runtime_device(preferred_gpu_index=1)
    print(f"🚀 Using device: {device}")
    if device_note:
        print(device_note)
    
    # Model file paths (absolute paths)
    model_path = "/mnt/data/wy2024/models/policy_net_state_dict.pth"
    model_info_path = "/mnt/data/wy2024/models/model_info.pth"
    
    try:
        # 1. Load trained model
        model, model_info = load_trained_model(model_path, model_info_path, device)
        
        # 2. Attack scenario testing
        print("\n" + "="*40)
        print("🎯 Attack Scenario Testing")
        print("="*40)
        print("1. Malicious With Everyone (ME) - Malicious nodes continuously provide harmful services")
        print("2. On-Off Attack (OOA) - Nodes periodically switch between honest/malicious behavior")
        print("3. Opportunistic Service Attack (OSA) - Nodes provide high-quality service only when reputation drops")
        print("4. All attack scenarios comparison test")
        
        attack_choice = input("\nPlease select attack scenario (1-4): ").strip()
        
        if attack_choice in ['1', '2', '3']:
            attack_types = ['ME', 'OOA', 'OSA']
            attack_type = attack_types[int(attack_choice) - 1]
            
            n_na = 10  # 默认生成10个NA
            
            print(f"\n[GEN] Generating {attack_type} attack scenario data...")
            test_case = create_attack_scenario_dataset([attack_type], n_na)[0]
            test_cases = [test_case]
            
            # Use model for prediction
            print("\n[PREDICT] Starting prediction...")
            predictions = []
            for i, test_case in enumerate(test_cases):
                prediction = predict_with_model(model, test_case, device, 
                                               export_params=False, 
                                               phase_name='initial_prediction', 
                                               attack_type=test_case['attack_type'])
                predictions.append(prediction)
                
                print(f"[DONE] {test_case['attack_type']} scenario: NA count={test_case['n_na']}, Best choice=NA_{prediction['best_na_idx']}, Q-value={prediction['best_q_value']:.4f}")
            
            # Analyze prediction results
            analyze_predictions(test_cases, predictions)
            analyze_attack_resistance(test_cases, predictions)
            
            # Visualization removed
            
            # 5 NA transaction simulation (自动执行)
            if test_cases:
                print("\n" + "="*40)
                print("🎯 Select Transaction Attack Mode")
                print("="*40)
                print("1. behavior (existing)")
                print("2. delay_only (only delay manipulation)")
                mode_choice = input("\nPlease select mode (1-2): ").strip()
                attack_mode = 'delay_only' if mode_choice == '2' else 'behavior'
                force_fail_on_max_delay = (attack_mode == 'behavior')

                base_dir = "/mnt/data/wy2024/Malicious behavior experiment"
                output_dir = os.path.join(
                    base_dir,
                    "experiment_code_data_delay_only" if attack_mode == 'delay_only' else "experiment_code_data",
                )
                os.makedirs(output_dir, exist_ok=True)

                print("\n🎯 Starting 5 NA transaction simulation (3 high reputation + 2 low reputation, including 1 malicious NA)")
                test_case = test_cases[0]
                prediction = predictions[0]
                
                # Select 5 NAs for simulation
                selected_data, selected_indices, malicious_na_idx = select_nas_for_simulation(test_case, prediction)
                
                # Set simulation parameters (固定50个事务)
                n_transactions = 50
                
                # Start transaction simulation
                print(f"\n[SIM] Starting simulation of {n_transactions} transactions...")
                evolution_data = simulate_transaction_evolution(
                    model,
                    test_case,
                    device,
                    n_transactions,
                    test_case['attack_type'],
                    attack_mode=attack_mode,
                    me_delay_step=0.02,
                    low_delay_range=(0.0, 0.3),
                    high_delay_range=(0.7, 1.0),
                    osa_cycle_length=20,
                    force_fail_on_max_delay=force_fail_on_max_delay,
                )
                
                # Export phase 1 reputation data
                print("\n[EXPORT] Exporting phase 1 reputation evolution data...")
                export_phase1_reputation_data_to_csv(evolution_data, os.path.join(output_dir, f"{attack_type}_phase1_reputation_evolution.csv"))
                
                # Save full evolution data for visualization
                if attack_mode == 'delay_only':
                    pickle_path = os.path.join(output_dir, f"{attack_type}_evolution_data.pkl")
                else:
                    pickle_path = os.path.join(base_dir, f"{attack_type}_evolution_data.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(evolution_data, f)
                print(f"[EXPORT] Full evolution data saved to {pickle_path}")
                
                # CSV export removed
                # print("\n[EXPORT] Exporting hunger evolution data...")
                # export_hunger_data_to_csv(evolution_data, f"{attack_type}_hunger_evolution.csv")
        
        elif attack_choice == '4':
            # All attack scenarios comparison
            n_na = 20  # 默认每个场景20个NA
            
            print(f"\n[GEN] Generating all attack scenario data ({n_na} NAs per scenario)...")
            
            attack_types = ['normal', 'ME', 'OOA', 'OSA']
            test_cases = create_attack_scenario_dataset(attack_types, n_na)
            
            # Use model for prediction
            print("\n[PREDICT] Starting prediction...")
            predictions = []
            for i, test_case in enumerate(test_cases):
                prediction = predict_with_model(model, test_case, device)
                predictions.append(prediction)
                
                print(f"[DONE] {test_case['attack_type']} scenario: NA count={test_case['n_na']}, Best choice=NA_{prediction['best_na_idx']}, Q-value={prediction['best_q_value']:.4f}")
            
            # Analyze prediction results
            analyze_predictions(test_cases, predictions)
            analyze_attack_resistance(test_cases, predictions)
            
            # Visualization removed
            
            # 5 NA transaction simulation (自动对所有场景执行)
            print("\n" + "="*40)
            print("🎯 Select Transaction Attack Mode")
            print("="*40)
            print("1. behavior (existing)")
            print("2. delay_only (only delay manipulation)")
            mode_choice = input("\nPlease select mode (1-2): ").strip()
            attack_mode = 'delay_only' if mode_choice == '2' else 'behavior'
            force_fail_on_max_delay = (attack_mode == 'behavior')

            base_dir = "/mnt/data/wy2024/Malicious behavior experiment"
            output_dir = os.path.join(
                base_dir,
                "experiment_code_data_delay_only" if attack_mode == 'delay_only' else "experiment_code_data",
            )
            os.makedirs(output_dir, exist_ok=True)

            print("\n🎯 Starting 5 NA transaction simulation for all scenarios...")
            for i, test_case in enumerate(test_cases):
                attack_type = test_case['attack_type']
                prediction = predictions[i]
                
                print(f"\n🎯 Starting 5 NA transaction simulation for {attack_type} scenario")
                
                # Select 5 NAs for simulation
                selected_data, selected_indices, malicious_na_idx = select_nas_for_simulation(test_case, prediction)
                
                # Set simulation parameters (固定100个事务)
                n_transactions = 100
                
                # Start transaction simulation
                print(f"[SIM] Starting simulation of {n_transactions} transactions...")
                evolution_data = simulate_transaction_evolution(
                    model,
                    test_case,
                    device,
                    n_transactions,
                    attack_type,
                    attack_mode=attack_mode,
                    me_delay_step=0.02,
                    low_delay_range=(0.0, 0.3),
                    high_delay_range=(0.7, 1.0),
                    osa_cycle_length=20,
                    force_fail_on_max_delay=force_fail_on_max_delay,
                )
                
                # Export phase 1 reputation data
                print(f"[EXPORT] Exporting phase 1 reputation evolution data for {attack_type}...")
                export_phase1_reputation_data_to_csv(evolution_data, os.path.join(output_dir, f"{attack_type}_phase1_reputation_evolution.csv"))
                
                # Save full evolution data for visualization
                if attack_mode == 'delay_only':
                    pickle_path = os.path.join(output_dir, f"{attack_type}_evolution_data.pkl")
                else:
                    pickle_path = os.path.join(base_dir, f"{attack_type}_evolution_data.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(evolution_data, f)
                print(f"[EXPORT] Full evolution data saved to {pickle_path}")
                
                # CSV export removed
                # print(f"[EXPORT] Exporting hunger evolution data for {attack_type}...")
                # export_hunger_data_to_csv(evolution_data, f"{attack_type}_hunger_evolution.csv")
        
        else:
            print("[ERROR] Invalid selection")
    
        print("\n[DONE] Testing completed!")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
