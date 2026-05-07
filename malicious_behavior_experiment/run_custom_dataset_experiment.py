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

def get_output_root() -> str:
    configured = os.environ.get("DRL_DPKI_OUTPUT_DIR")
    if configured:
        return configured
    legacy = "/mnt/data/wy2024"
    if os.path.isdir(legacy):
        return legacy
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "outputs")

OUTPUT_ROOT = get_output_root()

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
    
    print(f"Created {len(test_cases)} test cases with four-group strategy")
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
        save_path = os.path.join(
            OUTPUT_ROOT,
            "malicious_behavior_experiment",
            "experiment_code_data",
            f"{attack_type}_{phase_name}_na_parameters.csv",
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    
    print(f"[INIT] Phase 1 hunger values set to similar but distinct values; baseline: {base_hunger:.3f}")
    print(f"[INIT] Hunger range: {np.min(phase1_data[:, 3]):.3f} - {np.max(phase1_data[:, 3]):.3f}")
    for i in range(10):
        print(f"   NA_{i:02d}: {phase1_data[i, 3]:.4f}")
    
    # Initialize cumulative transaction counters (used to compute success rate from actual outcomes)
    phase1_total_tx = np.zeros(10, dtype=int)  # Cumulative total transactions
    phase1_success_count = np.zeros(10, dtype=int)  # Cumulative successful transactions
    
    print("[INIT] Initialized cumulative transaction counters; success rate will be computed from actual outcomes")
    phase1_evolution = {
        'time_points': list(range(51)),  # 0-50, but actual transactions are generated from 1-50
        'reputations': [],
        'na_count': 10,
        'phase': 'window_filling'
    }
    
    # Initialize Phase 1 weighted success-rate collection
    phase1_evolution['weighted_success_rates'] = []
    # Initialize Phase 1 actual success-rate collection
    phase1_evolution['actual_success_rates'] = []
    
    # Assign a per-NA window strategy (based on high/low reputation grouping)
    na_window_strategies = {}
    
    # High-reputation group (NA0-NA4): 5 different success-rate patterns
    high_rep_strategies = ['up_down', 'down_up', 'constant', 'always_up', 'always_down']
    for i, na_idx in enumerate(range(5)):
        na_window_strategies[na_idx] = high_rep_strategies[i]
        print(f"[INIT] High-reputation group NA{na_idx} assigned strategy: {high_rep_strategies[i]}")
    
    # Low-reputation group (NA5-NA9): 5 different success-rate patterns
    low_rep_strategies = ['up_down', 'down_up', 'constant', 'always_up', 'always_down']
    for i, na_idx in enumerate(range(5, 10)):
        na_window_strategies[na_idx] = low_rep_strategies[i]
        print(f"[INIT] Low-reputation group NA{na_idx} assigned strategy: {low_rep_strategies[i]}")
    
    # Initialize per-NA transaction counters and current pack buffers
    na_transaction_counters = {na_idx: 0 for na_idx in range(10)}
    na_current_transactions = {na_idx: [] for na_idx in range(10)}
    
    # Simulate sliding-window filling: generate 1 transaction per time step
    for time_step in range(51):
        # Generate 1 transaction per NA and accumulate
        for na_idx in range(10):
            current_features = phase1_data[na_idx]
            
            # Generate a single transaction
            if time_step > 0:  # Skip initial time point
                # Generate transaction performance parameters
                strategy = na_window_strategies[na_idx]
                
                # Base performance parameters
                base_success_rate = current_features[1]
                base_delay = current_features[2]
                
                # Compute time progress (0 to 1)
                progress = time_step / 50.0
                
                # Compute success-probability adjustment factor based on strategy
                if strategy == 'up_down':
                    # Success rate rises then falls (inverted U-shape)
                    if time_step <= 25:
                        # First half rises
                        factor = 0.7 + 0.6 * (time_step / 25.0)
                    else:
                        # Second half falls
                        factor = 1.3 - 0.6 * ((time_step - 25) / 25.0)
                    
                elif strategy == 'down_up':
                    # Success rate falls then rises (U-shape)
                    if time_step <= 25:
                        # First half falls
                        factor = 1.3 - 0.6 * (time_step / 25.0)
                    else:
                        # Second half rises
                        factor = 0.7 + 0.6 * ((time_step - 25) / 25.0)
                    
                elif strategy == 'constant':
                    # Success rate stays constant
                    factor = 1.0
                    
                elif strategy == 'always_up':
                    # Success rate always rises
                    factor = 0.7 + 0.6 * progress
                    
                elif strategy == 'always_down':
                    # Success rate always falls
                    factor = 1.3 - 0.6 * progress
                    
                else:
                    # Default: keep base performance
                    factor = 1.0
                
                # Compute transaction success probability at current time step
                transaction_success_probability = base_success_rate * factor
                transaction_success_probability = max(0.1, min(1.0, transaction_success_probability))
                
                # Compute delay
                transaction_delay = base_delay * (2.0 - factor)
                transaction_delay = max(0.1, min(2.0, transaction_delay))
                
                # If delay reaches 1.0, force the transaction to fail
                if force_fail_on_max_delay and transaction_delay >= 1.0:
                    is_success = False
                    print(f"[DELAY] NA{na_idx} delay reached {transaction_delay:.3f}; transaction forced to fail")
                else:
                    # Determine success based on the strategy-adjusted probability
                    is_success = np.random.random() < transaction_success_probability
                
                # Update cumulative transaction counters
                phase1_total_tx[na_idx] += 1
                if is_success:
                    phase1_success_count[na_idx] += 1
                
                # Compute the NA's actual success rate (based on cumulative counts)
                current_actual_success_rate = phase1_success_count[na_idx] / phase1_total_tx[na_idx] if phase1_total_tx[na_idx] > 0 else 0.0
                
                # Compute reputation change
                computed_reward = calculate_computed_reward(
                    current_actual_success_rate, current_features[0], current_features[3]
                )
                
                if is_success:
                    reputation_change = computed_reward * 20
                else:
                    reputation_change = -computed_reward * 100
                
                # Update reputation value
                phase1_data[na_idx, 0] = max(3000, min(10000, 
                    phase1_data[na_idx, 0] + reputation_change * 0.1))
                
                # Append transaction into the current pack buffer
                na_current_transactions[na_idx].append({
                    'success': is_success,
                    'delay': transaction_delay,  # Delay value
                    'reputation_change': reputation_change * 0.1
                })
                
                na_transaction_counters[na_idx] += 1
                
                # Pack every 10 transactions (consistent with Phase 2/3)
                if na_transaction_counters[na_idx] % 10 == 0:
                    transactions = na_current_transactions[na_idx]
                    if transactions:
                        # Compute pack statistics
                        success_count = sum(1 for t in transactions if t['success'])
                        total_count = len(transactions)
                        success_rate = success_count / total_count if total_count > 0 else 0.0
                        avg_delay = np.mean([t['delay'] for t in transactions])
                        total_reputation_change = sum(t['reputation_change'] for t in transactions)
                        
                        # Build pack data
                        pack_data = {
                            'success_rate': success_rate,
                            'avg_delay': avg_delay,
                            'transaction_count': total_count,
                            'total_reputation_change': total_reputation_change,
                            'weight': total_count,  # Time weight equals transaction count
                            'start_step': time_step - 9,
                            'end_step': time_step
                        }
                        
                        # Append to global sliding-window queue
                        na_window_queues_global[na_idx].append(pack_data)
                        
                        # Clear current transaction list
                        na_current_transactions[na_idx] = []
        
        phase1_evolution['reputations'].append(phase1_data[:, 0].copy())
        
        # Collect actual success-rate data at each time step (computed from cumulative transaction counts)
        actual_success_rates = []
        for na_idx in range(10):
            # Compute actual success rate based on cumulative transaction counts
            if phase1_total_tx[na_idx] > 0:
                actual_success_rate = phase1_success_count[na_idx] / phase1_total_tx[na_idx]
            else:
                # If there are no transactions yet, use the initial success rate
                actual_success_rate = phase1_data[na_idx, 1]
            
            actual_success_rates.append(actual_success_rate)
        
        phase1_evolution['actual_success_rates'].append(actual_success_rates.copy())
        
        # Collect weighted success-rate data every 5 time steps (consistent with Phase 2/3)
        if time_step % 5 == 0:
            weighted_success_rates = []
            for na_idx in range(10):
                summary = _get_na_window_summary(na_idx, na_window_queues_global)
                weighted_success_rates.append(summary['weighted_success_rate'])
            phase1_evolution['weighted_success_rates'].append(weighted_success_rates.copy())
        
        current_step_count += 1
    
    print(
        f"Phase 1 completed. Reputation range for 10 NAs: {phase1_data[:, 0].min():.0f} - {phase1_data[:, 0].max():.0f}"
    )
    
    # Update phase1_data to the actual state at the end of Phase 1
    # Use the reputation values from the last time step of Phase 1
    final_reputations = phase1_evolution['reputations'][-1]  # Reputation values at the last time step
    phase1_data[:, 0] = final_reputations  # Update reputation values
    
    # Success rate is now computed purely from cumulative transaction counts.
    # phase1_data[:, 1] is computed on demand from phase1_success_count/phase1_total_tx.
    
    # Compute and update delay values (inverse relationship with success rate)
    for na_idx in range(10):
        strategy = na_window_strategies[na_idx]
        base_delay = test_case['raw_data'][na_idx, 2]  # Base delay from raw data
        
        # Compute final delay based on strategy
        if strategy == 'up_down':
            # Success rate decreases at the end, so delay should increase
            factor = 1.3 - 0.6 * ((50 - 25) / 25.0)  # Success-rate factor at the last time step
            delay_factor = 2.0 - factor  # Delay factor is the inverse of success-rate factor
        elif strategy == 'down_up':
            # Success rate increases at the end, so delay should decrease
            factor = 0.7 + 0.6 * ((50 - 25) / 25.0)  # Success-rate factor at the last time step
            delay_factor = 2.0 - factor  # Delay factor is the inverse of success-rate factor
        elif strategy == 'constant':
            delay_factor = 1.0  # No change
        elif strategy == 'always_up':
            # Success rate always rises, so delay always falls
            factor = 1.0 + 0.3 * 1.0  # Final success-rate factor
            delay_factor = 2.0 - factor  # Delay factor is the inverse of success-rate factor
        elif strategy == 'always_down':
            # Success rate always falls, so delay always rises
            factor = 1.0 - 0.3 * 1.0  # Final success-rate factor
            delay_factor = 2.0 - factor  # Delay factor is the inverse of success-rate factor
        else:
            delay_factor = 1.0
        
        final_delay = base_delay * delay_factor
        final_delay = max(0.1, min(2.0, final_delay))  # Clamp delay range
        phase1_data[na_idx, 2] = final_delay
    
    print("Updated phase1_data state:")
    print(f"   Reputation range: {phase1_data[:, 0].min():.0f} - {phase1_data[:, 0].max():.0f}")
    print(f"   Success-rate range: {phase1_data[:, 1].min():.3f} - {phase1_data[:, 1].max():.3f}")
    print(f"   Delay range: {phase1_data[:, 2].min():.3f} - {phase1_data[:, 2].max():.3f}")
    
    # Phase 2: model selects 5 NAs
    print("\nPhase 2: selecting 5 NAs out of 10...")
    print("Using global sliding-window weighted features for model prediction")
    
    # Use global sliding window to compute weighted feature data
    weighted_data = np.zeros((10, 4))
    for na_idx in range(10):
        summary = _get_na_window_summary(na_idx, na_window_queues_global)
        weighted_data[na_idx, 0] = phase1_data[na_idx, 0]  # Current reputation value
        weighted_data[na_idx, 1] = summary['weighted_success_rate']  # Weighted success rate
        weighted_data[na_idx, 2] = summary['weighted_delay_grade']  # Weighted delay grade
        weighted_data[na_idx, 3] = phase1_data[na_idx, 3]  # Current hunger level
        print(
            f"   NA{na_idx}: weighted_success_rate={summary['weighted_success_rate']:.3f}, "
            f"weighted_delay_grade={summary['weighted_delay_grade']:.3f}"
        )
    
    # Predict using weighted feature data
    normalized_data = normalize_features(weighted_data)
    input_tensor = torch.FloatTensor(normalized_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = model(input_tensor)
        q_values_np = q_values.cpu().numpy().flatten()
    
    # Export all NA parameters for Phase 2 (including Q values), passing global sliding-window data
    export_na_parameters(
        weighted_data,  # Raw data (weighted)
        normalized_data,  # Normalized data
        q_values_np,  # Q values
        'phase2_selection',  # Phase name
        attack_type,  # Attack type
        na_window_queues=na_window_queues_global  # Global sliding-window data
    )
    
    # Select 3 high-reputation and 2 low-reputation NAs (strict threshold at 6600)
    reputations = phase1_data[:, 0]
    
    # Split high-reputation (>= 6600) and low-reputation (< 6600) NAs
    high_rep_indices = np.where(reputations >= 6600)[0]
    low_rep_indices = np.where(reputations < 6600)[0]
    
    print(
        f"Post-Phase-1 reputation distribution: "
        f"high-reputation NAs (>=6600): {len(high_rep_indices)}, "
        f"low-reputation NAs (<6600): {len(low_rep_indices)}"
    )
    
    # Ensure there are enough NAs to select
    if len(high_rep_indices) < 3:
        print(f"[WARN] Not enough high-reputation NAs for selecting 3; only {len(high_rep_indices)} available")
        # If there are not enough high-reputation NAs, pick the top 3 by reputation
        sorted_indices = np.argsort(reputations)[::-1]
        high_rep_selected = sorted_indices[:3]
    else:
        # Select the 3 highest Q-value NAs from the high-reputation group
        high_rep_scores = q_values_np[high_rep_indices]
        high_rep_selected = high_rep_indices[np.argsort(high_rep_scores)[-3:]]
    
    if len(low_rep_indices) < 2:
        print(f"[WARN] Not enough low-reputation NAs for selecting 2; only {len(low_rep_indices)} available")
        # If there are not enough low-reputation NAs, pick the bottom 2 by reputation from the remaining set
        remaining_indices = np.setdiff1d(np.arange(len(reputations)), high_rep_selected)
        if len(remaining_indices) >= 2:
            remaining_reputations = reputations[remaining_indices]
            sorted_remaining = remaining_indices[np.argsort(remaining_reputations)]
            low_rep_selected = sorted_remaining[:2]
        else:
            low_rep_selected = remaining_indices
    else:
        # Select the 2 highest Q-value NAs from the low-reputation group
        low_rep_scores = q_values_np[low_rep_indices]
        low_rep_selected = low_rep_indices[np.argsort(low_rep_scores)[-2:]]
    
    selected_indices_phase2 = np.concatenate([high_rep_selected, low_rep_selected])
    selected_data = phase1_data[selected_indices_phase2]
    
    # Determine malicious NAs (based on attack type)
    if attack_type == 'ME' and 'attack_node_indices' in test_case:
        # ME scenario: choose one malicious NA from the high-rep group and one from the low-rep group
        malicious_na_indices = []
        
        # Pick one malicious NA from the high-rep group (indices 0-2)
        high_rep_malicious_idx = np.random.choice(range(3))  # Choose from the first 3 high-rep NAs
        malicious_na_indices.append(high_rep_malicious_idx)
        
        # Pick one malicious NA from the low-rep group
        if len(low_rep_selected) > 0:
            low_rep_malicious_idx = np.random.choice(range(3, 3 + len(low_rep_selected)))  # Choose from low-rep NAs
            malicious_na_indices.append(low_rep_malicious_idx)
        else:
            # If there are no low-rep NAs, pick another one from the remaining high-rep NAs
            remaining_high_rep = [i for i in range(3) if i != high_rep_malicious_idx]
            if remaining_high_rep:
                additional_malicious_idx = np.random.choice(remaining_high_rep)
                malicious_na_indices.append(additional_malicious_idx)
        
        malicious_na_idx = malicious_na_indices[0] if len(malicious_na_indices) > 0 else None  # Backward compatibility
    else:
        # Updated logic: choose one malicious NA from the high-rep group and one from the low-rep group
        malicious_na_indices = []
        
        # Pick one malicious NA from the high-rep group (indices 0-2)
        high_rep_malicious_idx = np.random.choice(range(3))  # Choose from the first 3 high-rep NAs
        malicious_na_indices.append(high_rep_malicious_idx)
        
        # Pick one malicious NA from the low-rep group
        if len(low_rep_selected) > 0:
            low_rep_malicious_idx = np.random.choice(range(3, 3 + len(low_rep_selected)))  # Choose from low-rep NAs
            malicious_na_indices.append(low_rep_malicious_idx)
        else:
            # If there are no low-rep NAs, pick another one from the remaining high-rep NAs
            remaining_high_rep = [i for i in range(3) if i != high_rep_malicious_idx]
            if remaining_high_rep:
                additional_malicious_idx = np.random.choice(remaining_high_rep)
                malicious_na_indices.append(additional_malicious_idx)
        
        # Backward compatibility: set malicious_na_idx to the first malicious NA index
        malicious_na_idx = malicious_na_indices[0] if len(malicious_na_indices) > 0 else None
    
    print("Selected 5 NAs:")
    for i, orig_idx in enumerate(selected_indices_phase2):
        na_type = "malicious" if i in malicious_na_indices else ("high-reputation" if i < 3 else "low-reputation")
        print(
            f"   NA{i} (orig idx {orig_idx}, {na_type}): "
            f"rep={selected_data[i, 0]:.0f}, Q={q_values_np[orig_idx]:.4f}"
        )
    
    # Phase 2: transaction simulation for the selected 5 NAs
    print(f"\nPhase 2: starting simulation of {n_transactions} transactions...")
    current_data = selected_data.copy()
    
    # Initialize total_tx and success_count for selected NAs
    if 'total_tx' in test_case and 'success_count' in test_case:
        selected_total_tx = test_case['total_tx'][selected_indices_phase2].copy()
        selected_success_count = test_case['success_count'][selected_indices_phase2].copy()
    else:
        # If missing, generate based on current success rate
        selected_total_tx = np.random.randint(20, 81, 5)
        # Use dynamically computed success rates when cumulative data exists; otherwise use the initial success rate
        current_success_rates = np.where(selected_total_tx > 0, 
                                        selected_success_count / selected_total_tx, 
                                        current_data[:, 1])
        selected_success_count = (current_success_rates * selected_total_tx).astype(int)
    
    # Use global sliding-window system (no re-initialization)
    current_step_count = 50  # Phase 2 starts from step 50
    print("[INFO] Phase 2 uses the global sliding window system; existing window data will be reused")
    
    phase2_evolution = {
        'time_points': [50],  # Start from 50, append dynamically
        'reputations': [],
        'success_rates': [],
        'delay_grades': [],  # Delay-grade series
        'hunger_levels': [],  # Hunger-level series
        'weighted_success_rates': [],  # Weighted success-rate series
        'weighted_delay_grades': [],  # Weighted delay-grade series
        'actual_success_rates': [],  # Actual success-rate series
        'na_labels': [],
        'malicious_na_idx': malicious_na_idx,
        'malicious_na_indices': malicious_na_indices,
        'selected_indices': selected_indices_phase2,
        'na_count': 5,
        'phase': 'transaction_simulation_phase2',
        'window_summaries': []  # Window summary series
    }
    
    # Generate labels for selected 5 NAs
    for i in range(5):
        orig_idx = selected_indices_phase2[i]  # Use original NA index
        if i == malicious_na_idx:
            phase2_evolution['na_labels'].append(f'CA{orig_idx} (Malicious)')
        else:
            phase2_evolution['na_labels'].append(f'CA{orig_idx}')
    
    # Record initial state
    phase2_evolution['reputations'].append(current_data[:, 0].copy())
    # Use dynamically computed success rates
    current_success_rates = np.where(selected_total_tx > 0, 
                                    selected_success_count / selected_total_tx, 
                                    current_data[:, 1])
    phase2_evolution['success_rates'].append(current_success_rates.copy())
    phase2_evolution['delay_grades'].append(current_data[:, 2].copy())  # Delay-grade series
    phase2_evolution['hunger_levels'].append(current_data[:, 3].copy())  # Hunger-level series
    phase2_evolution['actual_success_rates'].append(current_success_rates.copy())  # Actual success-rate series
    
    # Record initial weighted data (global window system + original NA index)
    initial_weighted_success_rates = []
    initial_weighted_delay_grades = []
    for i in range(5):
        orig_idx = selected_indices_phase2[i]  # Use original NA index
        summary = _get_na_window_summary(orig_idx, na_window_queues_global)
        initial_weighted_success_rates.append(summary['weighted_success_rate'])
        initial_weighted_delay_grades.append(summary['weighted_delay_grade'])
        print(
            f"   NA{i} (orig idx {orig_idx}): weighted_success_rate={summary['weighted_success_rate']:.3f}, "
            f"weighted_delay_grade={summary['weighted_delay_grade']:.3f}"
        )
    phase2_evolution['weighted_success_rates'].append(initial_weighted_success_rates.copy())
    phase2_evolution['weighted_delay_grades'].append(initial_weighted_delay_grades.copy())
    
    print(f"[INFO] Phase 2 Initial State:")
    for i in range(5):
        print(f"   {phase2_evolution['na_labels'][i]}: Reputation={current_data[i, 0]:.0f}, Success Rate={current_data[i, 1]:.3f}")
    
    # Simulate Phase 2 transactions (50 transactions; time points 51-100)
    # Record per-transaction actual success rate and delay grade
    transaction_actual_success_rates = []
    transaction_actual_delay_grades = []
    
    for transaction in range(1, n_transactions + 1):
        # Record actual success rate and delay grade for all NAs at the current transaction
        current_transaction_success_rates = []
        current_transaction_delay_grades = []
        
        # Phase 2: all 5 selected NAs participate in every transaction
        for na_idx in range(5):
            # Each NA executes a transaction
            if na_idx in malicious_na_indices and attack_mode == 'delay_only':
                success_prob = current_data[na_idx, 1]
                delay_increase = 0.0
                malicious_success = None
            elif na_idx in malicious_na_indices:
                # Implement malicious behavior patterns by attack type
                if attack_type == 'ME':
                    # ME: all transactions fail
                    malicious_success = False
                    delay_increase = 0.1  # Increase delay
                    
                elif attack_type == 'OOA':
                    # OOA: transactions alternate between success and failure
                    if transaction % 2 == 0:  # Even transactions: success
                        malicious_success = True
                        delay_increase = np.random.uniform(-0.02, 0.02)
                    else:  # Odd transactions: failure
                        malicious_success = False
                        delay_increase = 0.3
                        
                elif attack_type == 'OSA':
                    # OSA: periodic failure/success, switching every 10 transactions
                    cycle_length = 10
                    cycle_position = (transaction - 1) % (cycle_length * 2)  # 20 transactions per full cycle
                    if cycle_position < cycle_length:  # First 10: failure
                        malicious_success = False
                        delay_increase = 0.25
                    else:  # Last 10: success
                        malicious_success = True
                        delay_increase = np.random.uniform(-0.02, 0.02)
                        
                else:
                    # Default malicious behavior (ME-like)
                    malicious_success = False
                    delay_increase = 0.05
                    
                # For malicious NAs, do not use success_prob; force success/failure directly
                success_prob = None  # Mark as malicious NA; skip probabilistic decision
            else:
                # Normal NA behavior
                success_prob = current_data[na_idx, 1]  # Fixed current success rate, no fluctuation
                delay_increase = np.random.uniform(-0.02, 0.02)
                malicious_success = None  # Mark as normal NA
            
            # Record the NA's actual success rate and delay grade
            # Compute actual success rate from cumulative counts
            actual_success_rate = selected_success_count[na_idx] / selected_total_tx[na_idx] if selected_total_tx[na_idx] > 0 else 0.0
            current_transaction_success_rates.append(actual_success_rate)
            # Compute the transaction's actual delay grade
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
                # OOA attack: delay grade cycles between honest and malicious phases
                if transaction % 2 == 0:  # Honest phase (even transactions)
                    actual_delay_grade = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # Malicious phase (odd transactions)
                    actual_delay_grade = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            elif na_idx in malicious_na_indices and attack_type == 'OSA':
                # OSA attack: delay grade cycles between normal and malicious phases
                cycle_length = 10
                cycle_position = (transaction - 1) % (cycle_length * 2)
                if cycle_position < cycle_length:  # First 10 transactions: normal behavior
                    actual_delay_grade = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # Last 10 transactions: malicious behavior
                    actual_delay_grade = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            else:
                # Other cases: cumulative change
                actual_delay_grade = max(0.0, min(1.0, current_data[na_idx, 2] + delay_increase))
            current_transaction_delay_grades.append(actual_delay_grade)
            
            # If delay grade reaches 1.0, force transaction failure
            if force_fail_on_max_delay and actual_delay_grade >= 1.0:
                success = False
                print(f"[DELAY] Phase2 NA{na_idx} delay grade reached {actual_delay_grade:.3f}; forcing transaction failure")
            else:
                # Simulate transaction success/failure
                if success_prob is None:  # Malicious NA: direct control
                    success = malicious_success
                else:  # Normal NA: probabilistic decision
                    success = np.random.random() < success_prob
            
            # Compute reputation change
            computed_reward = calculate_computed_reward(
                current_data[na_idx, 1], 
                current_data[na_idx, 0], 
                current_data[na_idx, 3]
            )
            
            # Update total_tx and success_count
            selected_total_tx[na_idx] += 1
            if success:
                selected_success_count[na_idx] += 1
                reputation_change = computed_reward * 20
                current_data[na_idx, 0] = min(10000, current_data[na_idx, 0] + reputation_change)
            else:
                reputation_change = computed_reward * 100
                current_data[na_idx, 0] = max(3000, current_data[na_idx, 0] - reputation_change)
            
            # Success rate is computed from cumulative transaction counts.
            # current_data[:, 1] is computed on demand from selected_success_count / selected_total_tx.
            
            # Update delay grade
            if na_idx in malicious_na_indices and attack_mode == 'delay_only':
                current_data[na_idx, 2] = actual_delay_grade
            elif na_idx in malicious_na_indices and attack_type == 'OOA':
                # OOA attack: delay grade cycles between honest and malicious phases (not cumulative)
                if transaction % 2 == 0:  # Honest phase (even transactions)
                    current_data[na_idx, 2] = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # Malicious phase (odd transactions)
                    current_data[na_idx, 2] = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            elif na_idx in malicious_na_indices and attack_type == 'OSA':
                # OSA attack: delay grade cycles between normal and malicious phases (not cumulative)
                cycle_length = 10
                cycle_position = (transaction - 1) % (cycle_length * 2)
                if cycle_position < cycle_length:  # First 10 transactions: normal behavior
                    current_data[na_idx, 2] = max(0.0, min(0.3, 0.1 + np.random.uniform(-0.05, 0.05)))
                else:  # Last 10 transactions: malicious behavior
                    current_data[na_idx, 2] = max(0.7, min(1.0, 0.9 + np.random.uniform(-0.1, 0.1)))
            else:
                # Other cases: cumulative change
                current_data[na_idx, 2] = max(0.0, min(1.0, current_data[na_idx, 2] + delay_increase))
            
            # Hunger level updates are applied after the transaction loop
            
            # [COLLECT] Collect sliding-window data (aligned with test.py): include hunger field
            transaction_data = {
                'success': success,
                'reputation_before': current_data[na_idx, 0] - reputation_change if success else current_data[na_idx, 0] + abs(reputation_change),
                'reputation_after': current_data[na_idx, 0],
                'delay': current_data[na_idx, 2],  # Use delay grade directly (0.0-1.0)
                'hunger': current_data[na_idx, 3],  # Include hunger field
                'step': current_step_count + transaction
            }
            # Use original NA index and the global window system
            orig_idx = selected_indices_phase2[na_idx]
            _update_na_window_queue(orig_idx, transaction_data, na_current_pack_global, current_step_count + transaction)
        
        # Update hunger levels for all 10 NAs (selected 5 reset to 0; others increase)
        _update_hunger_for_all_nas(selected_indices_phase2, global_current_step + transaction, phase1_data, 'phase2')
        global_current_step += 1
        
        # Sync selected NAs' hunger levels back to current_data
        for i, orig_idx in enumerate(selected_indices_phase2):
            current_data[i, 3] = phase1_data[orig_idx, 3]
        
        # [PACK] Periodically pack sliding-window data (using the global window system)
        if transaction % window_pack_interval == 0:
            _pack_all_na_windows(na_window_queues_global, na_current_pack_global, current_step_count + transaction)
            print(f"[PACK] Phase 2 - Transaction {transaction}: Window data packed for all NAs")
        
        # Store this transaction's actual success rates and delay grades
        transaction_actual_success_rates.append(current_transaction_success_rates.copy())
        transaction_actual_delay_grades.append(current_transaction_delay_grades.copy())
        
        # Record current state
        if transaction % 5 == 0:  # Record every 5 transactions
            phase2_evolution['time_points'].append(50 + transaction)
            phase2_evolution['reputations'].append(current_data[:, 0].copy())
            # Use dynamically computed success rates
            current_success_rates = np.where(selected_total_tx > 0, 
                                            selected_success_count / selected_total_tx, 
                                            current_data[:, 1])
            phase2_evolution['success_rates'].append(current_success_rates.copy())
            phase2_evolution['delay_grades'].append(current_data[:, 2].copy())  # Include delay grade data
            phase2_evolution['hunger_levels'].append(current_data[:, 3].copy())  # Include hunger level data
            # Use this transaction's actual success rates instead of baseline rates
            phase2_evolution['actual_success_rates'].append(current_transaction_success_rates.copy())
            
            # [SUMMARY] Record window statistics summary (global window system + original NA indices)
            window_summary = {}
            weighted_success_rates = []
            weighted_delay_grades = []
            for i in range(5):
                orig_idx = selected_indices_phase2[i]  # Use original NA index
                summary = _get_na_window_summary(orig_idx, na_window_queues_global)
                window_summary[i] = summary
                weighted_success_rates.append(summary['weighted_success_rate'])
                weighted_delay_grades.append(summary['weighted_delay_grade'])
            phase2_evolution['window_summaries'].append(window_summary)
            phase2_evolution['weighted_success_rates'].append(weighted_success_rates.copy())  # Include weighted success rate data
            phase2_evolution['weighted_delay_grades'].append(weighted_delay_grades.copy())  # Include weighted delay grade data
    
    # Store full per-transaction actual success rate and delay grade data
    phase2_evolution['transaction_actual_success_rates'] = transaction_actual_success_rates
    phase2_evolution['transaction_actual_delay_grades'] = transaction_actual_delay_grades
    
    print(f"[DONE] Phase 2 Transaction Simulation Completed!")
    print(f"[INFO] Phase 2 Final State:")
    for i in range(5):
        initial_rep = phase2_evolution['reputations'][0][i]
        final_rep = phase2_evolution['reputations'][-1][i]
        change = final_rep - initial_rep
        print(f"   {phase2_evolution['na_labels'][i]}: Reputation {initial_rep:.0f} → {final_rep:.0f} (Change: {change:+.0f})")
    
    # Write back the updated 5 NAs from Phase 2 into the original 10-NA array
    print("\nUpdating original 10 NAs with Phase 2 results...")
    updated_10_na_data = phase1_data.copy()
    for i, orig_idx in enumerate(selected_indices_phase2):
        updated_10_na_data[orig_idx] = current_data[i]
        print(f"   NA{orig_idx}: Updated with Phase 2 results (Reputation: {current_data[i, 0]:.0f})")
    
    # Phase 3: select 5 NAs from the updated 10 NAs
    print("\nPhase 3: selecting 5 NAs out of the updated 10 NAs...")
    
    # Select using updated data
    phase1_final_data = updated_10_na_data.copy()
    normalized_data = normalize_features(phase1_final_data)
    
    # Create a test-case format so predict_with_model can be reused
    phase3_test_case = {
        'raw_data': phase1_final_data,
        'normalized_data': normalized_data
    }
    
    # Phase 3 computes Q values from real-time data (does not use the initial prediction data)
    print("Phase 3: computing Q values from real-time data...")
    
    # Use the global sliding window to compute weighted feature data
    weighted_data_phase3 = np.zeros((10, 4))
    for na_idx in range(10):
        summary = _get_na_window_summary(na_idx, na_window_queues_global)
        weighted_data_phase3[na_idx, 0] = phase1_final_data[na_idx, 0]  # Reputation uses current value
        weighted_data_phase3[na_idx, 1] = summary['weighted_success_rate']  # Weighted success rate
        weighted_data_phase3[na_idx, 2] = summary['weighted_delay_grade']  # Weighted delay grade
        weighted_data_phase3[na_idx, 3] = phase1_final_data[na_idx, 3]  # Hunger uses current value
    
    # Run model inference using real-time weighted feature data
    normalized_data_phase3 = normalize_features(weighted_data_phase3)
    input_tensor = torch.FloatTensor(normalized_data_phase3).unsqueeze(0).to(device)
    
    with torch.no_grad():
        q_values = model(input_tensor)
        q_values_np = q_values.cpu().numpy().flatten()
    
    print(f"   Real-time Q-value range: {q_values_np.min():.4f} ~ {q_values_np.max():.4f}")
    
    # Select 3 high-reputation and 2 low-reputation NAs (threshold = 6600)
    reputations = phase1_final_data[:, 0]
    
    # Split high-reputation (>= 6600) and low-reputation (< 6600) NAs
    high_rep_indices = np.where(reputations >= 6600)[0]
    low_rep_indices = np.where(reputations < 6600)[0]
    
    print(
        f"Pre-Phase-3 reputation distribution: "
        f"high-reputation NAs (>=6600): {len(high_rep_indices)}, "
        f"low-reputation NAs (<6600): {len(low_rep_indices)}"
    )
    
    # Ensure there are enough NAs to select from
    if len(high_rep_indices) < 3:
        print(f"[WARN] Not enough high-reputation NAs for selecting 3; only {len(high_rep_indices)} available")
        sorted_indices = np.argsort(reputations)[::-1]
        high_rep_selected_phase3 = sorted_indices[:3]
    else:
        high_rep_scores = q_values_np[high_rep_indices]
        high_rep_selected_phase3 = high_rep_indices[np.argsort(high_rep_scores)[-3:]]
    
    if len(low_rep_indices) < 2:
        print(f"[WARN] Not enough low-reputation NAs for selecting 2; only {len(low_rep_indices)} available")
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
    
    # Phase 3 disables attack persistence and does not inherit Phase-2 malicious NAs.
    # All NAs follow normal logic in Phase 3; no malicious NAs are configured.
    malicious_na_indices_phase3 = []
    
    # Backward compatibility
    malicious_na_idx_phase3 = malicious_na_indices_phase3[0] if len(malicious_na_indices_phase3) > 0 else None
    
    print("Phase 3 selected 5 NAs:")
    for i, orig_idx in enumerate(selected_indices_phase3):
        na_type = "high-reputation" if i < 3 else "low-reputation"
        print(
            f"   NA{i} (orig idx {orig_idx}, {na_type}): "
            f"rep={selected_data_phase3[i, 0]:.0f}, Q={q_values_np[orig_idx]:.4f}"
        )
    
    # Phase 3: transaction simulation for the newly selected 5 NAs
    print(f"\nPhase 3: starting simulation of {n_transactions} transactions...")
    current_data_phase3 = selected_data_phase3.copy()
    
    # Initialize Phase-3 total_tx and success_count
    selected_total_tx_p3 = np.zeros(5, dtype=int)
    selected_success_count_p3 = np.zeros(5, dtype=int)
    
    for i, orig_idx in enumerate(selected_indices_phase3):
        if orig_idx in selected_indices_phase2:
            # If selected in Phase 2, inherit its data
            old_idx = np.where(selected_indices_phase2 == orig_idx)[0][0]
            selected_total_tx_p3[i] = selected_total_tx[old_idx]
            selected_success_count_p3[i] = selected_success_count[old_idx]
        else:
            # If newly selected, generate based on current success rate
            selected_total_tx_p3[i] = np.random.randint(20, 81)
            selected_success_count_p3[i] = int(current_data_phase3[i, 1] * selected_total_tx_p3[i])
        print(f"   NA{i} (orig idx {orig_idx}): total_tx={selected_total_tx_p3[i]}, success_count={selected_success_count_p3[i]}")
    
    # Use global sliding-window system (no re-initialization or inheritance)
    print("[INFO] Phase 3 uses the global sliding window system; all NAs keep their historical window data")
    for new_idx, orig_idx in enumerate(selected_indices_phase3):
        summary = _get_na_window_summary(orig_idx, na_window_queues_global)
        print(
            f"   NA{new_idx} (orig idx {orig_idx}): window_len={len(na_window_queues_global[orig_idx])}, "
            f"weighted_success_rate={summary['weighted_success_rate']:.3f}"
        )
    
    current_step_count = 100  # Phase 3 starts from step 100
    
    phase3_evolution = {
        'time_points': [100],  # Start from 100
        'reputations': [],
        'success_rates': [],
        'delay_grades': [],  # Include delay grade data
        'hunger_levels': [],  # Include hunger level data
        'weighted_success_rates': [],  # Include weighted success rate data
        'actual_success_rates': [],  # Include actual success rate data
        'weighted_delay_grades': [],  # Include weighted delay grade data
        'na_labels': [],
        'malicious_na_idx': malicious_na_idx_phase3,
        'malicious_na_indices': malicious_na_indices_phase3,  # Include malicious NA index list
        'selected_indices': selected_indices_phase3,
        'na_count': 5,
        'phase': 'transaction_simulation_phase3',
        'window_summaries': []  # Include window statistics summary
    }
    
    # Generate labels for selected 5 NAs in phase 3
    for i in range(5):
        orig_idx = selected_indices_phase3[i]  # Use original NA index
        if i == malicious_na_idx_phase3:
            phase3_evolution['na_labels'].append(f'CA{orig_idx} (Malicious)')
        else:
            phase3_evolution['na_labels'].append(f'CA{orig_idx}')
    
    # Record initial state
    phase3_evolution['reputations'].append(current_data_phase3[:, 0].copy())
    # Use dynamically computed success rates
    current_success_rates_p3 = np.where(selected_total_tx_p3 > 0, 
                                       selected_success_count_p3 / selected_total_tx_p3, 
                                       current_data_phase3[:, 1])
    phase3_evolution['success_rates'].append(current_success_rates_p3.copy())
    phase3_evolution['delay_grades'].append(current_data_phase3[:, 2].copy())  # Include delay grade data
    phase3_evolution['hunger_levels'].append(current_data_phase3[:, 3].copy())  # Include hunger level data
    phase3_evolution['actual_success_rates'].append(current_success_rates_p3.copy())  # Include actual success rate data
    
    # Record initial weighted data (global window system + original NA index)
    initial_weighted_success_rates = []
    initial_weighted_delay_grades = []
    for i in range(5):
        orig_idx = selected_indices_phase3[i]  # Use original NA index
        summary = _get_na_window_summary(orig_idx, na_window_queues_global)
        initial_weighted_success_rates.append(summary['weighted_success_rate'])
        initial_weighted_delay_grades.append(summary['weighted_delay_grade'])
        print(
            f"   NA{i} (orig idx {orig_idx}): initial_weighted_success_rate={summary['weighted_success_rate']:.3f}, "
            f"weighted_delay_grade={summary['weighted_delay_grade']:.3f}"
        )
    phase3_evolution['weighted_success_rates'].append(initial_weighted_success_rates.copy())
    phase3_evolution['weighted_delay_grades'].append(initial_weighted_delay_grades.copy())
    
    print(f"[INFO] Phase 3 Initial State:")
    for i in range(5):
        print(f"   {phase3_evolution['na_labels'][i]}: Reputation={current_data_phase3[i, 0]:.0f}, Success Rate={current_data_phase3[i, 1]:.3f}")
    
    # Simulate Phase 3 transactions
    for transaction in range(1, n_transactions + 1):
        # Phase 3: all 5 selected NAs participate in every transaction
        for na_idx in range(5):
            # Each NA executes a transaction
            # Phase 3 malicious-behavior logic
            if na_idx in malicious_na_indices_phase3:
                # Malicious NA: lower success rate
                success_prob = current_data_phase3[na_idx, 1] * 0.3  # Reduce success rate to 30%
                delay_increase = np.random.uniform(0.01, 0.05)  # Increase delay more for malicious NAs
                if transaction % 10 == 1:  # Print once per 10 transactions to avoid excessive output
                    print(
                        f"[ALERT] Phase 3 malicious NA{na_idx} attack: success_rate "
                        f"{current_data_phase3[na_idx, 1]:.3f} -> {success_prob:.3f}"
                    )
            else:
                # Normal NA behavior
                success_prob = current_data_phase3[na_idx, 1]  # Fixed current success rate, no fluctuation
                delay_increase = np.random.uniform(-0.02, 0.02)
            
            # If delay reaches 1.0, force the transaction to fail
            if force_fail_on_max_delay and current_data_phase3[na_idx, 2] >= 1.0:
                success = False
                print(
                    f"[DELAY] Phase3 NA{na_idx} delay={current_data_phase3[na_idx, 2]:.3f}; transaction forced to fail"
                )
            else:
                # Simulate transaction success/failure
                success = np.random.random() < success_prob
            
            # Compute reputation change
            computed_reward = calculate_computed_reward(
                current_data_phase3[na_idx, 1], 
                current_data_phase3[na_idx, 0], 
                current_data_phase3[na_idx, 3]
            )
            
            # Update total_tx and success_count
            selected_total_tx_p3[na_idx] += 1
            if success:
                selected_success_count_p3[na_idx] += 1
                reputation_change = computed_reward * 20
                current_data_phase3[na_idx, 0] = min(10000, current_data_phase3[na_idx, 0] + reputation_change)
            else:
                reputation_change = computed_reward * 100
                current_data_phase3[na_idx, 0] = max(3000, current_data_phase3[na_idx, 0] - reputation_change)
            
            # Success rate is computed from cumulative transaction counts.
            # current_data_phase3[:, 1] is computed on demand from selected_success_count_p3 / selected_total_tx_p3.
            
            # Update delay grade
            current_data_phase3[na_idx, 2] = max(0.0, min(1.0, current_data_phase3[na_idx, 2] + delay_increase))
            
            # Hunger level updates are applied after the transaction loop
            
            # Collect sliding-window data (aligned with test.py): include hunger field
            transaction_data = {
                'success': success,
                'reputation_before': current_data_phase3[na_idx, 0] - reputation_change if success else current_data_phase3[na_idx, 0] + abs(reputation_change),
                'reputation_after': current_data_phase3[na_idx, 0],
                'delay': current_data_phase3[na_idx, 2],  # Use delay grade directly (0.0-1.0)
                'hunger': current_data_phase3[na_idx, 3],  # Include hunger field
                'step': current_step_count + transaction
            }
            # Update the global window system using original NA indices
            orig_idx = selected_indices_phase3[na_idx]
            _update_na_window_queue(orig_idx, transaction_data, na_current_pack_global, current_step_count + transaction)
        
        # Update hunger levels for all 10 NAs (selected 5 reset to 0; others increase)
        _update_hunger_for_all_nas(selected_indices_phase3, global_current_step + transaction, updated_10_na_data, 'phase3')
        global_current_step += 1
        
        # Sync selected NAs' hunger levels back to current_data_phase3
        for i, orig_idx in enumerate(selected_indices_phase3):
            current_data_phase3[i, 3] = updated_10_na_data[orig_idx, 3]
        
        # [PACK] Periodically pack sliding-window data (global window system)
        if transaction % window_pack_interval == 0:
            _pack_all_na_windows(na_window_queues_global, na_current_pack_global, current_step_count + transaction)
            print(f"[PACK] Phase 3 - Transaction {transaction}: Window data packed for all NAs (global system)")
        
        # Record current state
        if transaction % 5 == 0:  # Record every 5 transactions
            phase3_evolution['time_points'].append(100 + transaction)
            phase3_evolution['reputations'].append(current_data_phase3[:, 0].copy())
            # Use dynamically computed success rates
            current_success_rates_p3 = np.where(selected_total_tx_p3 > 0, 
                                               selected_success_count_p3 / selected_total_tx_p3, 
                                               current_data_phase3[:, 1])
            phase3_evolution['success_rates'].append(current_success_rates_p3.copy())
            phase3_evolution['delay_grades'].append(current_data_phase3[:, 2].copy())  # Include delay grade data
            phase3_evolution['hunger_levels'].append(current_data_phase3[:, 3].copy())  # Include hunger level data
            phase3_evolution['actual_success_rates'].append(current_success_rates_p3.copy())  # Include actual success rate data
            
            # [SUMMARY] Record window statistics summary (global window system + original NA indices)
            window_summary = {}
            weighted_success_rates = []
            weighted_delay_grades = []
            for i in range(5):
                orig_idx = selected_indices_phase3[i]  # Use original NA index
                summary = _get_na_window_summary(orig_idx, na_window_queues_global)
                window_summary[i] = summary
                weighted_success_rates.append(summary['weighted_success_rate'])
                weighted_delay_grades.append(summary['weighted_delay_grade'])
            phase3_evolution['window_summaries'].append(window_summary)
            phase3_evolution['weighted_success_rates'].append(weighted_success_rates.copy())  # Include weighted success rate data
            phase3_evolution['weighted_delay_grades'].append(weighted_delay_grades.copy())  # Include weighted delay grade data
    
    print(f"[DONE] Phase 3 Transaction Simulation Completed!")
    print(f"[INFO] Phase 3 Final State:")
    for i in range(5):
        initial_rep = phase3_evolution['reputations'][0][i]
        final_rep = phase3_evolution['reputations'][-1][i]
        change = final_rep - initial_rep
        print(f"   {phase3_evolution['na_labels'][i]}: Reputation {initial_rep:.0f} → {final_rep:.0f} (Change: {change:+.0f})")
    
    # Export Phase 3 parameter file (Q values computed from real-time data)
    print("\nExporting Phase 3 parameter file...")
    
    # Reuse the real-time Q values and data computed above
    export_na_parameters(
        raw_data=weighted_data_phase3,
        normalized_data=normalized_data_phase3,
        q_values_np=q_values_np,  # Q values computed from real-time data
        phase_name='phase3_selection',
        attack_type=attack_type,
        na_window_queues=na_window_queues_global
    )
    
    print("[DONE] Exported Phase 3 parameter file (real-time data)")
    
    # Collect hunger and delay-grade data for all 10 NAs across phases
    all_hunger_data = []
    all_delay_grade_data = []
    
    # Preserve the original Phase-1 hunger data (before _update_hunger_for_all_nas mutates it)
    original_phase1_hunger = test_case['raw_data'][:, 3].copy()  # Hunger values from raw test data
    print(f"[DEBUG] Using original hunger data: {original_phase1_hunger}")
    
    for na_idx in range(10):
        na_hunger_timeline = []
        na_delay_grade_timeline = []
        
        # Phase 1: each NA keeps its original initial hunger value; delay grade is read from phase1_data
        na_initial_hunger = original_phase1_hunger[na_idx]  # Initial hunger value from raw data
        if na_idx == 0:  # Print debug info for the first NA only
            print(f"[DEBUG] NA_{na_idx:02d} initial hunger: {na_initial_hunger:.4f}")
            print(f"[DEBUG] phase1_data[{na_idx}, 3] = {phase1_data[na_idx, 3]:.4f}")
        for t in range(51):
            # During Phase 1, keep the initial hunger value unchanged
            na_hunger_timeline.append(na_initial_hunger)
            # During Phase 1, all NAs keep the initial delay grade
            na_delay_grade_timeline.append(phase1_data[na_idx, 2])
        
        # [DEBUG] Print Phase-1 end hunger state
        if na_idx in [10, 15, 16, 18, 19]:  # Previously problematic NAs
            print(f"[DEBUG] NA_{na_idx:02d} Phase1 end hunger: {na_initial_hunger:.4f} (should be non-zero)")
        
        # Phase 2: check whether this NA was selected
        if na_idx in selected_indices_phase2:
            # Selected NA: hunger reset to 0, delay grade uses phase2 data
            selected_idx = np.where(selected_indices_phase2 == na_idx)[0][0]
            for t in range(len(phase2_evolution['time_points'])):
                na_hunger_timeline.append(0.0)  # Selected NA hunger is 0
            for grades in phase2_evolution['delay_grades']:
                na_delay_grade_timeline.append(grades[selected_idx])
            # [DEBUG] Print selected NA state
            if na_idx in [10, 15, 16, 18, 19]:
                print(f"[DEBUG] NA_{na_idx:02d} selected in Phase2; hunger reset to 0")
        else:
            # Unselected NA: hunger keeps increasing from the Phase-1 end value; delay grade stays unchanged
            phase1_end_hunger = na_initial_hunger  # Hunger value at end of Phase 1
            for t in range(len(phase2_evolution['time_points'])):
                # Compute steps elapsed since Phase 1 end
                steps_since_phase1_end = (t + 1) * 5  # Phase 2 records every 5 transactions, starting from step 5
                # Continue growing from the Phase-1 end hunger value
                additional_hunger = np.log(1 + steps_since_phase1_end / 20.0) / np.log(11)
                hunger_value = min(1.0, phase1_end_hunger + additional_hunger * 0.5)  # Slightly reduce growth rate
                na_hunger_timeline.append(hunger_value)
                # Keep the initial delay grade
                na_delay_grade_timeline.append(phase1_data[na_idx, 2])
        
        # Phase 3: check whether this NA was selected
        if na_idx in selected_indices_phase3:
            # Selected NA: hunger reset to 0; delay grade uses Phase-3 data
            selected_idx = np.where(selected_indices_phase3 == na_idx)[0][0]
            for t in range(len(phase3_evolution['time_points'])):
                na_hunger_timeline.append(0.0)  # Selected NA hunger is 0
            for grades in phase3_evolution['delay_grades']:
                na_delay_grade_timeline.append(grades[selected_idx])
            # [DEBUG] Print selected NA state
            if na_idx in [10, 15, 16, 18, 19]:
                print(f"[DEBUG] NA_{na_idx:02d} selected in Phase3; hunger reset to 0")
        else:
            # Unselected NA: hunger keeps increasing from the Phase-2 end value; delay grade stays unchanged
            # Compute hunger value at the end of Phase 2
            if na_idx in selected_indices_phase2:
                phase2_end_hunger = 0.0  # If selected in Phase 2, it starts from 0
            else:
                # If also unselected in Phase 2, compute Phase-2 end hunger from Phase-1 value
                phase1_end_hunger = na_initial_hunger
                phase2_steps = len(phase2_evolution['time_points']) * 5  # Total steps in Phase 2
                additional_hunger_phase2 = np.log(1 + phase2_steps / 20.0) / np.log(11)
                phase2_end_hunger = min(1.0, phase1_end_hunger + additional_hunger_phase2 * 0.5)
            
            for t in range(len(phase3_evolution['time_points'])):
                # Compute steps elapsed since Phase 2 ended
                steps_since_phase2_end = (t + 1) * 5  # Phase 3 records every 5 transactions, starting from step 5
                # Continue growing from the Phase-2 end hunger value
                additional_hunger = np.log(1 + steps_since_phase2_end / 20.0) / np.log(11)
                hunger_value = min(1.0, phase2_end_hunger + additional_hunger * 0.5)  # Slightly reduce growth rate
                na_hunger_timeline.append(hunger_value)
                # Keep the initial delay grade
                na_delay_grade_timeline.append(phase1_data[na_idx, 2])
        
        all_hunger_data.append(na_hunger_timeline)
        all_delay_grade_data.append(na_delay_grade_timeline)
    
    # Collect status data for all NAs across phases
    na_states_records = []
    
    # Record Phase-1 end state
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
    
    # Record selected NAs' Phase-2 start state
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
    
    # Record selected NAs' Phase-2 end state
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
    
    # Record selected NAs' Phase-3 start state
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
    
    # Record selected NAs' Phase-3 end state
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
    
    # Record unselected NAs' states across phases (based on phase1_data and hunger computation)
    all_selected = np.unique(np.concatenate([selected_indices_phase2, selected_indices_phase3]))
    unselected_nas = [i for i in range(10) if i not in all_selected]
    
    for na_idx in unselected_nas:
        # Phase-2 start state (unselected)
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
        
        # Phase-2 end state (unselected)
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
        
        # If also unselected in Phase 3
        if na_idx not in selected_indices_phase3:
            # Phase-3 start state
            na_states_records.append({
                'NA_Index': na_idx,
                'Phase': 'Phase3_Start',
                'Reputation': phase1_data[na_idx, 0],
                'Success_Rate': phase1_data[na_idx, 1],
                'Delay_Grade': phase1_data[na_idx, 2],
                'Hunger_Level': phase2_end_hunger,
                'Selected': 'No'
            })
            
            # Phase-3 end state
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
    # print(f"[DATA] NA state records saved to: {states_csv_path}")
    print(f"[INFO] Recorded {len(na_states_records)} state records covering all 10 NAs across phases")
    
    # Collect weighted success rate data for all 10 NAs
    all_weighted_success_rates = []
    
    for na_idx in range(10):
        na_weighted_success_timeline = []
        
        # Phase 1: take weighted success rates from phase1_evolution
        for rates in phase1_evolution['weighted_success_rates']:
            na_weighted_success_timeline.append(rates[na_idx])
        
        # Phase 2: check whether this NA was selected
        if na_idx in selected_indices_phase2:
            # Selected NA: use Phase-2 weighted success rates (skip initial value)
            selected_idx = np.where(selected_indices_phase2 == na_idx)[0][0]
            for rates in phase2_evolution['weighted_success_rates'][1:]:  # Skip initial value
                na_weighted_success_timeline.append(rates[selected_idx])
        else:
            # Unselected NA: use window-summary data (skip initial value)
            for t in range(1, len(phase2_evolution['time_points'])):  # Skip initial value
                summary = _get_na_window_summary(na_idx, na_window_queues_global)
                na_weighted_success_timeline.append(summary['weighted_success_rate'])
        
        # Phase 3: check whether this NA was selected
        if na_idx in selected_indices_phase3:
            # Selected NA: use Phase-3 weighted success rates (skip initial value)
            selected_idx = np.where(selected_indices_phase3 == na_idx)[0][0]
            for rates in phase3_evolution['weighted_success_rates'][1:]:  # Skip initial value
                na_weighted_success_timeline.append(rates[selected_idx])
        else:
            # Unselected NA: use window-summary data (skip initial value)
            for t in range(1, len(phase3_evolution['time_points'])):  # Skip initial value
                summary = _get_na_window_summary(na_idx, na_window_queues_global)
                na_weighted_success_timeline.append(summary['weighted_success_rate'])
        
        all_weighted_success_rates.append(na_weighted_success_timeline)
    
    # Merge data from three phases
    evolution_data = {
        'phase1': phase1_evolution,
        'phase2': phase2_evolution,
        'phase3': phase3_evolution,
        'phase_separator_1': 50,  # First separator (Phase 1 ends; Phase 2 starts)
        'phase_separator_2': 100,  # Second separator (Phase 2 ends; Phase 3 starts)
        'attack_type': attack_type,
        'attack_mode': attack_mode,
        'all_hunger_data': all_hunger_data,
        'all_delay_grade_data': all_delay_grade_data,  # Include delay grade data for all NAs
        'all_weighted_success_rates': all_weighted_success_rates,  # Include weighted success rate data for all NAs
        'na_states_records': na_states_records  # Include state records
    }
    
    return evolution_data

def export_phase1_reputation_data_to_csv(evolution_data, save_path=None):
    """
    Export Phase-1 reputation evolution for all NAs to a CSV file.
    
    Args:
        evolution_data: A dict containing phase1_evolution data.
        save_path: Output path. If None, auto-generate.
    
    Returns:
        str: The saved file path.
    """
    if 'phase1' not in evolution_data:
        print("Error: evolution_data has no phase1 data")
        return None
    
    phase1_data = evolution_data['phase1']
    
    if save_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'phase1_reputation_evolution_{timestamp}.csv'
    
    # Prepare data
    time_points = phase1_data['time_points']
    reputations = phase1_data['reputations']
    na_count = phase1_data['na_count']
    
    # Build DataFrame
    data_dict = {'time_step': time_points}
    
    # Add a reputation column for each NA
    for na_idx in range(na_count):
        reputation_values = [reputations[t][na_idx] for t in range(len(time_points))]
        data_dict[f'NA{na_idx}_reputation'] = reputation_values
    
    df = pd.DataFrame(data_dict)
    
    # Save to CSV
    df.to_csv(save_path, index=False, encoding='utf-8')
    
    print(f"Phase-1 reputation evolution exported to: {save_path}")
    print(f"Data includes {na_count} NAs across {len(time_points)} time steps")
    
    # Summary statistics
    print("\n=== Phase-1 reputation evolution summary ===")
    for na_idx in range(na_count):
        col_name = f'NA{na_idx}_reputation'
        initial_rep = df[col_name].iloc[0]
        final_rep = df[col_name].iloc[-1]
        change = final_rep - initial_rep
        print(f"NA{na_idx}: initial={initial_rep:.1f}, final={final_rep:.1f}, change={change:+.1f}")
    
    return save_path

def export_hunger_data_to_csv(evolution_data, save_path=None):
    """
    Export hunger-level evolution for all NAs to a CSV file.
    
    Args:
        evolution_data: Evolution data dict containing hunger-level data.
        save_path: Output path. If None, use default.
    
    Returns:
        str: The saved file path.
    """
    if save_path is None:
        save_path = f"{evolution_data['attack_type']}_hunger_evolution.csv"
    
    # Get hunger data
    all_hunger_data = evolution_data['all_hunger_data']
    
    # Build the timeline
    phase1_time = list(range(0, 51))  # Phase 1: 0-50
    phase2_time = list(range(51, 101))  # Phase 2: 51-100
    phase3_time = list(range(101, 151))  # Phase 3: 101-150
    all_time_points = phase1_time + phase2_time + phase3_time
    
    # Prepare CSV rows
    csv_data = []
    
    # Create one row per time point
    for t_idx, time_point in enumerate(all_time_points):
        row = {'Time_Point': time_point}
        
        # Determine current phase
        if time_point <= 50:
            phase = 'Phase1'
        elif time_point <= 100:
            phase = 'Phase2'
        else:
            phase = 'Phase3'
        
        row['Phase'] = phase
        
        # Add hunger data for all 20 NAs
        for na_idx in range(20):
            if t_idx < len(all_hunger_data[na_idx]):
                hunger_value = all_hunger_data[na_idx][t_idx]
            else:
                # If data is missing, use the last known value
                hunger_value = all_hunger_data[na_idx][-1] if all_hunger_data[na_idx] else 0.0
            
            row[f'NA_{na_idx:02d}_Hunger'] = hunger_value
        
        csv_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(save_path, index=False)
    
    print(f"[EXPORT] Hunger data exported to: {save_path}")
    print(f"[INFO] Exported {len(csv_data)} time points covering hunger evolution for all 20 NAs")
    print(f"[INFO] Data range: time point {all_time_points[0]} to {all_time_points[-1]}")
    
    # Display summary statistics
    print("\n[STATS] Hunger data statistics:")
    for na_idx in range(20):
        hunger_values = [row[f'NA_{na_idx:02d}_Hunger'] for row in csv_data]
        min_hunger = min(hunger_values)
        max_hunger = max(hunger_values)
        avg_hunger = sum(hunger_values) / len(hunger_values)
        print(f"   NA_{na_idx:02d}: min={min_hunger:.3f}, max={max_hunger:.3f}, avg={avg_hunger:.3f}")
    
    return save_path

def select_nas_for_simulation(test_case, prediction):
    """
    Select 5 NAs from test data for simulation (3 high reputation + 2 low reputation)
    
    Args:
        test_case: Test case
        prediction: Prediction result
    
    Returns:
        selected_data: Data for the selected 5 NAs
        selected_indices: Indices of selected NAs
        malicious_na_idx: Index of malicious NA in the selected list
    """
    raw_data = test_case['raw_data']
    reputations = raw_data[:, 0]
    
    # Check whether this is an attack scenario
    is_attack_scenario = 'attack_type' in test_case and test_case['attack_type'] != 'normal'
    
    # Check whether there are multiple malicious nodes (ME scenario)
    attack_node_indices = test_case.get('attack_node_indices', [])
    if is_attack_scenario and len(attack_node_indices) > 0:
        if test_case['attack_type'] == 'ME' and len(attack_node_indices) == 2:
            # ME scenario: two malicious nodes (one in high-rep group, one in low-rep group)
            print(f"[ME] Detected scenario; malicious node indices: {attack_node_indices}")
            
            # Sort by reputation (excluding malicious nodes)
            available_indices = np.arange(len(reputations))
            available_indices = np.setdiff1d(available_indices, attack_node_indices)
            available_reputations = reputations[available_indices]
            sorted_available = available_indices[np.argsort(available_reputations)[::-1]]
            
            # Select 3 normal NAs (2 high-rep + 1 low-rep)
            high_rep_pool = sorted_available[:max(2, len(sorted_available)//2)]
            high_rep_selected = np.random.choice(high_rep_pool, size=min(2, len(high_rep_pool)), replace=False)
            
            low_rep_pool = sorted_available[-max(1, len(sorted_available)//3):]
            low_rep_selected = np.random.choice(low_rep_pool, size=min(1, len(low_rep_pool)), replace=False)
            
            # Merge selected NAs (put the two malicious nodes at the end)
            selected_indices = np.concatenate([high_rep_selected, low_rep_selected, attack_node_indices])
            malicious_na_idx = [3, 4]  # The two malicious nodes are the last two positions in the selected list
            
            print("[ME] Selected 5 NAs:")
            print(f"   High-reputation NAs: {high_rep_selected} (rep: {reputations[high_rep_selected]})")
            print(f"   Low-reputation NAs: {low_rep_selected} (rep: {reputations[low_rep_selected]})")
            print(f"   Malicious NAs: {attack_node_indices} (rep: {reputations[attack_node_indices]}) - ME")
            
        else:
            # Other attack scenarios: single malicious node
            attack_node_idx = attack_node_indices[0]
            print(f"[ATTACK] Detected scenario; malicious node index: {attack_node_idx}")
            
            # Sort by reputation (excluding malicious node)
            available_indices = np.arange(len(reputations))
            available_indices = available_indices[available_indices != attack_node_idx]
            available_reputations = reputations[available_indices]
            sorted_available = available_indices[np.argsort(available_reputations)[::-1]]
            
            # Select 4 normal NAs (2 high-rep + 2 low-rep)
            high_rep_pool = sorted_available[:max(2, len(sorted_available)//3)]
            high_rep_selected = np.random.choice(high_rep_pool, size=min(2, len(high_rep_pool)), replace=False)
            
            low_rep_pool = sorted_available[-max(2, len(sorted_available)//3):]
            low_rep_selected = np.random.choice(low_rep_pool, size=min(2, len(low_rep_pool)), replace=False)
            
            # Merge selected NAs (put malicious node at the end)
            selected_indices = np.concatenate([high_rep_selected, low_rep_selected, [attack_node_idx]])
            malicious_na_idx = 4  # Malicious node is the last one in the selected list
            
            print("[ATTACK] Selected 5 NAs:")
            print(f"   High-reputation NAs: {high_rep_selected} (rep: {reputations[high_rep_selected]})")
            print(f"   Low-reputation NAs: {low_rep_selected} (rep: {reputations[low_rep_selected]})")
            print(f"   Malicious NA: {attack_node_idx} (rep: {reputations[attack_node_idx]:.0f}) - {test_case['attack_type']}")
        
    else:
        # Normal scenario: strict threshold at 6600
        # Split high-reputation (>=6600) and low-reputation (<6600) NAs
        high_rep_indices = np.where(reputations >= 6600)[0]
        low_rep_indices = np.where(reputations < 6600)[0]
        
        print(
            f"[STATS] Reputation distribution: "
            f"high-reputation NAs (>=6600): {len(high_rep_indices)}, "
            f"low-reputation NAs (<6600): {len(low_rep_indices)}"
        )
        
        # Ensure there are enough NAs to select
        if len(high_rep_indices) < 3:
            print(f"[WARN] Not enough high-reputation NAs for selecting 3; only {len(high_rep_indices)} available")
            # If there are not enough high-reputation NAs, select the top 3 by reputation
            sorted_indices = np.argsort(reputations)[::-1]
            high_rep_selected = sorted_indices[:3]
        else:
            # Randomly select 3 from high-reputation NAs
            high_rep_selected = np.random.choice(high_rep_indices, size=3, replace=False)
        
        if len(low_rep_indices) < 2:
            print(f"[WARN] Not enough low-reputation NAs for selecting 2; only {len(low_rep_indices)} available")
            # If there are not enough low-reputation NAs, select the bottom 2 by reputation from the remaining set
            remaining_indices = np.setdiff1d(np.arange(len(reputations)), high_rep_selected)
            if len(remaining_indices) >= 2:
                remaining_reputations = reputations[remaining_indices]
                sorted_remaining = remaining_indices[np.argsort(remaining_reputations)]
                low_rep_selected = sorted_remaining[:2]
            else:
                low_rep_selected = remaining_indices
        else:
            # Randomly select 2 from low-reputation NAs
            low_rep_selected = np.random.choice(low_rep_indices, size=2, replace=False)
        
        # Merge selected NAs
        selected_indices = np.concatenate([high_rep_selected, low_rep_selected])
        
        # Randomly pick one as malicious (prefer the low-reputation group)
        if len(low_rep_selected) > 0:
            malicious_na_idx = np.random.choice(range(3, 3 + len(low_rep_selected)))  # Choose from low-reputation NAs
        else:
            malicious_na_idx = np.random.choice(range(len(selected_indices)))  # If none, choose randomly
        
        print("[NORMAL] Selected 5 NAs:")
        print(f"   High-reputation NAs: {high_rep_selected} (rep: {reputations[high_rep_selected]})")
        print(f"   Low-reputation NAs: {low_rep_selected} (rep: {reputations[low_rep_selected]})")
        print(
            f"   Simulated malicious NA: position {malicious_na_idx} in selected list "
            f"(orig idx: {selected_indices[malicious_na_idx]})"
        )
        
        # Validate selection result
        high_count = np.sum(reputations[selected_indices] >= 6600)
        low_count = np.sum(reputations[selected_indices] < 6600)
        print(f"[DONE] Validation: high-reputation NAs ({high_count}), low-reputation NAs ({low_count})")
    
    selected_data = raw_data[selected_indices]
    return selected_data, selected_indices, malicious_na_idx

def main():
    """
    Main function - Demonstrate how to test custom datasets with trained model
    """
    print("Testing Custom Dataset with Trained DQN Model")
    print("="*60)
    
    # Set device
    device, device_note = get_runtime_device(preferred_gpu_index=1)
    print(f"Using device: {device}")
    if device_note:
        print(device_note)
    
    # Model file paths (absolute paths)
    model_path = os.path.join(OUTPUT_ROOT, "models", "policy_net_state_dict.pth")
    model_info_path = os.path.join(OUTPUT_ROOT, "models", "model_info.pth")
    
    try:
        # 1. Load trained model
        model, model_info = load_trained_model(model_path, model_info_path, device)
        
        # 2. Attack scenario testing
        print("\n" + "="*40)
        print("Attack Scenario Testing")
        print("="*40)
        print("1. Malicious With Everyone (ME) - Malicious nodes continuously provide harmful services")
        print("2. On-Off Attack (OOA) - Nodes periodically switch between honest/malicious behavior")
        print("3. Opportunistic Service Attack (OSA) - Nodes provide high-quality service only when reputation drops")
        print("4. All attack scenarios comparison test")
        
        attack_choice = input("\nPlease select attack scenario (1-4): ").strip()
        
        if attack_choice in ['1', '2', '3']:
            attack_types = ['ME', 'OOA', 'OSA']
            attack_type = attack_types[int(attack_choice) - 1]
            
            n_na = 10  # Generate 10 NAs by default
            
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
            
            # 5-NA transaction simulation (auto-run)
            if test_cases:
                print("\n" + "="*40)
                print("Select Transaction Attack Mode")
                print("="*40)
                print("1. behavior (existing)")
                print("2. delay_only (only delay manipulation)")
                mode_choice = input("\nPlease select mode (1-2): ").strip()
                attack_mode = 'delay_only' if mode_choice == '2' else 'behavior'
                force_fail_on_max_delay = (attack_mode == 'behavior')

                base_dir = os.path.join(OUTPUT_ROOT, "malicious_behavior_experiment")
                output_dir = os.path.join(
                    base_dir,
                    "experiment_code_data_delay_only" if attack_mode == 'delay_only' else "experiment_code_data",
                )
                os.makedirs(output_dir, exist_ok=True)

                print("\nStarting 5-NA transaction simulation (3 high reputation + 2 low reputation, including 1 malicious NA)")
                test_case = test_cases[0]
                prediction = predictions[0]
                
                # Select 5 NAs for simulation
                selected_data, selected_indices, malicious_na_idx = select_nas_for_simulation(test_case, prediction)
                
                # Set simulation parameters (fixed 50 transactions)
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
            n_na = 20  # Default: 20 NAs per scenario
            
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
            
            # 5-NA transaction simulation (auto-run for all scenarios)
            print("\n" + "="*40)
            print("Select Transaction Attack Mode")
            print("="*40)
            print("1. behavior (existing)")
            print("2. delay_only (only delay manipulation)")
            mode_choice = input("\nPlease select mode (1-2): ").strip()
            attack_mode = 'delay_only' if mode_choice == '2' else 'behavior'
            force_fail_on_max_delay = (attack_mode == 'behavior')

            base_dir = os.path.join(OUTPUT_ROOT, "malicious_behavior_experiment")
            output_dir = os.path.join(
                base_dir,
                "experiment_code_data_delay_only" if attack_mode == 'delay_only' else "experiment_code_data",
            )
            os.makedirs(output_dir, exist_ok=True)

            print("\nStarting 5-NA transaction simulation for all scenarios...")
            for i, test_case in enumerate(test_cases):
                attack_type = test_case['attack_type']
                prediction = predictions[i]
                
                print(f"\nStarting 5-NA transaction simulation for {attack_type} scenario")
                
                # Select 5 NAs for simulation
                selected_data, selected_indices, malicious_na_idx = select_nas_for_simulation(test_case, prediction)
                
                # Set simulation parameters (fixed 100 transactions)
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
