import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only GPU 1 (0-based index)
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import random
from collections import deque
import time
from datetime import datetime

"""
Double DQN (DDQN) for Network Assistant Selection
=================================================

Key Features:
- Double DQN: Mitigates Q-value overestimation
- Experience replay: Improves sample efficiency
- Target network: Stabilizes training
- NA-count-invariant architecture: Supports any number of NAs

Core Idea:
1. Select actions using the policy network
2. Evaluate Q-values using the target network
3. Reduce maximization bias and overestimation

Use Case: Selection optimization with a fixed NA set
"""

# Record program start time
start_time = time.time()
start_datetime = datetime.now()
print(f"Program start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

def get_output_root() -> Path:
    configured = os.environ.get("DRL_DPKI_OUTPUT_DIR")
    if configured:
        return Path(configured)
    legacy = Path("/mnt/data/wy2024")
    if legacy.is_dir():
        return legacy
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "outputs"

OUTPUT_ROOT = get_output_root()

def set_all_seeds(seed=42):
    """
    Set all random seeds to improve reproducibility.
    """
    print(f"Setting random seed: {seed}")
    
    # Python built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU case
        
    # Ensure deterministic CUDA behavior (may affect performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("Random seed configuration complete. Training should be more stable.")

# Seed selection
# Common seed recommendations:
# - 42: Classic choice, often stable
# - 123: Simple and frequently works well
# - 2024: Easy to remember
# - 88: Performs well for some tasks
# - 666: Sometimes works well in RL experiments
# - 1337: Common "hacker" seed, sometimes good in DL experiments

RANDOM_SEED = 42  # Edit this value to try different seeds

print(f"Current seed: {RANDOM_SEED}")
print("Seed suggestions: 42, 123, 2024, 88, 666, 1337")

# Set global random seed
set_all_seeds(RANDOM_SEED)

# GPU availability check
print("GPU availability check:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Using CPU for training")

# Font configuration (use English fonts only to avoid missing-glyph warnings)
print("Configuring fonts...")

font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "font"))
if os.path.isdir(font_dir):
    font_files = [f for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
    if font_files:
        print(f"Found custom font files: {len(font_files)}. Loading...")
    loaded_custom_font = False
    for font_file in font_files:
        font_path = os.path.join(font_dir, font_file)
        try:
            fm.fontManager.addfont(font_path)
            loaded_custom_font = True
        except Exception as exc:
            print(f"Failed to load font {font_file}: {exc}")
else:
    print(f"Font directory not found: {font_dir}")

# Font priority: use English fonts only
if 'loaded_custom_font' in locals() and loaded_custom_font:
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # Render minus sign correctly
plt.rcParams['font.size'] = 10

print(f"Using font family: {plt.rcParams['font.family'][:2]}")
print("Font configuration complete")

def format_duration(seconds):
    """
    Format a duration in seconds into a human-readable string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}.{millisecs:03d}s"

# NA-count-invariant DQN architecture based on pure feature learning
# Advantages:
# 1. Scalability: supports any number of NAs without retraining
# 2. Generalization: learns general patterns of NA features instead of positions
# 3. Simplicity: parameter count stays fixed as NA count grows
# 4. Interpretability: each NA's Q-value depends only on its own features
class QNetwork(nn.Module):
    def __init__(self, n_features, n_na):  # Remove default parameters
        super().__init__()
        # NA feature encoder: maps each NA's features to a fixed dimension
        self.na_encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(0.01),  # Use LeakyReLU to avoid dying ReLU
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(32, 16)  # Encode each NA to 16 dims
        )
        
        # Q-value predictor: independently evaluates each NA
        self.q_predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),
            nn.Linear(16, 1)  # Output a single Q-value
        )
        
        # Initialize network weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights to avoid identical outputs."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier normal initialization
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    # Initialize bias with small random values
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x):
        batch_size = x.size(0)
        n_na = x.size(1)
        # Flatten all NA features
        x_flat = x.view(batch_size * n_na, -1)  # [batch*n_na, n_features]
        # Encode each NA independently
        na_embeddings = self.na_encoder(x_flat)  # [batch*n_na, 16]
        # Predict Q-value for each NA independently
        q_values_flat = self.q_predictor(na_embeddings)  # [batch*n_na, 1]
        # Reshape back to batch format
        q_values = q_values_flat.view(batch_size, n_na)  # [batch, n_na]
        return q_values

class DQNAgent:
    def __init__(self, n_features, n_na, lr, gamma,  # Remove all default parameters
                 epsilon_start, epsilon_end,  # Remove all default parameters
                 decay_steps, memory_size, batch_size, target_update,  # Remove all default parameters
                 min_memory_size, update_frequency, total_episodes):  # Remove all default parameters
        # Prefer GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"Double DQN Agent device: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            print(f"Double DQN Agent device: {self.device}")
        print("Using an NA-count-invariant network architecture (supports any number of NAs)")
        print("Algorithm: Double DQN (DDQN) to mitigate Q-value overestimation")
        print(f"Optimization config: Replay Buffer={memory_size}, Batch Size={batch_size}, Min Memory={min_memory_size}")
        
        self.n_na = n_na
        self.n_features = n_features
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.decay_steps   = decay_steps
        self.total_episodes = total_episodes  # Total training episodes
        self.step_count    = 0
        self.epsilon       = epsilon_start
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.min_memory_size = min_memory_size  # Minimum experiences before training starts
        self.update_frequency = update_frequency  # Train once every N steps
        self.step_counter = 0  # Step counter
        
        # Build networks and move to device: architecture is NA-count-invariant
        self.policy_net = QNetwork(n_features, n_na).to(self.device)
        self.target_net = QNetwork(n_features, n_na).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Use AdamW optimizer with AMSGrad
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr,
            weight_decay=1e-4,  # L2 regularization
            amsgrad=True       # Enable AMSGrad for more stable convergence
        )
        print("Optimizer: AdamW (weight_decay=1e-4, amsgrad=True)")
        print("   Pros: better generalization and stable convergence")
        print("   Notes: decoupled weight decay and AMSGrad reduces oscillation")
        
        # Add Cosine Annealing with Warm-up learning rate scheduler
        self.initial_lr = lr
        self.final_lr = lr * 0.1  # Final LR is 10% of the initial LR
        self.total_episodes = total_episodes
        print(f"LR schedule: Cosine Annealing with Warm-up {self.final_lr:.6f} → {lr:.6f} → {self.final_lr:.6f}")
        print("   First 5%: warm-up linearly increases to max LR")
        print("   Last 95%: cosine annealing smoothly decays to min LR")
        print("   Pros: avoids overly fast learning early and improves late convergence")
        print("   Formula: lr = min_lr + (max_lr - min_lr) * (1 + cos(pi * progress)) / 2")
        
        self.target_update = target_update
        self.learn_step = 0
        
        # Loss history
        self.loss_history = []
        self.loss_steps = []
        
        # Learning rate history
        self.lr_history = []
        
        # Epsilon history
        self.epsilon_history = []
        
        # Policy analysis history
        self.policy_entropy_history = []  # Policy entropy
        self.action_distribution_history = []  # Action distribution
        self.na_selection_frequency = np.zeros(n_na)  # Per-NA selection frequency

    def select_action(self, state, initial_reputations=None, available_mask=None, record_policy=False):
        """
        Select a single NA as the action; the network can separate high/low reputation NAs.
        
        Args:
            state: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations for grouping
            available_mask: shape (n_na,) - available NA mask; None means all available
            record_policy: bool - whether to record policy-analysis metrics
        
        Returns:
            int: selected NA index
        """
        # If no initial reputations are provided, fall back to current reputations
        if initial_reputations is None:
            reputations = state[:, 0]  # Use current reputations
        else:
            reputations = initial_reputations
        
        # Configure availability mask
        if available_mask is None:
            available_mask = np.ones(self.n_na, dtype=bool)
        
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0:
            return 0  # If nothing is available, fall back to 0

        # Compute Q-values and policy distribution (needed even with epsilon-greedy for analysis)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, n_na, 5]
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).cpu()  # [n_na]
        
        # Mask out unavailable NAs
        masked_q_values = q_values.clone()
        masked_q_values[~torch.from_numpy(available_mask)] = float('-inf')  # Unavailable NAs -> -inf
        
        # Softmax policy distribution (for entropy calculation)
        valid_q_values = q_values[available_mask]
        if len(valid_q_values) > 1:
            policy_probs = F.softmax(valid_q_values, dim=0).numpy()
            # Policy entropy
            policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-8))
        else:
            policy_entropy = 0.0
            policy_probs = np.array([1.0])
        
        # Record policy-analysis data
        if record_policy:
            self.policy_entropy_history.append(policy_entropy)
            # Record the full action distribution (including unavailable NAs)
            full_policy_dist = np.zeros(self.n_na)
            full_policy_dist[available_indices] = policy_probs
            self.action_distribution_history.append(full_policy_dist.copy())
        
        # Choose action
        if np.random.rand() < self.epsilon:
            # Random selection: pick one from available NAs
            action = np.random.choice(available_indices)
        else:
            # Greedy selection based on Q-values
            action = masked_q_values.argmax().item()
        
        # Update NA selection frequency
        if record_policy:
            self.na_selection_frequency[action] += 1
        
        return action
    
    def update_epsilon_by_episode(self, current_episode):
        """Episode-wise epsilon decay with a low late-stage floor."""
        decay_rate = 0.7
        min_epsilon = 0.005
        base_decay = self.epsilon_start * (decay_rate ** current_episode)
        self.epsilon = max(min_epsilon, base_decay)

    def update_learning_rate(self, current_episode):
        """
        Cosine Annealing with Warm-up learning rate scheduler.
        First 5%: warm-up linearly increases to max LR.
        Last 95%: cosine annealing smoothly decays to min LR.
        """
        if current_episode >= self.total_episodes:
            progress = 1.0
        else:
            progress = current_episode / self.total_episodes
        
        # Warm-up ratio
        warmup_ratio = 0.05  # Warm-up for the first 5%
        
        if progress <= warmup_ratio:
            # Warm-up: linearly increase to max LR
            warmup_progress = progress / warmup_ratio
            current_lr = self.final_lr + (self.initial_lr - self.final_lr) * warmup_progress
        else:
            # Cosine annealing phase
            cosine_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
            import math
            cosine_factor = (1 + math.cos(math.pi * cosine_progress)) / 2
            current_lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine_factor
        
        # Ensure LR does not go below the minimum
        current_lr = max(current_lr, self.final_lr)
        
        # Update optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        return current_lr
    
    def make_final_selection(self, state, initial_reputations):
        """
        Make the final selection using the trained DQN network (same logic as training).
        
        Args:
            state: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations for grouping
        
        Returns:
            dict: selection results, Q-values, and analysis info
        """
        # Use repeated single-selection to construct a final set
        selected_nas = self.make_multiple_selections(state, initial_reputations, 5, 'balanced')
        
        # Compute all NA Q-values for analysis
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        # Group information
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        low_rep_indices = np.where(low_rep_mask)[0]
        high_rep_indices = np.where(high_rep_mask)[0]
        
        # Random selection as a baseline comparison
        random_selected = []
        if len(low_rep_indices) > 0:
            num_low_select = min(2, len(low_rep_indices))
            random_low = np.random.choice(low_rep_indices, num_low_select, replace=False)
            random_selected.extend(random_low)
        if len(high_rep_indices) > 0:
            num_high_select = min(3, len(high_rep_indices))
            random_high = np.random.choice(high_rep_indices, num_high_select, replace=False)
            random_selected.extend(random_high)
        
        return {
            'dqn_selected': selected_nas,
            'random_selected': random_selected,
            'all_q_values': all_q_values,
            'low_rep_indices': low_rep_indices,
            'high_rep_indices': high_rep_indices
        }
    
    def export_na_selection_parameters(self, state, initial_reputations, env=None, output_file=None):
        """
        Export all NA parameters for the last model selection to a file.
        
        Args:
            state: shape (n_na, 4) - current state (reputation, success_rate, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations
            env: Environment object, used to obtain weighted success rate and weighted delay level
            output_file: output path; if None, use a default path
        
        Returns:
            str: output file path
        """
        if output_file is None:
            output_file = str(OUTPUT_ROOT / "src" / "na_selection_parameters.csv")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Compute all NA Q-values
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        # Make final selection
        selection_result = self.make_final_selection(state, initial_reputations)
        selected_nas = selection_result['dqn_selected']
        
        # Get weighted success rate and weighted delay grade (if env is provided)
        weighted_success_rates = []
        weighted_delay_levels = []
        if env is not None:
            for i in range(self.n_na):
                window_summary = env.get_na_window_summary(i)
                weighted_success_rates.append(window_summary['weighted_success_rate'])
                weighted_delay_levels.append(window_summary['weighted_delay_grade'])
        else:
            # If no env is provided, fall back to current-state values
            weighted_success_rates = state[:, 1].tolist()  # success_rate
            weighted_delay_levels = [0] * self.n_na  # Default values
        
        # Prepare export data
        export_data = []
        for i in range(self.n_na):
            na_data = {
                'NA_Index': i,
                'Q_Value': all_q_values[i],
                'Current_Reputation': state[i, 0],  # reputation
                'Success_Rate': state[i, 1],        # success_rate
                'Signature_Delay': env.signature_delay[i] if env is not None else state[i, 2],     # Use raw delay-level value
                'Hunger_Level': state[i, 3],        # hunger
                'Weighted_Success_Rate': weighted_success_rates[i],
                'Weighted_Delay_Grade': weighted_delay_levels[i],
                'Initial_Reputation': initial_reputations[i],
                'Reputation_Group': 'High' if initial_reputations[i] >= 6600 else 'Low',
                'Selected_by_DQN': i in selected_nas,
                'Selection_Rank': selected_nas.index(i) + 1 if i in selected_nas else None
            }
            export_data.append(na_data)
        
        # Sort by Q-value (descending)
        export_data.sort(key=lambda x: x['Q_Value'], reverse=True)
        
        # Add Q-value rank
        for rank, na_data in enumerate(export_data, 1):
            na_data['Q_Value_Rank'] = rank
        
        # Convert to DataFrame and save
        df = pd.DataFrame(export_data)
        df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"\nNA selection parameters exported to: {output_file}")
        print(f"   - Total NAs: {len(df)}")
        print(f"   - NAs selected by DQN: {len([x for x in export_data if x['Selected_by_DQN']])}")
        print(f"   - High-rep NAs: {len([x for x in export_data if x['Reputation_Group'] == 'High'])}")
        print(f"   - Low-rep NAs: {len([x for x in export_data if x['Reputation_Group'] == 'Low'])}")
        print(f"   - Q-value range: {df['Q_Value'].min():.6f} ~ {df['Q_Value'].max():.6f}")
        
        return output_file

    def store(self, s, a, r, s_, done):
        """
        Store experience (now supports single-NA actions).
        
        Args:
            s: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            a: int - selected NA index
            r: float - reward
            s_: shape (n_na, 5) - next state (reputation, success_rate, activity, signature_delay, hunger)
            done: bool - episode done
        """
        self.memory.append((s.copy(), int(a), r, s_.copy(), done))

    def update(self, global_step=None):
        # Check whether we have enough experience to train
        if len(self.memory) < self.min_memory_size:
            return None  # Not enough data; skip training
            
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples for one batch
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(np.array(states)).to(self.device)  # [batch_size, n_na, n_features]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # [batch_size, n_na, n_features]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # [batch_size, 1]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # [batch_size, 1]
        
        # Compute current Q-value
        current_q_values = self.policy_net(states)  # [batch_size, n_na]
        current_q = current_q_values.gather(1, actions)  # [batch_size, 1]
        
        # Compute target Q-value (Double DQN)
        with torch.no_grad():
            # DDQN: select next action with policy net, evaluate with target net
            # 1) Select best next action using the policy network
            next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]  # [batch_size, 1]
            
            # 2) Evaluate the Q-value of that action using the target network (avoid overestimation)
            next_q_values = self.target_net(next_states)  # [batch_size, n_na]
            next_q = next_q_values.gather(1, next_actions)  # [batch_size, 1]
            
            target = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q, target)
        
        # Record loss value and its corresponding training step
        self.loss_history.append(loss.item())
        if global_step is not None:
            self.loss_steps.append(global_step)
        else:
            self.loss_steps.append(self.learn_step)
        
        self.optimizer.zero_grad()
        loss.backward()    
        self.optimizer.step()

        # Update target network every target_update policy updates (balance stability and learning speed)
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()  # Return loss value

    def maybe_update(self, global_step=None):
        """
        Conditional update: train more frequently to strengthen gradient signal.
        
        Returns:
            list: Loss values for this update; None if no training happened.
        """
        self.step_counter += 1
        
        # Check whether update conditions are met
        if (self.step_counter % self.update_frequency == 0 and 
            len(self.memory) >= self.min_memory_size):
            
            # Run multiple training batches to strengthen gradient signal
            batch_losses = []
            training_batches = max(2, self.update_frequency)  # At least 2 batches; can match update frequency
            
            for _ in range(training_batches):
                loss = self.update(global_step=global_step)
                if loss is not None:
                    batch_losses.append(loss)
            
            return batch_losses if batch_losses else None
        
        return None

    def debug_q_values(self, state, initial_reputations):
        """Debug Q-value learning."""
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze().cpu()
        
        print(f"Q-value stats: min={q_values.min():.4f}, max={q_values.max():.4f}, "
              f"std={q_values.std():.4f}")
        
        # Check whether Q-values have meaningful variation
        if q_values.std() < 0.1:
            print("Warning: Q-value variance is too small; the network may not be learning useful information.")
        
        # Compare Q-values between low/high reputation groups
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        
        if np.any(low_rep_mask) and np.any(high_rep_mask):
            low_q_mean = q_values[low_rep_mask].mean()
            high_q_mean = q_values[high_rep_mask].mean()
            print(f"Mean Q (low-reputation): {low_q_mean:.4f}, Mean Q (high-reputation): {high_q_mean:.4f}")
            print(f"Q-value learning direction: {'correct' if high_q_mean > low_q_mean else 'incorrect'}")
        
        return q_values

    def make_multiple_selections(self, state, initial_reputations, num_selections=5, strategy='balanced'):
        """
        Build the final NA set by selecting one NA multiple times.
        
        Args:
            state: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations
            num_selections: int - number of NAs to select
            strategy: str - selection strategy ('balanced', 'top_q', 'by_group')
        
        Returns:
            list: selected NA indices
        """
        original_epsilon = self.epsilon
        self.epsilon = 0  # Pure greedy selection
        
        selected_nas = []
        available_mask = np.ones(self.n_na, dtype=bool)
        
        if strategy == 'balanced':
            # Balanced: sample from low/high reputation groups by ratio
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            # Target: 2 low-reputation + 3 high-reputation (or adjusted by availability)
            target_low = min(2, len(low_rep_indices), num_selections)
            target_high = min(num_selections - target_low, len(high_rep_indices))
            
            # Pick from low-reputation group
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            for _ in range(target_low):
                if np.any(low_mask):
                    selected_na = self.select_action(state, initial_reputations, low_mask)
                    selected_nas.append(selected_na)
                    low_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            # Pick from high-reputation group
            high_mask = np.zeros(self.n_na, dtype=bool)
            high_mask[high_rep_indices] = True
            high_mask &= available_mask  # Exclude already selected
            
            for _ in range(target_high):
                if np.any(high_mask):
                    selected_na = self.select_action(state, initial_reputations, high_mask)
                    selected_nas.append(selected_na)
                    high_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            # If still need more, pick from the remaining pool
            remaining_needed = num_selections - len(selected_nas)
            for _ in range(remaining_needed):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'top_q':
            # Simple: pick top-Q NAs
            for _ in range(min(num_selections, self.n_na)):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'by_group':
            # Grouped: pick low-reputation first, then high-reputation
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            # Pick low-reputation group first
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            while len(selected_nas) < num_selections and np.any(low_mask):
                selected_na = self.select_action(state, initial_reputations, low_mask)
                selected_nas.append(selected_na)
                low_mask[selected_na] = False
                available_mask[selected_na] = False
            
            # Then pick high-reputation group
            high_mask = np.zeros(self.n_na, dtype=bool)
            high_mask[high_rep_indices] = True
            high_mask &= available_mask
            
            while len(selected_nas) < num_selections and np.any(high_mask):
                selected_na = self.select_action(state, initial_reputations, high_mask)
                selected_nas.append(selected_na)
                high_mask[selected_na] = False
                available_mask[selected_na] = False
        
        self.epsilon = original_epsilon
        return selected_nas[:num_selections]

class Environment:
    def __init__(self, n_na, use_fixed_nas, enable_random_fluctuation, window_size=20):
        self.n_na = n_na
        self.use_fixed_nas = use_fixed_nas
        # Random fluctuation toggle
        self.enable_random_fluctuation = enable_random_fluctuation
        if enable_random_fluctuation:
            print("Random fluctuation mode enabled: success rate, hunger, delay, etc. vary over time.")
        else:
            print("Fixed mode: all parameters stay at base values; no random fluctuation.")

        # Hunger growth parameters (kept consistent with the baseline environment)
        self.hunger_growth_scale = 10.0
        self.hunger_growth_log_base = 11.0
        
        self.step_feature_success_rate = None
        self.step_feature_delay_grade = None
        self.sliding_window_enabled = False
        self.window_pack_interval = 0
        self.window_queue_size = 0
        self.ablate_sliding_window_size_one = False
        self.na_window_queues = None
        self.na_current_pack = None
        self.current_step_count = 0
        print("Sliding-window stats disabled: no buffer; state uses per-step features only.")
        
        if use_fixed_nas:
            # Train with a fixed NA set
            print(f"Using a more complex fixed NA set for training (total {n_na} NAs).")
            print("Design principle: each reputation group includes multiple NA types to create contrastive learning conditions.")
            
            # Note: do not reset the random seed; keep global seed consistency.
            # Generate a distinguishable set of NAs using the global RNG state.
            
            # Redesign NA distribution: create four distinct reputation ranges
            quarter = n_na // 4
            remaining = n_na % 4
            
            # Allocate NA counts per range
            group_sizes = [quarter] * 4
            for i in range(remaining):
                group_sizes[i] += 1
            
            print(
                f"NA grouping: very-low {group_sizes[0]}, low {group_sizes[1]}, high {group_sizes[2]}, very-high {group_sizes[3]}"
            )
            
            # Generate reputations for each group
            very_low_rep = np.random.uniform(3300, 4500, group_sizes[0])  # Very-low: 3300-4500
            low_rep = np.random.uniform(4500, 6600, group_sizes[1])  # Low: 4500-6600
            high_rep = np.random.uniform(6600, 8200, group_sizes[2])  # High: 6600-8200
            very_high_rep = np.random.uniform(8200, 9800, group_sizes[3])  # Very-high: 8200-9800
            
            self.fixed_initial_reputation = np.concatenate([very_low_rep, low_rep, high_rep, very_high_rep])
            
            # Diverse success-rate distributions per group to create a learning challenge (with malicious NAs)
            success_rates_list = []
            malicious_flags_list = []  # Track which NAs are malicious
            malicious_types_list = []  # Track malicious behavior type
            
            # Define malicious NA behavior types:
            # Type 1: "high success + high delay" - looks good but intentionally delays
            # Type 2: "high success + stealth attack" - attacks via other channels
            # Type 3: "mid success + highly unstable" - unpredictable behavior
            # Type 4: "classic low success" - clearly low quality
            
            # Very-low reputation (3300-4500): mixed types + malicious NAs
            # 30% truly low, 25% medium, 15% hidden high, 30% malicious
            vl_count_low = int(group_sizes[0] * 0.3)
            vl_count_med = int(group_sizes[0] * 0.25) 
            vl_count_high = int(group_sizes[0] * 0.15)
            vl_count_malicious = group_sizes[0] - vl_count_low - vl_count_med - vl_count_high
            
            vl_sr_low = np.random.uniform(0.25, 0.45, vl_count_low)  # Truly low
            vl_sr_med = np.random.uniform(0.55, 0.70, vl_count_med)  # Medium
            vl_sr_high = np.random.uniform(0.75, 0.90, vl_count_high)  # Hidden high
            
            # Malicious NA type mix for very-low group (simplified to two types)
            vl_mal_type1 = max(1, vl_count_malicious // 2)  # High success + high delay
            vl_mal_type4 = vl_count_malicious - vl_mal_type1  # Classic low success
            
            vl_sr_mal_type1 = np.random.uniform(0.65, 0.80, vl_mal_type1)  # High success camouflage
            vl_sr_mal_type4 = np.random.uniform(0.05, 0.30, vl_mal_type4)  # Low success
            
            vl_sr_malicious = np.concatenate([vl_sr_mal_type1, vl_sr_mal_type4])
            vl_malicious_types = np.concatenate([
                np.full(vl_mal_type1, 1), np.full(vl_mal_type4, 4)
            ])
            
            vl_success_rates = np.concatenate([vl_sr_low, vl_sr_med, vl_sr_high, vl_sr_malicious])
            vl_malicious_flags = np.concatenate([
                np.zeros(vl_count_low + vl_count_med + vl_count_high, dtype=bool),
                np.ones(vl_count_malicious, dtype=bool)
            ])
            vl_all_malicious_types = np.concatenate([
                np.zeros(vl_count_low + vl_count_med + vl_count_high, dtype=int),
                vl_malicious_types
            ])
            
            # Shuffle while keeping flags/types aligned
            shuffle_indices = np.random.permutation(len(vl_success_rates))
            vl_success_rates = vl_success_rates[shuffle_indices]
            vl_malicious_flags = vl_malicious_flags[shuffle_indices]
            vl_all_malicious_types = vl_all_malicious_types[shuffle_indices]
            
            success_rates_list.append(vl_success_rates)
            malicious_flags_list.append(vl_malicious_flags)
            malicious_types_list.append(vl_all_malicious_types)
            
            # Low reputation (4500-6600): more variation + malicious NAs
            # 25% low-mid, 30% medium, 20% promising, 25% malicious
            l_count_low = int(group_sizes[1] * 0.25)
            l_count_med = int(group_sizes[1] * 0.30)
            l_count_high = int(group_sizes[1] * 0.20)
            l_count_malicious = group_sizes[1] - l_count_low - l_count_med - l_count_high
            
            l_sr_low = np.random.uniform(0.35, 0.55, l_count_low)  # Low-mid
            l_sr_med = np.random.uniform(0.60, 0.75, l_count_med)  # Medium
            l_sr_high = np.random.uniform(0.80, 0.95, l_count_high)  # Promising
            
            # Malicious NA type mix for low group (simplified to two types)
            l_mal_type1 = max(1, l_count_malicious // 2)  # High success + high delay
            l_mal_type4 = l_count_malicious - l_mal_type1  # Classic low success (slightly higher here)
            
            l_sr_mal_type1 = np.random.uniform(0.75, 0.90, l_mal_type1)  # High success camouflage
            l_sr_mal_type4 = np.random.uniform(0.15, 0.40, l_mal_type4)  # Relatively low success (higher than very-low)
            
            l_sr_malicious = np.concatenate([l_sr_mal_type1, l_sr_mal_type4])
            l_malicious_types = np.concatenate([
                np.full(l_mal_type1, 1), np.full(l_mal_type4, 4)
            ])
            
            l_success_rates = np.concatenate([l_sr_low, l_sr_med, l_sr_high, l_sr_malicious])
            l_malicious_flags = np.concatenate([
                np.zeros(l_count_low + l_count_med + l_count_high, dtype=bool),
                np.ones(l_count_malicious, dtype=bool)
            ])
            l_all_malicious_types = np.concatenate([
                np.zeros(l_count_low + l_count_med + l_count_high, dtype=int),
                l_malicious_types
            ])
            
            shuffle_indices = np.random.permutation(len(l_success_rates))
            l_success_rates = l_success_rates[shuffle_indices]
            l_malicious_flags = l_malicious_flags[shuffle_indices]
            l_all_malicious_types = l_all_malicious_types[shuffle_indices]
            
            success_rates_list.append(l_success_rates)
            malicious_flags_list.append(l_malicious_flags)
            malicious_types_list.append(l_all_malicious_types)
            
            # High reputation (6600-8200): mostly good with exceptions + malicious NAs
            # 45% high, 25% medium, 10% unexpectedly low, 20% malicious
            h_count_high = int(group_sizes[2] * 0.45)
            h_count_med = int(group_sizes[2] * 0.25)
            h_count_low = int(group_sizes[2] * 0.10)
            h_count_malicious = group_sizes[2] - h_count_high - h_count_med - h_count_low
            
            h_sr_high = np.random.uniform(0.75, 0.90, h_count_high)  # High ability
            h_sr_med = np.random.uniform(0.60, 0.75, h_count_med)  # Medium ability
            h_sr_low = np.random.uniform(0.40, 0.60, h_count_low)  # Unexpectedly low
            
            # Malicious NA type mix for high group (simplified to one type)
            # Keep only Type 1 ("high success + high delay"); Type 4 is unrealistic here.
            h_mal_type1 = h_count_malicious  # All Type 1
            
            h_sr_mal_type1 = np.random.uniform(0.82, 0.95, h_mal_type1)  # Very high success camouflage
            
            h_sr_malicious = h_sr_mal_type1
            h_malicious_types = np.full(h_mal_type1, 1)  # All Type 1
            
            h_success_rates = np.concatenate([h_sr_high, h_sr_med, h_sr_low, h_sr_malicious])
            h_malicious_flags = np.concatenate([
                np.zeros(h_count_high + h_count_med + h_count_low, dtype=bool),
                np.ones(h_count_malicious, dtype=bool)
            ])
            h_all_malicious_types = np.concatenate([
                np.zeros(h_count_high + h_count_med + h_count_low, dtype=int),
                h_malicious_types
            ])
            
            shuffle_indices = np.random.permutation(len(h_success_rates))
            h_success_rates = h_success_rates[shuffle_indices]
            h_malicious_flags = h_malicious_flags[shuffle_indices]
            h_all_malicious_types = h_all_malicious_types[shuffle_indices]
            
            success_rates_list.append(h_success_rates)
            malicious_flags_list.append(h_malicious_flags)
            malicious_types_list.append(h_all_malicious_types)
            
            # Very-high reputation (8200-9800): mostly excellent with some variation + malicious NAs
            # 55% very high, 25% high, 5% medium, 15% malicious (ultimate camouflage)
            vh_count_very_high = int(group_sizes[3] * 0.55)
            vh_count_high = int(group_sizes[3] * 0.25)
            vh_count_med = int(group_sizes[3] * 0.05)
            vh_count_malicious = group_sizes[3] - vh_count_very_high - vh_count_high - vh_count_med
            
            vh_sr_very_high = np.random.uniform(0.85, 0.98, vh_count_very_high)  # Very high ability
            vh_sr_high = np.random.uniform(0.75, 0.85, vh_count_high)  # High ability
            vh_sr_med = np.random.uniform(0.65, 0.75, vh_count_med)  # Medium ability
            
            # Malicious NA type mix for very-high group (simplified to one type)
            # Keep only Type 1 ("very high success + intentional high delay").
            vh_mal_type1 = vh_count_malicious  # All Type 1
            
            vh_sr_mal_type1 = np.random.uniform(0.90, 0.98, vh_mal_type1)  # Extremely high success camouflage
            
            vh_sr_malicious = vh_sr_mal_type1
            vh_malicious_types = np.full(vh_mal_type1, 1)  # All Type 1
            
            vh_success_rates = np.concatenate([vh_sr_very_high, vh_sr_high, vh_sr_med, vh_sr_malicious])
            vh_malicious_flags = np.concatenate([
                np.zeros(vh_count_very_high + vh_count_high + vh_count_med, dtype=bool),
                np.ones(vh_count_malicious, dtype=bool)
            ])
            vh_all_malicious_types = np.concatenate([
                np.zeros(vh_count_very_high + vh_count_high + vh_count_med, dtype=int),
                vh_malicious_types
            ])
            
            shuffle_indices = np.random.permutation(len(vh_success_rates))
            vh_success_rates = vh_success_rates[shuffle_indices]
            vh_malicious_flags = vh_malicious_flags[shuffle_indices]
            vh_all_malicious_types = vh_all_malicious_types[shuffle_indices]
            
            success_rates_list.append(vh_success_rates)
            malicious_flags_list.append(vh_malicious_flags)
            malicious_types_list.append(vh_all_malicious_types)
            
            # Merge all group data
            self.fixed_initial_success_rate = np.concatenate(success_rates_list)
            self.fixed_malicious_flags = np.concatenate(malicious_flags_list)  # Persist malicious flags
            self.fixed_malicious_types = np.concatenate(malicious_types_list)  # Persist malicious type ids
            
            # Activity assignment: do not distinguish malicious NAs by activity; use the same rule as normal NAs.
            
            
            # Assign delay by malicious NA type: key manifestation of malicious behavior
            delay_list = []
            
            for group_idx, group_size in enumerate(group_sizes):
                malicious_mask = malicious_flags_list[group_idx]
                malicious_types = malicious_types_list[group_idx]
                normal_mask = ~malicious_mask
                
                # Normal NA delay distribution (higher reputation tends to lower delay grade)
                if group_idx == 0:  # Very-low group
                    normal_delay = np.random.uniform(0.3, 0.8, np.sum(normal_mask))
                elif group_idx == 1:  # Low group
                    normal_delay = np.random.uniform(0.25, 0.65, np.sum(normal_mask))
                elif group_idx == 2:  # High group
                    normal_delay = np.random.uniform(0.15, 0.5, np.sum(normal_mask))
                else:  # Very-high group
                    normal_delay = np.random.uniform(0.1, 0.4, np.sum(normal_mask))
                
                # Malicious NA delay distribution (designed per type)
                malicious_delay = np.zeros(np.sum(malicious_mask))
                mal_indices = np.where(malicious_mask)[0]
                
                for i, mal_idx in enumerate(mal_indices):
                    mal_type = malicious_types[mal_idx]
                    
                    if mal_type == 1:  # Type 1: high success + high delay (intentional delay attack)
                        # Intentionally creates higher delay grades to degrade system performance
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.7, 1.0)  # Extremely high delay grade
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.6, 0.9)  # Very high delay grade
                        elif group_idx == 2:
                            malicious_delay[i] = np.random.uniform(0.5, 0.8)  # High delay grade (relative to peers)
                        else:
                            malicious_delay[i] = np.random.uniform(0.4, 0.7)  # Noticeably higher (relative to peers)
                            
                    else:  # Type 4: classic low success
                        # Poor overall, including delay grade
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.6, 0.95)  # High delay grade
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.55, 0.9)  # High delay grade
                        # No Type 4 in high / very-high groups
                
                # Combine delays for normal and malicious NAs
                group_delay = np.zeros(group_size)
                group_delay[normal_mask] = normal_delay
                group_delay[malicious_mask] = malicious_delay
                delay_list.append(group_delay)
            
            self.fixed_initial_signature_delay = np.concatenate(delay_list)
            
            # Set a fixed initial hunger for each NA (0-50% uniform) to simulate initial load imbalance
            self.fixed_initial_hunger = np.random.uniform(0.0, 0.5, n_na)
            
            # Derive initial total transactions and success counts
            self.fixed_initial_total_tx = np.random.randint(20, 80, n_na)
            self.fixed_initial_success_count = (self.fixed_initial_success_rate * self.fixed_initial_total_tx).astype(int)
            
            # Delay configuration - use delay grade percentage (0.0-1.0)
            self.ideal_delay_grade = 0.0  # Ideal delay grade: 0% (best)
            self.max_acceptable_delay_grade = 1.0  # Max acceptable delay grade: 100% (worst)
            
            # Note: do not reset the random seed; keep global-seed determinism.
            
            print("Complex NA set preview (with refined malicious NA types):")
            print("id\tinit_rep\tsuccess_rate\tsign_delay(ms)\thunger\tdelay_grade\trep_group\tability_type\tmalicious_type")
            
            # Reputation grouping function
            def get_reputation_group(rep):
                if rep < 4500:
                    return "very-low"
                elif rep < 6600:
                    return "low"
                elif rep < 8200:
                    return "high"
                else:
                    return "very-high"
            
            # Malicious type description
            def get_malicious_type_desc(mal_type):
                if mal_type == 1:
                    return "malicious: high success + high delay"
                elif mal_type == 4:
                    return "malicious: classic low success"
                else:
                    return "normal"
            
            # Ability type (considers malicious flags/types)
            def get_ability_type(rep, success_rate, is_malicious, mal_type):
                if is_malicious:
                    if rep < 4500:
                        return f"malicious-obvious (T{mal_type})"
                    elif rep < 6600:
                        return f"malicious-low-tier (T{mal_type})"
                    elif rep < 8200:
                        return f"malicious-mid-tier (T{mal_type})"
                    else:
                        return f"malicious-high-tier (T{mal_type})"
                
                # Normal NA ability classification
                if rep < 4500:  # Very-low group
                    if success_rate >= 0.75:
                        return "hidden high performer"
                    elif success_rate >= 0.55:
                        return "medium ability"
                    else:
                        return "truly low ability"
                elif rep < 6600:  # Low group
                    if success_rate >= 0.80:
                        return "promising"
                    elif success_rate >= 0.60:
                        return "medium ability"
                    else:
                        return "low-mid ability"
                elif rep < 8200:  # High group
                    if success_rate >= 0.75:
                        return "high ability"
                    elif success_rate >= 0.60:
                        return "medium ability"
                    else:
                        return "unexpectedly low ability"
                else:  # Very-high group
                    if success_rate >= 0.85:
                        return "very high ability"
                    elif success_rate >= 0.75:
                        return "high ability"
                    else:
                        return "medium ability"
            
            for i in range(n_na):
                rep = self.fixed_initial_reputation[i]
                success_rate = self.fixed_initial_success_rate[i]
                
                delay = self.fixed_initial_signature_delay[i]
                delay_level = self.calculate_delay_performance(delay)
                is_malicious = self.fixed_malicious_flags[i]
                mal_type = self.fixed_malicious_types[i] if is_malicious else 0
                
                group = get_reputation_group(rep)
                ability_type = get_ability_type(rep, success_rate, is_malicious, mal_type)
                malicious_type_desc = get_malicious_type_desc(mal_type)
                
                print(f"{i}\t{rep:.1f}\t\t{success_rate:.3f}\t{delay:.1f}\t\t{self.fixed_initial_hunger[i]:.1%}\t\t{delay_level:.3f}\t\t{group}\t{ability_type}\t{malicious_type_desc}")
            
            # Detailed statistics (including malicious NA statistics)
            very_low_indices = np.where(self.fixed_initial_reputation < 4500)[0]
            low_indices = np.where((self.fixed_initial_reputation >= 4500) & (self.fixed_initial_reputation < 6600))[0]
            high_indices = np.where((self.fixed_initial_reputation >= 6600) & (self.fixed_initial_reputation < 8200))[0]
            very_high_indices = np.where(self.fixed_initial_reputation >= 8200)[0]
            
            print(f"\nRefined NA distribution stats (malicious type analysis):")
            
            # Malicious type counting helper
            def count_malicious_types(indices):
                type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                for i in indices:
                    if self.fixed_malicious_flags[i]:
                        mal_type = self.fixed_malicious_types[i]
                        type_counts[mal_type] += 1
                return type_counts
            
            # Very-low group stats
            if len(very_low_indices) > 0:
                vl_malicious = sum(1 for i in very_low_indices if self.fixed_malicious_flags[i])
                vl_normal = len(very_low_indices) - vl_malicious
                vl_mal_types = count_malicious_types(very_low_indices)
                vl_high_ability = sum(1 for i in very_low_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.75)
                vl_med_ability = sum(1 for i in very_low_indices if not self.fixed_malicious_flags[i] and 0.55 <= self.fixed_initial_success_rate[i] < 0.75)
                vl_low_ability = vl_normal - vl_high_ability - vl_med_ability
                print(
                    f"very-low ({len(very_low_indices)}): malicious {vl_malicious} [T1:{vl_mal_types[1]}, T2:{vl_mal_types[2]}, T3:{vl_mal_types[3]}, T4:{vl_mal_types[4]}] | "
                    f"normal {vl_normal} (hidden-high {vl_high_ability}, medium {vl_med_ability}, low {vl_low_ability})"
                )
            
            # Low group stats
            if len(low_indices) > 0:
                l_malicious = sum(1 for i in low_indices if self.fixed_malicious_flags[i])
                l_normal = len(low_indices) - l_malicious
                l_mal_types = count_malicious_types(low_indices)
                l_high_ability = sum(1 for i in low_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.80)
                l_med_ability = sum(1 for i in low_indices if not self.fixed_malicious_flags[i] and 0.60 <= self.fixed_initial_success_rate[i] < 0.80)
                l_low_ability = l_normal - l_high_ability - l_med_ability
                print(
                    f"low ({len(low_indices)}): malicious {l_malicious} [T1:{l_mal_types[1]}, T2:{l_mal_types[2]}, T3:{l_mal_types[3]}, T4:{l_mal_types[4]}] | "
                    f"normal {l_normal} (promising {l_high_ability}, medium {l_med_ability}, low-mid {l_low_ability})"
                )
            
            # High group stats
            if len(high_indices) > 0:
                h_malicious = sum(1 for i in high_indices if self.fixed_malicious_flags[i])
                h_normal = len(high_indices) - h_malicious
                h_mal_types = count_malicious_types(high_indices)
                h_high_ability = sum(1 for i in high_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.75)
                h_med_ability = sum(1 for i in high_indices if not self.fixed_malicious_flags[i] and 0.60 <= self.fixed_initial_success_rate[i] < 0.75)
                h_low_ability = h_normal - h_high_ability - h_med_ability
                print(
                    f"high ({len(high_indices)}): malicious {h_malicious} | normal {h_normal} "
                    f"(high {h_high_ability}, medium {h_med_ability}, unexpectedly-low {h_low_ability})"
                )
            
            # Very-high group stats
            if len(very_high_indices) > 0:
                vh_malicious = sum(1 for i in very_high_indices if self.fixed_malicious_flags[i])
                vh_normal = len(very_high_indices) - vh_malicious
                vh_very_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.85)
                vh_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and 0.75 <= self.fixed_initial_success_rate[i] < 0.85)
                vh_med_ability = vh_normal - vh_very_high_ability - vh_high_ability
                print(
                    f"very-high ({len(very_high_indices)}): malicious {vh_malicious} | normal {vh_normal} "
                    f"(very-high {vh_very_high_ability}, high {vh_high_ability}, medium {vh_med_ability})"
                )
            
            # Overall malicious NA stats
            total_malicious = np.sum(self.fixed_malicious_flags)
            total_normal = n_na - total_malicious
            print(
                f"\nOverall malicious NA stats: malicious {total_malicious} ({total_malicious/n_na:.1%}), "
                f"normal {total_normal} ({total_normal/n_na:.1%})"
            )
            

        else:
            # In non-fixed NA mode, delay parameters also need initialization
            self.ideal_delay = 100.0
            self.max_acceptable_delay = self.ideal_delay * 3
        
        # Initialize current parameters
        self.current_time = 30  # Start time point: 30 minutes
        
        
        # Call reset() to initialize all state
        self.reset()
    
    def calculate_delay_performance(self, delay_grade):
        """
        Compute delay grade (higher delay => higher grade).
        
        Args:
            delay_grade: delay grade (0.0-1.0)
            
        Returns:
            float: delay grade, where 0.0 is lowest and 1.0 is highest
        """
        # Clamp into [0.0, 1.0]
        return max(0.0, min(1.0, delay_grade))
    
    
    def reset(self):
        """
        Reset environment state. Uses fixed initial settings when enabled.
        """
        if self.use_fixed_nas:
            # Use fixed NA set; reset only dynamic state
            self.reputation = self.fixed_initial_reputation.copy()
            self.total_tx = self.fixed_initial_total_tx.copy()
            self.success_count = self.fixed_initial_success_count.copy()
            self.success_rate = self.fixed_initial_success_rate.copy()
            
            # Signature delay
            self.signature_delay = self.fixed_initial_signature_delay.copy()
            # Use fixed initial hunger (each NA has a preset initial hunger)
            self.hunger = self.fixed_initial_hunger.copy()
            # Reconstruct last-selected step from initial hunger to simulate history
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)
            for i in range(self.n_na):
                # hunger = log(1 + steps/20) / log(11)  =>  steps = 20 * (11^hunger - 1)
                if self.hunger[i] > 0:
                    estimated_steps = int(20 * (np.power(11, self.hunger[i]) - 1))
                    self.last_selected_time[i] = -estimated_steps
                else:
                    self.last_selected_time[i] = 0  # hunger=0 means "just selected"
        else:
            # Original random reset
            self.reputation = np.random.uniform(3300, 10000, self.n_na)
            self.total_tx = np.random.randint(1, 100, self.n_na)
            self.success_count = np.array([
                np.random.randint(0, self.total_tx[i] + 1) 
                for i in range(self.n_na)
            ])
            self.success_rate = self.success_count / self.total_tx
            
            # Randomly initialize signature delay
            self.signature_delay = np.random.uniform(50, 200, self.n_na)
            # Randomly initialize hunger (0-50% range) to simulate initial load differences
            self.hunger = np.random.uniform(0.0, 0.5, self.n_na)
            # Record last-selected step reconstructed from hunger
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)
            for i in range(self.n_na):
                if self.hunger[i] > 0:
                    estimated_steps = int(20 * (np.power(11, self.hunger[i]) - 1))
                    self.last_selected_time[i] = -estimated_steps
                else:
                    self.last_selected_time[i] = 0
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)

        # Reset time-related state
        self.current_time = 30
        # Training step counter (used to compute hunger)
        self.training_step = 0
        
        self.current_step_count = 0

        self.step_feature_success_rate = self.success_rate.copy()
        self.step_feature_delay_grade = self.signature_delay.copy()

        # Record starting reputation for this episode (fixed base reputation)
        self.episode_start_reputation = self.reputation.copy()
        # Reset last-step reputation
        self.last_step_reputation = self.reputation.copy()

        return self.get_state()
    def get_state(self):
        # Update: remove activity feature; keep 4 features (reputation, success_rate, signature_delay, hunger)
        success_feature = self.step_feature_success_rate if self.step_feature_success_rate is not None else self.success_rate
        delay_feature = self.step_feature_delay_grade if self.step_feature_delay_grade is not None else self.signature_delay

        raw_state = np.column_stack([
            self.reputation,
            success_feature,
            delay_feature,
            self.hunger
        ])

        # Normalize
        normalized_state = raw_state.copy()
        normalized_state[:, 0] = (self.reputation - 3000) / (10000 - 3000)  # Reputation normalization
        normalized_state[:, 1] = success_feature  # success_rate is already in [0,1]
        # delay grade is already in [0,1]
        normalized_state[:, 2] = delay_feature
        normalized_state[:, 3] = self.hunger  # hunger is already in [0,1]
        return normalized_state

    def step(self, action_na_index, selected_mask=None):
        """
        Simplified environment step that handles a single NA.
        
        Args:
            action_na_index: int - selected NA index
            selected_mask: kept for backward compatibility (unused)
        
        Returns:
            next_state: updated state
            reward: reward for a single NA
            done: whether episode ends (always False)
        """
        # Ensure index is within valid range
        na_idx = max(0, min(int(action_na_index), self.n_na - 1))
        
        # Record pre-action reputation and success rate
        old_reputation = self.reputation[na_idx]
        old_success_rate = self.success_rate[na_idx]
        
        # Update time
        self.current_time += 1
        # Update training step counter
        self.training_step += 1
        # self._update_hunger(na_idx)
        # Simulate per-transaction delay grade (controlled random fluctuation)
        base_delay_grade = self.signature_delay[na_idx]
        if self.enable_random_fluctuation:
            # Add ±0.05 fluctuation (delay grade range 0.0-1.0)
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
        else:
            # Fixed mode: add small fluctuation (±0.01) to simulate measurement noise
            current_delay_grade = base_delay_grade + np.random.uniform(-0.01, 0.01)
        current_delay_grade = np.clip(current_delay_grade, 0.0, 1.0)  # Clamp into 0.0-1.0
        
        # Success-rate random fluctuation (if enabled)
        effective_success_rate = self.success_rate[na_idx]
        if self.enable_random_fluctuation:
            # ±5% fluctuation, clamped into [0,1]
            fluctuation = np.random.uniform(-0.05, 0.05)
            effective_success_rate = np.clip(effective_success_rate + fluctuation, 0.0, 1.0)
        
        # Sample success based on (possibly fluctuating) success rate
        success = np.random.random() < effective_success_rate
        
        # Compute reward parameters (used for reputation updates)
        norm_rep = (old_reputation - 3300.0) / (10000.0 - 3300.0)
        if old_reputation < 6600:
            computed_reward = (0.4 * self.success_rate[na_idx]
                              + 0.4 * norm_rep
                              + 0.2 * self.hunger[na_idx])
        else:
            computed_reward = (0.4 * self.success_rate[na_idx]
                              + 0.2 * norm_rep
                              + 0.4 * self.hunger[na_idx])
        
        # Update NA parameters
        reputation_increase = 0
        reputation_decrease = 0

        if success:
            self.total_tx[na_idx] += 1
            self.success_count[na_idx] += 1
            self.success_rate[na_idx] = self.success_count[na_idx] / self.total_tx[na_idx]
            reputation_increase = computed_reward * 20
            self.reputation[na_idx] = min(10000, old_reputation + reputation_increase)
        else:
            self.total_tx[na_idx] += 1
            self.success_rate[na_idx] = self.success_count[na_idx] / self.total_tx[na_idx]
            reputation_decrease = computed_reward * 100
            self.reputation[na_idx] = max(3300, old_reputation - reputation_decrease)

        # Reward is based on NA quality metrics rather than a single transaction outcome.
        # Old: reward = (+1.0 if success else -1.0) + delay adjustment
        # New: reward = f(weighted_success_rate) + weighted_delay_adjustment + hunger_adjustment
        #
        # Rationale:
        # 1) More stable learning signal (less single-step randomness).
        # 2) Reflects underlying NA quality via historical/weighted metrics.
        # 3) Smoother reward gradient across [0.0, 1.0] success rates.
        # 4) Less noise: a single failure should not heavily penalize a high-quality NA.
        # 5) Load balancing: high-hunger, high-quality NAs can get extra reward.
        # 6) Uses weighted success rate and delay grade (from windowed stats when enabled).
        
        effective_delay_performance = self.calculate_delay_performance(current_delay_grade)

        if self.step_feature_success_rate is not None:
            self.step_feature_success_rate[na_idx] = float(effective_success_rate)
        if self.step_feature_delay_grade is not None:
            self.step_feature_delay_grade[na_idx] = float(current_delay_grade)
        
        # Base reward from weighted success rate; expand reward range and raise positive threshold.
        # Old: 2.5 * effective_success_rate - 1.5 (zero at success_rate=0.6)
        # New: 4.0 * effective_success_rate - 2.8 (zero at success_rate=0.7)
        weighted_success_rate_reward = 4.0 * effective_success_rate - 2.8  # Map [0,1] -> [-2.8, 1.2]
        
        # Delay adjustment based on weighted delay grade.
        # Delay grade range: [0,1], where 0 is best and 1 is worst.
        # Old: 0.3 * (1 - 2 * effective_delay_performance) -> [0.3, -0.3]
        # New: 0.8 * (1 - 2 * effective_delay_performance) -> [0.8, -0.8]
        weighted_delay_grade_bonus = 0.8 * (1 - 2 * effective_delay_performance)  # Map [0,1] -> [0.8, -0.8]
        
        # Hunger adjustment (with optional controlled noise).
        # Design principles:
        # 1) High success + low delay + high hunger => extra reward (prefer long-unselected good NAs).
        # 2) Low success or high delay + high hunger => tiny/no reward (avoid preferring low-quality NAs).
        # 3) Hunger weight depends on both weighted success rate and weighted delay performance.
        
        # Hunger random fluctuation (if enabled)
        effective_hunger = self.hunger[na_idx]
        if self.enable_random_fluctuation:
            # ±10% fluctuation, clamped into [0,1]
            hunger_fluctuation = np.random.uniform(-0.1, 0.1)
            effective_hunger = np.clip(effective_hunger + hunger_fluctuation, 0.0, 1.0)
        
        # Overall quality weight: 60% weighted success rate + 40% weighted delay performance.
        # Only NAs with both decent success and delay performance should receive notable hunger reward.
        # Note: delay performance uses 0 as best and 1 as worst, so use (1 - delay_performance).
        quality_weight = 0.6 * effective_success_rate + 0.4 * (1 - effective_delay_performance)
        
        # Dynamic hunger weight: higher quality => larger hunger weight.
        # Use a smooth non-linear function to amplify high-quality NAs.
        # Base weight range: 0.2 to 1.5; use exponent to increase non-linearity.
        base_hunger_weight = 0.2 + 1.3 * (quality_weight ** 1.5)  # Exponent 1.5 for stronger non-linearity
        
        # Further adjust by quality tier to enforce stronger differentiation
        if quality_weight >= 0.8:
            # Top-tier quality: maximum hunger weight
            dynamic_hunger_weight = min(1.8, base_hunger_weight * 1.2)  # Max 1.8
        elif quality_weight >= 0.6:
            # High quality: large hunger weight
            dynamic_hunger_weight = base_hunger_weight * 1.0  # Standard weight
        elif quality_weight >= 0.4:
            # Medium quality: medium hunger weight
            dynamic_hunger_weight = base_hunger_weight * 0.8  # -20%
        else:
            # Low quality: small hunger weight
            dynamic_hunger_weight = base_hunger_weight * 0.5  # -50%
        
        # Hunger reward
        hunger_bonus = dynamic_hunger_weight * quality_weight * effective_hunger
        
        # Total reward: weighted success rate + weighted delay grade + hunger bonus
        reward = weighted_success_rate_reward + weighted_delay_grade_bonus + hunger_bonus
        
        self._update_hunger(na_idx)
        
        return self.get_state(), reward, False

    def _update_na_window_queue(self, na_idx, transaction_data):
        """
        Update the current transaction pack for a given NA and accumulate behavior data.
        
        Args:
            na_idx: NA index
            transaction_data: dict containing transaction information
        """
        current_pack = self.na_current_pack[na_idx]
        
        # If this is the beginning of a new pack, record start step
        if len(current_pack['transactions']) == 0:
            current_pack['start_step'] = self.current_step_count
        
        # Add transaction to current pack
        current_pack['transactions'].append(transaction_data)
        current_pack['total_count'] += 1
        if transaction_data['success']:
            current_pack['success_count'] += 1
        
        # Record reputation change
        reputation_change = transaction_data['reputation_after'] - transaction_data['reputation_before']
        current_pack['reputation_changes'].append(reputation_change)
        
        # Update end step
        current_pack['end_step'] = self.current_step_count
    
    def _pack_all_na_windows(self):
        """
        Pack each NA's current transaction pack into its window queue.
        """
        for na_idx in range(self.n_na):
            current_pack = self.na_current_pack[na_idx]
            
            # Pack only when there are transactions
            if current_pack['total_count'] > 0:
                # Average delay of the pack
                avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions']) if current_pack['transactions'] else 0.0
                
                # Delay grade
                delay_grade = self.calculate_delay_performance(avg_delay)
                
                # Pack summary stats
                pack_summary = {
                    'start_step': current_pack['start_step'],
                    'end_step': current_pack['end_step'],
                    'transaction_count': current_pack['total_count'],
                    'success_count': current_pack['success_count'],
                    'success_rate': current_pack['success_count'] / current_pack['total_count'],
                    'total_reputation_change': sum(current_pack['reputation_changes']),
                    'avg_reputation_change': sum(current_pack['reputation_changes']) / len(current_pack['reputation_changes']) if current_pack['reputation_changes'] else 0,
                    'avg_delay': avg_delay,
                    'delay_grade': delay_grade,  # New: delay grade field
                    'transactions': current_pack['transactions'].copy()  # Keep raw transaction data
                }
                
                if self.ablate_sliding_window_size_one:
                    queue = self.na_window_queues[na_idx]
                    queue.clear()
                    repeat_n = max(int(self.window_queue_size), 1)
                    for _ in range(repeat_n):
                        replicated = dict(pack_summary)
                        replicated['transactions'] = pack_summary['transactions'].copy()
                        queue.append(replicated)
                else:
                    self.na_window_queues[na_idx].append(pack_summary)
                
                # Reset current pack
                current_pack['transactions'] = []
                current_pack['success_count'] = 0
                current_pack['total_count'] = 0
                current_pack['reputation_changes'] = []
                current_pack['start_step'] = 0
                current_pack['end_step'] = 0
    
    def get_na_window_summary(self, na_idx):
        """
        Get a statistical summary for a given NA's window queue.
        
        Args:
            na_idx: NA index
            
        Returns:
            dict: summary statistics of transactions in the window queue
        """
        if (not getattr(self, 'sliding_window_enabled', True)
                or self.na_window_queues is None
                or self.na_current_pack is None):
            current_success = float(self.step_feature_success_rate[na_idx]) if self.step_feature_success_rate is not None else float(self.success_rate[na_idx])
            current_delay_grade = float(self.step_feature_delay_grade[na_idx]) if self.step_feature_delay_grade is not None else float(self.signature_delay[na_idx])
            weighted_delay_grade = self.calculate_delay_performance(current_delay_grade)

            delay_grade = "A"
            if weighted_delay_grade > 0.75:
                delay_grade = "D"
            elif weighted_delay_grade > 0.5:
                delay_grade = "C"
            elif weighted_delay_grade > 0.25:
                delay_grade = "B"

            return {
                'queue_size': 0,
                'current_pack_size': 0,
                'total_transactions': 0,
                'total_success_count': 0,
                'overall_success_rate': current_success,
                'total_reputation_change': 0.0,
                'avg_pack_success_rate': current_success,
                'weighted_success_rate': current_success,
                'weighted_avg_delay': current_delay_grade,
                'weighted_delay_grade': weighted_delay_grade,
                'delay_grade': delay_grade,
                'step_range': None,
                'pack_details': []
            }

        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            return {
                'queue_size': 0,
                'current_pack_size': 0,
                'total_transactions': 0,
                'total_success_count': 0,
                'overall_success_rate': 0.0,
                'total_reputation_change': 0.0,
                'avg_pack_success_rate': 0.0,
                'weighted_success_rate': 0.0,  # No data: weighted success rate = 0
                'weighted_avg_delay': 0.0,  # No data: weighted average delay = 0
                'weighted_delay_grade': 0.0,  # No data: delay grade = 0 (worst)
                'delay_grade': "N/A",  # No data: show N/A
                'step_range': None
            }
        
        # Compute stats across all packs in the queue
        total_transactions = sum(pack['transaction_count'] for pack in queue)
        total_success_count = sum(pack['success_count'] for pack in queue)
        total_reputation_change = sum(pack['total_reputation_change'] for pack in queue)
        
        # Include the current accumulating pack
        total_transactions += current_pack['total_count']
        total_success_count += current_pack['success_count']
        total_reputation_change += sum(current_pack['reputation_changes'])
        
        # Overall success rate
        overall_success_rate = total_success_count / total_transactions if total_transactions > 0 else 0.0
        
        # Average pack success rate
        pack_success_rates = [pack['success_rate'] for pack in queue if pack['transaction_count'] > 0]
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            pack_success_rates.append(current_pack_success_rate)
        
        avg_pack_success_rate = sum(pack_success_rates) / len(pack_success_rates) if pack_success_rates else 0.0
        
        # Weighted success rate and weighted delay (newer packs get higher weight)
        weighted_success_rate = 0.0
        weighted_avg_delay = 0.0
        weighted_delay_grade = 0.0  # Weighted delay grade
        total_weight = 0.0
        
        # Assign weights for queue packs (newer => higher weight)
        if queue:
            for i, pack in enumerate(queue):
                # Linear weights; newest pack has the highest weight
                weight = i + 1  # Starts from 1
                
                # Average delay for this pack
                pack_avg_delay = 0.0
                if pack['transactions']:
                    pack_avg_delay = sum(tx['delay'] for tx in pack['transactions']) / len(pack['transactions'])
                
                # Use delay_grade in the pack if present
                pack_delay_grade = pack.get('delay_grade', 0.0)
                
                weighted_success_rate += pack['success_rate'] * weight
                weighted_avg_delay += pack_avg_delay * weight
                weighted_delay_grade += pack_delay_grade * weight
                total_weight += weight
        
        # Include current accumulating pack as the highest weight
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            
            # Average delay for the current pack
            current_pack_avg_delay = 0.0
            if current_pack['transactions']:
                current_pack_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions'])
            
            # Delay grade for the current pack
            current_pack_delay_grade = self.calculate_delay_performance(current_pack_avg_delay)
            
            # Current pack has the highest weight
            current_weight = len(queue) + 1
            weighted_success_rate += current_pack_success_rate * current_weight
            weighted_avg_delay += current_pack_avg_delay * current_weight
            weighted_delay_grade += current_pack_delay_grade * current_weight
            total_weight += current_weight
        
        # Normalize weighted values
        if total_weight > 0:
            weighted_success_rate /= total_weight
            weighted_avg_delay /= total_weight
            weighted_delay_grade /= total_weight
        
        # If no history exists, use a default delay grade
        if total_weight == 0:
            weighted_delay_grade = 0.5  # Medium performance
        
        # Letter-grade display logic based on delay grade
        delay_grade = "A"
        if weighted_delay_grade > 0.75:  # Delay grade > 75%
            delay_grade = "D"
        elif weighted_delay_grade > 0.5:  # Delay grade > 50%
            delay_grade = "C"
        elif weighted_delay_grade > 0.25:  # Delay grade > 25%
            delay_grade = "B"
        else:  # Delay grade <= 25%
            delay_grade = "A"
        
        # Step range
        step_range = None
        if queue:
            min_step = min(pack['start_step'] for pack in queue)
            max_step = max(pack['end_step'] for pack in queue)
            if current_pack['total_count'] > 0:
                max_step = max(max_step, current_pack['end_step'])
            step_range = (min_step, max_step)
        elif current_pack['total_count'] > 0:
            step_range = (current_pack['start_step'], current_pack['end_step'])
        
        return {
            'queue_size': len(queue),
            'current_pack_size': current_pack['total_count'],
            'total_transactions': total_transactions,
            'total_success_count': total_success_count,
            'overall_success_rate': overall_success_rate,
            'total_reputation_change': total_reputation_change,
            'avg_pack_success_rate': avg_pack_success_rate,
            'weighted_success_rate': weighted_success_rate,
            'weighted_avg_delay': weighted_avg_delay,
            'weighted_delay_grade': weighted_delay_grade,  # Weighted delay grade
            'delay_grade': delay_grade,
            'step_range': step_range,
            'pack_details': [
                {
                    'step_range': (pack['start_step'], pack['end_step']),
                    'transaction_count': pack['transaction_count'],
                    'success_rate': pack['success_rate'],
                    'reputation_change': pack['total_reputation_change'],
                    'weight': i + 1,  # Pack weight
                    'avg_delay': sum(tx['delay'] for tx in pack['transactions']) / len(pack['transactions']) if pack['transactions'] else 0.0
                } for i, pack in enumerate(queue)
            ]
        }
    
    def print_sliding_window_summary(self, max_nas_to_show=10):
        """
        Print a sliding-window statistics summary.
        
        Args:
            max_nas_to_show: maximum number of NAs to show
        """
        if (not getattr(self, 'sliding_window_enabled', True)
                or self.na_window_queues is None
                or self.na_current_pack is None):
            print("\nSliding-window stats disabled: buffer-free ablation mode.")
            return

        print(
            f"\nSliding-window queue summary (pack interval: {self.window_pack_interval} steps, queue capacity: {self.window_queue_size})"
        )
        print("=" * 140)
        
        summaries = self.get_all_nas_window_summary()
        
        # Sort by weighted success rate
        sorted_nas = sorted(summaries.items(), key=lambda x: x[1]['weighted_success_rate'], reverse=True)
        
        print(
            f"{'NA':<6} {'queue':<6} {'pack':<8} {'tx':<8} {'w_succ':<12} {'w_delay':<10} {'grade':<8} {'rep_delta':<10} {'steps':<15}"
        )
        print("-" * 140)
        
        shown_count = 0
        for na_idx, summary in sorted_nas:
            if shown_count >= max_nas_to_show:
                break
                
            if summary['total_transactions'] > 0:
                step_range_str = f"{summary['step_range']}" if summary['step_range'] else "None"
                print(f"NA{na_idx:<4} {summary['queue_size']:<6} {summary['current_pack_size']:<8} {summary['total_transactions']:<8} {summary['weighted_success_rate']:<12.3f} {summary['weighted_avg_delay']:<10.1f} {summary['delay_grade']:<8} {summary['total_reputation_change']:<10.1f} {step_range_str:<15}")
                shown_count += 1
        
        # Overall stats
        total_transactions = sum(s['total_transactions'] for s in summaries.values())
        total_success = sum(s['total_success_count'] for s in summaries.values())
        active_nas = sum(1 for s in summaries.values() if s['total_transactions'] > 0)
        
        # System-level weighted metrics
        system_weighted_success_rate = 0.0
        system_weighted_delay = 0.0
        total_weight = 0.0
        
        for summary in summaries.values():
            if summary['total_transactions'] > 0:
                weight = summary['total_transactions']  # Use transaction count as weight
                system_weighted_success_rate += summary['weighted_success_rate'] * weight
                system_weighted_delay += summary['weighted_avg_delay'] * weight
                total_weight += weight
        
        if total_weight > 0:
            system_weighted_success_rate /= total_weight
            system_weighted_delay /= total_weight
        
        print("-" * 140)
        print(f"Overall: {active_nas} NAs have transactions, total tx: {total_transactions}, total success: {total_success}")
        if total_transactions > 0:
            print(
                f"System overall success rate: {total_success/total_transactions:.3f}, system weighted success rate: {system_weighted_success_rate:.3f}"
            )
            print(
                f"System weighted avg delay: {system_weighted_delay:.1%}, avg per active NA: {total_transactions/max(1, active_nas):.1f} tx"
            )
        print(f"Current step: {self.current_step_count}")
        
        if shown_count < active_nas:
            print(f"(Showing top {shown_count} active NAs; {active_nas - shown_count} more have transactions.)")

    def show_na_detailed_queue(self, na_idx, max_packs_to_show=5):
        """
        Show detailed queue information for a specific NA, including weight computation.
        
        Args:
            na_idx: NA index
            max_packs_to_show: maximum number of packs to show
        """
        if na_idx >= self.n_na:
            print(f"Invalid NA index: {na_idx}")
            return
        
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\nNA{na_idx} detailed queue info")
        print("=" * 100)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("No transactions for this NA.")
            return
        
        # Show queue packs (weight increases with recency)
        print(f"Packs in queue (total {len(queue)}; enqueue order; newer => higher weight):")
        print(f"{'pack':<8} {'weight':<6} {'steps':<15} {'tx':<8} {'succ':<10} {'avg_delay':<10} {'rep_delta':<10}")
        print("-" * 100)
        
        shown_packs = 0
        for i, pack in enumerate(queue):
            if shown_packs >= max_packs_to_show:
                break
            
            weight = i + 1
            step_range = f"({pack['start_step']}-{pack['end_step']})"
            avg_delay = pack.get('avg_delay', 0.0)
            
            print(
                f"pack{i+1:<4} {weight:<6} {step_range:<15} {pack['transaction_count']:<8} {pack['success_rate']:<10.3f} {avg_delay:<10.1%} {pack['total_reputation_change']:<10.1f}"
            )
            shown_packs += 1
        
        # Current accumulating pack
        if current_pack['total_count'] > 0:
            current_weight = len(queue) + 1
            current_success_rate = current_pack['success_count'] / current_pack['total_count']
            current_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions']) if current_pack['transactions'] else 0.0
            current_step_range = f"({current_pack['start_step']}-{current_pack['end_step']})"
            current_reputation_change = sum(current_pack['reputation_changes'])
            
            print(
                f"current {current_weight:<6} {current_step_range:<15} {current_pack['total_count']:<8} {current_success_rate:<10.3f} {current_avg_delay:<10.1%} {current_reputation_change:<10.1f}"
            )
        
        # Weighted results
        summary = self.get_na_window_summary(na_idx)
        print("\nWeighted results:")
        print(f"  - weighted success rate: {summary['weighted_success_rate']:.3f}")
        print(f"  - weighted avg delay: {summary['weighted_avg_delay']:.1%}")
        print(f"  - delay grade: {summary['delay_grade']}")
        print(f"  - total transactions: {summary['total_transactions']}")
        print(f"  - queue capacity used: {len(queue)}/{self.window_queue_size}")
        
        if len(queue) < max_packs_to_show and len(queue) > shown_packs:
            print(f"\n({len(queue) - shown_packs} more packs not shown.)")

    def get_all_nas_window_summary(self):
        """
        Get sliding-window summaries for all NAs.
        
        Returns:
            dict: summaries keyed by NA index
        """
        summaries = {}
        for na_idx in range(self.n_na):
            summaries[na_idx] = self.get_na_window_summary(na_idx)
        return summaries
    
    def print_na_queue_details(self, na_idx, max_packs_to_show=5):
        """
        Print detailed queue information for a specific NA.
        
        Args:
            na_idx: NA index
            max_packs_to_show: maximum number of packs to show
        """
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\nNA{na_idx} detailed queue info:")
        print("=" * 80)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("  No transactions for this NA.")
            return
        
        # Queue packs
        if len(queue) > 0:
            print(f"Packed data (queue size: {len(queue)}/{self.window_queue_size}):")
            
            # Show most recent packs
            recent_packs = list(queue)[-max_packs_to_show:]
            for i, pack in enumerate(recent_packs):
                pack_idx = len(queue) - max_packs_to_show + i
                if pack_idx < 0:
                    pack_idx = i
                
                print(
                    f"  pack#{pack_idx+1}: steps {pack['start_step']}-{pack['end_step']}, "
                    f"tx {pack['transaction_count']}, success {pack['success_rate']:.2f}, "
                    f"rep_delta {pack['total_reputation_change']:+.2f}"
                )
            
            if len(queue) > max_packs_to_show:
                print(f"  ... ({len(queue) - max_packs_to_show} older packs not shown)")
        
        # Current accumulating pack
        if current_pack['total_count'] > 0:
            print(f"\nCurrent accumulating pack:")
            print(
                f"  steps {current_pack['start_step']}-{current_pack['end_step']}, "
                f"tx {current_pack['total_count']}, "
                f"success {current_pack['success_count']} "
                f"({current_pack['success_count']/current_pack['total_count']:.2f}), "
                f"rep_delta {sum(current_pack['reputation_changes']):+.2f}"
            )
        else:
            print(f"\nCurrent accumulating pack: empty")
        
        # Summary
        summary = self.get_na_window_summary(na_idx)
        print(f"\nSummary:")
        print(f"  total transactions: {summary['total_transactions']}")
        print(f"  total success: {summary['total_success_count']}")
        print(f"  overall success rate: {summary['overall_success_rate']:.2f}")
        print(f"  total rep_delta: {summary['total_reputation_change']:+.2f}")
        if summary['step_range']:
            print(f"  step range: {summary['step_range'][0]} - {summary['step_range'][1]}")
        print("=" * 80)

    def _update_hunger(self, selected_na_idx):
        """
        Update hunger for all NAs.
        
        Logic:
        1. Reset selected NA hunger to 0.
        2. Increase other NAs' hunger over time.
        3. Clamp hunger into [0, 1].
        
        Args:
            selected_na_idx: selected NA index
        """
        # Update selected NA record
        self.last_selected_time[selected_na_idx] = self.training_step

        # Update hunger for all NAs
        for i in range(self.n_na):
            steps_since_selected = self.training_step - self.last_selected_time[i]

            if steps_since_selected <= 0:
                self.hunger[i] = 0.0
            else:
                normalized_steps = steps_since_selected / self.hunger_growth_scale
                self.hunger[i] = min(1.0, np.log(1 + normalized_steps) / np.log(self.hunger_growth_log_base))

        # Reset selected NA hunger to 0
        self.hunger[selected_na_idx] = 0.0

def simulate_selected_nas_training(env, selected_nas, title, steps=100):
    """
    Simulate transactions for selected NAs; track reputation changes only (no rewards).
    
    Args:
        env: environment object
        selected_nas: list of selected NA indices
        title: title string
        steps: number of simulated steps
    
    Returns:
        dict: per-NA simulation traces
    """
    print(f"\n{title} - NA transaction simulation ({steps} transactions):")
    print("=" * 80)
    
    # Save current environment state (post-training state)
    original_reputation = env.reputation.copy()
    original_success_rate = env.success_rate.copy()
    
    original_current_time = env.current_time
    
    original_total_tx = env.total_tx.copy()
    original_success_count = env.success_count.copy()
    
    # Per-NA simulation traces
    training_results = {}
    
    for na_idx in selected_nas:
        # Reset environment to fixed initial state
        env.reputation = env.fixed_initial_reputation.copy()
        env.success_rate = env.fixed_initial_success_rate.copy()
        
        env.current_time = 0  # Reset time
        
        env.total_tx = env.fixed_initial_total_tx.copy()
        env.success_count = env.fixed_initial_success_count.copy()
        
        initial_rep = env.reputation[na_idx]
        initial_success_rate = env.success_rate[na_idx]
        initial_hunger = env.hunger[na_idx]
        initial_signature_delay = env.signature_delay[na_idx]
        
        reputation_history = [initial_rep]
        transaction_results = []  # Detailed per-transaction results
        
        print(f"\nNA {na_idx} transaction trace:")
        print(
            f"   Initial: rep={initial_rep:.2f}, success_rate={initial_success_rate:.3f}, "
            f"sign_delay={initial_signature_delay:.1f}ms, hunger={initial_hunger:.3f}"
        )
        
        # Simulate transaction processing
        for step in range(steps):
            # Record pre-transaction state
            old_reputation = env.reputation[na_idx]
            old_success_rate = env.success_rate[na_idx]
            old_hunger = env.hunger[na_idx]
            
            env.current_time += 1
            
            
            # Simulate per-transaction signature delay grade (add small random noise)
            base_delay_grade = env.signature_delay[na_idx]
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
            current_delay_grade = max(0.0, min(1.0, current_delay_grade))  # Clamp into [0.0, 1.0]
            
            # Determine transaction success
            if current_delay_grade > env.max_acceptable_delay_grade:
                success = False
                failure_reason = "signature delay grade too high"
            else:
                # Use raw success rate (do not adjust by delay performance)
                success = np.random.random() < env.success_rate[na_idx]
                failure_reason = None
            
            # Compute reputation update parameters
            norm_rep = (old_reputation - 3300.0) / (10000.0 - 3300.0)
            if old_reputation < 6600:
                computed_factor = (0.4 * env.success_rate[na_idx] + 
                                 0.4 * norm_rep + 
                                 0.2 * env.hunger[na_idx])
            else:
                computed_factor = (0.4 * env.success_rate[na_idx] + 
                                 0.2 * norm_rep + 
                                 0.4 * env.hunger[na_idx])
            
            # Update NA state and reputation
            if success:
                env.total_tx[na_idx] += 1
                env.success_count[na_idx] += 1
                env.success_rate[na_idx] = env.success_count[na_idx] / env.total_tx[na_idx]
                reputation_change = computed_factor * 20
                env.reputation[na_idx] = min(10000, old_reputation + reputation_change)
            else:
                env.total_tx[na_idx] += 1
                env.success_rate[na_idx] = env.success_count[na_idx] / env.total_tx[na_idx]
                reputation_change = -computed_factor * 100
                env.reputation[na_idx] = max(3300, old_reputation + reputation_change)
            
            # Record transaction result
            transaction_result = {
                'step': step + 1,
                'success': success,
                'failure_reason': failure_reason,
                'signature_delay': current_delay_grade,
                'old_reputation': old_reputation,
                'new_reputation': env.reputation[na_idx],
                'reputation_change': reputation_change,
                'old_success_rate': old_success_rate,
                'new_success_rate': env.success_rate[na_idx],
                'hunger': env.hunger[na_idx]
            }
            transaction_results.append(transaction_result)
            reputation_history.append(env.reputation[na_idx])
        
        # Aggregate summary stats
        final_rep = reputation_history[-1]
        total_rep_change = final_rep - initial_rep
        successful_transactions = sum(1 for t in transaction_results if t['success'])
        transaction_success_rate = successful_transactions / steps
        avg_signature_delay = np.mean([t['signature_delay'] for t in transaction_results])
        final_success_rate = env.success_rate[na_idx]
        final_hunger = env.hunger[na_idx]
        
        # Failure breakdown
        delay_failures = sum(1 for t in transaction_results if t['failure_reason'] == 'signature delay grade too high')
        random_failures = steps - successful_transactions - delay_failures
        
        training_results[na_idx] = {
            'initial_reputation': initial_rep,
            'final_reputation': final_rep,
            'reputation_change': total_rep_change,
            'initial_success_rate': initial_success_rate,
            'final_success_rate': final_success_rate,
            'transaction_success_rate': transaction_success_rate,
            'successful_transactions': successful_transactions,
            'total_transactions': steps,
            'avg_signature_delay': avg_signature_delay,
            'delay_failures': delay_failures,
            'random_failures': random_failures,
            'initial_hunger': initial_hunger,
            'final_hunger': final_hunger,
            'reputation_history': reputation_history,
            'transaction_results': transaction_results
        }
        
        print(f"NA {na_idx} summary:")
        print(f"   Reputation: {initial_rep:.2f} -> {final_rep:.2f} ({total_rep_change:+.2f})")
        print(f"   Success rate: {initial_success_rate:.3f} -> {final_success_rate:.3f}")
        print(f"   Transaction success rate: {successful_transactions}/{steps} ({transaction_success_rate:.1%})")
        print(f"   Avg signature delay grade: {avg_signature_delay:.1%}")
        print(f"   Failures: delay-too-high {delay_failures}, random {random_failures}")
        print(f"   Hunger: {initial_hunger:.3f} -> {final_hunger:.3f}")
    
    # Restore original environment state
    env.reputation = original_reputation
    env.success_rate = original_success_rate
    
    env.current_time = original_current_time
    
    env.total_tx = original_total_tx
    env.success_count = original_success_count
    
    return training_results

def create_policy_analysis_plots(agent, env):
    """
    Create policy analysis plots:
    1. Policy distribution
    2. Policy entropy curve
    3. NA selection frequency heatmap
    """
    print("\n" + "="*80)
    print("Creating policy analysis plots")
    print("="*80)
    
    # Font settings (kept CJK-capable to avoid missing glyphs if needed)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create 3x1 subplots (larger canvas)
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # 1) Policy distribution - average action probabilities in recent episodes
    ax1 = axes[0]
    
    if len(agent.action_distribution_history) > 0:
        # Average last N distributions
        recent_distributions = agent.action_distribution_history[-10:]  # Last 10
        avg_distribution = np.mean(recent_distributions, axis=0)
        
        # NA indices
        na_indices = np.arange(len(avg_distribution))
        
        # Color by initial reputation group
        colors = []
        for i in range(env.n_na):
            if env.episode_start_reputation[i] < 6600:
                colors.append('lightcoral')  # Low reputation group
            else:
                colors.append('lightblue')   # High reputation group
        
        bars = ax1.bar(na_indices, avg_distribution, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Annotate top selection probabilities
        top_5_indices = np.argsort(avg_distribution)[-5:]
        for idx in top_5_indices:
            ax1.annotate(f'NA{idx}\n{avg_distribution[idx]:.3f}', 
                        xy=(idx, avg_distribution[idx]), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')
        
        ax1.set_title('Policy Distribution (Final Strategy)\nAverage Action Probabilities in Recent Episodes', 
                     fontsize=12, fontweight='bold', pad=20)  # Increase title padding
        ax1.set_xlabel('NA Index')
        ax1.set_ylabel('Selection Probability')
        
        # Show all NA indices
        ax1.set_xticks(na_indices)
        ax1.set_xticklabels([f'NA{i}' for i in na_indices], rotation=45, ha='right')
        
        ax1.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightcoral', label='Low Reputation Group (<6600)'),
                          Patch(facecolor='lightblue', label='High Reputation Group (≥6600)')]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        print("Policy distribution plot created (average of last 10 episodes).")
        print(f"   Top-5 NA selection probabilities: {[f'NA{i}({avg_distribution[i]:.3f})' for i in top_5_indices]}")
    else:
        ax1.text(0.5, 0.5, 'No Policy Distribution Data Available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Policy Distribution - No Data')
    
    # 2) Policy entropy
    ax2 = axes[1]
    
    if len(agent.policy_entropy_history) > 0:
        episodes = np.arange(len(agent.policy_entropy_history))
        entropies = agent.policy_entropy_history
        entropy_change = 0.0
        
        # Entropy curve
        ax2.plot(episodes, entropies, 'b-', linewidth=2, alpha=0.8, label='Policy Entropy')
        
        # Trend line
        if len(entropies) > 10:
            z = np.polyfit(episodes, entropies, 1)
            p = np.poly1d(z)
            ax2.plot(episodes, p(episodes), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
        
        # Key stats
        if len(entropies) > 0:
            initial_entropy = entropies[0]
            final_entropy = entropies[-1]
            max_entropy = np.max(entropies)
            min_entropy = np.min(entropies)
            
            ax2.axhline(y=np.log(env.n_na), color='g', linestyle=':', alpha=0.7, 
                       label=f'Max Possible Entropy (log({env.n_na})={np.log(env.n_na):.2f})')
            
            # Annotate key stats
            ax2.text(0.02, 0.65, f'Initial: {initial_entropy:.3f}\nFinal: {final_entropy:.3f}\n'
                                  f'Max: {max_entropy:.3f}\nMin: {min_entropy:.3f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_title('Policy Entropy Over Training\n(Lower entropy = more deterministic policy)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Step (every 10 episodes)')
        ax2.set_ylabel('Entropy (bits)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02))
        
        # Trend analysis
        if len(entropies) > 1:
            entropy_change = entropies[-1] - entropies[0]
            if entropy_change < -0.1:
                trend_text = "Policy becomes more deterministic (entropy decreases)"
            elif entropy_change > 0.1:
                trend_text = "Policy becomes more random (entropy increases)"
            else:
                trend_text = "Policy relatively stable"
        else:
            trend_text = "Insufficient data"
            
        print(f"Policy entropy plot created - {trend_text}")
        print(f"   Entropy change: {entropies[0]:.3f} -> {entropies[-1]:.3f} (Δ{entropy_change:.3f})")
    else:
        ax2.text(0.5, 0.5, 'No Policy Entropy Data Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Policy Entropy - No Data')
    
    # 3) NA selection frequency heatmap
    ax3 = axes[2]
    
    if np.sum(agent.na_selection_frequency) > 0:
        # Normalize selection frequency into percentages
        selection_freq_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        
        # Reshape into matrix for display
        n_cols = min(10, env.n_na)  # Up to 10 NAs per row
        n_rows = int(np.ceil(env.n_na / n_cols))
        
        # Fill matrix
        heatmap_data = np.zeros((n_rows, n_cols))
        for i in range(env.n_na):
            row = i // n_cols
            col = i % n_cols
            heatmap_data[row, col] = selection_freq_percent[i]
        
        # Heatmap
        im = ax3.imshow(heatmap_data, cmap='Reds', aspect='auto', interpolation='nearest')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Selection Frequency (%)', rotation=270, labelpad=20)
        
        # Annotate cells
        for i in range(n_rows):
            for j in range(n_cols):
                na_idx = i * n_cols + j
                if na_idx < env.n_na:
                    text_color = 'white' if heatmap_data[i, j] > np.max(heatmap_data) * 0.5 else 'black'
                    ax3.text(j, i, f'NA{na_idx}\n{heatmap_data[i, j]:.1f}%',
                            ha='center', va='center', color=text_color, fontsize=8)
        
        # Axes labels
        ax3.set_xticks(range(n_cols))
        ax3.set_yticks(range(n_rows))
        ax3.set_xticklabels([f'Col{i}' for i in range(n_cols)])
        ax3.set_yticklabels([f'Row{i}' for i in range(n_rows)])
        
        ax3.set_title('NA Selection Frequency Heatmap\n(Percentage of times each NA was selected)', 
                     fontsize=12, fontweight='bold')
        
        # Selection frequency analysis
        most_selected = np.argmax(selection_freq_percent)
        least_selected = np.argmin(selection_freq_percent[selection_freq_percent > 0])
        
        print("NA selection frequency heatmap created.")
        print(f"   Most selected: NA{most_selected} ({selection_freq_percent[most_selected]:.1f}%)")
        print(f"   Least selected: NA{least_selected} ({selection_freq_percent[least_selected]:.1f}%)")
        
        # Over-concentration check
        top_3_percent = np.sum(np.sort(selection_freq_percent)[-3:])
        if top_3_percent > 70:
            print(f"   Warning: Top-3 NAs account for {top_3_percent:.1f}% selections; potential overfitting.")
        else:
            print(f"   Selection spread looks reasonable: Top-3 account for {top_3_percent:.1f}%.")
    else:
        ax3.text(0.5, 0.5, 'No Selection Frequency Data Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('NA Selection Frequency - No Data')
    
    plt.tight_layout(pad=4.0)
    
    # Save plot
    filename = str(OUTPUT_ROOT / "src" / "policy_analysis.png")
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nPolicy analysis plot saved: {filename}")
    print("="*80)
    
    # Detailed report
    print("\nPolicy analysis report:")
    
    if len(agent.policy_entropy_history) > 0:
        print("1. Policy entropy:")
        print(f"   - Initial entropy: {agent.policy_entropy_history[0]:.3f}")
        print(f"   - Final entropy: {agent.policy_entropy_history[-1]:.3f}")
        print(f"   - Max possible entropy: {np.log(env.n_na):.3f} (fully random)")
        print(f"   - Determinism: {(1 - agent.policy_entropy_history[-1]/np.log(env.n_na))*100:.1f}%")
    
    if np.sum(agent.na_selection_frequency) > 0:
        print(f"\n2. NA selection:")
        selection_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        selected_nas = np.where(selection_percent > 0)[0]
        print(f"   - Selected NAs: {len(selected_nas)}/{env.n_na}")
        print(f"   - Selection diversity: {len(selected_nas)/env.n_na*100:.1f}%")
        
        # By reputation group
        low_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                   if env.episode_start_reputation[i] < 6600])
        high_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                    if env.episode_start_reputation[i] >= 6600])
        print(f"   - Low reputation group selection: {low_rep_selections:.1f}%")
        print(f"   - High reputation group selection: {high_rep_selections:.1f}%")
    
    if len(agent.action_distribution_history) > 0:
        print(f"\n3. Policy distribution:")
        recent_dist = np.mean(agent.action_distribution_history[-10:], axis=0)
        top_5_nas = np.argsort(recent_dist)[-5:]
        print(f"   - Top-5 preferred NAs: {[f'NA{i}({recent_dist[i]:.2f})' for i in top_5_nas]}")
        concentration = np.sum(np.sort(recent_dist)[-5:])
        print(f"   - Top-5 concentration: {concentration:.1%}")
        
        if concentration > 0.8:
            print(f"   - Warning: policy is highly concentrated; potential overfitting.")
        elif concentration < 0.3:
            print(f"   - Warning: policy is too diffuse; may be undertrained.")
        else:
            print(f"   - Policy concentration looks moderate.")

def train_dqn(n_na, n_episodes, steps_per_episode, lr, 
              update_frequency, use_fixed_nas, enable_random_fluctuation,
              generate_episode_gifs=False):  # Removed default args
    
    # Parameter checks
    print("Training parameter check:")
    print(f"  NAs: {n_na}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Learning rate: {lr}")
    print(f"  Update frequency: every {update_frequency} steps")
    print(f"  Use fixed NA set: {use_fixed_nas}")
    print(f"  Random fluctuation mode: {enable_random_fluctuation}")
    
    # Basic sanity checks
    total_steps = n_episodes * steps_per_episode
    print(f"  Total training steps: {total_steps}")
    
    if n_na < 5:
        print("Warning: too few NAs; this may affect learning.")
    if n_na > 30:
        print("Warning: too many NAs; you may need more training steps.")
    if lr > 0.01:
        print("Warning: learning rate is high; training may be unstable.")
    if total_steps < 1000:
        print("Warning: too few training steps; the network may be undertrained.")
    
    # Create environment and agent
    env = Environment(n_na, use_fixed_nas=use_fixed_nas, enable_random_fluctuation=enable_random_fluctuation)
    agent = DQNAgent(n_features=4, n_na=n_na, lr=lr, gamma=0.9,  # After removing activity, n_features=4
                    epsilon_start=1.0, epsilon_end=0.01, decay_steps=15000,
                    memory_size=50000, batch_size=256, target_update=20,
                    min_memory_size=2000, update_frequency=update_frequency, 
                    total_episodes=n_episodes)
    history = []
    # Per-step reputation changes for the last episode
    reputation_last = []
    
    # Detailed data for the first and last episodes
    first_episode_data = []  # Reputation and selections for first episode
    last_episode_data = []   # Reputation and selections for last episode
    
    # Reward analysis
    step_rewards = []  # Smoothed step rewards
    step_reward_steps = []  # Corresponding training steps
    reward_capture_interval = 3  # Aggregate every 3 steps
    step_reward_buffer = []  # Buffer raw rewards for smoothing
    success_rates_per_episode = []  # Per-episode success rates
    
    # Training stats
    total_training_steps = 0
    total_updates = 0
    
    # Warm-up: collect experience without training
    print("Warm-up: collecting initial experience...")
    warmup_steps = 0
    while len(agent.memory) < agent.min_memory_size:
        state = env.reset()  # Reset environment state while keeping fixed NA traits
        for _ in range(10):  # Collect 10 steps per environment state
            action = agent.select_action(state, env.episode_start_reputation)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            warmup_steps += 1
            
            # Stop warm-up once minimum memory is reached
            if len(agent.memory) >= agent.min_memory_size:
                break
    
    print(f"Warm-up done: collected {len(agent.memory)} experiences in {warmup_steps} steps.")
    print("Training strategy: use a fixed NA set; select a single NA each step.")
    print(f"Start training. Update frequency: train every {update_frequency} steps...")
    
    # Print the fixed NA characteristics for the first episode to confirm determinism
    print("\nFixed NA set characteristics (constant throughout training):")
    print("id\tinit_rep\tsuccess_rate\tsign_delay(ms)\thunger\tdelay_grade\tgroup")
    for i in range(n_na):
        rep = env.fixed_initial_reputation[i] if env.use_fixed_nas else env.reputation[i]
        sr = env.fixed_initial_success_rate[i] if env.use_fixed_nas else env.success_rate[i]
        delay = env.fixed_initial_signature_delay[i] if env.use_fixed_nas else env.signature_delay[i]
        hunger = env.fixed_initial_hunger[i] if env.use_fixed_nas else env.hunger[i]
        delay_perf = env.calculate_delay_performance(delay)
        group = "low" if rep < 6600 else "high"
        print(f"{i}\t{rep:.1f}\t\t{sr:.3f}\t{delay:.1f}\t\t{hunger:.1%}\t\t{delay_perf:.1%}\t\t{group}")
    
    for ep in range(n_episodes):
        # Update epsilon at the start of each episode (episode-based decay)
        agent.update_epsilon_by_episode(ep)
        
        # Update learning rate at the start of each episode (stage-wise decay)
        current_lr = agent.update_learning_rate(ep)
        
        # Record learning rate / epsilon history
        if ep % 10 == 0:  # Record every 10 episodes
            agent.lr_history.append(current_lr)
            agent.epsilon_history.append(agent.epsilon)
        
        # Log learning rate every 100 episodes
        if ep % 100 == 0 and ep > 0:
            print(f"Episode {ep}: Learning Rate = {current_lr:.6f}")
        
        state = env.reset()  # Reset environment state while keeping fixed NA traits unchanged
        
        # For the first and last episode, print NA parameters for verification
        if ep == 0:
            print(f"\nFirst episode (#{ep}) fixed NA parameters:")
            print("id\treputation\tsuccess_rate\tsign_delay(ms)\tdelay_grade\tgroup")
            for i in range(min(n_na, 10)):  # Print only the first 10 to avoid long output
                group = "low" if env.reputation[i] < 6600 else "high"
                delay_perf = env.calculate_delay_performance(env.signature_delay[i])
                print(f"{i}\t{env.reputation[i]:.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.3f}\t\t{group}")
            if n_na > 10:
                print(f"... (total {n_na} NAs)")
        elif ep == n_episodes - 1:
            print(f"\nLast episode (#{ep}) NA parameters (should match initial):")
            print("id\treputation\tsuccess_rate\tsign_delay(ms)\tdelay_grade\tgroup")
            for i in range(min(n_na, 10)):  # Print only the first 10 to avoid long output
                group = "low" if env.reputation[i] < 6600 else "high"
                delay_perf = env.calculate_delay_performance(env.signature_delay[i])
                print(f"{i}\t{env.reputation[i]:.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{group}")
            if n_na > 10:
                print(f"... (total {n_na} NAs)")
        
        # Initialize per-episode accumulators
        ep_reward = 0
        ep_step_rewards = []  # Per-step rewards in this episode
        ep_successes = 0  # Count of positive-reward steps
        
        # Record episode data (first and last episodes only)
        is_first_episode = (ep == 0)
        is_last_episode = (ep == n_episodes - 1)
        
        if is_first_episode or is_last_episode:
            episode_reputation_history = [env.reputation.copy()]  # Reputation after each step
            episode_selected_nas = []  # Selected NA per step
        
        # For the last episode, record initial reputation
        if is_last_episode:
            reputation_last.append(env.reputation.copy())
        
        for t in range(steps_per_episode):
            record_policy = (ep % 10 == 0) or is_last_episode  # Record every 10 episodes + last episode
            action_na = agent.select_action(state, initial_reputations=env.episode_start_reputation, record_policy=record_policy)
            next_state, reward, done = env.step(action_na)
            agent.store(state, action_na, reward, next_state, done)
            
            total_training_steps += 1
            
            # Conditional updates rather than training every step
            batch_losses = agent.maybe_update(global_step=total_training_steps)
            if batch_losses is not None:
                total_updates += len(batch_losses)
                # Only print detailed training info for the last episode
                if ep == n_episodes - 1:
                    avg_loss = np.mean(batch_losses)
                    # Skip detailed loss prints to reduce log noise
                
            # Accumulate per-episode reward
            ep_reward += reward
            ep_step_rewards.append(reward)

            # Aggregate rewards across a few steps to reduce curve noise
            step_reward_buffer.append(reward)
            if len(step_reward_buffer) >= reward_capture_interval:
                averaged_reward = sum(step_reward_buffer) / len(step_reward_buffer)
                step_rewards.append(averaged_reward)
                step_reward_steps.append(total_training_steps)
                step_reward_buffer.clear()
            
            # Count positive-reward steps as "successes"
            if reward > 0:
                ep_successes += 1
            
            state = next_state
            
            # Record detailed data for first/last episodes
            if is_first_episode or is_last_episode:
                episode_reputation_history.append(env.reputation.copy())
                episode_selected_nas.append(action_na)
        if is_last_episode:
            reputation_last.append(env.reputation.copy())
        
        # Calculate episode statistics
        ep_success_rate = ep_successes / steps_per_episode if steps_per_episode > 0 else 0
        ep_avg_step_reward = np.mean(ep_step_rewards) if ep_step_rewards else 0
        ep_reward_std = np.std(ep_step_rewards) if ep_step_rewards else 0
        
        # Calculate Q-value statistics (every 10 episodes to reduce overhead)
        if ep % 10 == 0:  # Only compute Q-value stats every 10 episodes
            with torch.no_grad():
                current_state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                all_q_values = agent.policy_net(current_state).squeeze(0).cpu().numpy()
                q_mean = np.mean(all_q_values)
                q_max = np.max(all_q_values)
                q_min = np.min(all_q_values)
                q_std = np.std(all_q_values)
        else:
            # Reuse the previous Q-value statistics (reduce overhead)
            if ep > 0 and len(history) > 0:
                last_q_data = history[-1]
                q_mean = last_q_data.get('q_mean', 0)
                q_max = last_q_data.get('q_max', 0)
                q_min = last_q_data.get('q_min', 0)
                q_std = last_q_data.get('q_std', 0)
            else:
                q_mean = q_max = q_min = q_std = 0
        
        # Store data for analysis
        success_rates_per_episode.append(ep_success_rate)
        
        # Save data from the first and last episodes
        if is_first_episode:
            first_episode_data = {
                'reputation_history': episode_reputation_history,
                'selected_nas': episode_selected_nas,
                'episode_num': ep + 1
            }
            # Export all NA parameters at the end of the first episode
            try:
                print("\nEnd of first episode: exporting all NA parameters...")
                export_file = agent.export_na_selection_parameters(
                    state=state, 
                    initial_reputations=env.episode_start_reputation,
                    env=env
                )
                print(f"First-episode NA parameter export succeeded. File: {export_file}")
            except Exception as e:
                print(f"First-episode NA parameter export failed: {e}")
        elif is_last_episode:
            last_episode_data = {
                'reputation_history': episode_reputation_history,
                'selected_nas': episode_selected_nas,
                'episode_num': ep + 1
            }
        
        history.append({
            'episode': ep,
            'total_steps': total_training_steps,
            'reward': ep_reward,
            'mean_reputation': env.reputation.mean(),
            'success_rate': ep_success_rate,
            'avg_step_reward': ep_avg_step_reward,
            'reward_std': ep_reward_std,
            'q_mean': q_mean,
            'q_max': q_max,
            'q_min': q_min,
            'q_std': q_std
        })
        if (ep+1) % 20 == 0:
            memory_status = f"memory: {len(agent.memory)}/{agent.min_memory_size}"
            update_ratio = total_updates / max(1, total_training_steps) * 100
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Mean Reputation: {env.reputation.mean():.2f}, "
                  f"Success Rate: {ep_success_rate:.3f}, Avg Step Reward: {ep_avg_step_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}, {memory_status}, update rate: {update_ratio:.1f}%")
        
        # Run Q-value learning quality checks every 50 episodes
        if (ep + 1) % 50 == 0 and ep > 0:
            print(f"\nEpisode {ep+1} - Q-value learning quality check:")
            q_values = agent.debug_q_values(state, env.episode_start_reputation)
            
            # Check whether the network learned a reasonable NA selection policy
            low_rep_indices = np.where(env.episode_start_reputation < 6600)[0]
            high_rep_indices = np.where(env.episode_start_reputation >= 6600)[0]
            
            if len(low_rep_indices) > 0 and len(high_rep_indices) > 0:
                # Count how many of the top-5 Q-value NAs fall into each group
                top_5_indices = np.argsort(q_values.cpu().numpy())[-5:]
                top_5_low_count = np.sum([idx in low_rep_indices for idx in top_5_indices])
                top_5_high_count = np.sum([idx in high_rep_indices for idx in top_5_indices])
                
                print(
                    f"  Top-5 Q-value NAs: low-reputation group {top_5_low_count}, "
                    f"high-reputation group {top_5_high_count}"
                )
                
                # Evaluate learning quality
                if top_5_high_count >= 3:
                    print("  OK: learning quality looks good (prefers high-reputation NAs)")
                elif top_5_high_count >= 2:
                    print("  WARN: learning quality is moderate (policy may need improvement)")
                else:
                    print("  BAD: learning quality is poor (hyperparameters may need tuning)")
            
            # Every 100 episodes, print sliding-window summary
            if (ep + 1) % 100 == 0:
                env.print_sliding_window_summary(max_nas_to_show=10)
        
        # Print details at key epsilon transition points
        first_transition = int(0.15 * n_episodes)  # First transition point: 15%
        second_transition = int(0.5 * n_episodes)  # Second transition point: 50%
        
        if ep == first_transition or ep == first_transition + 1:
            print(
                f"Episode {ep+1}: Epsilon first transition, ε = {agent.epsilon:.4f} "
                f"({('very-fast decay -> fast decay' if ep == first_transition else 'fast decay phase')})"
            )
        elif ep == second_transition or ep == second_transition + 1:
            print(
                f"Episode {ep+1}: Epsilon second transition, ε = {agent.epsilon:.4f} "
                f"({('fast decay -> fine-tuning' if ep == second_transition else 'fine-tuning phase')})"
            )
        elif ep in [n_episodes-10, n_episodes-5, n_episodes-1]:
            print(f"Episode {ep+1}: approaching end of training, ε = {agent.epsilon:.4f}")
    
    # Handle remaining rewards not filling the aggregation window
    if step_reward_buffer:
        averaged_reward = sum(step_reward_buffer) / len(step_reward_buffer)
        step_rewards.append(averaged_reward)
        step_reward_steps.append(total_training_steps)

    # Ensure the last episode LR/epsilon are logged (if the final episode is not a multiple of 10)
    final_episode = n_episodes - 1  # Index of the final episode
    if final_episode % 10 != 0:  # If the final episode is not a multiple of 10
        current_lr = agent.update_learning_rate(final_episode)
        agent.lr_history.append(current_lr)
        agent.epsilon_history.append(agent.epsilon)
        print(
            f"Log final Episode {final_episode}: Learning Rate = {current_lr:.6f}, "
            f"Epsilon = {agent.epsilon:.6f}"
        )
    
    # Print training statistics
    print("\nTraining summary:")
    print(f"Total training steps: {total_training_steps}")
    print(f"Total network updates: {total_updates}")
    print(f"Update frequency: try training every {update_frequency} steps")
    print(f"Actual update rate: {total_updates/max(1, total_training_steps)*100:.2f}%")
    print(f"Target-network update count: {agent.learn_step//agent.target_update}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Key logic checkpoints
    print("\nPost-training key logic checks:")
    
    # 1. Check whether network parameters changed
    final_state = env.get_state()
    with torch.no_grad():
        initial_q = torch.zeros(n_na)  # Assumed initial Q values
        final_q = agent.policy_net(torch.FloatTensor(final_state).unsqueeze(0).to(agent.device)).squeeze().cpu()
    
    q_magnitude = torch.norm(final_q).item()
    print(f"  Q-network weight magnitude: {q_magnitude:.4f}")
    
    if q_magnitude < 0.1:
        print("  BAD: network may be under-trained")
    elif q_magnitude > 100:
        print("  WARN: network may be overfitting or learning rate is too high")
    else:
        print("  OK: network weight magnitude looks normal")
    
    # 2. Check replay buffer utilization
    print(
        f"  Replay buffer utilization: {len(agent.memory)}/{agent.min_memory_size} "
          f"({len(agent.memory)/agent.min_memory_size*100:.1f}%)")
    
    # 3. Check whether the training frequency is reasonable
    expected_updates = total_training_steps // update_frequency
    update_efficiency = total_updates / max(1, expected_updates) * 100
    print(
        f"  Training efficiency: {update_efficiency:.1f}% "
        f"(expected {expected_updates} updates, actual {total_updates})"
    )
    
    # 4. Check loss convergence
    if len(agent.loss_history) > 100:
        early_loss = np.mean(agent.loss_history[:50])
        late_loss = np.mean(agent.loss_history[-50:])
        loss_improvement = (early_loss - late_loss) / early_loss * 100
        print(
            f"  Loss improvement: {loss_improvement:.1f}% "
            f"(early: {early_loss:.4f} -> late: {late_loss:.4f})"
        )
        
        if loss_improvement > 20:
            print("  OK: loss converged well")
        elif loss_improvement > 5:
            print("  WARN: limited loss improvement; may need more training")
        else:
            print("  BAD: no clear loss improvement; check hyperparameters")
    else:
        print("  WARN: insufficient training data to assess loss convergence")
    
    # Use the trained DQN for the final selection
    print("\n" + "="*80)
    print("Use the trained DQN network for the final NA selection")
    print("Training setup: fixed NA pool; select one NA per step; final set is built by repeated selections")
    print("="*80)
    
    final_state = env.get_state()
    selection_result = agent.make_final_selection(final_state, env.episode_start_reputation)
    
    dqn_selected = selection_result['dqn_selected']
    random_selected = selection_result['random_selected']
    all_q_values = selection_result['all_q_values']
    low_rep_indices = selection_result['low_rep_indices']
    high_rep_indices = selection_result['high_rep_indices']
    
    print("\nNA status at the end of the current episode:")
    print("ID\tInitRep\t\tCurRep\t\tDeltaRep\tsuccess_rate\tSigDelay(ms)\tDelayLevel\tQ\t\tGroup")
    for i in range(min(n_na, 15)):  # Limit output to avoid overly long logs
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "LowInitRep" if initial_rep < 6600 else "HighInitRep"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        print(
            f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t"
            f"{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t"
            f"{all_q_values[i]:.4f}\t\t{group}"
        )
    if n_na > 15:
        print(f"... (total {n_na} NAs, showing first 15 only)")
    
    print(f"\n[DQN] DQN-constructed set of {len(dqn_selected)} NAs (target: 2 low + 3 high):")
    print("ID\tInitRep\t\tCurRep\t\tDeltaRep\tsuccess_rate\tSigDelay(ms)\tDelayLevel\tQ\t\tGroup\t\tMalicious")
    dqn_low_count = 0
    dqn_high_count = 0
    dqn_total_reputation = 0
    dqn_total_q_value = 0
    dqn_malicious_count = 0  # Count malicious NAs selected
    
    for i in dqn_selected:
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "LowInitRep" if initial_rep < 6600 else "HighInitRep"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        
        # Check whether the NA is malicious and get its type
        is_malicious = env.fixed_malicious_flags[i]
        if is_malicious:
            dqn_malicious_count += 1
            mal_type = env.fixed_malicious_types[i]
            if mal_type == 1:
                malicious_info = "MALICIOUS T1 (high success + high delay)"
            elif mal_type == 4:
                malicious_info = "MALICIOUS T4 (classic low success)"
            else:
                malicious_info = "MALICIOUS (unknown type)"
        else:
            malicious_info = "NORMAL"
            
        if initial_rep < 6600:
            dqn_low_count += 1
        else:
            dqn_high_count += 1
        dqn_total_reputation += current_rep
        dqn_total_q_value += all_q_values[i]
        print(
            f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t"
            f"{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}\t{malicious_info}"
        )
    
    # Print malicious NA statistics for the DQN selection
    if dqn_malicious_count > 0:
        print(
            f"WARNING: DQN selection contains {dqn_malicious_count} malicious NAs. "
            "This indicates the malicious NAs may be successfully camouflaged or the DQN is not identifying them well."
        )
    
    print(f"\n[Random] Randomly selected set of {len(random_selected)} NAs (for comparison):")
    print("ID\tInitRep\t\tCurRep\t\tDeltaRep\tsuccess_rate\tSigDelay(ms)\tDelayLevel\tQ\t\tGroup\t\tMalicious")
    random_low_count = 0
    random_high_count = 0
    random_total_reputation = 0
    random_total_q_value = 0
    random_malicious_count = 0  # Count malicious NAs selected at random
    
    for i in random_selected:
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "LowInitRep" if initial_rep < 6600 else "HighInitRep"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        
        # Check whether the NA is malicious and get its type
        is_malicious = env.fixed_malicious_flags[i]
        if is_malicious:
            random_malicious_count += 1
            mal_type = env.fixed_malicious_types[i]
            if mal_type == 1:
                malicious_info = "MALICIOUS T1 (high success + high delay)"
            elif mal_type == 4:
                malicious_info = "MALICIOUS T4 (classic low success)"
            else:
                malicious_info = "MALICIOUS (unknown type)"
        else:
            malicious_info = "NORMAL"
            
        if initial_rep < 6600:
            random_low_count += 1
        else:
            random_high_count += 1
        random_total_reputation += current_rep
        random_total_q_value += all_q_values[i]
        print(
            f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t"
            f"{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}\t{malicious_info}"
        )
    
    # Print malicious NA statistics for the random selection
    if random_malicious_count > 0:
        print(
            f"NOTE: random selection contains {random_malicious_count} malicious NAs "
            "(expected for a random baseline)."
        )
    
    # Simulate full training behavior for the selected NA sets
    dqn_training_results = simulate_selected_nas_training(env, dqn_selected, "DQN-selected NA training simulation", steps=100)
    
    # Print sliding-window summary
    env.print_sliding_window_summary(max_nas_to_show=15)
    random_training_results = simulate_selected_nas_training(env, random_selected, "Random-selected NA training simulation", steps=100)
    
    print("\nSelection policy comparison (full training simulation):")
    print(f"{'Metric':<25} {'DQN':<15} {'Random':<15} {'DQN Edge':<15}")
    print("-" * 70)
    print(f"{'Low-rep group count':<25} {dqn_low_count:<15} {random_low_count:<15} {('meets target' if dqn_low_count >= 2 else 'below target'):<15}")
    print(f"{'High-rep group count':<25} {dqn_high_count:<15} {random_high_count:<15} {('meets target' if dqn_high_count >= 3 else 'below target'):<15}")
    print(f"{'Malicious NA count':<25} {dqn_malicious_count:<15} {random_malicious_count:<15} {('fewer' if dqn_malicious_count < random_malicious_count else ('same' if dqn_malicious_count == random_malicious_count else 'more')):<15}")
    
    # Compute post-training statistics
    dqn_avg_initial_rep = sum(dqn_training_results[i]['initial_reputation'] for i in dqn_selected) / len(dqn_selected)
    dqn_avg_final_rep = sum(dqn_training_results[i]['final_reputation'] for i in dqn_selected) / len(dqn_selected)
    dqn_avg_rep_change = sum(dqn_training_results[i]['reputation_change'] for i in dqn_selected) / len(dqn_selected)
    dqn_avg_transaction_success_rate = sum(dqn_training_results[i]['transaction_success_rate'] for i in dqn_selected) / len(dqn_selected)
    dqn_total_successful_transactions = sum(dqn_training_results[i]['successful_transactions'] for i in dqn_selected)
    dqn_avg_signature_delay = sum(dqn_training_results[i]['avg_signature_delay'] for i in dqn_selected) / len(dqn_selected)
    
    random_avg_initial_rep = sum(random_training_results[i]['initial_reputation'] for i in random_selected) / len(random_selected)
    random_avg_final_rep = sum(random_training_results[i]['final_reputation'] for i in random_selected) / len(random_selected)
    random_avg_rep_change = sum(random_training_results[i]['reputation_change'] for i in random_selected) / len(random_selected)
    random_avg_transaction_success_rate = sum(random_training_results[i]['transaction_success_rate'] for i in random_selected) / len(random_selected)
    random_total_successful_transactions = sum(random_training_results[i]['successful_transactions'] for i in random_selected)
    random_avg_signature_delay = sum(random_training_results[i]['avg_signature_delay'] for i in random_selected) / len(random_selected)
    
    print(f"{'Avg reputation (pre)':<25} {dqn_avg_initial_rep:<15.2f} {random_avg_initial_rep:<15.2f} {dqn_avg_initial_rep - random_avg_initial_rep:+.2f}")
    print(f"{'Avg reputation (post)':<25} {dqn_avg_final_rep:<15.2f} {random_avg_final_rep:<15.2f} {dqn_avg_final_rep - random_avg_final_rep:+.2f}")
    print(f"{'Avg reputation gain':<25} {dqn_avg_rep_change:<15.2f} {random_avg_rep_change:<15.2f} {dqn_avg_rep_change - random_avg_rep_change:+.2f}")
    print(f"{'Tx success rate':<25} {dqn_avg_transaction_success_rate:<15.1%} {random_avg_transaction_success_rate:<15.1%} {'+' if dqn_avg_transaction_success_rate > random_avg_transaction_success_rate else ''}{(dqn_avg_transaction_success_rate - random_avg_transaction_success_rate):.1%}")
    print(f"{'Total successful txs':<25} {dqn_total_successful_transactions:<15} {random_total_successful_transactions:<15} {dqn_total_successful_transactions - random_total_successful_transactions:+}")
    print(f"{'Avg signature delay(ms)':<25} {dqn_avg_signature_delay:<15.1f} {random_avg_signature_delay:<15.1f} {dqn_avg_signature_delay - random_avg_signature_delay:+.1f}")
    
    # Create animated comparison of selection strategies
    print("\nCreating selection strategy comparison GIF...")
    create_comparison_strategy_gif(env, dqn_selected, random_selected, dqn_training_results, random_training_results)
    
    dqn_avg_q_value = dqn_total_q_value / len(dqn_selected) if dqn_selected else 0
    random_avg_q_value = random_total_q_value / len(random_selected) if random_selected else 0
    print(f"{'Avg Q value':<25} {dqn_avg_q_value:<15.4f} {random_avg_q_value:<15.4f} {dqn_avg_q_value - random_avg_q_value:+.4f}")
    
    print("\nQ-value analysis (fixed NA pool):")
    if len(low_rep_indices) > 0:
        low_q_values = all_q_values[low_rep_indices]
        print(f"Low-rep group Q range: {low_q_values.min():.4f} ~ {low_q_values.max():.4f} (mean: {low_q_values.mean():.4f})")
    if len(high_rep_indices) > 0:
        high_q_values = all_q_values[high_rep_indices]
        print(f"High-rep group Q range: {high_q_values.min():.4f} ~ {high_q_values.max():.4f} (mean: {high_q_values.mean():.4f})")
    
    # Signature delay analysis
    print("\nSignature delay analysis:")
    all_delays = [env.signature_delay[i] for i in range(n_na)]
    all_delay_perfs = [env.calculate_delay_performance(delay) for delay in all_delays]
    
    print(f"Signature delay range: {min(all_delays):.1f}ms ~ {max(all_delays):.1f}ms (mean: {np.mean(all_delays):.1f}ms)")
    print(f"Delay level: {min(all_delay_perfs):.3f} ~ {max(all_delay_perfs):.3f} (mean: {np.mean(all_delay_perfs):.3f})")
    
    # Group-wise delay analysis
    if len(low_rep_indices) > 0:
        low_delays = [all_delays[i] for i in low_rep_indices]
        low_delay_perfs = [all_delay_perfs[i] for i in low_rep_indices]
        print(f"Low-rep group delay: {np.mean(low_delays):.1f}ms (level: {np.mean(low_delay_perfs):.3f})")
    if len(high_rep_indices) > 0:
        high_delays = [all_delays[i] for i in high_rep_indices]
        high_delay_perfs = [all_delay_perfs[i] for i in high_rep_indices]
        print(f"High-rep group delay: {np.mean(high_delays):.1f}ms (level: {np.mean(high_delay_perfs):.3f})")
    
    # Delay-level interpretation
    print("Delay level is normalized to [0.0, 1.0] (0.0 = fastest, 1.0 = slowest).")
    
    # Correlation analysis between Q-values and NA composite quality
    print("\nQ-value learning effectiveness analysis:")
    print("ID\tInitRep\tSuccess\tQ\t\tSigDelay(ms)\tHunger\tDelayLevel\tScore\t\tAssessment")
    
    # Compute per-NA composite quality score in [0, 1]
    na_scores = []
    for i in range(n_na):
        initial_rep = env.episode_start_reputation[i]
        success_rate = env.success_rate[i]
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        # Score = 40% success + 30% delay + 20% reputation + 10% hunger
        rep_score = (initial_rep - 3300) / (10000 - 3300)  # Normalize reputation to [0, 1]
        composite_score = (0.4 * success_rate + 
                           0.3 * delay_perf + 
                           0.2 * rep_score + 
                           0.1 * env.hunger[i])
        na_scores.append(composite_score)
    
    na_scores = np.array(na_scores)
    q_values_array = np.array(all_q_values)
    
    # Correlation between Q-values and composite scores
    correlation = np.corrcoef(na_scores, q_values_array)[0, 1]
    
    for i in range(n_na):
        initial_rep = env.episode_start_reputation[i]
        success_rate = env.success_rate[i]
        q_val = all_q_values[i]
        delay = env.signature_delay[i]
        delay_perf = env.calculate_delay_performance(delay)
        composite_score = na_scores[i]
        
        # Assessment heuristic: Q-values should align with composite scores
        q_percentile = (q_val - q_values_array.min()) / (q_values_array.max() - q_values_array.min())
        score_percentile = (composite_score - na_scores.min()) / (na_scores.max() - na_scores.min())
        
        # Compare percentile ranks (allow a tolerance of ±0.3)
        rank_diff = abs(q_percentile - score_percentile)
        if rank_diff <= 0.3:
            reasonable = "OK"
        elif rank_diff <= 0.5:
            reasonable = "MID"
        else:
            reasonable = "BAD"
        
        print(
            f"{i}\t{initial_rep:.1f}\t\t{success_rate:.3f}\t{q_val:.4f}\t\t{delay:.1f}\t\t"
            f"{env.hunger[i]:.1%}\t\t{delay_perf:.1%}\t\t{composite_score:.3f}\t\t{reasonable}"
        )
    
    print("\nQ-value learning effectiveness summary:")
    print(f"Correlation (Q vs composite score): {correlation:.3f}")
    if correlation > 0.7:
        print("OK: strong alignment (Q-values reflect composite quality well).")
    elif correlation > 0.5:
        print("OK: good alignment (Q-values largely reflect composite quality).")
    elif correlation > 0.3:
        print("MID: moderate alignment (may need more training).")
    else:
        print("BAD: weak alignment (consider checking training settings).")
    
    # Use the DQN result as the final selection
    selected_idx = dqn_selected

    # Plot reputation changes for the last episode
    plt.figure(figsize=(12, 8))
    for i in range(n_na):
        # Use different plot styles based on whether the NA is selected
        if i in selected_idx:
            # Selected NAs: thick line and markers
            plt.plot(range(len(reputation_last)), [r[i] for r in reputation_last], 
                    linewidth=3, marker='o', markersize=4, markevery=10,
                    label=f'[DQN] NA {i} (DQN Selected)')
        else:
            # Unselected NAs: thin lines
            plt.plot(range(len(reputation_last)), [r[i] for r in reputation_last], 
                    linewidth=1, alpha=0.7, label=f'NA {i}')
    
    # Mark selected NAs at the final step
    for i in selected_idx:
        final_reputation = env.reputation[i]
        plt.scatter(len(reputation_last)-1, final_reputation, 
                   s=200, marker='*', color='gold', edgecolor='red', linewidth=2,
                   zorder=10)
        # Annotate Q-value
        q_value = all_q_values[i]
        plt.annotate(f'DQN Selected NA{i}\nQ-value: {q_value:.4f}', 
                    xy=(len(reputation_last)-1, final_reputation),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=8, ha='left')
    
    # Add the 6600 threshold line
    plt.axhline(y=6600, color='red', linestyle='--', linewidth=2, label='Reputation Threshold (6600)')
    
    plt.xlabel('Step')
    plt.ylabel('Reputation')
    plt.title('Last Episode Reputation Trajectory - Fixed NA Set Training Results\n([DQN] = Final Multi-Selection Combination)')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure locally
    last_episode_path = str(OUTPUT_ROOT / "src" / "last_episode_reputation_trajectory.png")
    Path(last_episode_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(last_episode_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Last episode reputation trajectory saved as: {last_episode_path}")
    
    # Optional: generate NA reputation GIFs for the first and last episodes
    if generate_episode_gifs:
        print("\nGenerating NA reputation dynamic GIF animations for first and last episodes...")
        if first_episode_data:
            create_episode_reputation_animation(first_episode_data, n_na, "First Episode")
        if last_episode_data:
            create_episode_reputation_animation(last_episode_data, n_na, "Last Episode")
    else:
        print("\nSkipping episode reputation GIF generation (disabled by configuration).")
    
    # Create policy analysis plots
    create_policy_analysis_plots(agent, env)
    
    # Persist step reward history for plotting
    agent.step_reward_history = step_rewards
    agent.step_reward_steps = step_reward_steps
    
    return pd.DataFrame(history), agent.loss_history, agent

def create_episode_reputation_animation(episode_data, n_na, title_prefix):
    """
    Create a GIF animation of NA reputation changes within a single episode.
    
    Args:
        episode_data: Dict containing reputation history and selection history.
        n_na: Number of NAs.
        title_prefix: Title prefix for the plot.
    """
    reputation_history = episode_data['reputation_history']
    selected_nas = episode_data['selected_nas']
    episode_num = episode_data['episode_num']
    
    print(f"Creating GIF animation for {title_prefix} Episode {episode_num}...")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Assign a distinct color per NA
    colors = plt.cm.tab20(np.linspace(0, 1, n_na))
    
    # Compute data range
    steps = range(len(reputation_history))
    all_reputations = [rep for step_rep in reputation_history for rep in step_rep]
    min_reputation = min(all_reputations) - 100
    max_reputation = max(all_reputations) + 200
    
    # Configure axis ranges
    ax.set_xlim(0, len(steps) - 1)
    ax.set_ylim(min_reputation, max_reputation)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Reputation', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add the 6600 threshold line
    ax.axhline(y=6600, color='red', linestyle='--', linewidth=2, 
               label='Reputation Threshold (6600)', alpha=0.8)
    
    # Initialize lines and points
    lines = []
    points = []
    selected_markers = []  # Special markers for selected NAs
    
    for i in range(n_na):
        # Base line (normal thickness)
        line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.7, label=f'NA {i}')
        lines.append(line)
        
        # Current point
        point, = ax.plot([], [], 'o', color=colors[i], markersize=8, alpha=0.9)
        points.append(point)
        
        # Selection marker (red star)
        marker, = ax.plot([], [], '*', color='red', markersize=20, alpha=1.0)
        selected_markers.append(marker)
    
    # Overlay text for current step and selection
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, verticalalignment='top', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Legend
    if n_na <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    def animate(frame):
        """Animation update function."""
        current_step = frame
        
        # Update title
        selected_na = selected_nas[current_step] if current_step < len(selected_nas) else -1
        ax.set_title(f'{title_prefix} (Episode {episode_num}) - Step {current_step + 1}\n' + 
                    f'Selected NA: {selected_na}' if selected_na != -1 else f'{title_prefix} (Episode {episode_num}) - Initial State', 
                    fontsize=14, weight='bold')
        
        # Update lines and points for each NA
        for i in range(n_na):
            # Collect all points up to the current step
            if current_step < len(reputation_history):
                x_data = list(range(current_step + 1))
                y_data = [reputation_history[j][i] for j in range(current_step + 1)]
                
                # Whether the current NA is selected at this step
                is_selected = (current_step < len(selected_nas) and selected_nas[current_step] == i)
                
                # Style line based on selection
                if is_selected:
                    # Selected NA: thicker line
                    lines[i].set_linewidth(6)
                    lines[i].set_alpha(1.0)
                else:
                    # Unselected NA: normal thickness
                    lines[i].set_linewidth(2)
                    lines[i].set_alpha(0.7)
                
                # Update line data
                lines[i].set_data(x_data, y_data)
                
                # Update current point
                if len(x_data) > 0:
                    points[i].set_data([x_data[-1]], [y_data[-1]])
                
                # Update selection marker
                if is_selected:
                    selected_markers[i].set_data([current_step], [reputation_history[current_step][i]])
                else:
                    selected_markers[i].set_data([], [])
            else:
                # Clear data
                lines[i].set_data([], [])
                points[i].set_data([], [])
                selected_markers[i].set_data([], [])
        
        # Update overlay text
        if current_step < len(selected_nas):
            selected_na = selected_nas[current_step]
            step_text.set_text(f'Step: {current_step + 1}\nSelected NA: {selected_na}\n' + 
                              f'NA{selected_na} Current Reputation: {reputation_history[current_step + 1][selected_na]:.1f}' 
                              if current_step + 1 < len(reputation_history) else f'Step: {current_step + 1}\nSelected NA: {selected_na}')
        else:
            step_text.set_text(f'Step: {current_step + 1}\nInitial State')
        
        return lines + points + selected_markers + [step_text]
    
    # Create animation (includes initial state + each step); faster playback
    total_frames = len(reputation_history)
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=400, blit=True, repeat=True)  # 400ms interval
    
    # Save animation
    filename = f'{title_prefix.replace(" ", "_").lower()}_episode_{episode_num}_reputation_animation.gif'
    
    try:
        # Ensure the output directory exists
        animations_dir = OUTPUT_ROOT / "episode_animations"
        animations_dir.mkdir(parents=True, exist_ok=True)
        full_path = str(animations_dir / filename)
        
        anim.save(full_path, writer='pillow', fps=2.5)
        print(f"{title_prefix} GIF animation saved as: {full_path}")
    except Exception as e:
        print(f"Error saving {title_prefix} animation: {e}")
    
    # Free memory
    plt.close(fig)

def create_comparison_strategy_gif(env, dqn_selected, random_selected, dqn_training_results, random_training_results, steps=100):
    """
    Create an animated comparison of selection strategies, showing reputation trajectories
    for DQN-selected and randomly selected NAs during simulated training.
    
    Args:
        env: Environment instance.
        dqn_selected: List of NA indices selected by DQN.
        random_selected: List of NA indices selected randomly.
        dqn_training_results: Training simulation results for DQN-selected NAs.
        random_training_results: Training simulation results for randomly selected NAs.
        steps: Number of simulation steps.
    """
    print("Creating selection strategy comparison GIF...")
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get reputation history data for all selected NAs
    all_selected_nas = list(set(dqn_selected + random_selected))
    
    # Assign distinct colors for each NA
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 base colors
    additional_colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20 additional colors
    all_colors = np.vstack([colors, additional_colors])  # 30 colors total
    
    # Assign colors to DQN-selected NAs
    dqn_colors = {}
    for i, na_idx in enumerate(dqn_selected):
        dqn_colors[na_idx] = all_colors[i % len(all_colors)]
    
    # Assign colors to random-selected NAs (offset to avoid clashing with DQN colors)
    random_colors = {}
    for i, na_idx in enumerate(random_selected):
        color_idx = (i + len(dqn_selected)) % len(all_colors)
        random_colors[na_idx] = all_colors[color_idx]
    
    # Compute data range
    all_reputations = []
    max_steps = 0
    
    # Collect reputation histories for DQN-selected NAs
    for na_idx in dqn_selected:
        if na_idx in dqn_training_results:
            rep_history = dqn_training_results[na_idx]['reputation_history']
            all_reputations.extend(rep_history)
            max_steps = max(max_steps, len(rep_history))
    
    # Collect reputation histories for random-selected NAs
    for na_idx in random_selected:
        if na_idx in random_training_results:
            rep_history = random_training_results[na_idx]['reputation_history']
            all_reputations.extend(rep_history)
            max_steps = max(max_steps, len(rep_history))
    
    if not all_reputations:
        print("Not enough data to create the animation.")
        return
    
    min_reputation = min(all_reputations) - 100
    max_reputation = max(all_reputations) + 200
    
    # Configure both subplots
    for ax, title in zip([ax1, ax2], ['DQN Selected NA Reputation Trajectory', 'Random Selected NA Reputation Trajectory']):
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(min_reputation, max_reputation)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Reputation', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=6600, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Threshold (6600)')
    
    # Initialize line/point objects
    dqn_lines = {}
    random_lines = {}
    dqn_points = {}
    random_points = {}
    
    # DQN-selected NA lines
    for i, na_idx in enumerate(dqn_selected):
        if na_idx in dqn_training_results:
            dqn_color = dqn_colors[na_idx]
            
            # If the NA is malicious, use a special linestyle/marker
            is_malicious = env.fixed_malicious_flags[na_idx]
            if is_malicious:
                mal_type = env.fixed_malicious_types[na_idx]
                # Malicious NAs: dashed line + square marker
                line, = ax1.plot([], [], color=dqn_color, linewidth=4, linestyle='--', alpha=0.9, 
                               label=f'[MAL-T{mal_type}] NA {na_idx}' if i < 5 else "")
                point, = ax1.plot([], [], 's', color=dqn_color, markersize=10, alpha=0.9, 
                                markeredgecolor='red', markeredgewidth=2)  # Square marker with red edge
            else:
                # Normal NAs: solid line + circle marker
                line, = ax1.plot([], [], color=dqn_color, linewidth=3, alpha=0.8, 
                               label=f'[NORM] NA {na_idx}' if i < 5 else "")
                point, = ax1.plot([], [], 'o', color=dqn_color, markersize=8, alpha=0.9)
                
            dqn_lines[na_idx] = line
            dqn_points[na_idx] = point
    
    # Random-selected NA lines
    for i, na_idx in enumerate(random_selected):
        if na_idx in random_training_results:
            random_color = random_colors[na_idx]
            
            # If the NA is malicious, use a special linestyle/marker
            is_malicious = env.fixed_malicious_flags[na_idx]
            if is_malicious:
                mal_type = env.fixed_malicious_types[na_idx]
                # Malicious NAs: dashed line + square marker
                line, = ax2.plot([], [], color=random_color, linewidth=4, linestyle='--', alpha=0.9, 
                               label=f'[MAL-T{mal_type}] NA {na_idx}' if i < 5 else "")
                point, = ax2.plot([], [], 's', color=random_color, markersize=10, alpha=0.9, 
                                markeredgecolor='red', markeredgewidth=2)  # Square marker with red edge
            else:
                # Normal NAs: solid line + circle marker
                line, = ax2.plot([], [], color=random_color, linewidth=3, alpha=0.8, 
                               label=f'[NORM] NA {na_idx}' if i < 5 else "")
                point, = ax2.plot([], [], 'o', color=random_color, markersize=8, alpha=0.9)
                
            random_lines[na_idx] = line
            random_points[na_idx] = point
    
    # Legends
    ax1.legend(loc='upper left', fontsize='small')
    ax2.legend(loc='upper left', fontsize='small')
    
    # Stats overlay text
    dqn_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    random_text = ax2.text(0.02, 0.02, '', transform=ax2.transAxes, fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    def animate(frame):
        """Animation update function."""
        current_step = frame
        
        # Update DQN-selected NAs
        dqn_current_reps = []
        for na_idx in dqn_selected:
            if na_idx in dqn_training_results and na_idx in dqn_lines:
                rep_history = dqn_training_results[na_idx]['reputation_history']
                if current_step < len(rep_history):
                    x_data = list(range(current_step + 1))
                    y_data = rep_history[:current_step + 1]
                    
                    dqn_lines[na_idx].set_data(x_data, y_data)
                    if len(x_data) > 0:
                        dqn_points[na_idx].set_data([x_data[-1]], [y_data[-1]])
                        dqn_current_reps.append(y_data[-1])
        
        # Update random-selected NAs
        random_current_reps = []
        for na_idx in random_selected:
            if na_idx in random_training_results and na_idx in random_lines:
                rep_history = random_training_results[na_idx]['reputation_history']
                if current_step < len(rep_history):
                    x_data = list(range(current_step + 1))
                    y_data = rep_history[:current_step + 1]
                    
                    random_lines[na_idx].set_data(x_data, y_data)
                    if len(x_data) > 0:
                        random_points[na_idx].set_data([x_data[-1]], [y_data[-1]])
                        random_current_reps.append(y_data[-1])
        
        # Update stats
        if dqn_current_reps:
            dqn_avg = np.mean(dqn_current_reps)
            dqn_std = np.std(dqn_current_reps)
            dqn_text.set_text(f'Step: {current_step}\nDQN Selection Stats:\nAvg Reputation: {dqn_avg:.1f}\nStd Dev: {dqn_std:.1f}\nNA Count: {len(dqn_current_reps)}')
        
        if random_current_reps:
            random_avg = np.mean(random_current_reps)
            random_std = np.std(random_current_reps)
            random_text.set_text(f'Step: {current_step}\nRandom Selection Stats:\nAvg Reputation: {random_avg:.1f}\nStd Dev: {random_std:.1f}\nNA Count: {len(random_current_reps)}')
        
        return list(dqn_lines.values()) + list(dqn_points.values()) + list(random_lines.values()) + list(random_points.values()) + [dqn_text, random_text]
    
    # Create animation
    total_frames = max_steps
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=200, blit=True, repeat=True)  # 200ms interval
    
    # Save animation
    filename = 'strategy_comparison_reputation_animation.gif'
    
    try:
        # Ensure the output directory exists
        animations_dir = OUTPUT_ROOT / "episode_animations"
        animations_dir.mkdir(parents=True, exist_ok=True)
        full_path = str(animations_dir / filename)
        
        anim.save(full_path, writer='pillow', fps=5)
        print(f"Selection strategy comparison GIF saved as: {full_path}")
        
        # Print brief stats
        print("Animation summary:")
        print(f"   DQN-selected NAs: {len(dqn_selected)} - {dqn_selected}")
        print(f"   Random-selected NAs: {len(random_selected)} - {random_selected}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {total_frames/5:.1f}s")
        
    except Exception as e:
        print(f"Failed to save selection strategy comparison GIF: {e}")
    
    # Free memory
    plt.close(fig)

# Train and visualize
if __name__ == "__main__":
    print("DQN-based NA selection optimization - 20 NA training configuration")
    print("="*60)
    
    # Tuned training parameters: fixed NA pool for improved stability
    df, loss_history, agent = train_dqn(
        n_na=20,             # Number of NAs: 20
        n_episodes=50,       # Episodes: 50 (stage-wise LR decay)
        steps_per_episode=200,  # Steps per episode: 200
        lr=0.001,            # Learning rate: 0.001
        update_frequency=4,  # Update frequency: train every 4 steps
        use_fixed_nas=True,  # Use a fixed NA pool
        enable_random_fluctuation=False  # Disable random fluctuation
    )
    
    print(f"\nTraining completed. Generated {len(df)} episodes of training data.")
    print(f"Loss history length: {len(loss_history)} training steps")
    
    # Save trained model
    print("\nSaving trained model...")
    try:
        model_dir = OUTPUT_ROOT / "models_ablation_no_slidewindow"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save state_dicts for policy/target networks (recommended)
        policy_state_path = str(model_dir / "policy_net_state_dict.pth")
        target_state_path = str(model_dir / "target_net_state_dict.pth")
        policy_full_path = str(model_dir / "policy_net_complete.pth")
        target_full_path = str(model_dir / "target_net_complete.pth")
        model_info_path = str(model_dir / "model_info.pth")

        torch.save(agent.policy_net.state_dict(), policy_state_path)
        torch.save(agent.target_net.state_dict(), target_state_path)
        
        # Save full models (fallback)
        torch.save(agent.policy_net, policy_full_path)
        torch.save(agent.target_net, target_full_path)
        
        # Save training parameters and metadata
        model_info = {
            'n_na': 20,
            'n_features': 4,
            'n_episodes': 50,  # Actual training episode count
            'learning_rate': 0.001,  # Actual learning rate
            'final_epsilon': agent.epsilon,
            'total_training_steps': len(loss_history),
            'final_loss': loss_history[-1] if loss_history else None,
            'final_reward': df['reward'].iloc[-1] if len(df) > 0 else None
        }
        torch.save(model_info, model_info_path)
        
        print("Model saved successfully.")
        print(f"   - Policy net state_dict: {policy_state_path}")
        print(f"   - Target net state_dict: {target_state_path}")
        print(f"   - Policy net (full): {policy_full_path}")
        print(f"   - Target net (full): {target_full_path}")
        print(f"   - Model metadata: {model_info_path}")
        
    except Exception as e:
        print(f"Model save failed: {e}")
    
    # Create a multi-panel figure to analyze training progress (includes Q-value monitoring)
    # Increase font sizes for more readable PDF exports
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    plt.figure(figsize=(24, 16))
    
    # Panel 1: step-level rewards (aligned with loss sampling)
    plt.subplot(3, 3, 1)
    reward_steps = getattr(agent, 'step_reward_steps', [])
    reward_history = getattr(agent, 'step_reward_history', [])

    if reward_steps and reward_history and len(reward_steps) == len(reward_history):
        window_size = min(100, max(1, len(reward_history)))
        if window_size > 1:
            reward_series = pd.Series(reward_history)
            rolling_mean = reward_series.rolling(window=window_size, min_periods=1).mean()
            rolling_std = reward_series.rolling(window=window_size, min_periods=1).std().fillna(0.0)
            lower = (rolling_mean - rolling_std).to_numpy()
            upper = (rolling_mean + rolling_std).to_numpy()
            mean_vals = rolling_mean.to_numpy()
            plt.fill_between(reward_steps, lower, upper, color='royalblue', alpha=0.15, label='±1σ band')
            plt.plot(reward_steps, mean_vals, label=f'Smoothed Reward (window={window_size})', color='blue', linewidth=2)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.4)
    else:
        # Fallback to episode-level plotting
        fallback_steps = df['total_steps'] if 'total_steps' in df.columns else df['episode']
        window_size = min(100, len(df))
        if len(df) >= window_size:
            smoothed_rewards = df['reward'].rolling(window=window_size, center=True, min_periods=1).mean()
            reward_std = df['reward'].rolling(window=window_size, center=True, min_periods=1).std().fillna(0.0)
            lower = (smoothed_rewards - reward_std).to_numpy()
            upper = (smoothed_rewards + reward_std).to_numpy()
            mean_vals = smoothed_rewards.to_numpy()
            plt.fill_between(fallback_steps, lower, upper, color='lightblue', alpha=0.2, label='±1σ band')
            plt.plot(fallback_steps, mean_vals, label=f'Smoothed Reward (window={window_size})', color='blue', linewidth=2)
        plt.title('Episode Rewards (Fallback)')
        plt.xlabel('Training Steps')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.grid(True, alpha=0.4)
    
    # Panel 2: mean reputation
    plt.subplot(3, 3, 2)
    # Raw series (lower alpha)
    plt.plot(df['episode'], df['mean_reputation'], alpha=0.3, label='Raw Mean Reputation', color='lightgreen')
    
    # Smoothed curve
    window_size = 20
    if len(df) >= window_size:
        smoothed_reputation = df['mean_reputation'].rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(df['episode'], smoothed_reputation, label=f'Smoothed Mean Reputation ({window_size})', color='darkgreen', linewidth=2)
    
    plt.title('Mean Reputation')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reputation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 3: loss curve
    plt.subplot(3, 3, 3)
    if len(loss_history) > 0:
        loss_steps = agent.loss_steps if hasattr(agent, 'loss_steps') and len(agent.loss_steps) == len(loss_history) else list(range(len(loss_history)))
        # Compute moving average for smoothing
        window_size = min(100, len(loss_history) // 10)  # 1/10 of length, capped at 100
        if window_size > 1:
            smoothed_loss = []
            for i in range(len(loss_history)):
                start_idx = max(0, i - window_size + 1)
                smoothed_loss.append(np.mean(loss_history[start_idx:i+1]))
            plt.plot(loss_steps, smoothed_loss, label=f'Smoothed Loss (window={window_size})', alpha=0.8)
        
        # Raw loss (lower alpha)
        plt.plot(loss_steps, loss_history, alpha=0.3, label='Raw Loss', color='gray')
        plt.title('Loss Function Curve')
        plt.xlabel('Training Steps')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Loss Function Curve')
    
    # Panel 4: reward standard deviation (20-episode window)
    plt.subplot(3, 3, 4)
    # Compute reward std over a rolling 20-episode window
    reward_std_data = []
    episode_data = []
    window_size = 20
    
    for i in range(window_size-1, len(df)):
        window_rewards = df['reward'].iloc[i-window_size+1:i+1]
        reward_std = window_rewards.std()
        reward_std_data.append(reward_std)
        episode_data.append(df['episode'].iloc[i])
    
    if reward_std_data:
        plt.plot(episode_data, reward_std_data, label='Reward Std (20-episode window)', color='green', alpha=0.4)
        # Smoothed line
        if len(reward_std_data) >= 20:
            reward_std_series = pd.Series(reward_std_data)
            smoothed_reward_std = reward_std_series.rolling(window=20, center=True, min_periods=1).mean()
            plt.plot(episode_data, smoothed_reward_std, label='Smoothed Reward Std', color='darkgreen', linewidth=2)
    
    plt.title('Reward Standard Deviation (20-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Reward Std')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 5: mean step reward
    plt.subplot(3, 3, 5)
    plt.plot(df['episode'], df['avg_step_reward'], label='Avg Step Reward', color='orange', alpha=0.2)
    # Smoothed line with confidence band
    if len(df) >= 20:
        smoothed_step_reward = df['avg_step_reward'].rolling(window=20, center=True, min_periods=1).mean()
        step_reward_std = df['avg_step_reward'].rolling(window=20, center=True, min_periods=1).std().fillna(0.0)
        lower = (smoothed_step_reward - step_reward_std).to_numpy()
        upper = (smoothed_step_reward + step_reward_std).to_numpy()
        mean_vals = smoothed_step_reward.to_numpy()
        plt.fill_between(df['episode'], lower, upper, color='orange', alpha=0.15, label='±1σ band')
        plt.plot(df['episode'], mean_vals, label='Smoothed Step Reward', color='darkorange', linewidth=2)
    plt.title('Average Step Reward')
    plt.xlabel('Episode')
    plt.ylabel('Avg Step Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 6: step reward std dev (volatility)
    plt.subplot(3, 3, 6)
    plt.plot(df['episode'], df['reward_std'], label='Step Reward Std Dev', color='red', alpha=0.4)
    # Smoothed line
    if len(df) >= 20:
        smoothed_std = df['reward_std'].rolling(window=20, center=True, min_periods=1).mean()
        plt.plot(df['episode'], smoothed_std, label='Smoothed Std Dev', color='darkred', linewidth=2)
    plt.title('Step Reward Volatility (Std Dev)')
    plt.xlabel('Episode')
    plt.ylabel('Reward Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 7: Q-value range
    plt.subplot(3, 3, 7)
    plt.plot(df['episode'], df['q_max'], label='Q Max', color='red', alpha=0.7)
    plt.plot(df['episode'], df['q_min'], label='Q Min', color='blue', alpha=0.7)
    plt.fill_between(df['episode'], df['q_min'], df['q_max'], alpha=0.2, color='purple')
    plt.title('Q-Value Range (Min-Max)')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for wide Q-value ranges
    
    # Panel 8: Q-value mean and std dev
    plt.subplot(3, 3, 8)
    plt.plot(df['episode'], df['q_mean'], label='Q Mean', color='purple', alpha=0.7)
    plt.fill_between(df['episode'], 
                     df['q_mean'] - df['q_std'], 
                     df['q_mean'] + df['q_std'], 
                     alpha=0.3, color='purple', label='Q Mean ± Std')
    plt.title('Q-Value Mean and Std Dev')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for wide Q-value ranges
    
    # Panel 9: learning rate and epsilon schedule
    plt.subplot(3, 3, 9)
    if hasattr(agent, 'lr_history') and agent.lr_history:
        # Fix the x-axis: generate correct episode indices
        lr_episodes = []
        total_episodes = len(df)  # Total episodes from the dataframe
        for i in range(len(agent.lr_history)):
            if i < len(agent.lr_history) - 1:
                # Earlier points are multiples of 10
                lr_episodes.append(i * 10)
            else:
                # Final point: if total episodes is not a multiple of 10, use the true last episode index
                final_episode = total_episodes - 1
                if final_episode % 10 == 0:
                    lr_episodes.append(i * 10)  # Exactly a multiple of 10
                else:
                    lr_episodes.append(final_episode)  # Use the true last episode index
        
        # Use dual y-axes for learning rate and epsilon
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Learning rate (left y-axis, log scale)
        line1 = ax1.plot(lr_episodes, agent.lr_history, label='Learning Rate', color='orange', linewidth=2)
        ax1.set_ylabel('Learning Rate', color='orange')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='orange')
        
        # Epsilon (right y-axis, linear scale)
        if hasattr(agent, 'epsilon_history') and agent.epsilon_history:
            line2 = ax2.plot(lr_episodes, agent.epsilon_history, label='Epsilon (exploration rate)', color='purple', linewidth=2)
            ax2.set_ylabel('Epsilon', color='purple')
            ax2.tick_params(axis='y', labelcolor='purple')
        
        plt.title('AdamW+AMSGrad Learning Rate & Epsilon')
        plt.xlabel('Episode')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Training Schedule Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('AdamW+AMSGrad Learning Rate & Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Values')
    
    plt.tight_layout()
    
    # Save training result figures locally
    main_png = str(OUTPUT_ROOT / "src" / "training_results.png")
    Path(main_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(main_png, dpi=300, bbox_inches='tight')
    print(f"Training result figure saved as: {main_png}")

    # Save reward curve as a single-page PDF
    try:
        # Use Times New Roman and larger fonts for PDF export
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'Times', 'serif'],
            'font.size': 25,
            'axes.titlesize': 25,
            'axes.labelsize': 25,
            'xtick.labelsize': 23,
            'ytick.labelsize': 23,
            'legend.fontsize': 23,
        })
        plt.figure(figsize=(10, 6))
        # Prefer step-level reward data (if available)
        reward_steps = getattr(agent, 'step_reward_steps', [])
        reward_history = getattr(agent, 'step_reward_history', [])
        if reward_steps and reward_history and len(reward_steps) == len(reward_history):
            window_size = min(100, max(1, len(reward_history)))
            if window_size > 1:
                import pandas as _pd
                reward_series = _pd.Series(reward_history)
                rolling_mean = reward_series.rolling(window=window_size, min_periods=1).mean()
                rolling_std = reward_series.rolling(window=window_size, min_periods=1).std().fillna(0.0)
                lower = (rolling_mean - rolling_std).to_numpy()
                upper = (rolling_mean + rolling_std).to_numpy()
                mean_vals = rolling_mean.to_numpy()
                plt.fill_between(reward_steps, lower, upper, color='royalblue', alpha=0.15, label='±1σ band')
                plt.plot(reward_steps, mean_vals, label=f'Smoothed Reward (window={window_size})', color='blue', linewidth=2)
            else:
                plt.plot(reward_steps, reward_history, label='Reward', color='blue')
            plt.xlabel('Training Steps')
            plt.ylabel('Reward')
            plt.legend()
            plt.grid(True, alpha=0.4)
        else:
            # Fallback to episode-level plotting
            fallback_steps = df['total_steps'] if 'total_steps' in df.columns else df['episode']
            window_size = min(100, len(df))
            if len(df) >= window_size:
                smoothed_rewards = df['reward'].rolling(window=window_size, center=True, min_periods=1).mean()
                reward_std = df['reward'].rolling(window=window_size, center=True, min_periods=1).std().fillna(0.0)
                lower = (smoothed_rewards - reward_std).to_numpy()
                upper = (smoothed_rewards + reward_std).to_numpy()
                mean_vals = smoothed_rewards.to_numpy()
                plt.fill_between(fallback_steps, lower, upper, color='lightblue', alpha=0.2, label='±1σ band')
                plt.plot(fallback_steps, mean_vals, label=f'Smoothed Reward (window={window_size})', color='blue', linewidth=2)
            else:
                plt.plot(df['episode'], df['reward'], label='Reward', color='blue')
            plt.xlabel('Training Steps')
            plt.ylabel('Total Reward')
            plt.legend()
            plt.grid(True, alpha=0.4)

        reward_pdf = str(OUTPUT_ROOT / "src" / "training_results_reward.pdf")
        Path(reward_pdf).parent.mkdir(parents=True, exist_ok=True)
        # Ensure vector fonts in PDF (embed TrueType as Type42)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.tight_layout()
        plt.savefig(reward_pdf, bbox_inches='tight')
        plt.close()
        print(f"Saved reward PDF: {reward_pdf}")
    except Exception as e:
        print(f"Failed to save reward PDF: {e}")

    # Save MSE loss curve as a single-page PDF
    try:
        # Use Times New Roman and larger fonts for PDF export
        plt.rcParams.update({
            'font.family': ['Times New Roman', 'Times', 'serif'],
            'font.size': 25,
            'axes.titlesize': 25,
            'axes.labelsize': 25,
            'xtick.labelsize': 23,
            'ytick.labelsize': 23,
            'legend.fontsize': 23,
        })
        plt.figure(figsize=(10, 6))
        if len(loss_history) > 0:
            loss_steps = agent.loss_steps if hasattr(agent, 'loss_steps') and len(agent.loss_steps) == len(loss_history) else list(range(len(loss_history)))
            window_size = min(100, len(loss_history) // 10) if len(loss_history) >= 10 else 1
            if window_size > 1:
                smoothed_loss = []
                for i in range(len(loss_history)):
                    start_idx = max(0, i - window_size + 1)
                    smoothed_loss.append(np.mean(loss_history[start_idx:i+1]))
                plt.plot(loss_steps, smoothed_loss, label=f'Smoothed Loss (window={window_size})', alpha=0.8)
            plt.plot(loss_steps, loss_history, alpha=0.3, label='Raw Loss', color='gray')
            plt.xlabel('Training Steps')
            plt.ylabel('MSE Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=plt.gca().transAxes)

        loss_pdf = str(OUTPUT_ROOT / "src" / "training_results_loss.pdf")
        Path(loss_pdf).parent.mkdir(parents=True, exist_ok=True)
        # Ensure vector fonts in PDF (embed TrueType as Type42)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.tight_layout()
        plt.savefig(loss_pdf, bbox_inches='tight')
        plt.close()
        print(f"Saved loss PDF: {loss_pdf}")
    except Exception as e:
        print(f"Failed to save loss PDF: {e}")

    # Close the main figure
    plt.close()
    
    # Print loss statistics
    if len(loss_history) > 0:
        print("\nLoss statistics:")
        print(f"Total training steps: {len(loss_history)}")
        print(f"Mean loss: {np.mean(loss_history):.6f}")
        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Min loss: {np.min(loss_history):.6f}")
        print(f"Max loss: {np.max(loss_history):.6f}")
        
        # Check convergence
        if len(loss_history) > 1000:
            recent_loss = np.mean(loss_history[-1000:])  # Mean loss over the last 1000 steps
            early_loss = np.mean(loss_history[:1000])    # Mean loss over the first 1000 steps
            improvement = (early_loss - recent_loss) / early_loss * 100
            print(f"Loss improvement: {improvement:.2f}%")
    else:
        print("\nWARNING: no loss data recorded (training too short or batch_size too large).")
    
    # Detailed reward volatility analysis
    print("\nEpisode reward volatility analysis:")
    print(f"Total episodes: {len(df)}")
    print(f"Mean episode reward: {df['reward'].mean():.2f}")
    print(f"Episode reward std: {df['reward'].std():.2f}")
    print(f"Episode reward range: {df['reward'].min():.2f} ~ {df['reward'].max():.2f}")
    print(f"Reward coefficient of variation (CV): {df['reward'].std() / abs(df['reward'].mean()):.2f}")
    
    print("\nPer-step reward statistics:")
    print(f"Mean step reward: {df['avg_step_reward'].mean():.2f}")
    print(f"Step reward volatility (mean std): {df['reward_std'].mean():.2f}")
    print(f"Overall success rate: {df['success_rate'].mean():.3f}")
    
    print("\nPotential causes of reward volatility:")
    high_volatility_episodes = df[df['reward_std'] > df['reward_std'].quantile(0.8)]
    print(f"High-volatility episodes: {len(high_volatility_episodes)} ({len(high_volatility_episodes)/len(df)*100:.1f}%)")
    
    if len(high_volatility_episodes) > 0:
        print(f"Mean reward (high-volatility): {high_volatility_episodes['reward'].mean():.2f}")
        print(f"Mean success rate (high-volatility): {high_volatility_episodes['success_rate'].mean():.3f}")
    
    # Analyze changes across early/mid/late phases
    early_episodes = df.iloc[:len(df)//3]  # First third
    middle_episodes = df.iloc[len(df)//3:2*len(df)//3]  # Middle third
    late_episodes = df.iloc[2*len(df)//3:]  # Last third
    
    print("\nTraining phase analysis:")
    print(f"Early (Episodes 0-{len(early_episodes)}): mean reward={early_episodes['reward'].mean():.2f}, volatility={early_episodes['reward'].std():.2f}")
    print(f"Mid (Episodes {len(early_episodes)}-{len(early_episodes)+len(middle_episodes)}): mean reward={middle_episodes['reward'].mean():.2f}, volatility={middle_episodes['reward'].std():.2f}")
    print(f"Late (Episodes {len(early_episodes)+len(middle_episodes)}-{len(df)}): mean reward={late_episodes['reward'].mean():.2f}, volatility={late_episodes['reward'].std():.2f}")
    
    improvement = (late_episodes['reward'].mean() - early_episodes['reward'].mean()) / abs(early_episodes['reward'].mean()) * 100
    volatility_change = (late_episodes['reward'].std() - early_episodes['reward'].std()) / early_episodes['reward'].std() * 100
    
    print("\nTraining effects:")
    print(f"Reward improvement: {improvement:+.1f}%")
    print(f"Volatility change: {volatility_change:+.1f}% ({'improved' if volatility_change < 0 else 'worsened'})")
    
    # Record end time and total runtime
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n" + "="*80)
    print("Program runtime summary")
    print(f"="*80)
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_duration(total_duration)}")
    print("Training throughput:")
    print(f"   - Total episodes: {len(df)}")
    print(f"   - Avg time per episode: {total_duration/len(df):.2f}s")
    if len(loss_history) > 0:
        print(f"   - Total training steps: {len(loss_history)}")
        print(f"   - Avg time per step: {total_duration/len(loss_history)*1000:.2f}ms")
    print(f"="*80)
    print("Program finished.")
    
    # Export parameters from the final model NA selection
    print(f"\n" + "="*80)
    print("Export NA selection parameters")
    print("="*80)
    
    try:
        # Create a fresh environment instance to obtain the current state
        export_env = Environment(n_na=20, use_fixed_nas=True, enable_random_fluctuation=False)
        export_env.reset()
        
        # Get current state and initial reputations
        current_state = export_env.get_state()
        initial_reputations = export_env.episode_start_reputation
        
        # Export NA selection parameters
        export_file = agent.export_na_selection_parameters(
            state=current_state, 
            initial_reputations=initial_reputations,
            env=export_env
        )
        
        print("NA selection parameter export succeeded.")
        
    except Exception as e:
        print(f"NA selection parameter export failed: {e}")
    
    # Demonstrate network architecture flexibility
    print(f"\n" + "="*80)
    print("Network architecture flexibility demo - handling different NA counts")
    print("="*80)
    
    def test_network_flexibility(trained_agent):
        """Demonstrate that the trained network can handle different NA counts."""
        print("Testing whether the network can handle different NA counts...")
        
        # Test different NA counts
        test_cases = [5, 10, 15, 25, 30]
        
        for n_test_na in test_cases:
            # Create test data with 4 features (reputation, success_rate, signature_delay, hunger)
            test_state = np.zeros((n_test_na, 4))
            test_state[:, 0] = np.random.uniform(3300, 10000, n_test_na)  # reputation
            test_state[:, 1] = np.random.uniform(0.2, 0.9, n_test_na)     # success_rate  
            test_state[:, 2] = np.random.uniform(50, 250, n_test_na)      # signature_delay (ms)
            test_state[:, 3] = np.random.uniform(0.0, 1.0, n_test_na)     # hunger (0%-100%)

            # Normalize
            normalized_test_state = test_state.copy()
            normalized_test_state[:, 0] = (test_state[:, 0] - 3000) / (10000 - 3000)
            normalized_test_state[:, 1] = test_state[:, 1]
            normalized_test_state[:, 2] = np.clip((test_state[:, 2] - 50) / (300 - 50), 0, 1)
            normalized_test_state[:, 3] = test_state[:, 3]
            
            # Convert to tensor
            test_tensor = torch.FloatTensor(normalized_test_state).unsqueeze(0).to(trained_agent.device)
            
            # Forward pass
            with torch.no_grad():
                q_values = trained_agent.policy_net(test_tensor)
            
            print(f"NA count: {n_test_na:2d} -> Q shape: {q_values.shape} | max Q: {q_values.max().item():.4f}")
        
        print("Network successfully handled all tested NA counts.")
        print("Benefit: a trained model can be applied directly to any NA count.")
        print("Features: reputation, success_rate, signature_delay, hunger")
    
    test_network_flexibility(agent)
