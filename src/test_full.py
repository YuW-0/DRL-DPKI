import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only GPU 1 (0-based index)
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
- Double DQN: mitigates Q-value overestimation
- Experience replay: improves sample efficiency
- Target network: stabilizes training
- NA-count-invariant architecture: supports any number of NAs

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
        Select a single NA action while distinguishing high/low-reputation NAs.
        
        Args:
            state: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations, used for grouping
            available_mask: shape (n_na,) - mask of selectable NAs; None means all NAs are selectable
            record_policy: bool - whether to record policy analysis data
        
        Returns:
            int: selected NA index
        """
        # If initial reputations are not provided, fall back to current reputations
        if initial_reputations is None:
            reputations = state[:, 0]  # Use current reputations
        else:
            reputations = initial_reputations
        
        # Set available NA mask
        if available_mask is None:
            available_mask = np.ones(self.n_na, dtype=bool)
        
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0:
            return 0  # If no NA is available, default to 0

        # Compute Q-values and policy distribution (needed for analysis even with epsilon-greedy)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, n_na, 5]
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).cpu()  # [n_na]
        
        # Consider Q-values of available NAs only
        masked_q_values = q_values.clone()
        masked_q_values[~torch.from_numpy(available_mask)] = float('-inf')  # Set unavailable NAs to -inf
        
        # Compute softmax policy distribution (for entropy)
        valid_q_values = q_values[available_mask]
        if len(valid_q_values) > 1:
            policy_probs = F.softmax(valid_q_values, dim=0).numpy()
            # Compute policy entropy
            policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-8))
        else:
            policy_entropy = 0.0
            policy_probs = np.array([1.0])
        
        # Record policy analysis data
        if record_policy:
            self.policy_entropy_history.append(policy_entropy)
            # Record full action distribution (including unavailable NAs)
            full_policy_dist = np.zeros(self.n_na)
            full_policy_dist[available_indices] = policy_probs
            self.action_distribution_history.append(full_policy_dist.copy())
        
        # Select action
        if np.random.rand() < self.epsilon:
            # Random selection: choose one from available NAs
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
        Cosine annealing with warm-up learning-rate scheduler.
        First 5%: warm-up, linearly increases LR to the maximum.
        Last 95%: cosine annealing, smoothly decays LR to the minimum.
        """
        if current_episode >= self.total_episodes:
            progress = 1.0
        else:
            progress = current_episode / self.total_episodes
        
        # Set warm-up ratio
        warmup_ratio = 0.05  # Warm-up for the first 5%
        
        if progress <= warmup_ratio:
            # Warm-up: linearly increase to max LR
            warmup_progress = progress / warmup_ratio
            current_lr = self.final_lr + (self.initial_lr - self.final_lr) * warmup_progress
        else:
            # Cosine annealing
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
        Make the final selection using the trained DQN network, consistent with training.
        
        Args:
            state: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations, used for grouping
        
        Returns:
            dict: selection results, Q-values, and analysis information
        """
        # Use multiple single selections to form a combination
        selected_nas = self.make_multiple_selections(state, initial_reputations, 5, 'balanced')
        
        # Compute Q-values for all NAs for analysis
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        # Grouping info
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        low_rep_indices = np.where(low_rep_mask)[0]
        high_rep_indices = np.where(high_rep_mask)[0]
        
        # Random selection for comparison
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
        Export NA parameters for the last model selection to a file.
        
        Args:
            state: shape (n_na, 4) - current state (reputation, success_rate, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations
            env: Environment object - used to get weighted success rate and weighted delay grade
            output_file: str - output file path; if None, use the default path
        
        Returns:
            str: path to the exported file
        """
        if output_file is None:
            output_file = '/mnt/data/wy2024/src/na_selection_parameters.csv'
        
        # Compute Q-values for all NAs
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        # Run final selection to obtain the selection result
        selection_result = self.make_final_selection(state, initial_reputations)
        selected_nas = selection_result['dqn_selected']
        
        # Get weighted success rate and weighted delay grade (if an env is provided)
        weighted_success_rates = []
        weighted_delay_levels = []
        if env is not None:
            for i in range(self.n_na):
                window_summary = env.get_na_window_summary(i)
                weighted_success_rates.append(window_summary['weighted_success_rate'])
                weighted_delay_levels.append(window_summary['weighted_delay_grade'])
        else:
            # If env is not provided, use values from the current state
            weighted_success_rates = state[:, 1].tolist()  # success_rate
            weighted_delay_levels = [0] * self.n_na  # Default value
        
        # Prepare export data
        export_data = []
        for i in range(self.n_na):
            na_data = {
                'NA_Index': i,
                'Q_Value': all_q_values[i],
                'Current_Reputation': state[i, 0],  # reputation
                'Success_Rate': state[i, 1],        # success_rate
                'Signature_Delay': env.signature_delay[i] if env is not None else state[i, 2],     # Use raw delay grade value
                'Hunger_Level': state[i, 3],        # hunger
                'Weighted_Success_Rate': weighted_success_rates[i],
                'Weighted_Delay_Grade': weighted_delay_levels[i],
                'Initial_Reputation': initial_reputations[i],
                'Reputation_Group': 'High' if initial_reputations[i] >= 6600 else 'Low',
                'Selected_by_DQN': i in selected_nas,
                'Selection_Rank': selected_nas.index(i) + 1 if i in selected_nas else None
            }
            export_data.append(na_data)
        
        # Sort by Q-value in descending order
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
        print(f"   - High-reputation NAs: {len([x for x in export_data if x['Reputation_Group'] == 'High'])}")
        print(f"   - Low-reputation NAs: {len([x for x in export_data if x['Reputation_Group'] == 'Low'])}")
        print(f"   - Q-value range: {df['Q_Value'].min():.6f} ~ {df['Q_Value'].max():.6f}")
        
        return output_file

    def store(self, s, a, r, s_, done):
        """
        Store experience; now supports a single-NA action.
        
        Args:
            s: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            a: int - selected NA index
            r: float - reward
            s_: shape (n_na, 5) - next state (reputation, success_rate, activity, signature_delay, hunger)
            done: bool - whether the episode is done
        """
        self.memory.append((s.copy(), int(a), r, s_.copy(), done))

    def update(self, global_step=None):
        # Check whether there is enough experience to train
        if len(self.memory) < self.min_memory_size:
            return None  # Not enough data; skip training
            
        if len(self.memory) < self.batch_size:
            return None  # Not enough samples for one batch
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors and move to device, keeping original dimensions
        states = torch.FloatTensor(np.array(states)).to(self.device)  # [batch_size, n_na, n_features]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # [batch_size, n_na, n_features]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # [batch_size, 1]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # [batch_size, 1]
        
        # Compute current Q-values
        current_q_values = self.policy_net(states)  # [batch_size, n_na]
        current_q = current_q_values.gather(1, actions)  # [batch_size, 1]
        
        # Compute target Q-values (Double DQN)
        with torch.no_grad():
            # DDQN: select action with policy net, evaluate with target net
            # 1) Choose the best action in the next state using the policy network
            next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]  # [batch_size, 1]
            
            # 2) Evaluate the chosen action using the target network (avoid overestimation)
            next_q_values = self.target_net(next_states)  # [batch_size, n_na]
            next_q = next_q_values.gather(1, next_actions)  # [batch_size, 1]
            
            target = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q, target)
        
        # Record loss values and corresponding training steps
        self.loss_history.append(loss.item())
        if global_step is not None:
            self.loss_steps.append(global_step)
        else:
            self.loss_steps.append(self.learn_step)
        
        self.optimizer.zero_grad()
        loss.backward()    
        self.optimizer.step()

        # Update the target network only every N policy-network updates (balance stability vs speed)
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()  # Return loss value

    def maybe_update(self, global_step=None):
        """
        Conditional updates: train more frequently to strengthen gradient signals.
        
        Returns:
            list: loss values for this training burst; None if no training happened
        """
        self.step_counter += 1
        
        # Check whether update conditions are met
        if (self.step_counter % self.update_frequency == 0 and 
            len(self.memory) >= self.min_memory_size):
            
            # Train multiple batches to strengthen gradient signals
            batch_losses = []
            training_batches = max(2, self.update_frequency)  # Train multiple rounds per trigger
            
            for _ in range(training_batches):
                loss = self.update(global_step=global_step)
                if loss is not None:
                    batch_losses.append(loss)
            
            return batch_losses if batch_losses else None
        
        return None

    def debug_q_values(self, state, initial_reputations):
        """Debug Q-value learning behavior."""
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze().cpu()
        
        print(f"Q-value stats: min={q_values.min():.4f}, max={q_values.max():.4f}, "
              f"std={q_values.std():.4f}")
        
        # Check whether Q-values have meaningful variation
        if q_values.std() < 0.1:
            print("Warning: Q-value variation is too small; the network may not have learned useful information")
        
        # Analyze Q-value differences between low/high-reputation groups
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        
        if np.any(low_rep_mask) and np.any(high_rep_mask):
            low_q_mean = q_values[low_rep_mask].mean()
            high_q_mean = q_values[high_rep_mask].mean()
            print(f"Low-reputation mean Q: {low_q_mean:.4f}, high-reputation mean Q: {high_q_mean:.4f}")
            print(f"Q-value learning direction: {'correct' if high_q_mean > low_q_mean else 'wrong'}")
        
        return q_values

    def make_multiple_selections(self, state, initial_reputations, num_selections=5, strategy='balanced'):
        """
        Compose the final NA set by performing multiple single selections.
        
        Args:
            state: shape (n_na, 5) - current state (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - initial reputations
            num_selections: int - number of NAs to select
            strategy: str - selection strategy ('balanced', 'top_q', 'by_group')
        
        Returns:
            list: selected NA indices
        """
        original_epsilon = self.epsilon
        self.epsilon = 0  # Use a purely greedy policy
        
        selected_nas = []
        available_mask = np.ones(self.n_na, dtype=bool)
        
        if strategy == 'balanced':
            # Balanced strategy: select from low/high-reputation groups in proportion
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            # Target: select 2 low-reputation + 3 high-reputation (or adjusted by availability)
            target_low = min(2, len(low_rep_indices), num_selections)
            target_high = min(num_selections - target_low, len(high_rep_indices))
            
            # Select from low-reputation group
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            for _ in range(target_low):
                if np.any(low_mask):
                    selected_na = self.select_action(state, initial_reputations, low_mask)
                    selected_nas.append(selected_na)
                    low_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            # Select from high-reputation group
            high_mask = np.zeros(self.n_na, dtype=bool)
            high_mask[high_rep_indices] = True
            high_mask &= available_mask  # Exclude already selected
            
            for _ in range(target_high):
                if np.any(high_mask):
                    selected_na = self.select_action(state, initial_reputations, high_mask)
                    selected_nas.append(selected_na)
                    high_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            # If more are needed, select from the remaining pool
            remaining_needed = num_selections - len(selected_nas)
            for _ in range(remaining_needed):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'top_q':
            # Simple strategy: pick top-Q NAs
            for _ in range(min(num_selections, self.n_na)):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'by_group':
            # Group strategy: select low-reputation first, then high-reputation
            # Grouping
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            # Select low-reputation group first
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            while len(selected_nas) < num_selections and np.any(low_mask):
                selected_na = self.select_action(state, initial_reputations, low_mask)
                selected_nas.append(selected_na)
                low_mask[selected_na] = False
                available_mask[selected_na] = False
            
            # Then select high-reputation group
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
            print("Random fluctuation mode enabled: success rate, hunger, delay, etc. will change dynamically")
        else:
            print("Fixed mode: all parameters stay at their base values (no random fluctuation)")

        # Hunger growth control parameters (kept consistent with the baseline environment)
        self.hunger_growth_scale = 10.0
        self.hunger_growth_log_base = 11.0
        
        # Sliding-window queue system per NA
        self.window_pack_interval = 5  # Pack once every 5 steps
        self.window_queue_size = window_size  # Max queue capacity per NA
        self.na_window_queues = {}  # Per-NA sliding-window queues
        self.current_step_count = 0  # Current step counter
        self.na_current_pack = {}  # Per-NA in-progress transaction pack
        
        for na_id in range(n_na):
            self.na_window_queues[na_id] = deque(maxlen=window_size)  # Queue stores packed transaction data
            self.na_current_pack[na_id] = {
                'transactions': [],
                'success_count': 0,
                'total_count': 0,
                'reputation_changes': [],
                'start_step': 0,
                'end_step': 0
            }
        print(
            f"Sliding-window queues: per-NA queues, pack every {self.window_pack_interval} steps, "
            f"queue capacity {window_size}"
        )
        
        if use_fixed_nas:
            # Train with a fixed NA set
            print(f"Training with a more complex fixed NA set (total {n_na} NAs)")
            print("Design principle: each reputation group contains diverse NA types for contrastive learning")
            
            # Note: do not reset random seeds; rely on the global seed for consistency
            # Generate a fixed batch of distinguishable NAs (using the global RNG state)
            
            # Redesign NA distribution: create 4 explicit reputation ranges
            quarter = n_na // 4
            remaining = n_na % 4
            
            # Allocate NA counts per range
            group_sizes = [quarter] * 4
            for i in range(remaining):
                group_sizes[i] += 1
            
            print(
                f"NA grouping: very-low {group_sizes[0]}, low {group_sizes[1]}, "
                f"high {group_sizes[2]}, very-high {group_sizes[3]}"
            )
            
            # Generate reputations per group
            very_low_rep = np.random.uniform(3300, 4500, group_sizes[0])    # Very-low: 3300-4500
            low_rep = np.random.uniform(4500, 6600, group_sizes[1])         # Low: 4500-6600
            high_rep = np.random.uniform(6600, 8200, group_sizes[2])        # High: 6600-8200
            very_high_rep = np.random.uniform(8200, 9800, group_sizes[3])   # Very-high: 8200-9800
            
            self.fixed_initial_reputation = np.concatenate([very_low_rep, low_rep, high_rep, very_high_rep])
            
            # Create diverse success-rate distributions per group (including complex malicious NAs)
            success_rates_list = []
            malicious_flags_list = []  # Which NAs are malicious
            malicious_types_list = []  # Malicious NA types
            
            # Define complex malicious NA behavior types:
            # Type 1: "high success + high delay" - looks good but intentionally delays
            # Type 2: "high success + stealth attack" - attacks through other means
            # Type 3: "medium success + extremely unstable" - unpredictable behavior
            # Type 4: "classic low success" - clearly low quality
            
            # Very-low group (3300-4500): diverse types + complex malicious NAs
            # 30% truly low-skill, 25% medium-skill, 15% hidden strong, 30% malicious (multiple types)
            vl_count_low = int(group_sizes[0] * 0.3)
            vl_count_med = int(group_sizes[0] * 0.25) 
            vl_count_high = int(group_sizes[0] * 0.15)
            vl_count_malicious = group_sizes[0] - vl_count_low - vl_count_med - vl_count_high
            
            vl_sr_low = np.random.uniform(0.25, 0.45, vl_count_low)     # Truly low-skill
            vl_sr_med = np.random.uniform(0.55, 0.70, vl_count_med)     # Medium-skill
            vl_sr_high = np.random.uniform(0.75, 0.90, vl_count_high)   # Hidden strong
            
            # Malicious NA mix for very-low group (simplified to two types)
            vl_mal_type1 = max(1, vl_count_malicious // 2)      # High success + high delay
            vl_mal_type4 = vl_count_malicious - vl_mal_type1    # Classic low success
            
            vl_sr_mal_type1 = np.random.uniform(0.65, 0.80, vl_mal_type1)  # High-success disguise
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
            
            # Shuffle order (keep flag and type aligned)
            shuffle_indices = np.random.permutation(len(vl_success_rates))
            vl_success_rates = vl_success_rates[shuffle_indices]
            vl_malicious_flags = vl_malicious_flags[shuffle_indices]
            vl_all_malicious_types = vl_all_malicious_types[shuffle_indices]
            
            success_rates_list.append(vl_success_rates)
            malicious_flags_list.append(vl_malicious_flags)
            malicious_types_list.append(vl_all_malicious_types)
            
            # Low group (4500-6600): more variation + complex malicious NAs
            # 25% lower-medium skill, 30% medium skill, 20% high-potential, 25% malicious NAs
            l_count_low = int(group_sizes[1] * 0.25)
            l_count_med = int(group_sizes[1] * 0.30)
            l_count_high = int(group_sizes[1] * 0.20)
            l_count_malicious = group_sizes[1] - l_count_low - l_count_med - l_count_high
            
            l_sr_low = np.random.uniform(0.35, 0.55, l_count_low)       # Lower-medium skill
            l_sr_med = np.random.uniform(0.60, 0.75, l_count_med)       # Medium skill
            l_sr_high = np.random.uniform(0.80, 0.95, l_count_high)     # High-potential
            
            # Malicious NA mix for low group (simplified to two types)
            l_mal_type1 = max(1, l_count_malicious // 2)      # High success + high delay
            l_mal_type4 = l_count_malicious - l_mal_type1     # Classic low success (slightly higher than very-low group)
            
            l_sr_mal_type1 = np.random.uniform(0.75, 0.90, l_mal_type1)  # High-success disguise
            l_sr_mal_type4 = np.random.uniform(0.15, 0.40, l_mal_type4)  # Relatively low success (higher than very-low group)
            
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
            
            # High group (6600-8200): mostly strong with exceptions + complex malicious NAs
            # 45% high skill, 25% medium skill, 10% unexpectedly low skill, 20% malicious NAs
            h_count_high = int(group_sizes[2] * 0.45)
            h_count_med = int(group_sizes[2] * 0.25)
            h_count_low = int(group_sizes[2] * 0.10)
            h_count_malicious = group_sizes[2] - h_count_high - h_count_med - h_count_low
            
            h_sr_high = np.random.uniform(0.75, 0.90, h_count_high)     # High skill
            h_sr_med = np.random.uniform(0.60, 0.75, h_count_med)       # Medium skill
            h_sr_low = np.random.uniform(0.40, 0.60, h_count_low)       # Unexpectedly low skill
            
            # Malicious NA mix for high group (simplified to one type)
            # Keep Type 1 only (high success + high delay), as Type 4 is unrealistic for high group
            h_mal_type1 = h_count_malicious  # All are Type 1
            
            h_sr_mal_type1 = np.random.uniform(0.82, 0.95, h_mal_type1)  # Very high success disguise
            
            h_sr_malicious = h_sr_mal_type1
            h_malicious_types = np.full(h_mal_type1, 1)  # All are Type 1
            
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
            
            # Very-high group (8200-9800): mostly excellent with variation + complex malicious NAs
            # 55% very high skill, 25% high skill, 5% medium skill, 15% malicious NAs (ultimate disguise)
            vh_count_very_high = int(group_sizes[3] * 0.55)
            vh_count_high = int(group_sizes[3] * 0.25)
            vh_count_med = int(group_sizes[3] * 0.05)
            vh_count_malicious = group_sizes[3] - vh_count_very_high - vh_count_high - vh_count_med
            
            vh_sr_very_high = np.random.uniform(0.85, 0.98, vh_count_very_high)  # Very high skill
            vh_sr_high = np.random.uniform(0.75, 0.85, vh_count_high)            # High skill
            vh_sr_med = np.random.uniform(0.65, 0.75, vh_count_med)              # Medium skill
            
            # Malicious NA mix for very-high group (simplified to one type)
            # Keep Type 1 only (very high success + intentionally high delay)
            vh_mal_type1 = vh_count_malicious  # All are Type 1
            
            vh_sr_mal_type1 = np.random.uniform(0.90, 0.98, vh_mal_type1)  # Very high success disguise
            
            vh_sr_malicious = vh_sr_mal_type1
            vh_malicious_types = np.full(vh_mal_type1, 1)  # All are Type 1
            
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
            self.fixed_malicious_flags = np.concatenate(malicious_flags_list)  # Malicious flags
            self.fixed_malicious_types = np.concatenate(malicious_types_list)  # Malicious types
            
            # Activity distribution: malicious NAs do not differ by activity; use the same rule as normal NAs
            
            
            # Allocate delay by malicious type (key manifestation of malicious behavior)
            delay_list = []
            
            for group_idx, group_size in enumerate(group_sizes):
                malicious_mask = malicious_flags_list[group_idx]
                malicious_types = malicious_types_list[group_idx]
                normal_mask = ~malicious_mask
                
                # Normal NA delay-grade distribution (higher reputation => lower delay grade)
                if group_idx == 0:  # Very-low group
                    normal_delay = np.random.uniform(0.3, 0.8, np.sum(normal_mask))
                elif group_idx == 1:  # Low group
                    normal_delay = np.random.uniform(0.25, 0.65, np.sum(normal_mask))
                elif group_idx == 2:  # High group
                    normal_delay = np.random.uniform(0.15, 0.5, np.sum(normal_mask))
                else:  # Very-high group
                    normal_delay = np.random.uniform(0.1, 0.4, np.sum(normal_mask))
                
                # Malicious NA delay allocation (designed per type)
                malicious_delay = np.zeros(np.sum(malicious_mask))
                mal_indices = np.where(malicious_mask)[0]
                
                for i, mal_idx in enumerate(mal_indices):
                    mal_type = malicious_types[mal_idx]
                    
                    if mal_type == 1:  # Type 1: high success + high delay (intentional delay attack)
                        # These malicious NAs intentionally increase delay to degrade system performance
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.7, 1.0)  # Extremely high delay grade
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.6, 0.9)  # Very high delay grade
                        elif group_idx == 2:
                            malicious_delay[i] = np.random.uniform(0.5, 0.8)  # High delay grade (relative to normals in same group)
                        else:
                            malicious_delay[i] = np.random.uniform(0.4, 0.7)  # Noticeably higher than normal (same group)
                            
                    else:  # Type 4: classic low success
                        # These malicious NAs perform poorly overall, including delay grade
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.6, 0.95)  # High delay grade
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.55, 0.9)  # High delay grade
                        # Type 4 does not appear in high/very-high groups
                
                # Combine normal and malicious delays
                group_delay = np.zeros(group_size)
                group_delay[normal_mask] = normal_delay
                group_delay[malicious_mask] = malicious_delay
                delay_list.append(group_delay)
            
            self.fixed_initial_signature_delay = np.concatenate(delay_list)
            
            # Fixed initial hunger per NA (randomly distributed in 0-50% to simulate uneven initial load)
            self.fixed_initial_hunger = np.random.uniform(0.0, 0.5, n_na)
            
            # Set total transactions and success counts accordingly
            self.fixed_initial_total_tx = np.random.randint(20, 80, n_na)
            self.fixed_initial_success_count = (self.fixed_initial_success_rate * self.fixed_initial_total_tx).astype(int)
            
            # Delay configuration (as a normalized delay grade in 0.0-1.0)
            self.ideal_delay_grade = 0.0  # Ideal delay grade: 0% (best)
            self.max_acceptable_delay_grade = 1.0  # Max acceptable delay grade: 100% (worst)
            
            # Note: do not reset RNG seeds; keep global determinism
            
            print("Complex NA set preview (with refined malicious NA types):")
            print("Idx\tInitialRep\tSuccessRate\tSignatureDelay(ms)\tHunger\tDelayGrade\tRepGroup\t\tAbilityType\t\tMaliciousType")
            
            # Reputation group helper
            def get_reputation_group(rep):
                if rep < 4500:
                    return "Very-low"
                elif rep < 6600:
                    return "Low"
                elif rep < 8200:
                    return "High"
                else:
                    return "Very-high"
            
            # Malicious type label helper
            def get_malicious_type_desc(mal_type):
                if mal_type == 1:
                    return "Malicious: high success + high delay"
                elif mal_type == 4:
                    return "Malicious: classic low success"
                else:
                    return "Normal"
            
            # Ability type helper (accounts for malicious flag/type)
            def get_ability_type(rep, success_rate, is_malicious, mal_type):
                if is_malicious:
                    if rep < 4500:
                        return f"Blatantly malicious (T{mal_type})"
                    elif rep < 6600:
                        return f"Low-tier malicious (T{mal_type})"
                    elif rep < 8200:
                        return f"Mid-tier malicious (T{mal_type})"
                    else:
                        return f"High-tier malicious (T{mal_type})"
                
                # Ability categories for non-malicious NAs
                if rep < 4500:  # Very-low group
                    if success_rate >= 0.75:
                        return "Hidden strong"
                    elif success_rate >= 0.55:
                        return "Medium skill"
                    else:
                        return "Truly low-skill"
                elif rep < 6600:  # Low group
                    if success_rate >= 0.80:
                        return "High-potential"
                    elif success_rate >= 0.60:
                        return "Medium skill"
                    else:
                        return "Lower-medium skill"
                elif rep < 8200:  # High group
                    if success_rate >= 0.75:
                        return "High skill"
                    elif success_rate >= 0.60:
                        return "Medium skill"
                    else:
                        return "Unexpectedly low skill"
                else:  # Very-high group
                    if success_rate >= 0.85:
                        return "Very high skill"
                    elif success_rate >= 0.75:
                        return "High skill"
                    else:
                        return "Medium skill"
            
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
            
            # Detailed statistics (including malicious NA stats)
            very_low_indices = np.where(self.fixed_initial_reputation < 4500)[0]
            low_indices = np.where((self.fixed_initial_reputation >= 4500) & (self.fixed_initial_reputation < 6600))[0]
            high_indices = np.where((self.fixed_initial_reputation >= 6600) & (self.fixed_initial_reputation < 8200))[0]
            very_high_indices = np.where(self.fixed_initial_reputation >= 8200)[0]
            
            print("\nNA distribution statistics (malicious-type breakdown):")
            
            # Malicious type counter
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
                    f"Very-low ({len(very_low_indices)}): malicious {vl_malicious} "
                    f"[T1:{vl_mal_types[1]}, T2:{vl_mal_types[2]}, T3:{vl_mal_types[3]}, T4:{vl_mal_types[4]}] | "
                    f"normal {vl_normal} (hidden strong {vl_high_ability}, medium {vl_med_ability}, low-skill {vl_low_ability})"
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
                    f"Low ({len(low_indices)}): malicious {l_malicious} "
                    f"[T1:{l_mal_types[1]}, T2:{l_mal_types[2]}, T3:{l_mal_types[3]}, T4:{l_mal_types[4]}] | "
                    f"normal {l_normal} (high-potential {l_high_ability}, medium {l_med_ability}, lower-medium {l_low_ability})"
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
                    f"High ({len(high_indices)}): malicious {h_malicious} | normal {h_normal} "
                    f"(high-skill {h_high_ability}, medium {h_med_ability}, unexpectedly-low {h_low_ability})"
                )
            
            # Very-high group stats
            if len(very_high_indices) > 0:
                vh_malicious = sum(1 for i in very_high_indices if self.fixed_malicious_flags[i])
                vh_normal = len(very_high_indices) - vh_malicious
                vh_very_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.85)
                vh_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and 0.75 <= self.fixed_initial_success_rate[i] < 0.85)
                vh_med_ability = vh_normal - vh_very_high_ability - vh_high_ability
                print(
                    f"Very-high ({len(very_high_indices)}): malicious {vh_malicious} | normal {vh_normal} "
                    f"(very-high-skill {vh_very_high_ability}, high-skill {vh_high_ability}, medium {vh_med_ability})"
                )
            
            # Overall malicious NA stats
            total_malicious = np.sum(self.fixed_malicious_flags)
            total_normal = n_na - total_malicious
            print(
                f"\nOverall malicious NA stats: malicious {total_malicious} ({total_malicious/n_na:.1%}), "
                f"normal {total_normal} ({total_normal/n_na:.1%})"
            )
            

        else:
            # Delay parameters also need initialization in non-fixed-NA mode
            self.ideal_delay = 100.0
            self.max_acceptable_delay = self.ideal_delay * 3
        
        # Initialize current parameters
        self.current_time = 30  # Start time point in minutes
        
        
        # Call reset to initialize all parameters
        self.reset()
    
    def calculate_delay_performance(self, delay_grade):
        """
        Calculate delay grade (larger delay means a larger grade).
        
        Args:
            delay_grade: delay grade in [0.0, 1.0]
            
        Returns:
            float: delay grade, where 0.0 is the lowest and 1.0 is the highest
        """
        # Return the input delay grade, clipped to [0.0, 1.0]
        return max(0.0, min(1.0, delay_grade))
    
    
    def reset(self):
        """
        Reset environment state with fixed initial reputations to reduce dynamic changes.
        """
        if self.use_fixed_nas:
            # Use fixed NA set; only reset dynamic state
            self.reputation = self.fixed_initial_reputation.copy()
            self.total_tx = self.fixed_initial_total_tx.copy()
            self.success_count = self.fixed_initial_success_count.copy()
            self.success_rate = self.fixed_initial_success_rate.copy()
            
            # Signature delay
            self.signature_delay = self.fixed_initial_signature_delay.copy()
            # Use fixed initial hunger (each NA has a preset initial hunger)
            self.hunger = self.fixed_initial_hunger.copy()
            # Record each NA's last-selected step (inferred from initial hunger)
            # Convert initial hunger to a corresponding "last-selected time" to simulate selection history
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)
            for i in range(self.n_na):
                # Infer last-selected step from hunger
                # hunger = log(1 + steps/20) / log(11); invert to solve for steps
                if self.hunger[i] > 0:
                    # steps = 20 * (11^hunger - 1)
                    estimated_steps = int(20 * (np.power(11, self.hunger[i]) - 1))
                    self.last_selected_time[i] = -estimated_steps
                else:
                    self.last_selected_time[i] = 0  # Hunger=0 means just selected
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
            # Randomly initialize hunger (0-50%) to simulate different initial load levels
            self.hunger = np.random.uniform(0.0, 0.5, self.n_na)
            # Record each NA's last-selected step (inferred from hunger)
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
        
        # Reset per-NA sliding-window queue system
        self.current_step_count = 0
        for na_id in range(self.n_na):
            self.na_window_queues[na_id].clear()
            self.na_current_pack[na_id] = {
                'transactions': [],
                'success_count': 0,
                'total_count': 0,
                'reputation_changes': [],
                'start_step': 0,
                'end_step': 0
            }

        # Record initial reputations at the start of the episode (fixed baseline reputations)
        self.episode_start_reputation = self.reputation.copy()
        # Reset last-step reputation record
        self.last_step_reputation = self.reputation.copy()

        return self.get_state()
    def get_state(self):
        # Remove the activity feature; keep 4 features (reputation, success_rate, signature_delay, hunger)
        raw_state = np.column_stack([
            self.reputation,
            self.success_rate,
            self.signature_delay,
            self.hunger
        ])

        # Normalize
        normalized_state = raw_state.copy()
        normalized_state[:, 0] = (self.reputation - 3000) / (10000 - 3000)  # Reputation normalization
        normalized_state[:, 1] = self.success_rate  # success_rate already in [0, 1]
        # Signature delay is already a delay grade in [0, 1]; no further normalization
        normalized_state[:, 2] = self.signature_delay
        normalized_state[:, 3] = self.hunger  # hunger already in [0, 1]
        return normalized_state

    def step(self, action_na_index, selected_mask=None):
        """
        Simplified environment step that processes a single NA.
        
        Args:
            action_na_index: int - selected NA index
            selected_mask: kept for compatibility (currently unused)
        
        Returns:
            next_state: updated state
            reward: reward for this single NA
            done: whether the episode is done (always False)
        """
        # Ensure the index is within bounds
        na_idx = max(0, min(int(action_na_index), self.n_na - 1))
        
        # Record reputation and success rate before the action
        old_reputation = self.reputation[na_idx]
        old_success_rate = self.success_rate[na_idx]
        
        # Update time
        self.current_time += 1
        # Update training step count
        self.training_step += 1
        self._update_hunger(na_idx)
        # Simulate delay grade of the current transaction (controlled fluctuation)
        base_delay_grade = self.signature_delay[na_idx]
        if self.enable_random_fluctuation:
            # Add ±0.05 random fluctuation (delay grade in 0.0-1.0)
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
        else:
            # Fixed mode: add a small fluctuation (±0.01) to simulate measurement error
            current_delay_grade = base_delay_grade + np.random.uniform(-0.01, 0.01)
        current_delay_grade = np.clip(current_delay_grade, 0.0, 1.0)  # Clip to [0.0, 1.0]
        
        # Success-rate fluctuation (if enabled)
        effective_success_rate = self.success_rate[na_idx]
        if self.enable_random_fluctuation:
            # Success rate fluctuates by ±5% and stays within [0, 1]
            fluctuation = np.random.uniform(-0.05, 0.05)
            effective_success_rate = np.clip(effective_success_rate + fluctuation, 0.0, 1.0)
        
        # Determine transaction success based on (possibly fluctuated) success rate
        success = np.random.random() < effective_success_rate
        
        # Compute reward-related terms (used for reputation update)
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

        # Reward is based on NA quality rather than current transaction outcome
        #
        # Before: reward = success ? +1.0 : -1.0 + delay adjustment
        # After:  reward = 2.0 * weighted_success_rate - 1.0 + weighted-delay adjustment + hunger adjustment
        #
        # Benefits:
        # 1) More stable learning signal (less sensitive to single-transaction randomness)
        # 2) Reflects true NA capability (based on weighted historical success rate)
        # 3) Smooth reward gradient (weighted success rate in [0,1] -> reward linearly from -1 to +1)
        # 4) Less noise (one failure won't give a strong negative reward to a high-quality NA)
        # 5) Load balancing: a high-hunger, high-quality NA gets extra reward
        # 6) Uses sliding-window weighted success rate and delay grade
        
        # Get per-NA sliding-window summary (includes weighting)
        window_summary = self.get_na_window_summary(na_idx)
        
        # Use weighted success rate instead of raw success rate
        # Strategy:
        # 1) If the window has history, use weighted success rate (recent data gets higher weight)
        # 2) If the window is empty, use the current success rate (latest statistics)
        # 3) With no data, weighted success rate is 0.0, which yields negative reward and encourages exploration
        if window_summary['total_transactions'] > 0:
            effective_success_rate = window_summary['weighted_success_rate']
            effective_delay_performance = window_summary['weighted_delay_grade']
        else:
            # Fallback when no historical window data exists
            effective_success_rate = self.success_rate[na_idx]
            effective_delay_performance = self.calculate_delay_performance(current_delay_grade)

        if window_summary['total_transactions'] < 5:
            if self.total_tx[na_idx] > 0:
                overall_success_rate = self.success_count[na_idx] / self.total_tx[na_idx]
                blended_total = max(window_summary['total_transactions'], 0)
                smoothing_target = 5
                effective_success_rate = (
                    effective_success_rate * blended_total +
                    overall_success_rate * (smoothing_target - blended_total)
                ) / smoothing_target
        
        # Base reward from weighted success rate: widen reward range and raise positive threshold
        # Before: 2.5 * effective_success_rate - 1.5 (0 reward at success rate 0.6)
        # After:  4.0 * effective_success_rate - 2.8 (0 reward at success rate 0.7)
        weighted_success_rate_reward = 4.0 * effective_success_rate - 2.8  # Map [0,1] to [-2.8, 1.2]
        
        # Delay adjustment: based on weighted delay grade, widen reward range for delay performance
        # Delay grade in [0,1], 0 is best and 1 is worst
        # Before: 0.3 * (1 - 2 * effective_delay_performance), range [0.3, -0.3]
        # After:  0.8 * (1 - 2 * effective_delay_performance), range [0.8, -0.8]
        weighted_delay_grade_bonus = 0.8 * (1 - 2 * effective_delay_performance)  # Map [0,1] to [0.8, -0.8]
        
        # Hunger adjustment (controlled random fluctuation)
        # Design principles:
        # 1) High success + low delay + high hunger => extra reward (prefer good NAs not selected recently)
        # 2) Low success or high delay + high hunger => small/no reward (do not prioritize low-quality NAs)
        # 3) Hunger weight considers both weighted success rate and weighted delay performance
        
        # Hunger fluctuation (if enabled)
        effective_hunger = self.hunger[na_idx]
        if self.enable_random_fluctuation:
            # Hunger fluctuates by ±10% and stays within [0, 1]
            hunger_fluctuation = np.random.uniform(-0.1, 0.1)
            effective_hunger = np.clip(effective_hunger + hunger_fluctuation, 0.0, 1.0)
        
        # Composite quality weight: 60% weighted success + 40% weighted delay performance
        # Only NAs with both good success and delay can receive a significant hunger bonus
        # Delay performance: 0 is best and 1 is worst, so use (1 - delay)
        quality_weight = 0.6 * effective_success_rate + 0.4 * (1 - effective_delay_performance)
        
        # Dynamic hunger weight: higher quality => larger hunger weight
        # Use a smooth nonlinear function to amplify high-quality NAs
        # Base weight range: 0.2 to 1.5, using power to enhance nonlinearity
        base_hunger_weight = 0.2 + 1.3 * (quality_weight ** 1.5)  # Use power 1.5 to strengthen nonlinearity
        
        # Further adjust by quality band for clearer separation
        if quality_weight >= 0.8:
            # Top-quality NAs: highest hunger weight with a higher cap
            dynamic_hunger_weight = min(1.8, base_hunger_weight * 1.2)  # Cap at 1.8
        elif quality_weight >= 0.6:
            # High-quality NAs: standard weight
            dynamic_hunger_weight = base_hunger_weight * 1.0  # Standard
        elif quality_weight >= 0.4:
            # Medium-quality NAs: reduce weight by 20%
            dynamic_hunger_weight = base_hunger_weight * 0.8  # -20%
        else:
            # Low-quality NAs: reduce weight by 50%
            dynamic_hunger_weight = base_hunger_weight * 0.5  # -50%
        
        # Compute hunger bonus
        hunger_bonus = dynamic_hunger_weight * quality_weight * effective_hunger
        
        # Final reward: weighted success rate + weighted delay grade + hunger bonus
        reward = weighted_success_rate_reward + weighted_delay_grade_bonus + hunger_bonus
        
        # Record transaction behavior into the per-NA sliding-window queue
        transaction_data = {
            'success': success,
            'delay': current_delay_grade,
            'reputation_before': old_reputation,
            'reputation_after': self.reputation[na_idx],
            'success_rate_before': old_success_rate,
            'success_rate_after': self.success_rate[na_idx],
            'hunger': self.hunger[na_idx],
            'step': self.training_step,
            'effective_success_rate': effective_success_rate,
            'effective_delay_performance': effective_delay_performance,
            'computed_reward': computed_reward,
            'quality_weight': quality_weight,
            'dynamic_hunger_weight': dynamic_hunger_weight,
            'hunger_bonus': hunger_bonus,
            'final_reward': reward,
            'weighted_success_rate_reward': effective_success_rate,
            'weighted_delay_grade_bonus': effective_delay_performance
        }
        self._update_na_window_queue(na_idx, transaction_data)
        
        # Check whether to pack window queues
        self.current_step_count += 1
        if self.current_step_count % self.window_pack_interval == 0:
            self._pack_all_na_windows()
        
        # Update hunger after reward calculation (reward uses the pre-update hunger level)
        self._update_hunger(na_idx)
        
        return self.get_state(), reward, False

    def _update_na_window_queue(self, na_idx, transaction_data):
        """
        Update the current transaction pack for a given NA by accumulating transaction data.
        
        Args:
            na_idx: NA index
            transaction_data: dict containing transaction info
        """
        current_pack = self.na_current_pack[na_idx]
        
        # If this is the start of a new pack, record the start step
        if len(current_pack['transactions']) == 0:
            current_pack['start_step'] = self.current_step_count
        
        # Add transaction to the current pack
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
        Pack each NA's current transaction pack into its corresponding queue.
        """
        for na_idx in range(self.n_na):
            current_pack = self.na_current_pack[na_idx]
            
            # Pack only when there are transactions
            if current_pack['total_count'] > 0:
                # Compute average delay within this pack
                avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions']) if current_pack['transactions'] else 0.0
                
                # Compute delay grade
                delay_grade = self.calculate_delay_performance(avg_delay)
                
                # Compute pack summary statistics
                pack_summary = {
                    'start_step': current_pack['start_step'],
                    'end_step': current_pack['end_step'],
                    'transaction_count': current_pack['total_count'],
                    'success_count': current_pack['success_count'],
                    'success_rate': current_pack['success_count'] / current_pack['total_count'],
                    'total_reputation_change': sum(current_pack['reputation_changes']),
                    'avg_reputation_change': sum(current_pack['reputation_changes']) / len(current_pack['reputation_changes']) if current_pack['reputation_changes'] else 0,
                    'avg_delay': avg_delay,
                    'delay_grade': delay_grade,
                    'transactions': current_pack['transactions'].copy()
                }
                
                # Append to queue (capacity limit is handled by deque)
                self.na_window_queues[na_idx].append(pack_summary)
                
                # Reset the current pack
                current_pack['transactions'] = []
                current_pack['success_count'] = 0
                current_pack['total_count'] = 0
                current_pack['reputation_changes'] = []
                current_pack['start_step'] = 0
                current_pack['end_step'] = 0
    
    def get_na_window_summary(self, na_idx):
        """
        Get a statistical summary of the sliding-window queue for a given NA.
        
        Args:
            na_idx: NA index
            
        Returns:
            dict: summary stats of transaction behavior within the window queue
        """
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
                'weighted_success_rate': 0.0,
                'weighted_avg_delay': 0.0,
                'weighted_delay_grade': 0.0,
                'delay_grade': "N/A",
                'step_range': None
            }
        
        # Aggregate stats across all packs in the queue
        total_transactions = sum(pack['transaction_count'] for pack in queue)
        total_success_count = sum(pack['success_count'] for pack in queue)
        total_reputation_change = sum(pack['total_reputation_change'] for pack in queue)
        
        # Include the currently accumulating pack
        total_transactions += current_pack['total_count']
        total_success_count += current_pack['success_count']
        total_reputation_change += sum(current_pack['reputation_changes'])
        
        # Compute overall success rate
        overall_success_rate = total_success_count / total_transactions if total_transactions > 0 else 0.0
        
        # Compute average pack success rate
        pack_success_rates = [pack['success_rate'] for pack in queue if pack['transaction_count'] > 0]
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            pack_success_rates.append(current_pack_success_rate)
        
        avg_pack_success_rate = sum(pack_success_rates) / len(pack_success_rates) if pack_success_rates else 0.0
        
        # Compute weighted success rate and weighted delay (newer packs get higher weight)
        weighted_success_rate = 0.0
        weighted_avg_delay = 0.0
        weighted_delay_grade = 0.0
        total_weight = 0.0
        
        # Weights for queue packs (newer packs get higher weight)
        if queue:
            for i, pack in enumerate(queue):
                # Weight: linearly increasing, newest pack has the highest weight
                weight = i + 1  # Weights start at 1
                
                # Compute average delay for this pack
                pack_avg_delay = 0.0
                if pack['transactions']:
                    pack_avg_delay = sum(tx['delay'] for tx in pack['transactions']) / len(pack['transactions'])
                
                # Use the delay_grade field in the pack (if present)
                pack_delay_grade = pack.get('delay_grade', 0.0)
                
                weighted_success_rate += pack['success_rate'] * weight
                weighted_avg_delay += pack_avg_delay * weight
                weighted_delay_grade += pack_delay_grade * weight
                total_weight += weight
        
        # If there is a currently accumulating pack, include it with the highest weight
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            
            # Compute average delay for the current pack
            current_pack_avg_delay = 0.0
            if current_pack['transactions']:
                current_pack_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions'])
            
            # Compute delay grade for the current pack
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
        
        # If no weight was accumulated, fall back to a default delay grade
        if total_weight == 0:
            # With no historical data, use a medium delay grade
            weighted_delay_grade = 0.5
        
        # Letter grade based on delay grade
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
            'weighted_delay_grade': weighted_delay_grade,
            'delay_grade': delay_grade,
            'step_range': step_range,
            'pack_details': [
                {
                    'step_range': (pack['start_step'], pack['end_step']),
                    'transaction_count': pack['transaction_count'],
                    'success_rate': pack['success_rate'],
                    'reputation_change': pack['total_reputation_change'],
                    'weight': i + 1,
                    'avg_delay': sum(tx['delay'] for tx in pack['transactions']) / len(pack['transactions']) if pack['transactions'] else 0.0
                } for i, pack in enumerate(queue)
            ]
        }
    
    def print_sliding_window_summary(self, max_nas_to_show=10):
        """
        Print a sliding-window queue summary.
        
        Args:
            max_nas_to_show: max number of NAs to show
        """
        print(
            f"\nSliding-window queue summary (pack interval: {self.window_pack_interval} steps, "
            f"queue capacity: {self.window_queue_size})"
        )
        print("=" * 140)
        
        summaries = self.get_all_nas_window_summary()
        
        # Sort by weighted success rate (show better-performing NAs first)
        sorted_nas = sorted(summaries.items(), key=lambda x: x[1]['weighted_success_rate'], reverse=True)
        
        print(
            f"{'NA':<6} {'Queue':<6} {'CurPack':<8} {'TxTotal':<8} {'W_Success':<12} "
            f"{'W_Delay':<10} {'Grade':<8} {'RepDelta':<10} {'StepRange':<15}"
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
        
        # Aggregate overall stats
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
        print(
            f"Overall: {active_nas} NAs have transactions, total tx: {total_transactions}, "
            f"total successes: {total_success}"
        )
        if total_transactions > 0:
            print(
                f"System overall success rate: {total_success/total_transactions:.3f}, "
                f"system weighted success rate: {system_weighted_success_rate:.3f}"
            )
            print(
                f"System weighted avg delay: {system_weighted_delay:.1%}, "
                f"avg per active NA: {total_transactions/max(1, active_nas):.1f} tx"
            )
        print(f"Current step: {self.current_step_count}")
        
        if shown_count < active_nas:
            print(f"(Showing {shown_count} NAs; {active_nas - shown_count} more have transactions)")

    def show_na_detailed_queue(self, na_idx, max_packs_to_show=5):
        """
        Show detailed queue information for a given NA (including weight calculation).
        
        Args:
            na_idx: NA index
            max_packs_to_show: max number of packs to show
        """
        if na_idx >= self.n_na:
            print(f"Invalid NA index: {na_idx}")
            return
        
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\nNA{na_idx} detailed queue info")
        print("=" * 100)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("No transaction records for this NA")
            return
        
        # Show packs in the queue (weights increase with recency)
        print(f"Packs in queue (total {len(queue)}, by enqueue order; newer means higher weight):")
        print(
            f"{'Pack':<8} {'Weight':<6} {'StepRange':<15} {'Tx':<8} {'Success':<10} "
            f"{'AvgDelay':<10} {'RepDelta':<10}"
        )
        print("-" * 100)
        
        shown_packs = 0
        for i, pack in enumerate(queue):
            if shown_packs >= max_packs_to_show:
                break
            
            weight = i + 1
            step_range = f"({pack['start_step']}-{pack['end_step']})"
            avg_delay = pack.get('avg_delay', 0.0)
            
            print(
                f"Pack{i+1:<4} {weight:<6} {step_range:<15} {pack['transaction_count']:<8} "
                f"{pack['success_rate']:<10.3f} {avg_delay:<10.1%} {pack['total_reputation_change']:<10.1f}"
            )
            shown_packs += 1
        
        # Show the currently accumulating pack
        if current_pack['total_count'] > 0:
            current_weight = len(queue) + 1
            current_success_rate = current_pack['success_count'] / current_pack['total_count']
            current_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions']) if current_pack['transactions'] else 0.0
            current_step_range = f"({current_pack['start_step']}-{current_pack['end_step']})"
            current_reputation_change = sum(current_pack['reputation_changes'])
            
            print(
                f"CurPack {current_weight:<6} {current_step_range:<15} {current_pack['total_count']:<8} "
                f"{current_success_rate:<10.3f} {current_avg_delay:<10.1%} {current_reputation_change:<10.1f}"
            )
        
        # Weighted calculation result
        summary = self.get_na_window_summary(na_idx)
        print("\nWeighted result:")
        print(f"  - Weighted success rate: {summary['weighted_success_rate']:.3f}")
        print(f"  - Weighted avg delay: {summary['weighted_avg_delay']:.1%}")
        print(f"  - Delay grade: {summary['delay_grade']}")
        print(f"  - Total transactions: {summary['total_transactions']}")
        print(f"  - Queue usage: {len(queue)}/{self.window_queue_size}")
        
        if len(queue) < max_packs_to_show and len(queue) > shown_packs:
            print(f"\n({len(queue) - shown_packs} more packs not shown)")

    def get_all_nas_window_summary(self):
        """
        Get sliding-window summaries for all NAs.
        
        Returns:
            dict: summary dict keyed by NA index
        """
        summaries = {}
        for na_idx in range(self.n_na):
            summaries[na_idx] = self.get_na_window_summary(na_idx)
        return summaries
    
    def print_na_queue_details(self, na_idx, max_packs_to_show=5):
        """
        Print detailed queue info for a given NA.
        
        Args:
            na_idx: NA index
            max_packs_to_show: max number of packs to show
        """
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\nNA{na_idx} detailed queue info:")
        print("=" * 80)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("  No transaction records for this NA")
            return
        
        # Show packs in the queue
        if len(queue) > 0:
            print(f"Packed data (queue size: {len(queue)}/{self.window_queue_size}):")
            
            # Show the most recent packs
            recent_packs = list(queue)[-max_packs_to_show:]
            for i, pack in enumerate(recent_packs):
                pack_idx = len(queue) - max_packs_to_show + i
                if pack_idx < 0:
                    pack_idx = i
                
                print(
                    f"  Pack#{pack_idx+1}: steps {pack['start_step']}-{pack['end_step']}, "
                    f"tx {pack['transaction_count']}, success {pack['success_rate']:.2f}, "
                    f"rep delta {pack['total_reputation_change']:+.2f}"
                )
            
            if len(queue) > max_packs_to_show:
                print(f"  ... ({len(queue) - max_packs_to_show} earlier packs)")
        
        # Show the currently accumulating pack
        if current_pack['total_count'] > 0:
            print("\nCurrent accumulating pack:")
            print(
                f"  steps {current_pack['start_step']}-{current_pack['end_step']}, "
                f"tx {current_pack['total_count']}, "
                f"success {current_pack['success_count']} "
                f"({current_pack['success_count']/current_pack['total_count']:.2f}), "
                f"rep delta {sum(current_pack['reputation_changes']):+.2f}"
            )
        else:
            print("\nCurrent accumulating pack: empty")
        
        # Summary
        summary = self.get_na_window_summary(na_idx)
        print("\nSummary:")
        print(f"  Total transactions: {summary['total_transactions']}")
        print(f"  Total successes: {summary['total_success_count']}")
        print(f"  Overall success rate: {summary['overall_success_rate']:.2f}")
        print(f"  Total reputation change: {summary['total_reputation_change']:+.2f}")
        if summary['step_range']:
            print(f"  Step range: {summary['step_range'][0]} - {summary['step_range'][1]}")
        print("=" * 80)

    def _update_hunger(self, selected_na_idx):
        """
        Update hunger values for all NAs.
        
        Logic:
        1) Reset the selected NA's hunger to 0
        2) Increase hunger for other NAs over time
        3) Keep hunger within [0, 1]
        
        Args:
            selected_na_idx: currently selected NA index
        """
        # Update last-selected record for the selected NA
        self.last_selected_time[selected_na_idx] = self.training_step

        # Compute hunger for all NAs
        for i in range(self.n_na):
            steps_since_selected = self.training_step - self.last_selected_time[i]

            if steps_since_selected <= 0:
                self.hunger[i] = 0.0
            else:
                normalized_steps = steps_since_selected / self.hunger_growth_scale
                self.hunger[i] = min(1.0, np.log(1 + normalized_steps) / np.log(self.hunger_growth_log_base))

        # Reset hunger for the selected NA
        self.hunger[selected_na_idx] = 0.0

def simulate_selected_nas_training(env, selected_nas, title, steps=100):
    """
    Simulate transactions for selected NAs (reputation change only, no reward).
    
    Args:
        env: environment object
        selected_nas: list of selected NA indices
        title: title
        steps: number of simulated steps
    
    Returns:
        dict: transaction trajectory for each NA
    """
    print(f"\n{title} - NA transaction simulation ({steps} transactions):")
    print("=" * 80)
    
    # Save the current environment state (post-training state)
    original_reputation = env.reputation.copy()
    original_success_rate = env.success_rate.copy()
    
    original_current_time = env.current_time
    
    original_total_tx = env.total_tx.copy()
    original_success_count = env.success_count.copy()
    
    # Create transaction trajectory records for selected NAs
    training_results = {}
    
    for na_idx in selected_nas:
        # Reset environment to the initial fixed state (use initial parameters)
        env.reputation = env.fixed_initial_reputation.copy()
        env.success_rate = env.fixed_initial_success_rate.copy()
        
        env.current_time = 0  # Reset to initial time
        
        env.total_tx = env.fixed_initial_total_tx.copy()
        env.success_count = env.fixed_initial_success_count.copy()
        
        initial_rep = env.reputation[na_idx]
        initial_success_rate = env.success_rate[na_idx]
        initial_hunger = env.hunger[na_idx]
        initial_signature_delay = env.signature_delay[na_idx]
        
        reputation_history = [initial_rep]
        transaction_results = []  # Detailed per-transaction results
        
        print(f"\nNA {na_idx} transaction trajectory:")
        print(
            f"   Initial: reputation={initial_rep:.2f}, success_rate={initial_success_rate:.3f}, "
            f"signature_delay={initial_signature_delay:.1f}ms, hunger={initial_hunger:.3f}"
        )
        
        # Simulate transaction processing
        for step in range(steps):
            # Record pre-transaction state
            old_reputation = env.reputation[na_idx]
            old_success_rate = env.success_rate[na_idx]
            old_hunger = env.hunger[na_idx]
            
            env.current_time += 1
            
            
            # Simulate signature delay grade for the current transaction (add random jitter)
            base_delay_grade = env.signature_delay[na_idx]
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
            current_delay_grade = max(0.0, min(1.0, current_delay_grade))  # Clamp to [0.0, 1.0]
            
            # Determine whether the transaction succeeds
            if current_delay_grade > env.max_acceptable_delay_grade:
                success = False
                failure_reason = "Signature delay grade too high"
            else:
                # Use the original success rate directly (no delay-performance adjustment)
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
        
        # Compute overall statistics
        final_rep = reputation_history[-1]
        total_rep_change = final_rep - initial_rep
        successful_transactions = sum(1 for t in transaction_results if t['success'])
        transaction_success_rate = successful_transactions / steps
        avg_signature_delay = np.mean([t['signature_delay'] for t in transaction_results])
        final_success_rate = env.success_rate[na_idx]
        final_hunger = env.hunger[na_idx]
        
        # Count failure reasons
        delay_failures = sum(1 for t in transaction_results if t['failure_reason'] == 'Signature delay grade too high')
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
        
        print(f"NA {na_idx} transaction summary:")
        print(f"   Reputation change: {initial_rep:.2f} → {final_rep:.2f} ({total_rep_change:+.2f})")
        print(f"   Success rate change: {initial_success_rate:.3f} → {final_success_rate:.3f}")
        print(f"   Transaction success rate: {successful_transactions}/{steps} ({transaction_success_rate:.1%})")
        print(f"   Avg signature delay grade: {avg_signature_delay:.1%}")
        print(f"   Failures: delay grade too high {delay_failures}, random {random_failures}")
        print(f"   Hunger change: {initial_hunger:.3f} → {final_hunger:.3f}")
    
    # Restore the original environment state
    env.reputation = original_reputation
    env.success_rate = original_success_rate
    
    env.current_time = original_current_time
    
    env.total_tx = original_total_tx
    env.success_count = original_success_count
    
    return training_results

def create_policy_analysis_plots(agent, env):
    """
    Create policy analysis plots:
    1. Policy distribution plot
    2. Policy entropy curve
    3. NA selection frequency heatmap
    """
    print("\n" + "="*80)
    print("Create policy analysis plots")
    print("="*80)
    
    # Set font parameters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create 3x1 subplots with increased height
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))  # Increase width and height
    
    # 1. Policy distribution plot - action probability distribution of the final policy
    ax1 = axes[0]
    
    if len(agent.action_distribution_history) > 0:
        # Average over the most recent policy distributions
        recent_distributions = agent.action_distribution_history[-10:]  # Last 10
        avg_distribution = np.mean(recent_distributions, axis=0)
        
        # Create NA indices
        na_indices = np.arange(len(avg_distribution))
        
        # Color by initial reputation group
        colors = []
        for i in range(env.n_na):
            if env.episode_start_reputation[i] < 6600:
                colors.append('lightcoral')  # Low reputation group - red
            else:
                colors.append('lightblue')   # High reputation group - blue
        
        bars = ax1.bar(na_indices, avg_distribution, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Mark NAs with the highest selection probabilities
        top_5_indices = np.argsort(avg_distribution)[-5:]
        for idx in top_5_indices:
            ax1.annotate(f'NA{idx}\n{avg_distribution[idx]:.3f}', 
                        xy=(idx, avg_distribution[idx]), 
                        xytext=(0, 15), textcoords='offset points',
                        ha='center', fontsize=9, fontweight='bold')
        
        ax1.set_title('Policy Distribution (Final Strategy)\nAverage Action Probabilities in Recent Episodes', 
                     fontsize=12, fontweight='bold', pad=20)
        ax1.set_xlabel('NA Index')
        ax1.set_ylabel('Selection Probability')
        
        # Show all NA indices on x-axis
        ax1.set_xticks(na_indices)
        ax1.set_xticklabels([f'NA{i}' for i in na_indices], rotation=45, ha='right')
        
        ax1.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightcoral', label='Low Reputation Group (<6600)'),
                          Patch(facecolor='lightblue', label='High Reputation Group (≥6600)')]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        print("Policy distribution plot completed (average over last 10 episodes)")
        print(f"   Top-5 NA selection probabilities: {[f'NA{i}({avg_distribution[i]:.3f})' for i in top_5_indices]}")
    else:
        ax1.text(0.5, 0.5, 'No Policy Distribution Data Available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Policy Distribution - No Data')
    
    # 2. Policy entropy curve
    ax2 = axes[1]
    
    if len(agent.policy_entropy_history) > 0:
        episodes = np.arange(len(agent.policy_entropy_history))
        entropies = agent.policy_entropy_history
        
        # Plot entropy
        ax2.plot(episodes, entropies, 'b-', linewidth=2, alpha=0.8, label='Policy Entropy')
        
        # Trend line
        if len(entropies) > 10:
            z = np.polyfit(episodes, entropies, 1)
            p = np.poly1d(z)
            ax2.plot(episodes, p(episodes), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
        
        # Key points
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
        
        # Analyze entropy trend
        if len(entropies) > 1:
            entropy_change = entropies[-1] - entropies[0]
            if entropy_change < -0.1:
                trend_text = "OK: policy becomes more deterministic (entropy decreases)"
            elif entropy_change > 0.1:
                trend_text = "WARN: policy becomes more random (entropy increases)"
            else:
                trend_text = "OK: policy is relatively stable"
        else:
            trend_text = "INFO: insufficient data"
            
        print(f"Policy entropy plot completed - {trend_text}")
        print(f"   Entropy change: {entropies[0]:.3f} → {entropies[-1]:.3f} (Δ{entropy_change:.3f})")
    else:
        ax2.text(0.5, 0.5, 'No Policy Entropy Data Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Policy Entropy - No Data')
    
    # 3. NA selection frequency heatmap
    ax3 = axes[2]
    
    if np.sum(agent.na_selection_frequency) > 0:
        # Normalize selection frequency to percentage
        selection_freq_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        
        # Create heatmap data matrix
        n_cols = min(10, env.n_na)  # Max 10 NAs per row
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
        
        # Cell labels
        for i in range(n_rows):
            for j in range(n_cols):
                na_idx = i * n_cols + j
                if na_idx < env.n_na:
                    text_color = 'white' if heatmap_data[i, j] > np.max(heatmap_data) * 0.5 else 'black'
                    ax3.text(j, i, f'NA{na_idx}\n{heatmap_data[i, j]:.1f}%',
                            ha='center', va='center', color=text_color, fontsize=8)
        
        # Axes
        ax3.set_xticks(range(n_cols))
        ax3.set_yticks(range(n_rows))
        ax3.set_xticklabels([f'Col{i}' for i in range(n_cols)])
        ax3.set_yticklabels([f'Row{i}' for i in range(n_rows)])
        
        ax3.set_title('NA Selection Frequency Heatmap\n(Percentage of times each NA was selected)', 
                     fontsize=12, fontweight='bold')
        
        # Analyze selection frequency
        most_selected = np.argmax(selection_freq_percent)
        least_selected = np.argmin(selection_freq_percent[selection_freq_percent > 0])
        
        print("NA selection frequency heatmap completed")
        print(f"   Most selected: NA{most_selected} ({selection_freq_percent[most_selected]:.1f}%)")
        print(f"   Least selected: NA{least_selected} ({selection_freq_percent[least_selected]:.1f}%)")
        
        # Check for over-concentration on a few NAs
        top_3_percent = np.sum(np.sort(selection_freq_percent)[-3:])
        if top_3_percent > 70:
            print(f"   WARN: top-3 NAs account for {top_3_percent:.1f}% selections (possible overfitting)")
        else:
            print(f"   OK: selection distribution looks reasonable (top-3 = {top_3_percent:.1f}%)")
    else:
        ax3.text(0.5, 0.5, 'No Selection Frequency Data Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('NA Selection Frequency - No Data')
    
    plt.tight_layout(pad=4.0)
    
    # Save plots
    filename = f'/mnt/data/wy2024/src/policy_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\nPolicy analysis plots saved: {filename}")
    print("="*80)
    
    # Detailed policy analysis report
    print("\nPolicy analysis report:")
    
    if len(agent.policy_entropy_history) > 0:
        print("1. Policy entropy:")
        print(f"   - Initial entropy: {agent.policy_entropy_history[0]:.3f}")
        print(f"   - Final entropy: {agent.policy_entropy_history[-1]:.3f}")
        print(f"   - Max entropy: {np.log(env.n_na):.3f} (fully random)")
        print(f"   - Determinism: {(1 - agent.policy_entropy_history[-1]/np.log(env.n_na))*100:.1f}%")
    
    if np.sum(agent.na_selection_frequency) > 0:
        print("\n2. NA selection:")
        selection_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        selected_nas = np.where(selection_percent > 0)[0]
        print(f"   - Selected NA count: {len(selected_nas)}/{env.n_na}")
        print(f"   - Selection diversity: {len(selected_nas)/env.n_na*100:.1f}%")
        
        # Grouped by reputation threshold
        low_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                   if env.episode_start_reputation[i] < 6600])
        high_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                    if env.episode_start_reputation[i] >= 6600])
        print(f"   - Low-reputation group rate: {low_rep_selections:.1f}%")
        print(f"   - High-reputation group rate: {high_rep_selections:.1f}%")
    
    if len(agent.action_distribution_history) > 0:
        print("\n3. Action distribution:")
        recent_dist = np.mean(agent.action_distribution_history[-10:], axis=0)
        top_5_nas = np.argsort(recent_dist)[-5:]
        print(f"   - Top-5 preferred NAs: {[f'NA{i}({recent_dist[i]:.2f})' for i in top_5_nas]}")
        concentration = np.sum(np.sort(recent_dist)[-5:])
        print(f"   - Top-5 concentration: {concentration:.1%}")
        
        if concentration > 0.8:
            print("   - WARN: policy is highly concentrated; may be overfitting")
        elif concentration < 0.3:
            print("   - WARN: policy is too diffuse; may be under-trained")
        else:
            print("   - OK: policy concentration looks reasonable")

def train_dqn(n_na, n_episodes, steps_per_episode, lr, 
              update_frequency, use_fixed_nas, enable_random_fluctuation,
              generate_episode_gifs=False):  # Keep defaults minimal/explicit
    
    # Parameter checks and sanity validation
    print("Training parameter check:")
    print(f"  NA count: {n_na}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Learning rate: {lr}")
    print(f"  Update frequency: every {update_frequency} steps")
    print(f"  Use fixed NA pool: {use_fixed_nas}")
    print(f"  Random fluctuation mode: {enable_random_fluctuation}")
    
    # Validate parameter ranges
    total_steps = n_episodes * steps_per_episode
    print(f"  Total training steps: {total_steps}")
    
    if n_na < 5:
        print("WARN: NA count is low; this may hurt learning quality")
    if n_na > 30:
        print("WARN: NA count is high; you may need more training steps")
    if lr > 0.01:
        print("WARN: learning rate is high; training may become unstable")
    if total_steps < 1000:
        print("WARN: total training steps are low; the network may be under-trained")
    
    # Create environment (optionally fixed NA pool)
    env = Environment(n_na, use_fixed_nas=use_fixed_nas, enable_random_fluctuation=enable_random_fluctuation)
    agent = DQNAgent(n_features=4, n_na=n_na, lr=lr, gamma=0.9,  # 4 features (activity removed); smaller gamma increases Q-value contrast
                    epsilon_start=1.0, epsilon_end=0.01, decay_steps=15000,
                    memory_size=50000, batch_size=256, target_update=20,
                    min_memory_size=2000, update_frequency=update_frequency, 
                    total_episodes=n_episodes)
    history = []
    # Per-step reputation trajectory for the last episode
    reputation_last = []
    
    # Detailed data for first/last episodes
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
    print("Training strategy: use a fixed NA pool; select a single NA each step.")
    print(f"Start training. Update frequency: train every {update_frequency} steps...")
    
    # Print the fixed NA characteristics for the first episode to confirm determinism
    print("\nFixed NA pool characteristics (constant throughout training):")
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
        # 🔧 在每个episode开始时更新epsilon（基于episode数的衰减）
        agent.update_epsilon_by_episode(ep)
        
        # 🔧 在每个episode开始时更新学习率（分阶段衰减）
        current_lr = agent.update_learning_rate(ep)
        
        # 记录学习率和epsilon历史
        if ep % 10 == 0:  # 每10个episode记录一次
            agent.lr_history.append(current_lr)
            agent.epsilon_history.append(agent.epsilon)
        
        # 每100个episode显示一次学习率变化
        if ep % 100 == 0 and ep > 0:
            print(f"📈 Episode {ep}: Learning Rate = {current_lr:.6f}")
        
        state = env.reset()  # 重置环境状态，但保持固定NA特征不变
        
        # 如果是第一个和最后一个episode, 打印该episode的NA参数确认
        if ep == 0:
            print(f"\n第一个Episode (#{ep}) 固定NA参数确认:")
            print("编号\treputation\tsuccess_rate\t签名延迟(ms)\t延迟等级\t分组")
            for i in range(min(n_na, 10)):  # 只打印前10个避免输出过长
                group = "低信誉组" if env.reputation[i] < 6600 else "高信誉组"
                delay_perf = env.calculate_delay_performance(env.signature_delay[i])
                print(f"{i}\t{env.reputation[i]:.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.3f}\t\t{group}")
            if n_na > 10:
                print(f"... (共{n_na}个NA)")
        elif ep == n_episodes - 1:
            print(f"\n最后一个Episode (#{ep}) NA参数 (应与初始相同):")
            print("编号\treputation\tsuccess_rate\t签名延迟(ms)\t延迟等级\t分组")
            for i in range(min(n_na, 10)):  # 只打印前10个避免输出过长
                group = "低信誉组" if env.reputation[i] < 6600 else "高信誉组"
                delay_perf = env.calculate_delay_performance(env.signature_delay[i])
                print(f"{i}\t{env.reputation[i]:.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{group}")
            if n_na > 10:
                print(f"... (共{n_na}个NA)")
        
        # 初始化本轮总奖励和统计变量
        ep_reward = 0
        ep_step_rewards = []  # 本episode的每步奖励
        ep_successes = 0  # 本episode的成功次数
        
        # 记录episode数据（第一个和最后一个episode）
        is_first_episode = (ep == 0)
        is_last_episode = (ep == n_episodes - 1)
        
        if is_first_episode or is_last_episode:
            episode_reputation_history = [env.reputation.copy()]  # 记录每步后的信誉值
            episode_selected_nas = []  # 记录每步选中的NA
        
        # 如果是最后一次episode，记录初始信誉
        if is_last_episode:
            reputation_last.append(env.reputation.copy())
        
        for t in range(steps_per_episode):
            # 修改：每次选择单个NA，在每10个episode记录策略分析数据
            record_policy = (ep % 10 == 0) or is_last_episode  # 每10个episode或最后一个episode记录
            action_na = agent.select_action(state, initial_reputations=env.episode_start_reputation, record_policy=record_policy)
            next_state, reward, done = env.step(action_na)
            agent.store(state, action_na, reward, next_state, done)
            
            total_training_steps += 1
            
            # 🔧 关键修改：使用条件更新而不是每步都训练
            batch_losses = agent.maybe_update(global_step=total_training_steps)
            if batch_losses is not None:
                total_updates += len(batch_losses)
                # 只在最后一个episode打印详细训练信息
                if ep == n_episodes - 1:
                    avg_loss = np.mean(batch_losses)
                    # 删除详细损失打印 - 减少输出冗余
                
            # 累加本轮奖励
            ep_reward += reward
            ep_step_rewards.append(reward)

            # 聚合若干步的奖励，减少曲线抖动
            step_reward_buffer.append(reward)
            if len(step_reward_buffer) >= reward_capture_interval:
                averaged_reward = sum(step_reward_buffer) / len(step_reward_buffer)
                step_rewards.append(averaged_reward)
                step_reward_steps.append(total_training_steps)
                step_reward_buffer.clear()
            
            # 记录成功次数（通过奖励正负判断）
            if reward > 0:
                ep_successes += 1
            
            state = next_state
            
            # 记录第一个和最后一个episode的详细数据
            if is_first_episode or is_last_episode:
                episode_reputation_history.append(env.reputation.copy())
                episode_selected_nas.append(action_na)
                  # 记录最后episode每步结束后的信誉
        if is_last_episode:
            reputation_last.append(env.reputation.copy())
        
        # 计算本episode的统计信息
        ep_success_rate = ep_successes / steps_per_episode if steps_per_episode > 0 else 0
        ep_avg_step_reward = np.mean(ep_step_rewards) if ep_step_rewards else 0
        ep_reward_std = np.std(ep_step_rewards) if ep_step_rewards else 0
        
        # 📈 计算Q值统计信息（每10个episode监测一次，减少开销）
        if ep % 10 == 0:  # 只在每10个episode计算Q值统计
            with torch.no_grad():
                current_state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                all_q_values = agent.policy_net(current_state).squeeze(0).cpu().numpy()
                q_mean = np.mean(all_q_values)
                q_max = np.max(all_q_values)
                q_min = np.min(all_q_values)
                q_std = np.std(all_q_values)
        else:
            # 使用上一次的Q值统计（减少计算开销）
            if ep > 0 and len(history) > 0:
                last_q_data = history[-1]
                q_mean = last_q_data.get('q_mean', 0)
                q_max = last_q_data.get('q_max', 0)
                q_min = last_q_data.get('q_min', 0)
                q_std = last_q_data.get('q_std', 0)
            else:
                q_mean = q_max = q_min = q_std = 0
        
        # 存储数据用于分析
        success_rates_per_episode.append(ep_success_rate)
        
        # 保存第一个和最后一个episode的数据
        if is_first_episode:
            first_episode_data = {
                'reputation_history': episode_reputation_history,
                'selected_nas': episode_selected_nas,
                'episode_num': ep + 1
            }
            # 🎯 在第一个episode结束时导出所有NA的参数
            try:
                print(f"\n第一个Episode结束，导出所有NA的参数...")
                export_file = agent.export_na_selection_parameters(
                    state=state, 
                    initial_reputations=env.episode_start_reputation,
                    env=env
                )
                print(f"✅ 第一个Episode NA参数导出成功！文件: {export_file}")
            except Exception as e:
                print(f"❌ 第一个Episode NA参数导出失败: {e}")
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
            memory_status = f"内存: {len(agent.memory)}/{agent.min_memory_size}"
            update_ratio = total_updates / max(1, total_training_steps) * 100
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Mean Reputation: {env.reputation.mean():.2f}, "
                  f"Success Rate: {ep_success_rate:.3f}, Avg Step Reward: {ep_avg_step_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}, {memory_status}, 更新率: {update_ratio:.1f}%")
        
        # 💡 每50个episode进行Q值学习质量检查
        if (ep + 1) % 50 == 0 and ep > 0:
            print(f"\n🔍 Episode {ep+1} - Q值学习质量检查:")
            q_values = agent.debug_q_values(state, env.episode_start_reputation)
            
            # 检查网络是否学到了合理的NA选择策略
            low_rep_indices = np.where(env.episode_start_reputation < 6600)[0]
            high_rep_indices = np.where(env.episode_start_reputation >= 6600)[0]
            
            if len(low_rep_indices) > 0 and len(high_rep_indices) > 0:
                # 计算top-5 Q值NA的分组情况
                top_5_indices = np.argsort(q_values.cpu().numpy())[-5:]
                top_5_low_count = np.sum([idx in low_rep_indices for idx in top_5_indices])
                top_5_high_count = np.sum([idx in high_rep_indices for idx in top_5_indices])
                
                print(f"  Top-5 Q值NA: 低信誉组{top_5_low_count}个, 高信誉组{top_5_high_count}个")
                
                # 评估学习质量
                if top_5_high_count >= 3:
                    print("  ✅ 学习质量良好：优先选择高信誉NA")
                elif top_5_high_count >= 2:
                    print("  ⚠️  学习质量一般：选择策略需改进")
                else:
                    print("  ❌ 学习质量差：可能需要调整参数")
            
            # 🔧 每100个episode显示滑动窗口统计
            if (ep + 1) % 100 == 0:
                env.print_sliding_window_summary(max_nas_to_show=10)
        
        # 🔧 在关键的epsilon转换点显示详细信息
        first_transition = int(0.15 * n_episodes)  # 第一个转换点：15%
        second_transition = int(0.5 * n_episodes)  # 第二个转换点：50%
        
        if ep == first_transition or ep == first_transition + 1:
            print(f"🔄 Episode {ep+1}: Epsilon第一转换点，ε = {agent.epsilon:.4f} "
                  f"({'极快衰减→快速衰减' if ep == first_transition else '快速衰减阶段'})")
        elif ep == second_transition or ep == second_transition + 1:
            print(f"🔄 Episode {ep+1}: Epsilon第二转换点，ε = {agent.epsilon:.4f} "
                  f"({'快速衰减→精细调优' if ep == second_transition else '精细调优阶段'})")
        elif ep in [n_episodes-10, n_episodes-5, n_episodes-1]:
            print(f"🎯 Episode {ep+1}: 接近训练结束，ε = {agent.epsilon:.4f}")
    
    # 处理未满聚合窗口的剩余奖励
    if step_reward_buffer:
        averaged_reward = sum(step_reward_buffer) / len(step_reward_buffer)
        step_rewards.append(averaged_reward)
        step_reward_steps.append(total_training_steps)

    # 🔧 确保记录最后一个episode的学习率和epsilon（如果最后一个episode不是10的倍数）
    final_episode = n_episodes - 1  # 最后一个episode的索引
    if final_episode % 10 != 0:  # 如果最后一个episode不是10的倍数
        current_lr = agent.update_learning_rate(final_episode)
        agent.lr_history.append(current_lr)
        agent.epsilon_history.append(agent.epsilon)
        print(f"📈 记录最后一个Episode {final_episode}: Learning Rate = {current_lr:.6f}, Epsilon = {agent.epsilon:.6f}")
    
    # 打印训练统计信息
    print(f"\n训练统计:")
    print(f"总训练步数: {total_training_steps}")
    print(f"总网络更新次数: {total_updates}")
    print(f"更新频率: 每{update_frequency}步尝试训练一次")
    print(f"实际更新率: {total_updates/max(1, total_training_steps)*100:.2f}%")
    print(f"目标网络更新次数: {agent.learn_step//agent.target_update}")
    print(f"最终Epsilon值: {agent.epsilon:.4f}")
    
    # 🔧 关键逻辑检查点
    print(f"\n🔍 训练完成后的关键逻辑检查:")
    
    # 1. 检查网络参数是否有更新
    final_state = env.get_state()
    with torch.no_grad():
        initial_q = torch.zeros(n_na)  # 假设的初始Q值
        final_q = agent.policy_net(torch.FloatTensor(final_state).unsqueeze(0).to(agent.device)).squeeze().cpu()
    
    q_magnitude = torch.norm(final_q).item()
    print(f"  Q值网络权重幅度: {q_magnitude:.4f}")
    
    if q_magnitude < 0.1:
        print("  ❌ 网络可能没有充分训练")
    elif q_magnitude > 100:
        print("  ⚠️  网络可能过拟合或学习率过高")
    else:
        print("  ✅ 网络权重幅度正常")
    
    # 2. 检查经验回放缓冲区状态
    print(f"  经验缓冲区利用率: {len(agent.memory)}/{agent.min_memory_size} "
          f"({len(agent.memory)/agent.min_memory_size*100:.1f}%)")
    
    # 3. 检查训练频率是否合理
    expected_updates = total_training_steps // update_frequency
    update_efficiency = total_updates / max(1, expected_updates) * 100
    print(f"  训练效率: {update_efficiency:.1f}% (期望更新{expected_updates}次，实际{total_updates}次)")
    
    # 4. 检查损失收敛情况
    if len(agent.loss_history) > 100:
        early_loss = np.mean(agent.loss_history[:50])
        late_loss = np.mean(agent.loss_history[-50:])
        loss_improvement = (early_loss - late_loss) / early_loss * 100
        print(f"  损失改善: {loss_improvement:.1f}% (早期:{early_loss:.4f} → 后期:{late_loss:.4f})")
        
        if loss_improvement > 20:
            print("  ✅ 损失收敛良好")
        elif loss_improvement > 5:
            print("  ⚠️  损失改善有限，可能需要更多训练")
        else:
            print("  ❌ 损失没有明显改善，检查超参数设置")
    else:
        print("  ⚠️  训练数据不足，无法评估损失收敛")
    
    # 使用训练好的DQN进行最终选择
    print("\n" + "="*80)
    print("使用训练好的DQN网络进行最终NA选择")
    print("💡 训练方式：固定NA集合，每个step选择单个NA，最终组合通过多次选择形成")
    print("="*80)
    
    final_state = env.get_state()
    selection_result = agent.make_final_selection(final_state, env.episode_start_reputation)
    
    dqn_selected = selection_result['dqn_selected']
    random_selected = selection_result['random_selected']
    all_q_values = selection_result['all_q_values']
    low_rep_indices = selection_result['low_rep_indices']
    high_rep_indices = selection_result['high_rep_indices']
    
    print(f"\n当前Episode结束时的NA状态:")
    print("编号\t初始信誉\t当前信誉\t信誉变化\tsuccess_rate\t签名延迟(ms)\t延迟等级\tQ值\t\t分组")
    for i in range(min(n_na, 15)):  # 限制输出数量，避免过长
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "低初始信誉组" if initial_rep < 6600 else "高初始信誉组"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
    print(f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}")
    if n_na > 15:
        print(f"... (共{n_na}个NA，仅显示前15个)")
    
    print(f"\n[DQN] DQN通过多次单选组成的{len(dqn_selected)}个NA组合 (目标:2低+3高):")
    print("编号\t初始信誉\t当前信誉\t信誉变化\tsuccess_rate\t签名延迟(ms)\t延迟等级\tQ值\t\t分组\t\t\t恶意标识")
    dqn_low_count = 0
    dqn_high_count = 0
    dqn_total_reputation = 0
    dqn_total_q_value = 0
    dqn_malicious_count = 0  # 统计选中的恶意NA数量
    
    for i in dqn_selected:
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "低初始信誉组" if initial_rep < 6600 else "高初始信誉组"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        
        # 🚨 检查是否为恶意NA并获取类型
        is_malicious = env.fixed_malicious_flags[i]
        if is_malicious:
            dqn_malicious_count += 1
            mal_type = env.fixed_malicious_types[i]
            if mal_type == 1:
                malicious_info = "🚨恶意T1(高成功+高延迟)"
            elif mal_type == 4:
                malicious_info = "🚨恶意T4(传统低成功)"
            else:
                malicious_info = "🚨恶意(未知类型)"
                malicious_info = "🚨恶意(未知类型)"
        else:
            malicious_info = "✅正常"
            
        if initial_rep < 6600:
            dqn_low_count += 1
        else:
            dqn_high_count += 1
        dqn_total_reputation += current_rep
        dqn_total_q_value += all_q_values[i]
    print(f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}\t{malicious_info}")
    
    # 🎯 显示DQN选择的恶意NA统计
    if dqn_malicious_count > 0:
        print(f"⚠️  警告: DQN选择中包含 {dqn_malicious_count} 个恶意NA！这表明恶意NA成功伪装或DQN识别能力不足。")
    
    print(f"\n🎲 随机选择的{len(random_selected)}个NA组合 (作为对比):")
    print("编号\t初始信誉\t当前信誉\t信誉变化\tsuccess_rate\t签名延迟(ms)\t延迟等级\tQ值\t\t分组\t\t\t恶意标识")
    random_low_count = 0
    random_high_count = 0
    random_total_reputation = 0
    random_total_q_value = 0
    random_malicious_count = 0  # 统计随机选中的恶意NA数量
    
    for i in random_selected:
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "低初始信誉组" if initial_rep < 6600 else "高初始信誉组"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        
        # 🚨 检查是否为恶意NA并获取类型
        is_malicious = env.fixed_malicious_flags[i]
        if is_malicious:
            random_malicious_count += 1
            mal_type = env.fixed_malicious_types[i]
            if mal_type == 1:
                malicious_info = "🚨恶意T1(高成功+高延迟)"
            elif mal_type == 4:
                malicious_info = "🚨恶意T4(传统低成功)"
            else:
                malicious_info = "🚨恶意(未知类型)"
        else:
            malicious_info = "✅正常"
            
        if initial_rep < 6600:
            random_low_count += 1
        else:
            random_high_count += 1
        random_total_reputation += current_rep
        random_total_q_value += all_q_values[i]
    print(f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}\t{malicious_info}")
    
    # 🎯 显示随机选择的恶意NA统计
    if random_malicious_count > 0:
        print(f"⚠️  随机选择中包含 {random_malicious_count} 个恶意NA (这是预期的随机结果)。")
    
    # 🎯 模拟选中NA的完整训练过程
    dqn_training_results = simulate_selected_nas_training(env, dqn_selected, "DQN选择的NA训练模拟", steps=100)
    
    # 🔧 显示滑动窗口统计摘要
    env.print_sliding_window_summary(max_nas_to_show=15)
    random_training_results = simulate_selected_nas_training(env, random_selected, "随机选择的NA训练模拟", steps=100)
    
    print(f"\n选择策略对比分析 (完整训练模拟对比):")
    print(f"{'指标':<25} {'DQN选择':<15} {'随机选择':<15} {'DQN优势':<15}")
    print("-" * 70)
    print(f"{'低信誉组数量':<25} {dqn_low_count:<15} {random_low_count:<15} {'✓' if dqn_low_count >= 2 else '✗':<15}")
    print(f"{'高信誉组数量':<25} {dqn_high_count:<15} {random_high_count:<15} {'✓' if dqn_high_count >= 3 else '✗':<15}")
    print(f"{'🚨恶意NA数量':<25} {dqn_malicious_count:<15} {random_malicious_count:<15} {'⚠️ 更少' if dqn_malicious_count < random_malicious_count else ('❌ 相同' if dqn_malicious_count == random_malicious_count else '❌ 更多'):<15}")
    
    # 计算训练后的统计数据
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
    
    print(f"{'训练前平均信誉':<25} {dqn_avg_initial_rep:<15.2f} {random_avg_initial_rep:<15.2f} {dqn_avg_initial_rep - random_avg_initial_rep:+.2f}")
    print(f"{'训练后平均信誉':<25} {dqn_avg_final_rep:<15.2f} {random_avg_final_rep:<15.2f} {dqn_avg_final_rep - random_avg_final_rep:+.2f}")
    print(f"{'平均信誉提升':<25} {dqn_avg_rep_change:<15.2f} {random_avg_rep_change:<15.2f} {dqn_avg_rep_change - random_avg_rep_change:+.2f}")
    print(f"{'事务成功率':<25} {dqn_avg_transaction_success_rate:<15.1%} {random_avg_transaction_success_rate:<15.1%} {'+' if dqn_avg_transaction_success_rate > random_avg_transaction_success_rate else ''}{(dqn_avg_transaction_success_rate - random_avg_transaction_success_rate):.1%}")
    print(f"{'总成功事务数':<25} {dqn_total_successful_transactions:<15} {random_total_successful_transactions:<15} {dqn_total_successful_transactions - random_total_successful_transactions:+}")
    print(f"{'平均签名延迟(ms)':<25} {dqn_avg_signature_delay:<15.1f} {random_avg_signature_delay:<15.1f} {dqn_avg_signature_delay - random_avg_signature_delay:+.1f}")
    
    # 🎬 创建选择策略对比动图
    print(f"\n🎬 创建选择策略对比动图...")
    create_comparison_strategy_gif(env, dqn_selected, random_selected, dqn_training_results, random_training_results)
    
    dqn_avg_q_value = dqn_total_q_value / len(dqn_selected) if dqn_selected else 0
    random_avg_q_value = random_total_q_value / len(random_selected) if random_selected else 0
    print(f"{'平均Q值':<25} {dqn_avg_q_value:<15.4f} {random_avg_q_value:<15.4f} {dqn_avg_q_value - random_avg_q_value:+.4f}")
    
    print(f"\n💡 Q值分析 (固定NA集合):")
    if len(low_rep_indices) > 0:
        low_q_values = all_q_values[low_rep_indices]
        print(f"低信誉组Q值范围: {low_q_values.min():.4f} ~ {low_q_values.max():.4f} (平均: {low_q_values.mean():.4f})")
    if len(high_rep_indices) > 0:
        high_q_values = all_q_values[high_rep_indices]
        print(f"高信誉组Q值范围: {high_q_values.min():.4f} ~ {high_q_values.max():.4f} (平均: {high_q_values.mean():.4f})")
    
    # 🚀 新增：签名延迟分析
    print(f"\n⚡ 签名延迟性能分析:")
    all_delays = [env.signature_delay[i] for i in range(n_na)]
    all_delay_perfs = [env.calculate_delay_performance(delay) for delay in all_delays]
    
    print(f"签名延迟范围: {min(all_delays):.1f}ms ~ {max(all_delays):.1f}ms (平均: {np.mean(all_delays):.1f}ms)")
    print(f"延迟等级: {min(all_delay_perfs):.3f} ~ {max(all_delay_perfs):.3f} (平均: {np.mean(all_delay_perfs):.3f})")
    
    # 分组延迟分析
    if len(low_rep_indices) > 0:
        low_delays = [all_delays[i] for i in low_rep_indices]
        low_delay_perfs = [all_delay_perfs[i] for i in low_rep_indices]
        print(f"低信誉组延迟: {np.mean(low_delays):.1f}ms (等级: {np.mean(low_delay_perfs):.3f})")
    if len(high_rep_indices) > 0:
        high_delays = [all_delays[i] for i in high_rep_indices]
        high_delay_perfs = [all_delay_perfs[i] for i in high_rep_indices]
        print(f"高信誉组延迟: {np.mean(high_delays):.1f}ms (等级: {np.mean(high_delay_perfs):.3f})")
    
    # 延迟等级标准提醒
    print(f"延迟标准: 延迟等级范围0.0-1.0 (0.0=最快, 1.0=最慢)")
    
    # 🔧 添加Q值与NA综合质量的相关性分析
    print(f"\n🔍 Q值学习效果分析:")
    print("编号\t初始信誉\t成功率\tQ值\t\t签名延迟等级\t饥饿度\t延迟等级\t综合评分\t是否合理")
    
    # 计算每个NA的综合质量评分 (0-1范围)
    na_scores = []
    for i in range(n_na):
        initial_rep = env.episode_start_reputation[i]
        success_rate = env.success_rate[i]
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        # 综合评分：成功率40% + 延迟性能30% + 信誉20% + 饥饿度10%
        rep_score = (initial_rep - 3300) / (10000 - 3300)  # 归一化信誉 [0,1]
        composite_score = (0.4 * success_rate + 
                           0.3 * delay_perf + 
                           0.2 * rep_score + 
                           0.1 * env.hunger[i])
        na_scores.append(composite_score)
    
    na_scores = np.array(na_scores)
    q_values_array = np.array(all_q_values)
    
    # 计算Q值与综合评分的相关性
    correlation = np.corrcoef(na_scores, q_values_array)[0, 1]
    
    for i in range(n_na):
        initial_rep = env.episode_start_reputation[i]
        success_rate = env.success_rate[i]
        q_val = all_q_values[i]
        delay = env.signature_delay[i]
        delay_perf = env.calculate_delay_performance(delay)
        composite_score = na_scores[i]
        
        # 改进的合理性判断：Q值应该与综合评分正相关
        # 高综合评分的NA应该有高Q值，低综合评分的NA应该有低Q值
        q_percentile = (q_val - q_values_array.min()) / (q_values_array.max() - q_values_array.min())
        score_percentile = (composite_score - na_scores.min()) / (na_scores.max() - na_scores.min())
        
        # 判断Q值排名与综合评分排名是否一致（允许±0.3的误差）
        rank_diff = abs(q_percentile - score_percentile)
        if rank_diff <= 0.3:
            reasonable = "✓ 合理"
        elif rank_diff <= 0.5:
            reasonable = "△ 一般"
        else:
            reasonable = "✗ 不合理"
        
        print(f"{i}\t{initial_rep:.1f}\t\t{success_rate:.3f}\t{q_val:.4f}\t\t{delay:.1%}\t\t{env.hunger[i]:.1%}\t\t{delay_perf:.1%}\t\t{composite_score:.3f}\t\t{reasonable}")
    
    print(f"\n📈 Q值学习效果统计:")
    print(f"Q值与综合评分相关系数: {correlation:.3f}")
    if correlation > 0.7:
        print("✅ 学习效果优秀：Q值能很好地反映NA的综合质量")
    elif correlation > 0.5:
        print("✓ 学习效果良好：Q值基本能反映NA的综合质量")
    elif correlation > 0.3:
        print("△ 学习效果一般：Q值部分反映NA的综合质量，可能需要更多训练")
    else:
        print("✗ 学习效果较差：Q值与NA综合质量相关性较弱，建议检查训练参数")
    
    # 使用DQN选择的结果作为最终选择
    selected_idx = dqn_selected

    # 画最后一次episode的信誉变化
    plt.figure(figsize=(12, 8))
    for i in range(n_na):
        # 根据是否被选中设置不同的绘图样式
        if i in selected_idx:
            # DQN选中的NA用粗线和特殊标记
            plt.plot(range(len(reputation_last)), [r[i] for r in reputation_last], 
                    linewidth=3, marker='o', markersize=4, markevery=10,
                    label=f'[DQN] NA {i} (DQN Selected)')
        else:
            # 未选中的NA用细线
            plt.plot(range(len(reputation_last)), [r[i] for r in reputation_last], 
                    linewidth=1, alpha=0.7, label=f'NA {i}')
    
    # 在图上标记最终选中NA的位置
    for i in selected_idx:
        final_reputation = env.reputation[i]
        plt.scatter(len(reputation_last)-1, final_reputation, 
                   s=200, marker='*', color='gold', edgecolor='red', linewidth=2,
                   zorder=10)
        # 添加文本标注，显示Q值
        q_value = all_q_values[i]
        plt.annotate(f'DQN Selected NA{i}\nQ-value: {q_value:.4f}', 
                    xy=(len(reputation_last)-1, final_reputation),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=8, ha='left')
    
    # 添加6600分界线
    plt.axhline(y=6600, color='red', linestyle='--', linewidth=2, label='Reputation Threshold (6600)')
    
    plt.xlabel('Step')
    plt.ylabel('Reputation')
    plt.title('Last Episode Reputation Trajectory - Fixed NA Set Training Results\n([DQN] = Final Multi-Selection Combination)')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片到本地
    plt.savefig('/mnt/data/wy2024/src/last_episode_reputation_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Last episode reputation trajectory saved as: /mnt/data/wy2024/src/last_episode_reputation_trajectory.png")
    
    # 🔧 可选：生成第一个和最后一个episode的NA信誉动态变化图
    if generate_episode_gifs:
        print("\nGenerating NA reputation dynamic GIF animations for first and last episodes...")
        if first_episode_data:
            create_episode_reputation_animation(first_episode_data, n_na, "First Episode")
        if last_episode_data:
            create_episode_reputation_animation(last_episode_data, n_na, "Last Episode")
    else:
        print("\nSkipping episode reputation GIF generation (disabled by configuration).")
    
    # 生成策略分析图表
    create_policy_analysis_plots(agent, env)
    
    # 保存逐步奖励历史以便绘图
    agent.step_reward_history = step_rewards
    agent.step_reward_steps = step_reward_steps
    
    return pd.DataFrame(history), agent.loss_history, agent

def create_episode_reputation_animation(episode_data, n_na, title_prefix):
    """
    创建单个episode的NA信誉动态变化GIF动画
    
    Args:
        episode_data: 包含信誉历史和选择历史的字典
        n_na: NA数量
        title_prefix: 图表标题前缀
    """
    reputation_history = episode_data['reputation_history']
    selected_nas = episode_data['selected_nas']
    episode_num = episode_data['episode_num']
    
    print(f"正在创建{title_prefix} Episode {episode_num}的GIF动画...")
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 为每个NA分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, n_na))
    
    # 计算数据范围
    steps = range(len(reputation_history))
    all_reputations = [rep for step_rep in reputation_history for rep in step_rep]
    min_reputation = min(all_reputations) - 100
    max_reputation = max(all_reputations) + 200
    
    # 设置坐标轴范围
    ax.set_xlim(0, len(steps) - 1)
    ax.set_ylim(min_reputation, max_reputation)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Reputation', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加6600分界线
    ax.axhline(y=6600, color='red', linestyle='--', linewidth=2, 
               label='Reputation Threshold (6600)', alpha=0.8)
    
    # 初始化线条和点
    lines = []
    points = []
    selected_markers = []  # 用于选中NA的特殊标记
    
    for i in range(n_na):
        # 基础线条（正常粗细）
        line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.7, label=f'NA {i}')
        lines.append(line)
        
        # 当前位置点
        point, = ax.plot([], [], 'o', color=colors[i], markersize=8, alpha=0.9)
        points.append(point)
        
        # 选中标记（红色星标）
        marker, = ax.plot([], [], '*', color='red', markersize=20, alpha=1.0)
        selected_markers.append(marker)
    
    # 添加当前步骤和选择信息的文本
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, verticalalignment='top', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # 添加图例
    if n_na <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    def animate(frame):
        """动画更新函数"""
        current_step = frame
        
        # 更新标题
        selected_na = selected_nas[current_step] if current_step < len(selected_nas) else -1
        ax.set_title(f'{title_prefix} (Episode {episode_num}) - Step {current_step + 1}\n' + 
                    f'Selected NA: {selected_na}' if selected_na != -1 else f'{title_prefix} (Episode {episode_num}) - Initial State', 
                    fontsize=14, weight='bold')
        
        # 更新每个NA的线条和点
        for i in range(n_na):
            # 获取当前步数之前的所有数据点
            if current_step < len(reputation_history):
                x_data = list(range(current_step + 1))
                y_data = [reputation_history[j][i] for j in range(current_step + 1)]
                
                # 判断当前NA是否被选中
                is_selected = (current_step < len(selected_nas) and selected_nas[current_step] == i)
                
                # 根据是否选中设置线条样式
                if is_selected:
                    # 选中的NA：整条线变粗
                    lines[i].set_linewidth(6)
                    lines[i].set_alpha(1.0)
                else:
                    # 未选中的NA：正常粗细
                    lines[i].set_linewidth(2)
                    lines[i].set_alpha(0.7)
                
                # 更新线条数据
                lines[i].set_data(x_data, y_data)
                
                # 更新当前点
                if len(x_data) > 0:
                    points[i].set_data([x_data[-1]], [y_data[-1]])
                
                # 更新选中标记
                if is_selected:
                    selected_markers[i].set_data([current_step], [reputation_history[current_step][i]])
                else:
                    selected_markers[i].set_data([], [])
            else:
                # 清空数据
                lines[i].set_data([], [])
                points[i].set_data([], [])
                selected_markers[i].set_data([], [])
        
        # 更新文本信息
        if current_step < len(selected_nas):
            selected_na = selected_nas[current_step]
            step_text.set_text(f'Step: {current_step + 1}\nSelected NA: {selected_na}\n' + 
                              f'NA{selected_na} Current Reputation: {reputation_history[current_step + 1][selected_na]:.1f}' 
                              if current_step + 1 < len(reputation_history) else f'Step: {current_step + 1}\nSelected NA: {selected_na}')
        else:
            step_text.set_text(f'Step: {current_step + 1}\nInitial State')
        
        return lines + points + selected_markers + [step_text]
    
    # 创建动画（包含初始状态 + 每个step）- 加快速度
    total_frames = len(reputation_history)
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=400, blit=True, repeat=True)  # 从800ms改为400ms，速度翻倍
    
    # 保存动画
    filename = f'{title_prefix.replace(" ", "_").lower()}_episode_{episode_num}_reputation_animation.gif'
    
    try:
        # 确保目录存在
        os.makedirs('/mnt/data/wy2024/episode_animations', exist_ok=True)
        full_path = os.path.join('/mnt/data/wy2024/episode_animations', filename)
        
        anim.save(full_path, writer='pillow', fps=2.5)  # 从1.25提高到2.5 FPS
        print(f"✅ {title_prefix} GIF animation saved as: {full_path}")
    except Exception as e:
        print(f"❌ Error saving {title_prefix} animation: {e}")
    
    # 清理内存
    plt.close(fig)

def create_comparison_strategy_gif(env, dqn_selected, random_selected, dqn_training_results, random_training_results, steps=100):
    """
    创建选择策略对比动图，显示DQN选择和随机选择的NA在训练过程中的信誉变化轨迹
    
    Args:
        env: 环境对象
        dqn_selected: DQN选择的NA索引列表
        random_selected: 随机选择的NA索引列表
        dqn_training_results: DQN选择NA的训练结果
        random_training_results: 随机选择NA的训练结果
        steps: 模拟步数
    """
    print("🎬 正在创建选择策略对比动图...")
    
    # 创建图形和坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 获取所有选中NA的信誉历史数据
    all_selected_nas = list(set(dqn_selected + random_selected))
    
    # 🎨 为每个NA分配不同的颜色
    # 使用matplotlib的颜色循环，确保每个NA都有独特的颜色
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 获取10种不同颜色
    additional_colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 额外20种颜色
    all_colors = np.vstack([colors, additional_colors])  # 合并为30种颜色
    
    # 为DQN选择的NA分配颜色
    dqn_colors = {}
    for i, na_idx in enumerate(dqn_selected):
        dqn_colors[na_idx] = all_colors[i % len(all_colors)]
    
    # 为随机选择的NA分配颜色（确保与DQN选择的颜色不同）
    random_colors = {}
    for i, na_idx in enumerate(random_selected):
        color_idx = (i + len(dqn_selected)) % len(all_colors)
        random_colors[na_idx] = all_colors[color_idx]
    
    # 计算数据范围
    all_reputations = []
    max_steps = 0
    
    # 收集DQN选择NA的信誉历史
    for na_idx in dqn_selected:
        if na_idx in dqn_training_results:
            rep_history = dqn_training_results[na_idx]['reputation_history']
            all_reputations.extend(rep_history)
            max_steps = max(max_steps, len(rep_history))
    
    # 收集随机选择NA的信誉历史  
    for na_idx in random_selected:
        if na_idx in random_training_results:
            rep_history = random_training_results[na_idx]['reputation_history']
            all_reputations.extend(rep_history)
            max_steps = max(max_steps, len(rep_history))
    
    if not all_reputations:
        print("❌ 没有足够的数据创建动图")
        return
    
    min_reputation = min(all_reputations) - 100
    max_reputation = max(all_reputations) + 200
    
    # 设置两个子图的基本属性
    for ax, title in zip([ax1, ax2], ['DQN Selected NA Reputation Trajectory', 'Random Selected NA Reputation Trajectory']):
        ax.set_xlim(0, max_steps - 1)
        ax.set_ylim(min_reputation, max_reputation)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Reputation', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=6600, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Threshold (6600)')
    
    # 初始化线条
    dqn_lines = {}
    random_lines = {}
    dqn_points = {}
    random_points = {}
    
    # DQN选择的NA线条
    for i, na_idx in enumerate(dqn_selected):
        if na_idx in dqn_training_results:
            dqn_color = dqn_colors[na_idx]
            
            # 🚨 检查是否为恶意NA，如果是则使用特殊线型
            is_malicious = env.fixed_malicious_flags[na_idx]
            if is_malicious:
                mal_type = env.fixed_malicious_types[na_idx]
                # 恶意NA使用虚线和特殊标记
                line, = ax1.plot([], [], color=dqn_color, linewidth=4, linestyle='--', alpha=0.9, 
                               label=f'[MAL-T{mal_type}] NA {na_idx}' if i < 5 else "")
                point, = ax1.plot([], [], 's', color=dqn_color, markersize=10, alpha=0.9, 
                                markeredgecolor='red', markeredgewidth=2)  # 方形标记，红边框
            else:
                # 正常NA使用实线和圆形标记
                line, = ax1.plot([], [], color=dqn_color, linewidth=3, alpha=0.8, 
                               label=f'[NORM] NA {na_idx}' if i < 5 else "")
                point, = ax1.plot([], [], 'o', color=dqn_color, markersize=8, alpha=0.9)
                
            dqn_lines[na_idx] = line
            dqn_points[na_idx] = point
    
    # 随机选择的NA线条
    for i, na_idx in enumerate(random_selected):
        if na_idx in random_training_results:
            random_color = random_colors[na_idx]
            
            # 🚨 检查是否为恶意NA，如果是则使用特殊线型
            is_malicious = env.fixed_malicious_flags[na_idx]
            if is_malicious:
                mal_type = env.fixed_malicious_types[na_idx]
                # 恶意NA使用虚线和特殊标记
                line, = ax2.plot([], [], color=random_color, linewidth=4, linestyle='--', alpha=0.9, 
                               label=f'[MAL-T{mal_type}] NA {na_idx}' if i < 5 else "")
                point, = ax2.plot([], [], 's', color=random_color, markersize=10, alpha=0.9, 
                                markeredgecolor='red', markeredgewidth=2)  # 方形标记，红边框
            else:
                # 正常NA使用实线和圆形标记
                line, = ax2.plot([], [], color=random_color, linewidth=3, alpha=0.8, 
                               label=f'[NORM] NA {na_idx}' if i < 5 else "")
                point, = ax2.plot([], [], 'o', color=random_color, markersize=8, alpha=0.9)
                
            random_lines[na_idx] = line
            random_points[na_idx] = point
    
    # 添加图例
    ax1.legend(loc='upper left', fontsize='small')
    ax2.legend(loc='upper left', fontsize='small')
    
    # 添加统计信息文本
    dqn_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    random_text = ax2.text(0.02, 0.02, '', transform=ax2.transAxes, fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    def animate(frame):
        """动画更新函数"""
        current_step = frame
        
        # 更新DQN选择的NA
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
        
        # 更新随机选择的NA
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
        
        # 更新统计信息
        if dqn_current_reps:
            dqn_avg = np.mean(dqn_current_reps)
            dqn_std = np.std(dqn_current_reps)
            dqn_text.set_text(f'Step: {current_step}\nDQN Selection Stats:\nAvg Reputation: {dqn_avg:.1f}\nStd Dev: {dqn_std:.1f}\nNA Count: {len(dqn_current_reps)}')
        
        if random_current_reps:
            random_avg = np.mean(random_current_reps)
            random_std = np.std(random_current_reps)
            random_text.set_text(f'Step: {current_step}\nRandom Selection Stats:\nAvg Reputation: {random_avg:.1f}\nStd Dev: {random_std:.1f}\nNA Count: {len(random_current_reps)}')
        
        return list(dqn_lines.values()) + list(dqn_points.values()) + list(random_lines.values()) + list(random_points.values()) + [dqn_text, random_text]
    
    # 创建动画
    total_frames = max_steps
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=200, blit=True, repeat=True)  # 200ms间隔
    
    # 保存动画
    filename = 'strategy_comparison_reputation_animation.gif'
    
    try:
        # 确保目录存在
        os.makedirs('/mnt/data/wy2024/episode_animations', exist_ok=True)
        full_path = os.path.join('/mnt/data/wy2024/episode_animations', filename)
        
        anim.save(full_path, writer='pillow', fps=5)  # 5 FPS
        print(f"✅ 选择策略对比动图已保存: {full_path}")
        
        # 打印简要统计
        print(f"动图统计信息:")
        print(f"   DQN选择的NA: {len(dqn_selected)}个 - {dqn_selected}")
        print(f"   随机选择的NA: {len(random_selected)}个 - {random_selected}")
        print(f"   动画总帧数: {total_frames}")
        print(f"   动画时长: {total_frames/5:.1f}秒")
        
    except Exception as e:
        print(f"❌ 保存选择策略对比动图失败: {e}")
    
    # 清理内存
    plt.close(fig)

#  训练并可视化
if __name__ == "__main__":
    print("🎯 基于DQN的NA选择优化 - 20个NA训练配置")
    print("="*60)
    
    # 🔧 优化的训练参数 - 使用固定NA集合，改进稳定性
    df, loss_history, agent = train_dqn(
        n_na=20,             # NA数量：20个
        n_episodes=50,      # 训练轮数：60轮训练 (分阶段学习率衰减)
        steps_per_episode=200,# 每轮步数：200步
        lr=0.001,            # 学习率：0.001，确保网络能学到差异
        update_frequency=4,  # 更新频率：每4步训练一次，加速学习
        use_fixed_nas=True,  # 使用固定NA集合
        enable_random_fluctuation=False  # 关闭随机波动
    )
    
    print(f"\n✅ 训练完成！生成了 {len(df)} 个Episode的训练数据")
    print(f"📈 损失历史记录: {len(loss_history)} 个训练步骤")
    
    # 保存训练好的模型
    print("\n💾 保存训练好的模型...")
    try:
        # 确保模型保存目录存在
        os.makedirs('/mnt/data/wy2024/models', exist_ok=True)
        
        # 保存策略网络和目标网络的状态字典（推荐方式）
        torch.save(agent.policy_net.state_dict(), '/mnt/data/wy2024/models/policy_net_state_dict.pth')
        torch.save(agent.target_net.state_dict(), '/mnt/data/wy2024/models/target_net_state_dict.pth')
        
        # 保存完整的模型（备用方式）
        torch.save(agent.policy_net, '/mnt/data/wy2024/models/policy_net_complete.pth')
        torch.save(agent.target_net, '/mnt/data/wy2024/models/target_net_complete.pth')
        
        # 保存训练参数和配置信息
        model_info = {
            'n_na': 20,
            'n_features': 4,
            'n_episodes': 50,  # 更新为实际训练轮数
            'learning_rate': 0.001,  # 更新为实际学习率
            'final_epsilon': agent.epsilon,
            'total_training_steps': len(loss_history),
            'final_loss': loss_history[-1] if loss_history else None,
            'final_reward': df['reward'].iloc[-1] if len(df) > 0 else None
        }
        torch.save(model_info, '/mnt/data/wy2024/models/model_info.pth')
        
        print("✅ 模型保存成功！")
        print("   - 策略网络状态字典: /mnt/data/wy2024/models/policy_net_state_dict.pth")
        print("   - 目标网络状态字典: /mnt/data/wy2024/models/target_net_state_dict.pth")
        print("   - 完整策略网络: /mnt/data/wy2024/models/policy_net_complete.pth")
        print("   - 完整目标网络: /mnt/data/wy2024/models/target_net_complete.pth")
        print("   - 模型配置信息: /mnt/data/wy2024/models/model_info.pth")
        
    except Exception as e:
        print(f"❌ 模型保存失败: {e}")
    
    # 创建包含八个子图的图形来详细分析训练进展（包含Q值监测）
    # 放大图内字体以便导出为 PDF 时更易阅读
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    plt.figure(figsize=(24, 16))
    
    # 第一个子图：逐步奖励（与损失采样方式一致）
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
        # 回退到基于episode的绘制
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
    
    # 第二个子图：平均信誉
    plt.subplot(3, 3, 2)
    # 原始数据（透明度较低）
    plt.plot(df['episode'], df['mean_reputation'], alpha=0.3, label='Raw Mean Reputation', color='lightgreen')
    
    # 添加平滑曲线 - 使用改进的平滑方法确保连接起始点和终点
    window_size = 20
    if len(df) >= window_size:
        smoothed_reputation = df['mean_reputation'].rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(df['episode'], smoothed_reputation, label=f'Smoothed Mean Reputation ({window_size})', color='darkgreen', linewidth=2)
    
    plt.title('Mean Reputation')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reputation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第三个子图：损失函数曲线
    plt.subplot(3, 3, 3)
    if len(loss_history) > 0:
        loss_steps = agent.loss_steps if hasattr(agent, 'loss_steps') and len(agent.loss_steps) == len(loss_history) else list(range(len(loss_history)))
        # 计算移动平均以平滑曲线
        window_size = min(100, len(loss_history) // 10)  # 窗口大小为总长度的1/10，最小为100
        if window_size > 1:
            smoothed_loss = []
            for i in range(len(loss_history)):
                start_idx = max(0, i - window_size + 1)
                smoothed_loss.append(np.mean(loss_history[start_idx:i+1]))
            plt.plot(loss_steps, smoothed_loss, label=f'Smoothed Loss (window={window_size})', alpha=0.8)
        
        # 原始损失曲线（透明度较低）
        plt.plot(loss_steps, loss_history, alpha=0.3, label='Raw Loss', color='gray')
        plt.title('Loss Function Curve')
        plt.xlabel('Training Steps')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Loss Function Curve')
    
    # 第四个子图：奖励标准差（每20集滑动窗口）
    plt.subplot(3, 3, 4)
    # 计算每20集滑动窗口的奖励标准差
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
        # 添加平滑线 - 使用改进的平滑方法
        if len(reward_std_data) >= 20:
            reward_std_series = pd.Series(reward_std_data)
            smoothed_reward_std = reward_std_series.rolling(window=20, center=True, min_periods=1).mean()
            plt.plot(episode_data, smoothed_reward_std, label='Smoothed Reward Std', color='darkgreen', linewidth=2)
    
    plt.title('Reward Standard Deviation (20-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Reward Std')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第五个子图：平均单步奖励
    plt.subplot(3, 3, 5)
    plt.plot(df['episode'], df['avg_step_reward'], label='Avg Step Reward', color='orange', alpha=0.2)
    # 添加平滑线与置信带 - 使用改进的平滑方法
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
    
    # 第六个子图：奖励标准差（波动性分析）
    plt.subplot(3, 3, 6)
    plt.plot(df['episode'], df['reward_std'], label='Step Reward Std Dev', color='red', alpha=0.4)
    # 添加平滑线 - 使用改进的平滑方法
    if len(df) >= 20:
        smoothed_std = df['reward_std'].rolling(window=20, center=True, min_periods=1).mean()
        plt.plot(df['episode'], smoothed_std, label='Smoothed Std Dev', color='darkred', linewidth=2)
    plt.title('Step Reward Volatility (Std Dev)')
    plt.xlabel('Episode')
    plt.ylabel('Reward Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 第七个子图：Q值范围监测
    plt.subplot(3, 3, 7)
    plt.plot(df['episode'], df['q_max'], label='Q Max', color='red', alpha=0.7)
    plt.plot(df['episode'], df['q_min'], label='Q Min', color='blue', alpha=0.7)
    plt.fill_between(df['episode'], df['q_min'], df['q_max'], alpha=0.2, color='purple')
    plt.title('Q-Value Range (Min-Max)')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标显示大范围的Q值
    
    # 第八个子图：Q值均值和标准差
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
    plt.yscale('log')  # 使用对数坐标显示大范围的Q值
    
    # 第九个子图：学习率和Epsilon变化（训练调度）
    plt.subplot(3, 3, 9)
    if hasattr(agent, 'lr_history') and agent.lr_history:
        # 修复横坐标：生成正确的episode序列
        lr_episodes = []
        total_episodes = len(df)  # 从dataframe获取总episode数
        for i in range(len(agent.lr_history)):
            if i < len(agent.lr_history) - 1:
                # 前面的数据点都是10的倍数
                lr_episodes.append(i * 10)
            else:
                # 最后一个数据点：如果总episode数不是10的倍数，使用实际的最后一个episode
                final_episode = total_episodes - 1
                if final_episode % 10 == 0:
                    lr_episodes.append(i * 10)  # 正好是10的倍数
                else:
                    lr_episodes.append(final_episode)  # 使用实际的最后一个episode
        
        # 双y轴显示学习率和epsilon
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # 学习率 (左y轴，对数坐标)
        line1 = ax1.plot(lr_episodes, agent.lr_history, label='Learning Rate', color='orange', linewidth=2)
        ax1.set_ylabel('Learning Rate', color='orange')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='orange')
        
        # Epsilon (右y轴，线性坐标)
        if hasattr(agent, 'epsilon_history') and agent.epsilon_history:
            line2 = ax2.plot(lr_episodes, agent.epsilon_history, label='Epsilon (探索率)', color='purple', linewidth=2)
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
    
    # 保存训练结果图表到本地
    main_png = '/mnt/data/wy2024/src/training_results.png'
    plt.savefig(main_png, dpi=300, bbox_inches='tight')
    print(f"训练结果图表已保存为: {main_png}")

    # 单独保存 Reward 曲线为单页 PDF
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
        # 优先使用 step-level reward 数据（如果存在）
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
            # 回退到基于 episode 的绘制
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

        reward_pdf = '/mnt/data/wy2024/src/training_results_reward.pdf'
        # Ensure vector fonts in PDF (embed TrueType as Type42)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.tight_layout()
        plt.savefig(reward_pdf, bbox_inches='tight')
        plt.close()
        print(f"Saved reward PDF: {reward_pdf}")
    except Exception as e:
        print(f"Failed to save reward PDF: {e}")

    # 单独保存 MSE Loss 曲线为单页 PDF
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

        loss_pdf = '/mnt/data/wy2024/src/training_results_loss.pdf'
        # Ensure vector fonts in PDF (embed TrueType as Type42)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.tight_layout()
        plt.savefig(loss_pdf, bbox_inches='tight')
        plt.close()
        print(f"Saved loss PDF: {loss_pdf}")
    except Exception as e:
        print(f"Failed to save loss PDF: {e}")

    # 关闭主图的 figure
    plt.close()
    
    # 打印损失统计信息
    if len(loss_history) > 0:
        print(f"\n损失函数统计:")
        print(f"总训练步数: {len(loss_history)}")
        print(f"平均损失: {np.mean(loss_history):.6f}")
        print(f"最终损失: {loss_history[-1]:.6f}")
        print(f"最小损失: {np.min(loss_history):.6f}")
        print(f"最大损失: {np.max(loss_history):.6f}")
        
        # 检查是否收敛
        if len(loss_history) > 1000:
            recent_loss = np.mean(loss_history[-1000:])  # 最后1000步的平均损失
            early_loss = np.mean(loss_history[:1000])    # 前1000步的平均损失
            improvement = (early_loss - recent_loss) / early_loss * 100
            print(f"损失改善程度: {improvement:.2f}%")
    else:
        print("\n警告: 没有记录到损失数据，可能是训练步数不足或batch_size设置过大")
    
    # 💡 添加详细的奖励波动分析
    print(f"\nEpisode奖励波动分析:")
    print(f"总Episode数: {len(df)}")
    print(f"平均Episode奖励: {df['reward'].mean():.2f}")
    print(f"Episode奖励标准差: {df['reward'].std():.2f}")
    print(f"Episode奖励范围: {df['reward'].min():.2f} ~ {df['reward'].max():.2f}")
    print(f"奖励变异系数 (CV): {df['reward'].std() / abs(df['reward'].mean()):.2f}")
    
    print(f"\n🎲 单步奖励统计:")
    print(f"平均单步奖励: {df['avg_step_reward'].mean():.2f}")
    print(f"单步奖励波动性: {df['reward_std'].mean():.2f}")
    print(f"总体成功率: {df['success_rate'].mean():.3f}")
    
    print(f"\n🔍 奖励波动原因分析:")
    high_volatility_episodes = df[df['reward_std'] > df['reward_std'].quantile(0.8)]
    print(f"高波动episode数量: {len(high_volatility_episodes)} ({len(high_volatility_episodes)/len(df)*100:.1f}%)")
    
    if len(high_volatility_episodes) > 0:
        print(f"高波动episode平均奖励: {high_volatility_episodes['reward'].mean():.2f}")
        print(f"高波动episode平均成功率: {high_volatility_episodes['success_rate'].mean():.3f}")
    
    # 分析前中后期的变化
    early_episodes = df.iloc[:len(df)//3]  # 前1/3
    middle_episodes = df.iloc[len(df)//3:2*len(df)//3]  # 中1/3  
    late_episodes = df.iloc[2*len(df)//3:]  # 后1/3
    
    print(f"\n📈 训练阶段分析:")
    print(f"前期 (Episodes 0-{len(early_episodes)}): 平均奖励={early_episodes['reward'].mean():.2f}, 波动={early_episodes['reward'].std():.2f}")
    print(f"中期 (Episodes {len(early_episodes)}-{len(early_episodes)+len(middle_episodes)}): 平均奖励={middle_episodes['reward'].mean():.2f}, 波动={middle_episodes['reward'].std():.2f}")
    print(f"后期 (Episodes {len(early_episodes)+len(middle_episodes)}-{len(df)}): 平均奖励={late_episodes['reward'].mean():.2f}, 波动={late_episodes['reward'].std():.2f}")
    
    improvement = (late_episodes['reward'].mean() - early_episodes['reward'].mean()) / abs(early_episodes['reward'].mean()) * 100
    volatility_change = (late_episodes['reward'].std() - early_episodes['reward'].std()) / early_episodes['reward'].std() * 100
    
    print(f"\n训练效果:")
    print(f"奖励改善: {improvement:+.1f}%")
    print(f"波动变化: {volatility_change:+.1f}% ({'改善' if volatility_change < 0 else '恶化'})")
    
    # 🕐 记录程序结束时间和总耗时
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n" + "="*80)
    print(f"⏰ 程序运行时间统计")
    print(f"="*80)
    print(f"🚀 启动时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏁 结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时: {format_duration(total_duration)}")
    print(f"训练效率:")
    print(f"   - 总Episodes: {len(df)}")
    print(f"   - 平均每Episode耗时: {total_duration/len(df):.2f}秒")
    if len(loss_history) > 0:
        print(f"   - 总训练步数: {len(loss_history)}")
        print(f"   - 平均每步耗时: {total_duration/len(loss_history)*1000:.2f}毫秒")
    print(f"="*80)
    print(f"🏁 程序执行完成！")
    
    # 导出最后一次模型选择NA时的所有参数
    print(f"\n" + "="*80)
    print("导出NA选择参数")
    print("="*80)
    
    try:
        # 创建环境实例以获取当前状态
        export_env = Environment(n_na=20, use_fixed_nas=True, enable_random_fluctuation=False)
        export_env.reset()
        
        # 获取当前状态和初始信誉值
        current_state = export_env.get_state()
        initial_reputations = export_env.episode_start_reputation
        
        # 导出NA选择参数
        export_file = agent.export_na_selection_parameters(
            state=current_state, 
            initial_reputations=initial_reputations,
            env=export_env
        )
        
        print(f"✅ NA选择参数导出成功！")
        
    except Exception as e:
        print(f"❌ NA选择参数导出失败: {e}")
    
    # 演示网络架构的灵活性
    print(f"\n" + "="*80)
    print("🧪 演示网络架构的灵活性 - 处理不同数量的NA")
    print("="*80)
    
    def test_network_flexibility(trained_agent):
        """演示训练好的网络可以处理不同数量的NA"""
        print("测试网络是否能处理不同数量的NA...")
        
        # 测试不同数量的NA
        test_cases = [5, 10, 15, 25, 30]
        
        for n_test_na in test_cases:
            # 创建包含4个特征的测试数据 (reputation, success_rate, signature_delay, hunger)
            test_state = np.zeros((n_test_na, 4))
            test_state[:, 0] = np.random.uniform(3300, 10000, n_test_na)  # reputation
            test_state[:, 1] = np.random.uniform(0.2, 0.9, n_test_na)     # success_rate  
            test_state[:, 2] = np.random.uniform(50, 250, n_test_na)      # signature_delay (ms)
            test_state[:, 3] = np.random.uniform(0.0, 1.0, n_test_na)     # hunger (0%-100%)

            # 归一化
            normalized_test_state = test_state.copy()
            normalized_test_state[:, 0] = (test_state[:, 0] - 3000) / (10000 - 3000)
            normalized_test_state[:, 1] = test_state[:, 1]
            normalized_test_state[:, 2] = np.clip((test_state[:, 2] - 50) / (300 - 50), 0, 1)
            normalized_test_state[:, 3] = test_state[:, 3]
            
            # 转换为tensor
            test_tensor = torch.FloatTensor(normalized_test_state).unsqueeze(0).to(trained_agent.device)
            
            # 前向传播
            with torch.no_grad():
                q_values = trained_agent.policy_net(test_tensor)
            
            print(f"✅ NA数量: {n_test_na:2d} -> Q值形状: {q_values.shape} | 最高Q值: {q_values.max().item():.4f}")
        
        print("🎉 网络成功处理了所有不同数量的NA！")
        print("💡 优势：训练好的模型可以直接应用于任意数量的NA场景")
        print("特征说明：reputation(信誉), success_rate(成功率), activity(活跃度), signature_delay(签名延迟), hunger(饥饿度)")
    
    test_network_flexibility(agent)
