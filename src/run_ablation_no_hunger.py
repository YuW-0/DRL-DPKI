import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
================================================

Key features:
- Double DQN to reduce Q-value overestimation
- Experience replay to improve sample efficiency
- Target network for training stability
- NA-count-invariant architecture to support arbitrary NA counts

Core idea:
1) Select actions with the policy network
2) Evaluate selected actions with the target network
"""

start_time = time.time()
start_datetime = datetime.now()
print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

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
    Set all random seeds for reproducibility.
    """
    print(f"Setting random seed: {seed}")
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("Random seed set.")

RANDOM_SEED = 42

print(f"Seed: {RANDOM_SEED}")

set_all_seeds(RANDOM_SEED)

print("GPU check:")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("Using CPU for training")

print("Configuring fonts...")

font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "font"))
if os.path.isdir(font_dir):
    font_files = [f for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
    if font_files:
        print(f"Found {len(font_files)} custom font files. Loading...")
    loaded_custom_font = False
    for font_file in font_files:
        font_path = os.path.join(font_dir, font_file)
        try:
            fm.fontManager.addfont(font_path)
            loaded_custom_font = True
        except Exception as exc:
            print(f"[WARN] Failed to load font {font_file}: {exc}")
else:
    print(f"[WARN] Font directory not found: {font_dir}")

if 'loaded_custom_font' in locals() and loaded_custom_font:
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

print(f"Font family: {plt.rcParams['font.family'][:2]}")
print("Font configuration done")

def format_duration(seconds):
    """
    Format a duration in seconds.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours}h{minutes}m{secs}s"
    elif minutes > 0:
        return f"{minutes}m{secs}s"
    else:
        return f"{secs}.{millisecs:03d}s"

class QNetwork(nn.Module):
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
        """Initialize network weights."""
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

class DQNAgent:
    def __init__(self, n_features, n_na, lr, gamma,
                 epsilon_start, epsilon_end,
                 decay_steps, memory_size, batch_size, target_update,
                 min_memory_size, update_frequency, total_episodes):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"Double DQN Agent device: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            print(f"Double DQN Agent device: {self.device}")
        print("Using NA-count-invariant architecture")
        print("Algorithm: Double DQN (DDQN)")
        print(f"Replay buffer={memory_size}, batch_size={batch_size}, min_memory={min_memory_size}")
        
        self.n_na = n_na
        self.n_features = n_features
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.decay_steps   = decay_steps
        self.total_episodes = total_episodes
        self.step_count    = 0
        self.epsilon       = epsilon_start
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.min_memory_size = min_memory_size
        self.update_frequency = update_frequency
        self.step_counter = 0
        
        self.policy_net = QNetwork(n_features, n_na).to(self.device)
        self.target_net = QNetwork(n_features, n_na).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr,
            weight_decay=1e-4,
            amsgrad=True
        )
        print("Optimizer: AdamW (weight_decay=1e-4, amsgrad=True)")
        
        self.initial_lr = lr
        self.final_lr = lr * 0.1
        self.total_episodes = total_episodes
        print(
            "LR schedule: Cosine annealing with warm-up "
            f"{self.final_lr:.6f} -> {lr:.6f} -> {self.final_lr:.6f}"
        )
        
        self.target_update = target_update
        self.learn_step = 0
        
        self.loss_history = []
        self.loss_steps = []
        
        self.lr_history = []
        
        self.epsilon_history = []
        
        self.policy_entropy_history = []
        self.action_distribution_history = []
        self.na_selection_frequency = np.zeros(n_na)

    def select_action(self, state, initial_reputations=None, available_mask=None, record_policy=False):
        """
        Select a single NA action.
        
        Args:
            state: ndarray, shape (n_na, n_features)
            initial_reputations: ndarray, shape (n_na,)
            available_mask: ndarray[bool], shape (n_na,), optional
            record_policy: bool
        
        Returns:
            int: Selected NA index
        """
        if initial_reputations is None:
            reputations = state[:, 0]
        else:
            reputations = initial_reputations
        
        if available_mask is None:
            available_mask = np.ones(self.n_na, dtype=bool)
        
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0:
            return 0

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).cpu()  # [n_na]
        
        masked_q_values = q_values.clone()
        masked_q_values[~torch.from_numpy(available_mask)] = float('-inf')
        
        valid_q_values = q_values[available_mask]
        if len(valid_q_values) > 1:
            policy_probs = F.softmax(valid_q_values, dim=0).numpy()
            policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-8))
        else:
            policy_entropy = 0.0
            policy_probs = np.array([1.0])
        
        if record_policy:
            self.policy_entropy_history.append(policy_entropy)
            full_policy_dist = np.zeros(self.n_na)
            full_policy_dist[available_indices] = policy_probs
            self.action_distribution_history.append(full_policy_dist.copy())
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(available_indices)
        else:
            action = masked_q_values.argmax().item()
        
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
        Cosine annealing learning-rate schedule with warm-up.
        """
        if current_episode >= self.total_episodes:
            progress = 1.0
        else:
            progress = current_episode / self.total_episodes
        
        warmup_ratio = 0.05
        
        if progress <= warmup_ratio:
            warmup_progress = progress / warmup_ratio
            current_lr = self.final_lr + (self.initial_lr - self.final_lr) * warmup_progress
        else:
            cosine_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
            import math
            cosine_factor = (1 + math.cos(math.pi * cosine_progress)) / 2
            current_lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine_factor
        
        current_lr = max(current_lr, self.final_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        return current_lr
    
    def make_final_selection(self, state, initial_reputations):
        """
        Run final selection using the trained DQN policy.
        
        Args:
            state: ndarray, shape (n_na, n_features)
            initial_reputations: ndarray, shape (n_na,)
        
        Returns:
            dict: selection results and analysis data
        """
        selected_nas = self.make_multiple_selections(state, initial_reputations, 5, 'balanced')
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        low_rep_indices = np.where(low_rep_mask)[0]
        high_rep_indices = np.where(high_rep_mask)[0]
        
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
        Export per-NA parameters at the last selection step.
        
        Args:
            state: ndarray, shape (n_na, n_features)
            initial_reputations: ndarray, shape (n_na,)
            env: optional environment object
            output_file: optional output path
        
        Returns:
            str: output path
        """
        if output_file is None:
            output_file = str(OUTPUT_ROOT / "src" / "na_selection_parameters.csv")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        selection_result = self.make_final_selection(state, initial_reputations)
        selected_nas = selection_result['dqn_selected']
        
        weighted_success_rates = []
        weighted_delay_levels = []
        if env is not None:
            for i in range(self.n_na):
                window_summary = env.get_na_window_summary(i)
                weighted_success_rates.append(window_summary['weighted_success_rate'])
                weighted_delay_levels.append(window_summary['weighted_delay_grade'])
        else:
            weighted_success_rates = state[:, 1].tolist()
            weighted_delay_levels = [0] * self.n_na
        
        export_data = []
        for i in range(self.n_na):
            na_data = {
                'NA_Index': i,
                'Q_Value': all_q_values[i],
                'Current_Reputation': state[i, 0],
                'Success_Rate': state[i, 1],
                'Signature_Delay': env.signature_delay[i] if env is not None else state[i, 2],
                'Hunger_Level': state[i, 3],
                'Weighted_Success_Rate': weighted_success_rates[i],
                'Weighted_Delay_Grade': weighted_delay_levels[i],
                'Initial_Reputation': initial_reputations[i],
                'Reputation_Group': 'High' if initial_reputations[i] >= 6600 else 'Low',
                'Selected_by_DQN': i in selected_nas,
                'Selection_Rank': selected_nas.index(i) + 1 if i in selected_nas else None
            }
            export_data.append(na_data)
        
        export_data.sort(key=lambda x: x['Q_Value'], reverse=True)
        
        for rank, na_data in enumerate(export_data, 1):
            na_data['Q_Value_Rank'] = rank
        
        df = pd.DataFrame(export_data)
        df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"\nNA selection parameters exported: {output_file}")
        print(f"   - Total NAs: {len(df)}")
        print(f"   - DQN-selected NAs: {len([x for x in export_data if x['Selected_by_DQN']])}")
        print(f"   - High-rep NAs: {len([x for x in export_data if x['Reputation_Group'] == 'High'])}")
        print(f"   - Low-rep NAs: {len([x for x in export_data if x['Reputation_Group'] == 'Low'])}")
        print(f"   - Q range: {df['Q_Value'].min():.6f} ~ {df['Q_Value'].max():.6f}")
        
        return output_file

    def store(self, s, a, r, s_, done):
        """
        Store experience (single-NA action).
        
        Args:
            s: ndarray, shape (n_na, n_features)
            a: int
            r: float
            s_: ndarray, shape (n_na, n_features)
            done: bool
        """
        self.memory.append((s.copy(), int(a), r, s_.copy(), done))

    def update(self, global_step=None):
        if len(self.memory) < self.min_memory_size:
            return None
            
        if len(self.memory) < self.batch_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(states)
        current_q = current_q_values.gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]
            next_q_values = self.target_net(next_states)
            next_q = next_q_values.gather(1, next_actions)
            
            target = rewards + self.gamma * next_q * (1 - dones)
        
        loss = F.mse_loss(current_q, target)
        
        self.loss_history.append(loss.item())
        if global_step is not None:
            self.loss_steps.append(global_step)
        else:
            self.loss_steps.append(self.learn_step)
        
        self.optimizer.zero_grad()
        loss.backward()    
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def maybe_update(self, global_step=None):
        """
        Conditional updates with potentially more frequent training.
        
        Returns:
            list: Loss values for this update batch, or None.
        """
        self.step_counter += 1
        
        if (self.step_counter % self.update_frequency == 0 and 
            len(self.memory) >= self.min_memory_size):
            
            batch_losses = []
            training_batches = max(2, self.update_frequency)
            
            for _ in range(training_batches):
                loss = self.update(global_step=global_step)
                if loss is not None:
                    batch_losses.append(loss)
            
            return batch_losses if batch_losses else None
        
        return None

    def debug_q_values(self, state, initial_reputations):
        """Debug Q-value learning status."""
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze().cpu()
        
        print(f"Q stats: min={q_values.min():.4f}, max={q_values.max():.4f}, "
              f"std={q_values.std():.4f}")
        
        if q_values.std() < 0.1:
            print("[WARN] Q-value variance is small; the network may not be learning useful signals")
        
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        
        if np.any(low_rep_mask) and np.any(high_rep_mask):
            low_q_mean = q_values[low_rep_mask].mean()
            high_q_mean = q_values[high_rep_mask].mean()
            print(f"Low-rep mean Q: {low_q_mean:.4f}, high-rep mean Q: {high_q_mean:.4f}")
            print(f"Direction: {'correct' if high_q_mean > low_q_mean else 'incorrect'}")
        
        return q_values

    def make_multiple_selections(self, state, initial_reputations, num_selections=5, strategy='balanced'):
        """
        Compose a final NA set by repeated single selections.
        
        Args:
            state: ndarray, shape (n_na, n_features)
            initial_reputations: ndarray, shape (n_na,)
            num_selections: int
            strategy: str ('balanced', 'top_q', 'by_group')
        
        Returns:
            list: selected NA indices
        """
        original_epsilon = self.epsilon
        self.epsilon = 0
        
        selected_nas = []
        available_mask = np.ones(self.n_na, dtype=bool)
        
        if strategy == 'balanced':
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            target_low = min(2, len(low_rep_indices), num_selections)
            target_high = min(num_selections - target_low, len(high_rep_indices))
            
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            for _ in range(target_low):
                if np.any(low_mask):
                    selected_na = self.select_action(state, initial_reputations, low_mask)
                    selected_nas.append(selected_na)
                    low_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            high_mask = np.zeros(self.n_na, dtype=bool)
            high_mask[high_rep_indices] = True
            high_mask &= available_mask
            
            for _ in range(target_high):
                if np.any(high_mask):
                    selected_na = self.select_action(state, initial_reputations, high_mask)
                    selected_nas.append(selected_na)
                    high_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            remaining_needed = num_selections - len(selected_nas)
            for _ in range(remaining_needed):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'top_q':
            for _ in range(min(num_selections, self.n_na)):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'by_group':
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            while len(selected_nas) < num_selections and np.any(low_mask):
                selected_na = self.select_action(state, initial_reputations, low_mask)
                selected_nas.append(selected_na)
                low_mask[selected_na] = False
                available_mask[selected_na] = False
            
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
    def __init__(self, n_na, use_fixed_nas, enable_random_fluctuation, window_size=20,
                 freeze_hunger=False, hunger_constant=0.0, hunger_weight=1.0, enable_hunger_update=True):
        self.n_na = n_na
        self.use_fixed_nas = use_fixed_nas
        self.enable_random_fluctuation = enable_random_fluctuation
        if enable_random_fluctuation:
            print("Random fluctuation enabled: success rate, hunger, and delay may vary dynamically")
        else:
            print("Fixed mode: parameters stay at baseline without random fluctuation")

        self.hunger_growth_scale = 10.0
        self.hunger_growth_log_base = 11.0
        self.freeze_hunger = freeze_hunger
        self.hunger_constant = float(np.clip(hunger_constant, 0.0, 1.0))
        self.hunger_weight = float(hunger_weight)
        self.enable_hunger_update = enable_hunger_update
        
        self.window_pack_interval = 5
        self.window_queue_size = window_size
        self.na_window_queues = {}
        self.current_step_count = 0
        self.na_current_pack = {}
        
        for na_id in range(n_na):
            self.na_window_queues[na_id] = deque(maxlen=window_size)
            self.na_current_pack[na_id] = {
                'transactions': [],
                'success_count': 0,
                'total_count': 0,
                'reputation_changes': [],
                'start_step': 0,
                'end_step': 0
            }
        print(
            "Sliding-window queues: each NA maintains its own queue; "
            f"pack every {self.window_pack_interval} steps; queue size={window_size}"
        )
        
        if use_fixed_nas:
            print(f"Using a fixed NA set for training (n_na={n_na})")
            
            quarter = n_na // 4
            remaining = n_na % 4
            
            group_sizes = [quarter] * 4
            for i in range(remaining):
                group_sizes[i] += 1
            
            print(
                "NA group sizes: "
                f"very_low={group_sizes[0]}, low={group_sizes[1]}, high={group_sizes[2]}, very_high={group_sizes[3]}"
            )
            
            very_low_rep = np.random.uniform(3300, 4500, group_sizes[0])
            low_rep = np.random.uniform(4500, 6600, group_sizes[1])
            high_rep = np.random.uniform(6600, 8200, group_sizes[2])
            very_high_rep = np.random.uniform(8200, 9800, group_sizes[3])
            
            self.fixed_initial_reputation = np.concatenate([very_low_rep, low_rep, high_rep, very_high_rep])
            
            success_rates_list = []
            malicious_flags_list = []
            malicious_types_list = []
            
            # Malicious types:
            # - Type 1: high success + high delay
            # - Type 4: low success (baseline malicious)
            vl_count_low = int(group_sizes[0] * 0.3)
            vl_count_med = int(group_sizes[0] * 0.25) 
            vl_count_high = int(group_sizes[0] * 0.15)
            vl_count_malicious = group_sizes[0] - vl_count_low - vl_count_med - vl_count_high
            
            vl_sr_low = np.random.uniform(0.25, 0.45, vl_count_low)
            vl_sr_med = np.random.uniform(0.55, 0.70, vl_count_med)
            vl_sr_high = np.random.uniform(0.75, 0.90, vl_count_high)
            
            vl_mal_type1 = max(1, vl_count_malicious // 2)
            vl_mal_type4 = vl_count_malicious - vl_mal_type1
            
            vl_sr_mal_type1 = np.random.uniform(0.65, 0.80, vl_mal_type1)
            vl_sr_mal_type4 = np.random.uniform(0.05, 0.30, vl_mal_type4)
            
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
            
            shuffle_indices = np.random.permutation(len(vl_success_rates))
            vl_success_rates = vl_success_rates[shuffle_indices]
            vl_malicious_flags = vl_malicious_flags[shuffle_indices]
            vl_all_malicious_types = vl_all_malicious_types[shuffle_indices]
            
            success_rates_list.append(vl_success_rates)
            malicious_flags_list.append(vl_malicious_flags)
            malicious_types_list.append(vl_all_malicious_types)
            
            l_count_low = int(group_sizes[1] * 0.25)
            l_count_med = int(group_sizes[1] * 0.30)
            l_count_high = int(group_sizes[1] * 0.20)
            l_count_malicious = group_sizes[1] - l_count_low - l_count_med - l_count_high
            
            l_sr_low = np.random.uniform(0.35, 0.55, l_count_low)
            l_sr_med = np.random.uniform(0.60, 0.75, l_count_med)
            l_sr_high = np.random.uniform(0.80, 0.95, l_count_high)
            
            l_mal_type1 = max(1, l_count_malicious // 2)
            l_mal_type4 = l_count_malicious - l_mal_type1
            
            l_sr_mal_type1 = np.random.uniform(0.75, 0.90, l_mal_type1)
            l_sr_mal_type4 = np.random.uniform(0.15, 0.40, l_mal_type4)
            
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
            
            h_count_high = int(group_sizes[2] * 0.45)
            h_count_med = int(group_sizes[2] * 0.25)
            h_count_low = int(group_sizes[2] * 0.10)
            h_count_malicious = group_sizes[2] - h_count_high - h_count_med - h_count_low
            
            h_sr_high = np.random.uniform(0.75, 0.90, h_count_high)
            h_sr_med = np.random.uniform(0.60, 0.75, h_count_med)
            h_sr_low = np.random.uniform(0.40, 0.60, h_count_low)
            
            h_mal_type1 = h_count_malicious
            
            h_sr_mal_type1 = np.random.uniform(0.82, 0.95, h_mal_type1)
            
            h_sr_malicious = h_sr_mal_type1
            h_malicious_types = np.full(h_mal_type1, 1)
            
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
            
            vh_count_very_high = int(group_sizes[3] * 0.55)
            vh_count_high = int(group_sizes[3] * 0.25)
            vh_count_med = int(group_sizes[3] * 0.05)
            vh_count_malicious = group_sizes[3] - vh_count_very_high - vh_count_high - vh_count_med
            
            vh_sr_very_high = np.random.uniform(0.85, 0.98, vh_count_very_high)
            vh_sr_high = np.random.uniform(0.75, 0.85, vh_count_high)
            vh_sr_med = np.random.uniform(0.65, 0.75, vh_count_med)
            
            vh_mal_type1 = vh_count_malicious
            
            vh_sr_mal_type1 = np.random.uniform(0.90, 0.98, vh_mal_type1)
            
            vh_sr_malicious = vh_sr_mal_type1
            vh_malicious_types = np.full(vh_mal_type1, 1)
            
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
            
            self.fixed_initial_success_rate = np.concatenate(success_rates_list)
            self.fixed_malicious_flags = np.concatenate(malicious_flags_list)
            self.fixed_malicious_types = np.concatenate(malicious_types_list)
            
            delay_list = []
            
            for group_idx, group_size in enumerate(group_sizes):
                malicious_mask = malicious_flags_list[group_idx]
                malicious_types = malicious_types_list[group_idx]
                normal_mask = ~malicious_mask
                
                if group_idx == 0:
                    normal_delay = np.random.uniform(0.3, 0.8, np.sum(normal_mask))
                elif group_idx == 1:
                    normal_delay = np.random.uniform(0.25, 0.65, np.sum(normal_mask))
                elif group_idx == 2:
                    normal_delay = np.random.uniform(0.15, 0.5, np.sum(normal_mask))
                else:
                    normal_delay = np.random.uniform(0.1, 0.4, np.sum(normal_mask))
                
                malicious_delay = np.zeros(np.sum(malicious_mask))
                mal_indices = np.where(malicious_mask)[0]
                
                for i, mal_idx in enumerate(mal_indices):
                    mal_type = malicious_types[mal_idx]
                    
                    if mal_type == 1:
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.7, 1.0)
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.6, 0.9)
                        elif group_idx == 2:
                            malicious_delay[i] = np.random.uniform(0.5, 0.8)
                        else:
                            malicious_delay[i] = np.random.uniform(0.4, 0.7)
                            
                    else:
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.6, 0.95)
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.55, 0.9)
                
                group_delay = np.zeros(group_size)
                group_delay[normal_mask] = normal_delay
                group_delay[malicious_mask] = malicious_delay
                delay_list.append(group_delay)
            
            self.fixed_initial_signature_delay = np.concatenate(delay_list)
            
            self.fixed_initial_hunger = np.random.uniform(0.0, 0.5, n_na)
            
            self.fixed_initial_total_tx = np.random.randint(20, 80, n_na)
            self.fixed_initial_success_count = (self.fixed_initial_success_rate * self.fixed_initial_total_tx).astype(int)
            
            self.ideal_delay_grade = 0.0
            self.max_acceptable_delay_grade = 1.0
            
            print("Fixed NA set preview (includes malicious types):")
            print("id\tinit_rep\tsuccess\tdelay_grade\thunger\tdelay_grade2\trep_group\tability\tmalicious")
            
            def get_reputation_group(rep):
                if rep < 4500:
                    return "very_low"
                elif rep < 6600:
                    return "low"
                elif rep < 8200:
                    return "high"
                else:
                    return "very_high"
            
            def get_malicious_type_desc(mal_type):
                if mal_type == 1:
                    return "MAL-T1(high_success_high_delay)"
                elif mal_type == 4:
                    return "MAL-T4(low_success)"
                else:
                    return "normal"
            
            def get_ability_type(rep, success_rate, is_malicious, mal_type):
                if is_malicious:
                    if rep < 4500:
                        return f"malicious_obvious(T{mal_type})"
                    elif rep < 6600:
                        return f"malicious_low(T{mal_type})"
                    elif rep < 8200:
                        return f"malicious_mid(T{mal_type})"
                    else:
                        return f"malicious_high(T{mal_type})"
                
                if rep < 4500:
                    if success_rate >= 0.75:
                        return "hidden_high_performer"
                    elif success_rate >= 0.55:
                        return "medium"
                    else:
                        return "low"
                elif rep < 6600:
                    if success_rate >= 0.80:
                        return "rising_star"
                    elif success_rate >= 0.60:
                        return "medium"
                    else:
                        return "below_average"
                elif rep < 8200:
                    if success_rate >= 0.75:
                        return "high"
                    elif success_rate >= 0.60:
                        return "medium"
                    else:
                        return "unexpected_low"
                else:
                    if success_rate >= 0.85:
                        return "very_high"
                    elif success_rate >= 0.75:
                        return "high"
                    else:
                        return "medium"
            
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
            
            very_low_indices = np.where(self.fixed_initial_reputation < 4500)[0]
            low_indices = np.where((self.fixed_initial_reputation >= 4500) & (self.fixed_initial_reputation < 6600))[0]
            high_indices = np.where((self.fixed_initial_reputation >= 6600) & (self.fixed_initial_reputation < 8200))[0]
            very_high_indices = np.where(self.fixed_initial_reputation >= 8200)[0]
            
            print("\nNA distribution summary (with malicious breakdown):")
            
            def count_malicious_types(indices):
                type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                for i in indices:
                    if self.fixed_malicious_flags[i]:
                        mal_type = self.fixed_malicious_types[i]
                        type_counts[mal_type] += 1
                return type_counts
            
            if len(very_low_indices) > 0:
                vl_malicious = sum(1 for i in very_low_indices if self.fixed_malicious_flags[i])
                vl_normal = len(very_low_indices) - vl_malicious
                vl_mal_types = count_malicious_types(very_low_indices)
                vl_high_ability = sum(1 for i in very_low_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.75)
                vl_med_ability = sum(1 for i in very_low_indices if not self.fixed_malicious_flags[i] and 0.55 <= self.fixed_initial_success_rate[i] < 0.75)
                vl_low_ability = vl_normal - vl_high_ability - vl_med_ability
                print(
                    f"very_low ({len(very_low_indices)}): mal={vl_malicious} "
                    f"[T1:{vl_mal_types[1]}, T2:{vl_mal_types[2]}, T3:{vl_mal_types[3]}, T4:{vl_mal_types[4]}] "
                    f"| normal={vl_normal} (hidden={vl_high_ability}, med={vl_med_ability}, low={vl_low_ability})"
                )
            
            if len(low_indices) > 0:
                l_malicious = sum(1 for i in low_indices if self.fixed_malicious_flags[i])
                l_normal = len(low_indices) - l_malicious
                l_mal_types = count_malicious_types(low_indices)
                l_high_ability = sum(1 for i in low_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.80)
                l_med_ability = sum(1 for i in low_indices if not self.fixed_malicious_flags[i] and 0.60 <= self.fixed_initial_success_rate[i] < 0.80)
                l_low_ability = l_normal - l_high_ability - l_med_ability
                print(
                    f"low ({len(low_indices)}): mal={l_malicious} "
                    f"[T1:{l_mal_types[1]}, T2:{l_mal_types[2]}, T3:{l_mal_types[3]}, T4:{l_mal_types[4]}] "
                    f"| normal={l_normal} (high={l_high_ability}, med={l_med_ability}, low={l_low_ability})"
                )
            
            if len(high_indices) > 0:
                h_malicious = sum(1 for i in high_indices if self.fixed_malicious_flags[i])
                h_normal = len(high_indices) - h_malicious
                h_mal_types = count_malicious_types(high_indices)
                h_high_ability = sum(1 for i in high_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.75)
                h_med_ability = sum(1 for i in high_indices if not self.fixed_malicious_flags[i] and 0.60 <= self.fixed_initial_success_rate[i] < 0.75)
                h_low_ability = h_normal - h_high_ability - h_med_ability
                print(
                    f"high ({len(high_indices)}): mal={h_malicious} | "
                    f"normal={h_normal} (high={h_high_ability}, med={h_med_ability}, low={h_low_ability})"
                )
            
            if len(very_high_indices) > 0:
                vh_malicious = sum(1 for i in very_high_indices if self.fixed_malicious_flags[i])
                vh_normal = len(very_high_indices) - vh_malicious
                vh_very_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.85)
                vh_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and 0.75 <= self.fixed_initial_success_rate[i] < 0.85)
                vh_med_ability = vh_normal - vh_very_high_ability - vh_high_ability
                print(
                    f"very_high ({len(very_high_indices)}): mal={vh_malicious} | "
                    f"normal={vh_normal} (very_high={vh_very_high_ability}, high={vh_high_ability}, med={vh_med_ability})"
                )
            
            total_malicious = np.sum(self.fixed_malicious_flags)
            total_normal = n_na - total_malicious
            print(
                f"\nMalicious total: {total_malicious} ({total_malicious/n_na:.1%}), "
                f"normal total: {total_normal} ({total_normal/n_na:.1%})"
            )
            

        else:
            self.ideal_delay = 100.0
            self.max_acceptable_delay = self.ideal_delay * 3
        
        self.current_time = 30
        
        
        self.reset()
    
    def calculate_delay_performance(self, delay_grade):
        """
        Delay grade in [0.0, 1.0], where larger means slower.
        """
        return max(0.0, min(1.0, delay_grade))
    
    
    def reset(self):
        """
        Reset environment state.
        """
        if self.use_fixed_nas:
            self.reputation = self.fixed_initial_reputation.copy()
            self.total_tx = self.fixed_initial_total_tx.copy()
            self.success_count = self.fixed_initial_success_count.copy()
            self.success_rate = self.fixed_initial_success_rate.copy()
            
            self.signature_delay = self.fixed_initial_signature_delay.copy()
            self.hunger = self.fixed_initial_hunger.copy()
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)
            for i in range(self.n_na):
                if self.hunger[i] > 0:
                    estimated_steps = int(20 * (np.power(11, self.hunger[i]) - 1))
                    self.last_selected_time[i] = -estimated_steps
                else:
                    self.last_selected_time[i] = 0
        else:
            pass
            self.reputation = np.random.uniform(3300, 10000, self.n_na)
            self.total_tx = np.random.randint(1, 100, self.n_na)
            self.success_count = np.array([
                np.random.randint(0, self.total_tx[i] + 1) 
                for i in range(self.n_na)
            ])
            self.success_rate = self.success_count / self.total_tx
            
            # Randomly initialize signature delay
            self.signature_delay = np.random.uniform(50, 200, self.n_na)
            # Randomly initialize hunger (0-50%) to simulate initial load
            self.hunger = np.random.uniform(0.0, 0.5, self.n_na)
            # Record last selected timestep for each NA (estimated from hunger)
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
        # Training step counter for hunger calculation
        self.training_step = 0

        if self.freeze_hunger:
            self.hunger = np.full(self.n_na, self.hunger_constant, dtype=float)
            self.last_selected_time = np.zeros(self.n_na, dtype=int)
        
        # Reset sliding-window queue system for all NAs
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

        # Record initial reputation at the start of this episode
        self.episode_start_reputation = self.reputation.copy()
        # Reset previous-step reputation record
        self.last_step_reputation = self.reputation.copy()

        return self.get_state()
    def get_state(self):
        # Remove activity feature; keep 4 features (reputation, success_rate, signature_delay, hunger)
        raw_state = np.column_stack([
            self.reputation,
            self.success_rate,
            self.signature_delay,
            self.hunger
        ])

        # Normalize
        normalized_state = raw_state.copy()
        normalized_state[:, 0] = (self.reputation - 3000) / (10000 - 3000)  # reputation normalization
        normalized_state[:, 1] = self.success_rate  # already in [0, 1]
        # signature_delay is already a delay grade in [0, 1]
        normalized_state[:, 2] = self.signature_delay
        normalized_state[:, 3] = self.hunger  # already in [0, 1]
        return normalized_state

    def step(self, action_na_index, selected_mask=None):
        """
        Simplified environment step; handles a single NA.
        
        Args:
            action_na_index: int - selected NA index
            selected_mask: kept for compatibility (currently unused)
        
        Returns:
            next_state: updated state
            reward: reward for the selected NA
            done: always False
        """
        # Ensure index is within valid range
        na_idx = max(0, min(int(action_na_index), self.n_na - 1))
        
        # Record reputation and success rate before the action
        old_reputation = self.reputation[na_idx]
        old_success_rate = self.success_rate[na_idx]
        
        # Update time
        self.current_time += 1
        # Update training step
        self.training_step += 1
        # self._update_hunger(na_idx)
        # Simulate delay grade for the current transaction (optionally with random fluctuation)
        base_delay_grade = self.signature_delay[na_idx]
        if self.enable_random_fluctuation:
            # Add ±0.05 random fluctuation (delay grade range 0.0-1.0)
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
        else:
            # Fixed mode: add small fluctuation (±0.01) to simulate measurement noise
            current_delay_grade = base_delay_grade + np.random.uniform(-0.01, 0.01)
        current_delay_grade = np.clip(current_delay_grade, 0.0, 1.0)
        
        # Success-rate random fluctuation (if enabled)
        effective_success_rate = self.success_rate[na_idx]
        if self.enable_random_fluctuation:
            # Success rate can fluctuate ±5% but stays within [0, 1]
            fluctuation = np.random.uniform(-0.05, 0.05)
            effective_success_rate = np.clip(effective_success_rate + fluctuation, 0.0, 1.0)
        
        # Determine transaction success based on (possibly fluctuated) success rate
        success = np.random.random() < effective_success_rate
        
        # Compute reward-related terms (used for reputation update)
        norm_rep = (old_reputation - 3300.0) / (10000.0 - 3300.0)
        hunger_term = self.hunger_weight * self.hunger[na_idx]
        if old_reputation < 6600:
            computed_reward = (0.4 * self.success_rate[na_idx]
                              + 0.4 * norm_rep
                              + 0.2 * hunger_term)
        else:
            computed_reward = (0.4 * self.success_rate[na_idx]
                              + 0.2 * norm_rep
                              + 0.4 * hunger_term)
        
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

        # Reward uses NA-level statistics and is not directly tied to single-transaction success.
        # It is based on weighted success rate, weighted delay grade, and hunger bonus.
        
        # Sliding-window summary for this NA (includes weighting)
        window_summary = self.get_na_window_summary(na_idx)
        
        # Use weighted success rate when window has data; otherwise fall back to current success rate.
        if window_summary['total_transactions'] > 0:
            effective_success_rate = window_summary['weighted_success_rate']
            effective_delay_performance = window_summary['weighted_delay_grade']
        else:
            # Fallback when there is no history
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
        
        # Base reward from weighted success rate; increase range and raise positive threshold.
        weighted_success_rate_reward = 4.0 * effective_success_rate - 2.8
        
        # Delay performance adjustment based on weighted delay grade in [0, 1].
        weighted_delay_grade_bonus = 0.8 * (1 - 2 * effective_delay_performance)
        
        # Hunger bonus (optionally with random fluctuation)
        effective_hunger = self.hunger[na_idx]
        if self.enable_random_fluctuation:
            # Hunger can fluctuate ±10% but stays within [0, 1]
            hunger_fluctuation = np.random.uniform(-0.1, 0.1)
            effective_hunger = np.clip(effective_hunger + hunger_fluctuation, 0.0, 1.0)
        
        # Composite quality weight: 60% weighted success rate + 40% delay performance.
        quality_weight = 0.6 * effective_success_rate + 0.4 * (1 - effective_delay_performance)
        
        # Dynamic hunger weight: higher-quality NAs get larger hunger incentives.
        base_hunger_weight = 0.2 + 1.3 * (quality_weight ** 1.5)
        
        if quality_weight >= 0.8:
            dynamic_hunger_weight = min(1.8, base_hunger_weight * 1.2)
        elif quality_weight >= 0.6:
            dynamic_hunger_weight = base_hunger_weight * 1.0
        elif quality_weight >= 0.4:
            dynamic_hunger_weight = base_hunger_weight * 0.8
        else:
            dynamic_hunger_weight = base_hunger_weight * 0.5
        
        # Compute hunger bonus
        hunger_bonus = dynamic_hunger_weight * quality_weight * effective_hunger
        hunger_bonus = hunger_bonus * self.hunger_weight
        
        # Final reward
        reward = weighted_success_rate_reward + weighted_delay_grade_bonus + hunger_bonus
        
        # Record transaction into the NA sliding-window queue
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
        
        # Pack window queues when reaching the interval
        self.current_step_count += 1
        if self.current_step_count % self.window_pack_interval == 0:
            self._pack_all_na_windows()
        
        # Update hunger after reward calculation
        if self.enable_hunger_update and not self.freeze_hunger:
            self._update_hunger(na_idx)
        
        return self.get_state(), reward, False

    def _update_na_window_queue(self, na_idx, transaction_data):
        """
        Update current transaction pack for the given NA.
        
        Args:
            na_idx: NA index
            transaction_data: dictionary of transaction fields
        """
        current_pack = self.na_current_pack[na_idx]
        
        # Record pack start step
        if len(current_pack['transactions']) == 0:
            current_pack['start_step'] = self.current_step_count
        
        # Append transaction to current pack
        current_pack['transactions'].append(transaction_data)
        current_pack['total_count'] += 1
        if transaction_data['success']:
            current_pack['success_count'] += 1
        
        # Record reputation change
        reputation_change = transaction_data['reputation_after'] - transaction_data['reputation_before']
        current_pack['reputation_changes'].append(reputation_change)
        
        # Update pack end step
        current_pack['end_step'] = self.current_step_count
    
    def _pack_all_na_windows(self):
        """
        Pack all NAs' current transaction packs into their sliding-window queues.
        """
        for na_idx in range(self.n_na):
            current_pack = self.na_current_pack[na_idx]
            
            # Pack only when transactions exist
            if current_pack['total_count'] > 0:
                # Average delay within the pack
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
                    'delay_grade': delay_grade,
                    'transactions': current_pack['transactions'].copy()
                }
                
                # Append to queue (deque will enforce capacity)
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
        Get statistics summary from the NA sliding-window queue.
        
        Args:
            na_idx: NA index
            
        Returns:
            dict: summary of transactions in the window queue
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
        
        # Aggregate statistics from all packs in the queue
        total_transactions = sum(pack['transaction_count'] for pack in queue)
        total_success_count = sum(pack['success_count'] for pack in queue)
        total_reputation_change = sum(pack['total_reputation_change'] for pack in queue)
        
        # Include the current in-progress pack
        total_transactions += current_pack['total_count']
        total_success_count += current_pack['success_count']
        total_reputation_change += sum(current_pack['reputation_changes'])
        
        # Overall success rate
        overall_success_rate = total_success_count / total_transactions if total_transactions > 0 else 0.0
        
        # Average success rate across packs
        pack_success_rates = [pack['success_rate'] for pack in queue if pack['transaction_count'] > 0]
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            pack_success_rates.append(current_pack_success_rate)
        
        avg_pack_success_rate = sum(pack_success_rates) / len(pack_success_rates) if pack_success_rates else 0.0
        
        # Compute weighted success rate and weighted delay (newer packs have higher weight)
        weighted_success_rate = 0.0
        weighted_avg_delay = 0.0
        weighted_delay_grade = 0.0
        total_weight = 0.0
        
        # Compute weights for packs in the queue (newer packs have higher weight)
        if queue:
            for i, pack in enumerate(queue):
                # Linear weights, increasing with recency
                weight = i + 1
                
                # Average delay for this pack
                pack_avg_delay = 0.0
                if pack['transactions']:
                    pack_avg_delay = sum(tx['delay'] for tx in pack['transactions']) / len(pack['transactions'])
                
                # Use delay_grade in pack if present
                pack_delay_grade = pack.get('delay_grade', 0.0)
                
                weighted_success_rate += pack['success_rate'] * weight
                weighted_avg_delay += pack_avg_delay * weight
                weighted_delay_grade += pack_delay_grade * weight
                total_weight += weight
        
        # Include current in-progress pack with the highest weight
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            
            # Average delay in current pack
            current_pack_avg_delay = 0.0
            if current_pack['transactions']:
                current_pack_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions'])
            
            # Delay grade in current pack
            current_pack_delay_grade = self.calculate_delay_performance(current_pack_avg_delay)
            
            # Highest weight for current pack
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
        
        # Default when no weighted delay grade was computed
        if total_weight == 0:
            weighted_delay_grade = 0.5
        
        # Letter grade mapping based on delay grade
        delay_grade = "A"
        if weighted_delay_grade > 0.75:
            delay_grade = "D"
        elif weighted_delay_grade > 0.5:
            delay_grade = "C"
        elif weighted_delay_grade > 0.25:
            delay_grade = "B"
        else:
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
        Print sliding-window summary.
        
        Args:
            max_nas_to_show: maximum number of NAs to show
        """
        print(f"\nSliding-window queue summary (pack interval: {self.window_pack_interval} steps, queue capacity: {self.window_queue_size})")
        print("=" * 140)
        
        summaries = self.get_all_nas_window_summary()
        
        # Sort by weighted success rate (best first)
        sorted_nas = sorted(summaries.items(), key=lambda x: x[1]['weighted_success_rate'], reverse=True)
        
        print(f"{'NA':<6} {'Queue':<6} {'CurPack':<8} {'TotalTx':<8} {'W_Succ':<12} {'W_Delay':<10} {'Grade':<8} {'RepΔ':<10} {'StepRange':<15}")
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
                weight = summary['total_transactions']
                system_weighted_success_rate += summary['weighted_success_rate'] * weight
                system_weighted_delay += summary['weighted_avg_delay'] * weight
                total_weight += weight
        
        if total_weight > 0:
            system_weighted_success_rate /= total_weight
            system_weighted_delay /= total_weight
        
        print("-" * 140)
        print(f"Overall: {active_nas} NAs have transactions, total_tx={total_transactions}, total_success={total_success}")
        if total_transactions > 0:
            print(f"System success rate: {total_success/total_transactions:.3f}, system weighted success: {system_weighted_success_rate:.3f}")
            print(f"System weighted avg delay: {system_weighted_delay:.1%}, tx per active NA: {total_transactions/max(1, active_nas):.1f}")
        print(f"Current step: {self.current_step_count}")
        
        if shown_count < active_nas:
            print(f"(Only showing top {shown_count}; {active_nas - shown_count} more NAs have transactions)")

    def show_na_detailed_queue(self, na_idx, max_packs_to_show=5):
        """
        Show detailed queue for a given NA, including weight calculation.
        
        Args:
            na_idx: NA index
            max_packs_to_show: maximum number of packs to show
        """
        if na_idx >= self.n_na:
            print(f"Invalid NA index: {na_idx}")
            return
        
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\nNA{na_idx} detailed queue")
        print("=" * 100)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("No transactions recorded for this NA.")
            return
        
        # Queue packs (older to newer)
        print(f"Queue packs (total {len(queue)}; newer packs have higher weight):")
        print(f"{'Pack':<8} {'Weight':<6} {'StepRange':<15} {'Tx':<8} {'Succ':<10} {'AvgDelay':<10} {'RepΔ':<10}")
        print("-" * 100)
        
        shown_packs = 0
        for i, pack in enumerate(queue):
            if shown_packs >= max_packs_to_show:
                break
            
            weight = i + 1
            step_range = f"({pack['start_step']}-{pack['end_step']})"
            avg_delay = pack.get('avg_delay', 0.0)
            
            print(f"{i+1:<8} {weight:<6} {step_range:<15} {pack['transaction_count']:<8} {pack['success_rate']:<10.3f} {avg_delay:<10.1%} {pack['total_reputation_change']:<10.1f}")
            shown_packs += 1
        
        # Current in-progress pack
        if current_pack['total_count'] > 0:
            current_weight = len(queue) + 1
            current_success_rate = current_pack['success_count'] / current_pack['total_count']
            current_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions']) if current_pack['transactions'] else 0.0
            current_step_range = f"({current_pack['start_step']}-{current_pack['end_step']})"
            current_reputation_change = sum(current_pack['reputation_changes'])
            
            print(f"Current  {current_weight:<6} {current_step_range:<15} {current_pack['total_count']:<8} {current_success_rate:<10.3f} {current_avg_delay:<10.1%} {current_reputation_change:<10.1f}")
        
        # Weighted results
        summary = self.get_na_window_summary(na_idx)
        print("\nWeighted results:")
        print(f"  - Weighted success rate: {summary['weighted_success_rate']:.3f}")
        print(f"  - Weighted avg delay: {summary['weighted_avg_delay']:.1%}")
        print(f"  - Delay grade: {summary['delay_grade']}")
        print(f"  - Total transactions: {summary['total_transactions']}")
        print(f"  - Queue usage: {len(queue)}/{self.window_queue_size}")
        
        if len(queue) < max_packs_to_show and len(queue) > shown_packs:
            print(f"\n({len(queue) - shown_packs} more packs not shown)")

    def get_all_nas_window_summary(self):
        """
        Get sliding-window summary for all NAs.
        
        Returns:
            dict: summary keyed by NA index
        """
        summaries = {}
        for na_idx in range(self.n_na):
            summaries[na_idx] = self.get_na_window_summary(na_idx)
        return summaries
    
    def print_na_queue_details(self, na_idx, max_packs_to_show=5):
        """
        Print detailed queue information for a given NA.
        
        Args:
            na_idx: NA index
            max_packs_to_show: maximum number of packs to show
        """
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\nNA{na_idx} detailed queue:")
        print("=" * 80)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("  No transactions recorded for this NA.")
            return
        
        # Packed data in queue
        if len(queue) > 0:
            print(f"Packed data (queue size: {len(queue)}/{self.window_queue_size}):")
            
            # Show recent packs
            recent_packs = list(queue)[-max_packs_to_show:]
            for i, pack in enumerate(recent_packs):
                pack_idx = len(queue) - max_packs_to_show + i
                if pack_idx < 0:
                    pack_idx = i
                
                print(f"  Pack#{pack_idx+1}: steps {pack['start_step']}-{pack['end_step']}, "
                      f"tx={pack['transaction_count']}, success={pack['success_rate']:.2f}, "
                      f"rep_delta={pack['total_reputation_change']:+.2f}")
            
            if len(queue) > max_packs_to_show:
                print(f"  ... ({len(queue) - max_packs_to_show} older packs not shown)")
        
        # Current in-progress pack
        if current_pack['total_count'] > 0:
            print("\nCurrent pack:")
            print(f"  steps {current_pack['start_step']}-{current_pack['end_step']}, "
                  f"tx={current_pack['total_count']}, "
                  f"success={current_pack['success_count']} "
                  f"({current_pack['success_count']/current_pack['total_count']:.2f}), "
                  f"rep_delta={sum(current_pack['reputation_changes']):+.2f}")
        else:
            print("\nCurrent pack: empty")
        
        # Summary stats
        summary = self.get_na_window_summary(na_idx)
        print("\nSummary:")
        print(f"  Total transactions: {summary['total_transactions']}")
        print(f"  Total successes: {summary['total_success_count']}")
        print(f"  Overall success rate: {summary['overall_success_rate']:.2f}")
        print(f"  Total reputation delta: {summary['total_reputation_change']:+.2f}")
        if summary['step_range']:
            print(f"  Step range: {summary['step_range'][0]} - {summary['step_range'][1]}")
        print("=" * 80)

    def _update_hunger(self, selected_na_idx):
        """
        Update hunger for all NAs.
        
        Logic:
        1. Reset hunger of the selected NA to 0
        2. Increase hunger of other NAs over time
        3. Clamp hunger to [0, 1]
        
        Args:
            selected_na_idx: currently selected NA index
        """
        # Update record for selected NA
        self.last_selected_time[selected_na_idx] = self.training_step

        # Update hunger for all NAs
        for i in range(self.n_na):
            steps_since_selected = self.training_step - self.last_selected_time[i]

            if steps_since_selected <= 0:
                self.hunger[i] = 0.0
            else:
                normalized_steps = steps_since_selected / self.hunger_growth_scale
                self.hunger[i] = min(1.0, np.log(1 + normalized_steps) / np.log(self.hunger_growth_log_base))

        # Ensure selected NA hunger is reset
        self.hunger[selected_na_idx] = 0.0

def simulate_selected_nas_training(env, selected_nas, title, steps=100):
    """
    Simulate transactions for selected NAs. Only reputation changes are computed.
    
    Args:
        env: environment object
        selected_nas: list of selected NA indices
        title: title
        steps: number of simulated steps
    
    Returns:
        dict: per-NA transaction trajectories
    """
    print(f"\n{title} - NA transaction simulation ({steps} transactions):")
    print("=" * 80)
    
    # Save current environment state (post-training state)
    original_reputation = env.reputation.copy()
    original_success_rate = env.success_rate.copy()
    
    original_current_time = env.current_time
    
    original_total_tx = env.total_tx.copy()
    original_success_count = env.success_count.copy()
    
    # Track trajectories for selected NAs
    training_results = {}
    
    for na_idx in selected_nas:
        # Reset environment to initial fixed state (use initial parameters)
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
        transaction_results = []  # Detailed results for each transaction
        
        print(f"\nNA {na_idx} transaction trace:")
        print(f"   Initial: reputation={initial_rep:.2f}, success_rate={initial_success_rate:.3f}, signature_delay={initial_signature_delay:.1f}ms, hunger={initial_hunger:.3f}")
        
        # Simulate transaction processing
        for step in range(steps):
            # Record pre-transaction state
            old_reputation = env.reputation[na_idx]
            old_success_rate = env.success_rate[na_idx]
            old_hunger = env.hunger[na_idx]
            
            env.current_time += 1
            
            
            # Simulate signature delay grade for the current transaction (with random fluctuation)
            base_delay_grade = env.signature_delay[na_idx]
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
            current_delay_grade = max(0.0, min(1.0, current_delay_grade))
            
            # Determine transaction success
            if current_delay_grade > env.max_acceptable_delay_grade:
                success = False
                failure_reason = "signature delay grade too high"
            else:
                # Use base success rate without delay-performance adjustment
                success = np.random.random() < env.success_rate[na_idx]
                failure_reason = None
            
            # Reputation update parameters
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
        
        # Aggregate statistics
        final_rep = reputation_history[-1]
        total_rep_change = final_rep - initial_rep
        successful_transactions = sum(1 for t in transaction_results if t['success'])
        transaction_success_rate = successful_transactions / steps
        avg_signature_delay = np.mean([t['signature_delay'] for t in transaction_results])
        final_success_rate = env.success_rate[na_idx]
        final_hunger = env.hunger[na_idx]
        
        # Failure reasons
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
        print(f"   Reputation: {initial_rep:.2f} → {final_rep:.2f} ({total_rep_change:+.2f})")
        print(f"   Success rate: {initial_success_rate:.3f} → {final_success_rate:.3f}")
        print(f"   Transaction success: {successful_transactions}/{steps} ({transaction_success_rate:.1%})")
        print(f"   Avg signature delay grade: {avg_signature_delay:.1%}")
        print(f"   Failures: delay_too_high={delay_failures}, random={random_failures}")
        print(f"   Hunger: {initial_hunger:.3f} → {final_hunger:.3f}")
    
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
    1) policy distribution
    2) policy entropy curve
    3) NA selection frequency heatmap
    """
    print("\n" + "="*80)
    print("Creating policy analysis plots")
    print("="*80)
    
    # Font setup (fallback list)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create 3x1 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # 1) Policy distribution - action probabilities in recent episodes
    ax1 = axes[0]
    
    if len(agent.action_distribution_history) > 0:
        # Average the last N distributions
        recent_distributions = agent.action_distribution_history[-10:]
        avg_distribution = np.mean(recent_distributions, axis=0)
        
        # NA indices
        na_indices = np.arange(len(avg_distribution))
        
        # Color by initial reputation group
        colors = []
        for i in range(env.n_na):
            if env.episode_start_reputation[i] < 6600:
                colors.append('lightcoral')
            else:
                colors.append('lightblue')
        
        bars = ax1.bar(na_indices, avg_distribution, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Annotate top-selected NAs
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
        
        print("Policy distribution plot: done (averaged over last 10 episodes)")
        print(f"  Top-5 selection probabilities: {[f'NA{i}({avg_distribution[i]:.3f})' for i in top_5_indices]}")
    else:
        ax1.text(0.5, 0.5, 'No Policy Distribution Data Available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Policy Distribution - No Data')
    
    # 2) Policy entropy curve
    ax2 = axes[1]
    
    if len(agent.policy_entropy_history) > 0:
        episodes = np.arange(len(agent.policy_entropy_history))
        entropies = agent.policy_entropy_history
        
        # Plot entropy curve
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
            
            # Annotation
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
        
        # Entropy trend
        if len(entropies) > 1:
            entropy_change = entropies[-1] - entropies[0]
            if entropy_change < -0.1:
                trend_text = "policy becomes more deterministic (entropy decreases)"
            elif entropy_change > 0.1:
                trend_text = "policy becomes more random (entropy increases)"
            else:
                trend_text = "policy is relatively stable"
        else:
            trend_text = "insufficient data"
            
        print(f"Policy entropy curve: done ({trend_text})")
        print(f"  Entropy change: {entropies[0]:.3f} → {entropies[-1]:.3f} (Δ{entropy_change:.3f})")
    else:
        ax2.text(0.5, 0.5, 'No Policy Entropy Data Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Policy Entropy - No Data')
    
    # 3) NA selection frequency heatmap
    ax3 = axes[2]
    
    if np.sum(agent.na_selection_frequency) > 0:
        # Normalize selection frequencies to percentages
        selection_freq_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        
        # Reshape into matrix for display
        n_cols = min(10, env.n_na)
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
        
        # Axis ticks/labels
        ax3.set_xticks(range(n_cols))
        ax3.set_yticks(range(n_rows))
        ax3.set_xticklabels([f'Col{i}' for i in range(n_cols)])
        ax3.set_yticklabels([f'Row{i}' for i in range(n_rows)])
        
        ax3.set_title('NA Selection Frequency Heatmap\n(Percentage of times each NA was selected)', 
                     fontsize=12, fontweight='bold')
        
        # Frequency analysis
        most_selected = np.argmax(selection_freq_percent)
        least_selected = np.argmin(selection_freq_percent[selection_freq_percent > 0])
        
        print("NA selection frequency heatmap: done")
        print(f"  Most selected: NA{most_selected} ({selection_freq_percent[most_selected]:.1f}%)")
        print(f"  Least selected: NA{least_selected} ({selection_freq_percent[least_selected]:.1f}%)")
        
        # Concentration / potential overfitting
        top_3_percent = np.sum(np.sort(selection_freq_percent)[-3:])
        if top_3_percent > 70:
            print(f"  Warning: Top-3 NAs take {top_3_percent:.1f}% selections; possible overfitting")
        else:
            print(f"  Distribution looks reasonable: Top-3 take {top_3_percent:.1f}%")
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
        print("1. Entropy")
        print(f"   - Initial: {agent.policy_entropy_history[0]:.3f}")
        print(f"   - Final: {agent.policy_entropy_history[-1]:.3f}")
        print(f"   - Max possible: {np.log(env.n_na):.3f} (fully random)")
        print(f"   - Determinism: {(1 - agent.policy_entropy_history[-1]/np.log(env.n_na))*100:.1f}%")
    
    if np.sum(agent.na_selection_frequency) > 0:
        print("\n2. NA selection")
        selection_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        selected_nas = np.where(selection_percent > 0)[0]
        print(f"   - Selected NAs: {len(selected_nas)}/{env.n_na}")
        print(f"   - Diversity: {len(selected_nas)/env.n_na*100:.1f}%")
        
        # By reputation group
        low_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                   if env.episode_start_reputation[i] < 6600])
        high_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                    if env.episode_start_reputation[i] >= 6600])
        print(f"   - Low-rep group share: {low_rep_selections:.1f}%")
        print(f"   - High-rep group share: {high_rep_selections:.1f}%")
    
    if len(agent.action_distribution_history) > 0:
        print("\n3. Distribution")
        recent_dist = np.mean(agent.action_distribution_history[-10:], axis=0)
        top_5_nas = np.argsort(recent_dist)[-5:]
        print(f"   - Top-5 NAs: {[f'NA{i}({recent_dist[i]:.2f})' for i in top_5_nas]}")
        concentration = np.sum(np.sort(recent_dist)[-5:])
        print(f"   - Top-5 concentration: {concentration:.1%}")
        
        if concentration > 0.8:
            print("   - Warning: policy is highly concentrated; possible overfitting")
        elif concentration < 0.3:
            print("   - Warning: policy is too diffuse; possible undertraining")
        else:
            print("   - Concentration looks reasonable")

def train_dqn(n_na, n_episodes, steps_per_episode, lr, 
              update_frequency, use_fixed_nas, enable_random_fluctuation,
              generate_episode_gifs=False,
              freeze_hunger=False, hunger_constant=0.0, hunger_weight=1.0, enable_hunger_update=True):
    
    # Parameter checks
    print("Training parameter check:")
    print(f"  NA count: {n_na}")
    print(f"  Episodes: {n_episodes}")  
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Learning rate: {lr}")
    print(f"  Update frequency: every {update_frequency} steps")
    print(f"  Use fixed NA set: {use_fixed_nas}")
    print(f"  Random fluctuation mode: {enable_random_fluctuation}")
    
    # Sanity checks
    total_steps = n_episodes * steps_per_episode
    print(f"  Total training steps: {total_steps}")
    
    if n_na < 5:
        print("Warning: NA count is small; learning may be affected")
    if n_na > 30:
        print("Warning: NA count is large; may need more training steps")
    if lr > 0.01:
        print("Warning: learning rate is high; training may be unstable")
    if total_steps < 1000:
        print("Warning: total training steps are low; network may underfit")
    
    # Create environment
    env = Environment(
        n_na,
        use_fixed_nas=use_fixed_nas,
        enable_random_fluctuation=enable_random_fluctuation,
        freeze_hunger=freeze_hunger,
        hunger_constant=hunger_constant,
        hunger_weight=hunger_weight,
        enable_hunger_update=enable_hunger_update
    )
    agent = DQNAgent(n_features=4, n_na=n_na, lr=lr, gamma=0.9,  # 4 features after removing activity; lower gamma to increase Q-value separation
                    epsilon_start=1.0, epsilon_end=0.01, decay_steps=15000,
                    memory_size=50000, batch_size=256, target_update=20,  # target network update interval
                    min_memory_size=2000, update_frequency=update_frequency, 
                    total_episodes=n_episodes)
    history = []
    # Reputation trajectory in the last episode
    reputation_last = []
    
    # Detailed data for the first and last episode
    first_episode_data = []
    last_episode_data = []
    
    # Reward tracking
    step_rewards = []
    step_reward_steps = []
    reward_capture_interval = 3
    step_reward_buffer = []
    success_rates_per_episode = []
    
    # Training stats
    total_training_steps = 0
    total_updates = 0
    
    # Warm-up phase: collect experience without training
    print("Warm-up: collecting initial experience...")
    warmup_steps = 0
    while len(agent.memory) < agent.min_memory_size:
        state = env.reset()
        for _ in range(10):
            action = agent.select_action(state, env.episode_start_reputation)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            warmup_steps += 1
            
            # Stop when reaching minimum replay size
            if len(agent.memory) >= agent.min_memory_size:
                break
    
    print(f"Warm-up done: collected {len(agent.memory)} transitions in {warmup_steps} steps")
    print("Training strategy: fixed NA set, select one NA per step")
    print(f"Start training: train every {update_frequency} steps")
    
    # Print fixed NA features
    print("\nFixed NA set features (constant throughout training):")
    print("ID\tInitRep\t\tSuccRate\tSigDelay(ms)\tHunger\t\tDelayGrade\tGroup")
    for i in range(n_na):
        rep = env.fixed_initial_reputation[i] if env.use_fixed_nas else env.reputation[i]
        sr = env.fixed_initial_success_rate[i] if env.use_fixed_nas else env.success_rate[i]
        delay = env.fixed_initial_signature_delay[i] if env.use_fixed_nas else env.signature_delay[i]
        hunger = env.hunger[i] if env.freeze_hunger else (env.fixed_initial_hunger[i] if env.use_fixed_nas else env.hunger[i])
        delay_perf = env.calculate_delay_performance(delay)
        group = "Low-rep" if rep < 6600 else "High-rep"
        print(f"{i}\t{rep:.1f}\t\t{sr:.3f}\t{delay:.1f}\t\t{hunger:.1%}\t\t{delay_perf:.1%}\t\t{group}")
    
    for ep in range(n_episodes):
        # Update epsilon by episode
        agent.update_epsilon_by_episode(ep)
        
        # Update learning rate by episode
        current_lr = agent.update_learning_rate(ep)
        
        # Record learning rate and epsilon history
        if ep % 10 == 0:
            agent.lr_history.append(current_lr)
            agent.epsilon_history.append(agent.epsilon)
        
        # Print learning rate every 100 episodes
        if ep % 100 == 0 and ep > 0:
            print(f"Episode {ep}: Learning Rate = {current_lr:.6f}")
        
        state = env.reset()
        
        # Print NA params for first and last episode
        if ep == 0:
            print(f"\nFirst episode (#{ep}) NA params check:")
            print("ID\treputation\tsuccess_rate\tsig_delay(ms)\tdelay_grade\tgroup")
            for i in range(min(n_na, 10)):
                group = "Low-rep" if env.reputation[i] < 6600 else "High-rep"
                delay_perf = env.calculate_delay_performance(env.signature_delay[i])
                print(f"{i}\t{env.reputation[i]:.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.3f}\t\t{group}")
            if n_na > 10:
                print(f"... (total {n_na} NAs)")
        elif ep == n_episodes - 1:
            print(f"\nLast episode (#{ep}) NA params (should match initial):")
            print("ID\treputation\tsuccess_rate\tsig_delay(ms)\tdelay_grade\tgroup")
            for i in range(min(n_na, 10)):
                group = "Low-rep" if env.reputation[i] < 6600 else "High-rep"
                delay_perf = env.calculate_delay_performance(env.signature_delay[i])
                print(f"{i}\t{env.reputation[i]:.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{group}")
            if n_na > 10:
                print(f"... (total {n_na} NAs)")
        
        # Episode stats
        ep_reward = 0
        ep_step_rewards = []
        ep_successes = 0
        
        # Episode details for first/last
        is_first_episode = (ep == 0)
        is_last_episode = (ep == n_episodes - 1)
        
        if is_first_episode or is_last_episode:
            episode_reputation_history = [env.reputation.copy()]
            episode_selected_nas = []
        
        # Save initial reputations for the last episode
        if is_last_episode:
            reputation_last.append(env.reputation.copy())
        
        for t in range(steps_per_episode):
            # Record policy diagnostics every 10 episodes and in the last episode
            record_policy = (ep % 10 == 0) or is_last_episode
            action_na = agent.select_action(state, initial_reputations=env.episode_start_reputation, record_policy=record_policy)
            next_state, reward, done = env.step(action_na)
            agent.store(state, action_na, reward, next_state, done)
            
            total_training_steps += 1
            
            # Conditional update instead of training every step
            batch_losses = agent.maybe_update(global_step=total_training_steps)
            if batch_losses is not None:
                total_updates += len(batch_losses)
                # Only print detailed training info in the last episode
                if ep == n_episodes - 1:
                    avg_loss = np.mean(batch_losses)
                
            # Accumulate rewards
            ep_reward += reward
            ep_step_rewards.append(reward)

            # Aggregate rewards to reduce jitter
            step_reward_buffer.append(reward)
            if len(step_reward_buffer) >= reward_capture_interval:
                averaged_reward = sum(step_reward_buffer) / len(step_reward_buffer)
                step_rewards.append(averaged_reward)
                step_reward_steps.append(total_training_steps)
                step_reward_buffer.clear()
            
            # Count "success" by positive reward
            if reward > 0:
                ep_successes += 1
            
            state = next_state
            
            # Record details for first/last episode
            if is_first_episode or is_last_episode:
                episode_reputation_history.append(env.reputation.copy())
                episode_selected_nas.append(action_na)
        if is_last_episode:
            reputation_last.append(env.reputation.copy())
        
        # Episode summary statistics
        ep_success_rate = ep_successes / steps_per_episode if steps_per_episode > 0 else 0
        ep_avg_step_reward = np.mean(ep_step_rewards) if ep_step_rewards else 0
        ep_reward_std = np.std(ep_step_rewards) if ep_step_rewards else 0
        
        # Q-value statistics every 10 episodes
        if ep % 10 == 0:
            with torch.no_grad():
                current_state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                all_q_values = agent.policy_net(current_state).squeeze(0).cpu().numpy()
                q_mean = np.mean(all_q_values)
                q_max = np.max(all_q_values)
                q_min = np.min(all_q_values)
                q_std = np.std(all_q_values)
        else:
            # Reuse previous Q stats
            if ep > 0 and len(history) > 0:
                last_q_data = history[-1]
                q_mean = last_q_data.get('q_mean', 0)
                q_max = last_q_data.get('q_max', 0)
                q_min = last_q_data.get('q_min', 0)
                q_std = last_q_data.get('q_std', 0)
            else:
                q_mean = q_max = q_min = q_std = 0
        
        # Store for analysis
        success_rates_per_episode.append(ep_success_rate)
        
        # Save first/last episode data
        if is_first_episode:
            first_episode_data = {
                'reputation_history': episode_reputation_history,
                'selected_nas': episode_selected_nas,
                'episode_num': ep + 1
            }
            # Export NA parameters at the end of the first episode
            try:
                print("\nEnd of first episode: exporting NA parameters...")
                export_file = agent.export_na_selection_parameters(
                    state=state, 
                    initial_reputations=env.episode_start_reputation,
                    env=env
                )
                print(f"First-episode NA export: success. File: {export_file}")
            except Exception as e:
                print(f"First-episode NA export: failed: {e}")
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
            memory_status = f"Memory: {len(agent.memory)}/{agent.min_memory_size}"
            update_ratio = total_updates / max(1, total_training_steps) * 100
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Mean Reputation: {env.reputation.mean():.2f}, "
                  f"Success Rate: {ep_success_rate:.3f}, Avg Step Reward: {ep_avg_step_reward:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}, {memory_status}, Update Rate: {update_ratio:.1f}%")
        
        # Q-value learning quality check every 50 episodes
        if (ep + 1) % 50 == 0 and ep > 0:
            print(f"\nEpisode {ep+1} - Q-value learning quality check:")
            q_values = agent.debug_q_values(state, env.episode_start_reputation)
            
            # Check whether the network learned a reasonable selection strategy
            low_rep_indices = np.where(env.episode_start_reputation < 6600)[0]
            high_rep_indices = np.where(env.episode_start_reputation >= 6600)[0]
            
            if len(low_rep_indices) > 0 and len(high_rep_indices) > 0:
                # Group distribution within top-5 Q-value NAs
                top_5_indices = np.argsort(q_values.cpu().numpy())[-5:]
                top_5_low_count = np.sum([idx in low_rep_indices for idx in top_5_indices])
                top_5_high_count = np.sum([idx in high_rep_indices for idx in top_5_indices])
                
                print(f"  Top-5 Q-value NAs: low-rep={top_5_low_count}, high-rep={top_5_high_count}")
                
                # Quality assessment
                if top_5_high_count >= 3:
                    print("  Quality: good (prefers high-rep NAs)")
                elif top_5_high_count >= 2:
                    print("  Quality: ok (selection strategy can improve)")
                else:
                    print("  Quality: poor (may need parameter tuning)")
            
            # Sliding-window summary every 100 episodes
            if (ep + 1) % 100 == 0:
                env.print_sliding_window_summary(max_nas_to_show=10)
        
        # Key epsilon transition points
        first_transition = int(0.15 * n_episodes)
        second_transition = int(0.5 * n_episodes)
        
        if ep == first_transition or ep == first_transition + 1:
            print(f"Episode {ep+1}: epsilon transition 1, ε={agent.epsilon:.4f} "
                  f"({'very_fast→fast' if ep == first_transition else 'fast_decay'})")
        elif ep == second_transition or ep == second_transition + 1:
            print(f"Episode {ep+1}: epsilon transition 2, ε={agent.epsilon:.4f} "
                  f"({'fast→fine_tuning' if ep == second_transition else 'fine_tuning'})")
        elif ep in [n_episodes-10, n_episodes-5, n_episodes-1]:
            print(f"Episode {ep+1}: nearing the end of training, ε={agent.epsilon:.4f}")
    
    # Flush remaining aggregated rewards
    if step_reward_buffer:
        averaged_reward = sum(step_reward_buffer) / len(step_reward_buffer)
        step_rewards.append(averaged_reward)
        step_reward_steps.append(total_training_steps)

    # Ensure last episode learning rate and epsilon are recorded
    final_episode = n_episodes - 1
    if final_episode % 10 != 0:
        current_lr = agent.update_learning_rate(final_episode)
        agent.lr_history.append(current_lr)
        agent.epsilon_history.append(agent.epsilon)
        print(f"Final episode record ({final_episode}): Learning Rate = {current_lr:.6f}, Epsilon = {agent.epsilon:.6f}")
    
    # Training stats
    print("\nTraining stats:")
    print(f"Total steps: {total_training_steps}")
    print(f"Total updates: {total_updates}")
    print(f"Update frequency: try every {update_frequency} steps")
    print(f"Actual update ratio: {total_updates/max(1, total_training_steps)*100:.2f}%")
    print(f"Target net updates: {agent.learn_step//agent.target_update}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    
    # Key checks
    print("\nPost-training key checks:")
    
    # 1) Check whether network parameters updated
    final_state = env.get_state()
    with torch.no_grad():
        initial_q = torch.zeros(n_na)
        final_q = agent.policy_net(torch.FloatTensor(final_state).unsqueeze(0).to(agent.device)).squeeze().cpu()
    
    q_magnitude = torch.norm(final_q).item()
    print(f"  Q-network output magnitude: {q_magnitude:.4f}")
    
    if q_magnitude < 0.1:
        print("  Warning: network may be undertrained")
    elif q_magnitude > 100:
        print("  Warning: network may be overfitting or LR is too high")
    else:
        print("  Magnitude looks normal")
    
    # 2) Replay buffer state
    print(f"  Replay buffer usage: {len(agent.memory)}/{agent.min_memory_size} "
          f"({len(agent.memory)/agent.min_memory_size*100:.1f}%)")
    
    # 3) Training frequency efficiency
    expected_updates = total_training_steps // update_frequency
    update_efficiency = total_updates / max(1, expected_updates) * 100
    print(f"  Update efficiency: {update_efficiency:.1f}% (expected {expected_updates}, got {total_updates})")
    
    # 4) Loss convergence
    if len(agent.loss_history) > 100:
        early_loss = np.mean(agent.loss_history[:50])
        late_loss = np.mean(agent.loss_history[-50:])
        loss_improvement = (early_loss - late_loss) / early_loss * 100
        print(f"  Loss improvement: {loss_improvement:.1f}% (early:{early_loss:.4f} → late:{late_loss:.4f})")
        
        if loss_improvement > 20:
            print("  Loss converges well")
        elif loss_improvement > 5:
            print("  Warning: limited improvement; may need more training")
        else:
            print("  Warning: no clear improvement; check hyperparameters")
    else:
        print("  Warning: insufficient data to evaluate loss convergence")
    
    # Final selection using trained DQN
    print("\n" + "="*80)
    print("Final NA selection using trained DQN")
    print("Training mode: fixed NA set, select one NA per step, final set is formed by repeated selection")
    print("="*80)
    
    final_state = env.get_state()
    selection_result = agent.make_final_selection(final_state, env.episode_start_reputation)
    
    dqn_selected = selection_result['dqn_selected']
    random_selected = selection_result['random_selected']
    all_q_values = selection_result['all_q_values']
    low_rep_indices = selection_result['low_rep_indices']
    high_rep_indices = selection_result['high_rep_indices']
    
    print("\nNA state at the end of the episode:")
    print("ID\tInitRep\t\tCurRep\t\tRepΔ\t\tsuccess_rate\tsig_delay(ms)\tdelay_grade\tQ\t\tgroup")
    for i in range(min(n_na, 15)):
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "Low-init-rep" if initial_rep < 6600 else "High-init-rep"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        print(f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}")
    if n_na > 15:
        print(f"... (total {n_na} NAs, showing first 15)")
    
    print(f"\n[DQN] NA set built by repeated single selections (size={len(dqn_selected)}, target: 2 low + 3 high):")
    print("ID\tInitRep\t\tCurRep\t\tRepΔ\t\tsuccess_rate\tsig_delay(ms)\tdelay_grade\tQ\t\tgroup\t\tmalicious")
    dqn_low_count = 0
    dqn_high_count = 0
    dqn_total_reputation = 0
    dqn_total_q_value = 0
    dqn_malicious_count = 0  # Count malicious NAs in DQN selection
    
    for i in dqn_selected:
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "Low-init-rep" if initial_rep < 6600 else "High-init-rep"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        
        # Malicious NA check
        is_malicious = env.fixed_malicious_flags[i]
        if is_malicious:
            dqn_malicious_count += 1
            mal_type = env.fixed_malicious_types[i]
            if mal_type == 1:
                malicious_info = "malicious T1 (high success + high delay)"
            elif mal_type == 4:
                malicious_info = "malicious T4 (low success)"
            else:
                malicious_info = "malicious (unknown type)"
        else:
            malicious_info = "normal"
            
        if initial_rep < 6600:
            dqn_low_count += 1
        else:
            dqn_high_count += 1
        dqn_total_reputation += current_rep
        dqn_total_q_value += all_q_values[i]
        print(f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}\t{malicious_info}")
    
    # Malicious count in DQN selection
    if dqn_malicious_count > 0:
        print(f"Warning: DQN selection contains {dqn_malicious_count} malicious NAs")
    
    print(f"\n[Random] NA set (size={len(random_selected)}) for comparison:")
    print("ID\tInitRep\t\tCurRep\t\tRepΔ\t\tsuccess_rate\tsig_delay(ms)\tdelay_grade\tQ\t\tgroup\t\tmalicious")
    random_low_count = 0
    random_high_count = 0
    random_total_reputation = 0
    random_total_q_value = 0
    random_malicious_count = 0  # Count malicious NAs in random selection
    
    for i in random_selected:
        initial_rep = env.episode_start_reputation[i]
        current_rep = env.reputation[i]
        rep_change = current_rep - initial_rep
        group = "Low-init-rep" if initial_rep < 6600 else "High-init-rep"
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        
        # Malicious NA check
        is_malicious = env.fixed_malicious_flags[i]
        if is_malicious:
            random_malicious_count += 1
            mal_type = env.fixed_malicious_types[i]
            if mal_type == 1:
                malicious_info = "malicious T1 (high success + high delay)"
            elif mal_type == 4:
                malicious_info = "malicious T4 (low success)"
            else:
                malicious_info = "malicious (unknown type)"
        else:
            malicious_info = "normal"
            
        if initial_rep < 6600:
            random_low_count += 1
        else:
            random_high_count += 1
        random_total_reputation += current_rep
        random_total_q_value += all_q_values[i]
    print(f"{i}\t{initial_rep:.2f}\t\t{current_rep:.2f}\t\t{rep_change:+.2f}\t\t{env.success_rate[i]:.4f}\t\t{env.signature_delay[i]:.1f}\t\t{delay_perf:.1%}\t\t{all_q_values[i]:.4f}\t\t{group}\t{malicious_info}")
    
    # Random selection malicious NA summary
    if random_malicious_count > 0:
        print(f"Warning: random selection contains {random_malicious_count} malicious NAs (expected under random selection).")
    
    # Simulate the full training process for the selected NAs
    dqn_training_results = simulate_selected_nas_training(env, dqn_selected, "DQN selection training simulation", steps=100)
    
    # Sliding window statistics summary
    env.print_sliding_window_summary(max_nas_to_show=15)
    random_training_results = simulate_selected_nas_training(env, random_selected, "Random selection training simulation", steps=100)
    
    print(f"\nSelection strategy comparison (full training simulation):")
    print(f"{'Metric':<25} {'DQN':<15} {'Random':<15} {'DQN-Note':<15}")
    print("-" * 70)
    print(f"{'Low-init group count':<25} {dqn_low_count:<15} {random_low_count:<15} {'OK' if dqn_low_count >= 2 else 'NO':<15}")
    print(f"{'High-init group count':<25} {dqn_high_count:<15} {random_high_count:<15} {'OK' if dqn_high_count >= 3 else 'NO':<15}")
    print(f"{'Malicious NA count':<25} {dqn_malicious_count:<15} {random_malicious_count:<15} {'fewer' if dqn_malicious_count < random_malicious_count else ('same' if dqn_malicious_count == random_malicious_count else 'more'):<15}")
    
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
    
    print(f"{'Avg init reputation':<25} {dqn_avg_initial_rep:<15.2f} {random_avg_initial_rep:<15.2f} {dqn_avg_initial_rep - random_avg_initial_rep:+.2f}")
    print(f"{'Avg final reputation':<25} {dqn_avg_final_rep:<15.2f} {random_avg_final_rep:<15.2f} {dqn_avg_final_rep - random_avg_final_rep:+.2f}")
    print(f"{'Avg reputation change':<25} {dqn_avg_rep_change:<15.2f} {random_avg_rep_change:<15.2f} {dqn_avg_rep_change - random_avg_rep_change:+.2f}")
    print(f"{'Tx success rate':<25} {dqn_avg_transaction_success_rate:<15.1%} {random_avg_transaction_success_rate:<15.1%} {'+' if dqn_avg_transaction_success_rate > random_avg_transaction_success_rate else ''}{(dqn_avg_transaction_success_rate - random_avg_transaction_success_rate):.1%}")
    print(f"{'Total successful tx':<25} {dqn_total_successful_transactions:<15} {random_total_successful_transactions:<15} {dqn_total_successful_transactions - random_total_successful_transactions:+}")
    print(f"{'Avg sig delay(ms)':<25} {dqn_avg_signature_delay:<15.1f} {random_avg_signature_delay:<15.1f} {dqn_avg_signature_delay - random_avg_signature_delay:+.1f}")
    
    # Create selection strategy comparison GIF
    print(f"\nCreating selection strategy comparison GIF...")
    create_comparison_strategy_gif(env, dqn_selected, random_selected, dqn_training_results, random_training_results)
    
    dqn_avg_q_value = dqn_total_q_value / len(dqn_selected) if dqn_selected else 0
    random_avg_q_value = random_total_q_value / len(random_selected) if random_selected else 0
    print(f"{'Avg Q-value':<25} {dqn_avg_q_value:<15.4f} {random_avg_q_value:<15.4f} {dqn_avg_q_value - random_avg_q_value:+.4f}")
    
    print(f"\nQ-value analysis (fixed NA set):")
    if len(low_rep_indices) > 0:
        low_q_values = all_q_values[low_rep_indices]
        print(f"Low-init group Q range: {low_q_values.min():.4f} ~ {low_q_values.max():.4f} (mean: {low_q_values.mean():.4f})")
    if len(high_rep_indices) > 0:
        high_q_values = all_q_values[high_rep_indices]
        print(f"High-init group Q range: {high_q_values.min():.4f} ~ {high_q_values.max():.4f} (mean: {high_q_values.mean():.4f})")
    
    # Signature delay analysis
    print(f"\nSignature delay performance analysis:")
    all_delays = [env.signature_delay[i] for i in range(n_na)]
    all_delay_perfs = [env.calculate_delay_performance(delay) for delay in all_delays]
    
    print(f"Signature delay range: {min(all_delays):.1f}ms ~ {max(all_delays):.1f}ms (mean: {np.mean(all_delays):.1f}ms)")
    print(f"Delay grade range: {min(all_delay_perfs):.3f} ~ {max(all_delay_perfs):.3f} (mean: {np.mean(all_delay_perfs):.3f})")
    
    # Group delay analysis
    if len(low_rep_indices) > 0:
        low_delays = [all_delays[i] for i in low_rep_indices]
        low_delay_perfs = [all_delay_perfs[i] for i in low_rep_indices]
        print(f"Low-init group delay: {np.mean(low_delays):.1f}ms (grade: {np.mean(low_delay_perfs):.3f})")
    if len(high_rep_indices) > 0:
        high_delays = [all_delays[i] for i in high_rep_indices]
        high_delay_perfs = [all_delay_perfs[i] for i in high_rep_indices]
        print(f"High-init group delay: {np.mean(high_delays):.1f}ms (grade: {np.mean(high_delay_perfs):.3f})")
    
    # Delay grade interpretation
    print(f"Delay grade: range 0.0-1.0 (0.0=fastest, 1.0=slowest)")
    
    # Correlation analysis between Q-values and composite quality
    print(f"\nQ-value learning effectiveness analysis:")
    print("ID\tInitRep\tSuccessRate\tQ\t\tSigDelay\tHunger\tDelayGrade\tComposite\tReasonable")
    
    # Compute per-NA composite quality score (range 0-1)
    na_scores = []
    for i in range(n_na):
        initial_rep = env.episode_start_reputation[i]
        success_rate = env.success_rate[i]
        delay_perf = env.calculate_delay_performance(env.signature_delay[i])
        rep_score = (initial_rep - 3300) / (10000 - 3300)
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
        
        # Heuristic sanity check: Q-values should roughly correlate with composite scores
        q_percentile = (q_val - q_values_array.min()) / (q_values_array.max() - q_values_array.min())
        score_percentile = (composite_score - na_scores.min()) / (na_scores.max() - na_scores.min())
        
        # Acceptable mismatch tolerance: +/- 0.3 percentile
        rank_diff = abs(q_percentile - score_percentile)
        if rank_diff <= 0.3:
            reasonable = "OK"
        elif rank_diff <= 0.5:
            reasonable = "MID"
        else:
            reasonable = "BAD"
        
        print(f"{i}\t{initial_rep:.1f}\t\t{success_rate:.3f}\t{q_val:.4f}\t\t{delay:.1%}\t\t{env.hunger[i]:.1%}\t\t{delay_perf:.1%}\t\t{composite_score:.3f}\t\t{reasonable}")
    
    print(f"\nQ-value learning effectiveness summary:")
    print(f"Correlation coefficient (Q vs composite): {correlation:.3f}")
    if correlation > 0.7:
        print("Excellent: Q-values strongly reflect composite NA quality")
    elif correlation > 0.5:
        print("Good: Q-values generally reflect composite NA quality")
    elif correlation > 0.3:
        print("Fair: Q-values partially reflect composite NA quality; consider more training")
    else:
        print("Poor: weak correlation between Q-values and composite NA quality; review training parameters")
    
    # Use DQN selection as the final selected set
    selected_idx = dqn_selected

    # Plot last-episode reputation trajectories
    plt.figure(figsize=(12, 8))
    for i in range(n_na):
        # Style based on whether the NA is selected
        if i in selected_idx:
            # DQN-selected NAs: thicker line and marker
            plt.plot(range(len(reputation_last)), [r[i] for r in reputation_last], 
                    linewidth=3, marker='o', markersize=4, markevery=10,
                    label=f'[DQN] NA {i} (DQN Selected)')
        else:
            # Not selected: thinner line
            plt.plot(range(len(reputation_last)), [r[i] for r in reputation_last], 
                    linewidth=1, alpha=0.7, label=f'NA {i}')
    
    # Mark final selected NAs
    for i in selected_idx:
        final_reputation = env.reputation[i]
        plt.scatter(len(reputation_last)-1, final_reputation, 
                   s=200, marker='*', color='gold', edgecolor='red', linewidth=2,
                   zorder=10)
        # Add annotation text with Q-value
        q_value = all_q_values[i]
        plt.annotate(f'DQN Selected NA{i}\nQ-value: {q_value:.4f}', 
                    xy=(len(reputation_last)-1, final_reputation),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=8, ha='left')
    
    # Add threshold line at 6600
    plt.axhline(y=6600, color='red', linestyle='--', linewidth=2, label='Reputation Threshold (6600)')
    
    plt.xlabel('Step')
    plt.ylabel('Reputation')
    plt.title('Last Episode Reputation Trajectory - Fixed NA Set Training Results\n([DQN] = Final Multi-Selection Combination)')
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    last_episode_path = str(OUTPUT_ROOT / "src" / "last_episode_reputation_trajectory.png")
    Path(last_episode_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(last_episode_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Last episode reputation trajectory saved as: {last_episode_path}")
    
    # Optional: generate reputation animations for the first/last episode
    if generate_episode_gifs:
        print("\nGenerating NA reputation dynamic GIF animations for first and last episodes...")
        if first_episode_data:
            create_episode_reputation_animation(first_episode_data, n_na, "First Episode")
        if last_episode_data:
            create_episode_reputation_animation(last_episode_data, n_na, "Last Episode")
    else:
        print("\nSkipping episode reputation GIF generation (disabled by configuration).")
    
    # Generate policy analysis plots
    create_policy_analysis_plots(agent, env)
    
    # Store step-level reward history for plotting
    agent.step_reward_history = step_rewards
    agent.step_reward_steps = step_reward_steps
    
    return pd.DataFrame(history), agent.loss_history, agent

def create_episode_reputation_animation(episode_data, n_na, title_prefix):
    """
    Create a per-episode GIF animation of NA reputation dynamics.
    
    Args:
        episode_data: dict containing reputation history and selection history
        n_na: number of NAs
        title_prefix: plot title prefix
    """
    reputation_history = episode_data['reputation_history']
    selected_nas = episode_data['selected_nas']
    episode_num = episode_data['episode_num']
    
    print(f"Creating {title_prefix} Episode {episode_num} GIF animation...")
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Assign colors for each NA
    colors = plt.cm.tab20(np.linspace(0, 1, n_na))
    
    # Compute data range
    steps = range(len(reputation_history))
    all_reputations = [rep for step_rep in reputation_history for rep in step_rep]
    min_reputation = min(all_reputations) - 100
    max_reputation = max(all_reputations) + 200
    
    # Configure axes ranges
    ax.set_xlim(0, len(steps) - 1)
    ax.set_ylim(min_reputation, max_reputation)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Reputation', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add threshold line at 6600
    ax.axhline(y=6600, color='red', linestyle='--', linewidth=2, 
               label='Reputation Threshold (6600)', alpha=0.8)
    
    # Initialize lines and points
    lines = []
    points = []
    selected_markers = []
    
    for i in range(n_na):
        # Base line
        line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.7, label=f'NA {i}')
        lines.append(line)
        
        # Current position point
        point, = ax.plot([], [], 'o', color=colors[i], markersize=8, alpha=0.9)
        points.append(point)
        
        # Selection marker
        marker, = ax.plot([], [], '*', color='red', markersize=20, alpha=1.0)
        selected_markers.append(marker)
    
    # Add a text box for step/selection info
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=14, verticalalignment='top', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Add legend
    if n_na <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    
    def animate(frame):
        """Animation update callback."""
        current_step = frame
        
        # Update title
        selected_na = selected_nas[current_step] if current_step < len(selected_nas) else -1
        ax.set_title(f'{title_prefix} (Episode {episode_num}) - Step {current_step + 1}\n' + 
                    f'Selected NA: {selected_na}' if selected_na != -1 else f'{title_prefix} (Episode {episode_num}) - Initial State', 
                    fontsize=14, weight='bold')
        
        # Update lines/points per NA
        for i in range(n_na):
            # Fetch all points up to current step
            if current_step < len(reputation_history):
                x_data = list(range(current_step + 1))
                y_data = [reputation_history[j][i] for j in range(current_step + 1)]
                
                # Whether current NA is selected at this step
                is_selected = (current_step < len(selected_nas) and selected_nas[current_step] == i)
                
                # Style based on selection
                if is_selected:
                    # Selected NA: thicker line
                    lines[i].set_linewidth(6)
                    lines[i].set_alpha(1.0)
                else:
                    # Not selected: normal thickness
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
        
        # Update text box
        if current_step < len(selected_nas):
            selected_na = selected_nas[current_step]
            step_text.set_text(f'Step: {current_step + 1}\nSelected NA: {selected_na}\n' + 
                              f'NA{selected_na} Current Reputation: {reputation_history[current_step + 1][selected_na]:.1f}' 
                              if current_step + 1 < len(reputation_history) else f'Step: {current_step + 1}\nSelected NA: {selected_na}')
        else:
            step_text.set_text(f'Step: {current_step + 1}\nInitial State')
        
        return lines + points + selected_markers + [step_text]
    
    # Create animation (initial + each step)
    total_frames = len(reputation_history)
    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                 interval=400, blit=True, repeat=True)
    
    # Save animation
    filename = f'{title_prefix.replace(" ", "_").lower()}_episode_{episode_num}_reputation_animation.gif'
    
    try:
        # Ensure output directory exists
        animations_dir = OUTPUT_ROOT / "episode_animations"
        animations_dir.mkdir(parents=True, exist_ok=True)
        full_path = str(animations_dir / filename)
        
        anim.save(full_path, writer='pillow', fps=2.5)
        print(f"Saved {title_prefix} GIF animation: {full_path}")
    except Exception as e:
        print(f"Failed to save {title_prefix} animation: {e}")
    
    # Free memory
    plt.close(fig)

def create_comparison_strategy_gif(env, dqn_selected, random_selected, dqn_training_results, random_training_results, steps=100):
    """
    Create a GIF comparing DQN vs random selection reputation trajectories during training.
    
    Args:
        env: environment object
        dqn_selected: list of NA indices selected by DQN
        random_selected: list of NA indices selected randomly
        dqn_training_results: training results for DQN-selected NAs
        random_training_results: training results for randomly-selected NAs
        steps: simulation steps
    """
    print("Creating selection strategy comparison GIF...")
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Collect reputation histories for all selected NAs
    all_selected_nas = list(set(dqn_selected + random_selected))
    
    # Assign distinct colors for each NA (up to 30 unique colors)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    additional_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    all_colors = np.vstack([colors, additional_colors])
    
    # Color mapping for DQN-selected NAs
    dqn_colors = {}
    for i, na_idx in enumerate(dqn_selected):
        dqn_colors[na_idx] = all_colors[i % len(all_colors)]
    
    # Color mapping for randomly-selected NAs (offset to reduce collisions)
    random_colors = {}
    for i, na_idx in enumerate(random_selected):
        color_idx = (i + len(dqn_selected)) % len(all_colors)
        random_colors[na_idx] = all_colors[color_idx]
    
    # Compute data range
    all_reputations = []
    max_steps = 0
    
    # Collect DQN selection reputation histories
    for na_idx in dqn_selected:
        if na_idx in dqn_training_results:
            rep_history = dqn_training_results[na_idx]['reputation_history']
            all_reputations.extend(rep_history)
            max_steps = max(max_steps, len(rep_history))
    
    # Collect random selection reputation histories
    for na_idx in random_selected:
        if na_idx in random_training_results:
            rep_history = random_training_results[na_idx]['reputation_history']
            all_reputations.extend(rep_history)
            max_steps = max(max_steps, len(rep_history))
    
    if not all_reputations:
        print("Error: insufficient data to create GIF")
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
    
    # Initialize lines/points
    dqn_lines = {}
    random_lines = {}
    dqn_points = {}
    random_points = {}
    
    # DQN-selected NAs
    for i, na_idx in enumerate(dqn_selected):
        if na_idx in dqn_training_results:
            dqn_color = dqn_colors[na_idx]
            
            # If malicious, use special style
            is_malicious = env.fixed_malicious_flags[na_idx]
            if is_malicious:
                mal_type = env.fixed_malicious_types[na_idx]
                # Malicious NA: dashed line + square marker
                line, = ax1.plot([], [], color=dqn_color, linewidth=4, linestyle='--', alpha=0.9, 
                               label=f'[MAL-T{mal_type}] NA {na_idx}' if i < 5 else "")
                point, = ax1.plot([], [], 's', color=dqn_color, markersize=10, alpha=0.9, 
                                markeredgecolor='red', markeredgewidth=2)
            else:
                # Normal NA: solid line + circular marker
                line, = ax1.plot([], [], color=dqn_color, linewidth=3, alpha=0.8, 
                               label=f'[NORM] NA {na_idx}' if i < 5 else "")
                point, = ax1.plot([], [], 'o', color=dqn_color, markersize=8, alpha=0.9)
                
            dqn_lines[na_idx] = line
            dqn_points[na_idx] = point
    
    # Randomly-selected NAs
    for i, na_idx in enumerate(random_selected):
        if na_idx in random_training_results:
            random_color = random_colors[na_idx]
            
            # If malicious, use special style
            is_malicious = env.fixed_malicious_flags[na_idx]
            if is_malicious:
                mal_type = env.fixed_malicious_types[na_idx]
                # Malicious NA: dashed line + square marker
                line, = ax2.plot([], [], color=random_color, linewidth=4, linestyle='--', alpha=0.9, 
                               label=f'[MAL-T{mal_type}] NA {na_idx}' if i < 5 else "")
                point, = ax2.plot([], [], 's', color=random_color, markersize=10, alpha=0.9, 
                                markeredgecolor='red', markeredgewidth=2)
            else:
                # Normal NA: solid line + circular marker
                line, = ax2.plot([], [], color=random_color, linewidth=3, alpha=0.8, 
                               label=f'[NORM] NA {na_idx}' if i < 5 else "")
                point, = ax2.plot([], [], 'o', color=random_color, markersize=8, alpha=0.9)
                
            random_lines[na_idx] = line
            random_points[na_idx] = point
    
    # Add legend
    ax1.legend(loc='upper left', fontsize='small')
    ax2.legend(loc='upper left', fontsize='small')
    
    # Add stats text
    dqn_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    random_text = ax2.text(0.02, 0.02, '', transform=ax2.transAxes, fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    def animate(frame):
        """Animation update callback."""
        current_step = frame
        
        # Update DQN selection
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
        
        # Update random selection
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
        
        # Update stats text
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
                                 interval=200, blit=True, repeat=True)
    
    # Save animation
    filename = 'strategy_comparison_reputation_animation.gif'
    
    try:
        # Ensure output directory exists
        animations_dir = OUTPUT_ROOT / "episode_animations"
        animations_dir.mkdir(parents=True, exist_ok=True)
        full_path = str(animations_dir / filename)
        
        anim.save(full_path, writer='pillow', fps=5)
        print(f"Saved selection strategy comparison GIF: {full_path}")
        
        # Summary
        print("GIF summary:")
        print(f"   DQN-selected NAs: {len(dqn_selected)} - {dqn_selected}")
        print(f"   Random-selected NAs: {len(random_selected)} - {random_selected}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {total_frames/5:.1f}s")
        
    except Exception as e:
        print(f"Failed to save selection strategy comparison GIF: {e}")
    
    # Free memory
    plt.close(fig)

# Training and visualization
if __name__ == "__main__":
    print("DQN-based NA selection optimization - 20 NA training configuration")
    print("="*60)
    
    # Training parameters: fixed NA set for improved stability
    df, loss_history, agent = train_dqn(
        n_na=20,             # NA count: 20
        n_episodes=50,       # Episodes: 50
        steps_per_episode=200,  # Steps per episode: 200
        lr=0.001,            # Learning rate
        update_frequency=4,  # Update every N steps
        use_fixed_nas=True,  # Fixed NA set
        enable_random_fluctuation=False,  # Disable random fluctuation
        freeze_hunger=True,
        hunger_constant=0.0,
        hunger_weight=0.0,
        enable_hunger_update=False
    )
    
    print(f"\nTraining complete. Generated {len(df)} episodes of training data.")
    print(f"Loss history length: {len(loss_history)} training steps")
    
    # Save trained model
    print("\nSaving trained model...")
    try:
        # Ensure output directory exists
        model_dir = OUTPUT_ROOT / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save state_dicts
        policy_state_path = str(model_dir / "policy_net_state_dict.pth")
        target_state_path = str(model_dir / "target_net_state_dict.pth")
        policy_full_path = str(model_dir / "policy_net_complete.pth")
        target_full_path = str(model_dir / "target_net_complete.pth")
        model_info_path = str(model_dir / "model_info.pth")

        torch.save(agent.policy_net.state_dict(), policy_state_path)
        torch.save(agent.target_net.state_dict(), target_state_path)
        
        # Save full modules (fallback)
        torch.save(agent.policy_net, policy_full_path)
        torch.save(agent.target_net, target_full_path)
        
        # Save metadata
        model_info = {
            'n_na': 20,
            'n_features': 4,
            'n_episodes': 50,
            'learning_rate': 0.001,
            'final_epsilon': agent.epsilon,
            'total_training_steps': len(loss_history),
            'final_loss': loss_history[-1] if loss_history else None,
            'final_reward': df['reward'].iloc[-1] if len(df) > 0 else None
        }
        torch.save(model_info, model_info_path)
        
        print("Model saved successfully:")
        print(f"   - Policy state_dict: {policy_state_path}")
        print(f"   - Target state_dict: {target_state_path}")
        print(f"   - Full policy model: {policy_full_path}")
        print(f"   - Full target model: {target_full_path}")
        print(f"   - Model metadata: {model_info_path}")
        
    except Exception as e:
        print(f"Failed to save model: {e}")
    
    # Create a multi-subplot figure to analyze training progress (incl. Q monitoring)
    # Increase font sizes for PDF export
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })
    plt.figure(figsize=(24, 16))
    
    # Subplot 1: step-level reward (consistent with loss sampling)
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
        # Fallback to episode-level plot
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
    
    # Subplot 2: mean reputation
    plt.subplot(3, 3, 2)
    # Raw data (lower alpha)
    plt.plot(df['episode'], df['mean_reputation'], alpha=0.3, label='Raw Mean Reputation', color='lightgreen')
    
    # Add smoothing curve
    window_size = 20
    if len(df) >= window_size:
        smoothed_reputation = df['mean_reputation'].rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(df['episode'], smoothed_reputation, label=f'Smoothed Mean Reputation ({window_size})', color='darkgreen', linewidth=2)
    
    plt.title('Mean Reputation')
    plt.xlabel('Episode')
    plt.ylabel('Mean Reputation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: loss curve
    plt.subplot(3, 3, 3)
    if len(loss_history) > 0:
        loss_steps = agent.loss_steps if hasattr(agent, 'loss_steps') and len(agent.loss_steps) == len(loss_history) else list(range(len(loss_history)))
        # Compute moving average for smoothing
        window_size = min(100, len(loss_history) // 10)
        if window_size > 1:
            smoothed_loss = []
            for i in range(len(loss_history)):
                start_idx = max(0, i - window_size + 1)
                smoothed_loss.append(np.mean(loss_history[start_idx:i+1]))
            plt.plot(loss_steps, smoothed_loss, label=f'Smoothed Loss (window={window_size})', alpha=0.8)
        
        # Raw loss curve (lower alpha)
        plt.plot(loss_steps, loss_history, alpha=0.3, label='Raw Loss', color='gray')
        plt.title('Loss Function Curve')
        plt.xlabel('Training Steps')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Loss Function Curve')
    
    # Subplot 4: reward standard deviation (20-episode rolling window)
    plt.subplot(3, 3, 4)
    # Compute reward std for each 20-episode window
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
        # Add smoothed line
        if len(reward_std_data) >= 20:
            reward_std_series = pd.Series(reward_std_data)
            smoothed_reward_std = reward_std_series.rolling(window=20, center=True, min_periods=1).mean()
            plt.plot(episode_data, smoothed_reward_std, label='Smoothed Reward Std', color='darkgreen', linewidth=2)
    
    plt.title('Reward Standard Deviation (20-episode window)')
    plt.xlabel('Episode')
    plt.ylabel('Reward Std')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 5: average step reward
    plt.subplot(3, 3, 5)
    plt.plot(df['episode'], df['avg_step_reward'], label='Avg Step Reward', color='orange', alpha=0.2)
    # Add smoothing line and confidence band
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
    
    # Subplot 6: step reward volatility (std dev)
    plt.subplot(3, 3, 6)
    plt.plot(df['episode'], df['reward_std'], label='Step Reward Std Dev', color='red', alpha=0.4)
    # Add smoothed line
    if len(df) >= 20:
        smoothed_std = df['reward_std'].rolling(window=20, center=True, min_periods=1).mean()
        plt.plot(df['episode'], smoothed_std, label='Smoothed Std Dev', color='darkred', linewidth=2)
    plt.title('Step Reward Volatility (Std Dev)')
    plt.xlabel('Episode')
    plt.ylabel('Reward Standard Deviation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 7: Q-value range monitoring
    plt.subplot(3, 3, 7)
    plt.plot(df['episode'], df['q_max'], label='Q Max', color='red', alpha=0.7)
    plt.plot(df['episode'], df['q_min'], label='Q Min', color='blue', alpha=0.7)
    plt.fill_between(df['episode'], df['q_min'], df['q_max'], alpha=0.2, color='purple')
    plt.title('Q-Value Range (Min-Max)')
    plt.xlabel('Episode')
    plt.ylabel('Q-Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for a wide Q-value range
    
    # Subplot 8: Q-value mean and std dev
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
    plt.yscale('log')  # Log scale for a wide Q-value range
    
    # Subplot 9: learning rate and epsilon schedule
    plt.subplot(3, 3, 9)
    if hasattr(agent, 'lr_history') and agent.lr_history:
        # Build proper episode indices for logged schedule points
        lr_episodes = []
        total_episodes = len(df)
        for i in range(len(agent.lr_history)):
            if i < len(agent.lr_history) - 1:
                # Earlier points are multiples of 10
                lr_episodes.append(i * 10)
            else:
                # Final point: if total episodes is not a multiple of 10, use the last episode index
                final_episode = total_episodes - 1
                if final_episode % 10 == 0:
                    lr_episodes.append(i * 10)
                else:
                    lr_episodes.append(final_episode)
        
        # Use twin axes to plot learning rate and epsilon
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        # Learning rate (left axis, log scale)
        line1 = ax1.plot(lr_episodes, agent.lr_history, label='Learning Rate', color='orange', linewidth=2)
        ax1.set_ylabel('Learning Rate', color='orange')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='orange')
        
        # Epsilon (right axis, linear scale)
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
    
    # Save training result plots
    main_png = str(OUTPUT_ROOT / "src" / "training_results.png")
    Path(main_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(main_png, dpi=300, bbox_inches='tight')
    print(f"Training result plot saved: {main_png}")

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
            # Fallback to episode-level curve
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
    
    # Loss summary
    if len(loss_history) > 0:
        print(f"\nLoss statistics:")
        print(f"Total training steps: {len(loss_history)}")
        print(f"Mean loss: {np.mean(loss_history):.6f}")
        print(f"Final loss: {loss_history[-1]:.6f}")
        print(f"Min loss: {np.min(loss_history):.6f}")
        print(f"Max loss: {np.max(loss_history):.6f}")
        
        # Convergence check
        if len(loss_history) > 1000:
            recent_loss = np.mean(loss_history[-1000:])
            early_loss = np.mean(loss_history[:1000])
            improvement = (early_loss - recent_loss) / early_loss * 100
            print(f"Loss improvement: {improvement:.2f}%")
    else:
        print("\nWarning: no loss history recorded; training may be insufficient or batch_size too large")
    
    # Detailed reward volatility analysis
    print(f"\nEpisode reward volatility analysis:")
    print(f"Total episodes: {len(df)}")
    print(f"Mean episode reward: {df['reward'].mean():.2f}")
    print(f"Episode reward std dev: {df['reward'].std():.2f}")
    print(f"Episode reward range: {df['reward'].min():.2f} ~ {df['reward'].max():.2f}")
    print(f"Reward coefficient of variation (CV): {df['reward'].std() / abs(df['reward'].mean()):.2f}")
    
    print(f"\nStep reward statistics:")
    print(f"Mean step reward: {df['avg_step_reward'].mean():.2f}")
    print(f"Step reward volatility (mean std dev): {df['reward_std'].mean():.2f}")
    print(f"Overall success rate: {df['success_rate'].mean():.3f}")
    
    print(f"\nVolatility breakdown:")
    high_volatility_episodes = df[df['reward_std'] > df['reward_std'].quantile(0.8)]
    print(f"High-volatility episodes: {len(high_volatility_episodes)} ({len(high_volatility_episodes)/len(df)*100:.1f}%)")
    
    if len(high_volatility_episodes) > 0:
        print(f"High-volatility mean reward: {high_volatility_episodes['reward'].mean():.2f}")
        print(f"High-volatility mean success rate: {high_volatility_episodes['success_rate'].mean():.3f}")
    
    # Compare early/middle/late phases
    early_episodes = df.iloc[:len(df)//3]
    middle_episodes = df.iloc[len(df)//3:2*len(df)//3]
    late_episodes = df.iloc[2*len(df)//3:]
    
    print(f"\nTraining phase analysis:")
    print(f"Early (Episodes 0-{len(early_episodes)}): mean reward={early_episodes['reward'].mean():.2f}, volatility={early_episodes['reward'].std():.2f}")
    print(f"Mid (Episodes {len(early_episodes)}-{len(early_episodes)+len(middle_episodes)}): mean reward={middle_episodes['reward'].mean():.2f}, volatility={middle_episodes['reward'].std():.2f}")
    print(f"Late (Episodes {len(early_episodes)+len(middle_episodes)}-{len(df)}): mean reward={late_episodes['reward'].mean():.2f}, volatility={late_episodes['reward'].std():.2f}")
    
    improvement = (late_episodes['reward'].mean() - early_episodes['reward'].mean()) / abs(early_episodes['reward'].mean()) * 100
    volatility_change = (late_episodes['reward'].std() - early_episodes['reward'].std()) / early_episodes['reward'].std() * 100
    
    print(f"\nTraining outcomes:")
    print(f"Reward improvement: {improvement:+.1f}%")
    print(f"Volatility change: {volatility_change:+.1f}% ({'improved' if volatility_change < 0 else 'worsened'})")
    
    # Record end time and total duration
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time
    
    print(f"\n" + "="*80)
    print(f"Runtime summary")
    print(f"="*80)
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {format_duration(total_duration)}")
    print(f"Training efficiency:")
    print(f"   - Total episodes: {len(df)}")
    print(f"   - Avg time per episode: {total_duration/len(df):.2f}s")
    if len(loss_history) > 0:
        print(f"   - Total training steps: {len(loss_history)}")
        print(f"   - Avg time per step: {total_duration/len(loss_history)*1000:.2f}ms")
    print(f"="*80)
    print(f"Done.")
    
    # Export NA selection parameters from the trained agent
    print(f"\n" + "="*80)
    print("Export NA selection parameters")
    print("="*80)
    
    try:
        # Create an environment instance to generate a state snapshot
        export_env = Environment(
            n_na=20,
            use_fixed_nas=True,
            enable_random_fluctuation=False,
            freeze_hunger=True,
            hunger_constant=0.0,
            hunger_weight=0.0,
            enable_hunger_update=False
        )
        export_env.reset()
        
        # Get current state and initial reputations
        current_state = export_env.get_state()
        initial_reputations = export_env.episode_start_reputation
        
        # Export parameters
        export_file = agent.export_na_selection_parameters(
            state=current_state, 
            initial_reputations=initial_reputations,
            env=export_env
        )
        
        print(f"Exported NA selection parameters successfully.")
        
    except Exception as e:
        print(f"Failed to export NA selection parameters: {e}")
    
    # Demonstrate network flexibility
    print(f"\n" + "="*80)
    print("Network flexibility demo - handling different NA counts")
    print("="*80)
    
    def test_network_flexibility(trained_agent):
        """Demonstrate the trained network can handle different NA counts."""
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
        print("Benefit: the trained model can be applied to arbitrary NA counts.")
        print("Features: reputation, success_rate, activity, signature_delay, hunger")
    
    test_network_flexibility(agent)
