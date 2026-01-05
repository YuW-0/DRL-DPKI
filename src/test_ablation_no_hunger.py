import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只用第3张显卡（编号2）
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
🌟 Double DQN (DDQN) for Network Assistant Selection
===================================================

算法特性:
- 🚀 Double DQN: 解决Q值过估计问题
- 🔄 经验回放: 提高样本效率
- 🎯 目标网络: 增强训练稳定性
- 🧠 NA量无关架构: 支持任意数量的NA

核心改进:
1. 用policy网络选择动作
2. 用target网络评估Q值
3. 避免最大化偏差导致的过估计

适用场景: 固定NA集合的选择优化问题
"""

#  记录程序启动时间
start_time = time.time()
start_datetime = datetime.now()
print(f"程序启动时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

def set_all_seeds(seed=42):
    """
    设置所有随机种子以确保结果的可重现性
    """
    print(f"🔧 设置随机种子: {seed}")
    
    # Python内置random模块
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    
    # 如果使用CUDA，设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        
    # 确保CUDA操作的确定性（可能影响性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("随机种子设置完成，训练将更加稳定")

# 种子选择配置
# 常用的优秀种子推荐：
# - 42: 经典选择，通常效果稳定
# - 123: 数学简洁，经常有好效果  
# - 2024: 年份种子，容易记忆
# - 88: 在某些任务中表现优异
# - 666: 在强化学习中经常有好结果
# - 1337: 程序员最爱，在深度学习中效果不错

RANDOM_SEED = 42  # 🔧 修改这里来尝试不同的种子

print(f"当前使用种子: {RANDOM_SEED}")
print("种子建议: 42(经典), 123(简洁), 2024(年份), 88(优秀), 666(强化学习), 1337(深度学习)")

# 设置全局随机种子
set_all_seeds(RANDOM_SEED)

# GPU支持检查
print("GPU支持检查:")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
else:
    print("将使用CPU进行训练")

# 配置字体支持 - 仅使用英文字体避免中文字符缺失警告
print("🔧 配置字体...")

font_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "font"))
if os.path.isdir(font_dir):
    font_files = [f for f in os.listdir(font_dir) if f.lower().endswith(".ttf")]
    if font_files:
        print(f"🔠 发现自定义字体文件: {len(font_files)} 个，开始加载...")
    loaded_custom_font = False
    for font_file in font_files:
        font_path = os.path.join(font_dir, font_file)
        try:
            fm.fontManager.addfont(font_path)
            loaded_custom_font = True
        except Exception as exc:
            print(f"⚠️ 加载字体失败 {font_file}: {exc}")
else:
    print(f"⚠️ 未找到字体目录: {font_dir}")

# 设置字体优先级 - 只使用英文字体
if 'loaded_custom_font' in locals() and loaded_custom_font:
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
else:
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['font.size'] = 10

print(f"使用字体: {plt.rcParams['font.family'][:2]}")
print("字体配置完成")

def format_duration(seconds):
    """
    格式化时间显示
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours}小时{minutes}分钟{secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟{secs}秒"
    else:
        return f"{secs}.{millisecs:03d}秒"

# 与NA数量无关的DQN网络架构 - 基于纯特征学习
# 新架构优势：
# 1. 可扩展性：支持任意数量的NA，无需重新训练
# 2. 泛化能力：学习的是NA特征的通用模式，而非位置信息
# 3. 简洁性：网络参数数量固定，不随NA数量增长
# 4. 可解释性：每个NA的Q值完全基于其自身特征
class QNetwork(nn.Module):
    def __init__(self, n_features, n_na):  # 移除默认参数
        super().__init__()
        # NA特征编码器 - 将每个NA的特征映射到固定维度
        self.na_encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LeakyReLU(0.01),  # 改用LeakyReLU防止dying ReLU
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),  # 改用LeakyReLU
            nn.Dropout(0.2),
            nn.Linear(32, 16)  # 每个NA编码为16维
        )
        
        # Q值预测器 - 独立评估每个NA的价值
        self.q_predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.01),  # 改用LeakyReLU
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.01),  # 改用LeakyReLU
            nn.Linear(16, 1)  # 输出单个Q值
        )
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重，确保输出有差异性"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier正态分布初始化
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    # 偏置项使用小的随机值
                    nn.init.uniform_(module.bias, -0.1, 0.1)

    def forward(self, x):
        batch_size = x.size(0)
        n_na = x.size(1)
        # 将所有NA特征展平处理
        x_flat = x.view(batch_size * n_na, -1)  # [batch*n_na, n_features]
        # 每个NA独立编码特征
        na_embeddings = self.na_encoder(x_flat)  # [batch*n_na, 16]
        # 每个NA独立预测Q值
        q_values_flat = self.q_predictor(na_embeddings)  # [batch*n_na, 1]
        # 重新整理为批次格式
        q_values = q_values_flat.view(batch_size, n_na)  # [batch, n_na]
        return q_values

class DQNAgent:
    def __init__(self, n_features, n_na, lr, gamma,  # 移除所有默认参数
                 epsilon_start, epsilon_end,  # 移除所有默认参数
                 decay_steps, memory_size, batch_size, target_update,  # 移除所有默认参数
                 min_memory_size, update_frequency, total_episodes):  # 移除所有默认参数
        # 设置设备优先使用GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"🚀 Double DQN Agent 使用设备: {self.device} ({torch.cuda.get_device_name(0)})")
            print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = torch.device("cpu")
            print(f"🚀 Double DQN Agent 使用设备: {self.device}")
        print("🔧 使用与NA数量无关的网络架构 - 支持任意数量的NA")
        print("⚡ 算法升级: Double DQN (DDQN) - 解决Q值过估计问题")
        print(f"🔧 优化配置: Replay Buffer={memory_size}, Batch Size={batch_size}, Min Memory={min_memory_size}")
        
        self.n_na = n_na
        self.n_features = n_features
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.decay_steps   = decay_steps
        self.total_episodes = total_episodes  # 总训练episode数
        self.step_count    = 0
        self.epsilon       = epsilon_start
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.min_memory_size = min_memory_size  # 开始训练前的最小经验数量
        self.update_frequency = update_frequency  # 每隔多少步进行一次训练
        self.step_counter = 0  # 步数计数器
        
        # 创建网络并移动到设备 - 新架构不依赖NA数量
        self.policy_net = QNetwork(n_features, n_na).to(self.device)
        self.target_net = QNetwork(n_features, n_na).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # 使用 AdamW 优化器并启用 AMSGrad
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), 
            lr=lr,
            weight_decay=1e-4,  # L2正则化，防止过拟合
            amsgrad=True       # 启用AMSGrad，提高收敛稳定性
        )
        print(f"优化器: AdamW (weight_decay=1e-4, amsgrad=True)")
        print(f"   优势: 更好的泛化能力 + 稳定收敛")
        print(f"   特性: 解耦权重衰减 + AMSGrad防止震荡")
        
        # 添加Cosine Annealing with Warm-up学习率调度器
        self.initial_lr = lr
        self.final_lr = lr * 0.1  # 最终学习率为初始的10%
        self.total_episodes = total_episodes
        print(f"🔧 学习率调度: Cosine Annealing with Warm-up {self.final_lr:.6f} → {lr:.6f} → {self.final_lr:.6f}")
        print(f"   前5%: Warm-up线性增长到最大学习率")
        print(f"   后95%: Cosine余弦衰减到最小学习率")
        print(f"   优势: 避免初期过快学习+后期平滑收敛")
        print(f"   公式: lr = min_lr + (max_lr - min_lr) * (1 + cos(π * progress)) / 2")
        
        self.target_update = target_update
        self.learn_step = 0
        
        # 添加损失记录
        self.loss_history = []
        self.loss_steps = []
        
        # 添加学习率记录
        self.lr_history = []
        
        # 添加epsilon记录
        self.epsilon_history = []
        
        # 添加策略分析记录
        self.policy_entropy_history = []  # 记录策略熵
        self.action_distribution_history = []  # 记录动作分布
        self.na_selection_frequency = np.zeros(n_na)  # 记录每个NA被选中的频率

    def select_action(self, state, initial_reputations=None, available_mask=None, record_policy=False):
        """
        选择单个NA进行动作，但网络能区分高低信誉NA
        
        Args:
            state: shape (n_na, 5) - 当前状态 (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - 初始信誉值，用于分组
            available_mask: shape (n_na,) - 可选择的NA掩码，None表示所有NA都可选
            record_policy: bool - 是否记录策略分析数据
        
        Returns:
            int: 选中的单个NA索引
        """
        # 如果没有提供初始信誉，使用当前信誉作为替代
        if initial_reputations is None:
            reputations = state[:, 0]  # 使用当前信誉
        else:
            reputations = initial_reputations
        
        # 设置可用NA掩码
        if available_mask is None:
            available_mask = np.ones(self.n_na, dtype=bool)
        
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0:
            return 0  # 如果没有可选NA，默认选择0

        # 计算Q值和策略分布（即使是epsilon-greedy也需要计算Q值用于分析）
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, n_na, 5]
        
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0).cpu()  # [n_na]
        
        # 只考虑可用NA的Q值
        masked_q_values = q_values.clone()
        masked_q_values[~torch.from_numpy(available_mask)] = float('-inf')  # 不可用NA设为负无穷
        
        # 计算softmax策略分布（用于熵计算）
        valid_q_values = q_values[available_mask]
        if len(valid_q_values) > 1:
            policy_probs = F.softmax(valid_q_values, dim=0).numpy()
            # 计算策略熵
            policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-8))
        else:
            policy_entropy = 0.0
            policy_probs = np.array([1.0])
        
        # 记录策略分析数据
        if record_policy:
            self.policy_entropy_history.append(policy_entropy)
            # 记录完整的动作分布（包括不可用的NA）
            full_policy_dist = np.zeros(self.n_na)
            full_policy_dist[available_indices] = policy_probs
            self.action_distribution_history.append(full_policy_dist.copy())
        
        # 选择动作
        if np.random.rand() < self.epsilon:
            # 随机选择策略：从可用NA中随机选一个
            action = np.random.choice(available_indices)
        else:
            # 基于Q值的贪心选择策略
            action = masked_q_values.argmax().item()
        
        # 更新NA选择频率
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
        Cosine Annealing with Warm-up 学习率调度器
        前5%: Warm-up阶段，学习率线性增长到最大值
        后95%: Cosine Annealing，平滑衰减到最小值
        """
        if current_episode >= self.total_episodes:
            progress = 1.0
        else:
            progress = current_episode / self.total_episodes
        
        # 设置warm-up比例
        warmup_ratio = 0.05  # 前5%进行warm-up
        
        if progress <= warmup_ratio:
            # Warm-up阶段：线性增长到最大学习率
            warmup_progress = progress / warmup_ratio
            current_lr = self.final_lr + (self.initial_lr - self.final_lr) * warmup_progress
        else:
            # Cosine Annealing阶段
            cosine_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
            import math
            cosine_factor = (1 + math.cos(math.pi * cosine_progress)) / 2
            current_lr = self.final_lr + (self.initial_lr - self.final_lr) * cosine_factor
        
        # 确保学习率不低于最小值
        current_lr = max(current_lr, self.final_lr)
        
        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        return current_lr
    
    def make_final_selection(self, state, initial_reputations):
        """
        使用训练好的DQN网络进行最终选择，与训练过程完全一致
        
        Args:
            state: shape (n_na, 5) - 当前状态 (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - 初始信誉值，用于分组
        
        Returns:
            dict: 包含选择结果、Q值和分析信息
        """
        # 使用多次选择方法获得组合
        selected_nas = self.make_multiple_selections(state, initial_reputations, 5, 'balanced')
        
        # 计算所有NA的Q值用于分析
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        # 分组信息
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        low_rep_indices = np.where(low_rep_mask)[0]
        high_rep_indices = np.where(high_rep_mask)[0]
        
        # 随机选择作为对比
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
        导出最后一次模型选择NA时所有NA的参数到文件
        
        Args:
            state: shape (n_na, 4) - 当前状态 (reputation, success_rate, signature_delay, hunger)
            initial_reputations: shape (n_na,) - 初始信誉值
            env: Environment对象 - 用于获取加权成功率和加权延迟等级
            output_file: str - 输出文件路径，如果为None则使用默认路径
        
        Returns:
            str: 导出文件的路径
        """
        if output_file is None:
            output_file = '/mnt/data/wy2024/src/na_selection_parameters.csv'
        
        # 计算所有NA的Q值
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            all_q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()
        
        # 进行最终选择以获得选择结果
        selection_result = self.make_final_selection(state, initial_reputations)
        selected_nas = selection_result['dqn_selected']
        
        # 获取加权成功率和加权延迟等级（如果提供了环境对象）
        weighted_success_rates = []
        weighted_delay_levels = []
        if env is not None:
            for i in range(self.n_na):
                window_summary = env.get_na_window_summary(i)
                weighted_success_rates.append(window_summary['weighted_success_rate'])
                weighted_delay_levels.append(window_summary['weighted_delay_grade'])
        else:
            # 如果没有环境对象，使用当前状态的值
            weighted_success_rates = state[:, 1].tolist()  # success_rate
            weighted_delay_levels = [0] * self.n_na  # 默认值
        
        # 准备导出数据
        export_data = []
        for i in range(self.n_na):
            na_data = {
                'NA_Index': i,
                'Q_Value': all_q_values[i],
                'Current_Reputation': state[i, 0],  # reputation
                'Success_Rate': state[i, 1],        # success_rate
                'Signature_Delay': env.signature_delay[i] if env is not None else state[i, 2],     # 使用原始延迟等级值
                'Hunger_Level': state[i, 3],        # hunger
                'Weighted_Success_Rate': weighted_success_rates[i],
                'Weighted_Delay_Grade': weighted_delay_levels[i],
                'Initial_Reputation': initial_reputations[i],
                'Reputation_Group': 'High' if initial_reputations[i] >= 6600 else 'Low',
                'Selected_by_DQN': i in selected_nas,
                'Selection_Rank': selected_nas.index(i) + 1 if i in selected_nas else None
            }
            export_data.append(na_data)
        
        # 按Q值降序排序
        export_data.sort(key=lambda x: x['Q_Value'], reverse=True)
        
        # 添加Q值排名
        for rank, na_data in enumerate(export_data, 1):
            na_data['Q_Value_Rank'] = rank
        
        # 转换为DataFrame并保存
        df = pd.DataFrame(export_data)
        df.to_csv(output_file, index=False, float_format='%.6f')
        
        print(f"\nNA选择参数已导出到: {output_file}")
        print(f"   - 总NA数量: {len(df)}")
        print(f"   - DQN选择的NA: {len([x for x in export_data if x['Selected_by_DQN']])}")
        print(f"   - 高信誉NA数量: {len([x for x in export_data if x['Reputation_Group'] == 'High'])}")
        print(f"   - 低信誉NA数量: {len([x for x in export_data if x['Reputation_Group'] == 'Low'])}")
        print(f"   - Q值范围: {df['Q_Value'].min():.6f} ~ {df['Q_Value'].max():.6f}")
        
        return output_file

    def store(self, s, a, r, s_, done):
        """
        存储经验，现在支持单个NA动作
        
        Args:
            s: shape (n_na, 5) - 当前状态 (reputation, success_rate, activity, signature_delay, hunger)
            a: int - 选中的NA索引
            r: float - 奖励
            s_: shape (n_na, 5) - 下一状态 (reputation, success_rate, activity, signature_delay, hunger)
            done: bool - 是否结束
        """
        self.memory.append((s.copy(), int(a), r, s_.copy(), done))

    def update(self, global_step=None):
        # 检查是否有足够的经验进行训练
        if len(self.memory) < self.min_memory_size:
            return None  # 数据不够，跳过训练
            
        if len(self.memory) < self.batch_size:
            return None  # 样本数不足一个batch
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转为tensor并移动到设备，保持原有维度
        states = torch.FloatTensor(np.array(states)).to(self.device)  # [batch_size, n_na, n_features]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # [batch_size, n_na, n_features]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # [batch_size, 1]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)  # [batch_size, 1]
        
        # 计算当前Q值
        current_q_values = self.policy_net(states)  # [batch_size, n_na]
        current_q = current_q_values.gather(1, actions)  # [batch_size, 1]
        
        # 计算目标Q值 - Double DQN算法
        with torch.no_grad():
            # DDQN核心改进：用policy网络选择动作，用target网络评估Q值
            # 1. 用当前策略网络选择下一个状态的最佳动作
            next_actions = self.policy_net(next_states).max(1, keepdim=True)[1]  # [batch_size, 1]
            
            # 2. 用目标网络评估该动作的Q值（避免过估计）
            next_q_values = self.target_net(next_states)  # [batch_size, n_na]
            next_q = next_q_values.gather(1, next_actions)  # [batch_size, 1]
            
            target = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q, target)
        
        # 记录损失值及对应训练步数
        self.loss_history.append(loss.item())
        if global_step is not None:
            self.loss_steps.append(global_step)
        else:
            self.loss_steps.append(self.learn_step)
        
        self.optimizer.zero_grad()
        loss.backward()    
        self.optimizer.step()

        # 每30次策略网络更新后才更新目标网络（平衡稳定性与学习速度）
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()  # 返回损失值

    def maybe_update(self, global_step=None):
        """
        条件更新：更频繁的训练，增强梯度信号
        
        Returns:
            list: 本次批次训练的损失值列表，如果没有训练则返回None
        """
        self.step_counter += 1
        
        # 检查是否满足更新条件
        if (self.step_counter % self.update_frequency == 0 and 
            len(self.memory) >= self.min_memory_size):
            
            # 进行更频繁的批次训练：增加训练轮数以增强梯度信号
            batch_losses = []
            training_batches = max(2, self.update_frequency)  # 更频繁的训练，每4步训练4轮
            
            for _ in range(training_batches):
                loss = self.update(global_step=global_step)
                if loss is not None:
                    batch_losses.append(loss)
            
            return batch_losses if batch_losses else None
        
        return None

    def debug_q_values(self, state, initial_reputations):
        """调试Q值学习情况"""
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(self.device)).squeeze().cpu()
        
        print(f"Q值统计: min={q_values.min():.4f}, max={q_values.max():.4f}, "
              f"std={q_values.std():.4f}")
        
        # 检查Q值是否有意义的差异
        if q_values.std() < 0.1:
            print("⚠️  警告：Q值差异过小，网络可能没有学到有效信息")
        
        # 分析高低信誉组的Q值差异
        low_rep_mask = initial_reputations < 6600
        high_rep_mask = initial_reputations >= 6600
        
        if np.any(low_rep_mask) and np.any(high_rep_mask):
            low_q_mean = q_values[low_rep_mask].mean()
            high_q_mean = q_values[high_rep_mask].mean()
            print(f"低信誉组Q值平均: {low_q_mean:.4f}, 高信誉组Q值平均: {high_q_mean:.4f}")
            print(f"Q值学习方向: {'正确' if high_q_mean > low_q_mean else '错误'}")
        
        return q_values

    def make_multiple_selections(self, state, initial_reputations, num_selections=5, strategy='balanced'):
        """
        通过多次单选来组成最终的NA组合
        
        Args:
            state: shape (n_na, 5) - 当前状态 (reputation, success_rate, activity, signature_delay, hunger)
            initial_reputations: shape (n_na,) - 初始信誉值
            num_selections: int - 需要选择的NA数量
            strategy: str - 选择策略 ('balanced', 'top_q', 'by_group')
        
        Returns:
            list: 选中的NA索引列表
        """
        original_epsilon = self.epsilon
        self.epsilon = 0  # 使用纯贪心策略
        
        selected_nas = []
        available_mask = np.ones(self.n_na, dtype=bool)
        
        if strategy == 'balanced':
            # 平衡策略：按比例从高低信誉组选择
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            # 目标：选择2个低信誉 + 3个高信誉（或按比例调整）
            target_low = min(2, len(low_rep_indices), num_selections)
            target_high = min(num_selections - target_low, len(high_rep_indices))
            
            # 从低信誉组选择
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            for _ in range(target_low):
                if np.any(low_mask):
                    selected_na = self.select_action(state, initial_reputations, low_mask)
                    selected_nas.append(selected_na)
                    low_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            # 从高信誉组选择
            high_mask = np.zeros(self.n_na, dtype=bool)
            high_mask[high_rep_indices] = True
            high_mask &= available_mask  # 排除已选择的
            
            for _ in range(target_high):
                if np.any(high_mask):
                    selected_na = self.select_action(state, initial_reputations, high_mask)
                    selected_nas.append(selected_na)
                    high_mask[selected_na] = False
                    available_mask[selected_na] = False
            
            # 如果还需要更多，从剩余中选择
            remaining_needed = num_selections - len(selected_nas)
            for _ in range(remaining_needed):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'top_q':
            # 简单策略：选择Q值最高的几个
            for _ in range(min(num_selections, self.n_na)):
                if np.any(available_mask):
                    selected_na = self.select_action(state, initial_reputations, available_mask)
                    selected_nas.append(selected_na)
                    available_mask[selected_na] = False
                    
        elif strategy == 'by_group':
            # 按组策略：先选所有低信誉，再选高信誉
            # 分组
            low_rep_indices = np.where(initial_reputations < 6600)[0]
            high_rep_indices = np.where(initial_reputations >= 6600)[0]
            
            # 先选低信誉组
            low_mask = np.zeros(self.n_na, dtype=bool)
            low_mask[low_rep_indices] = True
            
            while len(selected_nas) < num_selections and np.any(low_mask):
                selected_na = self.select_action(state, initial_reputations, low_mask)
                selected_nas.append(selected_na)
                low_mask[selected_na] = False
                available_mask[selected_na] = False
            
            # 再选高信誉组
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
        # 🎲 随机波动控制开关
        self.enable_random_fluctuation = enable_random_fluctuation
        if enable_random_fluctuation:
            print("🎲 随机波动模式已启用: 成功率、饥饿度、延迟等会有动态变化")
        else:
            print("🔒 固定模式: 所有参数保持基础值，无随机波动")

        # 饥饿度增长控制参数，保持与baseline环境一致
        self.hunger_growth_scale = 10.0
        self.hunger_growth_log_base = 11.0
        self.freeze_hunger = freeze_hunger
        self.hunger_constant = float(np.clip(hunger_constant, 0.0, 1.0))
        self.hunger_weight = float(hunger_weight)
        self.enable_hunger_update = enable_hunger_update
        
        # 🚀 新增：为每个NA设置独立的滑动窗口队列系统
        self.window_pack_interval = 5  # 每5步打包一次
        self.window_queue_size = window_size  # 每个NA的队列最大容量
        self.na_window_queues = {}  # 每个NA的滑动窗口队列
        self.current_step_count = 0  # 当前步数计数
        self.na_current_pack = {}  # 每个NA当前正在积累的事务包
        
        for na_id in range(n_na):
            self.na_window_queues[na_id] = deque(maxlen=window_size)  # 队列存储打包的事务数据
            self.na_current_pack[na_id] = {
                'transactions': [],
                'success_count': 0,
                'total_count': 0,
                'reputation_changes': [],
                'start_step': 0,
                'end_step': 0
            }
        print(f"🔧 滑动窗口队列配置: 每个NA独立维护队列，每{self.window_pack_interval}步打包一次，队列容量{window_size}")
        
        if use_fixed_nas:
            # 🔧 使用固定的NA集合进行训练
            print(f"🎯 使用更复杂的NA集合进行训练 (共{n_na}个NA)")
            print("📋 设计原则：每个信誉组都包含多种类型的NA，形成对比学习环境")
            
            # 注意：不重新设置随机种子，使用全局种子保持一致性
            # 固定生成一批有区分度的NA（使用全局随机种子状态）
            
            # 🚀 重新设计NA分布策略：创建4个明确的信誉区间
            quarter = n_na // 4
            remaining = n_na % 4
            
            # 分配各区间的NA数量
            group_sizes = [quarter] * 4
            for i in range(remaining):
                group_sizes[i] += 1
            
            print(f"NA分组策略: 极低信誉组{group_sizes[0]}个, 低信誉组{group_sizes[1]}个, 高信誉组{group_sizes[2]}个, 极高信誉组{group_sizes[3]}个")
            
            # 生成各组的信誉值
            very_low_rep = np.random.uniform(3300, 4500, group_sizes[0])    # 极低信誉组: 3300-4500
            low_rep = np.random.uniform(4500, 6600, group_sizes[1])         # 低信誉组: 4500-6600  
            high_rep = np.random.uniform(6600, 8200, group_sizes[2])        # 高信誉组: 6600-8200
            very_high_rep = np.random.uniform(8200, 9800, group_sizes[3])   # 极高信誉组: 8200-9800
            
            self.fixed_initial_reputation = np.concatenate([very_low_rep, low_rep, high_rep, very_high_rep])
            
            # 🎨 为每个组设计多样化的成功率分布，创建学习挑战（包含复杂恶意NA）
            success_rates_list = []
            malicious_flags_list = []  # 记录哪些NA是恶意的
            malicious_types_list = []  # 记录恶意NA的类型
            
            # 🚨 定义恶意NA的复杂行为类型：
            # Type 1: "高成功率+高延迟" - 表面优秀但故意拖延
            # Type 2: "高成功率+隐蔽攻击" - 通过其他方式进行攻击
            # Type 3: "中成功率+极不稳定" - 不可预测的行为
            # Type 4: "传统低成功率" - 明显的低质量
            
            # 极低信誉组 (3300-4500)：包含各种类型 + 复杂恶意NA
            # 30%真正低能力, 25%中等能力, 15%隐藏高手, 30%恶意NA（多种类型）
            vl_count_low = int(group_sizes[0] * 0.3)
            vl_count_med = int(group_sizes[0] * 0.25) 
            vl_count_high = int(group_sizes[0] * 0.15)
            vl_count_malicious = group_sizes[0] - vl_count_low - vl_count_med - vl_count_high
            
            vl_sr_low = np.random.uniform(0.25, 0.45, vl_count_low)     # 真正低能力
            vl_sr_med = np.random.uniform(0.55, 0.70, vl_count_med)     # 中等能力 
            vl_sr_high = np.random.uniform(0.75, 0.90, vl_count_high)   # 隐藏高手
            
            # 🚨 极低信誉组的恶意NA策略分配（简化为两种类型）
            vl_mal_type1 = max(1, vl_count_malicious // 2)      # 高成功率+高延迟
            vl_mal_type4 = vl_count_malicious - vl_mal_type1    # 传统低成功率
            
            vl_sr_mal_type1 = np.random.uniform(0.65, 0.80, vl_mal_type1)  # 高成功率伪装
            vl_sr_mal_type4 = np.random.uniform(0.05, 0.30, vl_mal_type4)  # 低成功率
            
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
            
            # 随机打乱顺序（保持flag和type对应关系）
            shuffle_indices = np.random.permutation(len(vl_success_rates))
            vl_success_rates = vl_success_rates[shuffle_indices]
            vl_malicious_flags = vl_malicious_flags[shuffle_indices]
            vl_all_malicious_types = vl_all_malicious_types[shuffle_indices]
            
            success_rates_list.append(vl_success_rates)
            malicious_flags_list.append(vl_malicious_flags)
            malicious_types_list.append(vl_all_malicious_types)
            
            # 低信誉组 (4500-6600)：更多变化 + 复杂恶意NA
            # 25%中低能力, 30%中等能力, 20%潜力股, 25%恶意NA
            l_count_low = int(group_sizes[1] * 0.25)
            l_count_med = int(group_sizes[1] * 0.30)
            l_count_high = int(group_sizes[1] * 0.20)
            l_count_malicious = group_sizes[1] - l_count_low - l_count_med - l_count_high
            
            l_sr_low = np.random.uniform(0.35, 0.55, l_count_low)       # 中低能力
            l_sr_med = np.random.uniform(0.60, 0.75, l_count_med)       # 中等能力
            l_sr_high = np.random.uniform(0.80, 0.95, l_count_high)     # 潜力股
            
            # 🚨 低信誉组的恶意NA策略分配（简化为两种类型）
            l_mal_type1 = max(1, l_count_malicious // 2)      # 高成功率+高延迟
            l_mal_type4 = l_count_malicious - l_mal_type1     # 传统低成功率（但在低信誉组会稍高一些）
            
            l_sr_mal_type1 = np.random.uniform(0.75, 0.90, l_mal_type1)  # 高成功率伪装
            l_sr_mal_type4 = np.random.uniform(0.15, 0.40, l_mal_type4)  # 相对低成功率（比极低组稍高）
            
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
            
            # 高信誉组 (6600-8200)：主要优秀但有例外 + 复杂恶意NA
            # 45%高能力, 25%中等能力, 10%意外低能力, 20%恶意NA
            h_count_high = int(group_sizes[2] * 0.45)
            h_count_med = int(group_sizes[2] * 0.25)
            h_count_low = int(group_sizes[2] * 0.10)
            h_count_malicious = group_sizes[2] - h_count_high - h_count_med - h_count_low
            
            h_sr_high = np.random.uniform(0.75, 0.90, h_count_high)     # 高能力
            h_sr_med = np.random.uniform(0.60, 0.75, h_count_med)       # 中等能力
            h_sr_low = np.random.uniform(0.40, 0.60, h_count_low)       # 意外低能力
            
            # 🚨 高信誉组的恶意NA策略分配（简化为一种类型）
            # 高信誉组只保留Type 1（高成功率+高延迟），因为Type 4在高信誉组不合理
            h_mal_type1 = h_count_malicious  # 全部为Type 1
            
            h_sr_mal_type1 = np.random.uniform(0.82, 0.95, h_mal_type1)  # 非常高的成功率伪装
            
            h_sr_malicious = h_sr_mal_type1
            h_malicious_types = np.full(h_mal_type1, 1)  # 全部为Type 1
            
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
            
            # 极高信誉组 (8200-9800)：大部分优秀但仍有变化 + 复杂恶意NA
            # 55%极高能力, 25%高能力, 5%中等能力, 15%恶意NA(终极伪装)
            vh_count_very_high = int(group_sizes[3] * 0.55)
            vh_count_high = int(group_sizes[3] * 0.25)
            vh_count_med = int(group_sizes[3] * 0.05)
            vh_count_malicious = group_sizes[3] - vh_count_very_high - vh_count_high - vh_count_med
            
            vh_sr_very_high = np.random.uniform(0.85, 0.98, vh_count_very_high)  # 极高能力
            vh_sr_high = np.random.uniform(0.75, 0.85, vh_count_high)            # 高能力
            vh_sr_med = np.random.uniform(0.65, 0.75, vh_count_med)              # 中等能力
            
            # 🚨 极高信誉组的恶意NA策略分配（简化为一种类型）
            # 极高信誉组只保留Type 1（极高成功率+故意高延迟）
            vh_mal_type1 = vh_count_malicious  # 全部为Type 1
            
            vh_sr_mal_type1 = np.random.uniform(0.90, 0.98, vh_mal_type1)  # 极高成功率伪装！
            
            vh_sr_malicious = vh_sr_mal_type1
            vh_malicious_types = np.full(vh_mal_type1, 1)  # 全部为Type 1
            
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
            
            # 合并所有组的数据
            self.fixed_initial_success_rate = np.concatenate(success_rates_list)
            self.fixed_malicious_flags = np.concatenate(malicious_flags_list)  # 保存恶意标记
            self.fixed_malicious_types = np.concatenate(malicious_types_list)  # 保存恶意类型
            
            # 🎨 分配活跃度：恶意NA不通过活跃度区分，使用与正常NA相同的分配规则
            
            
            # 🎯 根据恶意NA类型分配延迟：这是恶意行为的关键体现
            delay_list = []
            
            for group_idx, group_size in enumerate(group_sizes):
                malicious_mask = malicious_flags_list[group_idx]
                malicious_types = malicious_types_list[group_idx]
                normal_mask = ~malicious_mask
                
                # 正常NA的延迟等级分配（信誉越高，延迟等级越低）
                if group_idx == 0:  # 极低信誉组
                    normal_delay = np.random.uniform(0.3, 0.8, np.sum(normal_mask))
                elif group_idx == 1:  # 低信誉组
                    normal_delay = np.random.uniform(0.25, 0.65, np.sum(normal_mask))
                elif group_idx == 2:  # 高信誉组
                    normal_delay = np.random.uniform(0.15, 0.5, np.sum(normal_mask))
                else:  # 极高信誉组
                    normal_delay = np.random.uniform(0.1, 0.4, np.sum(normal_mask))
                
                # 🚨 恶意NA的延迟分配（根据类型进行精心设计）
                malicious_delay = np.zeros(np.sum(malicious_mask))
                mal_indices = np.where(malicious_mask)[0]
                
                for i, mal_idx in enumerate(mal_indices):
                    mal_type = malicious_types[mal_idx]
                    
                    if mal_type == 1:  # Type 1: 高成功率+高延迟（故意延迟攻击）
                        # 这类恶意NA故意制造高延迟等级来影响系统性能
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.7, 1.0)  # 极高延迟等级
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.6, 0.9)  # 很高延迟等级
                        elif group_idx == 2:
                            malicious_delay[i] = np.random.uniform(0.5, 0.8)  # 高延迟等级（相对同组正常NA）
                        else:
                            malicious_delay[i] = np.random.uniform(0.4, 0.7)  # 明显高于正常（相对同组）
                            
                    else:  # Type 4: 传统低成功率
                        # 这类恶意NA各方面都差，包括延迟等级
                        if group_idx == 0:
                            malicious_delay[i] = np.random.uniform(0.6, 0.95)  # 高延迟等级
                        elif group_idx == 1:
                            malicious_delay[i] = np.random.uniform(0.55, 0.9)  # 高延迟等级
                        # 高信誉组和极高信誉组不会有Type 4
                
                # 组合正常和恶意NA的延迟
                group_delay = np.zeros(group_size)
                group_delay[normal_mask] = normal_delay
                group_delay[malicious_mask] = malicious_delay
                delay_list.append(group_delay)
            
            self.fixed_initial_signature_delay = np.concatenate(delay_list)
            
            # 🍽️ 新增：为每个NA设置固定的初始饥饿度（0-50%随机分布，模拟初始负载不均衡）
            self.fixed_initial_hunger = np.random.uniform(0.0, 0.5, n_na)
            
            # 对应设置总事务数和成功数
            self.fixed_initial_total_tx = np.random.randint(20, 80, n_na)
            self.fixed_initial_success_count = (self.fixed_initial_success_rate * self.fixed_initial_total_tx).astype(int)
            
            # 延迟配置 - 改为延迟等级百分比 (0.0-1.0)
            self.ideal_delay_grade = 0.0  # 理想延迟等级：0% (最佳性能)
            self.max_acceptable_delay_grade = 1.0  # 最大可接受延迟等级：100% (最差性能)
            
            # 注意：不重置随机种子，保持全局种子的确定性
            
            print("🔍 复杂NA集合特征预览 (包含精细化恶意NA类型):")
            print("编号\t初始信誉\t成功率\t签名延迟(ms)\t饥饿度\t延迟等级\t信誉分组\t\t能力类型\t\t恶意类型")
            
            # 定义信誉分组函数
            def get_reputation_group(rep):
                if rep < 4500:
                    return "极低信誉组"
                elif rep < 6600:
                    return "低信誉组"
                elif rep < 8200:
                    return "高信誉组"
                else:
                    return "极高信誉组"
            
            # 定义恶意类型说明
            def get_malicious_type_desc(mal_type):
                if mal_type == 1:
                    return "🚨高成功率+高延迟"
                elif mal_type == 4:
                    return "🚨传统低成功率"
                else:
                    return "✅正常"
            
            # 定义能力类型函数（考虑恶意标记和类型）
            def get_ability_type(rep, success_rate, is_malicious, mal_type):
                if is_malicious:
                    if rep < 4500:
                        return f"🚨明显恶意(T{mal_type})"
                    elif rep < 6600:
                        return f"🚨低级恶意(T{mal_type})" 
                    elif rep < 8200:
                        return f"🚨中级恶意(T{mal_type})"
                    else:
                        return f"🚨高级恶意(T{mal_type})"
                
                # 正常NA的能力分类
                if rep < 4500:  # 极低信誉组
                    if success_rate >= 0.75:
                        return "🌟隐藏高手"
                    elif success_rate >= 0.55:
                        return "💪中等能力"
                    else:
                        return "📉真正低能力"
                elif rep < 6600:  # 低信誉组
                    if success_rate >= 0.80:
                        return "🚀潜力股"
                    elif success_rate >= 0.60:
                        return "⚖️中等能力"
                    else:
                        return "📉中低能力"
                elif rep < 8200:  # 高信誉组
                    if success_rate >= 0.75:
                        return "⭐高能力"
                    elif success_rate >= 0.60:
                        return "🔘中等能力"
                    else:
                        return "❗意外低能力"
                else:  # 极高信誉组
                    if success_rate >= 0.85:
                        return "👑极高能力"
                    elif success_rate >= 0.75:
                        return "⭐高能力"
                    else:
                        return "🔘中等能力"
            
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
            
            # 详细统计分析（包含恶意NA统计）
            very_low_indices = np.where(self.fixed_initial_reputation < 4500)[0]
            low_indices = np.where((self.fixed_initial_reputation >= 4500) & (self.fixed_initial_reputation < 6600))[0]
            high_indices = np.where((self.fixed_initial_reputation >= 6600) & (self.fixed_initial_reputation < 8200))[0]
            very_high_indices = np.where(self.fixed_initial_reputation >= 8200)[0]
            
            print(f"\n精细化NA分布统计 (恶意NA类型分析):")
            
            # 定义恶意类型统计函数
            def count_malicious_types(indices):
                type_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                for i in indices:
                    if self.fixed_malicious_flags[i]:
                        mal_type = self.fixed_malicious_types[i]
                        type_counts[mal_type] += 1
                return type_counts
            
            # 极低信誉组统计
            if len(very_low_indices) > 0:
                vl_malicious = sum(1 for i in very_low_indices if self.fixed_malicious_flags[i])
                vl_normal = len(very_low_indices) - vl_malicious
                vl_mal_types = count_malicious_types(very_low_indices)
                vl_high_ability = sum(1 for i in very_low_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.75)
                vl_med_ability = sum(1 for i in very_low_indices if not self.fixed_malicious_flags[i] and 0.55 <= self.fixed_initial_success_rate[i] < 0.75)
                vl_low_ability = vl_normal - vl_high_ability - vl_med_ability
                print(f"极低信誉组 ({len(very_low_indices)}个): 🚨恶意{vl_malicious}个[T1:{vl_mal_types[1]}, T2:{vl_mal_types[2]}, T3:{vl_mal_types[3]}, T4:{vl_mal_types[4]}] | 正常{vl_normal}个(隐藏高手{vl_high_ability}, 中等{vl_med_ability}, 低能力{vl_low_ability})")
            
            # 低信誉组统计
            if len(low_indices) > 0:
                l_malicious = sum(1 for i in low_indices if self.fixed_malicious_flags[i])
                l_normal = len(low_indices) - l_malicious
                l_mal_types = count_malicious_types(low_indices)
                l_high_ability = sum(1 for i in low_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.80)
                l_med_ability = sum(1 for i in low_indices if not self.fixed_malicious_flags[i] and 0.60 <= self.fixed_initial_success_rate[i] < 0.80)
                l_low_ability = l_normal - l_high_ability - l_med_ability
                print(f"低信誉组 ({len(low_indices)}个): 🚨恶意{l_malicious}个[T1:{l_mal_types[1]}, T2:{l_mal_types[2]}, T3:{l_mal_types[3]}, T4:{l_mal_types[4]}] | 正常{l_normal}个(潜力股{l_high_ability}, 中等{l_med_ability}, 中低{l_low_ability})")
            
            # 高信誉组统计
            if len(high_indices) > 0:
                h_malicious = sum(1 for i in high_indices if self.fixed_malicious_flags[i])
                h_normal = len(high_indices) - h_malicious
                h_mal_types = count_malicious_types(high_indices)
                h_high_ability = sum(1 for i in high_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.75)
                h_med_ability = sum(1 for i in high_indices if not self.fixed_malicious_flags[i] and 0.60 <= self.fixed_initial_success_rate[i] < 0.75)
                h_low_ability = h_normal - h_high_ability - h_med_ability
                print(f"高信誉组 ({len(high_indices)}个): 🚨恶意{h_malicious}个 | 正常{h_normal}个(高能力{h_high_ability}, 中等能力{h_med_ability}, 意外低能力{h_low_ability})")
            
            # 极高信誉组统计
            if len(very_high_indices) > 0:
                vh_malicious = sum(1 for i in very_high_indices if self.fixed_malicious_flags[i])
                vh_normal = len(very_high_indices) - vh_malicious
                vh_very_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and self.fixed_initial_success_rate[i] >= 0.85)
                vh_high_ability = sum(1 for i in very_high_indices if not self.fixed_malicious_flags[i] and 0.75 <= self.fixed_initial_success_rate[i] < 0.85)
                vh_med_ability = vh_normal - vh_very_high_ability - vh_high_ability
                print(f"极高信誉组 ({len(very_high_indices)}个): 🚨恶意{vh_malicious}个 | 正常{vh_normal}个(极高能力{vh_very_high_ability}, 高能力{vh_high_ability}, 中等能力{vh_med_ability})")
            
            # 总体恶意NA统计
            total_malicious = np.sum(self.fixed_malicious_flags)
            total_normal = n_na - total_malicious
            print(f"\n🚨 恶意NA总体统计: 恶意{total_malicious}个 ({total_malicious/n_na:.1%}), 正常{total_normal}个 ({total_normal/n_na:.1%})")
            

        else:
            # 非固定NA模式下也需要初始化延迟参数
            self.ideal_delay = 100.0
            self.max_acceptable_delay = self.ideal_delay * 3
        
        # 初始化当前参数
        self.current_time = 30  # 初始时间点设为30分钟
        
        
        # 调用reset初始化所有参数
        self.reset()
    
    def calculate_delay_performance(self, delay_grade):
        """
        计算延迟等级 (延迟越大，等级数值越大)
        
        Args:
            delay_grade: 延迟等级（0.0-1.0）
            
        Returns:
            float: 延迟等级，0.0表示最低延迟等级，1.0表示最高延迟等级
        """
        # 直接返回输入的延迟等级，确保在0.0-1.0范围内
        return max(0.0, min(1.0, delay_grade))
    
    
    def reset(self):
        """
        重置环境状态，固定初始信誉，减少动态调整。
        """
        if self.use_fixed_nas:
            # 使用固定的NA集合，只重置动态状态
            self.reputation = self.fixed_initial_reputation.copy()
            self.total_tx = self.fixed_initial_total_tx.copy()
            self.success_count = self.fixed_initial_success_count.copy()
            self.success_rate = self.fixed_initial_success_rate.copy()
            
            # 🚀 签名延迟
            self.signature_delay = self.fixed_initial_signature_delay.copy()
            # 🍽️ 新增：使用固定的初始饥饿度（每个NA都有预设的初始饥饿度）
            self.hunger = self.fixed_initial_hunger.copy()
            # 🍽️ 记录每个NA上次被选中的时间步（根据初始饥饿度反推）
            # 将初始饥饿度转换为对应的"上次选中时间"，模拟历史选择情况
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)
            for i in range(self.n_na):
                # 根据饥饿度反推上次被选中的时间
                # hunger = log(1 + steps/20) / log(11), 反解得到 steps
                if self.hunger[i] > 0:
                    # steps = 20 * (11^hunger - 1)
                    estimated_steps = int(20 * (np.power(11, self.hunger[i]) - 1))
                    self.last_selected_time[i] = -estimated_steps
                else:
                    self.last_selected_time[i] = 0  # 饥饿度为0表示刚被选中
        else:
            # 原来的随机重置方式
            self.reputation = np.random.uniform(3300, 10000, self.n_na)
            self.total_tx = np.random.randint(1, 100, self.n_na)
            self.success_count = np.array([
                np.random.randint(0, self.total_tx[i] + 1) 
                for i in range(self.n_na)
            ])
            self.success_rate = self.success_count / self.total_tx
            
            # 🚀 随机初始化签名延迟
            self.signature_delay = np.random.uniform(50, 200, self.n_na)
            # 🍽️ 新增：随机初始化饥饿度（0-50%范围，模拟不同的初始负载状态）
            self.hunger = np.random.uniform(0.0, 0.5, self.n_na)
            # 🍽️ 记录每个NA上次被选中的时间步（根据饥饿度反推）
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)
            for i in range(self.n_na):
                if self.hunger[i] > 0:
                    estimated_steps = int(20 * (np.power(11, self.hunger[i]) - 1))
                    self.last_selected_time[i] = -estimated_steps
                else:
                    self.last_selected_time[i] = 0
            self.last_selected_time = np.full(self.n_na, -10, dtype=int)

        # 重置时间相关状态  
        self.current_time = 30
        # 🍽️ 新增：当前训练步数计数器，用于计算饥饿度
        self.training_step = 0

        if self.freeze_hunger:
            self.hunger = np.full(self.n_na, self.hunger_constant, dtype=float)
            self.last_selected_time = np.zeros(self.n_na, dtype=int)
        
        # 🔧 新增：重置所有NA的滑动窗口队列系统
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

        # 记录本episode开始时的初始信誉（固定的基础信誉）
        self.episode_start_reputation = self.reputation.copy()
        # 重置上一步信誉记录
        self.last_step_reputation = self.reputation.copy()

        return self.get_state()
    def get_state(self):
        # 🚀 更新：移除活跃度(activity)特征，保留4个特征 (reputation, success_rate, signature_delay, hunger)
        raw_state = np.column_stack([
            self.reputation,
            self.success_rate,
            self.signature_delay,
            self.hunger
        ])

        # 归一化
        normalized_state = raw_state.copy()
        normalized_state[:, 0] = (self.reputation - 3000) / (10000 - 3000)  # reputation归一化
        normalized_state[:, 1] = self.success_rate  # success_rate已在[0,1]
        # 签名延迟已经是延迟等级[0,1]，无需归一化
        normalized_state[:, 2] = self.signature_delay
        normalized_state[:, 3] = self.hunger  # hunger已在[0,1]
        return normalized_state

    def step(self, action_na_index, selected_mask=None):
        """
        简化的环境step方法，只处理单个NA
        
        Args:
            action_na_index: int - 选中的NA索引
            selected_mask: 保留参数兼容性（暂未使用）
        
        Returns:
            next_state: 更新后的状态
            reward: 单个NA的奖励
            done: 是否结束（始终为False）
        """
        # 确保索引在有效范围内
        na_idx = max(0, min(int(action_na_index), self.n_na - 1))
        
        # 记录操作前的信誉和成功率
        old_reputation = self.reputation[na_idx]
        old_success_rate = self.success_rate[na_idx]
        
        # 更新时间
        self.current_time += 1
        # 🍽️ 新增：更新训练步数
        self.training_step += 1
        # self._update_hunger(na_idx)
        # 🚀 模拟当前事务的延迟等级（可控随机波动）
        base_delay_grade = self.signature_delay[na_idx]
        if self.enable_random_fluctuation:
            # 添加±0.05的随机波动（延迟等级范围0.0-1.0）
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
        else:
            # 固定模式：仅添加小幅波动(±0.01)模拟测量误差
            current_delay_grade = base_delay_grade + np.random.uniform(-0.01, 0.01)
        current_delay_grade = np.clip(current_delay_grade, 0.0, 1.0)  # 确保延迟等级在0.0-1.0范围内
        
        # 🎲 成功率随机波动（如果启用）
        effective_success_rate = self.success_rate[na_idx]
        if self.enable_random_fluctuation:
            # 成功率可上下浮动±5%，但不超出[0,1]范围
            fluctuation = np.random.uniform(-0.05, 0.05)
            effective_success_rate = np.clip(effective_success_rate + fluctuation, 0.0, 1.0)
        
        # 基于（可能波动的）成功率判断交易成功
        success = np.random.random() < effective_success_rate
        
        # 计算奖励参数（用于信誉更新）
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
        
        # 更新NA参数
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

        # 🚀 修改：奖励函数改为基于NA成功率，不受当前事务成功与否影响
        # 
        # 修改前：奖励 = success ? +1.0 : -1.0 + 延迟调整
        # 修改后：奖励 = 2.0 * weighted_success_rate - 1.0 + 加权延迟调整 + 饥饿度调整
        #  
        # 优势：
        # 1. 更稳定的学习信号：不受单次事务随机性影响
        # 2. 反映NA真实能力：基于加权成功率的历史表现
        # 3. 平滑的奖励梯度：加权成功率从0.0到1.0，奖励从-1.0到+1.0线性变化
        # 4. 避免噪声干扰：单次事务失败不会给高质量NA负奖励
        # 5. 🍽️ 负载均衡考虑：高饥饿度的高质量NA会获得额外奖励
        # 6. 🎯 新增：基于滑动窗口的加权成功率和延迟等级
        
        # 获取该NA的滑动窗口统计（包含加权信息）
        window_summary = self.get_na_window_summary(na_idx)
        
        # 使用加权成功率替代简单成功率
        # 策略：
        # 1. 如果窗口中有历史数据，使用加权成功率（体现历史趋势，越新的数据权重越高）
        # 2. 如果窗口中没有数据，使用当前实时成功率（基于最新的统计）
        # 3. 没有数据时，加权成功率返回0.0，会导致负奖励，促使探索该NA
        if window_summary['total_transactions'] > 0:
            effective_success_rate = window_summary['weighted_success_rate']
            effective_delay_performance = window_summary['weighted_delay_grade']
        else:
            # 没有历史数据时的回退策略
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
        
        # 基础奖励基于加权成功率，扩大奖励范围并提高正向奖励阈值
        # 修改前：2.5 * effective_success_rate - 1.5 (成功率0.6时获得0奖励)
        # 修改后：4.0 * effective_success_rate - 2.8 (成功率0.7时获得0奖励，提高正向奖励阈值)
        weighted_success_rate_reward = 4.0 * effective_success_rate - 2.8  # 将[0,1]映射到[-2.8,1.2]
        
        # 延迟性能调整：基于加权延迟等级，扩大高延迟性能奖励范围
        # 延迟等级范围[0,1]，0表示最佳，1表示最差
        # 修改前：0.3 * (1 - 2 * effective_delay_performance) 范围[0.3,-0.3]
        # 修改后：0.8 * (1 - 2 * effective_delay_performance) 范围[0.8,-0.8]，大幅扩大奖励范围
        weighted_delay_grade_bonus = 0.8 * (1 - 2 * effective_delay_performance)  # 将[0,1]映射到[0.8,-0.8]
        
        # 🍽️ 饥饿度调整（可控随机波动）
        # 设计原则：
        # 1. 高成功率+低延迟+高饥饿度 = 额外奖励（鼓励选择久未被选中的优质NA）
        # 2. 低成功率或高延迟+高饥饿度 = 微小奖励或无奖励（不应优先选择质量差的NA）
        # 3. 饥饿度权重同时考虑加权成功率和加权延迟性能
        
        # 🎲 饥饿度随机波动（如果启用）
        effective_hunger = self.hunger[na_idx]
        if self.enable_random_fluctuation:
            # 饥饿度可上下浮动±10%，但不超出[0,1]范围
            hunger_fluctuation = np.random.uniform(-0.1, 0.1)
            effective_hunger = np.clip(effective_hunger + hunger_fluctuation, 0.0, 1.0)
        
        # 综合质量权重：基于加权成功率60% + 加权延迟性能40%
        # 只有加权成功率和延迟性能都较好的NA才能获得显著的饥饿度奖励
        # 注意：延迟性能现在是0表示最佳，1表示最差，所以需要用(1-延迟性能)
        quality_weight = 0.6 * effective_success_rate + 0.4 * (1 - effective_delay_performance)
        
        # 动态饥饿度权重：质量越高，饥饿度权重越大，扩大权重范围
        # 使用平滑的非线性函数，确保高质量NA获得更大的饥饿度权重
        # 基础权重范围：0.2 到 1.5，大幅扩大权重范围，通过质量权重的平方来增强高质量NA的优势
        base_hunger_weight = 0.2 + 1.3 * (quality_weight ** 1.5)  # 使用1.5次方增强非线性效果
        
        # 进一步根据质量等级调整权重，确保明显的差异化，扩大权重范围
        if quality_weight >= 0.8:
            # 顶级质量NA：最大饥饿度权重，大幅提高上限
            dynamic_hunger_weight = min(1.8, base_hunger_weight * 1.2)  # 最高权重1.8
        elif quality_weight >= 0.6:
            # 高质量NA：较大饥饿度权重
            dynamic_hunger_weight = base_hunger_weight * 1.0  # 标准权重
        elif quality_weight >= 0.4:
            # 中等质量NA：中等饥饿度权重
            dynamic_hunger_weight = base_hunger_weight * 0.8  # 降低20%
        else:
            # 低质量NA：较小饥饿度权重
            dynamic_hunger_weight = base_hunger_weight * 0.5  # 降低50%
        
        # 计算饥饿度奖励
        hunger_bonus = dynamic_hunger_weight * quality_weight * effective_hunger
        hunger_bonus = hunger_bonus * self.hunger_weight
        
        # 综合奖励：基于加权成功率 + 加权延迟等级 + 饥饿度奖励
        reward = weighted_success_rate_reward + weighted_delay_grade_bonus + hunger_bonus
        
        # 🔧 新增：记录事务行为到该NA的滑动窗口队列
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
        
        # 检查是否需要进行队列打包
        self.current_step_count += 1
        if self.current_step_count % self.window_pack_interval == 0:
            self._pack_all_na_windows()
        
        # 更新饥饿度，保证奖励使用更新前的饥饿水平
        if self.enable_hunger_update and not self.freeze_hunger:
            self._update_hunger(na_idx)
        
        return self.get_state(), reward, False

    def _update_na_window_queue(self, na_idx, transaction_data):
        """
        🔧 更新指定NA的当前事务包，累积事务行为数据
        
        Args:
            na_idx: NA索引
            transaction_data: 包含事务信息的字典
        """
        current_pack = self.na_current_pack[na_idx]
        
        # 如果是新包的开始，记录开始步数
        if len(current_pack['transactions']) == 0:
            current_pack['start_step'] = self.current_step_count
        
        # 添加事务到当前包
        current_pack['transactions'].append(transaction_data)
        current_pack['total_count'] += 1
        if transaction_data['success']:
            current_pack['success_count'] += 1
        
        # 记录信誉变化
        reputation_change = transaction_data['reputation_after'] - transaction_data['reputation_before']
        current_pack['reputation_changes'].append(reputation_change)
        
        # 更新结束步数
        current_pack['end_step'] = self.current_step_count
    
    def _pack_all_na_windows(self):
        """
        🔧 将所有NA的当前事务包打包到各自的队列中
        """
        for na_idx in range(self.n_na):
            current_pack = self.na_current_pack[na_idx]
            
            # 只有当包含事务时才打包
            if current_pack['total_count'] > 0:
                # 计算包的平均延迟
                avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions']) if current_pack['transactions'] else 0.0
                
                # 计算延迟等级
                delay_grade = self.calculate_delay_performance(avg_delay)
                
                # 计算包的统计信息
                pack_summary = {
                    'start_step': current_pack['start_step'],
                    'end_step': current_pack['end_step'],
                    'transaction_count': current_pack['total_count'],
                    'success_count': current_pack['success_count'],
                    'success_rate': current_pack['success_count'] / current_pack['total_count'],
                    'total_reputation_change': sum(current_pack['reputation_changes']),
                    'avg_reputation_change': sum(current_pack['reputation_changes']) / len(current_pack['reputation_changes']) if current_pack['reputation_changes'] else 0,
                    'avg_delay': avg_delay,
                    'delay_grade': delay_grade,  # 新增：延迟等级字段
                    'transactions': current_pack['transactions'].copy()  # 保留详细事务数据
                }
                
                # 将打包的数据加入队列（自动处理容量限制）
                self.na_window_queues[na_idx].append(pack_summary)
                
                # 重置当前包
                current_pack['transactions'] = []
                current_pack['success_count'] = 0
                current_pack['total_count'] = 0
                current_pack['reputation_changes'] = []
                current_pack['start_step'] = 0
                current_pack['end_step'] = 0
    
    def get_na_window_summary(self, na_idx):
        """
        🔍 获取指定NA滑动窗口队列的统计摘要
        
        Args:
            na_idx: NA索引
            
        Returns:
            dict: 包含窗口队列内事务行为统计的摘要
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
                'weighted_success_rate': 0.0,  # 没有数据时，加权成功率为0
                'weighted_avg_delay': 0.0,     # 没有数据时，加权平均延迟为0
                'weighted_delay_grade': 0.0,   # 没有数据时，延迟等级为0（最差）
                'delay_grade': "N/A",          # 没有数据时显示N/A
                'step_range': None
            }
        
        # 计算队列中所有包的统计信息
        total_transactions = sum(pack['transaction_count'] for pack in queue)
        total_success_count = sum(pack['success_count'] for pack in queue)
        total_reputation_change = sum(pack['total_reputation_change'] for pack in queue)
        
        # 加上当前正在积累的包
        total_transactions += current_pack['total_count']
        total_success_count += current_pack['success_count']
        total_reputation_change += sum(current_pack['reputation_changes'])
        
        # 计算整体成功率
        overall_success_rate = total_success_count / total_transactions if total_transactions > 0 else 0.0
        
        # 计算平均包成功率
        pack_success_rates = [pack['success_rate'] for pack in queue if pack['transaction_count'] > 0]
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            pack_success_rates.append(current_pack_success_rate)
        
        avg_pack_success_rate = sum(pack_success_rates) / len(pack_success_rates) if pack_success_rates else 0.0
        
        # 计算加权成功率和加权延迟（越新的包权重越高）
        weighted_success_rate = 0.0
        weighted_avg_delay = 0.0
        weighted_delay_grade = 0.0  # 新增：加权延迟等级
        total_weight = 0.0
        
        # 为队列中的包计算权重（越新权重越高）
        if queue:
            for i, pack in enumerate(queue):
                # 权重计算：线性递增，最新的包权重最高
                weight = i + 1  # 权重从1开始递增
                
                # 计算该包的平均延迟
                pack_avg_delay = 0.0
                if pack['transactions']:
                    pack_avg_delay = sum(tx['delay'] for tx in pack['transactions']) / len(pack['transactions'])
                
                # 使用包中的delay_grade字段（如果存在）
                pack_delay_grade = pack.get('delay_grade', 0.0)
                
                weighted_success_rate += pack['success_rate'] * weight
                weighted_avg_delay += pack_avg_delay * weight
                weighted_delay_grade += pack_delay_grade * weight
                total_weight += weight
        
        # 如果有当前正在积累的包，也加入计算（权重最高）
        if current_pack['total_count'] > 0:
            current_pack_success_rate = current_pack['success_count'] / current_pack['total_count']
            
            # 计算当前包的平均延迟
            current_pack_avg_delay = 0.0
            if current_pack['transactions']:
                current_pack_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions'])
            
            # 计算当前包的延迟等级
            current_pack_delay_grade = self.calculate_delay_performance(current_pack_avg_delay)
            
            # 当前包权重最高
            current_weight = len(queue) + 1
            weighted_success_rate += current_pack_success_rate * current_weight
            weighted_avg_delay += current_pack_avg_delay * current_weight
            weighted_delay_grade += current_pack_delay_grade * current_weight
            total_weight += current_weight
        
        # 归一化加权值
        if total_weight > 0:
            weighted_success_rate /= total_weight
            weighted_avg_delay /= total_weight
            weighted_delay_grade /= total_weight
        
        # 如果没有计算出加权延迟等级，则使用默认值
        if total_weight == 0:
            # 没有历史数据时，使用中等延迟等级
            weighted_delay_grade = 0.5  # 中等性能
        
        # 基于延迟等级的字母等级显示逻辑
        delay_grade = "A"
        if weighted_delay_grade > 0.75:  # 延迟等级超过75%
            delay_grade = "D"
        elif weighted_delay_grade > 0.5:  # 延迟等级超过50%
            delay_grade = "C"
        elif weighted_delay_grade > 0.25:  # 延迟等级超过25%
            delay_grade = "B"
        else:  # 延迟等级25%以下
            delay_grade = "A"
        
        # 获取步数范围
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
            'weighted_delay_grade': weighted_delay_grade,  # 添加加权延迟等级
            'delay_grade': delay_grade,
            'step_range': step_range,
            'pack_details': [
                {
                    'step_range': (pack['start_step'], pack['end_step']),
                    'transaction_count': pack['transaction_count'],
                    'success_rate': pack['success_rate'],
                    'reputation_change': pack['total_reputation_change'],
                    'weight': i + 1,  # 包的权重
                    'avg_delay': sum(tx['delay'] for tx in pack['transactions']) / len(pack['transactions']) if pack['transactions'] else 0.0
                } for i, pack in enumerate(queue)
            ]
        }
    
    def print_sliding_window_summary(self, max_nas_to_show=10):
        """
        🔍 打印滑动窗口统计摘要
        
        Args:
            max_nas_to_show: 最多显示多少个NA的详细信息
        """
        print(f"\n🔧 滑动窗口队列统计摘要 (打包间隔: {self.window_pack_interval}步, 队列容量: {self.window_queue_size})")
        print("=" * 140)
        
        summaries = self.get_all_nas_window_summary()
        
        # 按加权成功率排序，优先显示性能更好的NA
        sorted_nas = sorted(summaries.items(), key=lambda x: x[1]['weighted_success_rate'], reverse=True)
        
        print(f"{'NA编号':<6} {'队列':<6} {'当前包':<8} {'总事务':<8} {'加权成功率':<12} {'加权延迟':<10} {'延迟等级':<8} {'信誉变化':<10} {'步数范围':<15}")
        print("-" * 140)
        
        shown_count = 0
        for na_idx, summary in sorted_nas:
            if shown_count >= max_nas_to_show:
                break
                
            if summary['total_transactions'] > 0:
                step_range_str = f"{summary['step_range']}" if summary['step_range'] else "None"
                print(f"NA{na_idx:<4} {summary['queue_size']:<6} {summary['current_pack_size']:<8} {summary['total_transactions']:<8} {summary['weighted_success_rate']:<12.3f} {summary['weighted_avg_delay']:<10.1f} {summary['delay_grade']:<8} {summary['total_reputation_change']:<10.1f} {step_range_str:<15}")
                shown_count += 1
        
        # 统计总体信息
        total_transactions = sum(s['total_transactions'] for s in summaries.values())
        total_success = sum(s['total_success_count'] for s in summaries.values())
        active_nas = sum(1 for s in summaries.values() if s['total_transactions'] > 0)
        
        # 计算系统级加权指标
        system_weighted_success_rate = 0.0
        system_weighted_delay = 0.0
        total_weight = 0.0
        
        for summary in summaries.values():
            if summary['total_transactions'] > 0:
                weight = summary['total_transactions']  # 以事务数作为权重
                system_weighted_success_rate += summary['weighted_success_rate'] * weight
                system_weighted_delay += summary['weighted_avg_delay'] * weight
                total_weight += weight
        
        if total_weight > 0:
            system_weighted_success_rate /= total_weight
            system_weighted_delay /= total_weight
        
        print("-" * 140)
        print(f"总体统计: {active_nas}个NA有事务记录, 总事务数: {total_transactions}, 总成功数: {total_success}")
        if total_transactions > 0:
            print(f"系统整体成功率: {total_success/total_transactions:.3f}, 系统加权成功率: {system_weighted_success_rate:.3f}")
            print(f"系统加权平均延迟: {system_weighted_delay:.1%}, 平均每个活跃NA: {total_transactions/max(1, active_nas):.1f}次事务")
        print(f"当前步数: {self.current_step_count}")
        
        if shown_count < active_nas:
            print(f"(只显示了前{shown_count}个最活跃的NA，还有{active_nas - shown_count}个NA有事务记录)")

    def show_na_detailed_queue(self, na_idx, max_packs_to_show=5):
        """
        🔍 显示指定NA的详细队列信息，包括权重计算
        
        Args:
            na_idx: NA索引
            max_packs_to_show: 最多显示多少个包的详细信息
        """
        if na_idx >= self.n_na:
            print(f"❌ 无效的NA索引: {na_idx}")
            return
        
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\nNA{na_idx} 详细队列信息")
        print("=" * 100)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("该NA暂无事务记录")
            return
        
        # 显示队列中的包（按权重从低到高）
        print(f"队列中的包 (共{len(queue)}个，按入队顺序，越新权重越高):")
        print(f"{'包序号':<8} {'权重':<6} {'步数范围':<15} {'事务数':<8} {'成功率':<10} {'平均延迟':<10} {'信誉变化':<10}")
        print("-" * 100)
        
        shown_packs = 0
        for i, pack in enumerate(queue):
            if shown_packs >= max_packs_to_show:
                break
            
            weight = i + 1
            step_range = f"({pack['start_step']}-{pack['end_step']})"
            avg_delay = pack.get('avg_delay', 0.0)
            
            print(f"包{i+1:<5} {weight:<6} {step_range:<15} {pack['transaction_count']:<8} {pack['success_rate']:<10.3f} {avg_delay:<10.1%} {pack['total_reputation_change']:<10.1f}")
            shown_packs += 1
        
        # 显示当前正在积累的包
        if current_pack['total_count'] > 0:
            current_weight = len(queue) + 1
            current_success_rate = current_pack['success_count'] / current_pack['total_count']
            current_avg_delay = sum(tx['delay'] for tx in current_pack['transactions']) / len(current_pack['transactions']) if current_pack['transactions'] else 0.0
            current_step_range = f"({current_pack['start_step']}-{current_pack['end_step']})"
            current_reputation_change = sum(current_pack['reputation_changes'])
            
            print(f"当前包 {current_weight:<6} {current_step_range:<15} {current_pack['total_count']:<8} {current_success_rate:<10.3f} {current_avg_delay:<10.1%} {current_reputation_change:<10.1f}")
        
        # 显示加权计算结果
        summary = self.get_na_window_summary(na_idx)
        print("\n📈 加权计算结果:")
        print(f"  - 加权成功率: {summary['weighted_success_rate']:.3f}")
        print(f"  - 加权平均延迟: {summary['weighted_avg_delay']:.1%}")
        print(f"  - 延迟等级: {summary['delay_grade']}")
        print(f"  - 总事务数: {summary['total_transactions']}")
        print(f"  - 队列容量使用: {len(queue)}/{self.window_queue_size}")
        
        if len(queue) < max_packs_to_show and len(queue) > shown_packs:
            print(f"\n(还有{len(queue) - shown_packs}个包未显示)")

    def get_all_nas_window_summary(self):
        """
        🔍 获取所有NA的滑动窗口统计摘要
        
        Returns:
            dict: 以NA索引为键的摘要字典
        """
        summaries = {}
        for na_idx in range(self.n_na):
            summaries[na_idx] = self.get_na_window_summary(na_idx)
        return summaries
    
    def print_na_queue_details(self, na_idx, max_packs_to_show=5):
        """
        🔍 打印指定NA的详细队列信息
        
        Args:
            na_idx: NA索引
            max_packs_to_show: 最多显示多少个包的详细信息
        """
        queue = self.na_window_queues[na_idx]
        current_pack = self.na_current_pack[na_idx]
        
        print(f"\n🔍 NA{na_idx} 详细队列信息:")
        print("=" * 80)
        
        if len(queue) == 0 and current_pack['total_count'] == 0:
            print("  该NA暂无任何事务记录")
            return
        
        # 显示队列中的包
        if len(queue) > 0:
            print(f"📦 已打包数据 (队列大小: {len(queue)}/{self.window_queue_size}):")
            
            # 显示最近的几个包
            recent_packs = list(queue)[-max_packs_to_show:]
            for i, pack in enumerate(recent_packs):
                pack_idx = len(queue) - max_packs_to_show + i
                if pack_idx < 0:
                    pack_idx = i
                
                print(f"  包#{pack_idx+1}: 步数{pack['start_step']}-{pack['end_step']}, "
                      f"事务{pack['transaction_count']}次, 成功率{pack['success_rate']:.2f}, "
                      f"信誉变化{pack['total_reputation_change']:+.2f}")
            
            if len(queue) > max_packs_to_show:
                print(f"  ... (还有{len(queue) - max_packs_to_show}个更早的包)")
        
        # 显示当前正在积累的包
        if current_pack['total_count'] > 0:
            print(f"\n📝 当前积累包:")
            print(f"  步数{current_pack['start_step']}-{current_pack['end_step']}, "
                  f"事务{current_pack['total_count']}次, "
                  f"成功{current_pack['success_count']}次 "
                  f"({current_pack['success_count']/current_pack['total_count']:.2f}), "
                  f"信誉变化{sum(current_pack['reputation_changes']):+.2f}")
        else:
            print(f"\n📝 当前积累包: 空")
        
        # 显示摘要统计
        summary = self.get_na_window_summary(na_idx)
        print(f"\n统计摘要:")
        print(f"  总事务数: {summary['total_transactions']}")
        print(f"  总成功数: {summary['total_success_count']}")
        print(f"  整体成功率: {summary['overall_success_rate']:.2f}")
        print(f"  总信誉变化: {summary['total_reputation_change']:+.2f}")
        if summary['step_range']:
            print(f"  步数范围: {summary['step_range'][0]} - {summary['step_range'][1]}")
        print("=" * 80)

    def _update_hunger(self, selected_na_idx):
        """
        🍽️ 更新所有NA的饥饿度
        
        逻辑：
        1. 被选中的NA饥饿度重置为0
        2. 其他NA的饥饿度按时间增长
        3. 饥饿度范围控制在[0, 1]（0%-100%）
        
        Args:
            selected_na_idx: 当前被选中的NA索引
        """
        # 更新被选中NA的记录
        self.last_selected_time[selected_na_idx] = self.training_step

        # 计算所有NA的饥饿度
        for i in range(self.n_na):
            steps_since_selected = self.training_step - self.last_selected_time[i]

            if steps_since_selected <= 0:
                self.hunger[i] = 0.0
            else:
                normalized_steps = steps_since_selected / self.hunger_growth_scale
                self.hunger[i] = min(1.0, np.log(1 + normalized_steps) / np.log(self.hunger_growth_log_base))

        # 被选中的NA饥饿度重置为0
        self.hunger[selected_na_idx] = 0.0

def simulate_selected_nas_training(env, selected_nas, title, steps=100):
    """
    模拟选中的NA运行事务，只计算信誉变化，不计算奖励
    
    Args:
        env: 环境对象
        selected_nas: 选中的NA索引列表
        title: 标题
        steps: 模拟的步数
    
    Returns:
        dict: 包含每个NA的事务运行轨迹
    """
    print(f"\n🎯 {title} - NA事务运行模拟 ({steps}次事务):")
    print("=" * 80)
    
    # 保存环境的当前状态（训练后的状态）
    original_reputation = env.reputation.copy()
    original_success_rate = env.success_rate.copy()
    
    original_current_time = env.current_time
    
    original_total_tx = env.total_tx.copy()
    original_success_count = env.success_count.copy()
    
    # 为选中的NA创建事务运行轨迹记录
    training_results = {}
    
    for na_idx in selected_nas:
        # 🚀 重置环境到初始固定状态（使用初始参数）
        env.reputation = env.fixed_initial_reputation.copy()
        env.success_rate = env.fixed_initial_success_rate.copy()
        
        env.current_time = 0  # 重置到初始时间
        
        env.total_tx = env.fixed_initial_total_tx.copy()
        env.success_count = env.fixed_initial_success_count.copy()
        
        initial_rep = env.reputation[na_idx]
        initial_success_rate = env.success_rate[na_idx]
        initial_hunger = env.hunger[na_idx]
        initial_signature_delay = env.signature_delay[na_idx]
        
        reputation_history = [initial_rep]
        transaction_results = []  # 记录每次事务的详细结果
        
        print(f"\n📋 NA {na_idx} 事务运行轨迹:")
        print(f"   初始属性: 信誉={initial_rep:.2f}, 成功率={initial_success_rate:.3f}, 签名延迟={initial_signature_delay:.1f}ms, 饥饿度={initial_hunger:.3f}")
        
        # 模拟事务处理过程
        for step in range(steps):
            # 记录事务前状态
            old_reputation = env.reputation[na_idx]
            old_success_rate = env.success_rate[na_idx]
            old_hunger = env.hunger[na_idx]
            
            env.current_time += 1
            
            
            # 模拟当前事务的签名延迟等级（在固定基础上添加随机波动）
            base_delay_grade = env.signature_delay[na_idx]
            current_delay_grade = base_delay_grade + np.random.uniform(-0.05, 0.05)
            current_delay_grade = max(0.0, min(1.0, current_delay_grade))  # 限制在[0.0, 1.0]范围内
            
            # 判断事务是否成功
            if current_delay_grade > env.max_acceptable_delay_grade:
                success = False
                failure_reason = "签名延迟等级过高"
            else:
                # 🚀 直接使用原始成功率，不基于延迟性能调整
                success = np.random.random() < env.success_rate[na_idx]
                failure_reason = None
            
            # 计算信誉更新参数
            norm_rep = (old_reputation - 3300.0) / (10000.0 - 3300.0)
            if old_reputation < 6600:
                computed_factor = (0.4 * env.success_rate[na_idx] + 
                                 0.4 * norm_rep + 
                                 0.2 * env.hunger[na_idx])
            else:
                computed_factor = (0.4 * env.success_rate[na_idx] + 
                                 0.2 * norm_rep + 
                                 0.4 * env.hunger[na_idx])
            
            # 更新NA状态和信誉
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
            
            # 记录事务结果
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
        
        # 计算总体统计结果
        final_rep = reputation_history[-1]
        total_rep_change = final_rep - initial_rep
        successful_transactions = sum(1 for t in transaction_results if t['success'])
        transaction_success_rate = successful_transactions / steps
        avg_signature_delay = np.mean([t['signature_delay'] for t in transaction_results])
        final_success_rate = env.success_rate[na_idx]
        final_hunger = env.hunger[na_idx]
        
        # 统计失败原因
        delay_failures = sum(1 for t in transaction_results if t['failure_reason'] == '签名延迟过高')
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
        
        print(f"NA {na_idx} 事务运行总结:")
        print(f"   信誉变化: {initial_rep:.2f} → {final_rep:.2f} ({total_rep_change:+.2f})")
        print(f"   成功率变化: {initial_success_rate:.3f} → {final_success_rate:.3f}")
        print(f"   事务成功率: {successful_transactions}/{steps} ({transaction_success_rate:.1%})")
        print(f"   平均签名延迟: {avg_signature_delay:.1%}")
        print(f"   失败原因: 延迟过高{delay_failures}次, 随机失败{random_failures}次")
        print(f"   饥饿度变化: {initial_hunger:.3f} → {final_hunger:.3f}")
    
    # 恢复环境原始状态
    env.reputation = original_reputation
    env.success_rate = original_success_rate
    
    env.current_time = original_current_time
    
    env.total_tx = original_total_tx
    env.success_count = original_success_count
    
    return training_results

def create_policy_analysis_plots(agent, env):
    """
    创建策略分析图表：
    1. 策略分布图
    2. 策略熵曲线
    3. NA被选中的频率热图
    """
    print("\n" + "="*80)
    print("🎨 创建策略分析图表")
    print("="*80)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建3x1的子图，增加图的高度
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))  # 增加宽度和高度
    
    # 1. 策略分布图 - 显示最终策略的动作概率分布
    ax1 = axes[0]
    
    if len(agent.action_distribution_history) > 0:
        # 获取最后几次的策略分布进行平均
        recent_distributions = agent.action_distribution_history[-10:]  # 最后10次
        avg_distribution = np.mean(recent_distributions, axis=0)
        
        # 创建NA索引
        na_indices = np.arange(len(avg_distribution))
        
        # 根据初始信誉分组着色
        colors = []
        for i in range(env.n_na):
            if env.episode_start_reputation[i] < 6600:
                colors.append('lightcoral')  # 低信誉组 - 红色
            else:
                colors.append('lightblue')   # 高信誉组 - 蓝色
        
        bars = ax1.bar(na_indices, avg_distribution, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # 标记被选中概率最高的NA
        top_5_indices = np.argsort(avg_distribution)[-5:]
        for idx in top_5_indices:
            ax1.annotate(f'NA{idx}\n{avg_distribution[idx]:.3f}', 
                        xy=(idx, avg_distribution[idx]), 
                        xytext=(0, 15), textcoords='offset points',  # 增加标注距离
                        ha='center', fontsize=9, fontweight='bold')
        
        ax1.set_title('Policy Distribution (Final Strategy)\nAverage Action Probabilities in Recent Episodes', 
                     fontsize=12, fontweight='bold', pad=20)  # 增加标题间距
        ax1.set_xlabel('NA Index')
        ax1.set_ylabel('Selection Probability')
        
        # 设置横坐标显示所有NA索引
        ax1.set_xticks(na_indices)  # 设置所有刻度位置
        ax1.set_xticklabels([f'NA{i}' for i in na_indices], rotation=45, ha='right')  # 旋转标签避免重叠
        
        ax1.grid(True, alpha=0.3)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightcoral', label='Low Reputation Group (<6600)'),
                          Patch(facecolor='lightblue', label='High Reputation Group (≥6600)')]
        ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))  # 移动到左上角
        
        print(f"✅ 策略分布图完成 - 基于最后10次episode的平均分布")
        print(f"   Top-5 NA选择概率: {[f'NA{i}({avg_distribution[i]:.3f})' for i in top_5_indices]}")
    else:
        ax1.text(0.5, 0.5, 'No Policy Distribution Data Available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Policy Distribution - No Data')
    
    # 2. 策略熵曲线
    ax2 = axes[1]
    
    if len(agent.policy_entropy_history) > 0:
        episodes = np.arange(len(agent.policy_entropy_history))
        entropies = agent.policy_entropy_history
        
        # 绘制熵曲线
        ax2.plot(episodes, entropies, 'b-', linewidth=2, alpha=0.8, label='Policy Entropy')
        
        # 添加趋势线
        if len(entropies) > 10:
            z = np.polyfit(episodes, entropies, 1)
            p = np.poly1d(z)
            ax2.plot(episodes, p(episodes), 'r--', alpha=0.7, label=f'Trend (slope={z[0]:.4f})')
        
        # 标记关键点
        if len(entropies) > 0:
            initial_entropy = entropies[0]
            final_entropy = entropies[-1]
            max_entropy = np.max(entropies)
            min_entropy = np.min(entropies)
            
            ax2.axhline(y=np.log(env.n_na), color='g', linestyle=':', alpha=0.7, 
                       label=f'Max Possible Entropy (log({env.n_na})={np.log(env.n_na):.2f})')
            
            # 注释关键信息
            ax2.text(0.02, 0.65, f'Initial: {initial_entropy:.3f}\nFinal: {final_entropy:.3f}\n'
                                  f'Max: {max_entropy:.3f}\nMin: {min_entropy:.3f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax2.set_title('Policy Entropy Over Training\n(Lower entropy = more deterministic policy)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Step (every 10 episodes)')
        ax2.set_ylabel('Entropy (bits)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02))  # 移动到右下角
        
        # 分析熵的变化趋势
        if len(entropies) > 1:
            entropy_change = entropies[-1] - entropies[0]
            if entropy_change < -0.1:
                trend_text = "✅ 策略趋于确定 (熵下降)"
            elif entropy_change > 0.1:
                trend_text = "⚠️ 策略变得随机 (熵上升)"
            else:
                trend_text = "➡️ 策略相对稳定"
        else:
            trend_text = "➡️ 数据不足"
            
        print(f"✅ 策略熵曲线完成 - {trend_text}")
        print(f"   熵变化: {entropies[0]:.3f} → {entropies[-1]:.3f} (Δ{entropy_change:.3f})")
    else:
        ax2.text(0.5, 0.5, 'No Policy Entropy Data Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Policy Entropy - No Data')
    
    # 3. NA被选中的频率热图
    ax3 = axes[2]
    
    if np.sum(agent.na_selection_frequency) > 0:
        # 归一化选择频率为百分比
        selection_freq_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        
        # 创建热图数据 - 重新整理为矩阵形式便于显示
        n_cols = min(10, env.n_na)  # 每行最多显示10个NA
        n_rows = int(np.ceil(env.n_na / n_cols))
        
        # 填充数据到矩阵
        heatmap_data = np.zeros((n_rows, n_cols))
        for i in range(env.n_na):
            row = i // n_cols
            col = i % n_cols
            heatmap_data[row, col] = selection_freq_percent[i]
        
        # 创建热图
        im = ax3.imshow(heatmap_data, cmap='Reds', aspect='auto', interpolation='nearest')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
        cbar.set_label('Selection Frequency (%)', rotation=270, labelpad=20)
        
        # 添加数值标注
        for i in range(n_rows):
            for j in range(n_cols):
                na_idx = i * n_cols + j
                if na_idx < env.n_na:
                    text_color = 'white' if heatmap_data[i, j] > np.max(heatmap_data) * 0.5 else 'black'
                    ax3.text(j, i, f'NA{na_idx}\n{heatmap_data[i, j]:.1f}%',
                            ha='center', va='center', color=text_color, fontsize=8)
        
        # 设置坐标轴
        ax3.set_xticks(range(n_cols))
        ax3.set_yticks(range(n_rows))
        ax3.set_xticklabels([f'Col{i}' for i in range(n_cols)])
        ax3.set_yticklabels([f'Row{i}' for i in range(n_rows)])
        
        ax3.set_title('NA Selection Frequency Heatmap\n(Percentage of times each NA was selected)', 
                     fontsize=12, fontweight='bold')
        
        # 分析选择频率
        most_selected = np.argmax(selection_freq_percent)
        least_selected = np.argmin(selection_freq_percent[selection_freq_percent > 0])
        
        print(f"✅ NA选择频率热图完成")
        print(f"   最常选择: NA{most_selected} ({selection_freq_percent[most_selected]:.1f}%)")
        print(f"   最少选择: NA{least_selected} ({selection_freq_percent[least_selected]:.1f}%)")
        
        # 检查是否过拟合某些NA
        top_3_percent = np.sum(np.sort(selection_freq_percent)[-3:])
        if top_3_percent > 70:
            print(f"   ⚠️ 警告: Top-3 NA占选择的{top_3_percent:.1f}%，可能存在过拟合")
        else:
            print(f"   ✅ 选择分布合理: Top-3 NA占{top_3_percent:.1f}%")
    else:
        ax3.text(0.5, 0.5, 'No Selection Frequency Data Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('NA Selection Frequency - No Data')
    
    plt.tight_layout(pad=4.0)  # 增加子图间距，避免文字重叠
    
    # 保存图表
    filename = f'/mnt/data/wy2024/src/policy_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n策略分析图表已保存: {filename}")
    print("="*80)
    
    # 输出详细的策略分析报告
    print("\n📋 策略分析报告:")
    
    if len(agent.policy_entropy_history) > 0:
        print(f"1. 策略熵分析:")
        print(f"   - 初始熵: {agent.policy_entropy_history[0]:.3f}")
        print(f"   - 最终熵: {agent.policy_entropy_history[-1]:.3f}")
        print(f"   - 最大可能熵: {np.log(env.n_na):.3f} (完全随机)")
        print(f"   - 确定性程度: {(1 - agent.policy_entropy_history[-1]/np.log(env.n_na))*100:.1f}%")
    
    if np.sum(agent.na_selection_frequency) > 0:
        print(f"\n2. NA选择分析:")
        selection_percent = agent.na_selection_frequency / np.sum(agent.na_selection_frequency) * 100
        selected_nas = np.where(selection_percent > 0)[0]
        print(f"   - 被选择的NA数量: {len(selected_nas)}/{env.n_na}")
        print(f"   - 选择多样性: {len(selected_nas)/env.n_na*100:.1f}%")
        
        # 按信誉分组分析
        low_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                   if env.episode_start_reputation[i] < 6600])
        high_rep_selections = np.sum([selection_percent[i] for i in range(env.n_na) 
                                    if env.episode_start_reputation[i] >= 6600])
        print(f"   - 低信誉组选择率: {low_rep_selections:.1f}%")
        print(f"   - 高信誉组选择率: {high_rep_selections:.1f}%")
    
    if len(agent.action_distribution_history) > 0:
        print(f"\n3. 策略分布分析:")
        recent_dist = np.mean(agent.action_distribution_history[-10:], axis=0)
        top_5_nas = np.argsort(recent_dist)[-5:]
        print(f"   - Top-5 偏好NA: {[f'NA{i}({recent_dist[i]:.2f})' for i in top_5_nas]}")
        concentration = np.sum(np.sort(recent_dist)[-5:])
        print(f"   - Top-5集中度: {concentration:.1%}")
        
        if concentration > 0.8:
            print(f"   - ⚠️ 策略高度集中，可能过拟合")
        elif concentration < 0.3:
            print(f"   - ⚠️ 策略过于分散，可能学习不足")
        else:
            print(f"   - ✅ 策略集中度适中")

def train_dqn(n_na, n_episodes, steps_per_episode, lr, 
              update_frequency, use_fixed_nas, enable_random_fluctuation,
              generate_episode_gifs=False,
              freeze_hunger=False, hunger_constant=0.0, hunger_weight=1.0, enable_hunger_update=True):
    
    # 🔧 参数检查和逻辑验证
    print("🔧 训练参数检查:")
    print(f"  NA数量: {n_na}")
    print(f"  训练Episodes: {n_episodes}")  
    print(f"  每Episode步数: {steps_per_episode}")
    print(f"  学习率: {lr}")
    print(f"  更新频率: 每{update_frequency}步")
    print(f"  使用固定NA集合: {use_fixed_nas}")
    print(f"  随机波动模式: {enable_random_fluctuation}")
    
    # 验证参数合理性
    total_steps = n_episodes * steps_per_episode
    print(f"  总训练步数: {total_steps}")
    
    if n_na < 5:
        print("⚠️  警告：NA数量过少，可能影响学习效果")
    if n_na > 30:
        print("⚠️  警告：NA数量过多，可能需要增加训练步数")
    if lr > 0.01:
        print("⚠️  警告：学习率较高，可能导致训练不稳定")
    if total_steps < 1000:
        print("⚠️  警告：总训练步数较少，网络可能学习不充分")
    
    # 🔧 创建使用固定NA的环境
    env = Environment(
        n_na,
        use_fixed_nas=use_fixed_nas,
        enable_random_fluctuation=enable_random_fluctuation,
        freeze_hunger=freeze_hunger,
        hunger_constant=hunger_constant,
        hunger_weight=hunger_weight,
        enable_hunger_update=enable_hunger_update
    )
    agent = DQNAgent(n_features=4, n_na=n_na, lr=lr, gamma=0.9,  # 活跃度删除后特征数=4，降低gamma增加Q值差异
                    epsilon_start=1.0, epsilon_end=0.01, decay_steps=15000,
                    memory_size=50000, batch_size=256, target_update=20,  # 修改目标网络更新频率为10步
                    min_memory_size=2000, update_frequency=update_frequency, 
                    total_episodes=n_episodes)  # 补全所有必需参数
    history = []
    # 记录最后一次episode中每步的信誉变化
    reputation_last = []
    
    # 记录第一个和最后一个episode的详细数据
    first_episode_data = []  # 记录第一个episode的信誉变化和选择
    last_episode_data = []   # 记录最后一个episode的信誉变化和选择
    
    # 添加奖励分析变量
    step_rewards = []  # 记录平滑后的step奖励
    step_reward_steps = []  # 记录对应训练步数
    reward_capture_interval = 3  # 每3步聚合一次奖励
    step_reward_buffer = []  # 累积原始奖励用于平滑
    success_rates_per_episode = []  # 记录每个episode的成功率
    
    # 添加训练统计
    total_training_steps = 0
    total_updates = 0
    
    # 预热阶段：先收集经验，不进行训练
    print("🔥 预热阶段：收集初始经验...")
    warmup_steps = 0
    while len(agent.memory) < agent.min_memory_size:
        state = env.reset()  # 重置环境状态，但保持固定NA特征
        for _ in range(10):  # 每个环境状态下收集10步经验
            action = agent.select_action(state, env.episode_start_reputation)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, next_state, done)
            state = next_state
            warmup_steps += 1
            
            # 达到最小内存要求就停止预热
            if len(agent.memory) >= agent.min_memory_size:
                break
    
    print(f"✅ 预热完成！已收集 {len(agent.memory)} 条经验，共执行 {warmup_steps} 步")
    print(f"🎯 训练策略：使用固定NA集合，每个step选择单个NA执行动作")
    print(f"开始正式训练，更新频率: 每{update_frequency}步训练一次...")
    
    # 🔧 打印第一个episode的固定NA特征，确认使用固定集合
    print(f"\n📋 固定NA集合特征 (训练全程保持不变):")
    print("编号\t初始信誉\t成功率\t签名延迟(ms)\t饥饿度\t延迟等级\t预期分组")
    for i in range(n_na):
        rep = env.fixed_initial_reputation[i] if env.use_fixed_nas else env.reputation[i]
        sr = env.fixed_initial_success_rate[i] if env.use_fixed_nas else env.success_rate[i]
        delay = env.fixed_initial_signature_delay[i] if env.use_fixed_nas else env.signature_delay[i]
        hunger = env.hunger[i] if env.freeze_hunger else (env.fixed_initial_hunger[i] if env.use_fixed_nas else env.hunger[i])
        delay_perf = env.calculate_delay_performance(delay)
        group = "低信誉组" if rep < 6600 else "高信誉组"
        # 显示实际的初始饥饿度
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
        enable_random_fluctuation=False,  # 关闭随机波动
        freeze_hunger=True,
        hunger_constant=0.0,
        hunger_weight=0.0,
        enable_hunger_update=False
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