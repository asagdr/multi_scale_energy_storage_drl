import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import UpperLayerConfig
from config.model_config import ModelConfig
from .transformer_encoder import TransformerEncoder
from .balance_analyzer import BalanceAnalyzer
from .constraint_generator import ConstraintGenerator

class MultiObjectiveReplayBuffer:
    """多目标经验回放缓冲区"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward_vector, next_state, done, weight_vector):
        """添加经验"""
        experience = (state, action, reward_vector, next_state, done, weight_vector)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次数据"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.stack([e[0] for e in experiences])
        actions = torch.stack([e[1] for e in experiences])
        reward_vectors = torch.stack([e[2] for e in experiences])
        next_states = torch.stack([e[3] for e in experiences])
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.bool)
        weight_vectors = torch.stack([e[5] for e in experiences])
        
        return states, actions, reward_vectors, next_states, dones, weight_vectors
    
    def __len__(self):
        return len(self.buffer)

class MultiObjectiveActor(nn.Module):
    """多目标Actor网络"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 256,
                 num_objectives: int = 4):
        super(MultiObjectiveActor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_objectives = num_objectives
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 权重编码器（用于处理目标权重）
        self.weight_encoder = nn.Sequential(
            nn.Linear(num_objectives, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 动作头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Tanh()  # 输出[-1, 1]
        )
        
        # 多目标动作调制器
        self.objective_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Tanh()
            ) for _ in range(num_objectives)
        ])
        
        # 权重注意力机制
        self.weight_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            batch_first=True
        )
    
    def forward(self, state: torch.Tensor, weight_vector: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            weight_vector: 目标权重向量 [batch_size, num_objectives]
        """
        # 状态编码
        state_features = self.state_encoder(state)
        
        # 权重编码
        weight_features = self.weight_encoder(weight_vector)
        
        # 特征融合
        fused_features = torch.cat([state_features, weight_features], dim=-1)
        fusion_output = self.fusion_network(fused_features)
        
        # 基础动作
        base_action = self.action_head(fusion_output)
        
        # 多目标调制
        objective_actions = []
        for i, modulator in enumerate(self.objective_modulators):
            obj_action = modulator(fusion_output)
            objective_actions.append(obj_action)
        
        # 权重加权组合
        weighted_action = torch.zeros_like(base_action)
        for i, obj_action in enumerate(objective_actions):
            weighted_action += weight_vector[:, i:i+1] * obj_action
        
        # 最终动作
        final_action = 0.7 * base_action + 0.3 * weighted_action
        
        return final_action

class MultiObjectiveCritic(nn.Module):
    """多目标Critic网络"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 256,
                 num_objectives: int = 4):
        super(MultiObjectiveCritic, self).__init__()
        
        self.num_objectives = num_objectives
        
        # 状态-动作编码器
        self.state_action_encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 权重编码器
        self.weight_encoder = nn.Sequential(
            nn.Linear(num_objectives, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 价值头 - 为每个目标输出一个价值
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for _ in range(num_objectives)
        ])
        
        # 综合价值头
        self.combined_value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, 
                state: torch.Tensor, 
                action: torch.Tensor, 
                weight_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            action: 动作 [batch_size, action_dim]
            weight_vector: 目标权重向量 [batch_size, num_objectives]
        """
        # 状态-动作编码
        state_action = torch.cat([state, action], dim=-1)
        sa_features = self.state_action_encoder(state_action)
        
        # 权重编码
        weight_features = self.weight_encoder(weight_vector)
        
        # 特征融合
        combined_features = torch.cat([sa_features, weight_features], dim=-1)
        
        # 各目标价值
        objective_values = []
        for i, value_head in enumerate(self.value_heads):
            obj_value = value_head(combined_features)
            objective_values.append(obj_value)
        
        objective_values = torch.cat(objective_values, dim=-1)  # [batch_size, num_objectives]
        
        # 加权综合价值
        weighted_value = torch.sum(objective_values * weight_vector, dim=-1, keepdim=True)
        
        # 独立综合价值
        combined_value = self.combined_value_head(combined_features)
        
        return {
            'objective_values': objective_values,
            'weighted_value': weighted_value,
            'combined_value': combined_value
        }

class MultiObjectiveAgent(nn.Module):
    """
    多目标DRL智能体
    实现上层5分钟级决策：SOC均衡、温度均衡、寿命成本最小化、约束满足
    """
    
    def __init__(self,
                 config: UpperLayerConfig,
                 model_config: ModelConfig,
                 agent_id: str = "MultiObjectiveAgent_001"):
        """
        初始化多目标智能体
        
        Args:
            config: 上层配置
            model_config: 模型配置
            agent_id: 智能体ID
        """
        super(MultiObjectiveAgent, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.agent_id = agent_id
        
        # === 模型参数 ===
        self.state_dim = model_config.upper_state_dim
        self.action_dim = model_config.upper_action_dim
        self.hidden_dim = config.hidden_dim
        self.num_objectives = 4  # SOC均衡、温度均衡、寿命成本、约束满足
        
        # === 网络组件 ===
        # Transformer编码器
        self.transformer_encoder = TransformerEncoder(config, model_config, f"Encoder_{agent_id}")
        
        # 均衡分析器
        self.balance_analyzer = BalanceAnalyzer(config, model_config, f"BalanceAnalyzer_{agent_id}")
        
        # 约束生成器
        self.constraint_generator = ConstraintGenerator(config, model_config, f"ConstraintGen_{agent_id}")
        
        # Actor-Critic网络
        self.actor = MultiObjectiveActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_objectives=self.num_objectives
        )
        
        self.critic = MultiObjectiveCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_objectives=self.num_objectives
        )
        
        self.target_actor = MultiObjectiveActor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_objectives=self.num_objectives
        )
        
        self.target_critic = MultiObjectiveCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_objectives=self.num_objectives
        )
        
        # 复制网络权重
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # === 优化器 ===
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate * 2)
        
        # === 经验回放 ===
        self.replay_buffer = MultiObjectiveReplayBuffer(100000)
        
        # === 训练参数 ===
        self.gamma = 0.99
        self.tau = 0.001
        self.noise_std = 0.1
        self.batch_size = config.batch_size
        
        # === 目标权重 ===
        self.current_weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
        
        # === 训练统计 ===
        self.training_step = 0
        self.episode_count = 0
        self.total_rewards = []
        self.objective_rewards = [[] for _ in range(self.num_objectives)]
        
        print(f"✅ 多目标智能体初始化完成: {agent_id}")
        print(f"   状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        print(f"   目标数量: {self.num_objectives}")
    
    def select_action(self, 
                     state: torch.Tensor, 
                     weight_vector: Optional[torch.Tensor] = None,
                     add_noise: bool = True) -> torch.Tensor:
        """
        选择动作
        
        Args:
            state: 当前状态
            weight_vector: 目标权重向量
            add_noise: 是否添加探索噪声
        """
        if weight_vector is None:
            weight_vector = self.current_weights.unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state, weight_vector)
            
            if add_noise:
                noise = torch.randn_like(action) * self.noise_std
                action = torch.clamp(action + noise, -1.0, 1.0)
        
        return action
    
    def update_networks(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样批次数据
        states, actions, reward_vectors, next_states, dones, weight_vectors = \
            self.replay_buffer.sample(self.batch_size)
        
        # === 更新Critic ===
        self.critic.train()
        self.critic_optimizer.zero_grad()
        
        # 当前Q值
        current_q = self.critic(states, actions, weight_vectors)
        
        # 目标Q值
        with torch.no_grad():
            next_actions = self.target_actor(next_states, weight_vectors)
            target_q = self.target_critic(next_states, next_actions, weight_vectors)
            
            # 多目标TD目标
            target_objective_values = reward_vectors + self.gamma * target_q['objective_values'] * (~dones).unsqueeze(-1)
            target_weighted_value = reward_vectors.sum(dim=-1, keepdim=True) + \
                                  self.gamma * target_q['weighted_value'] * (~dones).unsqueeze(-1)
        
        # Critic损失
        objective_loss = F.mse_loss(current_q['objective_values'], target_objective_values)
        weighted_loss = F.mse_loss(current_q['weighted_value'], target_weighted_value)
        combined_loss = F.mse_loss(current_q['combined_value'], target_weighted_value)
        
        critic_loss = objective_loss + weighted_loss + combined_loss
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # === 更新Actor ===
        self.actor.train()
        self.actor_optimizer.zero_grad()
        
        # Actor损失
        new_actions = self.actor(states, weight_vectors)
        actor_q = self.critic(states, new_actions, weight_vectors)
        
        # 多目标策略梯度
        actor_loss = -actor_q['weighted_value'].mean()
        
        # 添加动作正则化
        action_reg = 0.01 * (new_actions ** 2).mean()
        actor_loss += action_reg
        
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # === 软更新目标网络 ===
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'objective_loss': objective_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'q_value': current_q['weighted_value'].mean().item()
        }
    
    def _soft_update(self, target_net, source_net):
        """软更新目标网络"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def add_experience(self, 
                      state: torch.Tensor, 
                      action: torch.Tensor, 
                      reward_vector: torch.Tensor, 
                      next_state: torch.Tensor, 
                      done: bool,
                      weight_vector: torch.Tensor):
        """添加经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward_vector, next_state, done, weight_vector)
    
    def calculate_multi_objective_rewards(self, 
                                        system_state: Dict[str, Any],
                                        action: torch.Tensor) -> torch.Tensor:
        """
        计算多目标奖励向量
        
        Args:
            system_state: 系统状态
            action: 执行的动作
            
        Returns:
            奖励向量 [soc_balance_reward, temp_balance_reward, lifetime_reward, constraint_reward]
        """
        rewards = torch.zeros(self.num_objectives)
        
        # === 1. SOC均衡奖励 ===
        soc_std = system_state.get('soc_std', 5.0)
        soc_reward = 1.0 - min(1.0, soc_std / 10.0)  # σ_SOC越小奖励越高
        rewards[0] = soc_reward
        
        # === 2. 温度均衡奖励 ===
        temp_std = system_state.get('temp_std', 5.0)
        temp_reward = 1.0 - min(1.0, temp_std / 15.0)
        rewards[1] = temp_reward
        
        # === 3. 寿命成本奖励 ===
        degradation_cost = system_state.get('degradation_cost_rate', 0.1)
        lifetime_reward = 1.0 - min(1.0, degradation_cost / 1.0)
        rewards[2] = lifetime_reward
        
        # === 4. 约束满足奖励 ===
        violation_count = system_state.get('constraint_violations', 0)
        constraint_reward = 1.0 if violation_count == 0 else max(0.0, 1.0 - violation_count * 0.2)
        rewards[3] = constraint_reward
        
        return rewards
    
    def adapt_objective_weights(self, 
                              system_state: Dict[str, Any],
                              performance_history: List[Dict]) -> torch.Tensor:
        """
        自适应调整目标权重
        
        Args:
            system_state: 当前系统状态
            performance_history: 性能历史
            
        Returns:
            调整后的权重向量
        """
        # 基于当前状态的紧迫性调整权重
        urgencies = torch.zeros(self.num_objectives)
        
        # SOC均衡紧迫性
        soc_std = system_state.get('soc_std', 0.0)
        urgencies[0] = min(1.0, soc_std / 5.0)
        
        # 温度均衡紧迫性
        max_temp = system_state.get('max_temperature', 25.0)
        urgencies[1] = min(1.0, max(0, max_temp - 40.0) / 20.0)
        
        # 寿命优化紧迫性
        min_soh = system_state.get('min_soh', 100.0)
        urgencies[2] = min(1.0, (100 - min_soh) / 20.0)
        
        # 约束违反紧迫性
        violation_count = system_state.get('constraint_violations', 0)
        urgencies[3] = min(1.0, violation_count / 5.0)
        
        # 基于历史性能调整
        if len(performance_history) > 10:
            recent_performance = performance_history[-10:]
            
            # 计算各目标的改善趋势
            for i in range(self.num_objectives):
                objective_values = [perf.get(f'objective_{i}_reward', 0.5) for perf in recent_performance]
                if len(objective_values) > 5:
                    trend = np.polyfit(range(len(objective_values)), objective_values, 1)[0]
                    if trend < -0.01:  # 性能下降
                        urgencies[i] += 0.2
        
        # 归一化权重
        total_urgency = urgencies.sum()
        if total_urgency > 0:
            new_weights = urgencies / total_urgency
        else:
            new_weights = torch.tensor([0.25, 0.25, 0.25, 0.25])
        
        # 平滑过渡
        self.current_weights = 0.9 * self.current_weights + 0.1 * new_weights
        
        return self.current_weights
    
    def generate_high_level_decision(self, 
                                   state: torch.Tensor,
                                   system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成高层决策
        
        Args:
            state: 状态向量
            system_state: 系统状态字典
            
        Returns:
            高层决策字典
        """
        # === 1. 均衡分析 ===
        balance_analysis = self.balance_analyzer(state)
        
        # === 2. 约束生成 ===
        constraint_matrix = self.constraint_generator(state)
        
        # === 3. 动作选择 ===
        action = self.select_action(state, self.current_weights.unsqueeze(0), add_noise=False)
        
        # === 4. 构建决策输出 ===
        decision = {
            # 控制指令
            'power_command_ratio': action[0, 0].item(),        # 功率指令比例 [-1,1]
            'soc_balance_weight': action[0, 1].item(),         # SOC均衡权重 [-1,1]
            'temp_balance_weight': action[0, 2].item(),        # 温度均衡权重 [-1,1]
            'lifetime_weight': action[0, 3].item(),           # 寿命优化权重 [-1,1]
            
            # 约束矩阵
            'constraint_matrix': constraint_matrix.to_matrix(),
            'constraint_level': constraint_matrix.constraint_level.value,
            
            # 均衡目标
            'target_soc_std': balance_analysis['target_soc_std'].item(),
            'target_temp_std': balance_analysis['target_temp_std'].item(),
            'balance_priorities': balance_analysis['balance_priorities'].cpu().numpy(),
            
            # 权重向量
            'objective_weights': self.current_weights.cpu().numpy(),
            
            # 分析结果
            'balance_score': balance_analysis['overall_balance_score'].item(),
            'soc_urgency': balance_analysis['soc_urgency'].item(),
            'thermal_urgency': balance_analysis['thermal_urgency'].item(),
            'degradation_urgency': balance_analysis['degradation_urgency'].item(),
            
            # 元信息
            'decision_timestamp': self.training_step,
            'agent_id': self.agent_id
        }
        
        return decision
    
    def train_episode(self, environment, max_steps: int = 100) -> Dict[str, float]:
        """训练一个回合"""
        state = environment.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        
        episode_reward = 0.0
        episode_objective_rewards = torch.zeros(self.num_objectives)
        
        for step in range(max_steps):
            # 动作选择
            action = self.select_action(state, self.current_weights.unsqueeze(0))
            
            # 环境交互
            next_state, reward, done, info = environment.step(action.squeeze(0).numpy())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            # 多目标奖励计算
            reward_vector = self.calculate_multi_objective_rewards(info, action)
            
            # 存储经验
            self.add_experience(state, action, reward_vector, next_state, done, self.current_weights)
            
            # 更新网络
            if len(self.replay_buffer) > self.batch_size:
                losses = self.update_networks()
            
            # 累积奖励
            episode_reward += reward
            episode_objective_rewards += reward_vector
            
            state = next_state
            
            if done:
                break
        
        # 记录统计
        self.episode_count += 1
        self.total_rewards.append(episode_reward)
        for i, obj_reward in enumerate(episode_objective_rewards):
            self.objective_rewards[i].append(obj_reward.item())
        
        # 计算回合统计
        episode_stats = {
            'episode_reward': episode_reward,
            'episode_length': step + 1,
            'soc_balance_reward': episode_objective_rewards[0].item(),
            'temp_balance_reward': episode_objective_rewards[1].item(),
            'lifetime_reward': episode_objective_rewards[2].item(),
            'constraint_reward': episode_objective_rewards[3].item(),
            'avg_q_value': losses.get('q_value', 0.0) if 'losses' in locals() else 0.0
        }
        
        return episode_stats
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        if not self.total_rewards:
            return {'error': 'No training episodes completed'}
        
        recent_rewards = self.total_rewards[-100:] if len(self.total_rewards) >= 100 else self.total_rewards
        
        stats = {
            'agent_id': self.agent_id,
            'total_episodes': self.episode_count,
            'training_steps': self.training_step,
            
            'performance': {
                'avg_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'max_reward': max(self.total_rewards),
                'min_reward': min(self.total_rewards)
            },
            
            'objectives': {
                f'objective_{i}_avg': np.mean(self.objective_rewards[i][-100:]) 
                if len(self.objective_rewards[i]) >= 100 else np.mean(self.objective_rewards[i])
                for i in range(self.num_objectives) if self.objective_rewards[i]
            },
            
            'current_weights': self.current_weights.cpu().numpy().tolist(),
            'replay_buffer_size': len(self.replay_buffer),
            
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'actor_parameters': sum(p.numel() for p in self.actor.parameters()),
                'critic_parameters': sum(p.numel() for p in self.critic.parameters())
            }
        }
        
        return stats
    
    def save_checkpoint(self, filepath: str) -> bool:
        """保存检查点"""
        try:
            checkpoint = {
                'agent_id': self.agent_id,
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'target_actor_state_dict': self.target_actor.state_dict(),
                'target_critic_state_dict': self.target_critic.state_dict(),
                
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                
                'current_weights': self.current_weights,
                'config': self.config,
                'model_config': self.model_config,
                
                'statistics': self.get_agent_statistics()
            }
            
            torch.save(checkpoint, filepath)
            print(f"✅ 智能体检查点已保存: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 保存检查点失败: {str(e)}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            self.current_weights = checkpoint['current_weights']
            self.training_step = checkpoint['training_step']
            self.episode_count = checkpoint['episode_count']
            
            print(f"✅ 智能体检查点已加载: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"MultiObjectiveAgent({self.agent_id}): "
                f"episodes={self.episode_count}, steps={self.training_step}, "
                f"objectives={self.num_objectives}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"MultiObjectiveAgent(agent_id='{self.agent_id}', "
                f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
                f"num_objectives={self.num_objectives})")
