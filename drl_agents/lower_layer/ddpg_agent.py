import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import random
import copy
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import LowerLayerConfig
from config.model_config import ModelConfig

class OrnsteinUhlenbeckNoise:
    """OU噪声生成器（用于连续动作探索）"""
    
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.size) * self.mu
    
    def sample(self):
        """采样噪声"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """采样批次数据"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.stack([e[0] for e in experiences])
        actions = torch.stack([e[1] for e in experiences])
        rewards = torch.tensor([e[2] for e in experiences], dtype=torch.float32)
        next_states = torch.stack([e[3] for e in experiences])
        dones = torch.tensor([e[4] for e in experiences], dtype=torch.bool)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor网络"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [400, 300],
                 max_action: float = 1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        # 构建网络层
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        action = self.network(state)
        return self.max_action * action

class Critic(nn.Module):
    """Critic网络"""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [400, 300]):
        super(Critic, self).__init__()
        
        # Q1网络
        q1_layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            q1_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        q1_layers.append(nn.Linear(input_dim, 1))
        self.q1_network = nn.Sequential(*q1_layers)
        
        # Q2网络（Twin Critic for TD3）
        q2_layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            q2_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        q2_layers.append(nn.Linear(input_dim, 1))
        self.q2_network = nn.Sequential(*q2_layers)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回双Q值"""
        state_action = torch.cat([state, action], dim=-1)
        
        q1 = self.q1_network(state_action)
        q2 = self.q2_network(state_action)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """仅返回Q1值"""
        state_action = torch.cat([state, action], dim=-1)
        return self.q1_network(state_action)

class DDPGAgent(nn.Module):
    """
    DDPG智能体（结合TD3改进）
    实现下层10ms级功率跟踪控制
    """
    
    def __init__(self,
                 config: LowerLayerConfig,
                 model_config: ModelConfig,
                 agent_id: str = "DDPGAgent_001"):
        """
        初始化DDPG智能体
        
        Args:
            config: 下层配置
            model_config: 模型配置
            agent_id: 智能体ID
        """
        super(DDPGAgent, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.agent_id = agent_id
        
        # === 模型参数 ===
        self.state_dim = model_config.lower_state_dim
        self.action_dim = model_config.lower_action_dim
        self.max_action = 1.0
        
        # === 网络初始化 ===
        # Actor网络
        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.actor_hidden,
            max_action=self.max_action
        )
        
        self.actor_target = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.actor_hidden,
            max_action=self.max_action
        )
        
        # Critic网络
        self.critic = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.critic_hidden
        )
        
        self.critic_target = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=config.critic_hidden
        )
        
        # 复制网络权重到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # === 优化器 ===
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # === 训练参数 ===
        self.gamma = config.gamma
        self.tau = config.tau
        self.policy_noise = config.policy_noise if hasattr(config, 'policy_noise') else 0.2
        self.noise_clip = config.noise_clip if hasattr(config, 'noise_clip') else 0.5
        self.policy_freq = config.policy_freq if hasattr(config, 'policy_freq') else 2
        
        # === 经验回放 ===
        buffer_size = config.buffer_size if hasattr(config, 'buffer_size') else 1000000
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 256
        
        # === 噪声生成器 ===
        self.noise = OrnsteinUhlenbeckNoise(
            size=self.action_dim,
            sigma=config.action_noise
        )
        
        # === 训练统计 ===
        self.total_it = 0
        self.training_step = 0
        self.episode_count = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # === 控制历史（用于分析） ===
        self.control_history: List[Dict] = []
        
        print(f"✅ DDPG智能体初始化完成: {agent_id}")
        print(f"   状态维度: {self.state_dim}, 动作维度: {self.action_dim}")
        print(f"   缓冲区大小: {buffer_size}, 批次大小: {self.batch_size}")
    
    def select_action(self, 
                     state: torch.Tensor, 
                     add_noise: bool = True,
                     noise_scale: float = 1.0) -> torch.Tensor:
        """
        选择动作
        
        Args:
            state: 当前状态
            add_noise: 是否添加探索噪声
            noise_scale: 噪声缩放因子
        """
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
            
            if add_noise:
                noise = torch.tensor(
                    self.noise.sample() * noise_scale,
                    dtype=torch.float32,
                    device=action.device
                )
                action = action + noise
                action = torch.clamp(action, -self.max_action, self.max_action)
        
        return action
    
    def add_experience(self, 
                      state: torch.Tensor, 
                      action: torch.Tensor, 
                      reward: float, 
                      next_state: torch.Tensor, 
                      done: bool):
        """添加经验到回放缓冲区"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> Dict[str, float]:
        """训练网络"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        self.total_it += 1
        
        # 采样批次数据
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # === 训练Critic ===
        with torch.no_grad():
            # 目标策略平滑化
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # 计算目标Q值（取双Q值的最小值）
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(-1) + (~dones).unsqueeze(-1) * self.gamma * target_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss_value = 0.0
        
        # === 延迟训练Actor ===
        if self.total_it % self.policy_freq == 0:
            # Actor损失
            actor_loss = -self.critic.q1(states, self.actor(states)).mean()
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            actor_loss_value = actor_loss.item()
            
            # 软更新目标网络
            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.actor_target, self.actor)
        
        # 记录损失
        self.actor_losses.append(actor_loss_value)
        self.critic_losses.append(critic_loss.item())
        self.training_step += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss_value,
            'q_value': current_q1.mean().item(),
            'target_q': target_q.mean().item()
        }
    
    def _soft_update(self, target_net, source_net):
        """软更新目标网络"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + source_param.data * self.tau
            )
    
    def track_power_command(self, 
                          current_state: torch.Tensor,
                          power_command: float,
                          constraints: Dict[str, float]) -> Dict[str, Any]:
        """
        功率跟踪控制
        
        Args:
            current_state: 当前状态
            power_command: 功率指令 (W)
            constraints: 约束字典
            
        Returns:
            控制结果
        """
        # 选择控制动作
        action = self.select_action(current_state, add_noise=False)
        
        # 解析动作
        power_control = action[0].item()      # 功率控制信号 [-1, 1]
        response_factor = action[1].item()    # 响应速度因子 [-1, 1]
        compensation = action[2].item()       # 误差补偿 [-1, 1]
        
        # 转换为实际控制信号
        max_power_change = constraints.get('max_power_change_rate', 1000.0)  # W/s
        actual_power_change = power_control * max_power_change * 0.01  # 10ms时间步
        
        # 响应速度调整
        response_speed = (response_factor + 1.0) / 2.0  # [0, 1]
        adjusted_power_change = actual_power_change * response_speed
        
        # 误差补偿
        power_error = constraints.get('power_error', 0.0)
        compensation_factor = (compensation + 1.0) / 2.0  # [0, 1]
        error_compensation = power_error * compensation_factor * 0.1
        
        # 最终控制输出
        final_power_change = adjusted_power_change + error_compensation
        
        # 记录控制历史
        control_record = {
            'timestamp': self.training_step,
            'power_command': power_command,
            'power_control': power_control,
            'response_factor': response_factor,
            'compensation': compensation,
            'actual_power_change': final_power_change,
            'power_error': power_error,
            'response_speed': response_speed
        }
        
        self.control_history.append(control_record)
        
        # 维护历史长度
        if len(self.control_history) > 10000:
            self.control_history.pop(0)
        
        return {
            'power_control_signal': final_power_change,
            'response_speed': response_speed,
            'error_compensation': error_compensation,
            'control_confidence': abs(power_control),  # 控制置信度
            'estimated_settling_time': self._estimate_settling_time(response_speed),
            'control_record': control_record
        }
    
    def _estimate_settling_time(self, response_speed: float) -> float:
        """估算稳定时间"""
        # 基于响应速度估算稳定时间
        base_settling_time = 0.1  # 100ms基础稳定时间
        settling_time = base_settling_time / max(0.1, response_speed)
        return min(settling_time, 1.0)  # 最大1秒
    
    def adapt_to_constraints(self, 
                           constraints: Dict[str, Any],
                           current_performance: Dict[str, float]) -> Dict[str, float]:
        """
        根据约束自适应调整
        
        Args:
            constraints: 约束信息
            current_performance: 当前性能指标
            
        Returns:
            调整后的参数
        """
        # 提取约束信息
        response_time_limit = constraints.get('response_time_limit', 0.1)  # s
        constraint_severity = constraints.get('constraint_severity', 0.0)
        
        # 提取性能指标
        tracking_error = current_performance.get('tracking_error', 0.0)
        response_time = current_performance.get('response_time', 0.1)
        
        # 自适应调整参数
        adjusted_params = {}
        
        # 根据约束严重程度调整噪声
        if constraint_severity > 0.7:
            adjusted_params['noise_scale'] = 0.1  # 严格约束下减少探索
        elif constraint_severity > 0.3:
            adjusted_params['noise_scale'] = 0.3
        else:
            adjusted_params['noise_scale'] = 0.5
        
        # 根据跟踪误差调整学习率
        if tracking_error > 0.1:
            adjusted_params['lr_multiplier'] = 1.5  # 增加学习率
        elif tracking_error < 0.01:
            adjusted_params['lr_multiplier'] = 0.8  # 降低学习率
        else:
            adjusted_params['lr_multiplier'] = 1.0
        
        # 根据响应时间调整更新频率
        if response_time > response_time_limit * 1.2:
            adjusted_params['update_frequency'] = 2  # 增加更新频率
        else:
            adjusted_params['update_frequency'] = 1
        
        return adjusted_params
    
    def evaluate_control_performance(self, window_size: int = 100) -> Dict[str, float]:
        """评估控制性能"""
        if len(self.control_history) < window_size:
            recent_history = self.control_history
        else:
            recent_history = self.control_history[-window_size:]
        
        if not recent_history:
            return {'error': 'No control history available'}
        
        # 计算性能指标
        power_errors = [abs(record['power_error']) for record in recent_history]
        response_speeds = [record['response_speed'] for record in recent_history]
        control_signals = [abs(record['power_control']) for record in recent_history]
        
        performance = {
            'avg_tracking_error': np.mean(power_errors),
            'max_tracking_error': max(power_errors),
            'avg_response_speed': np.mean(response_speeds),
            'control_effort': np.mean(control_signals),
            'control_smoothness': 1.0 - np.std(control_signals),  # 控制平滑度
            'tracking_accuracy': 1.0 - min(1.0, np.mean(power_errors) / 1000.0),  # 跟踪精度
            'response_consistency': 1.0 - np.std(response_speeds)  # 响应一致性
        }
        
        return performance
    
    def train_episode(self, environment, max_steps: int = 1000) -> Dict[str, float]:
        """训练一个回合"""
        state = environment.reset()
        state = torch.FloatTensor(state).unsqueeze(0)
        
        episode_reward = 0.0
        episode_steps = 0
        
        # 重置噪声
        self.noise.reset()
        
        for step in range(max_steps):
            # 动作选择
            action = self.select_action(state, add_noise=True)
            
            # 环境交互
            next_state, reward, done, info = environment.step(action.squeeze(0).numpy())
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            
            # 存储经验
            self.add_experience(state, action, reward, next_state, done)
            
            # 网络训练
            if len(self.replay_buffer) > self.batch_size:
                train_info = self.train()
            
            # 累积奖励
            episode_reward += reward
            episode_steps += 1
            
            state = next_state
            
            if done:
                break
        
        # 记录回合统计
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        
        # 计算回合统计
        episode_stats = {
            'episode_reward': episode_reward,
            'episode_length': episode_steps,
            'avg_reward_per_step': episode_reward / max(1, episode_steps),
            'total_training_steps': self.training_step,
            'replay_buffer_size': len(self.replay_buffer),
            'noise_std': self.noise.sigma,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0
        }
        
        return episode_stats
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        if not self.episode_rewards:
            return {'error': 'No training episodes completed'}
        
        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        control_performance = self.evaluate_control_performance()
        
        stats = {
            'agent_id': self.agent_id,
            'total_episodes': self.episode_count,
            'training_steps': self.training_step,
            'total_iterations': self.total_it,
            
            'reward_statistics': {
                'avg_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'max_reward': max(self.episode_rewards),
                'min_reward': min(self.episode_rewards),
                'reward_trend': self._calculate_reward_trend()
            },
            
            'loss_statistics': {
                'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0.0,
                'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0.0,
                'loss_trend': 'decreasing' if len(self.critic_losses) > 50 and 
                             np.mean(self.critic_losses[-25:]) < np.mean(self.critic_losses[-50:-25]) 
                             else 'stable'
            },
            
            'control_performance': control_performance,
            
            'network_info': {
                'actor_parameters': sum(p.numel() for p in self.actor.parameters()),
                'critic_parameters': sum(p.numel() for p in self.critic.parameters()),
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'model_size_mb': sum(p.numel() for p in self.parameters()) * 4 / (1024 * 1024)
            },
            
            'buffer_info': {
                'buffer_size': len(self.replay_buffer),
                'buffer_capacity': self.replay_buffer.capacity,
                'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.capacity
            }
        }
        
        return stats
    
    def _calculate_reward_trend(self) -> str:
        """计算奖励趋势"""
        if len(self.episode_rewards) < 20:
            return "insufficient_data"
        
        recent_20 = self.episode_rewards[-20:]
        first_half = np.mean(recent_20[:10])
        second_half = np.mean(recent_20[10:])
        
        if second_half > first_half + 0.1:
            return "improving"
        elif second_half < first_half - 0.1:
            return "declining"
        else:
            return "stable"
    
    def save_checkpoint(self, filepath: str) -> bool:
        """保存检查点"""
        try:
            checkpoint = {
                'agent_id': self.agent_id,
                'total_it': self.total_it,
                'training_step': self.training_step,
                'episode_count': self.episode_count,
                
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_target_state_dict': self.actor_target.state_dict(),
                'critic_target_state_dict': self.critic_target.state_dict(),
                
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                
                'config': self.config,
                'model_config': self.model_config,
                'noise_state': self.noise.state,
                
                'statistics': self.get_agent_statistics()
            }
            
            torch.save(checkpoint, filepath)
            print(f"✅ DDPG智能体检查点已保存: {filepath}")
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
            self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            
            self.total_it = checkpoint['total_it']
            self.training_step = checkpoint['training_step']
            self.episode_count = checkpoint['episode_count']
            self.noise.state = checkpoint['noise_state']
            
            print(f"✅ DDPG智能体检查点已加载: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"DDPGAgent({self.agent_id}): "
                f"episodes={self.episode_count}, steps={self.training_step}, "
                f"buffer_size={len(self.replay_buffer)}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"DDPGAgent(agent_id='{self.agent_id}', "
                f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
                f"training_steps={self.training_step})")
