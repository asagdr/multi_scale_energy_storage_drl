import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import threading
from collections import deque
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.training_config import UpperLayerConfig, TrainingConfig
from config.model_config import ModelConfig
from environment.storage_environment import StorageEnvironment
from drl_agents.upper_layer.multi_objective_agent import MultiObjectiveAgent
from drl_agents.upper_layer.pareto_optimizer import ParetoOptimizer
from drl_agents.upper_layer.balance_analyzer import BalanceAnalyzer
from drl_agents.upper_layer.constraint_generator import ConstraintGenerator

@dataclass
class UpperTrainingMetrics:
    """上层训练指标"""
    episode: int = 0
    epoch: int = 0
    
    # 多目标奖励
    soc_balance_reward: float = 0.0
    temp_balance_reward: float = 0.0
    lifetime_reward: float = 0.0
    constraint_reward: float = 0.0
    total_reward: float = 0.0
    
    # 网络损失
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    pareto_loss: float = 0.0
    
    # 性能指标
    soc_std: float = 0.0
    temp_std: float = 0.0
    degradation_rate: float = 0.0
    constraint_violations: int = 0
    
    # 帕累托指标
    hypervolume: float = 0.0
    pareto_front_size: int = 0
    solution_diversity: float = 0.0
    
    # 时间指标
    decision_time: float = 0.0
    training_time: float = 0.0

class UpperEnvironmentWrapper:
    """上层环境包装器（5分钟时间尺度）"""
    
    def __init__(self, base_environment: StorageEnvironment, time_scale: int = 300):
        """
        初始化上层环境包装器
        
        Args:
            base_environment: 基础环境
            time_scale: 时间尺度（秒）
        """
        self.base_env = base_environment
        self.time_scale = time_scale  # 5分钟 = 300秒
        self.steps_per_decision = time_scale // 10  # 每个决策的下层步数 (300s / 10ms = 30000)
        
        # 累积状态
        self.accumulated_states = deque(maxlen=100)  # 保存最近100个状态
        self.step_count = 0
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        base_state = self.base_env.reset()
        self.accumulated_states.clear()
        self.accumulated_states.append(base_state)
        self.step_count = 0
        
        return self._get_upper_state()
    
    def step(self, upper_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行上层动作（跨越多个时间步）
        
        Args:
            upper_action: 上层动作 [power_command_ratio, soc_weight, temp_weight, lifetime_weight]
        """
        # 解析上层动作
        power_command_ratio = upper_action[0]
        objective_weights = upper_action[1:4]
        
        # 累积奖励和信息
        total_reward = 0.0
        episode_info = {
            'soc_balance_scores': [],
            'temp_balance_scores': [],
            'lifetime_scores': [],
            'constraint_violations': 0,
            'power_tracking_errors': [],
            'response_times': [],
            'soc_values': [],
            'temperatures': [],
            'soh_values': []
        }
        
        # 执行多个下层步骤
        steps_to_execute = min(self.steps_per_decision, 1000)  # 限制最大步数
        
        for step in range(steps_to_execute):
            # 生成下层动作（简化为基于上层指令的控制）
            lower_action = self._generate_lower_action(power_command_ratio, step)
            
            # 执行下层步骤
            next_state, step_reward, done, step_info = self.base_env.step(lower_action)
            
            # 累积信息
            total_reward += step_reward
            self._accumulate_step_info(episode_info, step_info)
            
            # 更新累积状态
            self.accumulated_states.append(next_state)
            self.step_count += 1
            
            if done:
                break
        
        # 计算上层奖励
        upper_reward = self._calculate_upper_reward(episode_info, objective_weights)
        
        # 构建上层状态
        upper_state = self._get_upper_state()
        
        # 构建上层信息
        upper_info = self._build_upper_info(episode_info)
        
        return upper_state, upper_reward, done, upper_info
    
    def _generate_lower_action(self, power_command_ratio: float, step: int) -> np.ndarray:
        """生成下层动作（简化版本）"""
        # 基于上层功率指令生成下层控制动作
        base_power = power_command_ratio * 0.8  # 限制在80%以内
        
        # 添加小幅变化以模拟精细控制
        power_variation = 0.1 * np.sin(step * 0.1)
        response_factor = 0.5 + 0.3 * np.random.random()
        compensation = 0.2 * (np.random.random() - 0.5)
        
        return np.array([base_power + power_variation, response_factor, compensation])
    
    def _accumulate_step_info(self, episode_info: Dict, step_info: Dict):
        """累积步骤信息"""
        # SOC均衡信息
        if 'soc_values' in step_info:
            episode_info['soc_values'].extend(step_info['soc_values'])
        
        # 温度信息
        if 'temperatures' in step_info:
            episode_info['temperatures'].extend(step_info['temperatures'])
        
        # SOH信息
        if 'soh_values' in step_info:
            episode_info['soh_values'].extend(step_info['soh_values'])
        
        # 约束违反
        episode_info['constraint_violations'] += step_info.get('constraint_violations', 0)
        
        # 功率跟踪误差
        if 'power_tracking_error' in step_info:
            episode_info['power_tracking_errors'].append(step_info['power_tracking_error'])
        
        # 响应时间
        if 'response_time' in step_info:
            episode_info['response_times'].append(step_info['response_time'])
    
    def _calculate_upper_reward(self, episode_info: Dict, weights: np.ndarray) -> float:
        """计算上层奖励"""
        # SOC均衡奖励
        soc_values = episode_info.get('soc_values', [50.0])
        soc_std = np.std(soc_values) if len(soc_values) > 1 else 0.0
        soc_reward = max(0.0, 1.0 - soc_std / 10.0)  # σ_SOC越小奖励越高
        
        # 温度均衡奖励
        temperatures = episode_info.get('temperatures', [25.0])
        temp_std = np.std(temperatures) if len(temperatures) > 1 else 0.0
        temp_reward = max(0.0, 1.0 - temp_std / 15.0)
        
        # 寿命奖励
        soh_values = episode_info.get('soh_values', [100.0])
        min_soh = min(soh_values) if soh_values else 100.0
        lifetime_reward = min_soh / 100.0  # SOH越高奖励越高
        
        # 约束满足奖励
        violations = episode_info.get('constraint_violations', 0)
        constraint_reward = max(0.0, 1.0 - violations * 0.1)
        
        # 加权组合
        if len(weights) >= 3:
            total_reward = (weights[0] * soc_reward + 
                          weights[1] * temp_reward + 
                          weights[2] * lifetime_reward)
        else:
            total_reward = (soc_reward + temp_reward + lifetime_reward) / 3.0
        
        # 添加约束惩罚
        total_reward = total_reward * constraint_reward
        
        return total_reward
    
    def _get_upper_state(self) -> np.ndarray:
        """获取上层状态"""
        if not self.accumulated_states:
            return np.zeros(20)  # 默认状态维度
        
        # 使用最近的状态
        recent_states = list(self.accumulated_states)[-10:]  # 最近10个状态
        
        if len(recent_states) == 0:
            return np.zeros(20)
        
        # 计算统计特征
        states_matrix = np.array(recent_states)
        
        # 基本统计
        mean_state = np.mean(states_matrix, axis=0)
        std_state = np.std(states_matrix, axis=0)
        
        # 趋势特征
        if len(recent_states) > 1:
            trend = states_matrix[-1] - states_matrix[0]
        else:
            trend = np.zeros_like(mean_state)
        
        # 组合特征（取前面几个重要维度）
        upper_state = np.concatenate([
            mean_state[:5],   # 平均状态（前5维）
            std_state[:5],    # 状态标准差（前5维）
            trend[:5],        # 状态趋势（前5维）
            [self.step_count / 10000.0]  # 时间特征
        ])
        
        return upper_state[:20]  # 确保维度一致
    
    def _build_upper_info(self, episode_info: Dict) -> Dict:
        """构建上层信息"""
        # 计算关键指标
        soc_values = episode_info.get('soc_values', [])
        temperatures = episode_info.get('temperatures', [])
        soh_values = episode_info.get('soh_values', [])
        
        upper_info = {
            'soc_std': np.std(soc_values) if len(soc_values) > 1 else 0.0,
            'temp_std': np.std(temperatures) if len(temperatures) > 1 else 0.0,
            'min_soh': min(soh_values) if soh_values else 100.0,
            'constraint_violations': episode_info.get('constraint_violations', 0),
            'avg_power_error': np.mean(episode_info.get('power_tracking_errors', [0.0])),
            'avg_response_time': np.mean(episode_info.get('response_times', [0.05]))
        }
        
        return upper_info

class UpperLayerTrainer:
    """
    上层训练器
    专门负责5分钟级高层决策的训练
    """
    
    def __init__(self,
                 config: UpperLayerConfig,
                 model_config: ModelConfig,
                 trainer_id: str = "UpperTrainer_001"):
        """
        初始化上层训练器
        
        Args:
            config: 上层训练配置
            model_config: 模型配置
            trainer_id: 训练器ID
        """
        self.config = config
        self.model_config = model_config
        self.trainer_id = trainer_id
        
        # === 设置随机种子 ===
        self._set_random_seeds(config.random_seed)
        
        # === 初始化环境 ===
        from config.battery_params import BatteryParams
        from config.system_config import SystemConfig
        
        battery_params = BatteryParams()
        system_config = SystemConfig()
        
        base_env = StorageEnvironment(
            battery_params=battery_params,
            system_config=system_config,
            env_id=f"UpperEnv_{trainer_id}"
        )
        
        self.environment = UpperEnvironmentWrapper(base_env, time_scale=300)
        
        # === 初始化智能体和组件 ===
        self.agent = MultiObjectiveAgent(
            config=config,
            model_config=model_config,
            agent_id=f"UpperAgent_{trainer_id}"
        )
        
        self.pareto_optimizer = ParetoOptimizer(
            config=config,
            n_objectives=4,
            optimizer_id=f"ParetoOpt_{trainer_id}"
        )
        
        self.balance_analyzer = BalanceAnalyzer(
            config=config,
            model_config=model_config,
            analyzer_id=f"BalanceAnalyzer_{trainer_id}"
        )
        
        self.constraint_generator = ConstraintGenerator(
            config=config,
            model_config=model_config,
            generator_id=f"ConstraintGen_{trainer_id}"
        )
        
        # === 优化器设置 ===
        self.actor_optimizer = optim.Adam(
            self.agent.actor.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        self.critic_optimizer = optim.Adam(
            self.agent.critic.parameters(),
            lr=config.learning_rate * 2,
            weight_decay=1e-4
        )
        
        # === 学习率调度器 ===
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer,
            T_max=config.max_episodes // 4,
            eta_min=config.learning_rate * 0.1
        )
        
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer,
            T_max=config.max_episodes // 4,
            eta_min=config.learning_rate * 0.2
        )
        
        # === 训练状态 ===
        self.training_state = {
            'current_episode': 0,
            'current_epoch': 0,
            'total_episodes': config.max_episodes,
            'is_training': False,
            'best_hypervolume': 0.0,
            'best_total_reward': float('-inf')
        }
        
        # === 训练历史 ===
        self.training_history: List[UpperTrainingMetrics] = []
        self.pareto_solutions_history: List[Dict] = []
        
        # === 日志设置 ===
        self._setup_logging()
        
        # === 保存路径 ===
        self.save_dir = f"checkpoints/upper_layer/{trainer_id}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"✅ 上层训练器初始化完成: {trainer_id}")
        print(f"   时间尺度: 5分钟级决策")
        print(f"   目标数量: 4个 (SOC均衡, 温度均衡, 寿命成本, 约束满足)")
        print(f"   训练算法: Multi-Objective Transformer DRL + Pareto Optimization")
    
    def _set_random_seeds(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = f"logs/upper_layer/{self.trainer_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/upper_training.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"UpperTrainer_{self.trainer_id}")
    
    def train(self,
             max_episodes: Optional[int] = None,
             save_frequency: int = 50,
             eval_frequency: int = 25) -> Dict[str, Any]:
        """
        开始上层训练
        
        Args:
            max_episodes: 最大训练回合数
            save_frequency: 保存频率
            eval_frequency: 评估频率
            
        Returns:
            训练结果统计
        """
        if max_episodes is None:
            max_episodes = self.config.max_episodes
        
        self.training_state['is_training'] = True
        self.training_state['total_episodes'] = max_episodes
        
        self.logger.info(f"开始上层DRL训练: 目标回合数={max_episodes}")
        
        start_time = time.time()
        
        try:
            for episode in range(max_episodes):
                self.training_state['current_episode'] = episode
                
                # 训练一个回合
                episode_metrics = self._train_episode()
                
                # 记录训练指标
                self.training_history.append(episode_metrics)
                
                # 更新最佳性能
                self._update_best_performance(episode_metrics)
                
                # 学习率调度
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                
                # 定期保存
                if (episode + 1) % save_frequency == 0:
                    self._save_checkpoint(episode)
                
                # 定期评估
                if (episode + 1) % eval_frequency == 0:
                    eval_results = self._evaluate_model()
                    self.logger.info(f"回合 {episode}: 评估结果 = {eval_results}")
                
                # 日志输出
                if (episode + 1) % 10 == 0:
                    self._log_training_progress(episode_metrics)
                
                # 检查提前停止条件
                if self._should_early_stop():
                    self.logger.info(f"满足提前停止条件，在回合 {episode} 结束训练")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("上层训练被用户中断")
        except Exception as e:
            self.logger.error(f"上层训练过程中发生错误: {str(e)}")
            raise
        finally:
            self.training_state['is_training'] = False
            end_time = time.time()
            
            # 最终保存
            self._save_final_model()
            
            # 训练统计
            total_time = end_time - start_time
            training_stats = self._calculate_training_statistics(total_time)
            
            self.logger.info(f"上层训练完成! 总用时: {total_time:.2f}秒")
            
            return training_stats
    
    def _train_episode(self) -> UpperTrainingMetrics:
        """训练一个回合"""
        episode_start_time = time.time()
        
        # 重置环境
        state = self.environment.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 初始化回合指标
        episode_metrics = UpperTrainingMetrics(
            episode=self.training_state['current_episode']
        )
        
        # 累积奖励
        total_reward = 0.0
        objective_rewards = np.zeros(4)
        
        # 性能指标累积
        soc_stds = []
        temp_stds = []
        degradation_rates = []
        constraint_violations = 0
        
        # 帕累托解集合
        episode_solutions = []
        
        step = 0
        max_steps = 12  # 每个回合最大12个决策（1小时，每5分钟一次）
        
        while step < max_steps:
            decision_start_time = time.time()
            
            # === 1. 均衡分析 ===
            balance_analysis = self.balance_analyzer(state_tensor, return_detailed=True)
            
            # === 2. 约束生成 ===
            constraint_matrix = self.constraint_generator(state_tensor)
            
            # === 3. 多目标决策 ===
            # 获取当前权重向量
            weight_vector = self.pareto_optimizer.get_next_weight_vector()
            weight_tensor = torch.FloatTensor(weight_vector).unsqueeze(0)
            
            # 动作选择
            action = self.agent.select_action(state_tensor, weight_tensor, add_noise=True)
            
            # === 4. 环境交互 ===
            next_state, reward, done, info = self.environment.step(action.detach().cpu().numpy().squeeze())
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # === 5. 计算多目标奖励 ===
            multi_objective_rewards = self._calculate_multi_objective_rewards(info)
            
            # === 6. 添加到帕累托优化器 ===
            success = self.pareto_optimizer.add_solution(
                objectives=multi_objective_rewards,
                weights=weight_vector,
                action=action.detach().cpu().numpy().squeeze(),
                state=state_tensor.detach().cpu().numpy().squeeze()
            )
            
            if success:
                episode_solutions.append({
                    'objectives': multi_objective_rewards.copy(),
                    'weights': weight_vector.copy(),
                    'action': action.detach().cpu().numpy().squeeze().copy()
                })
            
            # === 7. 更新智能体 ===
            training_start_time = time.time()
            
            # 添加经验
            reward_tensor = torch.FloatTensor(multi_objective_rewards)
            self.agent.add_experience(
                state_tensor, action, reward_tensor, next_state_tensor, done, weight_tensor
            )
            
            # 更新网络
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                losses = self.agent.update_networks()
                episode_metrics.actor_loss += losses.get('actor_loss', 0.0)
                episode_metrics.critic_loss += losses.get('critic_loss', 0.0)
            
            training_time = time.time() - training_start_time
            episode_metrics.training_time += training_time
            
            # === 8. 累积指标 ===
            total_reward += reward
            objective_rewards += multi_objective_rewards
            
            # 性能指标
            soc_stds.append(info.get('soc_std', 0.0))
            temp_stds.append(info.get('temp_std', 0.0))
            degradation_rates.append(info.get('min_soh', 100.0))
            constraint_violations += info.get('constraint_violations', 0)
            
            # 决策时间
            decision_time = time.time() - decision_start_time
            episode_metrics.decision_time += decision_time
            
            # 更新状态
            state_tensor = next_state_tensor
            step += 1
            
            if done:
                break
        
        # === 计算回合指标 ===
        episode_end_time = time.time()
        
        episode_metrics.total_reward = total_reward
        episode_metrics.soc_balance_reward = objective_rewards[0]
        episode_metrics.temp_balance_reward = objective_rewards[1]
        episode_metrics.lifetime_reward = objective_rewards[2]
        episode_metrics.constraint_reward = objective_rewards[3]
        
        episode_metrics.soc_std = np.mean(soc_stds) if soc_stds else 0.0
        episode_metrics.temp_std = np.mean(temp_stds) if temp_stds else 0.0
        episode_metrics.degradation_rate = 100.0 - np.mean(degradation_rates) if degradation_rates else 0.0
        episode_metrics.constraint_violations = constraint_violations
        
        # 帕累托指标
        episode_metrics.hypervolume = self.pareto_optimizer.calculate_hypervolume()
        episode_metrics.pareto_front_size = len(self.pareto_optimizer.pareto_front.solutions)
        
        # 记录帕累托解
        if episode_solutions:
            self.pareto_solutions_history.append({
                'episode': self.training_state['current_episode'],
                'solutions': episode_solutions,
                'hypervolume': episode_metrics.hypervolume
            })
        
        return episode_metrics
    
    def _calculate_multi_objective_rewards(self, info: Dict[str, Any]) -> np.ndarray:
        """计算多目标奖励向量"""
        # 1. SOC均衡奖励
        soc_std = info.get('soc_std', 0.0)
        soc_reward = max(0.0, 1.0 - soc_std / 10.0)  # σ_SOC越小奖励越高
        
        # 2. 温度均衡奖励
        temp_std = info.get('temp_std', 0.0)
        temp_reward = max(0.0, 1.0 - temp_std / 15.0)
        
        # 3. 寿命奖励
        min_soh = info.get('min_soh', 100.0)
        lifetime_reward = min_soh / 100.0
        
        # 4. 约束满足奖励
        violations = info.get('constraint_violations', 0)
        constraint_reward = max(0.0, 1.0 - violations * 0.1)
        
        return np.array([soc_reward, temp_reward, lifetime_reward, constraint_reward])
    
    def _update_best_performance(self, metrics: UpperTrainingMetrics):
        """更新最佳性能"""
        # 更新最佳超体积
        if metrics.hypervolume > self.training_state['best_hypervolume']:
            self.training_state['best_hypervolume'] = metrics.hypervolume
            self._save_best_model('best_hypervolume')
        
        # 更新最佳总奖励
        if metrics.total_reward > self.training_state['best_total_reward']:
            self.training_state['best_total_reward'] = metrics.total_reward
            self._save_best_model('best_reward')
    
    def _evaluate_model(self) -> Dict[str, float]:
        """评估模型"""
        eval_episodes = 5
        eval_results = {
            'total_rewards': [],
            'soc_balance_rewards': [],
            'temp_balance_rewards': [],
            'lifetime_rewards': [],
            'constraint_rewards': [],
            'hypervolumes': [],
            'soc_stds': [],
            'temp_stds': []
        }
        
        for _ in range(eval_episodes):
            state = self.environment.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            episode_reward = 0.0
            episode_objectives = np.zeros(4)
            episode_soc_stds = []
            episode_temp_stds = []
            
            for step in range(12):  # 12个决策步
                # 使用确定性策略
                weight_vector = np.array([0.25, 0.25, 0.25, 0.25])  # 均匀权重
                weight_tensor = torch.FloatTensor(weight_vector).unsqueeze(0)
                
                action = self.agent.select_action(state_tensor, weight_tensor, add_noise=False)
                
                next_state, reward, done, info = self.environment.step(action.detach().cpu().numpy().squeeze())
                
                episode_reward += reward
                episode_objectives += self._calculate_multi_objective_rewards(info)
                episode_soc_stds.append(info.get('soc_std', 0.0))
                episode_temp_stds.append(info.get('temp_std', 0.0))
                
                state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                
                if done:
                    break
            
            eval_results['total_rewards'].append(episode_reward)
            eval_results['soc_balance_rewards'].append(episode_objectives[0])
            eval_results['temp_balance_rewards'].append(episode_objectives[1])
            eval_results['lifetime_rewards'].append(episode_objectives[2])
            eval_results['constraint_rewards'].append(episode_objectives[3])
            eval_results['soc_stds'].append(np.mean(episode_soc_stds))
            eval_results['temp_stds'].append(np.mean(episode_temp_stds))
        
        # 计算统计结果
        return {
            'mean_total_reward': np.mean(eval_results['total_rewards']),
            'std_total_reward': np.std(eval_results['total_rewards']),
            'mean_soc_balance': np.mean(eval_results['soc_balance_rewards']),
            'mean_temp_balance': np.mean(eval_results['temp_balance_rewards']),
            'mean_lifetime': np.mean(eval_results['lifetime_rewards']),
            'mean_constraint': np.mean(eval_results['constraint_rewards']),
            'mean_soc_std': np.mean(eval_results['soc_stds']),
            'mean_temp_std': np.mean(eval_results['temp_stds'])
        }
    
    def _should_early_stop(self) -> bool:
        """检查是否应该提前停止"""
        if len(self.training_history) < 50:
            return False
        
        # 检查最近50个回合的超体积是否停滞
        recent_hypervolumes = [metrics.hypervolume for metrics in self.training_history[-50:]]
        
        if len(recent_hypervolumes) >= 25:
            first_half = np.mean(recent_hypervolumes[:25])
            second_half = np.mean(recent_hypervolumes[25:])
            
            # 如果超体积改善小于1%，认为停滞
            improvement = (second_half - first_half) / max(abs(first_half), 1e-6)
            return improvement < 0.01
        
        return False
    
    def _log_training_progress(self, metrics: UpperTrainingMetrics):
        """记录训练进度"""
        self.logger.info(
            f"回合 {metrics.episode}: "
            f"总奖励={metrics.total_reward:.4f}, "
            f"SOC均衡={metrics.soc_balance_reward:.4f}, "
            f"温度均衡={metrics.temp_balance_reward:.4f}, "
            f"寿命={metrics.lifetime_reward:.4f}, "
            f"约束={metrics.constraint_reward:.4f}, "
            f"σ_SOC={metrics.soc_std:.3f}, "
            f"超体积={metrics.hypervolume:.4f}, "
            f"前沿大小={metrics.pareto_front_size}"
        )
    
    def _save_checkpoint(self, episode: int) -> str:
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}.pth")
        
        checkpoint = {
            'episode': episode,
            'training_state': self.training_state,
            'agent_state': self.agent.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'actor_scheduler_state': self.actor_scheduler.state_dict(),
            'critic_scheduler_state': self.critic_scheduler.state_dict(),
            'pareto_optimizer_state': self.pareto_optimizer.get_optimizer_statistics(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")
        
        return checkpoint_path
    
    def _save_best_model(self, model_type: str) -> str:
        """保存最佳模型"""
        best_model_path = os.path.join(self.save_dir, f"best_model_{model_type}.pth")
        
        best_model = {
            'episode': self.training_state['current_episode'],
            'model_type': model_type,
            'performance': {
                'best_hypervolume': self.training_state['best_hypervolume'],
                'best_total_reward': self.training_state['best_total_reward']
            },
            'agent_state': self.agent.state_dict(),
            'pareto_solutions': self.pareto_solutions_history[-1] if self.pareto_solutions_history else None,
            'config': self.config
        }
        
        torch.save(best_model, best_model_path)
        self.logger.info(f"最佳模型已保存: {best_model_path}")
        
        return best_model_path
    
    def _save_final_model(self):
        """保存最终模型"""
        # 保存最终检查点
        final_checkpoint = self._save_checkpoint(self.training_state['current_episode'])
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, "upper_training_history.json")
        with open(history_path, 'w') as f:
            serializable_history = []
            for metrics in self.training_history:
                serializable_history.append({
                    'episode': metrics.episode,
                    'total_reward': metrics.total_reward,
                    'soc_balance_reward': metrics.soc_balance_reward,
                    'temp_balance_reward': metrics.temp_balance_reward,
                    'lifetime_reward': metrics.lifetime_reward,
                    'constraint_reward': metrics.constraint_reward,
                    'hypervolume': metrics.hypervolume,
                    'pareto_front_size': metrics.pareto_front_size,
                    'soc_std': metrics.soc_std,
                    'temp_std': metrics.temp_std
                })
            json.dump(serializable_history, f, indent=2)
        
        # 保存帕累托前沿
        pareto_path = os.path.join(self.save_dir, "pareto_solutions.json")
        with open(pareto_path, 'w') as f:
            json.dump(self.pareto_solutions_history, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        self.logger.info(f"上层训练数据已保存: {self.save_dir}")
    
    def _calculate_training_statistics(self, total_time: float) -> Dict[str, Any]:
        """计算训练统计信息"""
        if not self.training_history:
            return {}
        
        # 基础统计
        total_rewards = [m.total_reward for m in self.training_history]
        hypervolumes = [m.hypervolume for m in self.training_history]
        soc_stds = [m.soc_std for m in self.training_history]
        temp_stds = [m.temp_std for m in self.training_history]
        
        stats = {
            'training_summary': {
                'total_episodes': len(self.training_history),
                'total_time': total_time,
                'avg_episode_time': total_time / len(self.training_history),
                'best_hypervolume': self.training_state['best_hypervolume'],
                'best_total_reward': self.training_state['best_total_reward']
            },
            
            'reward_statistics': {
                'mean_total_reward': np.mean(total_rewards),
                'std_total_reward': np.std(total_rewards),
                'max_total_reward': np.max(total_rewards),
                'min_total_reward': np.min(total_rewards)
            },
            
            'pareto_statistics': {
                'mean_hypervolume': np.mean(hypervolumes),
                'std_hypervolume': np.std(hypervolumes),
                'max_hypervolume': np.max(hypervolumes),
                'final_front_size': self.training_history[-1].pareto_front_size
            },
            
            'performance_statistics': {
                'mean_soc_std': np.mean(soc_stds),
                'mean_temp_std': np.mean(temp_stds),
                'soc_balance_achievement_rate': np.mean([1 if s < 2.0 else 0 for s in soc_stds]),
                'temp_balance_achievement_rate': np.mean([1 if s < 5.0 else 0 for s in temp_stds])
            }
        }
        
        return stats
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.training_state = checkpoint['training_state']
            self.agent.load_state_dict(checkpoint['agent_state'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state'])
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state'])
            self.training_history = checkpoint['training_history']
            
            self.logger.info(f"上层检查点加载成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"上层检查点加载失败: {str(e)}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        return {
            'trainer_id': self.trainer_id,
            'training_state': self.training_state.copy(),
            'pareto_front_size': len(self.pareto_optimizer.pareto_front.solutions),
            'current_hypervolume': self.pareto_optimizer.calculate_hypervolume(),
            'recent_performance': (
                self.training_history[-5:] if len(self.training_history) >= 5 
                else self.training_history
            )
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"UpperLayerTrainer({self.trainer_id}): "
                f"回合={self.training_state['current_episode']}/{self.training_state['total_episodes']}, "
                f"训练中={self.training_state['is_training']}")
