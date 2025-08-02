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

from config.training_config import LowerLayerConfig, TrainingConfig
from config.model_config import ModelConfig
from environment.storage_environment import StorageEnvironment
from drl_agents.lower_layer.ddpg_agent import DDPGAgent
from drl_agents.lower_layer.power_tracker import PowerTracker
from drl_agents.lower_layer.constraint_handler import ConstraintHandler
from drl_agents.lower_layer.temperature_compensator import TemperatureCompensator
from drl_agents.lower_layer.response_optimizer import ResponseOptimizer

@dataclass
class LowerTrainingMetrics:
    """下层训练指标"""
    episode: int = 0
    step: int = 0
    
    # 奖励分解
    power_tracking_reward: float = 0.0
    response_speed_reward: float = 0.0
    constraint_satisfaction_reward: float = 0.0
    control_smoothness_reward: float = 0.0
    total_reward: float = 0.0
    
    # 网络损失
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    q_value: float = 0.0
    
    # 控制性能
    power_tracking_error: float = 0.0
    response_time: float = 0.0
    constraint_violations: int = 0
    control_effort: float = 0.0
    
    # 温度补偿
    max_temperature: float = 0.0
    temp_compensation_active: bool = False
    cooling_enhancement: float = 0.0
    
    # 优化性能
    tracking_accuracy: float = 0.0
    stability_margin: float = 0.0
    
    # 时间指标
    control_time: float = 0.0
    optimization_time: float = 0.0

class LowerEnvironmentWrapper:
    """下层环境包装器（10ms时间尺度）"""
    
    def __init__(self, base_environment: StorageEnvironment, time_scale: float = 0.01):
        """
        初始化下层环境包装器
        
        Args:
            base_environment: 基础环境
            time_scale: 时间尺度（秒）
        """
        self.base_env = base_environment
        self.time_scale = time_scale  # 10ms = 0.01s
        
        # 上层指令缓存
        self.upper_commands = {
            'power_command': 0.0,
            'constraint_matrix': None,
            'weight_vector': [0.25, 0.25, 0.25, 0.25],
            'balance_targets': {'soc_target_std': 2.0, 'temp_target_std': 5.0}
        }
        
        # 控制历史
        self.control_history = deque(maxlen=1000)
        self.step_count = 0
        
        # 性能指标计算
        self.power_reference_history = deque(maxlen=100)
        self.power_actual_history = deque(maxlen=100)
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        base_state = self.base_env.reset()
        self.control_history.clear()
        self.power_reference_history.clear()
        self.power_actual_history.clear()
        self.step_count = 0
        
        return self._get_lower_state(base_state)
    
    def step(self, lower_action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行下层动作（10ms级控制）
        
        Args:
            lower_action: 下层动作 [power_control, response_factor, compensation]
        """
        # 解析下层动作
        power_control = lower_action[0]        # 功率控制信号 [-1, 1]
        response_factor = lower_action[1]      # 响应速度因子 [-1, 1]
        compensation = lower_action[2]         # 误差补偿 [-1, 1]
        
        # 转换为实际控制信号
        actual_power_command = self._convert_to_power_command(
            power_control, response_factor, compensation
        )
        
        # 执行基础环境步骤
        next_base_state, base_reward, done, base_info = self.base_env.step([actual_power_command])
        
        # 计算下层奖励
        lower_reward = self._calculate_lower_reward(
            lower_action, actual_power_command, base_info
        )
        
        # 构建下层状态
        lower_state = self._get_lower_state(next_base_state)
        
        # 构建下层信息
        lower_info = self._build_lower_info(lower_action, actual_power_command, base_info)
        
        # 记录控制历史
        self._record_control_step(lower_action, actual_power_command, lower_info)
        
        self.step_count += 1
        
        return lower_state, lower_reward, done, lower_info
    
    def update_upper_commands(self, commands: Dict[str, Any]):
        """更新上层指令"""
        self.upper_commands.update(commands)
    
    def _convert_to_power_command(self, 
                                 power_control: float,
                                 response_factor: float,
                                 compensation: float) -> float:
        """转换为实际功率指令"""
        # 基础功率指令
        base_power = self.upper_commands.get('power_command', 0.0)
        
        # 功率调节
        max_power_change = 1000.0  # 1kW最大变化
        power_adjustment = power_control * max_power_change
        
        # 响应速度调节
        response_speed = (response_factor + 1.0) / 2.0  # [0, 1]
        adjusted_power = base_power + power_adjustment * response_speed
        
        # 误差补偿
        if len(self.power_reference_history) > 0 and len(self.power_actual_history) > 0:
            power_error = self.power_reference_history[-1] - self.power_actual_history[-1]
            error_compensation = compensation * power_error * 0.1
            adjusted_power += error_compensation
        
        # 限制功率范围
        max_power = 50000.0  # 50kW
        adjusted_power = np.clip(adjusted_power, -max_power, max_power)
        
        return adjusted_power
    
    def _calculate_lower_reward(self, 
                               lower_action: np.ndarray,
                               actual_power: float,
                               base_info: Dict[str, Any]) -> float:
        """计算下层奖励"""
        # 1. 功率跟踪奖励
        power_reference = self.upper_commands.get('power_command', 0.0)
        power_error = abs(actual_power - power_reference)
        tracking_reward = max(0.0, 1.0 - power_error / 1000.0)  # 1kW容差
        
        # 2. 响应速度奖励
        response_factor = lower_action[1]
        response_speed = (response_factor + 1.0) / 2.0
        response_reward = response_speed
        
        # 3. 约束满足奖励
        constraint_violations = base_info.get('constraint_violations', 0)
        constraint_reward = max(0.0, 1.0 - constraint_violations * 0.2)
        
        # 4. 控制平滑性奖励
        if len(self.control_history) > 1:
            prev_action = self.control_history[-1]['action']
            action_change = np.linalg.norm(lower_action - prev_action)
            smoothness_reward = max(0.0, 1.0 - action_change / 2.0)
        else:
            smoothness_reward = 1.0
        
        # 权重组合
        weights = self.upper_commands.get('weight_vector', [0.4, 0.3, 0.2, 0.1])
        if len(weights) >= 4:
            total_reward = (weights[0] * tracking_reward + 
                           weights[1] * response_reward + 
                           weights[2] * constraint_reward + 
                           weights[3] * smoothness_reward)
        else:
            total_reward = 0.4 * tracking_reward + 0.3 * response_reward + 0.2 * constraint_reward + 0.1 * smoothness_reward
        
        return total_reward
    
    def _get_lower_state(self, base_state: np.ndarray) -> np.ndarray:
        """获取下层状态"""
        # 基础状态特征
        lower_state = base_state[:10].copy()  # 取前10维作为基础
        
        # 添加控制相关特征
        control_features = []
        
        # 功率跟踪误差
        power_reference = self.upper_commands.get('power_command', 0.0)
        current_power = base_state[4] if len(base_state) > 4 else 0.0
        power_error = power_reference - current_power
        control_features.append(power_error / 10000.0)  # 归一化
        
        # 功率变化率
        if len(self.power_actual_history) > 1:
            power_rate = (self.power_actual_history[-1] - self.power_actual_history[-2]) / self.time_scale
            control_features.append(power_rate / 100000.0)  # 归一化
        else:
            control_features.append(0.0)
        
        # 响应时间目标
        target_response_time = 0.05  # 50ms目标
        control_features.append(target_response_time / 0.1)  # 归一化
        
        # 约束严重程度
        constraint_severity = 0.0
        if self.upper_commands.get('constraint_matrix') is not None:
            constraint_severity = 0.5  # 简化表示
        control_features.append(constraint_severity)
        
        # 时间特征
        time_feature = (self.step_count % 1000) / 1000.0  # 周期性时间特征
        control_features.append(time_feature)
        
        # 组合状态
        extended_state = np.concatenate([lower_state, control_features])
        
        # 确保状态维度一致
        target_dim = 15
        if len(extended_state) > target_dim:
            extended_state = extended_state[:target_dim]
        elif len(extended_state) < target_dim:
            padding = np.zeros(target_dim - len(extended_state))
            extended_state = np.concatenate([extended_state, padding])
        
        return extended_state
    
    def _build_lower_info(self, 
                         lower_action: np.ndarray,
                         actual_power: float,
                         base_info: Dict[str, Any]) -> Dict[str, Any]:
        """构建下层信息"""
        power_reference = self.upper_commands.get('power_command', 0.0)
        power_error = abs(actual_power - power_reference)
        
        # 记录功率历史
        self.power_reference_history.append(power_reference)
        self.power_actual_history.append(actual_power)
        
        # 计算响应时间（简化）
        response_time = 0.01 + abs(lower_action[1]) * 0.04  # 10-50ms范围
        
        lower_info = {
            'power_tracking_error': power_error,
            'response_time': response_time,
            'constraint_violations': base_info.get('constraint_violations', 0),
            'control_effort': np.linalg.norm(lower_action),
            'actual_power_command': actual_power,
            'power_reference': power_reference,
            'tracking_accuracy': max(0.0, 1.0 - power_error / 1000.0),
            'response_speed': (lower_action[1] + 1.0) / 2.0,
            'control_smoothness': 1.0,  # 简化计算
            'temperatures': base_info.get('temperatures', [25.0] * 10),
            'max_temperature': base_info.get('max_temperature', 25.0)
        }
        
        return lower_info
    
    def _record_control_step(self, 
                           lower_action: np.ndarray,
                           actual_power: float,
                           lower_info: Dict[str, Any]):
        """记录控制步骤"""
        record = {
            'step': self.step_count,
            'timestamp': self.step_count * self.time_scale,
            'action': lower_action.copy(),
            'actual_power': actual_power,
            'power_error': lower_info['power_tracking_error'],
            'response_time': lower_info['response_time']
        }
        
        self.control_history.append(record)

class LowerLayerTrainer:
    """
    下层训练器
    专门负责10ms级底层控制的训练
    """
    
    def __init__(self,
                 config: LowerLayerConfig,
                 model_config: ModelConfig,
                 trainer_id: str = "LowerTrainer_001"):
        """
        初始化下层训练器
        
        Args:
            config: 下层训练配置
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
            env_id=f"LowerEnv_{trainer_id}"
        )
        
        self.environment = LowerEnvironmentWrapper(base_env, time_scale=0.01)
        
        # === 初始化智能体和组件 ===
        self.agent = DDPGAgent(
            config=config,
            model_config=model_config,
            agent_id=f"LowerAgent_{trainer_id}"
        )
        
        self.power_tracker = PowerTracker(
            config=config,
            model_config=model_config,
            tracker_id=f"PowerTracker_{trainer_id}"
        )
        
        self.constraint_handler = ConstraintHandler(
            config=config,
            model_config=model_config,
            handler_id=f"ConstraintHandler_{trainer_id}"
        )
        
        self.temperature_compensator = TemperatureCompensator(
            config=config,
            model_config=model_config,
            compensator_id=f"TempCompensator_{trainer_id}"
        )
        
        self.response_optimizer = ResponseOptimizer(
            config=config,
            model_config=model_config,
            optimizer_id=f"ResponseOpt_{trainer_id}"
        )
        
        # === 训练状态 ===
        self.training_state = {
            'current_episode': 0,
            'current_step': 0,
            'total_episodes': config.max_episodes,
            'total_steps': 0,
            'is_training': False,
            'best_tracking_accuracy': 0.0,
            'best_response_time': float('inf')
        }
        
        # === 训练历史 ===
        self.training_history: List[LowerTrainingMetrics] = []
        self.performance_history = deque(maxlen=1000)
        
        # === 自适应学习 ===
        self.adaptive_learning = {
            'noise_decay': 0.995,
            'exploration_schedule': 'linear',
            'curriculum_learning': True,
            'difficulty_level': 1.0
        }
        
        # === 日志设置 ===
        self._setup_logging()
        
        # === 保存路径 ===
        self.save_dir = f"checkpoints/lower_layer/{trainer_id}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"✅ 下层训练器初始化完成: {trainer_id}")
        print(f"   时间尺度: 10ms级控制")
        print(f"   目标: 功率跟踪, 响应优化, 约束满足, 控制平滑")
        print(f"   训练算法: DDPG with Constraints + Specialized Controllers")
    
    def _set_random_seeds(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = f"logs/lower_layer/{self.trainer_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/lower_training.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"LowerTrainer_{self.trainer_id}")
    
    def train(self,
             max_episodes: Optional[int] = None,
             max_steps_per_episode: int = 10000,
             save_frequency: int = 100,
             eval_frequency: int = 50) -> Dict[str, Any]:
        """
        开始下层训练
        
        Args:
            max_episodes: 最大训练回合数
            max_steps_per_episode: 每回合最大步数
            save_frequency: 保存频率
            eval_frequency: 评估频率
            
        Returns:
            训练结果统计
        """
        if max_episodes is None:
            max_episodes = self.config.max_episodes
        
        self.training_state['is_training'] = True
        self.training_state['total_episodes'] = max_episodes
        
        self.logger.info(f"开始下层DRL训练: 目标回合数={max_episodes}")
        
        start_time = time.time()
        
        try:
            for episode in range(max_episodes):
                self.training_state['current_episode'] = episode
                
                # 自适应难度调整
                self._adjust_training_difficulty(episode)
                
                # 训练一个回合
                episode_metrics = self._train_episode(max_steps_per_episode)
                
                # 记录训练指标
                self.training_history.append(episode_metrics)
                
                # 更新最佳性能
                self._update_best_performance(episode_metrics)
                
                # 自适应参数调整
                self._update_adaptive_parameters(episode_metrics)
                
                # 定期保存
                if (episode + 1) % save_frequency == 0:
                    self._save_checkpoint(episode)
                
                # 定期评估
                if (episode + 1) % eval_frequency == 0:
                    eval_results = self._evaluate_model()
                    self.logger.info(f"回合 {episode}: 评估结果 = {eval_results}")
                
                # 日志输出
                if (episode + 1) % 20 == 0:
                    self._log_training_progress(episode_metrics)
                
                # 检查提前停止条件
                if self._should_early_stop():
                    self.logger.info(f"满足提前停止条件，在回合 {episode} 结束训练")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("下层训练被用户中断")
        except Exception as e:
            self.logger.error(f"下层训练过程中发生错误: {str(e)}")
            raise
        finally:
            self.training_state['is_training'] = False
            end_time = time.time()
            
            # 最终保存
            self._save_final_model()
            
            # 训练统计
            total_time = end_time - start_time
            training_stats = self._calculate_training_statistics(total_time)
            
            self.logger.info(f"下层训练完成! 总用时: {total_time:.2f}秒")
            
            return training_stats
    
    def _adjust_training_difficulty(self, episode: int):
        """调整训练难度"""
        if not self.adaptive_learning['curriculum_learning']:
            return
        
        # 课程学习：逐渐增加难度
        total_episodes = self.training_state['total_episodes']
        progress = episode / total_episodes
        
        # 难度等级：1.0（简单）到 3.0（困难）
        if progress < 0.3:
            difficulty = 1.0 + progress  # 1.0 -> 1.3
        elif progress < 0.7:
            difficulty = 1.3 + (progress - 0.3) * 1.75  # 1.3 -> 2.0
        else:
            difficulty = 2.0 + (progress - 0.7) * 3.33  # 2.0 -> 3.0
        
        self.adaptive_learning['difficulty_level'] = min(3.0, difficulty)
        
        # 更新环境参数
        self._update_environment_difficulty(difficulty)
    
    def _update_environment_difficulty(self, difficulty: float):
        """更新环境难度"""
        # 根据难度调整功率指令的复杂性和约束严格程度
        commands = {
            'power_command': 5000.0 * difficulty * np.sin(time.time() * 0.1),  # 动态功率指令
            'constraint_matrix': None,  # 约束矩阵将根据难度生成
            'weight_vector': [0.25, 0.25, 0.25, 0.25],
            'balance_targets': {
                'soc_target_std': max(1.0, 3.0 - difficulty),
                'temp_target_std': max(2.0, 8.0 - difficulty * 2)
            }
        }
        
        self.environment.update_upper_commands(commands)
    
    def _train_episode(self, max_steps: int) -> LowerTrainingMetrics:
        """训练一个回合"""
        episode_start_time = time.time()
        
        # 重置环境
        state = self.environment.reset()
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 初始化回合指标
        episode_metrics = LowerTrainingMetrics(
            episode=self.training_state['current_episode']
        )
        
        # 累积奖励和指标
        total_reward = 0.0
        objective_rewards = np.zeros(4)  # [tracking, response, constraint, smoothness]
        
        # 性能指标累积
        tracking_errors = []
        response_times = []
        constraint_violations = 0
        control_efforts = []
        
        step = 0
        
        while step < max_steps:
            control_start_time = time.time()
            
            # === 1. 动作选择 ===
            # 计算噪声缩放
            noise_scale = self._calculate_noise_scale(step)
            action = self.agent.select_action(state_tensor, add_noise=True, noise_scale=noise_scale)
            
            # === 2. 约束处理 ===
            system_state = self._state_to_dict(state_tensor)
            constraint_result = self.constraint_handler.handle_constraints(
                action, system_state
            )
            constrained_action = constraint_result['constrained_action']
            
            # === 3. 温度补偿 ===
            temperatures = system_state.get('temperatures', [25.0] * 10)
            temp_profile = self.temperature_compensator.analyze_temperature_profile(np.array(temperatures))
            
            if temp_profile.max_temperature > 40.0:  # 温度补偿阈值
                temp_prediction = self.temperature_compensator.predict_thermal_behavior(
                    temp_profile, system_state, 5000.0
                )
                temp_compensation = self.temperature_compensator.generate_compensation_action(
                    temp_profile, temp_prediction, {}
                )
                
                # 应用温度补偿
                constrained_action = self.temperature_compensator.apply_temperature_compensation(
                    constrained_action, temp_compensation, system_state
                )
                
                episode_metrics.temp_compensation_active = True
                episode_metrics.cooling_enhancement = temp_compensation.cooling_enhancement
            
            # === 4. 功率跟踪控制 ===
            power_command = system_state.get('power_command', 0.0)
            current_power = system_state.get('current_power', 0.0)
            
            tracking_result = self.power_tracker.track_power(
                power_command, current_power, system_state
            )
            
            # === 5. 响应优化 ===
            if step % 100 == 0:  # 每100步优化一次响应
                # 模拟阶跃响应数据
                input_signal = np.ones(50) * power_command
                output_signal = np.linspace(current_power, power_command, 50)
                time_vector = np.linspace(0, 0.5, 50)
                
                response_metrics = self.response_optimizer.measure_step_response(
                    input_signal, output_signal, time_vector
                )
                
                optimization_result = self.response_optimizer.optimize_response(
                    response_metrics, system_state
                )
                
                episode_metrics.response_time = response_metrics.response_time
                episode_metrics.stability_margin = response_metrics.overshoot
            
            # === 6. 环境交互 ===
            next_state, reward, done, info = self.environment.step(
                constrained_action.detach().cpu().numpy().squeeze()
            )
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # === 7. 计算分解奖励 ===
            decomposed_rewards = self._calculate_decomposed_rewards(info, constraint_result)
            
            # === 8. 更新智能体 ===
            self.agent.add_experience(
                state_tensor, constrained_action, reward, next_state_tensor, done
            )
            
            if len(self.agent.replay_buffer) > self.agent.batch_size:
                losses = self.agent.train()
                episode_metrics.actor_loss += losses.get('actor_loss', 0.0)
                episode_metrics.critic_loss += losses.get('critic_loss', 0.0)
                episode_metrics.q_value += losses.get('q_value', 0.0)
            
            control_time = time.time() - control_start_time
            episode_metrics.control_time += control_time
            
            # === 9. 累积指标 ===
            total_reward += reward
            objective_rewards += decomposed_rewards
            
            # 性能指标
            tracking_errors.append(info.get('power_tracking_error', 0.0))
            response_times.append(info.get('response_time', 0.05))
            constraint_violations += info.get('constraint_violations', 0)
            control_efforts.append(info.get('control_effort', 0.0))
            
            # 记录性能历史
            self.performance_history.append({
                'step': step,
                'tracking_error': info.get('power_tracking_error', 0.0),
                'response_time': info.get('response_time', 0.05),
                'constraint_violations': info.get('constraint_violations', 0),
                'tracking_accuracy': info.get('tracking_accuracy', 0.0)
            })
            
            # 更新状态
            state_tensor = next_state_tensor
            step += 1
            self.training_state['current_step'] = step
            self.training_state['total_steps'] += 1
            
            if done:
                break
        
        # === 计算回合指标 ===
        episode_metrics.step = step
        episode_metrics.total_reward = total_reward
        episode_metrics.power_tracking_reward = objective_rewards[0]
        episode_metrics.response_speed_reward = objective_rewards[1]
        episode_metrics.constraint_satisfaction_reward = objective_rewards[2]
        episode_metrics.control_smoothness_reward = objective_rewards[3]
        
        episode_metrics.power_tracking_error = np.mean(tracking_errors) if tracking_errors else 0.0
        episode_metrics.response_time = np.mean(response_times) if response_times else 0.05
        episode_metrics.constraint_violations = constraint_violations
        episode_metrics.control_effort = np.mean(control_efforts) if control_efforts else 0.0
        
        episode_metrics.tracking_accuracy = max(0.0, 1.0 - episode_metrics.power_tracking_error / 1000.0)
        episode_metrics.max_temperature = max(temperatures) if temperatures else 25.0
        
        return episode_metrics
    
    def _calculate_noise_scale(self, step: int) -> float:
        """计算噪声缩放"""
        if self.adaptive_learning['exploration_schedule'] == 'linear':
            # 线性衰减
            initial_noise = 1.0
            final_noise = 0.1
            decay_rate = (initial_noise - final_noise) / 10000  # 10000步衰减
            noise_scale = max(final_noise, initial_noise - step * decay_rate)
        
        elif self.adaptive_learning['exploration_schedule'] == 'exponential':
            # 指数衰减
            noise_scale = 1.0 * (self.adaptive_learning['noise_decay'] ** step)
        
        else:
            noise_scale = 0.5  # 固定噪声
        
        return noise_scale
    
    def _state_to_dict(self, state_tensor: torch.Tensor) -> Dict[str, Any]:
        """将状态张量转换为字典"""
        state_array = state_tensor.squeeze().cpu().numpy()
        
        return {
            'soc': state_array[0] * 100.0 if len(state_array) > 0 else 50.0,
            'temperature': state_array[1] * 60.0 if len(state_array) > 1 else 25.0,
            'voltage': state_array[2] * 4.2 if len(state_array) > 2 else 3.4,
            'current': state_array[3] * 200.0 if len(state_array) > 3 else 0.0,
            'current_power': state_array[4] * 50000.0 if len(state_array) > 4 else 0.0,
            'power_command': self.environment.upper_commands.get('power_command', 0.0),
            'temperatures': [25.0 + i for i in range(10)],  # 模拟温度数据
            'constraint_violations': 0
        }
    
    def _calculate_decomposed_rewards(self, 
                                    info: Dict[str, Any],
                                    constraint_result: Dict[str, Any]) -> np.ndarray:
        """计算分解奖励"""
        # 1. 功率跟踪奖励
        tracking_error = info.get('power_tracking_error', 0.0)
        tracking_reward = max(0.0, 1.0 - tracking_error / 1000.0)
        
        # 2. 响应速度奖励
        response_time = info.get('response_time', 0.05)
        target_response_time = 0.05  # 50ms目标
        response_reward = max(0.0, 1.0 - abs(response_time - target_response_time) / target_response_time)
        
        # 3. 约束满足奖励
        constraint_penalty = constraint_result.get('constraint_penalty', 0.0)
        constraint_reward = max(0.0, 1.0 - constraint_penalty / 100.0)
        
        # 4. 控制平滑性奖励
        control_effort = info.get('control_effort', 0.0)
        smoothness_reward = max(0.0, 1.0 - control_effort / 2.0)
        
        return np.array([tracking_reward, response_reward, constraint_reward, smoothness_reward])
    
    def _update_best_performance(self, metrics: LowerTrainingMetrics):
        """更新最佳性能"""
        # 更新最佳跟踪精度
        if metrics.tracking_accuracy > self.training_state['best_tracking_accuracy']:
            self.training_state['best_tracking_accuracy'] = metrics.tracking_accuracy
            self._save_best_model('best_tracking')
        
        # 更新最佳响应时间
        if metrics.response_time < self.training_state['best_response_time']:
            self.training_state['best_response_time'] = metrics.response_time
            self._save_best_model('best_response')
    
    def _update_adaptive_parameters(self, metrics: LowerTrainingMetrics):
        """更新自适应参数"""
        # 根据性能调整学习参数
        if metrics.tracking_accuracy > 0.9:
            # 性能良好，可以减少探索
            self.adaptive_learning['noise_decay'] = max(0.99, self.adaptive_learning['noise_decay'])
        elif metrics.tracking_accuracy < 0.6:
            # 性能不佳，增加探索
            self.adaptive_learning['noise_decay'] = min(0.995, self.adaptive_learning['noise_decay'])
        
        # 根据约束违反情况调整
        if metrics.constraint_violations > 5:
            # 约束违反过多，需要更保守的策略
            self.agent.noise.sigma = max(0.05, self.agent.noise.sigma * 0.95)
    
    def _evaluate_model(self) -> Dict[str, float]:
        """评估模型"""
        eval_episodes = 10
        eval_results = {
            'tracking_errors': [],
            'response_times': [],
            'constraint_violations': [],
            'tracking_accuracies': [],
            'total_rewards': []
        }
        
        for _ in range(eval_episodes):
            state = self.environment.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            episode_reward = 0.0
            episode_tracking_errors = []
            episode_response_times = []
            episode_violations = 0
            episode_accuracies = []
            
            for step in range(1000):  # 1000步评估
                # 使用确定性策略
                action = self.agent.select_action(state_tensor, add_noise=False)
                
                # 约束处理
                system_state = self._state_to_dict(state_tensor)
                constraint_result = self.constraint_handler.handle_constraints(action, system_state)
                constrained_action = constraint_result['constrained_action']
                
                # 环境交互
                next_state, reward, done, info = self.environment.step(
                    constrained_action.detach().cpu().numpy().squeeze()
                )
                
                episode_reward += reward
                episode_tracking_errors.append(info.get('power_tracking_error', 0.0))
                episode_response_times.append(info.get('response_time', 0.05))
                episode_violations += info.get('constraint_violations', 0)
                episode_accuracies.append(info.get('tracking_accuracy', 0.0))
                
                state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                
                if done:
                    break
            
            eval_results['total_rewards'].append(episode_reward)
            eval_results['tracking_errors'].append(np.mean(episode_tracking_errors))
            eval_results['response_times'].append(np.mean(episode_response_times))
            eval_results['constraint_violations'].append(episode_violations)
            eval_results['tracking_accuracies'].append(np.mean(episode_accuracies))
        
        return {
            'mean_total_reward': np.mean(eval_results['total_rewards']),
            'mean_tracking_error': np.mean(eval_results['tracking_errors']),
            'mean_response_time': np.mean(eval_results['response_times']),
            'mean_constraint_violations': np.mean(eval_results['constraint_violations']),
            'mean_tracking_accuracy': np.mean(eval_results['tracking_accuracies']),
            'std_tracking_error': np.std(eval_results['tracking_errors']),
            'tracking_error_95th': np.percentile(eval_results['tracking_errors'], 95)
        }
    
    def _should_early_stop(self) -> bool:
        """检查是否应该提前停止"""
        if len(self.performance_history) < 500:
            return False
        
        # 检查最近500步的跟踪精度是否停滞
        recent_performance = list(self.performance_history)[-500:]
        recent_accuracies = [p['tracking_accuracy'] for p in recent_performance]
        
        if len(recent_accuracies) >= 250:
            first_half = np.mean(recent_accuracies[:250])
            second_half = np.mean(recent_accuracies[250:])
            
            # 如果精度改善小于1%，认为停滞
            improvement = (second_half - first_half) / max(abs(first_half), 1e-6)
            return improvement < 0.01
        
        return False
    
    def _log_training_progress(self, metrics: LowerTrainingMetrics):
        """记录训练进度"""
        self.logger.info(
            f"回合 {metrics.episode}: "
            f"总奖励={metrics.total_reward:.4f}, "
            f"跟踪精度={metrics.tracking_accuracy:.3f}, "
            f"响应时间={metrics.response_time*1000:.1f}ms, "
            f"跟踪误差={metrics.power_tracking_error:.1f}W, "
            f"约束违反={metrics.constraint_violations}, "
            f"温度补偿={'启用' if metrics.temp_compensation_active else '禁用'}"
        )
    
    def _save_checkpoint(self, episode: int) -> str:
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}.pth")
        
        checkpoint = {
            'episode': episode,
            'training_state': self.training_state,
            'agent_state': self.agent.state_dict(),
            'power_tracker_state': self.power_tracker.get_tracker_statistics(),
            'constraint_handler_state': self.constraint_handler.get_constraint_status(),
            'temperature_compensator_state': self.temperature_compensator.get_compensator_statistics(),
            'response_optimizer_state': self.response_optimizer.get_optimizer_statistics(),
            'adaptive_learning': self.adaptive_learning,
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
                'best_tracking_accuracy': self.training_state['best_tracking_accuracy'],
                'best_response_time': self.training_state['best_response_time']
            },
            'agent_state': self.agent.state_dict(),
            'config': self.config
        }
        
        torch.save(best_model, best_model_path)
        self.logger.info(f"最佳模型已保存: {best_model_path} ({model_type})")
        
        return best_model_path
    
    def _save_final_model(self):
        """保存最终模型"""
        # 保存最终检查点
        final_checkpoint = self._save_checkpoint(self.training_state['current_episode'])
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, "lower_training_history.json")
        with open(history_path, 'w') as f:
            serializable_history = []
            for metrics in self.training_history:
                serializable_history.append({
                    'episode': metrics.episode,
                    'total_reward': metrics.total_reward,
                    'tracking_accuracy': metrics.tracking_accuracy,
                    'power_tracking_error': metrics.power_tracking_error,
                    'response_time': metrics.response_time,
                    'constraint_violations': metrics.constraint_violations,
                    'actor_loss': metrics.actor_loss,
                    'critic_loss': metrics.critic_loss
                })
            json.dump(serializable_history, f, indent=2)
        
        # 保存性能历史
        performance_path = os.path.join(self.save_dir, "performance_history.json")
        with open(performance_path, 'w') as f:
            json.dump(list(self.performance_history), f, indent=2)
        
        self.logger.info(f"下层训练数据已保存: {self.save_dir}")
    
    def _calculate_training_statistics(self, total_time: float) -> Dict[str, Any]:
        """计算训练统计信息"""
        if not self.training_history:
            return {}
        
        # 基础统计
        total_rewards = [m.total_reward for m in self.training_history]
        tracking_accuracies = [m.tracking_accuracy for m in self.training_history]
        response_times = [m.response_time for m in self.training_history]
        tracking_errors = [m.power_tracking_error for m in self.training_history]
        
        stats = {
            'training_summary': {
                'total_episodes': len(self.training_history),
                'total_steps': self.training_state['total_steps'],
                'total_time': total_time,
                'avg_episode_time': total_time / len(self.training_history),
                'best_tracking_accuracy': self.training_state['best_tracking_accuracy'],
                'best_response_time': self.training_state['best_response_time']
            },
            
            'reward_statistics': {
                'mean_total_reward': np.mean(total_rewards),
                'std_total_reward': np.std(total_rewards),
                'max_total_reward': np.max(total_rewards),
                'min_total_reward': np.min(total_rewards)
            },
            
            'control_performance': {
                'mean_tracking_accuracy': np.mean(tracking_accuracies),
                'std_tracking_accuracy': np.std(tracking_accuracies),
                'mean_response_time': np.mean(response_times),
                'std_response_time': np.std(response_times),
                'mean_tracking_error': np.mean(tracking_errors),
                'tracking_error_95th': np.percentile(tracking_errors, 95)
            },
            
            'training_efficiency': {
                'convergence_episode': self._find_convergence_episode(),
                'final_performance_trend': self._calculate_final_trend(),
                'avg_steps_per_episode': np.mean([m.step for m in self.training_history])
            }
        }
        
        return stats
    
    def _find_convergence_episode(self) -> int:
        """寻找收敛回合"""
        if len(self.training_history) < 50:
            return -1
        
        # 基于跟踪精度的收敛检测
        window_size = 25
        stability_threshold = 0.02  # 2%变异系数
        
        for i in range(window_size, len(self.training_history)):
            window_accuracies = [
                metrics.tracking_accuracy 
                for metrics in self.training_history[i-window_size:i]
            ]
            
            mean_accuracy = np.mean(window_accuracies)
            std_accuracy = np.std(window_accuracies)
            
            if mean_accuracy > 0 and (std_accuracy / mean_accuracy) < stability_threshold:
                return i - window_size
        
        return -1
    
    def _calculate_final_trend(self) -> str:
        """计算最终趋势"""
        if len(self.training_history) < 20:
            return "insufficient_data"
        
        recent_accuracies = [metrics.tracking_accuracy for metrics in self.training_history[-20:]]
        
        # 线性拟合
        x = np.arange(len(recent_accuracies))
        slope = np.polyfit(x, recent_accuracies, 1)[0]
        
        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "declining"
        else:
            return "stable"
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.training_state = checkpoint['training_state']
            self.agent.load_state_dict(checkpoint['agent_state'])
            self.adaptive_learning = checkpoint['adaptive_learning']
            self.training_history = checkpoint['training_history']
            
            self.logger.info(f"下层检查点加载成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"下层检查点加载失败: {str(e)}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        recent_performance = list(self.performance_history)[-10:] if self.performance_history else []
        
        return {
            'trainer_id': self.trainer_id,
            'training_state': self.training_state.copy(),
            'adaptive_learning': self.adaptive_learning.copy(),
            'current_noise_scale': self.agent.noise.sigma,
            'recent_performance': recent_performance,
            'component_status': {
                'power_tracker': self.power_tracker.get_tracker_statistics(),
                'constraint_handler': self.constraint_handler.get_constraint_status(),
                'temperature_compensator': self.temperature_compensator.get_compensator_statistics(),
                'response_optimizer': self.response_optimizer.get_optimizer_statistics()
            }
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"LowerLayerTrainer({self.trainer_id}): "
                f"回合={self.training_state['current_episode']}/{self.training_state['total_episodes']}, "
                f"步数={self.training_state['total_steps']}, "
                f"训练中={self.training_state['is_training']}")
