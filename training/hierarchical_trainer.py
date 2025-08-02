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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.training_config import TrainingConfig, UpperLayerConfig, LowerLayerConfig
from config.model_config import ModelConfig
from .upper_trainer import UpperLayerTrainer
from .lower_trainer import LowerLayerTrainer
from drl_agents.communication.message_protocol import MessageProtocol, MessageType, Priority
from drl_agents.communication.information_flow import InformationFlow
from drl_agents.communication.data_exchange import DataExchange

@dataclass
class HierarchicalMetrics:
    """分层训练指标"""
    episode: int = 0
    
    # 上层指标
    upper_reward: float = 0.0
    upper_hypervolume: float = 0.0
    upper_pareto_size: int = 0
    soc_balance_score: float = 0.0
    temp_balance_score: float = 0.0
    
    # 下层指标
    lower_reward: float = 0.0
    tracking_accuracy: float = 0.0
    response_time: float = 0.0
    constraint_violations: int = 0
    
    # 协调指标
    communication_success_rate: float = 0.0
    synchronization_delay: float = 0.0
    coordination_efficiency: float = 0.0
    
    # 联合性能
    overall_performance: float = 0.0
    system_stability: float = 0.0
    energy_efficiency: float = 0.0
    
    # 时间指标
    upper_training_time: float = 0.0
    lower_training_time: float = 0.0
    coordination_time: float = 0.0

class CoordinationManager:
    """协调管理器"""
    
    def __init__(self, manager_id: str = "CoordManager_001"):
        self.manager_id = manager_id
        
        # 协调策略
        self.coordination_strategy = {
            'sync_frequency': 'adaptive',  # 'fixed', 'adaptive', 'event_driven'
            'sync_interval': 50,           # 基础同步间隔（回合数）
            'performance_threshold': 0.8,  # 性能阈值
            'coordination_weight': 0.3     # 协调权重
        }
        
        # 协调状态
        self.coordination_state = {
            'last_sync_episode': 0,
            'sync_count': 0,
            'performance_history': deque(maxlen=100),
            'coordination_quality': 0.0
        }
        
        # 信息交换缓冲区
        self.upper_to_lower_buffer = deque(maxlen=1000)
        self.lower_to_upper_buffer = deque(maxlen=1000)
        
    def should_synchronize(self, 
                          episode: int,
                          upper_performance: Dict[str, float],
                          lower_performance: Dict[str, float]) -> bool:
        """判断是否应该同步"""
        if self.coordination_strategy['sync_frequency'] == 'fixed':
            return episode % self.coordination_strategy['sync_interval'] == 0
        
        elif self.coordination_strategy['sync_frequency'] == 'adaptive':
            # 基于性能自适应同步
            episodes_since_sync = episode - self.coordination_state['last_sync_episode']
            
            if episodes_since_sync >= self.coordination_strategy['sync_interval']:
                return True
            
            # 性能下降时增加同步频率
            if (upper_performance.get('total_reward', 0) < 0.5 or 
                lower_performance.get('tracking_accuracy', 0) < 0.7):
                return episodes_since_sync >= self.coordination_strategy['sync_interval'] // 2
            
        elif self.coordination_strategy['sync_frequency'] == 'event_driven':
            # 基于事件驱动同步
            if (len(self.upper_to_lower_buffer) > 100 or 
                len(self.lower_to_upper_buffer) > 100):
                return True
        
        return False
    
    def coordinate_training(self,
                          upper_info: Dict[str, Any],
                          lower_info: Dict[str, Any]) -> Dict[str, Any]:
        """协调训练"""
        coordination_start_time = time.time()
        
        # 信息交换
        upper_to_lower_info = self._extract_upper_to_lower_info(upper_info)
        lower_to_upper_info = self._extract_lower_to_upper_info(lower_info)
        
        # 添加到缓冲区
        self.upper_to_lower_buffer.append(upper_to_lower_info)
        self.lower_to_upper_buffer.append(lower_to_upper_info)
        
        # 计算协调质量
        coordination_quality = self._calculate_coordination_quality(upper_info, lower_info)
        self.coordination_state['coordination_quality'] = coordination_quality
        
        # 生成协调指令
        coordination_commands = self._generate_coordination_commands(
            upper_to_lower_info, lower_to_upper_info
        )
        
        coordination_time = time.time() - coordination_start_time
        
        # 更新协调状态
        self.coordination_state['sync_count'] += 1
        
        return {
            'upper_commands': coordination_commands.get('to_upper', {}),
            'lower_commands': coordination_commands.get('to_lower', {}),
            'coordination_quality': coordination_quality,
            'coordination_time': coordination_time,
            'sync_success': True
        }
    
    def _extract_upper_to_lower_info(self, upper_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取上层到下层的信息"""
        return {
            'constraint_matrix': upper_info.get('constraint_matrix'),
            'objective_weights': upper_info.get('objective_weights', [0.25, 0.25, 0.25, 0.25]),
            'balance_targets': upper_info.get('balance_targets', {}),
            'power_commands': upper_info.get('power_commands', []),
            'performance_feedback': {
                'hypervolume': upper_info.get('hypervolume', 0.0),
                'pareto_size': upper_info.get('pareto_size', 0),
                'soc_balance': upper_info.get('soc_balance_score', 0.0),
                'temp_balance': upper_info.get('temp_balance_score', 0.0)
            }
        }
    
    def _extract_lower_to_upper_info(self, lower_info: Dict[str, Any]) -> Dict[str, Any]:
        """提取下层到上层的信息"""
        return {
            'tracking_performance': {
                'accuracy': lower_info.get('tracking_accuracy', 0.0),
                'error': lower_info.get('tracking_error', 0.0),
                'response_time': lower_info.get('response_time', 0.05)
            },
            'constraint_status': {
                'violations': lower_info.get('constraint_violations', 0),
                'satisfaction_rate': lower_info.get('constraint_satisfaction_rate', 1.0)
            },
            'control_status': {
                'control_effort': lower_info.get('control_effort', 0.0),
                'stability_margin': lower_info.get('stability_margin', 0.0),
                'temperature_status': lower_info.get('temperature_status', {})
            },
            'system_health': {
                'component_status': lower_info.get('component_status', {}),
                'performance_trend': lower_info.get('performance_trend', 'stable')
            }
        }
    
    def _calculate_coordination_quality(self,
                                      upper_info: Dict[str, Any],
                                      lower_info: Dict[str, Any]) -> float:
        """计算协调质量"""
        # 上层性能质量
        upper_quality = (
            upper_info.get('hypervolume', 0.0) * 0.3 +
            upper_info.get('soc_balance_score', 0.0) * 0.25 +
            upper_info.get('temp_balance_score', 0.0) * 0.25 +
            (1.0 - upper_info.get('constraint_violations', 0) * 0.1) * 0.2
        )
        
        # 下层性能质量
        lower_quality = (
            lower_info.get('tracking_accuracy', 0.0) * 0.4 +
            (1.0 - min(1.0, lower_info.get('response_time', 0.05) / 0.1)) * 0.3 +
            (1.0 - lower_info.get('constraint_violations', 0) * 0.1) * 0.3
        )
        
        # 协调一致性
        consistency = self._calculate_consistency(upper_info, lower_info)
        
        # 综合协调质量
        coordination_quality = 0.4 * upper_quality + 0.4 * lower_quality + 0.2 * consistency
        
        return max(0.0, min(1.0, coordination_quality))
    
    def _calculate_consistency(self,
                             upper_info: Dict[str, Any],
                             lower_info: Dict[str, Any]) -> float:
        """计算上下层一致性"""
        # 目标一致性：上层目标与下层执行的一致性
        target_power = upper_info.get('target_power', 0.0)
        actual_power = lower_info.get('actual_power', 0.0)
        
        if target_power != 0:
            power_consistency = 1.0 - abs(target_power - actual_power) / abs(target_power)
        else:
            power_consistency = 1.0
        
        # 约束一致性：上层约束与下层满足情况的一致性
        upper_constraints = upper_info.get('constraint_severity', 0.0)
        lower_violations = lower_info.get('constraint_violations', 0)
        
        constraint_consistency = 1.0 - min(1.0, lower_violations * 0.2)
        
        # 时间一致性：决策时间与执行时间的匹配
        decision_time = upper_info.get('decision_time', 300.0)  # 5分钟
        execution_time = lower_info.get('avg_execution_time', 0.01)  # 10ms
        
        time_ratio = execution_time / (decision_time / 30000)  # 期望比例
        time_consistency = 1.0 - abs(1.0 - time_ratio)
        
        # 综合一致性
        consistency = 0.5 * power_consistency + 0.3 * constraint_consistency + 0.2 * time_consistency
        
        return max(0.0, min(1.0, consistency))
    
    def _generate_coordination_commands(self,
                                      upper_to_lower: Dict[str, Any],
                                      lower_to_upper: Dict[str, Any]) -> Dict[str, Any]:
        """生成协调指令"""
        commands = {
            'to_upper': {},
            'to_lower': {}
        }
        
        # 给上层的指令
        tracking_performance = lower_to_upper.get('tracking_performance', {})
        if tracking_performance.get('accuracy', 0) < 0.7:
            # 跟踪性能不佳，建议调整上层目标
            commands['to_upper']['adjust_targets'] = {
                'power_command_adjustment': -0.1,
                'constraint_relaxation': 0.1
            }
        
        # 给下层的指令
        performance_feedback = upper_to_lower.get('performance_feedback', {})
        if performance_feedback.get('hypervolume', 0) < 0.5:
            # 上层性能不佳，建议增强下层跟踪
            commands['to_lower']['enhance_tracking'] = {
                'increase_learning_rate': 0.1,
                'reduce_exploration': 0.2
            }
        
        # 约束协调
        constraint_status = lower_to_upper.get('constraint_status', {})
        if constraint_status.get('violations', 0) > 5:
            commands['to_upper']['tighten_constraints'] = True
            commands['to_lower']['increase_constraint_weight'] = 0.2
        
        return commands

class HierarchicalTrainer:
    """
    分层联合训练器
    协调上下层DRL的联合训练
    """
    
    def __init__(self,
                 config: TrainingConfig,
                 model_config: ModelConfig,
                 trainer_id: str = "HierarchicalTrainer_001"):
        """
        初始化分层联合训练器
        
        Args:
            config: 训练配置
            model_config: 模型配置
            trainer_id: 训练器ID
        """
        self.config = config
        self.model_config = model_config
        self.trainer_id = trainer_id
        
        # === 初始化子训练器 ===
        self.upper_trainer = UpperLayerTrainer(
            config=config.upper_config,
            model_config=model_config,
            trainer_id=f"Upper_{trainer_id}"
        )
        
        self.lower_trainer = LowerLayerTrainer(
            config=config.lower_config,
            model_config=model_config,
            trainer_id=f"Lower_{trainer_id}"
        )
        
        # === 初始化协调管理器 ===
        self.coordination_manager = CoordinationManager(f"Coord_{trainer_id}")
        
        # === 初始化通信系统 ===
        self.message_protocol = MessageProtocol(node_id=f"Hierarchical_{trainer_id}")
        self.information_flow = InformationFlow(
            flow_id=f"HierarchicalFlow_{trainer_id}",
            message_protocol=self.message_protocol
        )
        self.data_exchange = DataExchange(
            exchange_id=f"HierarchicalExchange_{trainer_id}",
            message_protocol=self.message_protocol,
            information_flow=self.information_flow
        )
        
        # === 训练模式 ===
        self.training_modes = {
            'sequential': False,      # 顺序训练：先上层后下层
            'parallel': True,         # 并行训练：同时训练
            'alternating': False,     # 交替训练：轮流训练
            'adaptive': False         # 自适应训练：根据性能选择
        }
        
        # === 训练状态 ===
        self.training_state = {
            'current_episode': 0,
            'total_episodes': config.max_episodes,
            'is_training': False,
            'training_mode': 'parallel',
            'sync_episode': 0,
            'best_overall_performance': 0.0
        }
        
        # === 训练历史 ===
        self.training_history: List[HierarchicalMetrics] = []
        self.coordination_history: List[Dict] = []
        
        # === 线程池 ===
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # === 日志设置 ===
        self._setup_logging()
        
        # === 保存路径 ===
        self.save_dir = f"checkpoints/hierarchical/{trainer_id}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"✅ 分层联合训练器初始化完成: {trainer_id}")
        print(f"   训练模式: {'并行' if self.training_modes['parallel'] else '顺序'}")
        print(f"   协调策略: {self.coordination_manager.coordination_strategy['sync_frequency']}")
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = f"logs/hierarchical/{self.trainer_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/hierarchical_training.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"HierarchicalTrainer_{self.trainer_id}")
    
    def train(self,
             max_episodes: Optional[int] = None,
             save_frequency: int = 50,
             eval_frequency: int = 25,
             sync_frequency: Optional[int] = None) -> Dict[str, Any]:
        """
        开始分层联合训练
        
        Args:
            max_episodes: 最大训练回合数
            save_frequency: 保存频率
            eval_frequency: 评估频率
            sync_frequency: 同步频率
            
        Returns:
            训练结果统计
        """
        if max_episodes is None:
            max_episodes = self.config.max_episodes
        
        if sync_frequency is not None:
            self.coordination_manager.coordination_strategy['sync_interval'] = sync_frequency
        
        self.training_state['is_training'] = True
        self.training_state['total_episodes'] = max_episodes
        
        self.logger.info(f"开始分层联合训练: 目标回合数={max_episodes}")
        self.logger.info(f"训练模式: {self.training_state['training_mode']}")
        
        start_time = time.time()
        
        try:
            # 启动通信系统
            self.information_flow.start_flow_processing()
            
            if self.training_modes['parallel']:
                # 并行训练模式
                training_stats = self._parallel_training(max_episodes, save_frequency, eval_frequency)
            elif self.training_modes['sequential']:
                # 顺序训练模式
                training_stats = self._sequential_training(max_episodes, save_frequency, eval_frequency)
            elif self.training_modes['alternating']:
                # 交替训练模式
                training_stats = self._alternating_training(max_episodes, save_frequency, eval_frequency)
            else:
                # 自适应训练模式
                training_stats = self._adaptive_training(max_episodes, save_frequency, eval_frequency)
        
        except KeyboardInterrupt:
            self.logger.info("分层训练被用户中断")
            training_stats = {}
        except Exception as e:
            self.logger.error(f"分层训练过程中发生错误: {str(e)}")
            raise
        finally:
            self.training_state['is_training'] = False
            self.information_flow.stop_flow_processing()
            self.thread_pool.shutdown(wait=True)
            
            end_time = time.time()
            
            # 最终保存
            self._save_final_model()
            
            # 训练统计
            total_time = end_time - start_time
            if not training_stats:
                training_stats = self._calculate_training_statistics(total_time)
            
            self.logger.info(f"分层训练完成! 总用时: {total_time:.2f}秒")
            
            return training_stats
    
    def _parallel_training(self,
                          max_episodes: int,
                          save_frequency: int,
                          eval_frequency: int) -> Dict[str, Any]:
        """并行训练模式"""
        self.logger.info("启动并行训练模式")
        
        for episode in range(max_episodes):
            self.training_state['current_episode'] = episode
            
            # 并行训练一个回合
            episode_metrics = self._parallel_train_episode()
            
            # 记录训练指标
            self.training_history.append(episode_metrics)
            
            # 更新最佳性能
            self._update_best_performance(episode_metrics)
            
            # 协调检查
            if self._should_coordinate(episode):
                coordination_result = self._coordinate_layers(episode)
                self.coordination_history.append(coordination_result)
                self.training_state['sync_episode'] = episode
            
            # 定期保存和评估
            if (episode + 1) % save_frequency == 0:
                self._save_checkpoint(episode)
            
            if (episode + 1) % eval_frequency == 0:
                eval_results = self._evaluate_hierarchical_model()
                self.logger.info(f"回合 {episode}: 联合评估结果 = {eval_results}")
            
            # 日志输出
            if (episode + 1) % 10 == 0:
                self._log_training_progress(episode_metrics)
            
            # 检查提前停止条件
            if self._should_early_stop():
                self.logger.info(f"满足提前停止条件，在回合 {episode} 结束训练")
                break
        
        return self._calculate_training_statistics(time.time())
    
    def _parallel_train_episode(self) -> HierarchicalMetrics:
        """并行训练一个回合"""
        episode_start_time = time.time()
        episode = self.training_state['current_episode']
        
        # 初始化回合指标
        metrics = HierarchicalMetrics(episode=episode)
        
        # 并行执行上下层训练
        upper_future = self.thread_pool.submit(self._train_upper_episode_step)
        lower_future = self.thread_pool.submit(self._train_lower_episode_step)
        
        try:
            # 等待训练完成
            upper_result = upper_future.result(timeout=300)  # 5分钟超时
            lower_result = lower_future.result(timeout=300)
            
            # 协调时间计算
            coordination_start_time = time.time()
            
            # 信息交换
            self._exchange_information(upper_result, lower_result)
            
            coordination_time = time.time() - coordination_start_time
            
            # 合并指标
            metrics.upper_reward = upper_result.get('total_reward', 0.0)
            metrics.upper_hypervolume = upper_result.get('hypervolume', 0.0)
            metrics.upper_pareto_size = upper_result.get('pareto_size', 0)
            metrics.soc_balance_score = upper_result.get('soc_balance_score', 0.0)
            metrics.temp_balance_score = upper_result.get('temp_balance_score', 0.0)
            
            metrics.lower_reward = lower_result.get('total_reward', 0.0)
            metrics.tracking_accuracy = lower_result.get('tracking_accuracy', 0.0)
            metrics.response_time = lower_result.get('response_time', 0.05)
            metrics.constraint_violations = lower_result.get('constraint_violations', 0)
            
            # 协调指标
            metrics.communication_success_rate = self._calculate_communication_success_rate()
            metrics.synchronization_delay = coordination_time
            metrics.coordination_efficiency = self._calculate_coordination_efficiency(upper_result, lower_result)
            
            # 联合性能
            metrics.overall_performance = self._calculate_overall_performance(upper_result, lower_result)
            metrics.system_stability = self._calculate_system_stability(upper_result, lower_result)
            metrics.energy_efficiency = self._calculate_energy_efficiency(upper_result, lower_result)
            
            # 时间指标
            metrics.upper_training_time = upper_result.get('training_time', 0.0)
            metrics.lower_training_time = lower_result.get('training_time', 0.0)
            metrics.coordination_time = coordination_time
            
        except Exception as e:
            self.logger.error(f"并行训练回合失败: {str(e)}")
            # 返回默认指标
            metrics.overall_performance = 0.0
        
        return metrics
    
    def _train_upper_episode_step(self) -> Dict[str, Any]:
        """训练上层单步"""
        try:
            # 模拟上层训练步骤
            episode_metrics = self.upper_trainer._train_episode()
            
            return {
                'total_reward': episode_metrics.total_reward,
                'hypervolume': episode_metrics.hypervolume,
                'pareto_size': episode_metrics.pareto_front_size,
                'soc_balance_score': episode_metrics.soc_balance_reward,
                'temp_balance_score': episode_metrics.temp_balance_reward,
                'constraint_matrix': torch.randn(3, 4),  # 模拟约束矩阵
                'objective_weights': [0.25, 0.25, 0.25, 0.25],
                'training_time': episode_metrics.training_time,
                'decision_time': episode_metrics.decision_time
            }
            
        except Exception as e:
            self.logger.error(f"上层训练步骤失败: {str(e)}")
            return {'total_reward': 0.0, 'training_time': 0.0}
    
    def _train_lower_episode_step(self) -> Dict[str, Any]:
        """训练下层单步"""
        try:
            # 模拟下层训练步骤
            episode_metrics = self.lower_trainer._train_episode(1000)  # 1000步
            
            return {
                'total_reward': episode_metrics.total_reward,
                'tracking_accuracy': episode_metrics.tracking_accuracy,
                'response_time': episode_metrics.response_time,
                'constraint_violations': episode_metrics.constraint_violations,
                'control_effort': episode_metrics.control_effort,
                'training_time': episode_metrics.control_time,
                'performance_feedback': {
                    'tracking_error': episode_metrics.power_tracking_error,
                    'stability_margin': episode_metrics.stability_margin
                }
            }
            
        except Exception as e:
            self.logger.error(f"下层训练步骤失败: {str(e)}")
            return {'total_reward': 0.0, 'training_time': 0.0}
    
    def _sequential_training(self,
                           max_episodes: int,
                           save_frequency: int,
                           eval_frequency: int) -> Dict[str, Any]:
        """顺序训练模式"""
        self.logger.info("启动顺序训练模式")
        
        # 先训练上层
        self.logger.info("开始上层预训练...")
        upper_episodes = max_episodes // 3
        upper_stats = self.upper_trainer.train(upper_episodes)
        
        # 再训练下层
        self.logger.info("开始下层预训练...")
        lower_episodes = max_episodes // 3
        lower_stats = self.lower_trainer.train(lower_episodes)
        
        # 最后联合微调
        self.logger.info("开始联合微调...")
        joint_episodes = max_episodes - upper_episodes - lower_episodes
        joint_stats = self._parallel_training(joint_episodes, save_frequency, eval_frequency)
        
        # 合并统计
        combined_stats = {
            'sequential_training': True,
            'upper_pretraining': upper_stats,
            'lower_pretraining': lower_stats,
            'joint_finetuning': joint_stats
        }
        
        return combined_stats
    
    def _alternating_training(self,
                            max_episodes: int,
                            save_frequency: int,
                            eval_frequency: int) -> Dict[str, Any]:
        """交替训练模式"""
        self.logger.info("启动交替训练模式")
        
        for episode in range(max_episodes):
            self.training_state['current_episode'] = episode
            
            if episode % 2 == 0:
                # 偶数回合训练上层
                upper_result = self._train_upper_episode_step()
                lower_result = {'total_reward': 0.0, 'training_time': 0.0}  # 下层不训练
            else:
                # 奇数回合训练下层
                lower_result = self._train_lower_episode_step()
                upper_result = {'total_reward': 0.0, 'training_time': 0.0}  # 上层不训练
            
            # 创建回合指标
            metrics = HierarchicalMetrics(episode=episode)
            metrics.upper_reward = upper_result.get('total_reward', 0.0)
            metrics.lower_reward = lower_result.get('total_reward', 0.0)
            metrics.overall_performance = (metrics.upper_reward + metrics.lower_reward) / 2
            
            self.training_history.append(metrics)
            
            # 定期同步
            if episode % 10 == 9:  # 每10回合同步一次
                coordination_result = self._coordinate_layers(episode)
                self.coordination_history.append(coordination_result)
            
            # 定期保存和评估
            if (episode + 1) % save_frequency == 0:
                self._save_checkpoint(episode)
            
            if (episode + 1) % eval_frequency == 0:
                eval_results = self._evaluate_hierarchical_model()
                self.logger.info(f"回合 {episode}: 交替评估结果 = {eval_results}")
        
        return self._calculate_training_statistics(time.time())
    
    def _adaptive_training(self,
                          max_episodes: int,
                          save_frequency: int,
                          eval_frequency: int) -> Dict[str, Any]:
        """自适应训练模式"""
        self.logger.info("启动自适应训练模式")
        
        # 初始性能评估
        upper_performance = 0.5
        lower_performance = 0.5
        
        for episode in range(max_episodes):
            self.training_state['current_episode'] = episode
            
            # 根据性能选择训练策略
            if upper_performance < 0.6 and lower_performance < 0.6:
                # 两者都需要改进，并行训练
                metrics = self._parallel_train_episode()
            elif upper_performance < lower_performance:
                # 上层需要更多训练
                upper_result = self._train_upper_episode_step()
                lower_result = {'total_reward': lower_performance, 'training_time': 0.0}
                metrics = self._create_metrics_from_results(episode, upper_result, lower_result)
            else:
                # 下层需要更多训练
                lower_result = self._train_lower_episode_step()
                upper_result = {'total_reward': upper_performance, 'training_time': 0.0}
                metrics = self._create_metrics_from_results(episode, upper_result, lower_result)
            
            self.training_history.append(metrics)
            
            # 更新性能评估
            if len(self.training_history) >= 10:
                recent_metrics = self.training_history[-10:]
                upper_performance = np.mean([m.upper_reward for m in recent_metrics])
                lower_performance = np.mean([m.lower_reward for m in recent_metrics])
            
            # 协调检查
            if self._should_coordinate(episode):
                coordination_result = self._coordinate_layers(episode)
                self.coordination_history.append(coordination_result)
        
        return self._calculate_training_statistics(time.time())
    
    def _should_coordinate(self, episode: int) -> bool:
        """判断是否应该协调"""
        # 获取最近的上下层性能
        if len(self.training_history) < 5:
            return False
        
        recent_metrics = self.training_history[-5:]
        upper_performance = np.mean([m.upper_reward for m in recent_metrics])
        lower_performance = np.mean([m.lower_reward for m in recent_metrics])
        
        return self.coordination_manager.should_synchronize(
            episode,
            {'total_reward': upper_performance},
            {'tracking_accuracy': lower_performance}
        )
    
    def _coordinate_layers(self, episode: int) -> Dict[str, Any]:
        """协调上下层"""
        coordination_start_time = time.time()
        
        # 获取上下层信息
        upper_info = self.upper_trainer.get_training_status()
        lower_info = self.lower_trainer.get_training_status()
        
        # 执行协调
        coordination_result = self.coordination_manager.coordinate_training(upper_info, lower_info)
        
        # 应用协调指令
        self._apply_coordination_commands(coordination_result)
        
        coordination_time = time.time() - coordination_start_time
        
        coordination_record = {
            'episode': episode,
            'coordination_time': coordination_time,
            'coordination_quality': coordination_result.get('coordination_quality', 0.0),
            'upper_commands': coordination_result.get('upper_commands', {}),
            'lower_commands': coordination_result.get('lower_commands', {}),
            'sync_success': coordination_result.get('sync_success', False)
        }
        
        self.logger.info(f"回合 {episode}: 层间协调完成，质量={coordination_record['coordination_quality']:.3f}")
        
        return coordination_record
    
    def _apply_coordination_commands(self, coordination_result: Dict[str, Any]):
        """应用协调指令"""
        upper_commands = coordination_result.get('upper_commands', {})
        lower_commands = coordination_result.get('lower_commands', {})
        
        # 应用上层指令
        if 'adjust_targets' in upper_commands:
            # 调整上层目标
            adjustment = upper_commands['adjust_targets']
            self.logger.info(f"应用上层调整指令: {adjustment}")
        
        # 应用下层指令
        if 'enhance_tracking' in lower_commands:
            # 增强下层跟踪
            enhancement = lower_commands['enhance_tracking']
            if 'increase_learning_rate' in enhancement:
                # 增加学习率
                for param_group in self.lower_trainer.agent.actor_optimizer.param_groups:
                    param_group['lr'] *= (1 + enhancement['increase_learning_rate'])
            
            self.logger.info(f"应用下层增强指令: {enhancement}")
    
    def _exchange_information(self, upper_result: Dict, lower_result: Dict):
        """交换上下层信息"""
        try:
            # 上层到下层的信息
            if 'constraint_matrix' in upper_result:
                constraint_matrix = upper_result['constraint_matrix']
                self.data_exchange.exchange_constraint_matrix(
                    constraint_matrix=constraint_matrix,
                    target_node="lower_layer"
                )
            
            # 下层到上层的信息
            if 'performance_feedback' in lower_result:
                performance_data = lower_result['performance_feedback']
                self.data_exchange.exchange_performance_feedback(
                    performance_data=performance_data,
                    target_node="upper_layer"
                )
            
        except Exception as e:
            self.logger.error(f"信息交换失败: {str(e)}")
    
    def _calculate_communication_success_rate(self) -> float:
        """计算通信成功率"""
        # 基于数据交换器的统计
        exchange_stats = self.data_exchange.get_exchange_statistics()
        total_exchanges = exchange_stats.get('exchange_statistics', {}).get('total_exchanges', 1)
        successful_exchanges = exchange_stats.get('exchange_statistics', {}).get('successful_exchanges', 0)
        
        return successful_exchanges / total_exchanges if total_exchanges > 0 else 0.0
    
    def _calculate_coordination_efficiency(self, upper_result: Dict, lower_result: Dict) -> float:
        """计算协调效率"""
        # 基于信息一致性和性能改善
        upper_performance = upper_result.get('total_reward', 0.0)
        lower_performance = lower_result.get('total_reward', 0.0)
        
        # 信息延迟惩罚
        coordination_time = self.coordination_manager.coordination_state.get('coordination_quality', 0.0)
        
        # 协调效率 = 平均性能 * 协调质量
        efficiency = (upper_performance + lower_performance) / 2 * coordination_time
        
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_overall_performance(self, upper_result: Dict, lower_result: Dict) -> float:
        """计算整体性能"""
        # 上层权重40%，下层权重40%，协调权重20%
        upper_weight = 0.4
        lower_weight = 0.4
        coordination_weight = 0.2
        
        upper_performance = upper_result.get('total_reward', 0.0)
        lower_performance = lower_result.get('total_reward', 0.0)
        coordination_quality = self.coordination_manager.coordination_state.get('coordination_quality', 0.0)
        
        overall = (upper_weight * upper_performance + 
                  lower_weight * lower_performance + 
                  coordination_weight * coordination_quality)
        
        return max(0.0, overall)
    
    def _calculate_system_stability(self, upper_result: Dict, lower_result: Dict) -> float:
        """计算系统稳定性"""
        # 基于性能方差和约束违反
        if len(self.training_history) < 10:
            return 0.5
        
        recent_metrics = self.training_history[-10:]
        
        # 性能稳定性
        upper_rewards = [m.upper_reward for m in recent_metrics]
        lower_rewards = [m.lower_reward for m in recent_metrics]
        
        upper_stability = 1.0 - np.std(upper_rewards) / (np.mean(upper_rewards) + 1e-6)
        lower_stability = 1.0 - np.std(lower_rewards) / (np.mean(lower_rewards) + 1e-6)
        
        # 约束稳定性
        constraint_violations = [m.constraint_violations for m in recent_metrics]
        constraint_stability = 1.0 - np.mean(constraint_violations) / 10.0
        
        # 综合稳定性
        stability = 0.4 * upper_stability + 0.4 * lower_stability + 0.2 * constraint_stability
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_energy_efficiency(self, upper_result: Dict, lower_result: Dict) -> float:
        """计算能量效率"""
        # 基于功率跟踪精度和控制努力
        tracking_accuracy = lower_result.get('tracking_accuracy', 0.0)
        control_effort = lower_result.get('control_effort', 1.0)
        
        # 能量效率 = 跟踪精度 / 控制努力
        efficiency = tracking_accuracy / (control_effort + 0.1)
        
        return max(0.0, min(1.0, efficiency))
    
    def _create_metrics_from_results(self,
                                   episode: int,
                                   upper_result: Dict,
                                   lower_result: Dict) -> HierarchicalMetrics:
        """从结果创建指标"""
        metrics = HierarchicalMetrics(episode=episode)
        
        metrics.upper_reward = upper_result.get('total_reward', 0.0)
        metrics.lower_reward = lower_result.get('total_reward', 0.0)
        metrics.tracking_accuracy = lower_result.get('tracking_accuracy', 0.0)
        metrics.overall_performance = self._calculate_overall_performance(upper_result, lower_result)
        
        return metrics
    
    def _update_best_performance(self, metrics: HierarchicalMetrics):
        """更新最佳性能"""
        if metrics.overall_performance > self.training_state['best_overall_performance']:
            self.training_state['best_overall_performance'] = metrics.overall_performance
            self._save_best_model()
    
    def _evaluate_hierarchical_model(self) -> Dict[str, float]:
        """评估分层模型"""
        # 分别评估上下层
        upper_eval = self.upper_trainer._evaluate_model()
        lower_eval = self.lower_trainer._evaluate_model()
        
        # 协调评估
        coordination_quality = self.coordination_manager.coordination_state.get('coordination_quality', 0.0)
        communication_success_rate = self._calculate_communication_success_rate()
        
        return {
            'upper_mean_reward': upper_eval.get('mean_total_reward', 0.0),
            'upper_soc_balance': upper_eval.get('mean_soc_balance', 0.0),
            'upper_temp_balance': upper_eval.get('mean_temp_balance', 0.0),
            'lower_tracking_accuracy': lower_eval.get('mean_tracking_accuracy', 0.0),
            'lower_response_time': lower_eval.get('mean_response_time', 0.05),
            'coordination_quality': coordination_quality,
            'communication_success_rate': communication_success_rate,
            'overall_performance': (upper_eval.get('mean_total_reward', 0.0) + 
                                   lower_eval.get('mean_tracking_accuracy', 0.0)) / 2
        }
    
    def _should_early_stop(self) -> bool:
        """检查是否应该提前停止"""
        if len(self.training_history) < 50:
            return False
        
        # 检查整体性能是否停滞
        recent_performance = [m.overall_performance for m in self.training_history[-50:]]
        
        if len(recent_performance) >= 25:
            first_half = np.mean(recent_performance[:25])
            second_half = np.mean(recent_performance[25:])
            
            improvement = (second_half - first_half) / max(abs(first_half), 1e-6)
            return improvement < 0.01
        
        return False
    
    def _log_training_progress(self, metrics: HierarchicalMetrics):
        """记录训练进度"""
        self.logger.info(
            f"回合 {metrics.episode}: "
            f"整体性能={metrics.overall_performance:.4f}, "
            f"上层奖励={metrics.upper_reward:.4f}, "
            f"下层奖励={metrics.lower_reward:.4f}, "
            f"跟踪精度={metrics.tracking_accuracy:.3f}, "
            f"协调效率={metrics.coordination_efficiency:.3f}, "
            f"系统稳定性={metrics.system_stability:.3f}"
        )
    
    def _save_checkpoint(self, episode: int) -> str:
        """保存检查点"""
        checkpoint_path = os.path.join(self.save_dir, f"hierarchical_checkpoint_episode_{episode}.pth")
        
        checkpoint = {
            'episode': episode,
            'training_state': self.training_state,
            'upper_trainer_state': self.upper_trainer.get_training_status(),
            'lower_trainer_state': self.lower_trainer.get_training_status(),
            'coordination_manager_state': {
                'coordination_strategy': self.coordination_manager.coordination_strategy,
                'coordination_state': self.coordination_manager.coordination_state
            },
            'training_history': self.training_history,
            'coordination_history': self.coordination_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"分层检查点已保存: {checkpoint_path}")
        
        return checkpoint_path
    
    def _save_best_model(self) -> str:
        """保存最佳模型"""
        best_model_path = os.path.join(self.save_dir, "best_hierarchical_model.pth")
        
        best_model = {
            'episode': self.training_state['current_episode'],
            'best_overall_performance': self.training_state['best_overall_performance'],
            'upper_agent_state': self.upper_trainer.agent.state_dict(),
            'lower_agent_state': self.lower_trainer.agent.state_dict(),
            'coordination_state': self.coordination_manager.coordination_state,
            'config': self.config
        }
        
        torch.save(best_model, best_model_path)
        self.logger.info(f"最佳分层模型已保存: {best_model_path}")
        
        return best_model_path
    
    def _save_final_model(self):
        """保存最终模型"""
        # 保存最终检查点
        final_checkpoint = self._save_checkpoint(self.training_state['current_episode'])
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, "hierarchical_training_history.json")
        with open(history_path, 'w') as f:
            serializable_history = []
            for metrics in self.training_history:
                serializable_history.append({
                    'episode': metrics.episode,
                    'overall_performance': metrics.overall_performance,
                    'upper_reward': metrics.upper_reward,
                    'lower_reward': metrics.lower_reward,
                    'tracking_accuracy': metrics.tracking_accuracy,
                    'coordination_efficiency': metrics.coordination_efficiency,
                    'system_stability': metrics.system_stability,
                    'energy_efficiency': metrics.energy_efficiency
                })
            json.dump(serializable_history, f, indent=2)
        
        # 保存协调历史
        coordination_path = os.path.join(self.save_dir, "coordination_history.json")
        with open(coordination_path, 'w') as f:
            json.dump(self.coordination_history, f, indent=2)
        
        self.logger.info(f"分层训练数据已保存: {self.save_dir}")
    
    def _calculate_training_statistics(self, total_time: float) -> Dict[str, Any]:
        """计算训练统计信息"""
        if not self.training_history:
            return {}
        
        # 基础统计
        overall_performances = [m.overall_performance for m in self.training_history]
        upper_rewards = [m.upper_reward for m in self.training_history]
        lower_rewards = [m.lower_reward for m in self.training_history]
        tracking_accuracies = [m.tracking_accuracy for m in self.training_history]
        
        stats = {
            'training_summary': {
                'total_episodes': len(self.training_history),
                'total_time': total_time,
                'training_mode': self.training_state['training_mode'],
                'best_overall_performance': self.training_state['best_overall_performance'],
                'synchronizations': len(self.coordination_history)
            },
            
            'performance_statistics': {
                'mean_overall_performance': np.mean(overall_performances),
                'std_overall_performance': np.std(overall_performances),
                'max_overall_performance': np.max(overall_performances),
                'mean_upper_reward': np.mean(upper_rewards),
                'mean_lower_reward': np.mean(lower_rewards),
                'mean_tracking_accuracy': np.mean(tracking_accuracies)
            },
            
            'coordination_statistics': {
                'coordination_frequency': len(self.coordination_history) / len(self.training_history),
                'avg_coordination_quality': np.mean([c['coordination_quality'] for c in self.coordination_history]) if self.coordination_history else 0.0,
                'avg_coordination_time': np.mean([c['coordination_time'] for c in self.coordination_history]) if self.coordination_history else 0.0,
                'sync_success_rate': np.mean([c['sync_success'] for c in self.coordination_history]) if self.coordination_history else 0.0
            },
            
            'layer_performance': {
                'upper_layer_stats': self.upper_trainer._calculate_training_statistics(total_time) if hasattr(self.upper_trainer, '_calculate_training_statistics') else {},
                'lower_layer_stats': self.lower_trainer._calculate_training_statistics(total_time) if hasattr(self.lower_trainer, '_calculate_training_statistics') else {}
            }
        }
        
        return stats
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.training_state = checkpoint['training_state']
            self.training_history = checkpoint['training_history']
            self.coordination_history = checkpoint['coordination_history']
            
            # 加载协调管理器状态
            coord_state = checkpoint['coordination_manager_state']
            self.coordination_manager.coordination_strategy = coord_state['coordination_strategy']
            self.coordination_manager.coordination_state = coord_state['coordination_state']
            
            self.logger.info(f"分层检查点加载成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"分层检查点加载失败: {str(e)}")
            return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        upper_status = self.upper_trainer.get_training_status()
        lower_status = self.lower_trainer.get_training_status()
        
        return {
            'trainer_id': self.trainer_id,
            'hierarchical_state': self.training_state.copy(),
            'coordination_state': self.coordination_manager.coordination_state.copy(),
            'upper_layer_status': upper_status,
            'lower_layer_status': lower_status,
            'communication_status': {
                'message_protocol': self.message_protocol.get_statistics(),
                'information_flow': self.information_flow.get_flow_statistics(),
                'data_exchange': self.data_exchange.get_exchange_statistics()
            },
            'recent_performance': (
                self.training_history[-5:] if len(self.training_history) >= 5 
                else self.training_history
            )
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"HierarchicalTrainer({self.trainer_id}): "
                f"回合={self.training_state['current_episode']}/{self.training_state['total_episodes']}, "
                f"模式={self.training_state['training_mode']}, "
                f"同步={len(self.coordination_history)}, "
                f"训练中={self.training_state['is_training']}")
