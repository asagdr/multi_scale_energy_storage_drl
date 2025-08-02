import torch
import torch.nn as nn
import numpy as np
import time
import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
from enum import Enum
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.training_config import TrainingConfig, UpperLayerConfig, LowerLayerConfig
from config.model_config import ModelConfig
from .upper_trainer import UpperLayerTrainer
from .lower_trainer import LowerLayerTrainer

class PretrainingStage(Enum):
    """预训练阶段枚举"""
    INITIALIZATION = "initialization"
    UPPER_PRETRAINING = "upper_pretraining"
    LOWER_PRETRAINING = "lower_pretraining"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"
    JOINT_FINETUNING = "joint_finetuning"
    COMPLETED = "completed"

@dataclass
class PretrainingMetrics:
    """预训练指标"""
    stage: PretrainingStage
    stage_episode: int = 0
    total_episode: int = 0
    
    # 阶段性能
    stage_performance: float = 0.0
    performance_improvement: float = 0.0
    convergence_score: float = 0.0
    
    # 知识迁移
    knowledge_transfer_score: float = 0.0
    transfer_efficiency: float = 0.0
    
    # 时间指标
    stage_time: float = 0.0
    cumulative_time: float = 0.0
    
    # 阶段特定指标
    stage_specific_metrics: Dict[str, float] = field(default_factory=dict)

class CurriculumLearning:
    """课程学习管理器"""
    
    def __init__(self, curriculum_id: str = "Curriculum_001"):
        self.curriculum_id = curriculum_id
        
        # 课程配置
        self.curriculum_config = {
            'difficulty_progression': 'linear',  # 'linear', 'exponential', 'adaptive'
            'initial_difficulty': 0.3,           # 初始难度
            'final_difficulty': 1.0,             # 最终难度
            'adaptation_rate': 0.1,              # 自适应率
            'performance_threshold': 0.7         # 性能阈值
        }
        
        # 当前状态
        self.current_difficulty = self.curriculum_config['initial_difficulty']
        self.performance_history = []
        
    def get_current_difficulty(self, episode: int, total_episodes: int, performance: float = None) -> float:
        """获取当前难度"""
        if self.curriculum_config['difficulty_progression'] == 'linear':
            # 线性增长
            progress = episode / total_episodes
            self.current_difficulty = (
                self.curriculum_config['initial_difficulty'] + 
                progress * (self.curriculum_config['final_difficulty'] - self.curriculum_config['initial_difficulty'])
            )
            
        elif self.curriculum_config['difficulty_progression'] == 'exponential':
            # 指数增长
            progress = episode / total_episodes
            self.current_difficulty = (
                self.curriculum_config['initial_difficulty'] * 
                (self.curriculum_config['final_difficulty'] / self.curriculum_config['initial_difficulty']) ** progress
            )
            
        elif self.curriculum_config['difficulty_progression'] == 'adaptive':
            # 自适应调整
            if performance is not None:
                self.performance_history.append(performance)
                
                if len(self.performance_history) >= 10:
                    avg_performance = np.mean(self.performance_history[-10:])
                    
                    if avg_performance > self.curriculum_config['performance_threshold']:
                        # 性能良好，增加难度
                        self.current_difficulty = min(
                            self.curriculum_config['final_difficulty'],
                            self.current_difficulty + self.curriculum_config['adaptation_rate']
                        )
                    elif avg_performance < self.curriculum_config['performance_threshold'] * 0.8:
                        # 性能不佳，降低难度
                        self.current_difficulty = max(
                            self.curriculum_config['initial_difficulty'],
                            self.current_difficulty - self.curriculum_config['adaptation_rate']
                        )
        
        return self.current_difficulty
    
    def generate_curriculum_parameters(self, difficulty: float) -> Dict[str, Any]:
        """生成课程参数"""
        return {
            'scenario_complexity': difficulty,
            'noise_level': 0.1 + difficulty * 0.2,
            'constraint_strictness': 0.5 + difficulty * 0.5,
            'disturbance_magnitude': difficulty * 0.3,
            'multi_objective_weights': self._generate_weights(difficulty)
        }
    
    def _generate_weights(self, difficulty: float) -> List[float]:
        """生成多目标权重"""
        if difficulty < 0.5:
            # 简单阶段：专注单一目标
            return [0.7, 0.1, 0.1, 0.1]
        elif difficulty < 0.8:
            # 中等阶段：双目标
            return [0.4, 0.4, 0.1, 0.1]
        else:
            # 困难阶段：多目标均衡
            return [0.25, 0.25, 0.25, 0.25]

class KnowledgeTransfer:
    """知识迁移管理器"""
    
    def __init__(self, transfer_id: str = "KnowledgeTransfer_001"):
        self.transfer_id = transfer_id
        
        # 迁移策略
        self.transfer_strategies = {
            'feature_extraction': True,    # 特征提取
            'fine_tuning': True,          # 微调
            'progressive_unfreezing': True, # 渐进解冻
            'distillation': False         # 知识蒸馏
        }
        
        # 迁移历史
        self.transfer_history = []
        
    def transfer_upper_to_lower(self,
                               upper_agent: nn.Module,
                               lower_agent: nn.Module) -> Dict[str, float]:
        """上层到下层的知识迁移"""
        transfer_start_time = time.time()
        
        # 特征提取器迁移
        if self.transfer_strategies['feature_extraction']:
            transfer_score = self._transfer_feature_extractor(upper_agent, lower_agent)
        else:
            transfer_score = 0.0
        
        # 策略网络微调
        if self.transfer_strategies['fine_tuning']:
            fine_tuning_score = self._fine_tune_policy(upper_agent, lower_agent)
            transfer_score = (transfer_score + fine_tuning_score) / 2
        
        transfer_time = time.time() - transfer_start_time
        
        # 记录迁移历史
        transfer_record = {
            'direction': 'upper_to_lower',
            'transfer_score': transfer_score,
            'transfer_time': transfer_time,
            'strategies_used': [k for k, v in self.transfer_strategies.items() if v]
        }
        self.transfer_history.append(transfer_record)
        
        return {
            'transfer_score': transfer_score,
            'transfer_time': transfer_time,
            'transfer_success': transfer_score > 0.5
        }
    
    def transfer_lower_to_upper(self,
                               lower_agent: nn.Module,
                               upper_agent: nn.Module) -> Dict[str, float]:
        """下层到上层的知识迁移"""
        transfer_start_time = time.time()
        
        # 控制策略迁移
        control_transfer_score = self._transfer_control_strategy(lower_agent, upper_agent)
        
        # 约束处理迁移
        constraint_transfer_score = self._transfer_constraint_handling(lower_agent, upper_agent)
        
        # 综合迁移得分
        transfer_score = (control_transfer_score + constraint_transfer_score) / 2
        
        transfer_time = time.time() - transfer_start_time
        
        # 记录迁移历史
        transfer_record = {
            'direction': 'lower_to_upper',
            'transfer_score': transfer_score,
            'transfer_time': transfer_time,
            'control_transfer': control_transfer_score,
            'constraint_transfer': constraint_transfer_score
        }
        self.transfer_history.append(transfer_record)
        
        return {
            'transfer_score': transfer_score,
            'transfer_time': transfer_time,
            'transfer_success': transfer_score > 0.5
        }
    
    def _transfer_feature_extractor(self, source_agent: nn.Module, target_agent: nn.Module) -> float:
        """迁移特征提取器"""
        try:
            # 假设两个智能体都有feature_extractor属性
            if hasattr(source_agent, 'transformer_encoder') and hasattr(target_agent, 'neural_tracker'):
                source_features = source_agent.transformer_encoder.state_dict()
                target_features = target_agent.neural_tracker.state_dict()
                
                # 迁移兼容的层
                transferred_layers = 0
                total_layers = 0
                
                for source_key, source_param in source_features.items():
                    total_layers += 1
                    # 寻找兼容的目标层
                    for target_key, target_param in target_features.items():
                        if source_param.shape == target_param.shape:
                            target_param.data.copy_(source_param.data)
                            transferred_layers += 1
                            break
                
                return transferred_layers / max(total_layers, 1)
            
            return 0.3  # 默认迁移得分
            
        except Exception as e:
            print(f"特征提取器迁移失败: {str(e)}")
            return 0.0
    
    def _fine_tune_policy(self, source_agent: nn.Module, target_agent: nn.Module) -> float:
        """微调策略网络"""
        try:
            # 策略网络参数微调
            if hasattr(source_agent, 'actor') and hasattr(target_agent, 'actor'):
                source_policy = source_agent.actor.state_dict()
                target_policy = target_agent.actor.state_dict()
                
                # 计算参数相似度
                similarity_scores = []
                
                for source_key, source_param in source_policy.items():
                    for target_key, target_param in target_policy.items():
                        if source_param.shape == target_param.shape:
                            # 计算余弦相似度
                            source_flat = source_param.flatten()
                            target_flat = target_param.flatten()
                            
                            similarity = torch.cosine_similarity(
                                source_flat.unsqueeze(0), 
                                target_flat.unsqueeze(0)
                            ).item()
                            similarity_scores.append(abs(similarity))
                            break
                
                return np.mean(similarity_scores) if similarity_scores else 0.2
            
            return 0.2  # 默认微调得分
            
        except Exception as e:
            print(f"策略微调失败: {str(e)}")
            return 0.0
    
    def _transfer_control_strategy(self, source_agent: nn.Module, target_agent: nn.Module) -> float:
        """迁移控制策略"""
        # 简化的控制策略迁移
        return 0.6  # 模拟迁移得分
    
    def _transfer_constraint_handling(self, source_agent: nn.Module, target_agent: nn.Module) -> float:
        """迁移约束处理"""
        # 简化的约束处理迁移
        return 0.7  # 模拟迁移得分

class PretrainingPipeline:
    """
    预训练流水线
    实现分层DRL的渐进式预训练策略
    """
    
    def __init__(self,
                 config: TrainingConfig,
                 model_config: ModelConfig,
                 pipeline_id: str = "PretrainingPipeline_001"):
        """
        初始化预训练流水线
        
        Args:
            config: 训练配置
            model_config: 模型配置
            pipeline_id: 流水线ID
        """
        self.config = config
        self.model_config = model_config
        self.pipeline_id = pipeline_id
        
        # === 初始化子训练器 ===
        self.upper_trainer = UpperLayerTrainer(
            config=config.upper_config,
            model_config=model_config,
            trainer_id=f"PretrainUpper_{pipeline_id}"
        )
        
        self.lower_trainer = LowerLayerTrainer(
            config=config.lower_config,
            model_config=model_config,
            trainer_id=f"PretrainLower_{pipeline_id}"
        )
        
        # === 初始化管理器 ===
        self.curriculum_learning = CurriculumLearning(f"Curriculum_{pipeline_id}")
        self.knowledge_transfer = KnowledgeTransfer(f"Transfer_{pipeline_id}")
        
        # === 预训练配置 ===
        self.pretraining_config = {
            'upper_pretraining_episodes': config.pretraining_episodes // 3,
            'lower_pretraining_episodes': config.pretraining_episodes // 3,
            'joint_finetuning_episodes': config.pretraining_episodes // 3,
            'enable_curriculum_learning': True,
            'enable_knowledge_transfer': True,
            'progressive_difficulty': True,
            'early_stopping_patience': 50
        }
        
        # === 预训练状态 ===
        self.pretraining_state = {
            'current_stage': PretrainingStage.INITIALIZATION,
            'stage_episode': 0,
            'total_episode': 0,
            'stage_start_time': 0.0,
            'pipeline_start_time': 0.0,
            'is_pretraining': False
        }
        
        # === 预训练历史 ===
        self.pretraining_history: List[PretrainingMetrics] = []
        self.stage_performance_history: Dict[PretrainingStage, List[float]] = {
            stage: [] for stage in PretrainingStage
        }
        
        # === 日志设置 ===
        self._setup_logging()
        
        # === 保存路径 ===
        self.save_dir = f"checkpoints/pretraining/{pipeline_id}"
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"✅ 预训练流水线初始化完成: {pipeline_id}")
        print(f"   预训练回合: 上层={self.pretraining_config['upper_pretraining_episodes']}, "
              f"下层={self.pretraining_config['lower_pretraining_episodes']}, "
              f"联合={self.pretraining_config['joint_finetuning_episodes']}")
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = f"logs/pretraining/{self.pipeline_id}"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/pretraining_pipeline.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(f"PretrainingPipeline_{self.pipeline_id}")
    
    def run_pretraining(self) -> Dict[str, Any]:
        """
        运行完整的预训练流水线
        
        Returns:
            预训练结果统计
        """
        self.pretraining_state['is_pretraining'] = True
        self.pretraining_state['pipeline_start_time'] = time.time()
        
        self.logger.info("🚀 开始预训练流水线")
        
        try:
            # === 阶段1: 上层预训练 ===
            self.logger.info("📈 阶段1: 上层预训练")
            upper_stats = self._run_upper_pretraining()
            
            # === 阶段2: 下层预训练 ===
            self.logger.info("⚡ 阶段2: 下层预训练")
            lower_stats = self._run_lower_pretraining()
            
            # === 阶段3: 知识迁移 ===
            self.logger.info("🔄 阶段3: 知识迁移")
            transfer_stats = self._run_knowledge_transfer()
            
            # === 阶段4: 联合微调 ===
            self.logger.info("🎯 阶段4: 联合微调")
            finetuning_stats = self._run_joint_finetuning()
            
            # === 完成预训练 ===
            self._complete_pretraining()
            
            # 综合统计
            pipeline_stats = self._calculate_pipeline_statistics(
                upper_stats, lower_stats, transfer_stats, finetuning_stats
            )
            
            self.logger.info("✅ 预训练流水线完成")
            
            return pipeline_stats
            
        except Exception as e:
            self.logger.error(f"❌ 预训练流水线失败: {str(e)}")
            raise
        finally:
            self.pretraining_state['is_pretraining'] = False
    
    def _run_upper_pretraining(self) -> Dict[str, Any]:
        """运行上层预训练"""
        self._enter_stage(PretrainingStage.UPPER_PRETRAINING)
        
        episodes = self.pretraining_config['upper_pretraining_episodes']
        self.logger.info(f"开始上层预训练: {episodes} 回合")
        
        # 配置课程学习
        if self.pretraining_config['enable_curriculum_learning']:
            self._configure_upper_curriculum()
        
        # 执行上层预训练
        upper_stats = {}
        for episode in range(episodes):
            self.pretraining_state['stage_episode'] = episode
            self.pretraining_state['total_episode'] += 1
            
            # 获取当前难度
            if self.pretraining_config['progressive_difficulty']:
                difficulty = self.curriculum_learning.get_current_difficulty(episode, episodes)
                curriculum_params = self.curriculum_learning.generate_curriculum_parameters(difficulty)
                self._apply_upper_curriculum(curriculum_params)
            
            # 训练一个回合（简化版本）
            episode_metrics = self._simulate_upper_training_episode(episode)
            
            # 记录性能
            self.stage_performance_history[PretrainingStage.UPPER_PRETRAINING].append(
                episode_metrics.stage_performance
            )
            
            # 检查早停
            if self._should_early_stop_stage(PretrainingStage.UPPER_PRETRAINING):
                self.logger.info(f"上层预训练早停于回合 {episode}")
                break
            
            # 定期日志
            if (episode + 1) % 20 == 0:
                self.logger.info(f"上层预训练进度: {episode+1}/{episodes}, "
                               f"性能={episode_metrics.stage_performance:.3f}")
        
        # 保存上层预训练模型
        upper_checkpoint = self._save_stage_checkpoint(PretrainingStage.UPPER_PRETRAINING)
        
        upper_stats = {
            'episodes': episodes,
            'final_performance': self.stage_performance_history[PretrainingStage.UPPER_PRETRAINING][-1],
            'avg_performance': np.mean(self.stage_performance_history[PretrainingStage.UPPER_PRETRAINING]),
            'checkpoint_path': upper_checkpoint
        }
        
        self.logger.info(f"上层预训练完成: 最终性能={upper_stats['final_performance']:.3f}")
        
        return upper_stats
    
    def _run_lower_pretraining(self) -> Dict[str, Any]:
        """运行下层预训练"""
        self._enter_stage(PretrainingStage.LOWER_PRETRAINING)
        
        episodes = self.pretraining_config['lower_pretraining_episodes']
        self.logger.info(f"开始下层预训练: {episodes} 回合")
        
        # 配置课程学习
        if self.pretraining_config['enable_curriculum_learning']:
            self._configure_lower_curriculum()
        
        # 执行下层预训练
        for episode in range(episodes):
            self.pretraining_state['stage_episode'] = episode
            self.pretraining_state['total_episode'] += 1
            
            # 获取当前难度
            if self.pretraining_config['progressive_difficulty']:
                difficulty = self.curriculum_learning.get_current_difficulty(episode, episodes)
                curriculum_params = self.curriculum_learning.generate_curriculum_parameters(difficulty)
                self._apply_lower_curriculum(curriculum_params)
            
            # 训练一个回合
            episode_metrics = self._simulate_lower_training_episode(episode)
            
            # 记录性能
            self.stage_performance_history[PretrainingStage.LOWER_PRETRAINING].append(
                episode_metrics.stage_performance
            )
            
            # 检查早停
            if self._should_early_stop_stage(PretrainingStage.LOWER_PRETRAINING):
                self.logger.info(f"下层预训练早停于回合 {episode}")
                break
            
            # 定期日志
            if (episode + 1) % 20 == 0:
                self.logger.info(f"下层预训练进度: {episode+1}/{episodes}, "
                               f"性能={episode_metrics.stage_performance:.3f}")
        
        # 保存下层预训练模型
        lower_checkpoint = self._save_stage_checkpoint(PretrainingStage.LOWER_PRETRAINING)
        
        lower_stats = {
            'episodes': episodes,
            'final_performance': self.stage_performance_history[PretrainingStage.LOWER_PRETRAINING][-1],
            'avg_performance': np.mean(self.stage_performance_history[PretrainingStage.LOWER_PRETRAINING]),
            'checkpoint_path': lower_checkpoint
        }
        
        self.logger.info(f"下层预训练完成: 最终性能={lower_stats['final_performance']:.3f}")
        
        return lower_stats
    
    def _run_knowledge_transfer(self) -> Dict[str, Any]:
        """运行知识迁移"""
        self._enter_stage(PretrainingStage.KNOWLEDGE_TRANSFER)
        
        self.logger.info("开始层间知识迁移")
        
        transfer_stats = {
            'upper_to_lower': {},
            'lower_to_upper': {},
            'bidirectional_transfer': {}
        }
        
        if self.pretraining_config['enable_knowledge_transfer']:
            # 上层到下层迁移
            self.logger.info("执行上层→下层知识迁移")
            upper_to_lower = self.knowledge_transfer.transfer_upper_to_lower(
                self.upper_trainer.agent,
                self.lower_trainer.agent
            )
            transfer_stats['upper_to_lower'] = upper_to_lower
            
            # 下层到上层迁移
            self.logger.info("执行下层→上层知识迁移")
            lower_to_upper = self.knowledge_transfer.transfer_lower_to_upper(
                self.lower_trainer.agent,
                self.upper_trainer.agent
            )
            transfer_stats['lower_to_upper'] = lower_to_upper
            
            # 计算双向迁移效果
            bidirectional_score = (
                upper_to_lower['transfer_score'] + lower_to_upper['transfer_score']
            ) / 2
            
            transfer_stats['bidirectional_transfer'] = {
                'transfer_score': bidirectional_score,
                'transfer_success': bidirectional_score > 0.5,
                'total_transfer_time': (
                    upper_to_lower['transfer_time'] + lower_to_upper['transfer_time']
                )
            }
            
            self.logger.info(f"知识迁移完成: 双向得分={bidirectional_score:.3f}")
        else:
            self.logger.info("知识迁移已禁用")
        
        return transfer_stats
    
    def _run_joint_finetuning(self) -> Dict[str, Any]:
        """运行联合微调"""
        self._enter_stage(PretrainingStage.JOINT_FINETUNING)
        
        episodes = self.pretraining_config['joint_finetuning_episodes']
        self.logger.info(f"开始联合微调: {episodes} 回合")
        
        # 联合微调
        for episode in range(episodes):
            self.pretraining_state['stage_episode'] = episode
            self.pretraining_state['total_episode'] += 1
            
            # 联合训练一个回合
            episode_metrics = self._simulate_joint_training_episode(episode)
            
            # 记录性能
            self.stage_performance_history[PretrainingStage.JOINT_FINETUNING].append(
                episode_metrics.stage_performance
            )
            
            # 检查早停
            if self._should_early_stop_stage(PretrainingStage.JOINT_FINETUNING):
                self.logger.info(f"联合微调早停于回合 {episode}")
                break
            
            # 定期日志
            if (episode + 1) % 10 == 0:
                self.logger.info(f"联合微调进度: {episode+1}/{episodes}, "
                               f"性能={episode_metrics.stage_performance:.3f}")
        
        # 保存联合微调模型
        joint_checkpoint = self._save_stage_checkpoint(PretrainingStage.JOINT_FINETUNING)
        
        finetuning_stats = {
            'episodes': episodes,
            'final_performance': self.stage_performance_history[PretrainingStage.JOINT_FINETUNING][-1],
            'avg_performance': np.mean(self.stage_performance_history[PretrainingStage.JOINT_FINETUNING]),
            'checkpoint_path': joint_checkpoint
        }
        
        self.logger.info(f"联合微调完成: 最终性能={finetuning_stats['final_performance']:.3f}")
        
        return finetuning_stats
    
    def _simulate_upper_training_episode(self, episode: int) -> PretrainingMetrics:
        """模拟上层训练回合"""
        # 简化的上层训练模拟
        base_performance = 0.3 + episode * 0.01  # 逐渐提升
        noise = np.random.normal(0, 0.05)  # 添加噪声
        performance = max(0.0, min(1.0, base_performance + noise))
        
        metrics = PretrainingMetrics(
            stage=PretrainingStage.UPPER_PRETRAINING,
            stage_episode=episode,
            total_episode=self.pretraining_state['total_episode'],
            stage_performance=performance,
            stage_specific_metrics={
                'hypervolume': performance * 0.8,
                'pareto_front_size': int(10 + performance * 20),
                'soc_balance_score': performance * 0.9,
                'temp_balance_score': performance * 0.85
            }
        )
        
        self.pretraining_history.append(metrics)
        return metrics
    
    def _simulate_lower_training_episode(self, episode: int) -> PretrainingMetrics:
        """模拟下层训练回合"""
        # 简化的下层训练模拟
        base_performance = 0.4 + episode * 0.008  # 逐渐提升
        noise = np.random.normal(0, 0.03)  # 添加噪声
        performance = max(0.0, min(1.0, base_performance + noise))
        
        metrics = PretrainingMetrics(
            stage=PretrainingStage.LOWER_PRETRAINING,
            stage_episode=episode,
            total_episode=self.pretraining_state['total_episode'],
            stage_performance=performance,
            stage_specific_metrics={
                'tracking_accuracy': performance,
                'response_time': 0.05 * (1.0 - performance * 0.5),
                'constraint_satisfaction': performance * 0.95,
                'control_smoothness': performance * 0.9
            }
        )
        
        self.pretraining_history.append(metrics)
        return metrics
    
    def _simulate_joint_training_episode(self, episode: int) -> PretrainingMetrics:
        """模拟联合训练回合"""
        # 联合训练性能 = 上层性能 * 0.5 + 下层性能 * 0.5
        upper_performance = 0.7 + episode * 0.005
        lower_performance = 0.8 + episode * 0.003
        joint_performance = (upper_performance + lower_performance) / 2
        
        noise = np.random.normal(0, 0.02)
        performance = max(0.0, min(1.0, joint_performance + noise))
        
        metrics = PretrainingMetrics(
            stage=PretrainingStage.JOINT_FINETUNING,
            stage_episode=episode,
            total_episode=self.pretraining_state['total_episode'],
            stage_performance=performance,
            stage_specific_metrics={
                'upper_contribution': upper_performance,
                'lower_contribution': lower_performance,
                'coordination_efficiency': performance * 0.9,
                'overall_stability': performance * 0.95
            }
        )
        
        self.pretraining_history.append(metrics)
        return metrics
    
    def _enter_stage(self, stage: PretrainingStage):
        """进入新阶段"""
        self.pretraining_state['current_stage'] = stage
        self.pretraining_state['stage_episode'] = 0
        self.pretraining_state['stage_start_time'] = time.time()
        
        self.logger.info(f"进入预训练阶段: {stage.value}")
    
    def _configure_upper_curriculum(self):
        """配置上层课程学习"""
        # 配置上层专用的课程学习参数
        self.curriculum_learning.curriculum_config.update({
            'initial_difficulty': 0.2,
            'final_difficulty': 0.9,
            'performance_threshold': 0.6
        })
    
    def _configure_lower_curriculum(self):
        """配置下层课程学习"""
        # 配置下层专用的课程学习参数
        self.curriculum_learning.curriculum_config.update({
            'initial_difficulty': 0.3,
            'final_difficulty': 1.0,
            'performance_threshold': 0.7
        })
    
    def _apply_upper_curriculum(self, curriculum_params: Dict[str, Any]):
        """应用上层课程参数"""
        # 应用课程参数到上层训练器
        self.logger.debug(f"应用上层课程参数: {curriculum_params}")
    
    def _apply_lower_curriculum(self, curriculum_params: Dict[str, Any]):
        """应用下层课程参数"""
        # 应用课程参数到下层训练器
        self.logger.debug(f"应用下层课程参数: {curriculum_params}")
    
    def _should_early_stop_stage(self, stage: PretrainingStage) -> bool:
        """检查阶段是否应该早停"""
        performance_history = self.stage_performance_history[stage]
        
        if len(performance_history) < self.pretraining_config['early_stopping_patience']:
            return False
        
        # 检查最近的性能是否停滞
        recent_performance = performance_history[-self.pretraining_config['early_stopping_patience']:]
        
        # 计算性能改善
        first_half = np.mean(recent_performance[:len(recent_performance)//2])
        second_half = np.mean(recent_performance[len(recent_performance)//2:])
        
        improvement = (second_half - first_half) / max(abs(first_half), 1e-6)
        
        # 如果改善小于1%，认为收敛
        return improvement < 0.01
    
    def _save_stage_checkpoint(self, stage: PretrainingStage) -> str:
        """保存阶段检查点"""
        checkpoint_path = os.path.join(self.save_dir, f"{stage.value}_checkpoint.pth")
        
        checkpoint = {
            'stage': stage.value,
            'episode': self.pretraining_state['stage_episode'],
            'total_episode': self.pretraining_state['total_episode'],
            'upper_agent_state': self.upper_trainer.agent.state_dict(),
            'lower_agent_state': self.lower_trainer.agent.state_dict(),
            'stage_performance_history': self.stage_performance_history[stage],
            'curriculum_state': {
                'current_difficulty': self.curriculum_learning.current_difficulty,
                'performance_history': self.curriculum_learning.performance_history
            },
            'knowledge_transfer_history': self.knowledge_transfer.transfer_history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"阶段检查点已保存: {checkpoint_path}")
        
        return checkpoint_path
    
    def _complete_pretraining(self):
        """完成预训练"""
        self.pretraining_state['current_stage'] = PretrainingStage.COMPLETED
        
        # 保存最终预训练模型
        final_checkpoint = os.path.join(self.save_dir, "pretraining_final.pth")
        
        final_model = {
            'pipeline_id': self.pipeline_id,
            'completion_time': time.time(),
            'total_episodes': self.pretraining_state['total_episode'],
            'final_upper_state': self.upper_trainer.agent.state_dict(),
            'final_lower_state': self.lower_trainer.agent.state_dict(),
            'pretraining_history': [
                {
                    'stage': m.stage.value,
                    'episode': m.total_episode,
                    'performance': m.stage_performance,
                    'stage_specific': m.stage_specific_metrics
                } for m in self.pretraining_history
            ],
            'stage_statistics': self._calculate_stage_statistics(),
            'config': self.config
        }
        
        torch.save(final_model, final_checkpoint)
        
        # 保存预训练历史
        history_path = os.path.join(self.save_dir, "pretraining_history.json")
        with open(history_path, 'w') as f:
            json.dump(final_model['pretraining_history'], f, indent=2)
        
        self.logger.info(f"预训练完成，最终模型已保存: {final_checkpoint}")
    
    def _calculate_stage_statistics(self) -> Dict[str, Any]:
        """计算阶段统计"""
        stage_stats = {}
        
        for stage in PretrainingStage:
            if stage in self.stage_performance_history:
                performance_list = self.stage_performance_history[stage]
                if performance_list:
                    stage_stats[stage.value] = {
                        'episodes': len(performance_list),
                        'initial_performance': performance_list[0],
                        'final_performance': performance_list[-1],
                        'max_performance': max(performance_list),
                        'avg_performance': np.mean(performance_list),
                        'performance_improvement': performance_list[-1] - performance_list[0],
                        'convergence_episode': self._find_convergence_episode(performance_list)
                    }
        
        return stage_stats
    
    def _find_convergence_episode(self, performance_list: List[float]) -> int:
        """找到收敛回合"""
        if len(performance_list) < 20:
            return -1
        
        # 简化的收敛检测
        window_size = 10
        for i in range(window_size, len(performance_list)):
            window = performance_list[i-window_size:i]
            if np.std(window) < 0.02:  # 标准差小于2%
                return i - window_size
        
        return -1
    
    def _calculate_pipeline_statistics(self,
                                     upper_stats: Dict,
                                     lower_stats: Dict,
                                     transfer_stats: Dict,
                                     finetuning_stats: Dict) -> Dict[str, Any]:
        """计算流水线统计"""
        total_time = time.time() - self.pretraining_state['pipeline_start_time']
        
        pipeline_stats = {
            'pipeline_summary': {
                'pipeline_id': self.pipeline_id,
                'total_time': total_time,
                'total_episodes': self.pretraining_state['total_episode'],
                'stages_completed': len([s for s in self.stage_performance_history.values() if s])
            },
            
            'stage_results': {
                'upper_pretraining': upper_stats,
                'lower_pretraining': lower_stats,
                'knowledge_transfer': transfer_stats,
                'joint_finetuning': finetuning_stats
            },
            
            'overall_performance': {
                'final_upper_performance': upper_stats.get('final_performance', 0.0),
                'final_lower_performance': lower_stats.get('final_performance', 0.0),
                'final_joint_performance': finetuning_stats.get('final_performance', 0.0),
                'transfer_success_rate': (
                    transfer_stats.get('bidirectional_transfer', {}).get('transfer_success', False)
                ),
                'overall_improvement': self._calculate_overall_improvement()
            },
            
            'curriculum_learning': {
                'final_difficulty': self.curriculum_learning.current_difficulty,
                'difficulty_progression': self.curriculum_learning.curriculum_config['difficulty_progression'],
                'performance_history_length': len(self.curriculum_learning.performance_history)
            },
            
            'efficiency_metrics': {
                'episodes_per_hour': self.pretraining_state['total_episode'] / (total_time / 3600),
                'convergence_efficiency': self._calculate_convergence_efficiency(),
                'knowledge_transfer_efficiency': transfer_stats.get('bidirectional_transfer', {}).get('transfer_score', 0.0)
            }
        }
        
        return pipeline_stats
    
    def _calculate_overall_improvement(self) -> float:
        """计算整体改善"""
        if not self.pretraining_history:
            return 0.0
        
        # 比较第一个和最后一个阶段的性能
        first_performance = self.pretraining_history[0].stage_performance
        last_performance = self.pretraining_history[-1].stage_performance
        
        improvement = (last_performance - first_performance) / max(abs(first_performance), 1e-6)
        return improvement
    
    def _calculate_convergence_efficiency(self) -> float:
        """计算收敛效率"""
        stage_stats = self._calculate_stage_statistics()
        
        convergence_episodes = []
        for stage_stat in stage_stats.values():
            if stage_stat['convergence_episode'] > 0:
                convergence_episodes.append(stage_stat['convergence_episode'])
        
        if convergence_episodes:
            avg_convergence = np.mean(convergence_episodes)
            # 效率 = 1 / 平均收敛回合数（归一化）
            efficiency = 1.0 / (1.0 + avg_convergence / 100.0)
            return efficiency
        
        return 0.5  # 默认效率
    
    def load_stage_checkpoint(self, stage: PretrainingStage, checkpoint_path: str) -> bool:
        """加载阶段检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 加载智能体状态
            if 'upper_agent_state' in checkpoint:
                self.upper_trainer.agent.load_state_dict(checkpoint['upper_agent_state'])
            
            if 'lower_agent_state' in checkpoint:
                self.lower_trainer.agent.load_state_dict(checkpoint['lower_agent_state'])
            
            # 加载课程学习状态
            if 'curriculum_state' in checkpoint:
                curriculum_state = checkpoint['curriculum_state']
                self.curriculum_learning.current_difficulty = curriculum_state['current_difficulty']
                self.curriculum_learning.performance_history = curriculum_state['performance_history']
            
            # 加载知识迁移历史
            if 'knowledge_transfer_history' in checkpoint:
                self.knowledge_transfer.transfer_history = checkpoint['knowledge_transfer_history']
            
            # 加载性能历史
            if 'stage_performance_history' in checkpoint:
                self.stage_performance_history[stage] = checkpoint['stage_performance_history']
            
            self.logger.info(f"阶段检查点加载成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"阶段检查点加载失败: {str(e)}")
            return False
    
    def get_pretraining_status(self) -> Dict[str, Any]:
        """获取预训练状态"""
        return {
            'pipeline_id': self.pipeline_id,
            'pretraining_state': self.pretraining_state.copy(),
            'current_stage': self.pretraining_state['current_stage'].value,
            'stage_progress': {
                stage.value: len(history) for stage, history in self.stage_performance_history.items()
            },
            'curriculum_status': {
                'current_difficulty': self.curriculum_learning.current_difficulty,
                'progression_type': self.curriculum_learning.curriculum_config['difficulty_progression']
            },
            'recent_performance': (
                self.pretraining_history[-5:] if len(self.pretraining_history) >= 5 
                else self.pretraining_history
            )
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"PretrainingPipeline({self.pipeline_id}): "
                f"阶段={self.pretraining_state['current_stage'].value}, "
                f"总回合={self.pretraining_state['total_episode']}, "
                f"运行中={self.pretraining_state['is_pretraining']}")
