import numpy as np
import torch
import time
import os
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from training.hierarchical_trainer import HierarchicalTrainer
from training.upper_trainer import UpperLayerTrainer
from training.lower_trainer import LowerLayerTrainer
from training.pretraining_pipeline import PretrainingPipeline
from training.evaluation_suite import EvaluationSuite
from utils.logger import Logger
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer
from utils.checkpoint_manager import CheckpointManager
from utils.experiment_tracker import ExperimentTracker, ExperimentConfig
from data_processing.scenario_generator import ScenarioGenerator, ScenarioType
from data_processing.load_profile_generator import LoadProfileGenerator
from data_processing.weather_simulator import WeatherSimulator

class ExperimentType(Enum):
    """实验类型枚举"""
    SINGLE_OBJECTIVE = "single_objective"           # 单目标训练
    MULTI_OBJECTIVE = "multi_objective"             # 多目标训练
    HIERARCHICAL = "hierarchical"                   # 分层训练
    BENCHMARK = "benchmark"                         # 基准对比
    ROBUSTNESS = "robustness"                      # 鲁棒性测试
    GENERALIZATION = "generalization"               # 泛化性测试
    PRETRAINING = "pretraining"                    # 预训练实验
    ABLATION = "ablation"                          # 消融实验
    SENSITIVITY = "sensitivity"                     # 敏感性分析
    CASE_STUDY = "case_study"                      # 案例研究

@dataclass
class ExperimentSettings:
    """实验设置"""
    # 基础设置
    experiment_name: str
    experiment_type: ExperimentType
    description: str = ""
    
    # 训练设置
    total_episodes: int = 1000
    evaluation_frequency: int = 100
    save_frequency: int = 200
    
    # 环境设置
    scenario_types: List[ScenarioType] = field(default_factory=lambda: [ScenarioType.DAILY_CYCLE])
    environment_variations: int = 5
    
    # 模型设置
    use_pretraining: bool = True
    enable_hierarchical: bool = True
    
    # 评估设置
    evaluation_episodes: int = 50
    benchmark_methods: List[str] = field(default_factory=list)
    
    # 可视化设置
    enable_visualization: bool = True
    plot_frequency: int = 100
    
    # 资源设置
    device: str = "cpu"
    num_workers: int = 1
    
    # 随机性控制
    random_seed: Optional[int] = 42

@dataclass
class ExperimentResults:
    """实验结果"""
    experiment_id: str
    settings: ExperimentSettings
    
    # 训练结果
    training_metrics: Dict[str, List[float]] = field(default_factory=dict)
    evaluation_metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    # 最终性能
    final_performance: Dict[str, float] = field(default_factory=dict)
    best_performance: Dict[str, float] = field(default_factory=dict)
    
    # 模型检查点
    best_checkpoint_path: Optional[str] = None
    final_checkpoint_path: Optional[str] = None
    
    # 时间统计
    training_time: float = 0.0
    evaluation_time: float = 0.0
    total_time: float = 0.0
    
    # 收敛信息
    convergence_episode: Optional[int] = None
    convergence_achieved: bool = False
    
    # 错误和警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class BasicExperiment:
    """
    基础实验框架
    提供标准化的实验执行流程
    """
    
    def __init__(self, 
                 settings: ExperimentSettings,
                 experiment_id: Optional[str] = None):
        """
        初始化基础实验
        
        Args:
            settings: 实验设置
            experiment_id: 实验ID
        """
        self.settings = settings
        self.experiment_id = experiment_id or f"exp_{int(time.time()*1000)}"
        
        # === 设置随机种子 ===
        if settings.random_seed is not None:
            self._set_random_seeds(settings.random_seed)
        
        # === 初始化组件 ===
        self._initialize_components()
        
        # === 实验状态 ===
        self.is_running = False
        self.is_completed = False
        self.current_episode = 0
        
        # === 结果存储 ===
        self.results = ExperimentResults(
            experiment_id=self.experiment_id,
            settings=settings
        )
        
        print(f"✅ 基础实验初始化完成: {settings.experiment_name}")
        print(f"   实验ID: {self.experiment_id}")
        print(f"   类型: {settings.experiment_type.value}")
    
    def _set_random_seeds(self, seed: int):
        """设置随机种子"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _initialize_components(self):
        """初始化实验组件"""
        # 日志器
        self.logger = Logger(f"Experiment_{self.experiment_id}")
        
        # 配置
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()
        
        # 实验跟踪器
        self.experiment_tracker = ExperimentTracker()
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 可视化器
        if self.settings.enable_visualization:
            self.visualizer = Visualizer()
        
        # 检查点管理器
        self.checkpoint_manager = CheckpointManager()
        
        # 数据生成器
        self.scenario_generator = ScenarioGenerator()
        self.load_generator = LoadProfileGenerator()
        self.weather_simulator = WeatherSimulator()
        
        # 训练器（将在运行时初始化）
        self.trainer = None
        self.evaluator = None
    
    def run_experiment(self) -> ExperimentResults:
        """
        运行完整实验
        
        Returns:
            实验结果
        """
        experiment_start_time = time.time()
        
        try:
            self.logger.info(f"🚀 开始实验: {self.settings.experiment_name}")
            
            # 创建实验跟踪
            exp_config = ExperimentConfig(
                name=self.settings.experiment_name,
                description=self.settings.description,
                hyperparameters=self._get_hyperparameters(),
                random_seed=self.settings.random_seed,
                device=self.settings.device
            )
            
            exp_id = self.experiment_tracker.create_experiment(exp_config)
            self.experiment_tracker.start_experiment(exp_id)
            
            self.is_running = True
            
            # 阶段1: 准备实验环境
            self.logger.info("📋 阶段1: 准备实验环境")
            self._prepare_experiment()
            
            # 阶段2: 初始化模型和训练器
            self.logger.info("🔧 阶段2: 初始化模型和训练器")
            self._initialize_models()
            
            # 阶段3: 预训练（如果启用）
            if self.settings.use_pretraining:
                self.logger.info("📚 阶段3: 预训练")
                self._run_pretraining()
            
            # 阶段4: 主要训练
            self.logger.info("🎯 阶段4: 主要训练")
            training_start_time = time.time()
            self._run_training()
            self.results.training_time = time.time() - training_start_time
            
            # 阶段5: 最终评估
            self.logger.info("📊 阶段5: 最终评估")
            evaluation_start_time = time.time()
            self._run_final_evaluation()
            self.results.evaluation_time = time.time() - evaluation_start_time
            
            # 阶段6: 结果分析
            self.logger.info("📈 阶段6: 结果分析")
            self._analyze_results()
            
            # 阶段7: 生成报告
            self.logger.info("📑 阶段7: 生成报告")
            self._generate_report()
            
            # 完成实验
            self.results.total_time = time.time() - experiment_start_time
            self.is_completed = True
            self.is_running = False
            
            # 记录最终结果
            self.experiment_tracker.complete_experiment(
                exp_id, 
                final_results=self.results.final_performance
            )
            
            self.logger.info(f"✅ 实验完成: {self.settings.experiment_name}")
            self.logger.info(f"   总用时: {self.results.total_time:.2f}s")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ 实验失败: {str(e)}")
            self.results.errors.append(str(e))
            self.is_running = False
            
            # 记录实验失败
            if 'exp_id' in locals():
                self.experiment_tracker.fail_experiment(str(e), exp_id)
            
            raise
    
    def _prepare_experiment(self):
        """准备实验环境"""
        # 生成实验场景
        self.scenarios = []
        for scenario_type in self.settings.scenario_types:
            for i in range(self.settings.environment_variations):
                scenario = self.scenario_generator.generate_scenario(
                    scenario_type=scenario_type,
                    scenario_id=f"{self.experiment_id}_{scenario_type.value}_{i}"
                )
                self.scenarios.append(scenario)
        
        self.logger.info(f"生成了 {len(self.scenarios)} 个实验场景")
        
        # 创建实验目录
        self.experiment_dir = f"experiments/runs/{self.experiment_id}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 保存实验设置
        settings_path = os.path.join(self.experiment_dir, "settings.json")
        with open(settings_path, 'w') as f:
            json.dump({
                'experiment_name': self.settings.experiment_name,
                'experiment_type': self.settings.experiment_type.value,
                'description': self.settings.description,
                'total_episodes': self.settings.total_episodes,
                'random_seed': self.settings.random_seed,
                'device': self.settings.device
            }, f, indent=2)
    
    def _initialize_models(self):
        """初始化模型和训练器"""
        # 根据实验类型选择训练器
        if self.settings.experiment_type == ExperimentType.HIERARCHICAL or self.settings.enable_hierarchical:
            self.trainer = HierarchicalTrainer(
                config=self.training_config,
                model_config=self.model_config,
                trainer_id=f"trainer_{self.experiment_id}"
            )
        elif self.settings.experiment_type == ExperimentType.SINGLE_OBJECTIVE:
            # 使用下层训练器进行单目标训练
            self.trainer = LowerLayerTrainer(
                config=self.training_config.lower_config,
                model_config=self.model_config,
                trainer_id=f"trainer_{self.experiment_id}"
            )
        else:
            # 默认使用分层训练器
            self.trainer = HierarchicalTrainer(
                config=self.training_config,
                model_config=self.model_config,
                trainer_id=f"trainer_{self.experiment_id}"
            )
        
        # 初始化评估器
        self.evaluator = EvaluationSuite(
            config=self.training_config,
            model_config=self.model_config,
            suite_id=f"evaluator_{self.experiment_id}"
        )
        
        self.logger.info(f"训练器类型: {type(self.trainer).__name__}")
    
    def _run_pretraining(self):
        """运行预训练"""
        if not isinstance(self.trainer, HierarchicalTrainer):
            self.logger.warning("非分层训练器，跳过预训练")
            return
        
        try:
            # 创建预训练流水线
            pretraining_pipeline = PretrainingPipeline(
                config=self.training_config,
                model_config=self.model_config,
                pipeline_id=f"pretrain_{self.experiment_id}"
            )
            
            # 运行预训练
            pretraining_results = pretraining_pipeline.run_pretraining()
            
            # 记录预训练结果
            for stage, stats in pretraining_results['stage_results'].items():
                self.experiment_tracker.log_metric(
                    f"pretraining_{stage}_performance",
                    stats.get('final_performance', 0),
                    step=0
                )
            
            self.logger.info("预训练完成")
            
        except Exception as e:
            self.logger.error(f"预训练失败: {str(e)}")
            self.results.warnings.append(f"预训练失败: {str(e)}")
    
    def _run_training(self):
        """运行主要训练"""
        self.logger.info(f"开始训练 {self.settings.total_episodes} 个回合")
        
        # 训练循环
        for episode in range(self.settings.total_episodes):
            self.current_episode = episode
            
            try:
                # 选择场景
                scenario = self.scenarios[episode % len(self.scenarios)]
                
                # 执行训练步骤
                episode_metrics = self._train_episode(episode, scenario)
                
                # 记录训练指标
                for metric_name, value in episode_metrics.items():
                    if metric_name not in self.results.training_metrics:
                        self.results.training_metrics[metric_name] = []
                    self.results.training_metrics[metric_name].append(value)
                    
                    # 记录到实验跟踪器
                    self.experiment_tracker.log_metric(
                        metric_name, value, step=episode, episode=episode
                    )
                
                # 定期评估
                if (episode + 1) % self.settings.evaluation_frequency == 0:
                    eval_metrics = self._evaluate_performance(episode)
                    
                    for metric_name, value in eval_metrics.items():
                        if metric_name not in self.results.evaluation_metrics:
                            self.results.evaluation_metrics[metric_name] = []
                        self.results.evaluation_metrics[metric_name].append(value)
                        
                        # 记录到实验跟踪器
                        self.experiment_tracker.log_metric(
                            f"eval_{metric_name}", value, step=episode, episode=episode
                        )
                
                # 定期保存检查点
                if (episode + 1) % self.settings.save_frequency == 0:
                    self._save_checkpoint(episode, episode_metrics)
                
                # 定期可视化
                if (self.settings.enable_visualization and 
                    (episode + 1) % self.settings.plot_frequency == 0):
                    self._update_visualizations(episode)
                
                # 检查收敛
                if self._check_convergence(episode):
                    self.results.convergence_episode = episode
                    self.results.convergence_achieved = True
                    self.logger.info(f"训练收敛于第 {episode} 回合")
                    break
                
                # 进度日志
                if (episode + 1) % 100 == 0:
                    self.logger.info(f"训练进度: {episode + 1}/{self.settings.total_episodes}")
                
            except Exception as e:
                self.logger.error(f"第 {episode} 回合训练失败: {str(e)}")
                self.results.errors.append(f"Episode {episode}: {str(e)}")
                
                # 如果连续失败，停止训练
                if len(self.results.errors) > 10:
                    raise RuntimeError("连续训练失败过多，停止训练")
        
        self.logger.info("主要训练完成")
    
    def _train_episode(self, episode: int, scenario) -> Dict[str, float]:
        """训练单个回合"""
        # 这是一个模拟的训练过程
        # 在实际实现中，这里会调用训练器的具体训练方法
        
        # 模拟训练指标
        base_reward = 100 + episode * 0.5 + np.random.normal(0, 10)
        base_loss = max(0.1, 50 - episode * 0.02 + np.random.normal(0, 5))
        
        episode_metrics = {
            'episode_reward': base_reward,
            'actor_loss': base_loss,
            'critic_loss': base_loss * 0.8,
            'tracking_accuracy': min(1.0, 0.5 + episode * 0.001 + np.random.normal(0, 0.05)),
            'energy_efficiency': min(1.0, 0.6 + episode * 0.0008 + np.random.normal(0, 0.03)),
            'temperature_stability': min(1.0, 0.7 + episode * 0.0005 + np.random.normal(0, 0.02))
        }
        
        return episode_metrics
    
    def _evaluate_performance(self, episode: int) -> Dict[str, float]:
        """评估性能"""
        # 模拟评估过程
        eval_metrics = {}
        
        # 如果有训练指标，基于最近的训练表现生成评估指标
        if self.results.training_metrics:
            recent_window = min(10, len(self.results.training_metrics.get('episode_reward', [])))
            
            if recent_window > 0:
                recent_rewards = self.results.training_metrics['episode_reward'][-recent_window:]
                eval_metrics['avg_reward'] = np.mean(recent_rewards)
                eval_metrics['reward_std'] = np.std(recent_rewards)
                
                if 'tracking_accuracy' in self.results.training_metrics:
                    recent_accuracy = self.results.training_metrics['tracking_accuracy'][-recent_window:]
                    eval_metrics['avg_tracking_accuracy'] = np.mean(recent_accuracy)
                
                if 'energy_efficiency' in self.results.training_metrics:
                    recent_efficiency = self.results.training_metrics['energy_efficiency'][-recent_window:]
                    eval_metrics['avg_energy_efficiency'] = np.mean(recent_efficiency)
        
        return eval_metrics
    
    def _save_checkpoint(self, episode: int, metrics: Dict[str, float]):
        """保存检查点"""
        try:
            # 生成检查点ID
            checkpoint_id = f"{self.experiment_id}_ep{episode}"
            
            # 模拟保存训练器状态
            checkpoint_data = {
                'episode': episode,
                'metrics': metrics,
                'experiment_id': self.experiment_id,
                'trainer_state': 'simulated_state',  # 在实际实现中这里是真实的模型状态
                'random_state': np.random.get_state()
            }
            
            checkpoint_path = os.path.join(self.experiment_dir, f"checkpoint_{episode}.pth")
            
            # 保存检查点
            torch.save(checkpoint_data, checkpoint_path)
            
            # 记录检查点
            self.experiment_tracker.log_model_checkpoint(
                checkpoint_path, 
                is_best=self._is_best_performance(metrics)
            )
            
            # 更新最佳检查点
            if self._is_best_performance(metrics):
                self.results.best_checkpoint_path = checkpoint_path
            
            self.results.final_checkpoint_path = checkpoint_path
            
        except Exception as e:
            self.logger.error(f"保存检查点失败: {str(e)}")
    
    def _is_best_performance(self, metrics: Dict[str, float]) -> bool:
        """判断是否为最佳性能"""
        if not self.results.best_performance:
            return True
        
        # 以奖励作为主要指标
        current_reward = metrics.get('episode_reward', 0)
        best_reward = self.results.best_performance.get('episode_reward', 0)
        
        if current_reward > best_reward:
            self.results.best_performance = metrics.copy()
            return True
        
        return False
    
    def _check_convergence(self, episode: int) -> bool:
        """检查是否收敛"""
        if episode < 100:  # 至少训练100回合
            return False
        
        # 检查最近50回合的性能稳定性
        if 'episode_reward' in self.results.training_metrics:
            recent_rewards = self.results.training_metrics['episode_reward'][-50:]
            if len(recent_rewards) >= 50:
                # 计算变异系数
                cv = np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-6)
                return cv < 0.05  # 变异系数小于5%认为收敛
        
        return False
    
    def _update_visualizations(self, episode: int):
        """更新可视化"""
        if not self.settings.enable_visualization:
            return
        
        try:
            # 创建训练曲线图
            if self.results.training_metrics:
                training_data = {}
                for metric_name, values in self.results.training_metrics.items():
                    training_data[metric_name] = np.array(values)
                
                # 使用可视化器创建图表（这里是简化版本）
                # 在实际实现中会调用 visualizer 的方法
                pass
                
        except Exception as e:
            self.logger.warning(f"更新可视化失败: {str(e)}")
    
    def _run_final_evaluation(self):
        """运行最终评估"""
        self.logger.info("开始最终评估")
        
        try:
            # 加载最佳模型（如果有）
            if self.results.best_checkpoint_path:
                # 在实际实现中这里会加载真实的模型检查点
                pass
            
            # 在所有场景上评估
            final_metrics = {}
            
            for i, scenario in enumerate(self.scenarios):
                scenario_metrics = self._evaluate_on_scenario(scenario)
                
                for metric_name, value in scenario_metrics.items():
                    if metric_name not in final_metrics:
                        final_metrics[metric_name] = []
                    final_metrics[metric_name].append(value)
            
            # 计算最终性能指标
            for metric_name, values in final_metrics.items():
                self.results.final_performance[f"final_{metric_name}"] = np.mean(values)
                self.results.final_performance[f"final_{metric_name}_std"] = np.std(values)
            
            self.logger.info("最终评估完成")
            
        except Exception as e:
            self.logger.error(f"最终评估失败: {str(e)}")
            self.results.errors.append(f"最终评估失败: {str(e)}")
    
    def _evaluate_on_scenario(self, scenario) -> Dict[str, float]:
        """在特定场景上评估"""
        # 模拟场景评估
        scenario_metrics = {
            'reward': np.random.normal(150, 20),
            'tracking_accuracy': np.random.uniform(0.85, 0.95),
            'energy_efficiency': np.random.uniform(0.80, 0.92),
            'safety_margin': np.random.uniform(0.75, 0.90)
        }
        
        return scenario_metrics
    
    def _analyze_results(self):
        """分析实验结果"""
        self.logger.info("开始结果分析")
        
        # 分析训练趋势
        if self.results.training_metrics:
            for metric_name, values in self.results.training_metrics.items():
                if len(values) > 10:
                    # 计算趋势
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    self.results.final_performance[f"{metric_name}_trend"] = slope
        
        # 分析收敛性
        if self.results.convergence_achieved:
            self.results.final_performance['convergence_speed'] = (
                self.results.convergence_episode / self.settings.total_episodes
            )
        
        # 计算效率指标
        if self.results.training_time > 0:
            self.results.final_performance['training_efficiency'] = (
                self.current_episode / self.results.training_time
            )
        
        self.logger.info("结果分析完成")
    
    def _generate_report(self):
        """生成实验报告"""
        report = {
            'experiment_info': {
                'id': self.experiment_id,
                'name': self.settings.experiment_name,
                'type': self.settings.experiment_type.value,
                'description': self.settings.description,
                'completion_time': time.time()
            },
            'settings': {
                'total_episodes': self.settings.total_episodes,
                'random_seed': self.settings.random_seed,
                'device': self.settings.device,
                'use_pretraining': self.settings.use_pretraining,
                'enable_hierarchical': self.settings.enable_hierarchical
            },
            'results': {
                'final_performance': self.results.final_performance,
                'best_performance': self.results.best_performance,
                'convergence_achieved': self.results.convergence_achieved,
                'convergence_episode': self.results.convergence_episode,
                'training_time': self.results.training_time,
                'total_time': self.results.total_time
            },
            'diagnostics': {
                'errors_count': len(self.results.errors),
                'warnings_count': len(self.results.warnings),
                'errors': self.results.errors,
                'warnings': self.results.warnings
            }
        }
        
        # 保存报告
        report_path = os.path.join(self.experiment_dir, "experiment_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"实验报告已保存: {report_path}")
    
    def _get_hyperparameters(self) -> Dict[str, Any]:
        """获取超参数"""
        return {
            'total_episodes': self.settings.total_episodes,
            'evaluation_frequency': self.settings.evaluation_frequency,
            'random_seed': self.settings.random_seed,
            'device': self.settings.device,
            'use_pretraining': self.settings.use_pretraining,
            'enable_hierarchical': self.settings.enable_hierarchical
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """获取实验进度"""
        progress = {
            'experiment_id': self.experiment_id,
            'is_running': self.is_running,
            'is_completed': self.is_completed,
            'current_episode': self.current_episode,
            'total_episodes': self.settings.total_episodes,
            'progress_percentage': (self.current_episode / self.settings.total_episodes) * 100,
            'convergence_achieved': self.results.convergence_achieved
        }
        
        if self.results.training_metrics:
            # 添加最新指标
            for metric_name, values in self.results.training_metrics.items():
                if values:
                    progress[f'latest_{metric_name}'] = values[-1]
        
        return progress
    
    def stop_experiment(self):
        """停止实验"""
        if self.is_running:
            self.is_running = False
            self.logger.info("实验已手动停止")
    
    def save_experiment_state(self, file_path: str):
        """保存实验状态"""
        state = {
            'experiment_id': self.experiment_id,
            'settings': self.settings,
            'results': self.results,
            'current_episode': self.current_episode,
            'is_completed': self.is_completed
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"实验状态已保存: {file_path}")
    
    def load_experiment_state(self, file_path: str):
        """加载实验状态"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        self.experiment_id = state['experiment_id']
        self.current_episode = state['current_episode']
        self.is_completed = state['is_completed']
        
        # 重建结果对象
        self.results = ExperimentResults(**state['results'])
        
        self.logger.info(f"实验状态已加载: {file_path}")
    
    def __str__(self) -> str:
        """字符串表示"""
        status = "完成" if self.is_completed else ("运行中" if self.is_running else "未开始")
        return (f"BasicExperiment({self.settings.experiment_name}): "
                f"状态={status}, 进度={self.current_episode}/{self.settings.total_episodes}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"BasicExperiment(experiment_id='{self.experiment_id}', "
                f"type='{self.settings.experiment_type.value}', "
                f"episodes={self.current_episode}/{self.settings.total_episodes})")
