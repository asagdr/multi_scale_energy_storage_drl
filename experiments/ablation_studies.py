import numpy as np
import torch
import time
import itertools
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType, ExperimentResults
from utils.logger import Logger
from utils.metrics import MetricsCalculator

class AblationComponent(Enum):
    """消融组件枚举"""
    HIERARCHICAL_STRUCTURE = "hierarchical_structure"   # 分层结构
    TRANSFORMER_ENCODER = "transformer_encoder"         # Transformer编码器
    MULTI_OBJECTIVE = "multi_objective"                 # 多目标优化
    PRETRAINING = "pretraining"                         # 预训练
    KNOWLEDGE_TRANSFER = "knowledge_transfer"           # 知识迁移
    CURRICULUM_LEARNING = "curriculum_learning"         # 课程学习
    COMMUNICATION = "communication"                     # 层间通信
    CONSTRAINT_HANDLING = "constraint_handling"         # 约束处理
    TEMPERATURE_COMPENSATION = "temperature_compensation" # 温度补偿
    BALANCE_ANALYZER = "balance_analyzer"               # 均衡分析器
    PARETO_OPTIMIZER = "pareto_optimizer"              # 帕累托优化器
    RESPONSE_OPTIMIZER = "response_optimizer"           # 响应优化器

@dataclass
class AblationConfig:
    """消融实验配置"""
    study_name: str
    description: str = ""
    
    # 要消融的组件
    components_to_ablate: List[AblationComponent] = field(default_factory=list)
    
    # 基线配置（包含所有组件）
    baseline_config: ExperimentSettings = None
    
    # 每个配置的重复次数
    num_repetitions: int = 3
    
    # 是否进行组合消融
    combination_ablation: bool = False
    max_combination_size: int = 3
    
    # 评估指标
    primary_metrics: List[str] = field(default_factory=lambda: [
        'episode_reward', 'tracking_accuracy', 'energy_efficiency'
    ])
    
    # 统计显著性检验
    significance_test: bool = True
    confidence_level: float = 0.95

@dataclass
class AblationResult:
    """消融实验结果"""
    configuration_name: str
    ablated_components: List[AblationComponent]
    experiment_results: List[ExperimentResults]
    
    # 统计指标
    mean_performance: Dict[str, float] = field(default_factory=dict)
    std_performance: Dict[str, float] = field(default_factory=dict)
    
    # 与基线的比较
    performance_drop: Dict[str, float] = field(default_factory=dict)
    relative_drop: Dict[str, float] = field(default_factory=dict)
    
    # 统计显著性
    significance_test_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class AblationStudy:
    """
    消融实验研究
    系统性地移除模型组件以评估其重要性
    """
    
    def __init__(self, config: AblationConfig):
        """
        初始化消融实验
        
        Args:
            config: 消融实验配置
        """
        self.config = config
        self.study_id = f"ablation_{int(time.time()*1000)}"
        
        # 日志器
        self.logger = Logger(f"AblationStudy_{self.study_id}")
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 实验配置生成
        self.configurations = self._generate_configurations()
        
        # 结果存储
        self.results: Dict[str, AblationResult] = {}
        self.baseline_result: Optional[AblationResult] = None
        
        # 创建实验目录
        self.study_dir = f"experiments/ablation_studies/{self.study_id}"
        os.makedirs(self.study_dir, exist_ok=True)
        
        print(f"✅ 消融实验初始化完成: {config.study_name}")
        print(f"   研究ID: {self.study_id}")
        print(f"   配置数量: {len(self.configurations)}")
        print(f"   重复次数: {config.num_repetitions}")
    
    def run_study(self) -> Dict[str, AblationResult]:
        """
        运行完整的消融实验
        
        Returns:
            消融实验结果
        """
        study_start_time = time.time()
        
        self.logger.info(f"🚀 开始消融实验: {self.config.study_name}")
        self.logger.info(f"总配置数: {len(self.configurations)}")
        
        try:
            # 运行基线实验
            self.logger.info("📊 运行基线实验")
            self._run_baseline_experiments()
            
            # 运行消融实验
            for i, (config_name, settings) in enumerate(self.configurations.items()):
                self.logger.info(f"🔬 运行配置 {i+1}/{len(self.configurations)}: {config_name}")
                self._run_configuration(config_name, settings)
            
            # 分析结果
            self.logger.info("📈 分析消融结果")
            self._analyze_results()
            
            # 生成报告
            self.logger.info("📑 生成消融报告")
            self._generate_study_report()
            
            study_time = time.time() - study_start_time
            self.logger.info(f"✅ 消融实验完成，用时: {study_time:.2f}s")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"❌ 消融实验失败: {str(e)}")
            raise
    
    def _generate_configurations(self) -> Dict[str, ExperimentSettings]:
        """生成所有消融配置"""
        configurations = {}
        
        # 基线配置
        baseline_name = "baseline_all_components"
        configurations[baseline_name] = self.config.baseline_config
        
        # 单组件消融
        for component in self.config.components_to_ablate:
            config_name = f"ablate_{component.value}"
            ablated_config = self._create_ablated_config([component])
            configurations[config_name] = ablated_config
        
        # 组合消融（如果启用）
        if self.config.combination_ablation:
            for size in range(2, min(self.config.max_combination_size + 1, 
                                   len(self.config.components_to_ablate) + 1)):
                for combination in itertools.combinations(self.config.components_to_ablate, size):
                    config_name = f"ablate_{'_'.join([c.value for c in combination])}"
                    ablated_config = self._create_ablated_config(list(combination))
                    configurations[config_name] = ablated_config
        
        return configurations
    
    def _create_ablated_config(self, ablated_components: List[AblationComponent]) -> ExperimentSettings:
        """创建消融配置"""
        # 复制基线配置
        config = ExperimentSettings(
            experiment_name=f"{self.config.baseline_config.experiment_name}_ablated",
            experiment_type=self.config.baseline_config.experiment_type,
            description=f"消融组件: {[c.value for c in ablated_components]}",
            total_episodes=self.config.baseline_config.total_episodes,
            evaluation_frequency=self.config.baseline_config.evaluation_frequency,
            save_frequency=self.config.baseline_config.save_frequency,
            scenario_types=self.config.baseline_config.scenario_types,
            environment_variations=self.config.baseline_config.environment_variations,
            use_pretraining=self.config.baseline_config.use_pretraining,
            enable_hierarchical=self.config.baseline_config.enable_hierarchical,
            evaluation_episodes=self.config.baseline_config.evaluation_episodes,
            enable_visualization=False,  # 消融实验中禁用可视化以提高速度
            device=self.config.baseline_config.device,
            random_seed=self.config.baseline_config.random_seed
        )
        
        # 根据消融组件修改配置
        for component in ablated_components:
            config = self._apply_ablation(config, component)
        
        return config
    
    def _apply_ablation(self, config: ExperimentSettings, component: AblationComponent) -> ExperimentSettings:
        """应用特定组件的消融"""
        if component == AblationComponent.HIERARCHICAL_STRUCTURE:
            # 禁用分层结构，使用单层训练
            config.enable_hierarchical = False
            config.experiment_type = ExperimentType.SINGLE_OBJECTIVE
            
        elif component == AblationComponent.PRETRAINING:
            # 禁用预训练
            config.use_pretraining = False
            
        elif component == AblationComponent.MULTI_OBJECTIVE:
            # 使用单目标训练
            config.experiment_type = ExperimentType.SINGLE_OBJECTIVE
            
        # 其他组件的消融需要在模型层面实现
        # 这里记录消融的组件，实际的模型修改在训练器中进行
        
        return config
    
    def _run_baseline_experiments(self):
        """运行基线实验"""
        baseline_experiments = []
        
        for rep in range(self.config.num_repetitions):
            self.logger.info(f"基线实验重复 {rep + 1}/{self.config.num_repetitions}")
            
            # 设置不同的随机种子
            baseline_config = self.config.baseline_config
            if baseline_config.random_seed is not None:
                baseline_config.random_seed = baseline_config.random_seed + rep
            
            # 运行实验
            experiment = BasicExperiment(
                settings=baseline_config,
                experiment_id=f"{self.study_id}_baseline_rep{rep}"
            )
            
            result = experiment.run_experiment()
            baseline_experiments.append(result)
        
        # 创建基线结果
        self.baseline_result = AblationResult(
            configuration_name="baseline",
            ablated_components=[],
            experiment_results=baseline_experiments
        )
        
        # 计算基线统计
        self._compute_statistics(self.baseline_result)
        
        self.logger.info("基线实验完成")
    
    def _run_configuration(self, config_name: str, settings: ExperimentSettings):
        """运行特定配置的实验"""
        experiments = []
        
        # 从配置名称推断消融的组件
        ablated_components = self._extract_ablated_components(config_name)
        
        for rep in range(self.config.num_repetitions):
            self.logger.info(f"配置 {config_name} 重复 {rep + 1}/{self.config.num_repetitions}")
            
            # 设置不同的随机种子
            config_copy = settings
            if config_copy.random_seed is not None:
                config_copy.random_seed = config_copy.random_seed + rep
            
            # 运行实验
            experiment = BasicExperiment(
                settings=config_copy,
                experiment_id=f"{self.study_id}_{config_name}_rep{rep}"
            )
            
            result = experiment.run_experiment()
            experiments.append(result)
        
        # 创建消融结果
        ablation_result = AblationResult(
            configuration_name=config_name,
            ablated_components=ablated_components,
            experiment_results=experiments
        )
        
        # 计算统计
        self._compute_statistics(ablation_result)
        
        # 与基线比较
        if self.baseline_result:
            self._compare_with_baseline(ablation_result)
        
        # 统计显著性检验
        if self.config.significance_test and self.baseline_result:
            self._perform_significance_test(ablation_result)
        
        self.results[config_name] = ablation_result
        
        self.logger.info(f"配置 {config_name} 完成")
    
    def _extract_ablated_components(self, config_name: str) -> List[AblationComponent]:
        """从配置名称提取消融的组件"""
        ablated_components = []
        
        if config_name == "baseline_all_components":
            return ablated_components
        
        # 移除 "ablate_" 前缀
        if config_name.startswith("ablate_"):
            component_names = config_name[7:].split("_")
            
            for component_name in component_names:
                try:
                    component = AblationComponent(component_name)
                    ablated_components.append(component)
                except ValueError:
                    # 处理复合组件名称
                    for component in AblationComponent:
                        if component_name in component.value:
                            ablated_components.append(component)
                            break
        
        return ablated_components
    
    def _compute_statistics(self, result: AblationResult):
        """计算统计指标"""
        # 收集所有重复实验的指标
        metric_values = {}
        
        for exp_result in result.experiment_results:
            for metric_name in self.config.primary_metrics:
                if metric_name in exp_result.final_performance:
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(exp_result.final_performance[metric_name])
        
        # 计算均值和标准差
        for metric_name, values in metric_values.items():
            result.mean_performance[metric_name] = np.mean(values)
            result.std_performance[metric_name] = np.std(values)
    
    def _compare_with_baseline(self, result: AblationResult):
        """与基线比较"""
        for metric_name in self.config.primary_metrics:
            if (metric_name in result.mean_performance and 
                metric_name in self.baseline_result.mean_performance):
                
                baseline_mean = self.baseline_result.mean_performance[metric_name]
                ablation_mean = result.mean_performance[metric_name]
                
                # 绝对性能下降
                result.performance_drop[metric_name] = baseline_mean - ablation_mean
                
                # 相对性能下降
                if baseline_mean != 0:
                    result.relative_drop[metric_name] = (
                        (baseline_mean - ablation_mean) / baseline_mean * 100
                    )
                else:
                    result.relative_drop[metric_name] = 0.0
    
    def _perform_significance_test(self, result: AblationResult):
        """执行统计显著性检验"""
        from scipy import stats
        
        for metric_name in self.config.primary_metrics:
            if metric_name in result.mean_performance:
                # 收集基线和消融实验的数据
                baseline_values = []
                ablation_values = []
                
                for exp_result in self.baseline_result.experiment_results:
                    if metric_name in exp_result.final_performance:
                        baseline_values.append(exp_result.final_performance[metric_name])
                
                for exp_result in result.experiment_results:
                    if metric_name in exp_result.final_performance:
                        ablation_values.append(exp_result.final_performance[metric_name])
                
                if len(baseline_values) > 1 and len(ablation_values) > 1:
                    # 执行t检验
                    t_stat, p_value = stats.ttest_ind(baseline_values, ablation_values)
                    
                    # 计算效应大小（Cohen's d）
                    pooled_std = np.sqrt((np.var(baseline_values) + np.var(ablation_values)) / 2)
                    cohens_d = (np.mean(baseline_values) - np.mean(ablation_values)) / pooled_std
                    
                    result.significance_test_results[metric_name] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_significant': p_value < (1 - self.config.confidence_level),
                        'cohens_d': cohens_d,
                        'effect_size': self._interpret_effect_size(abs(cohens_d))
                    }
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """解释效应大小"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_results(self):
        """分析消融结果"""
        self.logger.info("开始结果分析")
        
        # 按重要性排序组件
        component_importance = {}
        
        for config_name, result in self.results.items():
            if result.ablated_components:  # 排除基线
                # 计算平均性能下降
                avg_drop = 0
                for metric_name in self.config.primary_metrics:
                    if metric_name in result.relative_drop:
                        avg_drop += abs(result.relative_drop[metric_name])
                
                avg_drop /= len(self.config.primary_metrics)
                
                # 如果是单组件消融，记录重要性
                if len(result.ablated_components) == 1:
                    component = result.ablated_components[0]
                    component_importance[component] = avg_drop
        
        # 排序并记录
        sorted_components = sorted(component_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        self.logger.info("组件重要性排序（按性能下降幅度）:")
        for component, importance in sorted_components:
            self.logger.info(f"  {component.value}: {importance:.2f}%")
    
    def _generate_study_report(self):
        """生成消融研究报告"""
        report = {
            'study_info': {
                'study_id': self.study_id,
                'study_name': self.config.study_name,
                'description': self.config.description,
                'num_configurations': len(self.configurations),
                'num_repetitions': self.config.num_repetitions,
                'primary_metrics': self.config.primary_metrics
            },
            'baseline_performance': {},
            'ablation_results': {},
            'component_importance_ranking': [],
            'summary_statistics': {},
            'recommendations': []
        }
        
        # 基线性能
        if self.baseline_result:
            report['baseline_performance'] = {
                'mean_performance': self.baseline_result.mean_performance,
                'std_performance': self.baseline_result.std_performance
            }
        
        # 消融结果
        for config_name, result in self.results.items():
            report['ablation_results'][config_name] = {
                'ablated_components': [c.value for c in result.ablated_components],
                'mean_performance': result.mean_performance,
                'std_performance': result.std_performance,
                'performance_drop': result.performance_drop,
                'relative_drop': result.relative_drop,
                'significance_test_results': result.significance_test_results
            }
        
        # 组件重要性排序
        component_importance = {}
        for result in self.results.values():
            if len(result.ablated_components) == 1:
                component = result.ablated_components[0]
                avg_drop = np.mean([abs(drop) for drop in result.relative_drop.values()])
                component_importance[component.value] = avg_drop
        
        sorted_importance = sorted(component_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        report['component_importance_ranking'] = sorted_importance
        
        # 汇总统计
        all_drops = []
        for result in self.results.values():
            for drop in result.relative_drop.values():
                all_drops.append(abs(drop))
        
        if all_drops:
            report['summary_statistics'] = {
                'avg_performance_drop': np.mean(all_drops),
                'max_performance_drop': np.max(all_drops),
                'min_performance_drop': np.min(all_drops),
                'std_performance_drop': np.std(all_drops)
            }
        
        # 生成建议
        if sorted_importance:
            most_important = sorted_importance[0]
            least_important = sorted_importance[-1]
            
            report['recommendations'] = [
                f"最重要组件: {most_important[0]} (性能下降 {most_important[1]:.2f}%)",
                f"最不重要组件: {least_important[0]} (性能下降 {least_important[1]:.2f}%)",
                "建议优先优化最重要的组件",
                "可以考虑简化最不重要的组件以提高效率"
            ]
        
        # 保存报告
        report_path = os.path.join(self.study_dir, "ablation_study_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"消融研究报告已保存: {report_path}")
        
        return report
    
    def get_component_importance(self) -> Dict[AblationComponent, float]:
        """获取组件重要性"""
        importance = {}
        
        for result in self.results.values():
            if len(result.ablated_components) == 1:
                component = result.ablated_components[0]
                avg_drop = np.mean([abs(drop) for drop in result.relative_drop.values()])
                importance[component] = avg_drop
        
        return importance
    
    def plot_results(self, save_path: Optional[str] = None):
        """绘制消融结果"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Ablation Study Results: {self.config.study_name}', fontsize=16)
            
            # 1. 组件重要性条形图
            importance = self.get_component_importance()
            if importance:
                components = list(importance.keys())
                values = list(importance.values())
                
                axes[0, 0].bar([c.value for c in components], values)
                axes[0, 0].set_title('Component Importance (Performance Drop %)')
                axes[0, 0].set_ylabel('Performance Drop (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. 性能对比图
            configs = []
            performances = []
            
            for config_name, result in self.results.items():
                if 'episode_reward' in result.mean_performance:
                    configs.append(config_name.replace('ablate_', ''))
                    performances.append(result.mean_performance['episode_reward'])
            
            if configs:
                axes[0, 1].bar(configs, performances)
                axes[0, 1].set_title('Performance Comparison')
                axes[0, 1].set_ylabel('Episode Reward')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. 相对性能下降热力图
            metrics = self.config.primary_metrics
            config_names = [name for name in self.results.keys() if name != 'baseline']
            
            if config_names and metrics:
                drop_matrix = []
                for config_name in config_names:
                    row = []
                    for metric in metrics:
                        drop = self.results[config_name].relative_drop.get(metric, 0)
                        row.append(drop)
                    drop_matrix.append(row)
                
                im = axes[1, 0].imshow(drop_matrix, cmap='Reds', aspect='auto')
                axes[1, 0].set_title('Relative Performance Drop (%)')
                axes[1, 0].set_xticks(range(len(metrics)))
                axes[1, 0].set_xticklabels(metrics)
                axes[1, 0].set_yticks(range(len(config_names)))
                axes[1, 0].set_yticklabels([name.replace('ablate_', '') for name in config_names])
                plt.colorbar(im, ax=axes[1, 0])
            
            # 4. 统计显著性
            significant_comparisons = []
            for result in self.results.values():
                for metric, test_result in result.significance_test_results.items():
                    if test_result['is_significant']:
                        significant_comparisons.append(f"{result.configuration_name}_{metric}")
            
            if significant_comparisons:
                axes[1, 1].bar(range(len(significant_comparisons)), 
                              [1] * len(significant_comparisons))
                axes[1, 1].set_title('Statistically Significant Differences')
                axes[1, 1].set_xticks(range(len(significant_comparisons)))
                axes[1, 1].set_xticklabels(significant_comparisons, rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"消融结果图表已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib不可用，无法绘制图表")
