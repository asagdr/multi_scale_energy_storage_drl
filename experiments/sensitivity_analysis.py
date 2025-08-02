import numpy as np
import torch
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import sys
import itertools

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType, ExperimentResults
from utils.logger import Logger
from utils.metrics import MetricsCalculator

class ParameterType(Enum):
    """参数类型枚举"""
    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    DISCOUNT_FACTOR = "discount_factor"
    EXPLORATION_RATE = "exploration_rate"
    NETWORK_HIDDEN_SIZE = "network_hidden_size"
    TARGET_UPDATE_FREQUENCY = "target_update_frequency"
    BUFFER_SIZE = "buffer_size"
    TEMPERATURE_COEFFICIENT = "temperature_coefficient"
    SOC_WEIGHT = "soc_weight"
    ENERGY_WEIGHT = "energy_weight"
    SAFETY_WEIGHT = "safety_weight"
    CONSTRAINT_PENALTY = "constraint_penalty"
    NOISE_LEVEL = "noise_level"
    EPISODE_LENGTH = "episode_length"
    CURRICULUM_DIFFICULTY = "curriculum_difficulty"

@dataclass
class ParameterRange:
    """参数范围定义"""
    param_type: ParameterType
    min_value: float
    max_value: float
    step_size: Optional[float] = None
    num_samples: int = 5
    scale: str = "linear"  # "linear" or "log"
    default_value: Optional[float] = None

@dataclass
class SensitivityConfig:
    """敏感性分析配置"""
    study_name: str
    description: str = ""
    
    # 要分析的参数
    parameters_to_analyze: List[ParameterRange] = field(default_factory=list)
    
    # 基线配置
    baseline_config: ExperimentSettings = None
    
    # 分析类型
    analysis_type: str = "one_at_a_time"  # "one_at_a_time", "factorial", "sobol"
    
    # 每个配置的重复次数
    num_repetitions: int = 3
    
    # 评估指标
    primary_metrics: List[str] = field(default_factory=lambda: [
        'episode_reward', 'tracking_accuracy', 'energy_efficiency'
    ])
    
    # 敏感性分析方法
    sensitivity_methods: List[str] = field(default_factory=lambda: [
        'local_sensitivity', 'global_sensitivity', 'sobol_indices'
    ])

@dataclass
class SensitivityResult:
    """敏感性分析结果"""
    parameter_config: Dict[ParameterType, float]
    experiment_results: List[ExperimentResults]
    
    # 统计指标
    mean_performance: Dict[str, float] = field(default_factory=dict)
    std_performance: Dict[str, float] = field(default_factory=dict)
    
    # 与基线的差异
    performance_difference: Dict[str, float] = field(default_factory=dict)
    relative_difference: Dict[str, float] = field(default_factory=dict)

@dataclass
class GlobalSensitivityResult:
    """全局敏感性分析结果"""
    parameter: ParameterType
    
    # 一阶敏感性指数
    first_order_index: Dict[str, float] = field(default_factory=dict)
    
    # 总敏感性指数
    total_index: Dict[str, float] = field(default_factory=dict)
    
    # 局部敏感性（梯度）
    local_sensitivity: Dict[str, float] = field(default_factory=dict)
    
    # 参数-性能关系
    parameter_response: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

class SensitivityAnalysis:
    """
    敏感性分析
    分析模型参数对性能的影响
    """
    
    def __init__(self, config: SensitivityConfig):
        """
        初始化敏感性分析
        
        Args:
            config: 敏感性分析配置
        """
        self.config = config
        self.study_id = f"sensitivity_{int(time.time()*1000)}"
        
        # 日志器
        self.logger = Logger(f"SensitivityAnalysis_{self.study_id}")
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 参数配置生成
        self.parameter_configurations = self._generate_parameter_configurations()
        
        # 结果存储
        self.results: Dict[str, SensitivityResult] = {}
        self.baseline_result: Optional[SensitivityResult] = None
        self.global_sensitivity_results: Dict[ParameterType, GlobalSensitivityResult] = {}
        
        # 创建研究目录
        self.study_dir = f"experiments/sensitivity_analysis/{self.study_id}"
        os.makedirs(self.study_dir, exist_ok=True)
        
        print(f"✅ 敏感性分析初始化完成: {config.study_name}")
        print(f"   研究ID: {self.study_id}")
        print(f"   参数配置数量: {len(self.parameter_configurations)}")
        print(f"   分析方法: {config.analysis_type}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        运行完整的敏感性分析
        
        Returns:
            敏感性分析结果
        """
        analysis_start_time = time.time()
        
        self.logger.info(f"🚀 开始敏感性分析: {self.config.study_name}")
        self.logger.info(f"分析类型: {self.config.analysis_type}")
        
        try:
            # 运行基线实验
            self.logger.info("📊 运行基线实验")
            self._run_baseline_experiments()
            
            # 运行参数变化实验
            self.logger.info("🔬 运行参数变化实验")
            self._run_parameter_experiments()
            
            # 计算敏感性指标
            self.logger.info("📈 计算敏感性指标")
            self._compute_sensitivity_indices()
            
            # 全局敏感性分析
            if "global_sensitivity" in self.config.sensitivity_methods:
                self.logger.info("🌐 执行全局敏感性分析")
                self._perform_global_sensitivity_analysis()
            
            # Sobol敏感性分析
            if "sobol_indices" in self.config.sensitivity_methods:
                self.logger.info("📊 执行Sobol敏感性分析")
                self._perform_sobol_analysis()
            
            # 生成分析报告
            self.logger.info("📑 生成敏感性分析报告")
            analysis_results = self._generate_analysis_report()
            
            analysis_time = time.time() - analysis_start_time
            self.logger.info(f"✅ 敏感性分析完成，用时: {analysis_time:.2f}s")
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"❌ 敏感性分析失败: {str(e)}")
            raise
    
    def _generate_parameter_configurations(self) -> List[Dict[ParameterType, float]]:
        """生成参数配置"""
        configurations = []
        
        if self.config.analysis_type == "one_at_a_time":
            # 一次一个参数变化
            configurations = self._generate_oat_configurations()
        elif self.config.analysis_type == "factorial":
            # 全因子设计
            configurations = self._generate_factorial_configurations()
        elif self.config.analysis_type == "sobol":
            # Sobol采样
            configurations = self._generate_sobol_configurations()
        
        return configurations
    
    def _generate_oat_configurations(self) -> List[Dict[ParameterType, float]]:
        """生成一次一个参数（OAT）配置"""
        configurations = []
        
        # 基线配置
        baseline_params = self._get_baseline_parameters()
        
        for param_range in self.config.parameters_to_analyze:
            param_values = self._generate_parameter_values(param_range)
            
            for value in param_values:
                config = baseline_params.copy()
                config[param_range.param_type] = value
                configurations.append(config)
        
        return configurations
    
    def _generate_factorial_configurations(self) -> List[Dict[ParameterType, float]]:
        """生成全因子设计配置"""
        # 为每个参数生成值
        parameter_values = {}
        for param_range in self.config.parameters_to_analyze:
            parameter_values[param_range.param_type] = self._generate_parameter_values(param_range)
        
        # 生成所有组合
        configurations = []
        param_types = list(parameter_values.keys())
        value_lists = list(parameter_values.values())
        
        for combination in itertools.product(*value_lists):
            config = {}
            for i, param_type in enumerate(param_types):
                config[param_type] = combination[i]
            
            # 添加未变化的基线参数
            baseline_params = self._get_baseline_parameters()
            for param_type, value in baseline_params.items():
                if param_type not in config:
                    config[param_type] = value
            
            configurations.append(config)
        
        return configurations
    
    def _generate_sobol_configurations(self) -> List[Dict[ParameterType, float]]:
        """生成Sobol采样配置"""
        try:
            from SALib.sample import sobol
            
            # 定义参数范围
            problem = {
                'num_vars': len(self.config.parameters_to_analyze),
                'names': [p.param_type.value for p in self.config.parameters_to_analyze],
                'bounds': [[p.min_value, p.max_value] for p in self.config.parameters_to_analyze]
            }
            
            # 生成Sobol样本
            num_samples = 1024  # Sobol样本数量
            samples = sobol.sample(problem, num_samples)
            
            # 转换为配置
            configurations = []
            baseline_params = self._get_baseline_parameters()
            
            for sample in samples:
                config = baseline_params.copy()
                for i, param_range in enumerate(self.config.parameters_to_analyze):
                    config[param_range.param_type] = sample[i]
                configurations.append(config)
            
            return configurations
            
        except ImportError:
            self.logger.warning("SALib不可用，使用随机采样代替Sobol采样")
            return self._generate_random_configurations()
    
    def _generate_random_configurations(self) -> List[Dict[ParameterType, float]]:
        """生成随机配置"""
        configurations = []
        baseline_params = self._get_baseline_parameters()
        
        num_samples = 100  # 随机样本数量
        for _ in range(num_samples):
            config = baseline_params.copy()
            for param_range in self.config.parameters_to_analyze:
                if param_range.scale == "log":
                    log_min = np.log10(param_range.min_value)
                    log_max = np.log10(param_range.max_value)
                    log_value = np.random.uniform(log_min, log_max)
                    value = 10 ** log_value
                else:
                    value = np.random.uniform(param_range.min_value, param_range.max_value)
                
                config[param_range.param_type] = value
            
            configurations.append(config)
        
        return configurations
    
    def _generate_parameter_values(self, param_range: ParameterRange) -> List[float]:
        """为单个参数生成值"""
        if param_range.step_size:
            # 使用步长
            if param_range.scale == "log":
                log_min = np.log10(param_range.min_value)
                log_max = np.log10(param_range.max_value)
                log_step = np.log10(param_range.step_size)
                log_values = np.arange(log_min, log_max + log_step, log_step)
                values = [10 ** log_val for log_val in log_values]
            else:
                values = list(np.arange(param_range.min_value, 
                                      param_range.max_value + param_range.step_size, 
                                      param_range.step_size))
        else:
            # 使用样本数量
            if param_range.scale == "log":
                log_min = np.log10(param_range.min_value)
                log_max = np.log10(param_range.max_value)
                log_values = np.linspace(log_min, log_max, param_range.num_samples)
                values = [10 ** log_val for log_val in log_values]
            else:
                values = list(np.linspace(param_range.min_value, 
                                        param_range.max_value, 
                                        param_range.num_samples))
        
        return values
    
    def _get_baseline_parameters(self) -> Dict[ParameterType, float]:
        """获取基线参数"""
        # 这里定义默认参数值
        baseline = {
            ParameterType.LEARNING_RATE: 0.001,
            ParameterType.BATCH_SIZE: 32,
            ParameterType.DISCOUNT_FACTOR: 0.99,
            ParameterType.EXPLORATION_RATE: 0.1,
            ParameterType.NETWORK_HIDDEN_SIZE: 256,
            ParameterType.TARGET_UPDATE_FREQUENCY: 100,
            ParameterType.BUFFER_SIZE: 10000,
            ParameterType.TEMPERATURE_COEFFICIENT: 1.0,
            ParameterType.SOC_WEIGHT: 1.0,
            ParameterType.ENERGY_WEIGHT: 1.0,
            ParameterType.SAFETY_WEIGHT: 2.0,
            ParameterType.CONSTRAINT_PENALTY: 10.0,
            ParameterType.NOISE_LEVEL: 0.01,
            ParameterType.EPISODE_LENGTH: 1000,
            ParameterType.CURRICULUM_DIFFICULTY: 1.0
        }
        
        # 使用用户提供的默认值覆盖
        for param_range in self.config.parameters_to_analyze:
            if param_range.default_value is not None:
                baseline[param_range.param_type] = param_range.default_value
        
        return baseline
    
    def _run_baseline_experiments(self):
        """运行基线实验"""
        baseline_experiments = []
        baseline_params = self._get_baseline_parameters()
        
        for rep in range(self.config.num_repetitions):
            self.logger.info(f"基线实验重复 {rep + 1}/{self.config.num_repetitions}")
            
            # 创建基线配置
            baseline_config = self._create_experiment_config(baseline_params, rep)
            
            # 运行实验
            experiment = BasicExperiment(
                settings=baseline_config,
                experiment_id=f"{self.study_id}_baseline_rep{rep}"
            )
            
            result = experiment.run_experiment()
            baseline_experiments.append(result)
        
        # 创建基线结果
        self.baseline_result = SensitivityResult(
            parameter_config=baseline_params,
            experiment_results=baseline_experiments
        )
        
        # 计算基线统计
        self._compute_result_statistics(self.baseline_result)
        
        self.logger.info("基线实验完成")
    
    def _run_parameter_experiments(self):
        """运行参数变化实验"""
        total_configs = len(self.parameter_configurations)
        
        for i, param_config in enumerate(self.parameter_configurations):
            config_name = f"config_{i}"
            self.logger.info(f"运行配置 {i+1}/{total_configs}: {config_name}")
            
            experiments = []
            
            for rep in range(self.config.num_repetitions):
                # 创建实验配置
                exp_config = self._create_experiment_config(param_config, rep)
                
                # 运行实验
                experiment = BasicExperiment(
                    settings=exp_config,
                    experiment_id=f"{self.study_id}_{config_name}_rep{rep}"
                )
                
                result = experiment.run_experiment()
                experiments.append(result)
            
            # 创建结果
            sensitivity_result = SensitivityResult(
                parameter_config=param_config,
                experiment_results=experiments
            )
            
            # 计算统计
            self._compute_result_statistics(sensitivity_result)
            
            # 与基线比较
            if self.baseline_result:
                self._compare_with_baseline(sensitivity_result)
            
            self.results[config_name] = sensitivity_result
    
    def _create_experiment_config(self, param_config: Dict[ParameterType, float], rep: int) -> ExperimentSettings:
        """创建实验配置"""
        config = ExperimentSettings(
            experiment_name=f"sensitivity_analysis_{self.study_id}",
            experiment_type=self.config.baseline_config.experiment_type,
            description=f"敏感性分析配置",
            total_episodes=200,  # 减少回合数以提高分析速度
            evaluation_frequency=50,
            save_frequency=100,
            scenario_types=self.config.baseline_config.scenario_types,
            environment_variations=2,  # 减少环境变化
            use_pretraining=False,  # 禁用预训练以提高速度
            enable_hierarchical=self.config.baseline_config.enable_hierarchical,
            evaluation_episodes=20,  # 减少评估回合
            enable_visualization=False,  # 禁用可视化
            device=self.config.baseline_config.device,
            random_seed=42 + rep if self.config.baseline_config.random_seed else None
        )
        
        # 注意：实际实现中需要将参数配置传递给训练器
        # 这里只是示例，实际需要修改训练配置
        
        return config
    
    def _compute_result_statistics(self, result: SensitivityResult):
        """计算结果统计"""
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
    
    def _compare_with_baseline(self, result: SensitivityResult):
        """与基线比较"""
        for metric_name in self.config.primary_metrics:
            if (metric_name in result.mean_performance and 
                metric_name in self.baseline_result.mean_performance):
                
                baseline_mean = self.baseline_result.mean_performance[metric_name]
                result_mean = result.mean_performance[metric_name]
                
                # 绝对差异
                result.performance_difference[metric_name] = result_mean - baseline_mean
                
                # 相对差异
                if baseline_mean != 0:
                    result.relative_difference[metric_name] = (
                        (result_mean - baseline_mean) / baseline_mean * 100
                    )
                else:
                    result.relative_difference[metric_name] = 0.0
    
    def _compute_sensitivity_indices(self):
        """计算敏感性指标"""
        for param_range in self.config.parameters_to_analyze:
            param_type = param_range.param_type
            
            # 收集该参数的所有结果
            param_results = []
            param_values = []
            
            for config_name, result in self.results.items():
                if param_type in result.parameter_config:
                    param_values.append(result.parameter_config[param_type])
                    param_results.append(result)
            
            if len(param_results) > 1:
                # 计算局部敏感性（梯度）
                local_sens = self._compute_local_sensitivity(param_values, param_results)
                
                # 存储结果
                if param_type not in self.global_sensitivity_results:
                    self.global_sensitivity_results[param_type] = GlobalSensitivityResult(parameter=param_type)
                
                self.global_sensitivity_results[param_type].local_sensitivity = local_sens
                
                # 存储参数-性能关系
                for metric_name in self.config.primary_metrics:
                    if metric_name not in self.global_sensitivity_results[param_type].parameter_response:
                        self.global_sensitivity_results[param_type].parameter_response[metric_name] = []
                    
                    for i, result in enumerate(param_results):
                        if metric_name in result.mean_performance:
                            self.global_sensitivity_results[param_type].parameter_response[metric_name].append(
                                (param_values[i], result.mean_performance[metric_name])
                            )
    
    def _compute_local_sensitivity(self, param_values: List[float], results: List[SensitivityResult]) -> Dict[str, float]:
        """计算局部敏感性"""
        local_sensitivity = {}
        
        for metric_name in self.config.primary_metrics:
            metric_values = []
            valid_params = []
            
            for i, result in enumerate(results):
                if metric_name in result.mean_performance:
                    metric_values.append(result.mean_performance[metric_name])
                    valid_params.append(param_values[i])
            
            if len(metric_values) > 1:
                # 计算数值梯度
                param_array = np.array(valid_params)
                metric_array = np.array(metric_values)
                
                # 排序以便计算梯度
                sorted_indices = np.argsort(param_array)
                sorted_params = param_array[sorted_indices]
                sorted_metrics = metric_array[sorted_indices]
                
                # 计算梯度（中心差分）
                gradients = []
                for i in range(1, len(sorted_params) - 1):
                    grad = (sorted_metrics[i+1] - sorted_metrics[i-1]) / (sorted_params[i+1] - sorted_params[i-1])
                    gradients.append(grad)
                
                if gradients:
                    local_sensitivity[metric_name] = np.mean(np.abs(gradients))
                else:
                    local_sensitivity[metric_name] = 0.0
        
        return local_sensitivity
    
    def _perform_global_sensitivity_analysis(self):
        """执行全局敏感性分析"""
        # 使用方差分解方法
        for param_type in [p.param_type for p in self.config.parameters_to_analyze]:
            if param_type in self.global_sensitivity_results:
                result = self.global_sensitivity_results[param_type]
                
                # 计算一阶敏感性指数
                first_order = self._compute_first_order_sensitivity(param_type)
                result.first_order_index = first_order
                
                # 计算总敏感性指数
                total = self._compute_total_sensitivity(param_type)
                result.total_index = total
    
    def _compute_first_order_sensitivity(self, param_type: ParameterType) -> Dict[str, float]:
        """计算一阶敏感性指数"""
        first_order_indices = {}
        
        # 收集数据
        param_values = []
        metric_data = {metric: [] for metric in self.config.primary_metrics}
        
        for result in self.results.values():
            if param_type in result.parameter_config:
                param_values.append(result.parameter_config[param_type])
                for metric_name in self.config.primary_metrics:
                    if metric_name in result.mean_performance:
                        metric_data[metric_name].append(result.mean_performance[metric_name])
                    else:
                        metric_data[metric_name].append(0)
        
        # 计算敏感性指数
        for metric_name, values in metric_data.items():
            if len(values) > 1:
                # 使用皮尔逊相关系数的平方作为近似
                correlation = np.corrcoef(param_values, values)[0, 1]
                first_order_indices[metric_name] = correlation ** 2
            else:
                first_order_indices[metric_name] = 0.0
        
        return first_order_indices
    
    def _compute_total_sensitivity(self, param_type: ParameterType) -> Dict[str, float]:
        """计算总敏感性指数"""
        # 简化实现：总敏感性 = 一阶敏感性 + 一些交互项估计
        total_indices = {}
        
        if param_type in self.global_sensitivity_results:
            first_order = self.global_sensitivity_results[param_type].first_order_index
            
            for metric_name, first_order_value in first_order.items():
                # 简化估计：总敏感性稍大于一阶敏感性
                total_indices[metric_name] = min(1.0, first_order_value * 1.2)
        
        return total_indices
    
    def _perform_sobol_analysis(self):
        """执行Sobol敏感性分析"""
        try:
            from SALib.analyze import sobol
            
            # 准备Sobol分析的数据
            problem = {
                'num_vars': len(self.config.parameters_to_analyze),
                'names': [p.param_type.value for p in self.config.parameters_to_analyze],
                'bounds': [[p.min_value, p.max_value] for p in self.config.parameters_to_analyze]
            }
            
            # 收集输出数据
            for metric_name in self.config.primary_metrics:
                Y = []
                for result in self.results.values():
                    if metric_name in result.mean_performance:
                        Y.append(result.mean_performance[metric_name])
                    else:
                        Y.append(0)
                
                if len(Y) > 0:
                    Y = np.array(Y)
                    
                    # 执行Sobol分析
                    Si = sobol.analyze(problem, Y, print_to_console=False)
                    
                    # 存储结果
                    for i, param_range in enumerate(self.config.parameters_to_analyze):
                        param_type = param_range.param_type
                        
                        if param_type not in self.global_sensitivity_results:
                            self.global_sensitivity_results[param_type] = GlobalSensitivityResult(parameter=param_type)
                        
                        self.global_sensitivity_results[param_type].first_order_index[metric_name] = Si['S1'][i]
                        self.global_sensitivity_results[param_type].total_index[metric_name] = Si['ST'][i]
            
        except ImportError:
            self.logger.warning("SALib不可用，跳过Sobol分析")
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """生成敏感性分析报告"""
        report = {
            'study_info': {
                'study_id': self.study_id,
                'study_name': self.config.study_name,
                'description': self.config.description,
                'analysis_type': self.config.analysis_type,
                'num_configurations': len(self.parameter_configurations),
                'num_repetitions': self.config.num_repetitions,
                'primary_metrics': self.config.primary_metrics
            },
            'baseline_performance': {},
            'parameter_sensitivity': {},
            'global_sensitivity': {},
            'parameter_rankings': {},
            'recommendations': []
        }
        
        # 基线性能
        if self.baseline_result:
            report['baseline_performance'] = {
                'mean_performance': self.baseline_result.mean_performance,
                'std_performance': self.baseline_result.std_performance
            }
        
        # 参数敏感性
        for param_type, result in self.global_sensitivity_results.items():
            report['parameter_sensitivity'][param_type.value] = {
                'local_sensitivity': result.local_sensitivity,
                'first_order_index': result.first_order_index,
                'total_index': result.total_index,
                'parameter_response': {
                    metric: [[p, v] for p, v in points] 
                    for metric, points in result.parameter_response.items()
                }
            }
        
        # 参数重要性排序
        for metric_name in self.config.primary_metrics:
            rankings = []
            for param_type, result in self.global_sensitivity_results.items():
                if metric_name in result.total_index:
                    rankings.append((param_type.value, result.total_index[metric_name]))
            
            rankings.sort(key=lambda x: x[1], reverse=True)
            report['parameter_rankings'][metric_name] = rankings
        
        # 生成建议
        if report['parameter_rankings']:
            for metric_name, rankings in report['parameter_rankings'].items():
                if rankings:
                    most_sensitive = rankings[0]
                    least_sensitive = rankings[-1]
                    
                    report['recommendations'].append(
                        f"对于{metric_name}，最敏感参数是{most_sensitive[0]}（敏感性指数：{most_sensitive[1]:.3f}）"
                    )
                    report['recommendations'].append(
                        f"对于{metric_name}，最不敏感参数是{least_sensitive[0]}（敏感性指数：{least_sensitive[1]:.3f}）"
                    )
        
        # 保存报告
        report_path = os.path.join(self.study_dir, "sensitivity_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"敏感性分析报告已保存: {report_path}")
        
        return report
    
    def get_most_sensitive_parameters(self, metric_name: str, top_k: int = 5) -> List[Tuple[ParameterType, float]]:
        """获取最敏感的参数"""
        sensitivities = []
        
        for param_type, result in self.global_sensitivity_results.items():
            if metric_name in result.total_index:
                sensitivities.append((param_type, result.total_index[metric_name]))
        
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        return sensitivities[:top_k]
    
    def plot_sensitivity_results(self, save_path: Optional[str] = None):
        """绘制敏感性分析结果"""
        try:
            import matplotlib.pyplot as plt
            
            num_params = len(self.global_sensitivity_results)
            num_metrics = len(self.config.primary_metrics)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Sensitivity Analysis Results: {self.config.study_name}', fontsize=16)
            
            # 1. 总敏感性指数热力图
            param_names = [p.value for p in self.global_sensitivity_results.keys()]
            sensitivity_matrix = []
            
            for param_type in self.global_sensitivity_results.keys():
                row = []
                for metric_name in self.config.primary_metrics:
                    if metric_name in self.global_sensitivity_results[param_type].total_index:
                        row.append(self.global_sensitivity_results[param_type].total_index[metric_name])
                    else:
                        row.append(0)
                sensitivity_matrix.append(row)
            
            if sensitivity_matrix:
                im = axes[0, 0].imshow(sensitivity_matrix, cmap='Reds', aspect='auto')
                axes[0, 0].set_title('Total Sensitivity Index')
                axes[0, 0].set_xticks(range(len(self.config.primary_metrics)))
                axes[0, 0].set_xticklabels(self.config.primary_metrics)
                axes[0, 0].set_yticks(range(len(param_names)))
                axes[0, 0].set_yticklabels(param_names)
                plt.colorbar(im, ax=axes[0, 0])
            
            # 2. 参数重要性排序
            if self.config.primary_metrics:
                metric = self.config.primary_metrics[0]
                rankings = self.get_most_sensitive_parameters(metric)
                
                if rankings:
                    params = [r[0].value for r in rankings]
                    values = [r[1] for r in rankings]
                    
                    axes[0, 1].bar(params, values)
                    axes[0, 1].set_title(f'Parameter Importance for {metric}')
                    axes[0, 1].set_ylabel('Total Sensitivity Index')
                    axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. 参数响应曲线
            if self.global_sensitivity_results and self.config.primary_metrics:
                param_type = list(self.global_sensitivity_results.keys())[0]
                metric = self.config.primary_metrics[0]
                
                if metric in self.global_sensitivity_results[param_type].parameter_response:
                    response_data = self.global_sensitivity_results[param_type].parameter_response[metric]
                    if response_data:
                        x_vals = [point[0] for point in response_data]
                        y_vals = [point[1] for point in response_data]
                        
                        axes[1, 0].scatter(x_vals, y_vals, alpha=0.7)
                        axes[1, 0].set_title(f'{param_type.value} vs {metric}')
                        axes[1, 0].set_xlabel(param_type.value)
                        axes[1, 0].set_ylabel(metric)
            
            # 4. 敏感性指数对比
            first_order_values = []
            total_values = []
            param_labels = []
            
            for param_type, result in self.global_sensitivity_results.items():
                if self.config.primary_metrics[0] in result.first_order_index:
                    first_order_values.append(result.first_order_index[self.config.primary_metrics[0]])
                    total_values.append(result.total_index[self.config.primary_metrics[0]])
                    param_labels.append(param_type.value)
            
            if first_order_values:
                x = np.arange(len(param_labels))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, first_order_values, width, label='First Order', alpha=0.8)
                axes[1, 1].bar(x + width/2, total_values, width, label='Total', alpha=0.8)
                axes[1, 1].set_title('Sensitivity Index Comparison')
                axes[1, 1].set_ylabel('Sensitivity Index')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(param_labels, rotation=45)
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"敏感性分析图表已保存: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib不可用，无法绘制图表")
