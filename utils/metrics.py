import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class MetricType(Enum):
    """指标类型枚举"""
    ACCURACY = "accuracy"                    # 准确性指标
    EFFICIENCY = "efficiency"                # 效率指标
    STABILITY = "stability"                  # 稳定性指标
    PERFORMANCE = "performance"              # 性能指标
    ECONOMIC = "economic"                    # 经济性指标
    ENVIRONMENTAL = "environmental"          # 环境指标
    SAFETY = "safety"                       # 安全性指标
    ROBUSTNESS = "robustness"               # 鲁棒性指标
    CONVERGENCE = "convergence"             # 收敛性指标
    MULTI_OBJECTIVE = "multi_objective"     # 多目标指标

@dataclass
class MetricResult:
    """指标计算结果"""
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str = ""
    description: str = ""
    confidence_interval: Optional[Tuple[float, float]] = None
    statistical_significance: Optional[float] = None
    benchmark_comparison: Optional[Dict[str, float]] = None
    calculation_time: float = field(default_factory=time.time)

@dataclass
class MetricSuite:
    """指标套件"""
    suite_id: str
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    summary_statistics: Dict[str, float] = field(default_factory=dict)
    calculation_timestamp: float = field(default_factory=time.time)
    data_quality_score: float = 0.0

class MetricsCalculator:
    """
    评估指标计算器
    提供全面的性能评估指标计算
    """
    
    def __init__(self, calculator_id: str = "MetricsCalculator_001"):
        """
        初始化指标计算器
        
        Args:
            calculator_id: 计算器ID
        """
        self.calculator_id = calculator_id
        
        # === 指标计算方法映射 ===
        self.metric_calculators = {
            # 准确性指标
            'tracking_accuracy': self._calc_tracking_accuracy,
            'prediction_accuracy': self._calc_prediction_accuracy,
            'control_accuracy': self._calc_control_accuracy,
            'estimation_accuracy': self._calc_estimation_accuracy,
            
            # 效率指标
            'energy_efficiency': self._calc_energy_efficiency,
            'computational_efficiency': self._calc_computational_efficiency,
            'power_efficiency': self._calc_power_efficiency,
            'round_trip_efficiency': self._calc_round_trip_efficiency,
            
            # 稳定性指标
            'voltage_stability': self._calc_voltage_stability,
            'temperature_stability': self._calc_temperature_stability,
            'soc_stability': self._calc_soc_stability,
            'system_stability': self._calc_system_stability,
            
            # 性能指标
            'response_time': self._calc_response_time,
            'settling_time': self._calc_settling_time,
            'overshoot': self._calc_overshoot,
            'steady_state_error': self._calc_steady_state_error,
            
            # 经济性指标
            'operation_cost': self._calc_operation_cost,
            'maintenance_cost': self._calc_maintenance_cost,
            'lifecycle_cost': self._calc_lifecycle_cost,
            'revenue_optimization': self._calc_revenue_optimization,
            
            # 环境指标
            'carbon_footprint': self._calc_carbon_footprint,
            'energy_waste': self._calc_energy_waste,
            'environmental_impact': self._calc_environmental_impact,
            
            # 安全性指标
            'safety_margin': self._calc_safety_margin,
            'fault_tolerance': self._calc_fault_tolerance,
            'emergency_response': self._calc_emergency_response,
            
            # 鲁棒性指标
            'noise_robustness': self._calc_noise_robustness,
            'parameter_sensitivity': self._calc_parameter_sensitivity,
            'disturbance_rejection': self._calc_disturbance_rejection,
            
            # 收敛性指标
            'convergence_rate': self._calc_convergence_rate,
            'convergence_stability': self._calc_convergence_stability,
            'learning_efficiency': self._calc_learning_efficiency,
            
            # 多目标指标
            'pareto_efficiency': self._calc_pareto_efficiency,
            'hypervolume': self._calc_hypervolume,
            'objective_balance': self._calc_objective_balance
        }
        
        # === 基准值 ===
        self.benchmarks = {
            'tracking_accuracy': {'good': 0.95, 'acceptable': 0.85, 'poor': 0.7},
            'energy_efficiency': {'good': 0.92, 'acceptable': 0.85, 'poor': 0.75},
            'response_time': {'good': 0.05, 'acceptable': 0.1, 'poor': 0.2},  # 秒
            'safety_margin': {'good': 0.8, 'acceptable': 0.6, 'poor': 0.4},
            'system_stability': {'good': 0.95, 'acceptable': 0.85, 'poor': 0.7}
        }
        
        # === 权重配置 ===
        self.metric_weights = {
            MetricType.ACCURACY: 0.25,
            MetricType.EFFICIENCY: 0.20,
            MetricType.STABILITY: 0.15,
            MetricType.PERFORMANCE: 0.15,
            MetricType.SAFETY: 0.15,
            MetricType.ECONOMIC: 0.10
        }
        
        # === 计算统计 ===
        self.calculation_stats = {
            'total_calculations': 0,
            'calculations_by_type': {metric_type: 0 for metric_type in MetricType},
            'calculation_time': 0.0,
            'failed_calculations': 0
        }
        
        print(f"✅ 指标计算器初始化完成: {calculator_id}")
        print(f"   支持指标: {len(self.metric_calculators)} 个")
    
    def calculate_metrics(self,
                         data: Dict[str, np.ndarray],
                         target_data: Optional[Dict[str, np.ndarray]] = None,
                         metric_names: Optional[List[str]] = None,
                         suite_id: Optional[str] = None) -> MetricSuite:
        """
        计算指标套件
        
        Args:
            data: 实际数据
            target_data: 目标/参考数据
            metric_names: 要计算的指标名称列表
            suite_id: 套件ID
            
        Returns:
            计算结果套件
        """
        calculation_start_time = time.time()
        
        if suite_id is None:
            suite_id = f"metrics_suite_{int(time.time()*1000)}"
        
        if metric_names is None:
            metric_names = list(self.metric_calculators.keys())
        
        # 初始化结果套件
        suite = MetricSuite(suite_id=suite_id)
        
        # 评估数据质量
        suite.data_quality_score = self._assess_data_quality(data)
        
        # 逐一计算指标
        successful_calculations = 0
        for metric_name in metric_names:
            if metric_name in self.metric_calculators:
                try:
                    calculator = self.metric_calculators[metric_name]
                    result = calculator(data, target_data)
                    
                    if result is not None:
                        # 添加基准比较
                        if metric_name in self.benchmarks:
                            result.benchmark_comparison = self._compare_with_benchmark(
                                result.value, metric_name
                            )
                        
                        suite.metrics[metric_name] = result
                        successful_calculations += 1
                        
                        # 更新统计
                        self.calculation_stats['calculations_by_type'][result.metric_type] += 1
                    
                except Exception as e:
                    print(f"⚠️ 指标 {metric_name} 计算失败: {str(e)}")
                    self.calculation_stats['failed_calculations'] += 1
        
        # 计算汇总统计
        suite.summary_statistics = self._calculate_summary_statistics(suite)
        
        # 更新统计
        calculation_time = time.time() - calculation_start_time
        self.calculation_stats['total_calculations'] += 1
        self.calculation_stats['calculation_time'] += calculation_time
        
        print(f"✅ 指标计算完成: {suite_id}")
        print(f"   成功计算: {successful_calculations}/{len(metric_names)} 个指标")
        
        return suite
    
    def _calc_tracking_accuracy(self, data: Dict[str, np.ndarray], 
                               target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算跟踪精度"""
        if target_data is None or 'power_reference' not in target_data or 'power_actual' not in data:
            return None
        
        reference = target_data['power_reference']
        actual = data['power_actual']
        
        # 确保长度一致
        min_length = min(len(reference), len(actual))
        reference = reference[:min_length]
        actual = actual[:min_length]
        
        # 计算跟踪误差
        tracking_error = np.abs(actual - reference)
        max_error = np.max(np.abs(reference)) if np.max(np.abs(reference)) > 0 else 1.0
        
        # 跟踪精度 = 1 - 平均相对误差
        accuracy = 1.0 - np.mean(tracking_error) / max_error
        accuracy = max(0.0, accuracy)
        
        return MetricResult(
            metric_name="tracking_accuracy",
            metric_type=MetricType.ACCURACY,
            value=accuracy,
            unit="",
            description="功率跟踪精度，值越接近1越好",
            confidence_interval=self._calculate_confidence_interval(tracking_error / max_error)
        )
    
    def _calc_prediction_accuracy(self, data: Dict[str, np.ndarray], 
                                 target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算预测精度"""
        if target_data is None or 'predicted' not in data or 'actual' not in target_data:
            return None
        
        predicted = data['predicted']
        actual = target_data['actual']
        
        min_length = min(len(predicted), len(actual))
        predicted = predicted[:min_length]
        actual = actual[:min_length]
        
        # 计算R²得分
        r2 = r2_score(actual, predicted)
        
        return MetricResult(
            metric_name="prediction_accuracy",
            metric_type=MetricType.ACCURACY,
            value=max(0.0, r2),
            unit="",
            description="预测精度R²得分，值越接近1越好"
        )
    
    def _calc_control_accuracy(self, data: Dict[str, np.ndarray], 
                              target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算控制精度"""
        if 'control_error' not in data:
            return None
        
        control_error = data['control_error']
        
        # 控制精度基于误差分布
        rms_error = np.sqrt(np.mean(control_error**2))
        max_error = np.max(np.abs(control_error))
        
        # 精度 = 1 / (1 + 归一化RMS误差)
        accuracy = 1.0 / (1.0 + rms_error / (max_error + 1e-6))
        
        return MetricResult(
            metric_name="control_accuracy",
            metric_type=MetricType.ACCURACY,
            value=accuracy,
            unit="",
            description="控制精度，基于RMS误差计算"
        )
    
    def _calc_estimation_accuracy(self, data: Dict[str, np.ndarray], 
                                 target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算估计精度"""
        if target_data is None or 'estimated_soc' not in data or 'true_soc' not in target_data:
            return None
        
        estimated = data['estimated_soc']
        true_soc = target_data['true_soc']
        
        min_length = min(len(estimated), len(true_soc))
        estimated = estimated[:min_length]
        true_soc = true_soc[:min_length]
        
        # MAE基于的精度
        mae = mean_absolute_error(true_soc, estimated)
        accuracy = 1.0 - mae / 100.0  # SOC范围0-100%
        
        return MetricResult(
            metric_name="estimation_accuracy",
            metric_type=MetricType.ACCURACY,
            value=max(0.0, accuracy),
            unit="",
            description="SOC估计精度"
        )
    
    def _calc_energy_efficiency(self, data: Dict[str, np.ndarray], 
                               target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算能量效率"""
        if 'energy_input' not in data or 'energy_output' not in data:
            return None
        
        energy_input = data['energy_input']
        energy_output = data['energy_output']
        
        # 总效率
        total_input = np.sum(energy_input[energy_input > 0])
        total_output = np.sum(energy_output[energy_output > 0])
        
        efficiency = total_output / (total_input + 1e-6) if total_input > 0 else 0
        efficiency = min(1.0, efficiency)  # 效率不能超过100%
        
        return MetricResult(
            metric_name="energy_efficiency",
            metric_type=MetricType.EFFICIENCY,
            value=efficiency,
            unit="",
            description="总体能量效率"
        )
    
    def _calc_computational_efficiency(self, data: Dict[str, np.ndarray], 
                                     target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算计算效率"""
        if 'computation_time' not in data or 'performance_score' not in data:
            return None
        
        computation_time = data['computation_time']
        performance_score = data['performance_score']
        
        # 计算效率 = 性能 / 时间
        avg_performance = np.mean(performance_score)
        avg_time = np.mean(computation_time)
        
        efficiency = avg_performance / (avg_time + 1e-6)
        
        return MetricResult(
            metric_name="computational_efficiency",
            metric_type=MetricType.EFFICIENCY,
            value=efficiency,
            unit="score/s",
            description="计算效率：性能得分每秒"
        )
    
    def _calc_power_efficiency(self, data: Dict[str, np.ndarray], 
                              target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算功率效率"""
        if 'power_delivered' not in data or 'power_consumed' not in data:
            return None
        
        power_delivered = data['power_delivered']
        power_consumed = data['power_consumed']
        
        # 功率效率
        delivered = np.sum(power_delivered[power_delivered > 0])
        consumed = np.sum(power_consumed[power_consumed > 0])
        
        efficiency = delivered / (consumed + 1e-6) if consumed > 0 else 0
        efficiency = min(1.0, efficiency)
        
        return MetricResult(
            metric_name="power_efficiency",
            metric_type=MetricType.EFFICIENCY,
            value=efficiency,
            unit="",
            description="功率传输效率"
        )
    
    def _calc_round_trip_efficiency(self, data: Dict[str, np.ndarray], 
                                   target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算往返效率"""
        if 'charge_energy' not in data or 'discharge_energy' not in data:
            return None
        
        charge_energy = data['charge_energy']
        discharge_energy = data['discharge_energy']
        
        total_charge = np.sum(charge_energy[charge_energy > 0])
        total_discharge = np.sum(discharge_energy[discharge_energy > 0])
        
        rt_efficiency = total_discharge / (total_charge + 1e-6) if total_charge > 0 else 0
        rt_efficiency = min(1.0, rt_efficiency)
        
        return MetricResult(
            metric_name="round_trip_efficiency",
            metric_type=MetricType.EFFICIENCY,
            value=rt_efficiency,
            unit="",
            description="往返效率：放电能量/充电能量"
        )
    
    def _calc_voltage_stability(self, data: Dict[str, np.ndarray], 
                               target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算电压稳定性"""
        if 'voltage' not in data:
            return None
        
        voltage = data['voltage']
        
        # 电压稳定性基于变异系数
        cv = np.std(voltage) / (np.mean(voltage) + 1e-6)
        stability = 1.0 / (1.0 + cv)
        
        return MetricResult(
            metric_name="voltage_stability",
            metric_type=MetricType.STABILITY,
            value=stability,
            unit="",
            description="电压稳定性，基于变异系数"
        )
    
    def _calc_temperature_stability(self, data: Dict[str, np.ndarray], 
                                   target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算温度稳定性"""
        if 'temperature' not in data:
            return None
        
        temperature = data['temperature']
        
        # 温度稳定性
        temp_range = np.max(temperature) - np.min(temperature)
        avg_temp = np.mean(temperature)
        
        # 稳定性 = 1 - 温度范围/平均温度
        stability = 1.0 - temp_range / (avg_temp + 1e-6)
        stability = max(0.0, stability)
        
        return MetricResult(
            metric_name="temperature_stability",
            metric_type=MetricType.STABILITY,
            value=stability,
            unit="",
            description="温度稳定性"
        )
    
    def _calc_soc_stability(self, data: Dict[str, np.ndarray], 
                           target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算SOC稳定性"""
        if 'soc' not in data:
            return None
        
        soc = data['soc']
        
        # SOC均衡度（多电池系统）
        if len(soc.shape) > 1:  # 多维数组
            soc_std = np.mean([np.std(soc[i, :]) for i in range(soc.shape[0])])
            stability = 1.0 - soc_std / 10.0  # 假设10%为最大可接受标准差
        else:
            # 单个SOC的稳定性
            soc_variation = np.std(soc)
            stability = 1.0 - soc_variation / 20.0  # 20%变化范围
        
        return MetricResult(
            metric_name="soc_stability",
            metric_type=MetricType.STABILITY,
            value=max(0.0, stability),
            unit="",
            description="SOC稳定性/均衡性"
        )
    
    def _calc_system_stability(self, data: Dict[str, np.ndarray], 
                              target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算系统稳定性"""
        stability_metrics = []
        
        # 收集各子系统稳定性
        if 'voltage' in data:
            voltage_stability = self._calc_voltage_stability(data, target_data)
            if voltage_stability:
                stability_metrics.append(voltage_stability.value)
        
        if 'temperature' in data:
            temp_stability = self._calc_temperature_stability(data, target_data)
            if temp_stability:
                stability_metrics.append(temp_stability.value)
        
        if 'soc' in data:
            soc_stability = self._calc_soc_stability(data, target_data)
            if soc_stability:
                stability_metrics.append(soc_stability.value)
        
        if not stability_metrics:
            return None
        
        # 综合稳定性
        overall_stability = np.mean(stability_metrics)
        
        return MetricResult(
            metric_name="system_stability",
            metric_type=MetricType.STABILITY,
            value=overall_stability,
            unit="",
            description="系统整体稳定性"
        )
    
    def _calc_response_time(self, data: Dict[str, np.ndarray], 
                           target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算响应时间"""
        if 'step_response' not in data or 'time' not in data:
            return None
        
        response = data['step_response']
        time_vec = data['time']
        
        # 找到稳态值（最后10%的平均值）
        steady_state = np.mean(response[-len(response)//10:])
        
        # 找到达到90%稳态值的时间
        target_value = 0.9 * steady_state
        
        # 从响应开始寻找
        for i, val in enumerate(response):
            if val >= target_value:
                response_time = time_vec[i] if i < len(time_vec) else time_vec[-1]
                break
        else:
            response_time = time_vec[-1]  # 未达到目标
        
        return MetricResult(
            metric_name="response_time",
            metric_type=MetricType.PERFORMANCE,
            value=response_time,
            unit="s",
            description="90%响应时间"
        )
    
    def _calc_settling_time(self, data: Dict[str, np.ndarray], 
                           target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算建立时间"""
        if 'step_response' not in data or 'time' not in data:
            return None
        
        response = data['step_response']
        time_vec = data['time']
        
        # 稳态值
        steady_state = np.mean(response[-len(response)//10:])
        
        # 找到最后一次超出±2%误差带的时间
        error_band = 0.02 * abs(steady_state)
        
        settling_time = time_vec[-1]  # 默认为最后时刻
        for i in range(len(response)-1, -1, -1):
            if abs(response[i] - steady_state) > error_band:
                settling_time = time_vec[i+1] if i+1 < len(time_vec) else time_vec[-1]
                break
        
        return MetricResult(
            metric_name="settling_time",
            metric_type=MetricType.PERFORMANCE,
            value=settling_time,
            unit="s",
            description="±2%建立时间"
        )
    
    def _calc_overshoot(self, data: Dict[str, np.ndarray], 
                       target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算超调量"""
        if 'step_response' not in data:
            return None
        
        response = data['step_response']
        
        # 稳态值
        steady_state = np.mean(response[-len(response)//10:])
        
        # 最大值
        max_value = np.max(response)
        
        # 超调量
        overshoot = (max_value - steady_state) / abs(steady_state) if steady_state != 0 else 0
        overshoot = max(0.0, overshoot * 100)  # 转换为百分比
        
        return MetricResult(
            metric_name="overshoot",
            metric_type=MetricType.PERFORMANCE,
            value=overshoot,
            unit="%",
            description="超调量百分比"
        )
    
    def _calc_steady_state_error(self, data: Dict[str, np.ndarray], 
                                target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算稳态误差"""
        if target_data is None or 'output' not in data or 'reference' not in target_data:
            return None
        
        output = data['output']
        reference = target_data['reference']
        
        # 稳态值（最后10%）
        steady_output = np.mean(output[-len(output)//10:])
        steady_reference = np.mean(reference[-len(reference)//10:])
        
        # 稳态误差
        sse = abs(steady_reference - steady_output) / abs(steady_reference) if steady_reference != 0 else 0
        
        return MetricResult(
            metric_name="steady_state_error",
            metric_type=MetricType.PERFORMANCE,
            value=sse,
            unit="",
            description="稳态误差"
        )
    
    def _calc_operation_cost(self, data: Dict[str, np.ndarray], 
                            target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算运行成本"""
        if 'power_consumption' not in data or 'electricity_price' not in data:
            return None
        
        power = data['power_consumption']  # kW
        price = data['electricity_price']   # ¥/kWh
        
        min_length = min(len(power), len(price))
        power = power[:min_length]
        price = price[:min_length]
        
        # 运行成本（假设每个时间步为1小时）
        operation_cost = np.sum(power * price)
        
        return MetricResult(
            metric_name="operation_cost",
            metric_type=MetricType.ECONOMIC,
            value=operation_cost,
            unit="¥",
            description="运行成本"
        )
    
    def _calc_maintenance_cost(self, data: Dict[str, np.ndarray], 
                              target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算维护成本"""
        if 'degradation_rate' not in data:
            return None
        
        degradation_rate = data['degradation_rate']
        
        # 简化的维护成本模型
        # 基于降解率估算维护需求
        avg_degradation = np.mean(degradation_rate)
        maintenance_cost = avg_degradation * 1000  # 简化计算
        
        return MetricResult(
            metric_name="maintenance_cost",
            metric_type=MetricType.ECONOMIC,
            value=maintenance_cost,
            unit="¥",
            description="维护成本估算"
        )
    
    def _calc_lifecycle_cost(self, data: Dict[str, np.ndarray], 
                            target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算生命周期成本"""
        lifecycle_cost = 0.0
        
        # 运行成本
        operation_result = self._calc_operation_cost(data, target_data)
        if operation_result:
            lifecycle_cost += operation_result.value
        
        # 维护成本
        maintenance_result = self._calc_maintenance_cost(data, target_data)
        if maintenance_result:
            lifecycle_cost += maintenance_result.value
        
        # 添加初始投资成本（简化）
        if 'initial_cost' in data:
            lifecycle_cost += data['initial_cost'][0]
        
        return MetricResult(
            metric_name="lifecycle_cost",
            metric_type=MetricType.ECONOMIC,
            value=lifecycle_cost,
            unit="¥",
            description="生命周期总成本"
        )
    
    def _calc_revenue_optimization(self, data: Dict[str, np.ndarray], 
                                  target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算收益优化"""
        if 'revenue' not in data or 'max_possible_revenue' not in data:
            return None
        
        actual_revenue = np.sum(data['revenue'])
        max_revenue = np.sum(data['max_possible_revenue'])
        
        optimization_ratio = actual_revenue / (max_revenue + 1e-6) if max_revenue > 0 else 0
        
        return MetricResult(
            metric_name="revenue_optimization",
            metric_type=MetricType.ECONOMIC,
            value=optimization_ratio,
            unit="",
            description="收益优化比例"
        )
    
    def _calc_carbon_footprint(self, data: Dict[str, np.ndarray], 
                              target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算碳足迹"""
        if 'energy_consumption' not in data or 'carbon_intensity' not in data:
            return None
        
        energy = data['energy_consumption']  # kWh
        carbon_intensity = data['carbon_intensity']  # kg CO2/kWh
        
        min_length = min(len(energy), len(carbon_intensity))
        energy = energy[:min_length]
        carbon_intensity = carbon_intensity[:min_length]
        
        carbon_footprint = np.sum(energy * carbon_intensity)
        
        return MetricResult(
            metric_name="carbon_footprint",
            metric_type=MetricType.ENVIRONMENTAL,
            value=carbon_footprint,
            unit="kg CO2",
            description="碳足迹"
        )
    
    def _calc_energy_waste(self, data: Dict[str, np.ndarray], 
                          target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算能量浪费"""
        if 'energy_input' not in data or 'useful_energy' not in data:
            return None
        
        energy_input = data['energy_input']
        useful_energy = data['useful_energy']
        
        total_input = np.sum(energy_input)
        total_useful = np.sum(useful_energy)
        
        energy_waste = total_input - total_useful
        waste_ratio = energy_waste / (total_input + 1e-6) if total_input > 0 else 0
        
        return MetricResult(
            metric_name="energy_waste",
            metric_type=MetricType.ENVIRONMENTAL,
            value=waste_ratio,
            unit="",
            description="能量浪费比例"
        )
    
    def _calc_environmental_impact(self, data: Dict[str, np.ndarray], 
                                  target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算环境影响"""
        impact_score = 0.0
        
        # 碳足迹影响
        carbon_result = self._calc_carbon_footprint(data, target_data)
        if carbon_result:
            impact_score += carbon_result.value / 1000  # 归一化
        
        # 能量浪费影响
        waste_result = self._calc_energy_waste(data, target_data)
        if waste_result:
            impact_score += waste_result.value
        
        # 综合环境影响（越小越好）
        environmental_score = 1.0 / (1.0 + impact_score)
        
        return MetricResult(
            metric_name="environmental_impact",
            metric_type=MetricType.ENVIRONMENTAL,
            value=environmental_score,
            unit="",
            description="环境影响评分（越高越好）"
        )
    
    def _calc_safety_margin(self, data: Dict[str, np.ndarray], 
                           target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算安全裕度"""
        if 'operating_values' not in data or 'safety_limits' not in data:
            return None
        
        operating_values = data['operating_values']
        safety_limits = data['safety_limits']
        
        # 计算到安全限制的最小距离
        distances = np.abs(safety_limits - operating_values)
        min_distance = np.min(distances)
        max_limit = np.max(np.abs(safety_limits))
        
        safety_margin = min_distance / (max_limit + 1e-6) if max_limit > 0 else 0
        
        return MetricResult(
            metric_name="safety_margin",
            metric_type=MetricType.SAFETY,
            value=safety_margin,
            unit="",
            description="安全裕度"
        )
    
    def _calc_fault_tolerance(self, data: Dict[str, np.ndarray], 
                             target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算故障容错性"""
        if 'fault_events' not in data or 'system_performance' not in data:
            return None
        
        fault_events = data['fault_events']
        performance = data['system_performance']
        
        # 故障期间的性能保持率
        fault_mask = fault_events > 0
        if np.sum(fault_mask) > 0:
            fault_performance = np.mean(performance[fault_mask])
            normal_performance = np.mean(performance[~fault_mask])
            
            tolerance = fault_performance / (normal_performance + 1e-6) if normal_performance > 0 else 0
        else:
            tolerance = 1.0  # 没有故障，容错性完美
        
        return MetricResult(
            metric_name="fault_tolerance",
            metric_type=MetricType.SAFETY,
            value=tolerance,
            unit="",
            description="故障容错性"
        )
    
    def _calc_emergency_response(self, data: Dict[str, np.ndarray], 
                                target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算应急响应"""
        if 'emergency_events' not in data or 'response_time' not in data:
            return None
        
        emergency_events = data['emergency_events']
        response_times = data['response_time']
        
        # 平均应急响应时间
        emergency_mask = emergency_events > 0
        if np.sum(emergency_mask) > 0:
            avg_response_time = np.mean(response_times[emergency_mask])
            # 响应性能：越快越好
            response_score = 1.0 / (1.0 + avg_response_time)
        else:
            response_score = 1.0  # 没有应急事件
        
        return MetricResult(
            metric_name="emergency_response",
            metric_type=MetricType.SAFETY,
            value=response_score,
            unit="",
            description="应急响应性能"
        )
    
    def _calc_noise_robustness(self, data: Dict[str, np.ndarray], 
                              target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算噪声鲁棒性"""
        if 'clean_performance' not in data or 'noisy_performance' not in data:
            return None
        
        clean_perf = data['clean_performance']
        noisy_perf = data['noisy_performance']
        
        # 性能保持率
        robustness = np.mean(noisy_perf) / (np.mean(clean_perf) + 1e-6)
        robustness = min(1.0, robustness)
        
        return MetricResult(
            metric_name="noise_robustness",
            metric_type=MetricType.ROBUSTNESS,
            value=robustness,
            unit="",
            description="噪声鲁棒性"
        )
    
    def _calc_parameter_sensitivity(self, data: Dict[str, np.ndarray], 
                                   target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算参数敏感性"""
        if 'parameter_variations' not in data or 'performance_variations' not in data:
            return None
        
        param_var = data['parameter_variations']
        perf_var = data['performance_variations']
        
        # 敏感性 = 性能变化 / 参数变化
        sensitivity = np.std(perf_var) / (np.std(param_var) + 1e-6)
        
        # 鲁棒性 = 1 / (1 + 敏感性)
        robustness = 1.0 / (1.0 + sensitivity)
        
        return MetricResult(
            metric_name="parameter_sensitivity",
            metric_type=MetricType.ROBUSTNESS,
            value=robustness,
            unit="",
            description="参数敏感性（低敏感性=高鲁棒性）"
        )
    
    def _calc_disturbance_rejection(self, data: Dict[str, np.ndarray], 
                                   target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算干扰抑制"""
        if 'disturbance_input' not in data or 'system_output' not in data:
            return None
        
        disturbance = data['disturbance_input']
        output = data['system_output']
        
        # 干扰抑制能力
        disturbance_magnitude = np.std(disturbance)
        output_variation = np.std(output)
        
        rejection = 1.0 - (output_variation / (disturbance_magnitude + 1e-6))
        rejection = max(0.0, rejection)
        
        return MetricResult(
            metric_name="disturbance_rejection",
            metric_type=MetricType.ROBUSTNESS,
            value=rejection,
            unit="",
            description="干扰抑制能力"
        )
    
    def _calc_convergence_rate(self, data: Dict[str, np.ndarray], 
                              target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算收敛率"""
        if 'training_rewards' not in data:
            return None
        
        rewards = data['training_rewards']
        
        # 找到收敛点（性能不再显著提升的点）
        window_size = min(50, len(rewards) // 4)
        if window_size < 10:
            return None
        
        for i in range(window_size, len(rewards)):
            recent_mean = np.mean(rewards[i-window_size:i])
            if i >= len(rewards) - window_size:
                break
            next_mean = np.mean(rewards[i:i+window_size])
            
            if abs(next_mean - recent_mean) / (recent_mean + 1e-6) < 0.01:  # 1%变化
                convergence_episode = i
                break
        else:
            convergence_episode = len(rewards)
        
        # 收敛率：越早收敛越好
        convergence_rate = 1.0 - (convergence_episode / len(rewards))
        
        return MetricResult(
            metric_name="convergence_rate",
            metric_type=MetricType.CONVERGENCE,
            value=convergence_rate,
            unit="",
            description="收敛率"
        )
    
    def _calc_convergence_stability(self, data: Dict[str, np.ndarray], 
                                   target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算收敛稳定性"""
        if 'training_rewards' not in data:
            return None
        
        rewards = data['training_rewards']
        
        # 最后25%的性能稳定性
        final_quarter = rewards[-len(rewards)//4:]
        stability = 1.0 - (np.std(final_quarter) / (np.mean(final_quarter) + 1e-6))
        stability = max(0.0, stability)
        
        return MetricResult(
            metric_name="convergence_stability",
            metric_type=MetricType.CONVERGENCE,
            value=stability,
            unit="",
            description="收敛稳定性"
        )
    
    def _calc_learning_efficiency(self, data: Dict[str, np.ndarray], 
                                 target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算学习效率"""
        if 'training_rewards' not in data:
            return None
        
        rewards = data['training_rewards']
        
        # 学习效率：最终性能 / 训练时间
        final_performance = np.mean(rewards[-10:])  # 最后10个episode的平均
        training_episodes = len(rewards)
        
        efficiency = final_performance / training_episodes
        
        return MetricResult(
            metric_name="learning_efficiency",
            metric_type=MetricType.CONVERGENCE,
            value=efficiency,
            unit="reward/episode",
            description="学习效率"
        )
    
    def _calc_pareto_efficiency(self, data: Dict[str, np.ndarray], 
                               target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算帕累托效率"""
        if 'objective_values' not in data:
            return None
        
        objectives = data['objective_values']  # shape: (n_solutions, n_objectives)
        
        if len(objectives.shape) != 2:
            return None
        
        # 计算帕累托前沿
        pareto_front = self._find_pareto_front(objectives)
        pareto_efficiency = len(pareto_front) / len(objectives)
        
        return MetricResult(
            metric_name="pareto_efficiency",
            metric_type=MetricType.MULTI_OBJECTIVE,
            value=pareto_efficiency,
            unit="",
            description="帕累托效率：前沿解的比例"
        )
    
    def _calc_hypervolume(self, data: Dict[str, np.ndarray], 
                         target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算超体积"""
        if 'objective_values' not in data or 'reference_point' not in data:
            return None
        
        objectives = data['objective_values']
        reference_point = data['reference_point']
        
        # 简化的超体积计算
        pareto_front = self._find_pareto_front(objectives)
        
        if len(pareto_front) == 0:
            hypervolume = 0.0
        else:
            # 计算每个解到参考点的体积贡献
            volumes = []
            for solution in pareto_front:
                volume = np.prod(np.maximum(0, solution - reference_point))
                volumes.append(volume)
            
            hypervolume = np.sum(volumes)
        
        return MetricResult(
            metric_name="hypervolume",
            metric_type=MetricType.MULTI_OBJECTIVE,
            value=hypervolume,
            unit="",
            description="超体积指标"
        )
    
    def _calc_objective_balance(self, data: Dict[str, np.ndarray], 
                               target_data: Optional[Dict[str, np.ndarray]] = None) -> Optional[MetricResult]:
        """计算目标平衡性"""
        if 'objective_values' not in data:
            return None
        
        objectives = data['objective_values']
        
        # 计算目标间的平衡性
        if len(objectives.shape) != 2 or objectives.shape[1] < 2:
            return None
        
        # 标准化目标值
        normalized_objectives = (objectives - np.min(objectives, axis=0)) / (
            np.max(objectives, axis=0) - np.min(objectives, axis=0) + 1e-6
        )
        
        # 计算每个解的目标平衡性（标准差越小越平衡）
        balance_scores = []
        for solution in normalized_objectives:
            balance = 1.0 - np.std(solution)
            balance_scores.append(max(0.0, balance))
        
        avg_balance = np.mean(balance_scores)
        
        return MetricResult(
            metric_name="objective_balance",
            metric_type=MetricType.MULTI_OBJECTIVE,
            value=avg_balance,
            unit="",
            description="目标平衡性"
        )
    
    def _find_pareto_front(self, objectives: np.ndarray) -> np.ndarray:
        """寻找帕累托前沿"""
        n_solutions = objectives.shape[0]
        pareto_front = []
        
        for i in range(n_solutions):
            is_dominated = False
            for j in range(n_solutions):
                if i != j:
                    # 检查是否被支配（假设目标是最大化）
                    if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(objectives[i])
        
        return np.array(pareto_front)
    
    def _assess_data_quality(self, data: Dict[str, np.ndarray]) -> float:
        """评估数据质量"""
        quality_scores = []
        
        for signal_name, signal_data in data.items():
            # 完整性
            completeness = 1.0 - np.sum(np.isnan(signal_data)) / len(signal_data)
            
            # 一致性
            if len(signal_data) > 1:
                diff_data = np.diff(signal_data)
                consistency = 1.0 - np.sum(np.abs(diff_data) > 3 * np.std(diff_data)) / len(diff_data)
            else:
                consistency = 1.0
            
            # 合理性
            finite_ratio = np.sum(np.isfinite(signal_data)) / len(signal_data)
            
            signal_quality = (completeness + consistency + finite_ratio) / 3
            quality_scores.append(signal_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """计算置信区间"""
        try:
            mean = np.mean(data)
            sem = stats.sem(data)  # 标准误差
            h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
            return (mean - h, mean + h)
        except:
            return (0.0, 0.0)
    
    def _compare_with_benchmark(self, value: float, metric_name: str) -> Dict[str, float]:
        """与基准值比较"""
        if metric_name not in self.benchmarks:
            return {}
        
        benchmark = self.benchmarks[metric_name]
        
        comparison = {}
        for level, threshold in benchmark.items():
            if metric_name in ['response_time']:  # 越小越好的指标
                comparison[f'vs_{level}'] = (threshold - value) / threshold if threshold > 0 else 0
            else:  # 越大越好的指标
                comparison[f'vs_{level}'] = (value - threshold) / threshold if threshold > 0 else 0
        
        return comparison
    
    def _calculate_summary_statistics(self, suite: MetricSuite) -> Dict[str, float]:
        """计算汇总统计"""
        if not suite.metrics:
            return {}
        
        # 按类型分组
        metrics_by_type = {}
        for metric in suite.metrics.values():
            if metric.metric_type not in metrics_by_type:
                metrics_by_type[metric.metric_type] = []
            metrics_by_type[metric.metric_type].append(metric.value)
        
        # 计算各类型平均值
        type_averages = {}
        for metric_type, values in metrics_by_type.items():
            type_averages[f'{metric_type.value}_avg'] = np.mean(values)
        
        # 加权综合得分
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_type, avg_value in type_averages.items():
            metric_type_enum = MetricType(metric_type.replace('_avg', ''))
            if metric_type_enum in self.metric_weights:
                weight = self.metric_weights[metric_type_enum]
                weighted_score += avg_value * weight
                total_weight += weight
        
        if total_weight > 0:
            type_averages['weighted_overall_score'] = weighted_score / total_weight
        
        # 添加其他统计信息
        all_values = [metric.value for metric in suite.metrics.values()]
        type_averages.update({
            'overall_mean': np.mean(all_values),
            'overall_std': np.std(all_values),
            'overall_min': np.min(all_values),
            'overall_max': np.max(all_values),
            'metrics_count': len(all_values)
        })
        
        return type_averages
    
    def create_metric_dashboard(self, suite: MetricSuite) -> Dict[str, Any]:
        """创建指标仪表板"""
        dashboard = {
            'suite_info': {
                'id': suite.suite_id,
                'timestamp': suite.calculation_timestamp,
                'data_quality': suite.data_quality_score,
                'metrics_count': len(suite.metrics)
            },
            'key_metrics': {},
            'performance_summary': {},
            'alerts': [],
            'recommendations': []
        }
        
        # 关键指标
        key_metric_names = ['tracking_accuracy', 'energy_efficiency', 'system_stability', 'safety_margin']
        for metric_name in key_metric_names:
            if metric_name in suite.metrics:
                dashboard['key_metrics'][metric_name] = {
                    'value': suite.metrics[metric_name].value,
                    'unit': suite.metrics[metric_name].unit,
                    'benchmark': suite.metrics[metric_name].benchmark_comparison
                }
        
        # 性能汇总
        dashboard['performance_summary'] = suite.summary_statistics
        
        # 生成警告
        for metric_name, metric in suite.metrics.items():
            if metric_name in self.benchmarks:
                benchmark = self.benchmarks[metric_name]
                if metric.value < benchmark.get('poor', 0):
                    dashboard['alerts'].append(f"{metric_name} 性能低于预期: {metric.value:.3f}")
        
        # 生成建议
        if suite.data_quality_score < 0.8:
            dashboard['recommendations'].append("数据质量较低，建议检查数据采集过程")
        
        if 'weighted_overall_score' in suite.summary_statistics:
            if suite.summary_statistics['weighted_overall_score'] < 0.7:
                dashboard['recommendations'].append("综合性能有待提升，建议重点关注低分指标")
        
        return dashboard
    
    def export_metrics(self, suite: MetricSuite, file_path: str, format: str = 'json'):
        """导出指标结果"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == 'json':
                export_data = {
                    'suite_id': suite.suite_id,
                    'calculation_timestamp': suite.calculation_timestamp,
                    'data_quality_score': suite.data_quality_score,
                    'summary_statistics': suite.summary_statistics,
                    'metrics': {}
                }
                
                for name, metric in suite.metrics.items():
                    export_data['metrics'][name] = {
                        'value': metric.value,
                        'unit': metric.unit,
                        'description': metric.description,
                        'type': metric.metric_type.value,
                        'confidence_interval': metric.confidence_interval,
                        'benchmark_comparison': metric.benchmark_comparison,
                        'calculation_time': metric.calculation_time
                    }
                
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                # 创建指标DataFrame
                metrics_data = []
                for name, metric in suite.metrics.items():
                    metrics_data.append({
                        'metric_name': name,
                        'metric_type': metric.metric_type.value,
                        'value': metric.value,
                        'unit': metric.unit,
                        'description': metric.description
                    })
                
                import pandas as pd
                df = pd.DataFrame(metrics_data)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            print(f"✅ 指标结果已导出: {file_path}")
            
        except Exception as e:
            print(f"❌ 指标导出失败: {str(e)}")
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """获取计算统计信息"""
        stats = self.calculation_stats.copy()
        
        if stats['total_calculations'] > 0:
            stats['avg_calculation_time'] = stats['calculation_time'] / stats['total_calculations']
            stats['success_rate'] = (stats['total_calculations'] - stats['failed_calculations']) / stats['total_calculations']
        else:
            stats['avg_calculation_time'] = 0
            stats['success_rate'] = 0
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"MetricsCalculator({self.calculator_id}): "
                f"计算次数={self.calculation_stats['total_calculations']}, "
                f"成功率={self.calculation_stats['total_calculations'] - self.calculation_stats['failed_calculations']}/{self.calculation_stats['total_calculations']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"MetricsCalculator(calculator_id='{self.calculator_id}', "
                f"metrics={len(self.metric_calculators)}, "
                f"total_calculations={self.calculation_stats['total_calculations']})")
