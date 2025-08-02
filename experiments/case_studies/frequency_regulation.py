import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from ..basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType
from utils.logger import Logger
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer, PlotConfig, PlotType

class FrequencyRegulationService(Enum):
    """频率调节服务类型"""
    PRIMARY_RESERVE = "primary_reserve"      # 一次调频
    SECONDARY_RESERVE = "secondary_reserve"  # 二次调频
    TERTIARY_RESERVE = "tertiary_reserve"   # 三次调频
    FAST_FREQUENCY_RESPONSE = "fast_frequency_response"  # 快速频率响应

@dataclass
class FrequencyRegulationConfig:
    """频率调节配置"""
    service_type: FrequencyRegulationService
    
    # 储能系统参数
    battery_capacity_kwh: float = 1000.0    # 电池容量
    max_power_kw: float = 500.0             # 最大功率
    response_time_ms: float = 100.0         # 响应时间
    ramp_rate_kw_per_s: float = 1000.0      # 爬坡率
    
    # 频率调节参数
    nominal_frequency: float = 50.0         # 标称频率 (Hz)
    deadband: float = 0.02                  # 死区 (Hz)
    droop_coefficient: float = 0.05         # 下垂系数
    regulation_capacity_mw: float = 1.0     # 调节容量
    
    # 服务参数
    service_duration_hours: int = 24        # 服务持续时间
    capacity_price: float = 50.0            # 容量电价 (元/MW/h)
    performance_price: float = 200.0        # 性能电价 (元/MWh)
    
    # 运行约束
    min_soc: float = 0.2                    # 最小SOC
    max_soc: float = 0.8                    # 最大SOC
    target_soc: float = 0.5                 # 目标SOC
    soc_recovery_rate: float = 0.1          # SOC恢复率

@dataclass
class FrequencyRegulationResults:
    """频率调节结果"""
    experiment_id: str
    config: FrequencyRegulationConfig
    
    # 频率调节性能
    avg_frequency_deviation: float = 0.0    # 平均频率偏差
    max_frequency_deviation: float = 0.0    # 最大频率偏差
    frequency_response_time: float = 0.0    # 频率响应时间
    regulation_accuracy: float = 0.0        # 调节精度
    
    # 服务质量
    availability_ratio: float = 0.0         # 可用率
    regulation_mileage: float = 0.0          # 调节里程
    performance_score: float = 0.0          # 性能评分
    
    # 经济收益
    capacity_revenue: float = 0.0           # 容量收益
    performance_revenue: float = 0.0        # 性能收益
    total_revenue: float = 0.0              # 总收益
    operation_cost: float = 0.0             # 运行成本
    net_profit: float = 0.0                 # 净收益
    
    # 系统状态
    avg_soc: float = 0.0                    # 平均SOC
    soc_deviation: float = 0.0              # SOC偏差
    cycle_count: int = 0                    # 循环次数
    energy_throughput_mwh: float = 0.0      # 能量吞吐量
    
    # 时间序列数据
    frequency_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    regulation_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    battery_power: np.ndarray = field(default_factory=lambda: np.array([]))
    battery_soc: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))

class FrequencyRegulationExperiment:
    """
    频率调节案例研究
    评估储能系统在频率调节服务中的技术和经济性能
    """
    
    def __init__(self, config: FrequencyRegulationConfig, experiment_id: Optional[str] = None):
        """
        初始化频率调节实验
        
        Args:
            config: 频率调节配置
            experiment_id: 实验ID
        """
        self.config = config
        self.experiment_id = experiment_id or f"freq_reg_{int(time.time()*1000)}"
        
        # 初始化组件
        self.logger = Logger(f"FrequencyRegulation_{self.experiment_id}")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # 生成频率信号
        self._generate_frequency_signal()
        
        # 创建实验目录
        self.experiment_dir = f"experiments/case_studies/frequency_regulation/{self.experiment_id}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"✅ 频率调节实验初始化完成: {config.service_type.value}")
        print(f"   实验ID: {self.experiment_id}")
        print(f"   调节容量: {config.regulation_capacity_mw:.1f} MW")
    
    def run_case_study(self) -> FrequencyRegulationResults:
        """
        运行频率调节案例研究
        
        Returns:
            频率调节结果
        """
        study_start_time = time.time()
        
        self.logger.info(f"🚀 开始频率调节案例研究: {self.config.service_type.value}")
        
        try:
            # 阶段1: 频率信号分析
            self.logger.info("📊 阶段1: 频率信号分析")
            frequency_analysis = self._analyze_frequency_signal()
            
            # 阶段2: 调节策略训练
            self.logger.info("🎯 阶段2: 调节策略训练")
            regulation_strategy = self._train_regulation_strategy()
            
            # 阶段3: 频率调节仿真
            self.logger.info("⚡ 阶段3: 频率调节仿真")
            simulation_results = self._simulate_frequency_regulation(regulation_strategy)
            
            # 阶段4: 性能评估
            self.logger.info("📈 阶段4: 性能评估")
            performance_metrics = self._evaluate_regulation_performance(simulation_results)
            
            # 阶段5: 经济性分析
            self.logger.info("💰 阶段5: 经济性分析")
            economic_analysis = self._analyze_regulation_economics(simulation_results, performance_metrics)
            
            # 阶段6: 结果整合
            self.logger.info("📊 阶段6: 结果整合")
            final_results = self._integrate_regulation_results(
                frequency_analysis, simulation_results, 
                performance_metrics, economic_analysis
            )
            
            # 生成报告
            self._generate_regulation_report(final_results)
            
            study_time = time.time() - study_start_time
            self.logger.info(f"✅ 频率调节案例研究完成，用时: {study_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 频率调节案例研究失败: {str(e)}")
            raise
    
    def _generate_frequency_signal(self):
        """生成频率信号"""
        # 生成时间序列（1秒分辨率）
        duration_seconds = self.config.service_duration_hours * 3600
        self.timestamps = np.arange(0, duration_seconds, 1)
        
        # 基础频率信号
        base_frequency = self.config.nominal_frequency
        
        # 添加不同频率成分的扰动
        frequency_deviation = np.zeros(len(self.timestamps))
        
        # 长期漂移（小时级）
        long_term = 0.05 * np.sin(2 * np.pi * self.timestamps / 3600)
        
        # 中期波动（分钟级）
        medium_term = 0.02 * np.sin(2 * np.pi * self.timestamps / 300) * np.random.uniform(0.5, 1.5, len(self.timestamps))
        
        # 短期扰动（秒级）
        short_term = 0.01 * np.random.normal(0, 1, len(self.timestamps))
        
        # 特殊事件（如大机组脱网）
        num_events = np.random.poisson(5)  # 平均5个事件
        for _ in range(num_events):
            event_time = np.random.randint(0, len(self.timestamps))
            event_duration = np.random.randint(30, 300)  # 30秒到5分钟
            event_magnitude = np.random.uniform(-0.2, 0.2)
            
            end_time = min(event_time + event_duration, len(self.timestamps))
            # 指数衰减的频率事件
            decay = np.exp(-np.arange(end_time - event_time) / 60)
            frequency_deviation[event_time:end_time] += event_magnitude * decay
        
        # 组合所有成分
        self.frequency_signal = base_frequency + long_term + medium_term + short_term + frequency_deviation
        
        # 确保频率在合理范围内
        self.frequency_signal = np.clip(self.frequency_signal, 49.5, 50.5)
        
        self.logger.info(f"生成频率信号: {len(self.frequency_signal)} 个数据点")
    
    def _analyze_frequency_signal(self) -> Dict[str, Any]:
        """分析频率信号"""
        frequency_deviation = self.frequency_signal - self.config.nominal_frequency
        
        analysis = {
            'mean_frequency': np.mean(self.frequency_signal),
            'frequency_std': np.std(self.frequency_signal),
            'max_positive_deviation': np.max(frequency_deviation),
            'max_negative_deviation': np.min(frequency_deviation),
            'rms_deviation': np.sqrt(np.mean(frequency_deviation**2)),
            'frequency_events': self._detect_frequency_events(),
            'regulation_demand': self._calculate_regulation_demand()
        }
        
        self.logger.info(f"频率信号分析完成 - RMS偏差: {analysis['rms_deviation']:.4f} Hz")
        
        return analysis
    
    def _detect_frequency_events(self) -> List[Dict[str, Any]]:
        """检测频率事件"""
        frequency_deviation = self.frequency_signal - self.config.nominal_frequency
        threshold = 0.1  # 0.1 Hz阈值
        
        events = []
        in_event = False
        event_start = 0
        
        for i, deviation in enumerate(frequency_deviation):
            if abs(deviation) > threshold and not in_event:
                # 事件开始
                in_event = True
                event_start = i
            elif abs(deviation) <= threshold and in_event:
                # 事件结束
                in_event = False
                event_duration = i - event_start
                event_magnitude = np.max(np.abs(frequency_deviation[event_start:i]))
                
                events.append({
                    'start_time': event_start,
                    'duration': event_duration,
                    'magnitude': event_magnitude,
                    'type': 'positive' if np.mean(frequency_deviation[event_start:i]) > 0 else 'negative'
                })
        
        return events
    
    def _calculate_regulation_demand(self) -> np.ndarray:
        """计算调节需求"""
        frequency_deviation = self.frequency_signal - self.config.nominal_frequency
        
        # 应用死区
        regulation_demand = np.where(
            np.abs(frequency_deviation) > self.config.deadband,
            frequency_deviation,
            0
        )
        
        # 应用下垂特性
        regulation_demand = -regulation_demand / self.config.droop_coefficient
        
        # 限制在调节容量范围内
        max_regulation = self.config.regulation_capacity_mw * 1000  # 转换为kW
        regulation_demand = np.clip(regulation_demand, -max_regulation, max_regulation)
        
        return regulation_demand
    
    def _train_regulation_strategy(self) -> Dict[str, Any]:
        """训练调节策略"""
        # 创建DRL训练配置（简化）
        experiment_settings = ExperimentSettings(
            experiment_name=f"frequency_regulation_training_{self.config.service_type.value}",
            experiment_type=ExperimentType.HIERARCHICAL,
            description="频率调节控制策略训练",
            total_episodes=300,
            evaluation_frequency=50,
            save_frequency=100,
            use_pretraining=True,
            enable_hierarchical=True,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        # 运行训练（简化版本）
        training_experiment = BasicExperiment(
            settings=experiment_settings,
            experiment_id=f"{self.experiment_id}_training"
        )
        
        training_results = training_experiment.run_experiment()
        
        regulation_strategy = {
            'type': 'drl_frequency_regulation',
            'model_path': training_results.best_checkpoint_path,
            'performance': training_results.best_performance,
            'training_time': training_results.training_time
        }
        
        self.logger.info("调节策略训练完成")
        
        return regulation_strategy
    
    def _simulate_frequency_regulation(self, regulation_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """仿真频率调节过程"""
        num_points = len(self.frequency_signal)
        regulation_demand = self._calculate_regulation_demand()
        
        # 初始化系统状态
        battery_soc = np.zeros(num_points)
        battery_power = np.zeros(num_points)
        regulation_signal = np.zeros(num_points)
        
        # 初始SOC
        current_soc = self.config.target_soc
        
        # 仿真控制过程
        for i in range(num_points):
            # 当前调节需求
            power_demand = regulation_demand[i]  # kW
            
            # SOC管理：如果SOC偏离目标太多，需要进行恢复
            soc_error = current_soc - self.config.target_soc
            soc_recovery_power = 0
            
            if abs(soc_error) > 0.1:  # SOC偏差超过10%
                # SOC恢复功率
                soc_recovery_power = -soc_error * self.config.soc_recovery_rate * self.config.battery_capacity_kwh
                soc_recovery_power = np.clip(soc_recovery_power, -self.config.max_power_kw/4, self.config.max_power_kw/4)
            
            # 总功率需求
            total_power_demand = power_demand + soc_recovery_power
            
            # 功率限制
            max_charge_power = min(
                self.config.max_power_kw,
                (self.config.max_soc - current_soc) * self.config.battery_capacity_kwh * 3600  # 1秒内的最大充电量
            )
            
            max_discharge_power = min(
                self.config.max_power_kw,
                (current_soc - self.config.min_soc) * self.config.battery_capacity_kwh * 3600  # 1秒内的最大放电量
            )
            
            # 应用功率限制
            actual_power = np.clip(total_power_demand, -max_discharge_power, max_charge_power)
            
            # 爬坡率限制
            if i > 0:
                max_power_change = self.config.ramp_rate_kw_per_s  # 1秒内的最大功率变化
                power_change = actual_power - battery_power[i-1]
                if abs(power_change) > max_power_change:
                    actual_power = battery_power[i-1] + np.sign(power_change) * max_power_change
            
            # 记录功率
            battery_power[i] = actual_power
            regulation_signal[i] = min(abs(actual_power), abs(power_demand)) * np.sign(power_demand)
            
            # 更新SOC
            energy_change = actual_power / 3600  # kWh (1秒 = 1/3600小时)
            
            if actual_power > 0:  # 充电
                energy_change *= 0.95  # 充电效率
            else:  # 放电
                energy_change /= 0.95  # 放电效率
            
            current_soc += energy_change / self.config.battery_capacity_kwh
            current_soc = np.clip(current_soc, self.config.min_soc, self.config.max_soc)
            battery_soc[i] = current_soc
        
        simulation_results = {
            'frequency_signal': self.frequency_signal,
            'regulation_demand': regulation_demand,
            'regulation_signal': regulation_signal,
            'battery_power': battery_power,
            'battery_soc': battery_soc,
            'timestamps': self.timestamps
        }
        
        self.logger.info("频率调节仿真完成")
        
        return simulation_results
    
    def _evaluate_regulation_performance(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估调节性能"""
        frequency_signal = simulation_results['frequency_signal']
        regulation_demand = simulation_results['regulation_demand']
        regulation_signal = simulation_results['regulation_signal']
        battery_soc = simulation_results['battery_soc']
        
        # 频率偏差分析
        frequency_deviation = frequency_signal - self.config.nominal_frequency
        avg_frequency_deviation = np.mean(np.abs(frequency_deviation))
        max_frequency_deviation = np.max(np.abs(frequency_deviation))
        
        # 调节精度
        regulation_error = regulation_demand - regulation_signal
        regulation_accuracy = 1.0 - np.mean(np.abs(regulation_error)) / (np.mean(np.abs(regulation_demand)) + 1e-6)
        
        # 响应时间（简化计算）
        response_time = self.config.response_time_ms / 1000  # 转换为秒
        
        # 可用率
        available_time = np.sum(
            (battery_soc > self.config.min_soc + 0.05) & 
            (battery_soc < self.config.max_soc - 0.05)
        )
        availability_ratio = available_time / len(battery_soc)
        
        # 调节里程
        regulation_mileage = np.sum(np.abs(np.diff(regulation_signal))) / 1000  # MW
        
        # 性能评分
        performance_score = min(1.0, regulation_accuracy * availability_ratio * 1.2)
        
        # SOC统计
        avg_soc = np.mean(battery_soc)
        soc_deviation = np.std(battery_soc)
        
        # 循环计数
        soc_changes = np.abs(np.diff(battery_soc))
        cycle_count = int(np.sum(soc_changes) / 2)  # 简化计算
        
        # 能量吞吐量
        energy_throughput = np.sum(np.abs(simulation_results['battery_power'])) / 1000 / 3600  # MWh
        
        performance_metrics = {
            'avg_frequency_deviation': avg_frequency_deviation,
            'max_frequency_deviation': max_frequency_deviation,
            'frequency_response_time': response_time,
            'regulation_accuracy': regulation_accuracy,
            'availability_ratio': availability_ratio,
            'regulation_mileage': regulation_mileage,
            'performance_score': performance_score,
            'avg_soc': avg_soc,
            'soc_deviation': soc_deviation,
            'cycle_count': cycle_count,
            'energy_throughput_mwh': energy_throughput
        }
        
        self.logger.info(f"性能评估完成 - 调节精度: {regulation_accuracy:.1%}")
        
        return performance_metrics
    
    def _analyze_regulation_economics(self, simulation_results: Dict[str, Any], 
                                    performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析调节经济性"""
        # 容量收益
        capacity_revenue = (self.config.regulation_capacity_mw * 
                          self.config.capacity_price * 
                          self.config.service_duration_hours *
                          performance_metrics['availability_ratio'])
        
        # 性能收益
        performance_revenue = (performance_metrics['regulation_mileage'] * 
                             self.config.performance_price * 
                             performance_metrics['performance_score'])
        
        # 总收益
        total_revenue = capacity_revenue + performance_revenue
        
        # 运行成本
        # 电池损耗成本
        cycle_cost_per_mwh = 50  # 元/MWh
        degradation_cost = performance_metrics['energy_throughput_mwh'] * cycle_cost_per_mwh
        
        # 维护成本
        maintenance_cost = self.config.max_power_kw * 0.1  # 简化估算：0.1元/kW/天
        
        # 其他运行成本
        operation_cost = degradation_cost + maintenance_cost
        
        # 净收益
        net_profit = total_revenue - operation_cost
        
        economic_analysis = {
            'capacity_revenue': capacity_revenue,
            'performance_revenue': performance_revenue,
            'total_revenue': total_revenue,
            'degradation_cost': degradation_cost,
            'maintenance_cost': maintenance_cost,
            'operation_cost': operation_cost,
            'net_profit': net_profit,
            'profit_margin': net_profit / total_revenue if total_revenue > 0 else 0,
            'revenue_per_mw': total_revenue / self.config.regulation_capacity_mw
        }
        
        self.logger.info(f"经济性分析完成 - 净收益: {net_profit:.0f} 元")
        
        return economic_analysis
    
    def _integrate_regulation_results(self, frequency_analysis: Dict[str, Any],
                                    simulation_results: Dict[str, Any],
                                    performance_metrics: Dict[str, Any],
                                    economic_analysis: Dict[str, Any]) -> FrequencyRegulationResults:
        """整合调节结果"""
        results = FrequencyRegulationResults(
            experiment_id=self.experiment_id,
            config=self.config
        )
        
        # 频率调节性能
        results.avg_frequency_deviation = performance_metrics['avg_frequency_deviation']
        results.max_frequency_deviation = performance_metrics['max_frequency_deviation']
        results.frequency_response_time = performance_metrics['frequency_response_time']
        results.regulation_accuracy = performance_metrics['regulation_accuracy']
        
        # 服务质量
        results.availability_ratio = performance_metrics['availability_ratio']
        results.regulation_mileage = performance_metrics['regulation_mileage']
        results.performance_score = performance_metrics['performance_score']
        
        # 经济收益
        results.capacity_revenue = economic_analysis['capacity_revenue']
        results.performance_revenue = economic_analysis['performance_revenue']
        results.total_revenue = economic_analysis['total_revenue']
        results.operation_cost = economic_analysis['operation_cost']
        results.net_profit = economic_analysis['net_profit']
        
        # 系统状态
        results.avg_soc = performance_metrics['avg_soc']
        results.soc_deviation = performance_metrics['soc_deviation']
        results.cycle_count = performance_metrics['cycle_count']
        results.energy_throughput_mwh = performance_metrics['energy_throughput_mwh']
        
        # 时间序列数据
        results.frequency_signal = simulation_results['frequency_signal']
        results.regulation_signal = simulation_results['regulation_signal']
        results.battery_power = simulation_results['battery_power']
        results.battery_soc = simulation_results['battery_soc']
        results.timestamps = simulation_results['timestamps']
        
        return results
    
    def _generate_regulation_report(self, results: FrequencyRegulationResults):
        """生成调节报告"""
        report = {
            'case_study_info': {
                'experiment_id': results.experiment_id,
                'service_type': results.config.service_type.value,
                'service_duration_hours': results.config.service_duration_hours,
                'regulation_capacity_mw': results.config.regulation_capacity_mw,
                'battery_capacity_kwh': results.config.battery_capacity_kwh
            },
            'frequency_regulation_performance': {
                'avg_frequency_deviation_hz': results.avg_frequency_deviation,
                'max_frequency_deviation_hz': results.max_frequency_deviation,
                'frequency_response_time_ms': results.frequency_response_time * 1000,
                'regulation_accuracy_percent': results.regulation_accuracy * 100,
                'availability_ratio_percent': results.availability_ratio * 100,
                'regulation_mileage_mw': results.regulation_mileage,
                'performance_score': results.performance_score
            },
            'economic_performance': {
                'revenue': {
                    'capacity_revenue': results.capacity_revenue,
                    'performance_revenue': results.performance_revenue,
                    'total_revenue': results.total_revenue
                },
                'costs': {
                    'operation_cost': results.operation_cost,
                    'net_profit': results.net_profit
                },
                'profitability': {
                    'profit_margin_percent': (results.net_profit / results.total_revenue * 100) if results.total_revenue > 0 else 0,
                    'revenue_per_mw_per_hour': results.total_revenue / (results.config.regulation_capacity_mw * results.config.service_duration_hours)
                }
            },
            'system_performance': {
                'avg_soc_percent': results.avg_soc * 100,
                'soc_deviation_percent': results.soc_deviation * 100,
                'cycle_count': results.cycle_count,
                'energy_throughput_mwh': results.energy_throughput_mwh,
                'utilization_rate': results.energy_throughput_mwh / (results.config.battery_capacity_kwh / 1000 * 2)  # 往返为2倍容量
            },
            'key_findings': [],
            'recommendations': []
        }
        
        # 关键发现
        if results.regulation_accuracy > 0.9:
            report['key_findings'].append(f"优秀的调节精度：{results.regulation_accuracy:.1%}")
        
        if results.availability_ratio > 0.95:
            report['key_findings'].append(f"高可用率：{results.availability_ratio:.1%}")
        
        if results.net_profit > 0:
            report['key_findings'].append(f"实现盈利：净收益 {results.net_profit:.0f} 元")
        
        # 建议
        if results.regulation_accuracy < 0.8:
            report['recommendations'].append("建议优化控制算法以提高调节精度")
        
        if results.availability_ratio < 0.9:
            report['recommendations'].append("建议优化SOC管理策略以提高可用率")
        
        if results.soc_deviation > 0.15:
            report['recommendations'].append("建议加强SOC平衡控制以减少SOC波动")
        
        # 保存报告
        report_path = os.path.join(self.experiment_dir, "frequency_regulation_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可视化
        self._create_regulation_visualizations(results)
        
        self.logger.info(f"频率调节报告已保存: {report_path}")
        
        return report
    
    def _create_regulation_visualizations(self, results: FrequencyRegulationResults):
        """创建频率调节可视化"""
        # 选择前1小时数据进行展示
        show_duration = 3600  # 1小时
        show_points = min(show_duration, len(results.timestamps))
        
        # 1. 频率和调节信号图
        freq_config = PlotConfig(
            plot_type=PlotType.LINE,
            title="频率信号和调节响应",
            x_label="时间 (秒)",
            y_label="频率 (Hz) / 调节功率 (kW)",
            width=1200,
            height=600,
            save_path=os.path.join(self.experiment_dir, "frequency_regulation.png")
        )
        
        freq_data = {
            'time': results.timestamps[:show_points],
            'frequency': results.frequency_signal[:show_points],
            'regulation_power': results.battery_power[:show_points] / 10  # 缩放以便显示
        }
        
        self.visualizer.create_plot(freq_data, freq_config)
        
        # 2. SOC变化图
        soc_config = PlotConfig(
            plot_type=PlotType.LINE,
            title="电池SOC变化",
            x_label="时间 (秒)",
            y_label="SOC (%)",
            width=1200,
            height=400,
            save_path=os.path.join(self.experiment_dir, "soc_variation.png")
        )
        
        soc_data = {
            'time': results.timestamps[:show_points],
            'soc': results.battery_soc[:show_points] * 100
        }
        
        self.visualizer.create_plot(soc_data, soc_config)
        
        # 3. 收益结构图
        revenue_config = PlotConfig(
            plot_type=PlotType.BAR,
            title="收益结构分析",
            x_label="收益类型",
            y_label="金额 (元)",
            width=800,
            height=600,
            save_path=os.path.join(self.experiment_dir, "revenue_structure.png")
        )
        
        revenue_data = {
            'capacity_revenue': results.capacity_revenue,
            'performance_revenue': results.performance_revenue,
            'operation_cost': -results.operation_cost,  # 负值表示成本
            'net_profit': results.net_profit
        }
        
        self.visualizer.create_plot(revenue_data, revenue_config)
        
        self.logger.info("频率调节可视化图表生成完成")
