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

from ..basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType, ExperimentResults
from data_processing.scenario_generator import ScenarioGenerator, ScenarioType
from data_processing.load_profile_generator import LoadProfileGenerator, LoadPattern
from utils.logger import Logger
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer, PlotConfig, PlotType

class PeakShavingScenario(Enum):
    """削峰填谷场景类型"""
    COMMERCIAL_BUILDING = "commercial_building"    # 商业建筑
    INDUSTRIAL_FACILITY = "industrial_facility"    # 工业设施
    RESIDENTIAL_COMPLEX = "residential_complex"    # 住宅小区
    MIXED_USE = "mixed_use"                        # 混合用途
    DATA_CENTER = "data_center"                    # 数据中心
    HOSPITAL = "hospital"                          # 医院

@dataclass
class PeakShavingConfig:
    """削峰填谷配置"""
    scenario_type: PeakShavingScenario
    
    # 负荷特性
    base_load_kw: float = 500.0          # 基础负荷 (kW)
    peak_load_kw: float = 1000.0         # 峰值负荷 (kW)
    load_profile_days: int = 30          # 负荷曲线天数
    
    # 储能系统参数
    battery_capacity_kwh: float = 500.0   # 电池容量 (kWh)
    max_power_kw: float = 250.0          # 最大功率 (kW)
    round_trip_efficiency: float = 0.9    # 往返效率
    
    # 电价结构
    peak_price: float = 1.2              # 峰时电价 (元/kWh)
    valley_price: float = 0.4            # 谷时电价 (元/kWh)
    normal_price: float = 0.7            # 平时电价 (元/kWh)
    demand_charge: float = 80.0          # 需量电费 (元/kW)
    
    # 削峰目标
    target_peak_reduction: float = 0.3   # 目标削峰比例
    peak_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(9, 12), (18, 22)])
    valley_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(23, 7)])
    
    # 约束条件
    min_soc: float = 0.1                 # 最小SOC
    max_soc: float = 0.9                 # 最大SOC
    max_cycle_depth: float = 0.8         # 最大循环深度

@dataclass
class PeakShavingResults:
    """削峰填谷结果"""
    experiment_id: str
    config: PeakShavingConfig
    
    # 削峰效果
    original_peak_load: float = 0.0      # 原始峰值负荷
    reduced_peak_load: float = 0.0       # 削峰后负荷
    peak_reduction_ratio: float = 0.0    # 削峰比例
    load_factor_improvement: float = 0.0 # 负荷率改善
    
    # 经济效益
    energy_cost_without_storage: float = 0.0    # 无储能时能量成本
    energy_cost_with_storage: float = 0.0       # 有储能时能量成本
    demand_cost_without_storage: float = 0.0    # 无储能时需量成本
    demand_cost_with_storage: float = 0.0       # 有储能时需量成本
    total_cost_savings: float = 0.0             # 总成本节省
    payback_period_years: float = 0.0           # 投资回收期
    
    # 系统性能
    battery_utilization: float = 0.0            # 电池利用率
    avg_cycle_depth: float = 0.0                # 平均循环深度
    total_cycles: int = 0                       # 总循环次数
    energy_throughput_mwh: float = 0.0          # 能量吞吐量
    
    # 时间序列数据
    load_profile: np.ndarray = field(default_factory=lambda: np.array([]))
    battery_power: np.ndarray = field(default_factory=lambda: np.array([]))
    battery_soc: np.ndarray = field(default_factory=lambda: np.array([]))
    net_load: np.ndarray = field(default_factory=lambda: np.array([]))
    electricity_price: np.ndarray = field(default_factory=lambda: np.array([]))

class PeakShavingExperiment:
    """
    削峰填谷案例研究
    评估储能系统在削峰填谷应用中的性能和经济性
    """
    
    def __init__(self, config: PeakShavingConfig, experiment_id: Optional[str] = None):
        """
        初始化削峰填谷实验
        
        Args:
            config: 削峰填谷配置
            experiment_id: 实验ID
        """
        self.config = config
        self.experiment_id = experiment_id or f"peak_shaving_{int(time.time()*1000)}"
        
        # 初始化组件
        self.logger = Logger(f"PeakShaving_{self.experiment_id}")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        self.load_generator = LoadProfileGenerator()
        
        # 生成负荷数据
        self._generate_load_data()
        
        # 生成电价数据
        self._generate_price_data()
        
        # 创建实验目录
        self.experiment_dir = f"experiments/case_studies/peak_shaving/{self.experiment_id}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"✅ 削峰填谷实验初始化完成: {config.scenario_type.value}")
        print(f"   实验ID: {self.experiment_id}")
        print(f"   负荷范围: {config.base_load_kw:.1f} - {config.peak_load_kw:.1f} kW")
    
    def run_case_study(self) -> PeakShavingResults:
        """
        运行削峰填谷案例研究
        
        Returns:
            削峰填谷结果
        """
        study_start_time = time.time()
        
        self.logger.info(f"🚀 开始削峰填谷案例研究: {self.config.scenario_type.value}")
        
        try:
            # 阶段1: 基线分析（无储能）
            self.logger.info("📊 阶段1: 基线分析（无储能）")
            baseline_results = self._analyze_baseline()
            
            # 阶段2: 储能控制策略训练
            self.logger.info("🎯 阶段2: 储能控制策略训练")
            control_strategy = self._train_control_strategy()
            
            # 阶段3: 削峰填谷仿真
            self.logger.info("⚡ 阶段3: 削峰填谷仿真")
            simulation_results = self._simulate_peak_shaving(control_strategy)
            
            # 阶段4: 经济性分析
            self.logger.info("💰 阶段4: 经济性分析")
            economic_analysis = self._analyze_economics(baseline_results, simulation_results)
            
            # 阶段5: 性能评估
            self.logger.info("📈 阶段5: 性能评估")
            performance_metrics = self._evaluate_performance(simulation_results)
            
            # 阶段6: 结果整合和可视化
            self.logger.info("📊 阶段6: 结果整合和可视化")
            final_results = self._integrate_results(
                baseline_results, simulation_results, 
                economic_analysis, performance_metrics
            )
            
            # 生成报告
            self._generate_case_study_report(final_results)
            
            study_time = time.time() - study_start_time
            self.logger.info(f"✅ 削峰填谷案例研究完成，用时: {study_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 削峰填谷案例研究失败: {str(e)}")
            raise
    
    def _generate_load_data(self):
        """生成负荷数据"""
        # 根据场景类型选择负荷模式
        load_pattern_map = {
            PeakShavingScenario.COMMERCIAL_BUILDING: LoadPattern.COMMERCIAL,
            PeakShavingScenario.INDUSTRIAL_FACILITY: LoadPattern.INDUSTRIAL,
            PeakShavingScenario.RESIDENTIAL_COMPLEX: LoadPattern.RESIDENTIAL,
            PeakShavingScenario.MIXED_USE: LoadPattern.MIXED,
            PeakShavingScenario.DATA_CENTER: LoadPattern.DATA_CENTER,
            PeakShavingScenario.HOSPITAL: LoadPattern.HOSPITAL
        }
        
        load_pattern = load_pattern_map.get(self.config.scenario_type, LoadPattern.COMMERCIAL)
        
        # 生成负荷曲线
        self.load_profile = self.load_generator.generate_load_profile(
            load_pattern=load_pattern,
            duration_hours=self.config.load_profile_days * 24,
            time_resolution_minutes=15,  # 15分钟分辨率
            parameters=self._get_load_parameters()
        )
        
        self.logger.info(f"生成负荷数据: {len(self.load_profile.load_values)} 个数据点")
    
    def _get_load_parameters(self):
        """获取负荷参数"""
        from data_processing.load_profile_generator import LoadParameters
        
        return LoadParameters(
            base_load=self.config.base_load_kw,
            peak_load=self.config.peak_load_kw,
            load_factor=0.7,
            peak_hours=self.config.peak_hours,
            valley_hours=self.config.valley_hours,
            noise_level=0.05,
            variation_coefficient=0.1
        )
    
    def _generate_price_data(self):
        """生成电价数据"""
        num_points = len(self.load_profile.timestamps)
        hours = self.load_profile.timestamps % 24
        
        # 初始化电价
        self.electricity_price = np.full(num_points, self.config.normal_price)
        
        # 设置峰时电价
        for start_hour, end_hour in self.config.peak_hours:
            peak_mask = (hours >= start_hour) & (hours <= end_hour)
            self.electricity_price[peak_mask] = self.config.peak_price
        
        # 设置谷时电价
        for start_hour, end_hour in self.config.valley_hours:
            if start_hour > end_hour:  # 跨午夜
                valley_mask = (hours >= start_hour) | (hours <= end_hour)
            else:
                valley_mask = (hours >= start_hour) & (hours <= end_hour)
            self.electricity_price[valley_mask] = self.config.valley_price
        
        self.logger.info("生成电价数据完成")
    
    def _analyze_baseline(self) -> Dict[str, Any]:
        """分析基线（无储能）情况"""
        load_data = self.load_profile.load_values
        price_data = self.electricity_price
        
        # 基线负荷特性
        baseline_results = {
            'peak_load': np.max(load_data),
            'min_load': np.min(load_data),
            'avg_load': np.mean(load_data),
            'load_factor': np.mean(load_data) / np.max(load_data),
            'load_variance': np.var(load_data)
        }
        
        # 基线成本计算
        time_resolution_hours = 0.25  # 15分钟 = 0.25小时
        
        # 能量成本
        energy_consumption = load_data * time_resolution_hours  # kWh
        energy_cost = np.sum(energy_consumption * price_data)
        
        # 需量成本（基于月最大需量）
        daily_peaks = []
        points_per_day = 96  # 24小时 * 4点/小时
        
        for day in range(0, len(load_data), points_per_day):
            day_data = load_data[day:day + points_per_day]
            if len(day_data) > 0:
                daily_peaks.append(np.max(day_data))
        
        if daily_peaks:
            monthly_peak = np.max(daily_peaks)
            demand_cost = monthly_peak * self.config.demand_charge * (self.config.load_profile_days / 30)
        else:
            demand_cost = 0
        
        baseline_results.update({
            'energy_cost': energy_cost,
            'demand_cost': demand_cost,
            'total_cost': energy_cost + demand_cost,
            'monthly_peak': monthly_peak if 'monthly_peak' in locals() else 0
        })
        
        self.logger.info(f"基线分析完成 - 峰值负荷: {baseline_results['peak_load']:.1f} kW")
        
        return baseline_results
    
    def _train_control_strategy(self) -> Dict[str, Any]:
        """训练储能控制策略"""
        # 创建DRL训练配置
        experiment_settings = ExperimentSettings(
            experiment_name=f"peak_shaving_training_{self.config.scenario_type.value}",
            experiment_type=ExperimentType.HIERARCHICAL,
            description="削峰填谷控制策略训练",
            total_episodes=500,  # 减少训练回合以加快案例研究
            evaluation_frequency=100,
            save_frequency=200,
            scenario_types=[ScenarioType.PEAK_SHAVING],
            environment_variations=3,
            use_pretraining=True,
            enable_hierarchical=True,
            evaluation_episodes=20,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        # 运行DRL训练
        training_experiment = BasicExperiment(
            settings=experiment_settings,
            experiment_id=f"{self.experiment_id}_training"
        )
        
        training_results = training_experiment.run_experiment()
        
        # 提取控制策略
        control_strategy = {
            'type': 'drl_trained',
            'model_path': training_results.best_checkpoint_path,
            'performance': training_results.best_performance,
            'training_time': training_results.training_time
        }
        
        self.logger.info("控制策略训练完成")
        
        return control_strategy
    
    def _simulate_peak_shaving(self, control_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """仿真削峰填谷过程"""
        load_data = self.load_profile.load_values
        num_points = len(load_data)
        
        # 初始化储能系统状态
        battery_soc = np.zeros(num_points)
        battery_power = np.zeros(num_points)
        net_load = np.zeros(num_points)
        
        # 初始SOC
        current_soc = 0.5  # 50%初始SOC
        
        # 简化的削峰控制逻辑（实际实现中会使用训练好的DRL策略）
        for i in range(num_points):
            current_load = load_data[i]
            current_hour = self.load_profile.timestamps[i] % 24
            
            # 判断是否为峰时或谷时
            is_peak_hour = any(start <= current_hour <= end for start, end in self.config.peak_hours)
            is_valley_hour = any(
                (start <= current_hour <= end if start <= end else 
                 current_hour >= start or current_hour <= end) 
                for start, end in self.config.valley_hours
            )
            
            # 控制策略
            target_power = 0  # 目标功率（正为充电，负为放电）
            
            if is_peak_hour:
                # 峰时：如果负荷高且SOC足够，则放电削峰
                peak_threshold = self.config.base_load_kw * (1 + self.config.target_peak_reduction)
                if current_load > peak_threshold and current_soc > self.config.min_soc:
                    max_discharge = min(
                        self.config.max_power_kw,
                        current_load - peak_threshold,
                        (current_soc - self.config.min_soc) * self.config.battery_capacity_kwh / 0.25  # 15分钟放电量
                    )
                    target_power = -max_discharge
                    
            elif is_valley_hour:
                # 谷时：如果SOC不足，则充电储能
                if current_soc < self.config.max_soc:
                    max_charge = min(
                        self.config.max_power_kw,
                        (self.config.max_soc - current_soc) * self.config.battery_capacity_kwh / 0.25
                    )
                    target_power = max_charge
            
            # 执行功率控制
            battery_power[i] = target_power
            
            # 更新SOC
            if target_power > 0:  # 充电
                energy_change = target_power * 0.25 * self.config.round_trip_efficiency
            else:  # 放电
                energy_change = target_power * 0.25 / self.config.round_trip_efficiency
            
            current_soc += energy_change / self.config.battery_capacity_kwh
            current_soc = np.clip(current_soc, self.config.min_soc, self.config.max_soc)
            
            battery_soc[i] = current_soc
            net_load[i] = current_load + battery_power[i]  # 净负荷 = 原负荷 + 电池功率
        
        simulation_results = {
            'load_profile': load_data,
            'battery_power': battery_power,
            'battery_soc': battery_soc,
            'net_load': net_load,
            'electricity_price': self.electricity_price,
            'timestamps': self.load_profile.timestamps
        }
        
        self.logger.info("削峰填谷仿真完成")
        
        return simulation_results
    
    def _analyze_economics(self, baseline_results: Dict[str, Any], 
                          simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析经济性"""
        net_load = simulation_results['net_load']
        price_data = simulation_results['electricity_price']
        
        # 有储能时的成本计算
        time_resolution_hours = 0.25
        
        # 能量成本
        energy_consumption_with_storage = np.maximum(0, net_load) * time_resolution_hours
        energy_cost_with_storage = np.sum(energy_consumption_with_storage * price_data)
        
        # 需量成本
        daily_peaks_with_storage = []
        points_per_day = 96
        
        for day in range(0, len(net_load), points_per_day):
            day_data = net_load[day:day + points_per_day]
            if len(day_data) > 0:
                daily_peaks_with_storage.append(np.max(day_data))
        
        if daily_peaks_with_storage:
            monthly_peak_with_storage = np.max(daily_peaks_with_storage)
            demand_cost_with_storage = (monthly_peak_with_storage * self.config.demand_charge * 
                                      (self.config.load_profile_days / 30))
        else:
            demand_cost_with_storage = 0
        
        # 成本节省计算
        energy_savings = baseline_results['energy_cost'] - energy_cost_with_storage
        demand_savings = baseline_results['demand_cost'] - demand_cost_with_storage
        total_savings = energy_savings + demand_savings
        
        # 年化收益和投资回收期
        annual_savings = total_savings * (365 / self.config.load_profile_days)
        
        # 简化的储能系统投资成本估算
        battery_cost_per_kwh = 1500  # 元/kWh
        pcs_cost_per_kw = 800       # 元/kW
        total_investment = (self.config.battery_capacity_kwh * battery_cost_per_kwh + 
                          self.config.max_power_kw * pcs_cost_per_kw)
        
        payback_period = total_investment / annual_savings if annual_savings > 0 else float('inf')
        
        economic_analysis = {
            'energy_cost_with_storage': energy_cost_with_storage,
            'demand_cost_with_storage': demand_cost_with_storage,
            'total_cost_with_storage': energy_cost_with_storage + demand_cost_with_storage,
            'energy_savings': energy_savings,
            'demand_savings': demand_savings,
            'total_savings': total_savings,
            'annual_savings': annual_savings,
            'total_investment': total_investment,
            'payback_period_years': payback_period,
            'roi_percent': (annual_savings / total_investment * 100) if total_investment > 0 else 0
        }
        
        self.logger.info(f"经济性分析完成 - 年化节省: {annual_savings:.0f} 元")
        
        return economic_analysis
    
    def _evaluate_performance(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估性能指标"""
        load_data = simulation_results['load_profile']
        net_load = simulation_results['net_load']
        battery_power = simulation_results['battery_power']
        battery_soc = simulation_results['battery_soc']
        
        # 削峰效果
        original_peak = np.max(load_data)
        reduced_peak = np.max(net_load)
        peak_reduction_ratio = (original_peak - reduced_peak) / original_peak
        
        # 负荷率改善
        original_load_factor = np.mean(load_data) / original_peak
        new_load_factor = np.mean(net_load) / reduced_peak if reduced_peak > 0 else 0
        load_factor_improvement = new_load_factor - original_load_factor
        
        # 电池利用率
        max_possible_energy = self.config.battery_capacity_kwh * (self.config.max_soc - self.config.min_soc)
        actual_energy_range = np.max(battery_soc) - np.min(battery_soc)
        battery_utilization = actual_energy_range / (self.config.max_soc - self.config.min_soc)
        
        # 循环分析
        soc_changes = np.abs(np.diff(battery_soc))
        avg_cycle_depth = np.mean(soc_changes) * 2  # 近似循环深度
        
        # 能量吞吐量
        energy_throughput = np.sum(np.abs(battery_power[battery_power != 0])) * 0.25 / 1000  # MWh
        
        # 估算循环次数
        total_cycles = np.sum(soc_changes) / 2  # 简化计算
        
        performance_metrics = {
            'original_peak_load': original_peak,
            'reduced_peak_load': reduced_peak,
            'peak_reduction_ratio': peak_reduction_ratio,
            'original_load_factor': original_load_factor,
            'new_load_factor': new_load_factor,
            'load_factor_improvement': load_factor_improvement,
            'battery_utilization': battery_utilization,
            'avg_cycle_depth': avg_cycle_depth,
            'total_cycles': int(total_cycles),
            'energy_throughput_mwh': energy_throughput
        }
        
        self.logger.info(f"性能评估完成 - 削峰比例: {peak_reduction_ratio:.1%}")
        
        return performance_metrics
    
    def _integrate_results(self, baseline_results: Dict[str, Any],
                          simulation_results: Dict[str, Any],
                          economic_analysis: Dict[str, Any],
                          performance_metrics: Dict[str, Any]) -> PeakShavingResults:
        """整合所有结果"""
        results = PeakShavingResults(
            experiment_id=self.experiment_id,
            config=self.config
        )
        
        # 削峰效果
        results.original_peak_load = performance_metrics['original_peak_load']
        results.reduced_peak_load = performance_metrics['reduced_peak_load']
        results.peak_reduction_ratio = performance_metrics['peak_reduction_ratio']
        results.load_factor_improvement = performance_metrics['load_factor_improvement']
        
        # 经济效益
        results.energy_cost_without_storage = baseline_results['energy_cost']
        results.energy_cost_with_storage = economic_analysis['energy_cost_with_storage']
        results.demand_cost_without_storage = baseline_results['demand_cost']
        results.demand_cost_with_storage = economic_analysis['demand_cost_with_storage']
        results.total_cost_savings = economic_analysis['total_savings']
        results.payback_period_years = economic_analysis['payback_period_years']
        
        # 系统性能
        results.battery_utilization = performance_metrics['battery_utilization']
        results.avg_cycle_depth = performance_metrics['avg_cycle_depth']
        results.total_cycles = performance_metrics['total_cycles']
        results.energy_throughput_mwh = performance_metrics['energy_throughput_mwh']
        
        # 时间序列数据
        results.load_profile = simulation_results['load_profile']
        results.battery_power = simulation_results['battery_power']
        results.battery_soc = simulation_results['battery_soc']
        results.net_load = simulation_results['net_load']
        results.electricity_price = simulation_results['electricity_price']
        
        return results
    
    def _generate_case_study_report(self, results: PeakShavingResults):
        """生成案例研究报告"""
        report = {
            'case_study_info': {
                'experiment_id': results.experiment_id,
                'scenario_type': results.config.scenario_type.value,
                'study_period_days': results.config.load_profile_days,
                'battery_capacity_kwh': results.config.battery_capacity_kwh,
                'max_power_kw': results.config.max_power_kw
            },
            'peak_shaving_performance': {
                'original_peak_load_kw': results.original_peak_load,
                'reduced_peak_load_kw': results.reduced_peak_load,
                'peak_reduction_ratio': results.peak_reduction_ratio,
                'peak_reduction_kw': results.original_peak_load - results.reduced_peak_load,
                'load_factor_improvement': results.load_factor_improvement
            },
            'economic_analysis': {
                'cost_without_storage': {
                    'energy_cost': results.energy_cost_without_storage,
                    'demand_cost': results.demand_cost_without_storage,
                    'total_cost': results.energy_cost_without_storage + results.demand_cost_without_storage
                },
                'cost_with_storage': {
                    'energy_cost': results.energy_cost_with_storage,
                    'demand_cost': results.demand_cost_with_storage,
                    'total_cost': results.energy_cost_with_storage + results.demand_cost_with_storage
                },
                'savings': {
                    'total_savings': results.total_cost_savings,
                    'annual_savings_estimate': results.total_cost_savings * (365 / results.config.load_profile_days),
                    'payback_period_years': results.payback_period_years
                }
            },
            'system_performance': {
                'battery_utilization': results.battery_utilization,
                'avg_cycle_depth': results.avg_cycle_depth,
                'total_cycles': results.total_cycles,
                'energy_throughput_mwh': results.energy_throughput_mwh
            },
            'key_findings': [],
            'recommendations': []
        }
        
        # 关键发现
        if results.peak_reduction_ratio > 0.2:
            report['key_findings'].append(f"显著削峰效果：削峰比例达到{results.peak_reduction_ratio:.1%}")
        
        if results.payback_period_years < 8:
            report['key_findings'].append(f"良好的经济性：投资回收期{results.payback_period_years:.1f}年")
        
        if results.battery_utilization > 0.7:
            report['key_findings'].append(f"高电池利用率：{results.battery_utilization:.1%}")
        
        # 建议
        if results.peak_reduction_ratio < results.config.target_peak_reduction:
            report['recommendations'].append("建议优化控制策略以达到目标削峰比例")
        
        if results.payback_period_years > 10:
            report['recommendations'].append("建议重新评估投资方案或寻找额外收益来源")
        
        if results.avg_cycle_depth > 0.8:
            report['recommendations'].append("建议控制循环深度以延长电池寿命")
        
        # 保存报告
        report_path = os.path.join(self.experiment_dir, "peak_shaving_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可视化
        self._create_visualizations(results)
        
        self.logger.info(f"案例研究报告已保存: {report_path}")
        
        return report
    
    def _create_visualizations(self, results: PeakShavingResults):
        """创建可视化图表"""
        # 1. 负荷曲线对比图
        load_comparison_config = PlotConfig(
            plot_type=PlotType.LINE,
            title="负荷曲线对比 - 削峰填谷效果",
            x_label="时间 (小时)",
            y_label="负荷 (kW)",
            width=1200,
            height=600,
            save_path=os.path.join(self.experiment_dir, "load_comparison.png")
        )
        
        # 选择前7天数据进行展示
        points_per_week = 7 * 96  # 7天 * 96点/天
        show_points = min(points_per_week, len(results.load_profile))
        
        load_data = {
            'time': results.timestamps[:show_points] / 4,  # 转换为小时
            'original_load': results.load_profile[:show_points],
            'net_load': results.net_load[:show_points]
        }
        
        self.visualizer.create_plot(load_data, load_comparison_config)
        
        # 2. 电池运行状态图
        battery_config = PlotConfig(
            plot_type=PlotType.LINE,
            title="电池运行状态",
            x_label="时间 (小时)",
            y_label="SOC (%)",
            width=1200,
            height=400,
            save_path=os.path.join(self.experiment_dir, "battery_status.png")
        )
        
        battery_data = {
            'time': results.timestamps[:show_points] / 4,
            'soc': results.battery_soc[:show_points] * 100,
            'power': results.battery_power[:show_points] / 10  # 缩放以便显示
        }
        
        self.visualizer.create_plot(battery_data, battery_config)
        
        # 3. 经济效益对比图
        economic_config = PlotConfig(
            plot_type=PlotType.BAR,
            title="经济效益对比",
            x_label="成本类型",
            y_label="成本 (元)",
            width=800,
            height=600,
            save_path=os.path.join(self.experiment_dir, "economic_comparison.png")
        )
        
        economic_data = {
            'without_storage_energy': results.energy_cost_without_storage,
            'with_storage_energy': results.energy_cost_with_storage,
            'without_storage_demand': results.demand_cost_without_storage,
            'with_storage_demand': results.demand_cost_with_storage
        }
        
        self.visualizer.create_plot(economic_data, economic_config)
        
        self.logger.info("可视化图表生成完成")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取案例研究摘要"""
        return {
            'experiment_id': self.experiment_id,
            'scenario_type': self.config.scenario_type.value,
            'system_configuration': {
                'battery_capacity_kwh': self.config.battery_capacity_kwh,
                'max_power_kw': self.config.max_power_kw,
                'round_trip_efficiency': self.config.round_trip_efficiency
            },
            'target_objectives': {
                'peak_reduction_target': self.config.target_peak_reduction,
                'cost_optimization': True,
                'load_factor_improvement': True
            }
        }
