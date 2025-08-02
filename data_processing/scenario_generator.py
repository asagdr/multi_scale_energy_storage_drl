import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ScenarioType(Enum):
    """场景类型枚举"""
    DAILY_CYCLE = "daily_cycle"                    # 日常循环
    SEASONAL_VARIATION = "seasonal_variation"      # 季节变化
    EMERGENCY_RESPONSE = "emergency_response"      # 应急响应
    GRID_SUPPORT = "grid_support"                  # 电网支持
    PEAK_SHAVING = "peak_shaving"                  # 削峰填谷
    FREQUENCY_REGULATION = "frequency_regulation"   # 频率调节
    ENERGY_ARBITRAGE = "energy_arbitrage"          # 能量套利
    RENEWABLE_INTEGRATION = "renewable_integration" # 可再生能源整合
    FAULT_SIMULATION = "fault_simulation"          # 故障仿真
    STRESS_TEST = "stress_test"                    # 压力测试

@dataclass
class ScenarioParameters:
    """场景参数"""
    duration: float = 24.0                 # 持续时间（小时）
    time_resolution: float = 0.01          # 时间分辨率（小时）
    complexity_level: float = 1.0          # 复杂度等级 [0.5, 3.0]
    disturbance_magnitude: float = 0.1     # 干扰幅度 [0.0, 1.0]
    noise_level: float = 0.02              # 噪声水平 [0.0, 0.1]
    
    # 环境参数
    ambient_temperature_range: Tuple[float, float] = (15.0, 35.0)  # 环境温度范围（℃）
    humidity_range: Tuple[float, float] = (30.0, 80.0)            # 湿度范围（%）
    
    # 负荷参数
    base_load: float = 10000.0             # 基础负荷（W）
    peak_load_ratio: float = 2.0           # 峰值负荷比例
    load_variation: float = 0.3            # 负荷变化幅度
    
    # 约束参数
    power_limit: float = 50000.0           # 功率限制（W）
    soc_range: Tuple[float, float] = (20.0, 90.0)  # SOC范围（%）
    temperature_limit: float = 45.0        # 温度限制（℃）
    
    # 随机性参数
    random_seed: Optional[int] = None       # 随机种子

@dataclass
class ScenarioData:
    """场景数据"""
    scenario_id: str
    scenario_type: ScenarioType
    parameters: ScenarioParameters
    
    # 时间序列数据
    timestamps: np.ndarray                  # 时间戳
    power_demand: np.ndarray               # 功率需求（W）
    power_price: np.ndarray                # 电价（元/kWh）
    ambient_temperature: np.ndarray        # 环境温度（℃）
    humidity: np.ndarray                   # 湿度（%）
    
    # 约束数据
    power_limits: np.ndarray               # 功率限制
    soc_targets: np.ndarray                # SOC目标
    temperature_limits: np.ndarray         # 温度限制
    
    # 事件数据
    events: List[Dict[str, Any]] = field(default_factory=list)  # 特殊事件
    
    # 元数据
    generation_time: float = field(default_factory=time.time)
    data_quality: Dict[str, float] = field(default_factory=dict)

class ScenarioGenerator:
    """
    仿真场景生成器
    生成多样化的储能系统运行场景
    """
    
    def __init__(self, generator_id: str = "ScenarioGenerator_001"):
        """
        初始化场景生成器
        
        Args:
            generator_id: 生成器ID
        """
        self.generator_id = generator_id
        
        # === 场景模板 ===
        self.scenario_templates = {
            ScenarioType.DAILY_CYCLE: self._get_daily_cycle_template(),
            ScenarioType.SEASONAL_VARIATION: self._get_seasonal_template(),
            ScenarioType.EMERGENCY_RESPONSE: self._get_emergency_template(),
            ScenarioType.GRID_SUPPORT: self._get_grid_support_template(),
            ScenarioType.PEAK_SHAVING: self._get_peak_shaving_template(),
            ScenarioType.FREQUENCY_REGULATION: self._get_frequency_regulation_template(),
            ScenarioType.ENERGY_ARBITRAGE: self._get_energy_arbitrage_template(),
            ScenarioType.RENEWABLE_INTEGRATION: self._get_renewable_integration_template(),
            ScenarioType.FAULT_SIMULATION: self._get_fault_simulation_template(),
            ScenarioType.STRESS_TEST: self._get_stress_test_template()
        }
        
        # === 生成统计 ===
        self.generation_stats = {
            'total_scenarios': 0,
            'scenarios_by_type': {scenario_type: 0 for scenario_type in ScenarioType},
            'total_data_points': 0,
            'generation_time': 0.0
        }
        
        # === 数据验证 ===
        self.data_validators = {
            'power_range_check': lambda x: np.all((x >= -100000) & (x <= 100000)),
            'temperature_range_check': lambda x: np.all((x >= -20) & (x <= 60)),
            'soc_range_check': lambda x: np.all((x >= 0) & (x <= 100)),
            'continuity_check': lambda x: np.all(np.abs(np.diff(x)) < np.std(x) * 3)
        }
        
        print(f"✅ 场景生成器初始化完成: {generator_id}")
        print(f"   支持场景类型: {len(self.scenario_templates)} 种")
    
    def generate_scenario(self,
                         scenario_type: ScenarioType,
                         parameters: Optional[ScenarioParameters] = None,
                         scenario_id: Optional[str] = None) -> ScenarioData:
        """
        生成指定类型的场景
        
        Args:
            scenario_type: 场景类型
            parameters: 场景参数
            scenario_id: 场景ID
            
        Returns:
            生成的场景数据
        """
        generation_start_time = time.time()
        
        # 使用默认参数或提供的参数
        if parameters is None:
            parameters = ScenarioParameters()
        
        # 生成场景ID
        if scenario_id is None:
            scenario_id = f"{scenario_type.value}_{int(time.time()*1000)}"
        
        # 设置随机种子
        if parameters.random_seed is not None:
            np.random.seed(parameters.random_seed)
        
        # 生成时间序列
        timestamps = self._generate_timestamps(parameters)
        
        # 获取场景模板
        template = self.scenario_templates[scenario_type]
        
        # 生成场景数据
        scenario_data = self._generate_scenario_data(
            scenario_type, parameters, template, timestamps, scenario_id
        )
        
        # 数据验证
        self._validate_scenario_data(scenario_data)
        
        # 更新统计
        generation_time = time.time() - generation_start_time
        self._update_generation_stats(scenario_type, len(timestamps), generation_time)
        
        print(f"✅ 场景生成完成: {scenario_id}")
        print(f"   类型: {scenario_type.value}, 数据点: {len(timestamps)}, 用时: {generation_time:.2f}s")
        
        return scenario_data
    
    def generate_batch_scenarios(self,
                                scenario_configs: List[Dict[str, Any]],
                                batch_id: Optional[str] = None) -> List[ScenarioData]:
        """
        批量生成场景
        
        Args:
            scenario_configs: 场景配置列表
            batch_id: 批次ID
            
        Returns:
            生成的场景数据列表
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time()*1000)}"
        
        batch_start_time = time.time()
        scenarios = []
        
        print(f"🚀 开始批量生成场景: {len(scenario_configs)} 个场景")
        
        for i, config in enumerate(scenario_configs):
            try:
                scenario_type = ScenarioType(config['type'])
                parameters = ScenarioParameters(**config.get('parameters', {}))
                scenario_id = config.get('id', f"{batch_id}_scenario_{i+1}")
                
                scenario = self.generate_scenario(scenario_type, parameters, scenario_id)
                scenarios.append(scenario)
                
                if (i + 1) % 10 == 0:
                    print(f"   进度: {i+1}/{len(scenario_configs)}")
                
            except Exception as e:
                print(f"⚠️ 场景 {i+1} 生成失败: {str(e)}")
        
        batch_time = time.time() - batch_start_time
        print(f"✅ 批量生成完成: {len(scenarios)}/{len(scenario_configs)} 个场景, 用时: {batch_time:.2f}s")
        
        return scenarios
    
    def _generate_timestamps(self, parameters: ScenarioParameters) -> np.ndarray:
        """生成时间戳"""
        num_points = int(parameters.duration / parameters.time_resolution)
        timestamps = np.linspace(0, parameters.duration, num_points)
        return timestamps
    
    def _generate_scenario_data(self,
                               scenario_type: ScenarioType,
                               parameters: ScenarioParameters,
                               template: Dict[str, Any],
                               timestamps: np.ndarray,
                               scenario_id: str) -> ScenarioData:
        """生成场景数据"""
        num_points = len(timestamps)
        
        # 基础环境数据
        ambient_temperature = self._generate_temperature_profile(
            timestamps, parameters, template
        )
        
        humidity = self._generate_humidity_profile(
            timestamps, parameters, template
        )
        
        # 负荷数据
        power_demand = self._generate_power_demand_profile(
            timestamps, parameters, template
        )
        
        # 电价数据
        power_price = self._generate_power_price_profile(
            timestamps, parameters, template
        )
        
        # 约束数据
        power_limits = self._generate_power_limits(
            timestamps, parameters, template
        )
        
        soc_targets = self._generate_soc_targets(
            timestamps, parameters, template
        )
        
        temperature_limits = self._generate_temperature_limits(
            timestamps, parameters, template
        )
        
        # 特殊事件
        events = self._generate_events(
            timestamps, parameters, template, scenario_type
        )
        
        # 数据质量评估
        data_quality = self._assess_data_quality({
            'power_demand': power_demand,
            'ambient_temperature': ambient_temperature,
            'humidity': humidity,
            'power_price': power_price
        })
        
        # 创建场景数据
        scenario_data = ScenarioData(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            parameters=parameters,
            timestamps=timestamps,
            power_demand=power_demand,
            power_price=power_price,
            ambient_temperature=ambient_temperature,
            humidity=humidity,
            power_limits=power_limits,
            soc_targets=soc_targets,
            temperature_limits=temperature_limits,
            events=events,
            data_quality=data_quality
        )
        
        return scenario_data
    
    def _generate_temperature_profile(self,
                                    timestamps: np.ndarray,
                                    parameters: ScenarioParameters,
                                    template: Dict[str, Any]) -> np.ndarray:
        """生成温度曲线"""
        num_points = len(timestamps)
        temp_min, temp_max = parameters.ambient_temperature_range
        
        # 基础日周期
        daily_cycle = np.sin(2 * np.pi * timestamps / 24 - np.pi/2) * 0.5 + 0.5
        base_temp = temp_min + (temp_max - temp_min) * daily_cycle
        
        # 添加季节性变化（如果是季节场景）
        if 'seasonal_factor' in template:
            seasonal_cycle = np.sin(2 * np.pi * timestamps / (24 * 365)) * template['seasonal_factor']
            base_temp += seasonal_cycle
        
        # 添加随机变化
        noise = np.random.normal(0, parameters.noise_level * (temp_max - temp_min), num_points)
        temperature = base_temp + noise
        
        # 应用复杂度调整
        if parameters.complexity_level > 1.0:
            # 添加高频变化
            high_freq = np.sin(2 * np.pi * timestamps * 4) * (parameters.complexity_level - 1.0) * 2
            temperature += high_freq
        
        # 限制范围
        temperature = np.clip(temperature, temp_min - 5, temp_max + 5)
        
        return temperature
    
    def _generate_humidity_profile(self,
                                 timestamps: np.ndarray,
                                 parameters: ScenarioParameters,
                                 template: Dict[str, Any]) -> np.ndarray:
        """生成湿度曲线"""
        num_points = len(timestamps)
        humidity_min, humidity_max = parameters.humidity_range
        
        # 与温度相关的湿度变化（反相关）
        temp_cycle = np.sin(2 * np.pi * timestamps / 24 - np.pi/2) * 0.5 + 0.5
        base_humidity = humidity_max - (humidity_max - humidity_min) * temp_cycle * 0.7
        
        # 添加独立的湿度变化
        humidity_cycle = np.sin(2 * np.pi * timestamps / 24 + np.pi/4) * 0.3
        base_humidity += humidity_cycle * (humidity_max - humidity_min) * 0.3
        
        # 添加噪声
        noise = np.random.normal(0, parameters.noise_level * (humidity_max - humidity_min), num_points)
        humidity = base_humidity + noise
        
        # 限制范围
        humidity = np.clip(humidity, humidity_min, humidity_max)
        
        return humidity
    
    def _generate_power_demand_profile(self,
                                     timestamps: np.ndarray,
                                     parameters: ScenarioParameters,
                                     template: Dict[str, Any]) -> np.ndarray:
        """生成功率需求曲线"""
        num_points = len(timestamps)
        base_load = parameters.base_load
        
        # 获取负荷模式
        load_pattern = template.get('load_pattern', 'typical')
        
        if load_pattern == 'residential':
            # 居民负荷：早晚高峰
            morning_peak = np.exp(-0.5 * ((timestamps % 24 - 7) / 2) ** 2)
            evening_peak = np.exp(-0.5 * ((timestamps % 24 - 19) / 3) ** 2)
            load_profile = 0.5 + 0.3 * morning_peak + 0.4 * evening_peak
            
        elif load_pattern == 'commercial':
            # 商业负荷：工作时间高峰
            work_hours = np.where((timestamps % 24 >= 8) & (timestamps % 24 <= 18), 1.0, 0.3)
            lunch_dip = np.exp(-0.5 * ((timestamps % 24 - 12) / 1) ** 2) * (-0.2)
            load_profile = work_hours + lunch_dip
            
        elif load_pattern == 'industrial':
            # 工业负荷：相对稳定
            base_industrial = 0.8 + 0.1 * np.sin(2 * np.pi * timestamps / 24)
            maintenance_dip = np.where((timestamps % 24 >= 2) & (timestamps % 24 <= 4), -0.3, 0)
            load_profile = base_industrial + maintenance_dip
            
        else:  # typical
            # 典型负荷：双峰模式
            peak1 = np.exp(-0.5 * ((timestamps % 24 - 10) / 3) ** 2) * 0.8
            peak2 = np.exp(-0.5 * ((timestamps % 24 - 20) / 2) ** 2) * 1.0
            valley = np.exp(-0.5 * ((timestamps % 24 - 3) / 2) ** 2) * (-0.3)
            load_profile = 0.6 + peak1 + peak2 + valley
        
        # 应用负荷比例
        power_demand = base_load * load_profile * parameters.peak_load_ratio
        
        # 添加变异性
        variation = np.random.normal(1.0, parameters.load_variation, num_points)
        power_demand *= variation
        
        # 添加干扰
        if parameters.disturbance_magnitude > 0:
            disturbance_times = np.random.random(num_points) < 0.05  # 5%概率
            disturbance_magnitude = np.random.normal(0, parameters.disturbance_magnitude * base_load, num_points)
            power_demand[disturbance_times] += disturbance_magnitude[disturbance_times]
        
        # 确保非负
        power_demand = np.maximum(power_demand, base_load * 0.1)
        
        return power_demand
    
    def _generate_power_price_profile(self,
                                    timestamps: np.ndarray,
                                    parameters: ScenarioParameters,
                                    template: Dict[str, Any]) -> np.ndarray:
        """生成电价曲线"""
        num_points = len(timestamps)
        base_price = template.get('base_price', 0.6)  # 元/kWh
        
        # 峰谷电价
        peak_hours_morning = (timestamps % 24 >= 8) & (timestamps % 24 <= 11)
        peak_hours_evening = (timestamps % 24 >= 18) & (timestamps % 24 <= 22)
        valley_hours = (timestamps % 24 >= 23) | (timestamps % 24 <= 7)
        
        price_multiplier = np.ones(num_points)
        price_multiplier[peak_hours_morning | peak_hours_evening] = 1.5  # 峰时
        price_multiplier[valley_hours] = 0.5  # 谷时
        
        power_price = base_price * price_multiplier
        
        # 添加市场波动
        market_volatility = template.get('market_volatility', 0.1)
        volatility = np.random.normal(1.0, market_volatility, num_points)
        power_price *= volatility
        
        # 确保价格合理
        power_price = np.clip(power_price, 0.1, 2.0)
        
        return power_price
    
    def _generate_power_limits(self,
                             timestamps: np.ndarray,
                             parameters: ScenarioParameters,
                             template: Dict[str, Any]) -> np.ndarray:
        """生成功率限制"""
        num_points = len(timestamps)
        base_limit = parameters.power_limit
        
        # 基本限制
        power_limits = np.full(num_points, base_limit)
        
        # 根据模板调整限制
        if template.get('dynamic_limits', False):
            # 动态限制：根据时间和条件变化
            time_factor = 0.8 + 0.2 * np.sin(2 * np.pi * timestamps / 24)
            power_limits *= time_factor
        
        # 添加随机限制事件
        limit_events = np.random.random(num_points) < 0.02  # 2%概率
        power_limits[limit_events] *= np.random.uniform(0.5, 0.8, np.sum(limit_events))
        
        return power_limits
    
    def _generate_soc_targets(self,
                            timestamps: np.ndarray,
                            parameters: ScenarioParameters,
                            template: Dict[str, Any]) -> np.ndarray:
        """生成SOC目标"""
        num_points = len(timestamps)
        soc_min, soc_max = parameters.soc_range
        
        # 基础SOC目标：根据电价优化
        price_cycle = np.sin(2 * np.pi * timestamps / 24) * 0.5 + 0.5
        soc_targets = soc_min + (soc_max - soc_min) * (1 - price_cycle)  # 低价时高SOC
        
        # 添加策略性调整
        strategy_type = template.get('strategy_type', 'peak_shaving')
        
        if strategy_type == 'peak_shaving':
            # 削峰填谷：峰时放电，谷时充电
            peak_times = (timestamps % 24 >= 18) & (timestamps % 24 <= 22)
            valley_times = (timestamps % 24 >= 23) | (timestamps % 24 <= 7)
            soc_targets[peak_times] = soc_min + (soc_max - soc_min) * 0.3  # 峰时低SOC
            soc_targets[valley_times] = soc_min + (soc_max - soc_min) * 0.8  # 谷时高SOC
        
        elif strategy_type == 'frequency_regulation':
            # 频率调节：保持中等SOC以便双向调节
            soc_targets = np.full(num_points, (soc_min + soc_max) / 2)
        
        # 添加平滑处理
        from scipy.ndimage import gaussian_filter1d
        soc_targets = gaussian_filter1d(soc_targets, sigma=2.0)
        
        return soc_targets
    
    def _generate_temperature_limits(self,
                                   timestamps: np.ndarray,
                                   parameters: ScenarioParameters,
                                   template: Dict[str, Any]) -> np.ndarray:
        """生成温度限制"""
        num_points = len(timestamps)
        base_limit = parameters.temperature_limit
        
        # 基本温度限制
        temperature_limits = np.full(num_points, base_limit)
        
        # 根据环境温度调整
        if template.get('adaptive_limits', False):
            # 高环境温度时降低限制
            temp_factor = 1.0 - (parameters.ambient_temperature_range[1] - 25) / 50
            temperature_limits *= np.clip(temp_factor, 0.8, 1.0)
        
        return temperature_limits
    
    def _generate_events(self,
                        timestamps: np.ndarray,
                        parameters: ScenarioParameters,
                        template: Dict[str, Any],
                        scenario_type: ScenarioType) -> List[Dict[str, Any]]:
        """生成特殊事件"""
        events = []
        
        # 根据场景类型生成特定事件
        if scenario_type == ScenarioType.EMERGENCY_RESPONSE:
            # 紧急事件
            event_time = np.random.uniform(2, 20)  # 2-20小时内发生
            events.append({
                'type': 'power_outage',
                'start_time': event_time,
                'duration': np.random.uniform(0.5, 3.0),  # 0.5-3小时
                'severity': np.random.uniform(0.5, 1.0),
                'description': '电网停电事件'
            })
        
        elif scenario_type == ScenarioType.FAULT_SIMULATION:
            # 故障事件
            num_faults = np.random.poisson(2)  # 平均2个故障
            for i in range(num_faults):
                fault_time = np.random.uniform(0, parameters.duration)
                fault_types = ['sensor_fault', 'actuator_fault', 'communication_fault', 'thermal_fault']
                events.append({
                    'type': np.random.choice(fault_types),
                    'start_time': fault_time,
                    'duration': np.random.uniform(0.1, 1.0),
                    'severity': np.random.uniform(0.2, 0.8),
                    'description': f'故障仿真事件 {i+1}'
                })
        
        elif scenario_type == ScenarioType.GRID_SUPPORT:
            # 电网支持事件
            support_requests = np.random.poisson(3)  # 平均3次支持请求
            for i in range(support_requests):
                request_time = np.random.uniform(0, parameters.duration)
                events.append({
                    'type': 'grid_support_request',
                    'start_time': request_time,
                    'duration': np.random.uniform(0.25, 2.0),
                    'power_request': np.random.uniform(5000, 20000),
                    'description': f'电网支持请求 {i+1}'
                })
        
        # 通用随机事件
        if np.random.random() < 0.3:  # 30%概率发生负荷突变
            surge_time = np.random.uniform(0, parameters.duration)
            events.append({
                'type': 'load_surge',
                'start_time': surge_time,
                'duration': np.random.uniform(0.1, 0.5),
                'magnitude': np.random.uniform(1.5, 3.0),
                'description': '负荷突增事件'
            })
        
        return events
    
    def _assess_data_quality(self, data_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """评估数据质量"""
        quality_metrics = {}
        
        for name, data in data_dict.items():
            # 连续性检查
            continuity_score = 1.0 - np.mean(np.abs(np.diff(data)) > 3 * np.std(data))
            
            # 范围合理性检查
            if name == 'power_demand':
                range_score = float(self.data_validators['power_range_check'](data))
            elif name == 'ambient_temperature':
                range_score = float(self.data_validators['temperature_range_check'](data))
            else:
                range_score = 1.0
            
            # 噪声水平评估
            noise_score = 1.0 - min(1.0, np.std(data) / (np.mean(np.abs(data)) + 1e-6))
            
            # 综合质量分数
            overall_score = (continuity_score + range_score + noise_score) / 3
            
            quality_metrics[f'{name}_quality'] = overall_score
            quality_metrics[f'{name}_continuity'] = continuity_score
            quality_metrics[f'{name}_range'] = range_score
            quality_metrics[f'{name}_noise'] = noise_score
        
        return quality_metrics
    
    def _validate_scenario_data(self, scenario_data: ScenarioData):
        """验证场景数据"""
        # 检查数据维度一致性
        expected_length = len(scenario_data.timestamps)
        data_arrays = [
            scenario_data.power_demand,
            scenario_data.power_price,
            scenario_data.ambient_temperature,
            scenario_data.humidity,
            scenario_data.power_limits,
            scenario_data.soc_targets,
            scenario_data.temperature_limits
        ]
        
        for i, data_array in enumerate(data_arrays):
            if len(data_array) != expected_length:
                raise ValueError(f"数据维度不一致: 数组 {i} 长度为 {len(data_array)}, 期望 {expected_length}")
        
        # 检查数据范围
        if not self.data_validators['power_range_check'](scenario_data.power_demand):
            print("⚠️ 功率需求数据超出合理范围")
        
        if not self.data_validators['temperature_range_check'](scenario_data.ambient_temperature):
            print("⚠️ 温度数据超出合理范围")
    
    def _update_generation_stats(self, scenario_type: ScenarioType, data_points: int, generation_time: float):
        """更新生成统计"""
        self.generation_stats['total_scenarios'] += 1
        self.generation_stats['scenarios_by_type'][scenario_type] += 1
        self.generation_stats['total_data_points'] += data_points
        self.generation_stats['generation_time'] += generation_time
    
    def _get_daily_cycle_template(self) -> Dict[str, Any]:
        """获取日常循环模板"""
        return {
            'load_pattern': 'typical',
            'base_price': 0.6,
            'market_volatility': 0.05,
            'seasonal_factor': 0.0,
            'strategy_type': 'peak_shaving',
            'dynamic_limits': False,
            'adaptive_limits': False
        }
    
    def _get_seasonal_template(self) -> Dict[str, Any]:
        """获取季节变化模板"""
        return {
            'load_pattern': 'residential',
            'base_price': 0.7,
            'market_volatility': 0.1,
            'seasonal_factor': 5.0,  # 更大的季节性变化
            'strategy_type': 'peak_shaving',
            'dynamic_limits': True,
            'adaptive_limits': True
        }
    
    def _get_emergency_template(self) -> Dict[str, Any]:
        """获取应急响应模板"""
        return {
            'load_pattern': 'industrial',
            'base_price': 0.8,
            'market_volatility': 0.2,
            'seasonal_factor': 0.0,
            'strategy_type': 'emergency_backup',
            'dynamic_limits': True,
            'adaptive_limits': True
        }
    
    def _get_grid_support_template(self) -> Dict[str, Any]:
        """获取电网支持模板"""
        return {
            'load_pattern': 'commercial',
            'base_price': 0.9,
            'market_volatility': 0.15,
            'seasonal_factor': 0.0,
            'strategy_type': 'frequency_regulation',
            'dynamic_limits': True,
            'adaptive_limits': False
        }
    
    def _get_peak_shaving_template(self) -> Dict[str, Any]:
        """获取削峰填谷模板"""
        return {
            'load_pattern': 'residential',
            'base_price': 0.6,
            'market_volatility': 0.08,
            'seasonal_factor': 0.0,
            'strategy_type': 'peak_shaving',
            'dynamic_limits': False,
            'adaptive_limits': False
        }
    
    def _get_frequency_regulation_template(self) -> Dict[str, Any]:
        """获取频率调节模板"""
        return {
            'load_pattern': 'typical',
            'base_price': 1.2,  # 频率调节高收益
            'market_volatility': 0.25,
            'seasonal_factor': 0.0,
            'strategy_type': 'frequency_regulation',
            'dynamic_limits': True,
            'adaptive_limits': False
        }
    
    def _get_energy_arbitrage_template(self) -> Dict[str, Any]:
        """获取能量套利模板"""
        return {
            'load_pattern': 'typical',
            'base_price': 0.5,
            'market_volatility': 0.3,  # 高波动性
            'seasonal_factor': 0.0,
            'strategy_type': 'arbitrage',
            'dynamic_limits': False,
            'adaptive_limits': False
        }
    
    def _get_renewable_integration_template(self) -> Dict[str, Any]:
        """获取可再生能源整合模板"""
        return {
            'load_pattern': 'renewable',
            'base_price': 0.4,
            'market_volatility': 0.2,
            'seasonal_factor': 2.0,
            'strategy_type': 'renewable_smoothing',
            'dynamic_limits': True,
            'adaptive_limits': True
        }
    
    def _get_fault_simulation_template(self) -> Dict[str, Any]:
        """获取故障仿真模板"""
        return {
            'load_pattern': 'typical',
            'base_price': 0.6,
            'market_volatility': 0.1,
            'seasonal_factor': 0.0,
            'strategy_type': 'fault_tolerant',
            'dynamic_limits': True,
            'adaptive_limits': True
        }
    
    def _get_stress_test_template(self) -> Dict[str, Any]:
        """获取压力测试模板"""
        return {
            'load_pattern': 'extreme',
            'base_price': 1.0,
            'market_volatility': 0.4,  # 极高波动
            'seasonal_factor': 0.0,
            'strategy_type': 'stress_response',
            'dynamic_limits': True,
            'adaptive_limits': True
        }
    
    def export_scenario(self, scenario_data: ScenarioData, file_path: str, format: str = 'json'):
        """导出场景数据"""
        try:
            if format.lower() == 'json':
                export_data = {
                    'scenario_id': scenario_data.scenario_id,
                    'scenario_type': scenario_data.scenario_type.value,
                    'parameters': {
                        'duration': scenario_data.parameters.duration,
                        'time_resolution': scenario_data.parameters.time_resolution,
                        'complexity_level': scenario_data.parameters.complexity_level,
                        'disturbance_magnitude': scenario_data.parameters.disturbance_magnitude,
                        'noise_level': scenario_data.parameters.noise_level,
                        'base_load': scenario_data.parameters.base_load,
                        'peak_load_ratio': scenario_data.parameters.peak_load_ratio
                    },
                    'timestamps': scenario_data.timestamps.tolist(),
                    'power_demand': scenario_data.power_demand.tolist(),
                    'power_price': scenario_data.power_price.tolist(),
                    'ambient_temperature': scenario_data.ambient_temperature.tolist(),
                    'humidity': scenario_data.humidity.tolist(),
                    'power_limits': scenario_data.power_limits.tolist(),
                    'soc_targets': scenario_data.soc_targets.tolist(),
                    'temperature_limits': scenario_data.temperature_limits.tolist(),
                    'events': scenario_data.events,
                    'data_quality': scenario_data.data_quality,
                    'generation_time': scenario_data.generation_time
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == 'csv':
                df = pd.DataFrame({
                    'timestamp': scenario_data.timestamps,
                    'power_demand': scenario_data.power_demand,
                    'power_price': scenario_data.power_price,
                    'ambient_temperature': scenario_data.ambient_temperature,
                    'humidity': scenario_data.humidity,
                    'power_limits': scenario_data.power_limits,
                    'soc_targets': scenario_data.soc_targets,
                    'temperature_limits': scenario_data.temperature_limits
                })
                df.to_csv(file_path, index=False)
            
            print(f"✅ 场景数据已导出: {file_path}")
            
        except Exception as e:
            print(f"❌ 场景数据导出失败: {str(e)}")
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        stats = self.generation_stats.copy()
        
        if stats['total_scenarios'] > 0:
            stats['avg_data_points_per_scenario'] = stats['total_data_points'] / stats['total_scenarios']
            stats['avg_generation_time_per_scenario'] = stats['generation_time'] / stats['total_scenarios']
        else:
            stats['avg_data_points_per_scenario'] = 0
            stats['avg_generation_time_per_scenario'] = 0
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"ScenarioGenerator({self.generator_id}): "
                f"生成场景={self.generation_stats['total_scenarios']}, "
                f"数据点={self.generation_stats['total_data_points']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"ScenarioGenerator(generator_id='{self.generator_id}', "
                f"scenario_types={len(self.scenario_templates)}, "
                f"total_scenarios={self.generation_stats['total_scenarios']})")
