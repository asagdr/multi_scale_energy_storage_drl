import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import math
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class LoadPattern(Enum):
    """负荷模式枚举"""
    RESIDENTIAL = "residential"        # 居民负荷
    COMMERCIAL = "commercial"          # 商业负荷
    INDUSTRIAL = "industrial"          # 工业负荷
    MIXED = "mixed"                    # 混合负荷
    ELECTRIC_VEHICLE = "electric_vehicle"  # 电动汽车负荷
    DATA_CENTER = "data_center"        # 数据中心负荷
    HOSPITAL = "hospital"              # 医院负荷
    SCHOOL = "school"                  # 学校负荷
    RETAIL = "retail"                  # 零售负荷
    MANUFACTURING = "manufacturing"     # 制造业负荷

class SeasonType(Enum):
    """季节类型枚举"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class WeekdayType(Enum):
    """工作日类型枚举"""
    WEEKDAY = "weekday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"
    HOLIDAY = "holiday"

@dataclass
class LoadParameters:
    """负荷参数"""
    base_load: float = 10000.0          # 基础负荷 (W)
    peak_load: float = 50000.0          # 峰值负荷 (W)
    load_factor: float = 0.7            # 负荷率
    diversity_factor: float = 0.8       # 需用系数
    
    # 时间特性
    peak_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(8, 12), (18, 22)])  # 峰值时段
    valley_hours: List[Tuple[int, int]] = field(default_factory=lambda: [(23, 7)])          # 谷值时段
    
    # 随机性参数
    noise_level: float = 0.1            # 噪声水平
    variation_coefficient: float = 0.15  # 变异系数
    correlation_factor: float = 0.8     # 相关性因子
    
    # 季节性参数
    seasonal_variation: float = 0.3     # 季节变化幅度
    weather_sensitivity: float = 0.2    # 天气敏感性
    
    # 特殊事件参数
    event_probability: float = 0.05     # 特殊事件概率
    event_magnitude: float = 2.0        # 事件影响幅度

@dataclass
class LoadProfile:
    """负荷曲线数据"""
    profile_id: str
    load_pattern: LoadPattern
    parameters: LoadParameters
    
    # 时间序列数据
    timestamps: np.ndarray
    load_values: np.ndarray            # 负荷值 (W)
    load_normalized: np.ndarray        # 归一化负荷
    
    # 负荷特征
    peak_load: float                   # 实际峰值负荷
    min_load: float                    # 最小负荷
    avg_load: float                    # 平均负荷
    load_factor: float                 # 实际负荷率
    
    # 统计特征
    load_variance: float               # 负荷方差
    peak_to_average_ratio: float       # 峰平比
    ramp_rate_max: float              # 最大爬坡率
    
    # 元数据
    generation_time: float = field(default_factory=time.time)
    quality_score: float = 0.0

class LoadProfileGenerator:
    """
    负荷曲线生成器
    生成各种类型的真实负荷曲线
    """
    
    def __init__(self, generator_id: str = "LoadProfileGenerator_001"):
        """
        初始化负荷曲线生成器
        
        Args:
            generator_id: 生成器ID
        """
        self.generator_id = generator_id
        
        # === 负荷模式模板 ===
        self.load_templates = {
            LoadPattern.RESIDENTIAL: self._get_residential_template(),
            LoadPattern.COMMERCIAL: self._get_commercial_template(),
            LoadPattern.INDUSTRIAL: self._get_industrial_template(),
            LoadPattern.MIXED: self._get_mixed_template(),
            LoadPattern.ELECTRIC_VEHICLE: self._get_ev_template(),
            LoadPattern.DATA_CENTER: self._get_datacenter_template(),
            LoadPattern.HOSPITAL: self._get_hospital_template(),
            LoadPattern.SCHOOL: self._get_school_template(),
            LoadPattern.RETAIL: self._get_retail_template(),
            LoadPattern.MANUFACTURING: self._get_manufacturing_template()
        }
        
        # === 季节性模板 ===
        self.seasonal_templates = {
            SeasonType.SPRING: {'cooling_factor': 0.2, 'heating_factor': 0.3, 'base_factor': 1.0},
            SeasonType.SUMMER: {'cooling_factor': 1.0, 'heating_factor': 0.0, 'base_factor': 1.2},
            SeasonType.AUTUMN: {'cooling_factor': 0.1, 'heating_factor': 0.4, 'base_factor': 0.9},
            SeasonType.WINTER: {'cooling_factor': 0.0, 'heating_factor': 1.0, 'base_factor': 1.1}
        }
        
        # === 工作日模板 ===
        self.weekday_templates = {
            WeekdayType.WEEKDAY: {'activity_factor': 1.0, 'peak_shift': 0.0},
            WeekdayType.SATURDAY: {'activity_factor': 0.8, 'peak_shift': 2.0},
            WeekdayType.SUNDAY: {'activity_factor': 0.6, 'peak_shift': 3.0},
            WeekdayType.HOLIDAY: {'activity_factor': 0.5, 'peak_shift': 4.0}
        }
        
        # === 生成统计 ===
        self.generation_stats = {
            'total_profiles': 0,
            'profiles_by_pattern': {pattern: 0 for pattern in LoadPattern},
            'total_data_points': 0,
            'generation_time': 0.0
        }
        
        print(f"✅ 负荷曲线生成器初始化完成: {generator_id}")
        print(f"   支持负荷模式: {len(self.load_templates)} 种")
    
    def generate_load_profile(self,
                            load_pattern: LoadPattern,
                            duration_hours: float = 24.0,
                            time_resolution_minutes: float = 1.0,
                            parameters: Optional[LoadParameters] = None,
                            season: SeasonType = SeasonType.SUMMER,
                            weekday_type: WeekdayType = WeekdayType.WEEKDAY,
                            profile_id: Optional[str] = None) -> LoadProfile:
        """
        生成负荷曲线
        
        Args:
            load_pattern: 负荷模式
            duration_hours: 持续时间（小时）
            time_resolution_minutes: 时间分辨率（分钟）
            parameters: 负荷参数
            season: 季节
            weekday_type: 工作日类型
            profile_id: 曲线ID
            
        Returns:
            生成的负荷曲线
        """
        generation_start_time = time.time()
        
        # 使用默认参数或提供的参数
        if parameters is None:
            parameters = LoadParameters()
        
        # 生成曲线ID
        if profile_id is None:
            profile_id = f"{load_pattern.value}_{int(time.time()*1000)}"
        
        # 生成时间序列
        timestamps = self._generate_timestamps(duration_hours, time_resolution_minutes)
        
        # 获取模板
        load_template = self.load_templates[load_pattern]
        seasonal_template = self.seasonal_templates[season]
        weekday_template = self.weekday_templates[weekday_type]
        
        # 生成基础负荷曲线
        base_profile = self._generate_base_profile(
            timestamps, parameters, load_template, seasonal_template, weekday_template
        )
        
        # 应用季节性调整
        seasonal_profile = self._apply_seasonal_adjustment(
            base_profile, timestamps, parameters, seasonal_template
        )
        
        # 应用天气影响
        weather_adjusted_profile = self._apply_weather_effects(
            seasonal_profile, timestamps, parameters, season
        )
        
        # 添加随机变化
        noisy_profile = self._add_random_variations(
            weather_adjusted_profile, parameters
        )
        
        # 应用特殊事件
        final_profile = self._apply_special_events(
            noisy_profile, timestamps, parameters
        )
        
        # 计算负荷特征
        load_features = self._calculate_load_features(final_profile, parameters)
        
        # 归一化
        normalized_profile = final_profile / np.max(final_profile)
        
        # 评估质量
        quality_score = self._assess_profile_quality(final_profile, parameters)
        
        # 创建负荷曲线对象
        load_profile = LoadProfile(
            profile_id=profile_id,
            load_pattern=load_pattern,
            parameters=parameters,
            timestamps=timestamps,
            load_values=final_profile,
            load_normalized=normalized_profile,
            peak_load=load_features['peak_load'],
            min_load=load_features['min_load'],
            avg_load=load_features['avg_load'],
            load_factor=load_features['load_factor'],
            load_variance=load_features['load_variance'],
            peak_to_average_ratio=load_features['peak_to_average_ratio'],
            ramp_rate_max=load_features['ramp_rate_max'],
            quality_score=quality_score
        )
        
        # 更新统计
        generation_time = time.time() - generation_start_time
        self._update_generation_stats(load_pattern, len(timestamps), generation_time)
        
        print(f"✅ 负荷曲线生成完成: {profile_id}")
        print(f"   模式: {load_pattern.value}, 峰值: {load_features['peak_load']:.0f}W, "
              f"负荷率: {load_features['load_factor']:.3f}")
        
        return load_profile
    
    def generate_batch_profiles(self,
                              profile_configs: List[Dict[str, Any]],
                              batch_id: Optional[str] = None) -> List[LoadProfile]:
        """
        批量生成负荷曲线
        
        Args:
            profile_configs: 曲线配置列表
            batch_id: 批次ID
            
        Returns:
            生成的负荷曲线列表
        """
        if batch_id is None:
            batch_id = f"batch_{int(time.time()*1000)}"
        
        batch_start_time = time.time()
        profiles = []
        
        print(f"🚀 开始批量生成负荷曲线: {len(profile_configs)} 条")
        
        for i, config in enumerate(profile_configs):
            try:
                load_pattern = LoadPattern(config['pattern'])
                duration = config.get('duration_hours', 24.0)
                resolution = config.get('time_resolution_minutes', 1.0)
                season = SeasonType(config.get('season', 'summer'))
                weekday = WeekdayType(config.get('weekday_type', 'weekday'))
                
                # 构建参数
                parameters = LoadParameters()
                if 'parameters' in config:
                    param_dict = config['parameters']
                    for key, value in param_dict.items():
                        if hasattr(parameters, key):
                            setattr(parameters, key, value)
                
                profile_id = config.get('id', f"{batch_id}_profile_{i+1}")
                
                profile = self.generate_load_profile(
                    load_pattern, duration, resolution, parameters, season, weekday, profile_id
                )
                profiles.append(profile)
                
                if (i + 1) % 10 == 0:
                    print(f"   进度: {i+1}/{len(profile_configs)}")
                
            except Exception as e:
                print(f"⚠️ 负荷曲线 {i+1} 生成失败: {str(e)}")
        
        batch_time = time.time() - batch_start_time
        print(f"✅ 批量生成完成: {len(profiles)}/{len(profile_configs)} 条曲线, 用时: {batch_time:.2f}s")
        
        return profiles
    
    def _generate_timestamps(self, duration_hours: float, resolution_minutes: float) -> np.ndarray:
        """生成时间戳"""
        resolution_hours = resolution_minutes / 60.0
        num_points = int(duration_hours / resolution_hours)
        timestamps = np.linspace(0, duration_hours, num_points)
        return timestamps
    
    def _generate_base_profile(self,
                             timestamps: np.ndarray,
                             parameters: LoadParameters,
                             load_template: Dict[str, Any],
                             seasonal_template: Dict[str, Any],
                             weekday_template: Dict[str, Any]) -> np.ndarray:
        """生成基础负荷曲线"""
        num_points = len(timestamps)
        hours = timestamps % 24  # 转换为小时
        
        # 获取模板参数
        peak_pattern = load_template['peak_pattern']
        valley_pattern = load_template['valley_pattern']
        base_level = load_template['base_level']
        
        # 初始化基础负荷
        base_profile = np.full(num_points, base_level * parameters.base_load)
        
        # 应用峰值模式
        for peak_start, peak_end in parameters.peak_hours:
            peak_mask = ((hours >= peak_start) & (hours <= peak_end))
            if peak_pattern == 'gaussian':
                peak_center = (peak_start + peak_end) / 2
                peak_width = (peak_end - peak_start) / 4
                peak_factor = np.exp(-0.5 * ((hours - peak_center) / peak_width) ** 2)
            elif peak_pattern == 'trapezoidal':
                peak_factor = np.where(peak_mask, 1.0, 0.0)
            else:  # linear
                peak_factor = np.maximum(0, 1 - np.abs(hours - (peak_start + peak_end) / 2) / ((peak_end - peak_start) / 2))
            
            base_profile += peak_factor * (parameters.peak_load - parameters.base_load) * 0.5
        
        # 应用谷值模式
        for valley_start, valley_end in parameters.valley_hours:
            if valley_start > valley_end:  # 跨午夜
                valley_mask = (hours >= valley_start) | (hours <= valley_end)
            else:
                valley_mask = (hours >= valley_start) & (hours <= valley_end)
            
            valley_reduction = 0.3 * parameters.base_load
            base_profile[valley_mask] -= valley_reduction
        
        # 应用工作日调整
        activity_factor = weekday_template['activity_factor']
        peak_shift = weekday_template['peak_shift']
        
        # 时间偏移
        if peak_shift != 0:
            shifted_hours = (hours + peak_shift) % 24
            # 重新计算基于偏移时间的负荷
            base_profile *= activity_factor
        else:
            base_profile *= activity_factor
        
        # 确保最小负荷
        base_profile = np.maximum(base_profile, parameters.base_load * 0.2)
        
        return base_profile
    
    def _apply_seasonal_adjustment(self,
                                 profile: np.ndarray,
                                 timestamps: np.ndarray,
                                 parameters: LoadParameters,
                                 seasonal_template: Dict[str, Any]) -> np.ndarray:
        """应用季节性调整"""
        # 获取季节因子
        base_factor = seasonal_template['base_factor']
        cooling_factor = seasonal_template['cooling_factor']
        heating_factor = seasonal_template['heating_factor']
        
        # 应用基础季节因子
        adjusted_profile = profile * base_factor
        
        # 添加制冷/制热负荷
        hours = timestamps % 24
        
        # 制冷负荷（通常在下午最高）
        cooling_pattern = np.exp(-0.5 * ((hours - 14) / 3) ** 2)  # 下午2点峰值
        cooling_load = cooling_pattern * cooling_factor * parameters.base_load * 0.3
        
        # 制热负荷（通常在早晚最高）
        heating_pattern = (np.exp(-0.5 * ((hours - 7) / 2) ** 2) + 
                          np.exp(-0.5 * ((hours - 19) / 2) ** 2))
        heating_load = heating_pattern * heating_factor * parameters.base_load * 0.25
        
        # 添加到总负荷
        adjusted_profile += (cooling_load + heating_load)
        
        return adjusted_profile
    
    def _apply_weather_effects(self,
                             profile: np.ndarray,
                             timestamps: np.ndarray,
                             parameters: LoadParameters,
                             season: SeasonType) -> np.ndarray:
        """应用天气影响"""
        # 简化的天气模型
        hours = timestamps % 24
        
        # 模拟天气变化
        if season == SeasonType.SUMMER:
            # 夏季：高温增加制冷负荷
            temp_effect = np.sin(2 * np.pi * (hours - 6) / 24) * 0.5 + 0.5  # 日温度变化
            weather_factor = 1.0 + temp_effect * parameters.weather_sensitivity * 0.5
        elif season == SeasonType.WINTER:
            # 冬季：低温增加制热负荷
            temp_effect = np.sin(2 * np.pi * (hours - 6) / 24) * (-0.5) + 0.5  # 反向温度效应
            weather_factor = 1.0 + temp_effect * parameters.weather_sensitivity * 0.4
        else:
            # 春秋季：温和的天气影响
            weather_factor = 1.0 + np.random.normal(0, parameters.weather_sensitivity * 0.1, len(profile))
        
        return profile * weather_factor
    
    def _add_random_variations(self,
                             profile: np.ndarray,
                             parameters: LoadParameters) -> np.ndarray:
        """添加随机变化"""
        # 高斯噪声
        noise = np.random.normal(0, parameters.noise_level * np.mean(profile), len(profile))
        
        # 相关噪声（模拟负荷的时间相关性）
        if parameters.correlation_factor > 0:
            # 简单的一阶自回归噪声
            corr_noise = np.zeros(len(profile))
            corr_noise[0] = np.random.normal(0, parameters.noise_level * np.mean(profile))
            for i in range(1, len(profile)):
                corr_noise[i] = (parameters.correlation_factor * corr_noise[i-1] + 
                               np.sqrt(1 - parameters.correlation_factor**2) * 
                               np.random.normal(0, parameters.noise_level * np.mean(profile)))
            noise = corr_noise
        
        # 周期性变化
        variation_period = 24 / 4  # 6小时周期
        periodic_variation = (np.sin(2 * np.pi * np.arange(len(profile)) / (variation_period * 60)) * 
                            parameters.variation_coefficient * np.mean(profile))
        
        # 组合所有变化
        varied_profile = profile + noise + periodic_variation
        
        # 确保非负
        varied_profile = np.maximum(varied_profile, np.mean(profile) * 0.1)
        
        return varied_profile
    
    def _apply_special_events(self,
                            profile: np.ndarray,
                            timestamps: np.ndarray,
                            parameters: LoadParameters) -> np.ndarray:
        """应用特殊事件"""
        event_profile = profile.copy()
        
        # 随机生成事件
        num_events = np.random.poisson(parameters.event_probability * len(profile) / 1440)  # 每天的事件数
        
        for _ in range(num_events):
            # 随机选择事件时间和持续时间
            event_start = np.random.randint(0, len(profile) - 60)  # 至少1小时空间
            event_duration = np.random.randint(15, 180)  # 15分钟到3小时
            event_end = min(event_start + event_duration, len(profile))
            
            # 随机选择事件类型和幅度
            event_type = np.random.choice(['surge', 'dip', 'step'])
            magnitude = np.random.uniform(0.5, parameters.event_magnitude)
            
            if event_type == 'surge':
                # 负荷激增
                event_profile[event_start:event_end] *= (1 + magnitude)
            elif event_type == 'dip':
                # 负荷骤降
                event_profile[event_start:event_end] *= (1 - magnitude * 0.5)
            else:  # step
                # 阶跃变化
                step_magnitude = magnitude * np.mean(profile) * 0.3
                event_profile[event_start:event_end] += step_magnitude
        
        return event_profile
    
    def _calculate_load_features(self,
                               profile: np.ndarray,
                               parameters: LoadParameters) -> Dict[str, float]:
        """计算负荷特征"""
        peak_load = np.max(profile)
        min_load = np.min(profile)
        avg_load = np.mean(profile)
        
        # 负荷率
        load_factor = avg_load / peak_load if peak_load > 0 else 0
        
        # 负荷方差
        load_variance = np.var(profile)
        
        # 峰平比
        peak_to_average_ratio = peak_load / avg_load if avg_load > 0 else 0
        
        # 最大爬坡率
        ramp_rates = np.abs(np.diff(profile))
        ramp_rate_max = np.max(ramp_rates) if len(ramp_rates) > 0 else 0
        
        return {
            'peak_load': peak_load,
            'min_load': min_load,
            'avg_load': avg_load,
            'load_factor': load_factor,
            'load_variance': load_variance,
            'peak_to_average_ratio': peak_to_average_ratio,
            'ramp_rate_max': ramp_rate_max
        }
    
    def _assess_profile_quality(self,
                              profile: np.ndarray,
                              parameters: LoadParameters) -> float:
        """评估负荷曲线质量"""
        quality_factors = []
        
        # 1. 平滑性评估
        smoothness = 1.0 - np.std(np.diff(profile)) / np.mean(profile)
        quality_factors.append(max(0, smoothness))
        
        # 2. 现实性评估（基于负荷率）
        load_factor = np.mean(profile) / np.max(profile)
        realistic_load_factor = 0.3 <= load_factor <= 0.9
        quality_factors.append(1.0 if realistic_load_factor else 0.5)
        
        # 3. 变化合理性
        max_change_rate = np.max(np.abs(np.diff(profile))) / np.mean(profile)
        reasonable_change = max_change_rate < 0.5  # 单步变化不超过50%
        quality_factors.append(1.0 if reasonable_change else 0.3)
        
        # 4. 峰谷特征
        peak_to_avg = np.max(profile) / np.mean(profile)
        reasonable_peak_ratio = 1.5 <= peak_to_avg <= 5.0
        quality_factors.append(1.0 if reasonable_peak_ratio else 0.7)
        
        # 综合质量分数
        quality_score = np.mean(quality_factors)
        
        return quality_score
    
    def _update_generation_stats(self, load_pattern: LoadPattern, data_points: int, generation_time: float):
        """更新生成统计"""
        self.generation_stats['total_profiles'] += 1
        self.generation_stats['profiles_by_pattern'][load_pattern] += 1
        self.generation_stats['total_data_points'] += data_points
        self.generation_stats['generation_time'] += generation_time
    
    def _get_residential_template(self) -> Dict[str, Any]:
        """获取居民负荷模板"""
        return {
            'peak_pattern': 'gaussian',
            'valley_pattern': 'flat',
            'base_level': 0.4,
            'peak_factor': 2.5,
            'description': '居民负荷：早晚双峰，夜间低谷'
        }
    
    def _get_commercial_template(self) -> Dict[str, Any]:
        """获取商业负荷模板"""
        return {
            'peak_pattern': 'trapezoidal',
            'valley_pattern': 'step',
            'base_level': 0.3,
            'peak_factor': 3.0,
            'description': '商业负荷：工作时间高峰，夜间低谷'
        }
    
    def _get_industrial_template(self) -> Dict[str, Any]:
        """获取工业负荷模板"""
        return {
            'peak_pattern': 'flat',
            'valley_pattern': 'slight_dip',
            'base_level': 0.8,
            'peak_factor': 1.2,
            'description': '工业负荷：相对稳定，维护时段略降'
        }
    
    def _get_mixed_template(self) -> Dict[str, Any]:
        """获取混合负荷模板"""
        return {
            'peak_pattern': 'mixed',
            'valley_pattern': 'moderate',
            'base_level': 0.5,
            'peak_factor': 2.0,
            'description': '混合负荷：综合特征'
        }
    
    def _get_ev_template(self) -> Dict[str, Any]:
        """获取电动汽车负荷模板"""
        return {
            'peak_pattern': 'evening_concentrated',
            'valley_pattern': 'deep_night',
            'base_level': 0.1,
            'peak_factor': 5.0,
            'description': '电动汽车负荷：晚间充电高峰'
        }
    
    def _get_datacenter_template(self) -> Dict[str, Any]:
        """获取数据中心负荷模板"""
        return {
            'peak_pattern': 'constant_high',
            'valley_pattern': 'minimal',
            'base_level': 0.9,
            'peak_factor': 1.1,
            'description': '数据中心负荷：基本恒定，小幅波动'
        }
    
    def _get_hospital_template(self) -> Dict[str, Any]:
        """获取医院负荷模板"""
        return {
            'peak_pattern': 'moderate_day',
            'valley_pattern': 'moderate_night',
            'base_level': 0.7,
            'peak_factor': 1.4,
            'description': '医院负荷：24小时运行，日间略高'
        }
    
    def _get_school_template(self) -> Dict[str, Any]:
        """获取学校负荷模板"""
        return {
            'peak_pattern': 'school_hours',
            'valley_pattern': 'vacation',
            'base_level': 0.2,
            'peak_factor': 4.0,
            'description': '学校负荷：上课时间高峰，假期低谷'
        }
    
    def _get_retail_template(self) -> Dict[str, Any]:
        """获取零售负荷模板"""
        return {
            'peak_pattern': 'business_hours',
            'valley_pattern': 'closed_hours',
            'base_level': 0.3,
            'peak_factor': 3.5,
            'description': '零售负荷：营业时间高峰'
        }
    
    def _get_manufacturing_template(self) -> Dict[str, Any]:
        """获取制造业负荷模板"""
        return {
            'peak_pattern': 'shift_based',
            'valley_pattern': 'shift_change',
            'base_level': 0.6,
            'peak_factor': 1.8,
            'description': '制造业负荷：基于班次的波动'
        }
    
    def analyze_load_profile(self, load_profile: LoadProfile) -> Dict[str, Any]:
        """分析负荷曲线特征"""
        analysis = {
            'basic_statistics': {
                'peak_load': load_profile.peak_load,
                'min_load': load_profile.min_load,
                'avg_load': load_profile.avg_load,
                'load_factor': load_profile.load_factor,
                'peak_to_average_ratio': load_profile.peak_to_average_ratio,
                'variance': load_profile.load_variance,
                'std_deviation': np.sqrt(load_profile.load_variance),
                'coefficient_of_variation': np.sqrt(load_profile.load_variance) / load_profile.avg_load
            },
            
            'temporal_characteristics': {
                'max_ramp_rate': load_profile.ramp_rate_max,
                'avg_ramp_rate': np.mean(np.abs(np.diff(load_profile.load_values))),
                'ramp_rate_std': np.std(np.abs(np.diff(load_profile.load_values)))
            },
            
            'peak_analysis': self._analyze_peaks(load_profile),
            'daily_pattern': self._analyze_daily_pattern(load_profile),
            'quality_assessment': {
                'overall_quality': load_profile.quality_score,
                'data_completeness': 1.0,  # 模拟数据完整性
                'pattern_consistency': self._assess_pattern_consistency(load_profile)
            }
        }
        
        return analysis
    
    def _analyze_peaks(self, load_profile: LoadProfile) -> Dict[str, Any]:
        """分析峰值特征"""
        from scipy.signal import find_peaks
        
        # 找到峰值
        peaks, properties = find_peaks(load_profile.load_values, 
                                     height=load_profile.avg_load * 1.2,
                                     distance=30)  # 至少30分钟间隔
        
        # 找到谷值
        valleys, _ = find_peaks(-load_profile.load_values,
                              height=-load_profile.avg_load * 0.8,
                              distance=30)
        
        peak_analysis = {
            'num_peaks': len(peaks),
            'num_valleys': len(valleys),
            'peak_times': (load_profile.timestamps[peaks] % 24).tolist() if len(peaks) > 0 else [],
            'valley_times': (load_profile.timestamps[valleys] % 24).tolist() if len(valleys) > 0 else [],
            'peak_values': load_profile.load_values[peaks].tolist() if len(peaks) > 0 else [],
            'valley_values': load_profile.load_values[valleys].tolist() if len(valleys) > 0 else [],
            'peak_symmetry': self._calculate_peak_symmetry(load_profile, peaks)
        }
        
        return peak_analysis
    
    def _analyze_daily_pattern(self, load_profile: LoadProfile) -> Dict[str, Any]:
        """分析日模式"""
        hours = load_profile.timestamps % 24
        
        # 按小时统计
        hourly_avg = []
        hourly_std = []
        
        for hour in range(24):
            hour_mask = (hours >= hour) & (hours < hour + 1)
            if np.any(hour_mask):
                hourly_avg.append(np.mean(load_profile.load_values[hour_mask]))
                hourly_std.append(np.std(load_profile.load_values[hour_mask]))
            else:
                hourly_avg.append(0)
                hourly_std.append(0)
        
        daily_pattern = {
            'hourly_average': hourly_avg,
            'hourly_std': hourly_std,
            'peak_hour': np.argmax(hourly_avg),
            'valley_hour': np.argmin(hourly_avg),
            'morning_rise_rate': self._calculate_morning_rise_rate(hourly_avg),
            'evening_decline_rate': self._calculate_evening_decline_rate(hourly_avg)
        }
        
        return daily_pattern
    
    def _assess_pattern_consistency(self, load_profile: LoadProfile) -> float:
        """评估模式一致性"""
        # 计算每小时的变异系数
        hours = load_profile.timestamps % 24
        cv_scores = []
        
        for hour in range(24):
            hour_mask = (hours >= hour) & (hours < hour + 1)
            if np.any(hour_mask) and np.sum(hour_mask) > 1:
                hour_values = load_profile.load_values[hour_mask]
                cv = np.std(hour_values) / (np.mean(hour_values) + 1e-6)
                cv_scores.append(cv)
        
        # 一致性 = 1 - 平均变异系数
        consistency = 1.0 - np.mean(cv_scores) if cv_scores else 0.5
        return max(0, min(1, consistency))
    
    def _calculate_peak_symmetry(self, load_profile: LoadProfile, peaks: np.ndarray) -> float:
        """计算峰值对称性"""
        if len(peaks) < 2:
            return 0.5
        
        # 简化的对称性计算：检查峰值的时间分布
        peak_hours = (load_profile.timestamps[peaks] % 24)
        peak_spacing = np.diff(np.sort(peak_hours))
        
        # 对称性基于峰值间隔的均匀性
        symmetry = 1.0 - np.std(peak_spacing) / (np.mean(peak_spacing) + 1e-6)
        return max(0, min(1, symmetry))
    
    def _calculate_morning_rise_rate(self, hourly_avg: List[float]) -> float:
        """计算晨峰上升率"""
        # 6-10点的上升率
        morning_hours = hourly_avg[6:11]
        if len(morning_hours) > 1:
            rise_rate = (morning_hours[-1] - morning_hours[0]) / len(morning_hours)
            return rise_rate / (np.mean(hourly_avg) + 1e-6)
        return 0.0
    
    def _calculate_evening_decline_rate(self, hourly_avg: List[float]) -> float:
        """计算晚峰下降率"""
        # 20-24点的下降率
        evening_hours = hourly_avg[20:24]
        if len(evening_hours) > 1:
            decline_rate = (evening_hours[0] - evening_hours[-1]) / len(evening_hours)
            return decline_rate / (np.mean(hourly_avg) + 1e-6)
        return 0.0
    
    def export_load_profile(self, load_profile: LoadProfile, file_path: str, format: str = 'csv'):
        """导出负荷曲线"""
        try:
            if format.lower() == 'csv':
                df = pd.DataFrame({
                    'timestamp': load_profile.timestamps,
                    'load_value': load_profile.load_values,
                    'load_normalized': load_profile.load_normalized
                })
                df.to_csv(file_path, index=False)
                
            elif format.lower() == 'json':
                export_data = {
                    'profile_id': load_profile.profile_id,
                    'load_pattern': load_profile.load_pattern.value,
                    'timestamps': load_profile.timestamps.tolist(),
                    'load_values': load_profile.load_values.tolist(),
                    'load_normalized': load_profile.load_normalized.tolist(),
                    'features': {
                        'peak_load': load_profile.peak_load,
                        'min_load': load_profile.min_load,
                        'avg_load': load_profile.avg_load,
                        'load_factor': load_profile.load_factor,
                        'peak_to_average_ratio': load_profile.peak_to_average_ratio
                    },
                    'quality_score': load_profile.quality_score,
                    'generation_time': load_profile.generation_time
                }
                
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            print(f"✅ 负荷曲线已导出: {file_path}")
            
        except Exception as e:
            print(f"❌ 负荷曲线导出失败: {str(e)}")
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息"""
        stats = self.generation_stats.copy()
        
        if stats['total_profiles'] > 0:
            stats['avg_data_points_per_profile'] = stats['total_data_points'] / stats['total_profiles']
            stats['avg_generation_time_per_profile'] = stats['generation_time'] / stats['total_profiles']
        else:
            stats['avg_data_points_per_profile'] = 0
            stats['avg_generation_time_per_profile'] = 0
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"LoadProfileGenerator({self.generator_id}): "
                f"生成曲线={self.generation_stats['total_profiles']}, "
                f"数据点={self.generation_stats['total_data_points']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"LoadProfileGenerator(generator_id='{self.generator_id}', "
                f"load_patterns={len(self.load_templates)}, "
                f"total_profiles={self.generation_stats['total_profiles']})")
