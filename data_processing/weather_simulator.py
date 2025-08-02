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

class WeatherCondition(Enum):
    """天气条件枚举"""
    CLEAR = "clear"                    # 晴天
    PARTLY_CLOUDY = "partly_cloudy"    # 多云
    CLOUDY = "cloudy"                  # 阴天
    OVERCAST = "overcast"              # 密云
    LIGHT_RAIN = "light_rain"          # 小雨
    MODERATE_RAIN = "moderate_rain"    # 中雨
    HEAVY_RAIN = "heavy_rain"          # 大雨
    SNOW = "snow"                      # 雪
    FOG = "fog"                        # 雾
    STORM = "storm"                    # 暴风雨

class ClimateZone(Enum):
    """气候区域枚举"""
    TROPICAL = "tropical"              # 热带
    SUBTROPICAL = "subtropical"        # 亚热带
    TEMPERATE = "temperate"            # 温带
    CONTINENTAL = "continental"        # 大陆性
    POLAR = "polar"                    # 极地
    DESERT = "desert"                  # 沙漠
    MEDITERRANEAN = "mediterranean"     # 地中海
    OCEANIC = "oceanic"                # 海洋性

@dataclass
class WeatherParameters:
    """天气参数"""
    # 温度参数
    annual_avg_temp: float = 15.0      # 年平均温度 (°C)
    temp_amplitude: float = 10.0       # 温度年振幅 (°C)
    daily_temp_range: float = 8.0      # 日温差 (°C)
    
    # 湿度参数
    annual_avg_humidity: float = 60.0  # 年平均湿度 (%)
    humidity_amplitude: float = 15.0   # 湿度年振幅 (%)
    daily_humidity_range: float = 20.0 # 日湿度变化 (%)
    
    # 太阳辐射参数
    max_solar_irradiance: float = 1000.0  # 最大太阳辐射 (W/m²)
    solar_variation: float = 0.15       # 太阳辐射变化幅度
    
    # 风速参数
    avg_wind_speed: float = 3.0         # 平均风速 (m/s)
    wind_variability: float = 2.0       # 风速变异性
    
    # 降水参数
    annual_precipitation: float = 800.0  # 年降水量 (mm)
    rainy_days_per_year: int = 120      # 年降雨天数
    
    # 大气压力参数
    avg_pressure: float = 1013.25       # 平均大气压 (hPa)
    pressure_variation: float = 20.0    # 大气压变化范围
    
    # 季节性参数
    seasonal_lag: float = 45.0          # 季节滞后 (天)
    climate_variability: float = 0.1    # 气候变异性
    
    # 极端事件参数
    extreme_event_probability: float = 0.02  # 极端事件概率
    extreme_magnitude: float = 2.0      # 极端事件幅度

@dataclass
class WeatherData:
    """天气数据"""
    data_id: str
    climate_zone: ClimateZone
    parameters: WeatherParameters
    
    # 时间序列数据
    timestamps: np.ndarray
    temperature: np.ndarray            # 温度 (°C)
    humidity: np.ndarray               # 相对湿度 (%)
    solar_irradiance: np.ndarray       # 太阳辐射 (W/m²)
    wind_speed: np.ndarray             # 风速 (m/s)
    wind_direction: np.ndarray         # 风向 (度)
    precipitation: np.ndarray          # 降水 (mm/h)
    atmospheric_pressure: np.ndarray   # 大气压 (hPa)
    
    # 天气状态
    weather_conditions: List[WeatherCondition]  # 天气条件序列
    
    # 计算属性
    heat_index: np.ndarray = field(init=False)  # 热指数
    wind_chill: np.ndarray = field(init=False)  # 风寒指数
    dew_point: np.ndarray = field(init=False)   # 露点温度
    
    # 元数据
    generation_time: float = field(default_factory=time.time)
    data_quality: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """计算派生属性"""
        self.heat_index = self._calculate_heat_index()
        self.wind_chill = self._calculate_wind_chill()
        self.dew_point = self._calculate_dew_point()
    
    def _calculate_heat_index(self) -> np.ndarray:
        """计算热指数"""
        T = self.temperature  # 华氏度转换
        T_f = T * 9/5 + 32
        RH = self.humidity
        
        # Rothfusz方程
        heat_index_f = (0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094)))
        
        # 高温高湿修正
        mask = (T_f >= 80) & (RH >= 40)
        if np.any(mask):
            hi_complex = (-42.379 + 2.04901523 * T_f[mask] + 10.14333127 * RH[mask] 
                         - 0.22475541 * T_f[mask] * RH[mask] - 6.83783e-3 * T_f[mask]**2 
                         - 5.481717e-2 * RH[mask]**2 + 1.22874e-3 * T_f[mask]**2 * RH[mask] 
                         + 8.5282e-4 * T_f[mask] * RH[mask]**2 - 1.99e-6 * T_f[mask]**2 * RH[mask]**2)
            heat_index_f[mask] = hi_complex
        
        # 转换回摄氏度
        heat_index_c = (heat_index_f - 32) * 5/9
        return heat_index_c
    
    def _calculate_wind_chill(self) -> np.ndarray:
        """计算风寒指数"""
        T = self.temperature
        V = self.wind_speed * 3.6  # 转换为 km/h
        
        # 只有低温时才计算风寒
        wind_chill = np.where(
            (T <= 10) & (V >= 4.8),
            13.12 + 0.6215 * T - 11.37 * (V**0.16) + 0.3965 * T * (V**0.16),
            T
        )
        
        return wind_chill
    
    def _calculate_dew_point(self) -> np.ndarray:
        """计算露点温度"""
        T = self.temperature
        RH = self.humidity
        
        # Magnus公式
        a = 17.27
        b = 237.7
        
        alpha = ((a * T) / (b + T)) + np.log(RH / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return dew_point

class WeatherSimulator:
    """
    天气仿真器
    生成真实的天气数据用于储能系统仿真
    """
    
    def __init__(self, simulator_id: str = "WeatherSimulator_001"):
        """
        初始化天气仿真器
        
        Args:
            simulator_id: 仿真器ID
        """
        self.simulator_id = simulator_id
        
        # === 气候区域模板 ===
        self.climate_templates = {
            ClimateZone.TROPICAL: self._get_tropical_template(),
            ClimateZone.SUBTROPICAL: self._get_subtropical_template(),
            ClimateZone.TEMPERATE: self._get_temperate_template(),
            ClimateZone.CONTINENTAL: self._get_continental_template(),
            ClimateZone.POLAR: self._get_polar_template(),
            ClimateZone.DESERT: self._get_desert_template(),
            ClimateZone.MEDITERRANEAN: self._get_mediterranean_template(),
            ClimateZone.OCEANIC: self._get_oceanic_template()
        }
        
        # === 天气模式概率 ===
        self.weather_transition_matrix = self._build_weather_transition_matrix()
        
        # === 仿真统计 ===
        self.simulation_stats = {
            'total_simulations': 0,
            'simulations_by_zone': {zone: 0 for zone in ClimateZone},
            'total_data_points': 0,
            'simulation_time': 0.0
        }
        
        print(f"✅ 天气仿真器初始化完成: {simulator_id}")
        print(f"   支持气候区域: {len(self.climate_templates)} 种")
    
    def simulate_weather(self,
                        climate_zone: ClimateZone,
                        duration_hours: float = 24.0 * 365,  # 默认一年
                        time_resolution_minutes: float = 60.0,  # 默认1小时
                        parameters: Optional[WeatherParameters] = None,
                        start_day_of_year: int = 1,
                        data_id: Optional[str] = None) -> WeatherData:
        """
        模拟天气数据
        
        Args:
            climate_zone: 气候区域
            duration_hours: 持续时间（小时）
            time_resolution_minutes: 时间分辨率（分钟）
            parameters: 天气参数
            start_day_of_year: 起始日期（年内第几天）
            data_id: 数据ID
            
        Returns:
            生成的天气数据
        """
        simulation_start_time = time.time()
        
        # 使用默认参数或提供的参数
        if parameters is None:
            template = self.climate_templates[climate_zone]
            parameters = WeatherParameters(**template)
        
        # 生成数据ID
        if data_id is None:
            data_id = f"{climate_zone.value}_{int(time.time()*1000)}"
        
        # 生成时间序列
        timestamps = self._generate_timestamps(duration_hours, time_resolution_minutes)
        
        # 计算日期相关参数
        days_from_start = (timestamps / 24.0) + start_day_of_year
        
        # 生成基础气象要素
        temperature = self._simulate_temperature(timestamps, days_from_start, parameters)
        humidity = self._simulate_humidity(timestamps, days_from_start, parameters, temperature)
        solar_irradiance = self._simulate_solar_irradiance(timestamps, days_from_start, parameters)
        wind_speed, wind_direction = self._simulate_wind(timestamps, parameters)
        precipitation = self._simulate_precipitation(timestamps, days_from_start, parameters)
        atmospheric_pressure = self._simulate_pressure(timestamps, parameters)
        
        # 生成天气条件序列
        weather_conditions = self._simulate_weather_conditions(
            timestamps, temperature, humidity, precipitation, solar_irradiance
        )
        
        # 应用天气条件的相互影响
        temperature, humidity, solar_irradiance = self._apply_weather_interactions(
            temperature, humidity, solar_irradiance, precipitation, weather_conditions
        )
        
        # 评估数据质量
        data_quality = self._assess_weather_quality({
            'temperature': temperature,
            'humidity': humidity,
            'solar_irradiance': solar_irradiance,
            'wind_speed': wind_speed,
            'precipitation': precipitation
        })
        
        # 创建天气数据对象
        weather_data = WeatherData(
            data_id=data_id,
            climate_zone=climate_zone,
            parameters=parameters,
            timestamps=timestamps,
            temperature=temperature,
            humidity=humidity,
            solar_irradiance=solar_irradiance,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            precipitation=precipitation,
            atmospheric_pressure=atmospheric_pressure,
            weather_conditions=weather_conditions,
            data_quality=data_quality
        )
        
        # 更新统计
        simulation_time = time.time() - simulation_start_time
        self._update_simulation_stats(climate_zone, len(timestamps), simulation_time)
        
        print(f"✅ 天气仿真完成: {data_id}")
        print(f"   气候区域: {climate_zone.value}, 数据点: {len(timestamps)}, 用时: {simulation_time:.2f}s")
        
        return weather_data
    
    def simulate_batch_weather(self,
                             simulation_configs: List[Dict[str, Any]],
                             batch_id: Optional[str] = None) -> List[WeatherData]:
        """
        批量天气仿真
        
        Args:
            simulation_configs: 仿真配置列表
            batch_id: 批次ID
            
        Returns:
            生成的天气数据列表
        """
        if batch_id is None:
            batch_id = f"weather_batch_{int(time.time()*1000)}"
        
        batch_start_time = time.time()
        weather_data_list = []
        
        print(f"🚀 开始批量天气仿真: {len(simulation_configs)} 个配置")
        
        for i, config in enumerate(simulation_configs):
            try:
                climate_zone = ClimateZone(config['climate_zone'])
                duration = config.get('duration_hours', 24.0 * 365)
                resolution = config.get('time_resolution_minutes', 60.0)
                start_day = config.get('start_day_of_year', 1)
                
                # 构建参数
                parameters = WeatherParameters()
                if 'parameters' in config:
                    param_dict = config['parameters']
                    for key, value in param_dict.items():
                        if hasattr(parameters, key):
                            setattr(parameters, key, value)
                
                data_id = config.get('id', f"{batch_id}_weather_{i+1}")
                
                weather_data = self.simulate_weather(
                    climate_zone, duration, resolution, parameters, start_day, data_id
                )
                weather_data_list.append(weather_data)
                
                if (i + 1) % 5 == 0:
                    print(f"   进度: {i+1}/{len(simulation_configs)}")
                
            except Exception as e:
                print(f"⚠️ 天气仿真 {i+1} 失败: {str(e)}")
        
        batch_time = time.time() - batch_start_time
        print(f"✅ 批量仿真完成: {len(weather_data_list)}/{len(simulation_configs)} 个数据集, 用时: {batch_time:.2f}s")
        
        return weather_data_list
    
    def _generate_timestamps(self, duration_hours: float, resolution_minutes: float) -> np.ndarray:
        """生成时间戳"""
        resolution_hours = resolution_minutes / 60.0
        num_points = int(duration_hours / resolution_hours)
        timestamps = np.linspace(0, duration_hours, num_points)
        return timestamps
    
    def _simulate_temperature(self,
                            timestamps: np.ndarray,
                            days_from_start: np.ndarray,
                            parameters: WeatherParameters) -> np.ndarray:
        """模拟温度"""
        num_points = len(timestamps)
        
        # 年周期（季节性变化）
        seasonal_phase = 2 * np.pi * (days_from_start - parameters.seasonal_lag) / 365.25
        annual_cycle = parameters.annual_avg_temp + parameters.temp_amplitude * np.sin(seasonal_phase)
        
        # 日周期
        hours = timestamps % 24
        daily_phase = 2 * np.pi * (hours - 6) / 24  # 最低温在早上6点
        daily_cycle = parameters.daily_temp_range * np.sin(daily_phase) / 2
        
        # 基础温度
        base_temperature = annual_cycle + daily_cycle
        
        # 添加随机变化
        temp_noise = np.random.normal(0, parameters.climate_variability * parameters.temp_amplitude, num_points)
        
        # 自相关噪声（天气的连续性）
        corr_factor = 0.9
        corr_noise = np.zeros(num_points)
        corr_noise[0] = temp_noise[0]
        for i in range(1, num_points):
            corr_noise[i] = corr_factor * corr_noise[i-1] + np.sqrt(1 - corr_factor**2) * temp_noise[i]
        
        # 最终温度
        temperature = base_temperature + corr_noise
        
        return temperature
    
    def _simulate_humidity(self,
                         timestamps: np.ndarray,
                         days_from_start: np.ndarray,
                         parameters: WeatherParameters,
                         temperature: np.ndarray) -> np.ndarray:
        """模拟湿度"""
        num_points = len(timestamps)
        
        # 年周期
        seasonal_phase = 2 * np.pi * days_from_start / 365.25
        annual_cycle = (parameters.annual_avg_humidity + 
                       parameters.humidity_amplitude * np.sin(seasonal_phase + np.pi))  # 与温度反相
        
        # 日周期（通常与温度反相关）
        hours = timestamps % 24
        daily_phase = 2 * np.pi * (hours - 18) / 24  # 最高湿度在傍晚
        daily_cycle = parameters.daily_humidity_range * np.sin(daily_phase) / 2
        
        # 温度相关性（负相关）
        temp_normalized = (temperature - np.mean(temperature)) / (np.std(temperature) + 1e-6)
        temp_effect = -10 * temp_normalized  # 温度每升高1标准差，湿度降低10%
        
        # 基础湿度
        base_humidity = annual_cycle + daily_cycle + temp_effect
        
        # 添加随机变化
        humidity_noise = np.random.normal(0, parameters.climate_variability * parameters.humidity_amplitude, num_points)
        
        # 最终湿度（限制在合理范围）
        humidity = base_humidity + humidity_noise
        humidity = np.clip(humidity, 10, 100)
        
        return humidity
    
    def _simulate_solar_irradiance(self,
                                 timestamps: np.ndarray,
                                 days_from_start: np.ndarray,
                                 parameters: WeatherParameters) -> np.ndarray:
        """模拟太阳辐射"""
        num_points = len(timestamps)
        hours = timestamps % 24
        
        # 年周期（太阳高度角变化）
        seasonal_phase = 2 * np.pi * days_from_start / 365.25
        seasonal_factor = 0.7 + 0.3 * np.sin(seasonal_phase - np.pi/2)  # 夏季最高
        
        # 日周期（太阳高度角）
        # 太阳升起时间和落下时间（简化模型）
        sunrise = 6.0
        sunset = 18.0
        solar_hours = sunset - sunrise
        
        # 只在白天有太阳辐射
        daylight_mask = (hours >= sunrise) & (hours <= sunset)
        solar_angle = np.zeros(num_points)
        
        # 计算太阳高度角（简化）
        daylight_hours = hours[daylight_mask]
        solar_noon = (sunrise + sunset) / 2
        angle_factor = np.sin(np.pi * (daylight_hours - sunrise) / solar_hours)
        solar_angle[daylight_mask] = angle_factor
        
        # 基础太阳辐射
        base_irradiance = (parameters.max_solar_irradiance * seasonal_factor * 
                          solar_angle * daylight_mask.astype(float))
        
        # 添加云层影响（随机衰减）
        cloud_factor = 1.0 - parameters.solar_variation * np.random.beta(2, 5, num_points)
        cloud_factor = np.clip(cloud_factor, 0.1, 1.0)
        
        # 最终太阳辐射
        solar_irradiance = base_irradiance * cloud_factor
        solar_irradiance = np.maximum(solar_irradiance, 0)
        
        return solar_irradiance
    
    def _simulate_wind(self,
                      timestamps: np.ndarray,
                      parameters: WeatherParameters) -> Tuple[np.ndarray, np.ndarray]:
        """模拟风速和风向"""
        num_points = len(timestamps)
        
        # 风速模拟（威布尔分布的时间序列）
        # 基础风速模式
        hours = timestamps % 24
        daily_wind_pattern = 1.0 + 0.3 * np.sin(2 * np.pi * (hours - 14) / 24)  # 下午风速较高
        
        # 随机风速变化（自相关）
        wind_noise = np.random.normal(0, parameters.wind_variability, num_points)
        corr_factor = 0.7
        corr_wind_noise = np.zeros(num_points)
        corr_wind_noise[0] = wind_noise[0]
        for i in range(1, num_points):
            corr_wind_noise[i] = corr_factor * corr_wind_noise[i-1] + np.sqrt(1 - corr_factor**2) * wind_noise[i]
        
        # 风速（确保非负）
        wind_speed = parameters.avg_wind_speed * daily_wind_pattern + corr_wind_noise
        wind_speed = np.maximum(wind_speed, 0.1)
        
        # 风向模拟（主导风向+随机变化）
        dominant_direction = 225  # 西南风（度）
        direction_variation = 45   # 变化范围
        
        # 风向随机游走
        direction_changes = np.random.normal(0, direction_variation/10, num_points)
        wind_direction = np.zeros(num_points)
        wind_direction[0] = dominant_direction
        
        for i in range(1, num_points):
            wind_direction[i] = wind_direction[i-1] + direction_changes[i]
            # 保持在0-360度范围内
            wind_direction[i] = wind_direction[i] % 360
        
        return wind_speed, wind_direction
    
    def _simulate_precipitation(self,
                              timestamps: np.ndarray,
                              days_from_start: np.ndarray,
                              parameters: WeatherParameters) -> np.ndarray:
        """模拟降水"""
        num_points = len(timestamps)
        
        # 降水概率模型
        # 年周期（雨季/旱季）
        seasonal_phase = 2 * np.pi * days_from_start / 365.25
        seasonal_rain_prob = 0.1 + 0.05 * np.sin(seasonal_phase + np.pi/2)  # 夏季多雨
        
        # 日周期（下午雷阵雨模式）
        hours = timestamps % 24
        daily_rain_prob = 1.0 + 0.5 * np.exp(-0.5 * ((hours - 15) / 3) ** 2)  # 下午3点最高
        
        # 综合降水概率
        rain_probability = seasonal_rain_prob * daily_rain_prob
        rain_probability *= (parameters.rainy_days_per_year / 365.25 / 24)  # 调整到小时概率
        
        # 生成降水事件
        precipitation = np.zeros(num_points)
        is_raining = np.random.random(num_points) < rain_probability
        
        # 降水强度（指数分布）
        rain_intensity = np.random.exponential(2.0, num_points)  # mm/h
        precipitation[is_raining] = rain_intensity[is_raining]
        
        # 连续性处理（雨通常持续一段时间）
        for i in range(1, num_points):
            if precipitation[i-1] > 0 and np.random.random() < 0.7:  # 70%概率持续
                if precipitation[i] == 0:
                    precipitation[i] = precipitation[i-1] * np.random.uniform(0.3, 0.9)
        
        return precipitation
    
    def _simulate_pressure(self,
                         timestamps: np.ndarray,
                         parameters: WeatherParameters) -> np.ndarray:
        """模拟大气压力"""
        num_points = len(timestamps)
        
        # 基础大气压
        base_pressure = np.full(num_points, parameters.avg_pressure)
        
        # 低频变化（天气系统）
        low_freq_period = 72  # 3天周期
        low_freq_phase = 2 * np.pi * timestamps / low_freq_period
        low_freq_variation = parameters.pressure_variation * 0.5 * np.sin(low_freq_phase)
        
        # 高频变化（日变化）
        hours = timestamps % 24
        daily_phase = 2 * np.pi * hours / 24
        daily_variation = 2.0 * np.sin(2 * daily_phase)  # 半日波
        
        # 随机变化
        pressure_noise = np.random.normal(0, parameters.pressure_variation * 0.1, num_points)
        
        # 自相关处理
        corr_factor = 0.95
        corr_pressure_noise = np.zeros(num_points)
        corr_pressure_noise[0] = pressure_noise[0]
        for i in range(1, num_points):
            corr_pressure_noise[i] = (corr_factor * corr_pressure_noise[i-1] + 
                                    np.sqrt(1 - corr_factor**2) * pressure_noise[i])
        
        # 最终大气压
        atmospheric_pressure = base_pressure + low_freq_variation + daily_variation + corr_pressure_noise
        
        return atmospheric_pressure
    
    def _simulate_weather_conditions(self,
                                   timestamps: np.ndarray,
                                   temperature: np.ndarray,
                                   humidity: np.ndarray,
                                   precipitation: np.ndarray,
                                   solar_irradiance: np.ndarray) -> List[WeatherCondition]:
        """模拟天气条件序列"""
        num_points = len(timestamps)
        weather_conditions = []
        
        for i in range(num_points):
            # 基于气象要素确定天气条件
            temp = temperature[i]
            humid = humidity[i]
            precip = precipitation[i]
            solar = solar_irradiance[i]
            hour = timestamps[i] % 24
            
            # 降水判断
            if precip > 10:
                if temp < 0:
                    condition = WeatherCondition.SNOW
                elif precip > 20:
                    condition = WeatherCondition.HEAVY_RAIN
                elif precip > 5:
                    condition = WeatherCondition.MODERATE_RAIN
                else:
                    condition = WeatherCondition.LIGHT_RAIN
            
            # 雾判断（高湿度 + 低温差）
            elif humid > 95 and 6 <= hour <= 10:
                condition = WeatherCondition.FOG
            
            # 云量判断（基于太阳辐射）
            else:
                # 计算理论太阳辐射
                if 6 <= hour <= 18:  # 白天
                    max_possible = 800  # 简化的最大可能辐射
                    cloud_cover = 1.0 - (solar / max_possible) if max_possible > 0 else 1.0
                    cloud_cover = np.clip(cloud_cover, 0, 1)
                    
                    if cloud_cover < 0.2:
                        condition = WeatherCondition.CLEAR
                    elif cloud_cover < 0.5:
                        condition = WeatherCondition.PARTLY_CLOUDY
                    elif cloud_cover < 0.8:
                        condition = WeatherCondition.CLOUDY
                    else:
                        condition = WeatherCondition.OVERCAST
                else:
                    # 夜间基于湿度判断
                    if humid < 70:
                        condition = WeatherCondition.CLEAR
                    elif humid < 85:
                        condition = WeatherCondition.PARTLY_CLOUDY
                    else:
                        condition = WeatherCondition.CLOUDY
            
            weather_conditions.append(condition)
        
        return weather_conditions
    
    def _apply_weather_interactions(self,
                                  temperature: np.ndarray,
                                  humidity: np.ndarray,
                                  solar_irradiance: np.ndarray,
                                  precipitation: np.ndarray,
                                  weather_conditions: List[WeatherCondition]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """应用天气要素间的相互影响"""
        # 降水对温度的影响（降温）
        rain_mask = precipitation > 0
        temperature[rain_mask] -= precipitation[rain_mask] * 0.5  # 降水降温效应
        
        # 降水对湿度的影响
        humidity[rain_mask] = np.minimum(humidity[rain_mask] + precipitation[rain_mask] * 2, 100)
        
        # 云层对太阳辐射的影响
        for i, condition in enumerate(weather_conditions):
            if condition in [WeatherCondition.CLOUDY, WeatherCondition.OVERCAST]:
                solar_irradiance[i] *= 0.3  # 云层遮挡
            elif condition == WeatherCondition.PARTLY_CLOUDY:
                solar_irradiance[i] *= 0.7
            elif condition in [WeatherCondition.LIGHT_RAIN, WeatherCondition.MODERATE_RAIN, WeatherCondition.HEAVY_RAIN]:
                solar_irradiance[i] *= 0.1  # 雨天遮挡严重
        
        return temperature, humidity, solar_irradiance
    
    def _assess_weather_quality(self, weather_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """评估天气数据质量"""
        quality_metrics = {}
        
        for name, data in weather_dict.items():
            # 连续性检查
            if len(data) > 1:
                continuity_score = 1.0 - np.mean(np.abs(np.diff(data)) > 3 * np.std(data))
            else:
                continuity_score = 1.0
            
            # 范围合理性检查
            if name == 'temperature':
                range_score = float(np.all((data >= -50) & (data <= 60)))
            elif name == 'humidity':
                range_score = float(np.all((data >= 0) & (data <= 100)))
            elif name == 'solar_irradiance':
                range_score = float(np.all((data >= 0) & (data <= 1500)))
            elif name == 'wind_speed':
                range_score = float(np.all((data >= 0) & (data <= 50)))
            elif name == 'precipitation':
                range_score = float(np.all((data >= 0) & (data <= 100)))
            else:
                range_score = 1.0
            
            # 物理一致性检查
            consistency_score = 1.0
            if name == 'temperature' and 'humidity' in weather_dict:
                # 温湿度负相关检查
                temp_humid_corr = np.corrcoef(data, weather_dict['humidity'])[0, 1]
                consistency_score = max(0, 1.0 + temp_humid_corr)  # 期望负相关
            
            # 综合质量分数
            overall_score = (continuity_score + range_score + consistency_score) / 3
            
            quality_metrics[f'{name}_quality'] = overall_score
            quality_metrics[f'{name}_continuity'] = continuity_score
            quality_metrics[f'{name}_range'] = range_score
            quality_metrics[f'{name}_consistency'] = consistency_score
        
        return quality_metrics
    
    def _build_weather_transition_matrix(self) -> Dict[WeatherCondition, Dict[WeatherCondition, float]]:
        """构建天气转换概率矩阵"""
        # 简化的马尔可夫转换矩阵
        transitions = {}
        
        # 晴天转换概率
        transitions[WeatherCondition.CLEAR] = {
            WeatherCondition.CLEAR: 0.7,
            WeatherCondition.PARTLY_CLOUDY: 0.2,
            WeatherCondition.CLOUDY: 0.08,
            WeatherCondition.LIGHT_RAIN: 0.02
        }
        
        # 多云转换概率
        transitions[WeatherCondition.PARTLY_CLOUDY] = {
            WeatherCondition.CLEAR: 0.3,
            WeatherCondition.PARTLY_CLOUDY: 0.4,
            WeatherCondition.CLOUDY: 0.2,
            WeatherCondition.LIGHT_RAIN: 0.1
        }
        
        # 阴天转换概率
        transitions[WeatherCondition.CLOUDY] = {
            WeatherCondition.PARTLY_CLOUDY: 0.2,
            WeatherCondition.CLOUDY: 0.4,
            WeatherCondition.OVERCAST: 0.2,
            WeatherCondition.LIGHT_RAIN: 0.15,
            WeatherCondition.MODERATE_RAIN: 0.05
        }
        
        # 其他天气条件的转换概率...
        # 为简化，这里只展示部分
        
        return transitions
    
    def _update_simulation_stats(self, climate_zone: ClimateZone, data_points: int, simulation_time: float):
        """更新仿真统计"""
        self.simulation_stats['total_simulations'] += 1
        self.simulation_stats['simulations_by_zone'][climate_zone] += 1
        self.simulation_stats['total_data_points'] += data_points
        self.simulation_stats['simulation_time'] += simulation_time
    
    def _get_tropical_template(self) -> Dict[str, Any]:
        """获取热带气候模板"""
        return {
            'annual_avg_temp': 26.0,
            'temp_amplitude': 3.0,
            'daily_temp_range': 6.0,
            'annual_avg_humidity': 80.0,
            'humidity_amplitude': 10.0,
            'annual_precipitation': 2000.0,
            'rainy_days_per_year': 200,
            'max_solar_irradiance': 1200.0,
            'avg_wind_speed': 2.5
        }
    
    def _get_subtropical_template(self) -> Dict[str, Any]:
        """获取亚热带气候模板"""
        return {
            'annual_avg_temp': 20.0,
            'temp_amplitude': 8.0,
            'daily_temp_range': 10.0,
            'annual_avg_humidity': 70.0,
            'humidity_amplitude': 15.0,
            'annual_precipitation': 1200.0,
            'rainy_days_per_year': 150,
            'max_solar_irradiance': 1000.0,
            'avg_wind_speed': 3.0
        }
    
    def _get_temperate_template(self) -> Dict[str, Any]:
        """获取温带气候模板"""
        return {
            'annual_avg_temp': 12.0,
            'temp_amplitude': 15.0,
            'daily_temp_range': 12.0,
            'annual_avg_humidity': 65.0,
            'humidity_amplitude': 20.0,
            'annual_precipitation': 800.0,
            'rainy_days_per_year': 120,
            'max_solar_irradiance': 900.0,
            'avg_wind_speed': 4.0
        }
    
    def _get_continental_template(self) -> Dict[str, Any]:
        """获取大陆性气候模板"""
        return {
            'annual_avg_temp': 8.0,
            'temp_amplitude': 20.0,
            'daily_temp_range': 15.0,
            'annual_avg_humidity': 55.0,
            'humidity_amplitude': 25.0,
            'annual_precipitation': 600.0,
            'rainy_days_per_year': 100,
            'max_solar_irradiance': 950.0,
            'avg_wind_speed': 5.0
        }
    
    def _get_polar_template(self) -> Dict[str, Any]:
        """获取极地气候模板"""
        return {
            'annual_avg_temp': -15.0,
            'temp_amplitude': 25.0,
            'daily_temp_range': 8.0,
            'annual_avg_humidity': 75.0,
            'humidity_amplitude': 15.0,
            'annual_precipitation': 200.0,
            'rainy_days_per_year': 50,
            'max_solar_irradiance': 600.0,
            'avg_wind_speed': 6.0
        }
    
    def _get_desert_template(self) -> Dict[str, Any]:
        """获取沙漠气候模板"""
        return {
            'annual_avg_temp': 25.0,
            'temp_amplitude': 12.0,
            'daily_temp_range': 20.0,
            'annual_avg_humidity': 25.0,
            'humidity_amplitude': 10.0,
            'annual_precipitation': 100.0,
            'rainy_days_per_year': 20,
            'max_solar_irradiance': 1300.0,
            'avg_wind_speed': 4.5
        }
    
    def _get_mediterranean_template(self) -> Dict[str, Any]:
        """获取地中海气候模板"""
        return {
            'annual_avg_temp': 18.0,
            'temp_amplitude': 10.0,
            'daily_temp_range': 12.0,
            'annual_avg_humidity': 60.0,
            'humidity_amplitude': 18.0,
            'annual_precipitation': 650.0,
            'rainy_days_per_year': 80,
            'max_solar_irradiance': 1100.0,
            'avg_wind_speed': 3.5
        }
    
    def _get_oceanic_template(self) -> Dict[str, Any]:
        """获取海洋性气候模板"""
        return {
            'annual_avg_temp': 15.0,
            'temp_amplitude': 8.0,
            'daily_temp_range': 8.0,
            'annual_avg_humidity': 75.0,
            'humidity_amplitude': 12.0,
            'annual_precipitation': 1000.0,
            'rainy_days_per_year': 180,
            'max_solar_irradiance': 800.0,
            'avg_wind_speed': 5.5
        }
    
    def analyze_weather_impact(self, weather_data: WeatherData) -> Dict[str, Any]:
        """分析天气对储能系统的影响"""
        analysis = {
            'thermal_impact': self._analyze_thermal_impact(weather_data),
            'performance_impact': self._analyze_performance_impact(weather_data),
            'cooling_demand': self._analyze_cooling_demand(weather_data),
            'extreme_conditions': self._analyze_extreme_conditions(weather_data)
        }
        
        return analysis
    
    def _analyze_thermal_impact(self, weather_data: WeatherData) -> Dict[str, Any]:
        """分析热影响"""
        temp = weather_data.temperature
        humid = weather_data.humidity
        wind = weather_data.wind_speed
        
        # 热应力指数
        heat_stress_hours = np.sum(weather_data.heat_index > 35)
        cold_stress_hours = np.sum(weather_data.wind_chill < -10)
        
        # 自然冷却潜力
        natural_cooling_potential = np.mean(wind * np.maximum(0, 25 - temp))
        
        thermal_impact = {
            'avg_temperature': np.mean(temp),
            'temp_range': np.max(temp) - np.min(temp),
            'heat_stress_hours': heat_stress_hours,
            'cold_stress_hours': cold_stress_hours,
            'high_humidity_hours': np.sum(humid > 80),
            'natural_cooling_potential': natural_cooling_potential,
            'thermal_cycling_stress': np.sum(np.abs(np.diff(temp)) > 10)
        }
        
        return thermal_impact
    
    def _analyze_performance_impact(self, weather_data: WeatherData) -> Dict[str, Any]:
        """分析性能影响"""
        temp = weather_data.temperature
        
        # 温度对电池性能的影响（简化模型）
        optimal_temp = 25.0
        temp_deviation = np.abs(temp - optimal_temp)
        
        # 容量影响（高温和低温都会降低容量）
        capacity_factor = np.where(temp > optimal_temp,
                                 1.0 - (temp - optimal_temp) * 0.005,  # 高温每度损失0.5%
                                 1.0 - (optimal_temp - temp) * 0.008)  # 低温每度损失0.8%
        capacity_factor = np.clip(capacity_factor, 0.6, 1.0)
        
        # 效率影响
        efficiency_factor = 1.0 - temp_deviation * 0.002  # 偏离最佳温度每度损失0.2%
        efficiency_factor = np.clip(efficiency_factor, 0.8, 1.0)
        
        performance_impact = {
            'avg_capacity_factor': np.mean(capacity_factor),
            'min_capacity_factor': np.min(capacity_factor),
            'avg_efficiency_factor': np.mean(efficiency_factor),
            'min_efficiency_factor': np.min(efficiency_factor),
            'optimal_temp_hours': np.sum(np.abs(temp - optimal_temp) < 5),
            'severe_temp_hours': np.sum(temp_deviation > 15)
        }
        
        return performance_impact
    
    def _analyze_cooling_demand(self, weather_data: WeatherData) -> Dict[str, Any]:
        """分析冷却需求"""
        temp = weather_data.temperature
        humid = weather_data.humidity
        solar = weather_data.solar_irradiance
        
        # 冷却负荷计算（简化）
        ambient_heat_load = np.maximum(0, temp - 25) * 100  # W per degree above 25°C
        solar_heat_load = solar * 0.1  # 10% solar heat gain
        total_heat_load = ambient_heat_load + solar_heat_load
        
        # 自然对流冷却潜力
        natural_cooling = weather_data.wind_speed * np.maximum(0, temp - 20) * 50
        
        # 净冷却需求
        net_cooling_demand = np.maximum(0, total_heat_load - natural_cooling)
        
        cooling_demand = {
            'avg_cooling_demand': np.mean(net_cooling_demand),
            'max_cooling_demand': np.max(net_cooling_demand),
            'total_cooling_energy': np.sum(net_cooling_demand) * len(weather_data.timestamps) / len(weather_data.timestamps),  # 简化积分
            'high_demand_hours': np.sum(net_cooling_demand > 1000),
            'natural_cooling_hours': np.sum(natural_cooling > ambient_heat_load)
        }
        
        return cooling_demand
    
    def _analyze_extreme_conditions(self, weather_data: WeatherData) -> Dict[str, Any]:
        """分析极端条件"""
        temp = weather_data.temperature
        wind = weather_data.wind_speed
        precip = weather_data.precipitation
        
        extreme_conditions = {
            'extreme_heat_events': np.sum(temp > 40),
            'extreme_cold_events': np.sum(temp < -10),
            'high_wind_events': np.sum(wind > 15),
            'heavy_rain_events': np.sum(precip > 20),
            'storm_conditions': len([c for c in weather_data.weather_conditions if c == WeatherCondition.STORM]),
            'consecutive_extreme_days': self._count_consecutive_extremes(temp)
        }
        
        return extreme_conditions
    
    def _count_consecutive_extremes(self, temperature: np.ndarray, threshold: float = 35.0) -> int:
        """计算连续极端天数"""
        extreme_mask = temperature > threshold
        max_consecutive = 0
        current_consecutive = 0
        
        for is_extreme in extreme_mask:
            if is_extreme:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def export_weather_data(self, weather_data: WeatherData, file_path: str, format: str = 'csv'):
        """导出天气数据"""
        try:
            if format.lower() == 'csv':
                df = pd.DataFrame({
                    'timestamp': weather_data.timestamps,
                    'temperature': weather_data.temperature,
                    'humidity': weather_data.humidity,
                    'solar_irradiance': weather_data.solar_irradiance,
                    'wind_speed': weather_data.wind_speed,
                    'wind_direction': weather_data.wind_direction,
                    'precipitation': weather_data.precipitation,
                    'atmospheric_pressure': weather_data.atmospheric_pressure,
                    'heat_index': weather_data.heat_index,
                    'wind_chill': weather_data.wind_chill,
                    'dew_point': weather_data.dew_point,
                    'weather_condition': [c.value for c in weather_data.weather_conditions]
                })
                df.to_csv(file_path, index=False)
                
            elif format.lower() == 'json':
                import json
                export_data = {
                    'data_id': weather_data.data_id,
                    'climate_zone': weather_data.climate_zone.value,
                    'timestamps': weather_data.timestamps.tolist(),
                    'temperature': weather_data.temperature.tolist(),
                    'humidity': weather_data.humidity.tolist(),
                    'solar_irradiance': weather_data.solar_irradiance.tolist(),
                    'wind_speed': weather_data.wind_speed.tolist(),
                    'wind_direction': weather_data.wind_direction.tolist(),
                    'precipitation': weather_data.precipitation.tolist(),
                    'atmospheric_pressure': weather_data.atmospheric_pressure.tolist(),
                    'weather_conditions': [c.value for c in weather_data.weather_conditions],
                    'data_quality': weather_data.data_quality,
                    'generation_time': weather_data.generation_time
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            print(f"✅ 天气数据已导出: {file_path}")
            
        except Exception as e:
            print(f"❌ 天气数据导出失败: {str(e)}")
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """获取仿真统计信息"""
        stats = self.simulation_stats.copy()
        
        if stats['total_simulations'] > 0:
            stats['avg_data_points_per_simulation'] = stats['total_data_points'] / stats['total_simulations']
            stats['avg_simulation_time_per_run'] = stats['simulation_time'] / stats['total_simulations']
        else:
            stats['avg_data_points_per_simulation'] = 0
            stats['avg_simulation_time_per_run'] = 0
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"WeatherSimulator({self.simulator_id}): "
                f"仿真次数={self.simulation_stats['total_simulations']}, "
                f"数据点={self.simulation_stats['total_data_points']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"WeatherSimulator(simulator_id='{self.simulator_id}', "
                f"climate_zones={len(self.climate_templates)}, "
                f"total_simulations={self.simulation_stats['total_simulations']})")
