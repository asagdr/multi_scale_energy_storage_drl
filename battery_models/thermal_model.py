import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig

class CoolingMode(Enum):
    """冷却模式枚举"""
    NATURAL = "natural"      # 自然冷却
    FORCED_AIR = "forced_air"  # 强制风冷
    LIQUID = "liquid"        # 液冷
    HYBRID = "hybrid"        # 混合冷却

class HeatSourceType(Enum):
    """热源类型枚举"""
    JOULE_HEATING = "joule"          # 焦耳热
    POLARIZATION_LOSS = "polarization"  # 极化损耗
    REACTION_HEAT = "reaction"       # 反应热
    EXTERNAL = "external"            # 外部热源

@dataclass
class ThermalState:
    """热状态数据结构"""
    # 温度状态
    core_temperature: float = 25.0      # ℃, 电池核心温度
    surface_temperature: float = 25.0   # ℃, 表面温度
    ambient_temperature: float = 25.0   # ℃, 环境温度
    
    # 热流状态
    heat_generation_rate: float = 0.0   # W, 产热功率
    heat_dissipation_rate: float = 0.0  # W, 散热功率
    net_heat_flow: float = 0.0          # W, 净热流
    
    # 温度梯度
    core_surface_gradient: float = 0.0  # ℃, 核心-表面温差
    surface_ambient_gradient: float = 0.0  # ℃, 表面-环境温差
    
    # 热容量和导热系数
    thermal_capacity: float = 1000.0    # J/K, 热容量
    thermal_conductivity: float = 2.0   # W/(m·K), 导热系数
    
    # 冷却状态
    cooling_power: float = 0.0          # W, 冷却功率
    cooling_efficiency: float = 1.0     # 冷却效率
    
    # 安全状态
    temperature_warning: bool = False    # 温度预警
    temperature_alarm: bool = False      # 温度报警
    thermal_runaway_risk: float = 0.0   # 热失控风险评估

@dataclass
class ThermalConstraints:
    """热约束数据结构"""
    # 电流约束 (基于温度)
    max_charge_current: float = 0.0     # A, 最大充电电流
    max_discharge_current: float = 0.0  # A, 最大放电电流
    
    # 功率约束 (基于温度)
    max_charge_power: float = 0.0       # W, 最大充电功率
    max_discharge_power: float = 0.0    # W, 最大放电功率
    
    # 温升约束
    max_temp_rise_rate: float = 2.0     # ℃/min, 最大温升速率
    max_temp_difference: float = 10.0   # ℃, 最大温差
    
    # 时间约束
    time_to_limit: float = float('inf') # s, 到达温度限制的时间
    cooling_time_required: float = 0.0  # s, 所需冷却时间

class ThermalModel:
    """
    电池热模型类
    实现完整的电池热行为建模，为DRL架构提供温度约束
    """
    
    def __init__(self, 
                 battery_params: BatteryParams,
                 system_config: Optional[SystemConfig] = None,
                 cooling_mode: CoolingMode = CoolingMode.FORCED_AIR,
                 cell_id: str = "ThermalCell_001"):
        """
        初始化热模型
        
        Args:
            battery_params: 电池参数
            system_config: 系统配置
            cooling_mode: 冷却模式
            cell_id: 电池ID
        """
        self.params = battery_params
        self.config = system_config
        self.cooling_mode = cooling_mode
        self.cell_id = cell_id
        
        # === 热模型参数 ===
        self._init_thermal_parameters()
        
        # === 初始化状态 ===
        self.state = ThermalState()
        self.state.ambient_temperature = self.params.NOMINAL_TEMP
        self.state.core_temperature = self.params.NOMINAL_TEMP
        self.state.surface_temperature = self.params.NOMINAL_TEMP
        
        # === 历史记录 ===
        self.temperature_history: List[Dict] = []
        self.constraint_history: List[ThermalConstraints] = []
        
        # === 仿真参数 ===
        self.time_step_count = 0
        self.total_time = 0.0
        
        # === 预警系统 ===
        self.warning_thresholds = {
            'high_temp': self.params.MAX_TEMP - 10.0,
            'temp_rise_rate': 5.0,  # ℃/min
            'temp_difference': 15.0  # ℃
        }
        
        print(f"✅ 热模型初始化完成: {cell_id} ({cooling_mode.value})")
    
    def _init_thermal_parameters(self):
        """初始化热模型参数"""
        # === 基础热物性参数 ===
        # 磷酸铁锂电池典型热参数
        self.thermal_params = {
            # 几何参数
            'cell_length': 0.174,      # m, 电池长度
            'cell_width': 0.121,       # m, 电池宽度  
            'cell_height': 0.0125,     # m, 电池厚度
            'cell_mass': 5.5,          # kg, 电池质量
            
            # 热物性参数
            'specific_heat': 900.0,    # J/(kg·K), 比热容
            'density': 2500.0,         # kg/m³, 密度
            'thermal_conductivity_x': 2.0,   # W/(m·K), x方向导热系数
            'thermal_conductivity_y': 2.0,   # W/(m·K), y方向导热系数
            'thermal_conductivity_z': 0.5,   # W/(m·K), z方向导热系数 (厚度方向较小)
            
            # 对流换热参数
            'convection_coeff_natural': 10.0,    # W/(m²·K), 自然对流换热系数
            'convection_coeff_forced': 50.0,     # W/(m²·K), 强制对流换热系数
            'convection_coeff_liquid': 500.0,    # W/(m²·K), 液冷换热系数
            
            # 热阻参数
            'contact_resistance': 0.001,    # K·m²/W, 接触热阻
            'packaging_resistance': 0.005   # K·m²/W, 封装热阻
        }
        
        # === 计算衍生参数 ===
        cell_volume = (self.thermal_params['cell_length'] * 
                      self.thermal_params['cell_width'] * 
                      self.thermal_params['cell_height'])
        
        self.thermal_params['cell_volume'] = cell_volume
        self.thermal_params['surface_area'] = 2 * (
            self.thermal_params['cell_length'] * self.thermal_params['cell_width'] +
            self.thermal_params['cell_length'] * self.thermal_params['cell_height'] +
            self.thermal_params['cell_width'] * self.thermal_params['cell_height']
        )
        
        # 热容量 (J/K)
        self.state.thermal_capacity = (self.thermal_params['cell_mass'] * 
                                     self.thermal_params['specific_heat'])
        
        # 根据冷却模式设置换热系数
        if self.cooling_mode == CoolingMode.NATURAL:
            self.convection_coefficient = self.thermal_params['convection_coeff_natural']
        elif self.cooling_mode == CoolingMode.FORCED_AIR:
            self.convection_coefficient = self.thermal_params['convection_coeff_forced']
        elif self.cooling_mode == CoolingMode.LIQUID:
            self.convection_coefficient = self.thermal_params['convection_coeff_liquid']
        else:  # HYBRID
            self.convection_coefficient = self.thermal_params['convection_coeff_forced']
    
    def calculate_heat_generation(self, 
                                current: float, 
                                voltage: float, 
                                soc: float,
                                internal_resistance: float) -> Dict[str, float]:
        """
        计算电池产热功率
        
        Args:
            current: 电流 (A)
            voltage: 端电压 (V)
            soc: SOC (%)
            internal_resistance: 内阻 (Ω)
            
        Returns:
            各类热源功率字典 (W)
        """
        heat_sources = {}
        
        # 1. 焦耳热 (I²R损耗)
        joule_heat = current**2 * internal_resistance
        heat_sources[HeatSourceType.JOULE_HEATING.value] = joule_heat
        
        # 2. 极化损耗热
        # 简化模型：基于电流和SOC的极化损耗
        polarization_resistance = self._get_polarization_resistance(soc, self.state.core_temperature)
        polarization_heat = current**2 * polarization_resistance
        heat_sources[HeatSourceType.POLARIZATION_LOSS.value] = polarization_heat
        
        # 3. 反应热 (熵热)
        # 充放电反应的熵变产生的热
        reaction_heat = self._calculate_reaction_heat(current, soc, self.state.core_temperature)
        heat_sources[HeatSourceType.REACTION_HEAT.value] = reaction_heat
        
        # 4. 外部热源 (环境影响)
        external_heat = self._calculate_external_heat_input()
        heat_sources[HeatSourceType.EXTERNAL.value] = external_heat
        
        # 总产热功率
        total_heat = sum(heat_sources.values())
        heat_sources['total'] = total_heat
        
        return heat_sources
    
    def _get_polarization_resistance(self, soc: float, temperature: float) -> float:
        """获取极化电阻"""
        # 基础极化电阻
        base_resistance = 0.0005  # Ω
        
        # SOC影响
        if soc < 20:
            soc_factor = 1.0 + (20 - soc) * 0.05
        elif soc > 80:
            soc_factor = 1.0 + (soc - 80) * 0.02
        else:
            soc_factor = 1.0
        
        # 温度影响 (阿伦尼乌斯关系)
        temp_factor = math.exp(1000 * (1/(temperature + 273.15) - 1/298.15))
        
        return base_resistance * soc_factor * temp_factor
    
    def _calculate_reaction_heat(self, current: float, soc: float, temperature: float) -> float:
        """计算反应热"""
        # 熵系数 (V/K) - 磷酸铁锂电池典型值
        entropy_coefficient = -0.0003  # V/K
        
        # 温度相关的熵变
        delta_entropy = entropy_coefficient * (temperature + 273.15)
        
        # 反应热功率 = I * T * dS/dT
        reaction_heat = current * (temperature + 273.15) * delta_entropy
        
        return reaction_heat
    
    def _calculate_external_heat_input(self) -> float:
        """计算外部热输入"""
        # 简化模型：主要考虑环境温度影响
        temp_diff = self.state.ambient_temperature - self.state.surface_temperature
        
        # 自然对流热传递
        external_heat = (self.thermal_params['convection_coeff_natural'] * 
                        self.thermal_params['surface_area'] * temp_diff)
        
        return external_heat
    
    def calculate_heat_dissipation(self, surface_temp: float, ambient_temp: float) -> Dict[str, float]:
        """
        计算散热功率
        
        Args:
            surface_temp: 表面温度 (℃)
            ambient_temp: 环境温度 (℃)
            
        Returns:
            散热功率字典 (W)
        """
        heat_dissipation = {}
        
        temp_diff = surface_temp - ambient_temp
        surface_area = self.thermal_params['surface_area']
        
        # 1. 对流散热
        convection_heat = self.convection_coefficient * surface_area * temp_diff
        heat_dissipation['convection'] = max(0, convection_heat)
        
        # 2. 辐射散热
        # 斯特藩-玻尔兹曼定律简化
        emissivity = 0.85  # 发射率
        stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        
        T_surf = surface_temp + 273.15  # K
        T_amb = ambient_temp + 273.15   # K
        
        radiation_heat = (emissivity * stefan_boltzmann * surface_area * 
                         (T_surf**4 - T_amb**4))
        heat_dissipation['radiation'] = max(0, radiation_heat)
        
        # 3. 主动冷却 (如果启用)
        active_cooling = self._calculate_active_cooling(temp_diff)
        heat_dissipation['active_cooling'] = active_cooling
        
        # 总散热功率
        total_dissipation = sum(heat_dissipation.values())
        heat_dissipation['total'] = total_dissipation
        
        return heat_dissipation
    
    def _calculate_active_cooling(self, temp_diff: float) -> float:
        """计算主动冷却功率"""
        if self.cooling_mode == CoolingMode.NATURAL:
            return 0.0
        
        # 根据温差启动主动冷却
        if temp_diff > 5.0:  # 温差超过5℃启动冷却
            if self.cooling_mode == CoolingMode.FORCED_AIR:
                # 风冷功率
                fan_power_ratio = min(1.0, (temp_diff - 5.0) / 15.0)  # 5-20℃线性调节
                max_cooling_power = 50.0  # W
                return max_cooling_power * fan_power_ratio
            
            elif self.cooling_mode == CoolingMode.LIQUID:
                # 液冷功率
                pump_power_ratio = min(1.0, (temp_diff - 5.0) / 10.0)  # 5-15℃线性调节
                max_cooling_power = 200.0  # W
                return max_cooling_power * pump_power_ratio
            
            elif self.cooling_mode == CoolingMode.HYBRID:
                # 混合冷却
                air_cooling = self._calculate_active_cooling(temp_diff) * 0.3  # 30%风冷
                liquid_cooling = self._calculate_active_cooling(temp_diff) * 0.7  # 70%液冷
                return air_cooling + liquid_cooling
        
        return 0.0
    
    def update_temperature(self, 
                          heat_generation: float, 
                          delta_t: float,
                          current: float = 0.0,
                          ambient_temp: Optional[float] = None) -> Dict[str, float]:
        """
        更新温度状态 (集总参数热模型)
        
        Args:
            heat_generation: 产热功率 (W)
            delta_t: 时间步长 (s)
            current: 电流 (A)
            ambient_temp: 环境温度 (℃)
            
        Returns:
            温度变化信息
        """
        if ambient_temp is not None:
            self.state.ambient_temperature = ambient_temp
        
        # === 1. 计算散热 ===
        heat_dissipation_dict = self.calculate_heat_dissipation(
            self.state.surface_temperature, 
            self.state.ambient_temperature
        )
        total_heat_dissipation = heat_dissipation_dict['total']
        
        # === 2. 净热流 ===
        net_heat_flow = heat_generation - total_heat_dissipation
        
        # === 3. 核心温度更新 ===
        # 简化的双节点模型：核心节点和表面节点
        
        # 核心温度变化
        thermal_capacity = self.state.thermal_capacity
        core_temp_change = (heat_generation * delta_t) / thermal_capacity
        
        # 核心到表面的热传导
        thermal_conductance = (self.thermal_params['thermal_conductivity_x'] * 
                             self.thermal_params['surface_area'] / 
                             self.thermal_params['cell_height'])
        
        core_to_surface_heat = (thermal_conductance * 
                               (self.state.core_temperature - self.state.surface_temperature))
        
        surface_temp_change = ((core_to_surface_heat - total_heat_dissipation) * delta_t / 
                              (thermal_capacity * 0.3))  # 表面热容量较小
        
        # === 4. 更新温度 ===
        old_core_temp = self.state.core_temperature
        old_surface_temp = self.state.surface_temperature
        
        self.state.core_temperature += core_temp_change - (core_to_surface_heat * delta_t / thermal_capacity)
        self.state.surface_temperature += surface_temp_change
        
        # 温度限制
        self.state.core_temperature = max(self.state.ambient_temperature - 5, 
                                        min(self.params.MAX_TEMP + 20, self.state.core_temperature))
        self.state.surface_temperature = max(self.state.ambient_temperature - 2,
                                           min(self.params.MAX_TEMP + 10, self.state.surface_temperature))
        
        # === 5. 更新状态变量 ===
        self.state.heat_generation_rate = heat_generation
        self.state.heat_dissipation_rate = total_heat_dissipation
        self.state.net_heat_flow = net_heat_flow
        self.state.core_surface_gradient = self.state.core_temperature - self.state.surface_temperature
        self.state.surface_ambient_gradient = self.state.surface_temperature - self.state.ambient_temperature
        self.state.cooling_power = heat_dissipation_dict.get('active_cooling', 0.0)
        
        # === 6. 安全检查 ===
        self._update_safety_status()
        
        # === 7. 返回信息 ===
        temp_info = {
            'core_temp_change': self.state.core_temperature - old_core_temp,
            'surface_temp_change': self.state.surface_temperature - old_surface_temp,
            'heat_generation': heat_generation,
            'heat_dissipation': total_heat_dissipation,
            'net_heat_flow': net_heat_flow,
            'core_temperature': self.state.core_temperature,
            'surface_temperature': self.state.surface_temperature,
            'thermal_gradient': self.state.core_surface_gradient
        }
        
        return temp_info
    
    def _update_safety_status(self):
        """更新安全状态"""
        # 温度预警
        if (self.state.core_temperature > self.warning_thresholds['high_temp'] or
            self.state.surface_temperature > self.warning_thresholds['high_temp']):
            self.state.temperature_warning = True
        else:
            self.state.temperature_warning = False
        
        # 温度报警
        if (self.state.core_temperature > self.params.MAX_TEMP or
            self.state.surface_temperature > self.params.MAX_TEMP):
            self.state.temperature_alarm = True
        else:
            self.state.temperature_alarm = False
        
        # 热失控风险评估 (简化)
        if self.state.core_temperature > 80:  # ℃
            risk_factor = (self.state.core_temperature - 80) / 20  # 80-100℃线性增长
            self.state.thermal_runaway_risk = min(1.0, risk_factor)
        else:
            self.state.thermal_runaway_risk = 0.0
    
    def calculate_thermal_constraints(self, 
                                    base_current_limits: Tuple[float, float],
                                    base_power_limits: Tuple[float, float]) -> ThermalConstraints:
        """
        计算基于温度的约束矩阵 C_t
        为上层DRL提供约束边界
        
        Args:
            base_current_limits: 基础电流限制 (max_charge, max_discharge)
            base_power_limits: 基础功率限制 (max_charge, max_discharge)
            
        Returns:
            热约束对象
        """
        constraints = ThermalConstraints()
        
        # === 1. 温度降额因子 ===
        temp_derating_factor = self._calculate_temperature_derating_factor()
        
        # === 2. 电流约束 ===
        max_charge_current_base, max_discharge_current_base = base_current_limits
        constraints.max_charge_current = max_charge_current_base * temp_derating_factor
        constraints.max_discharge_current = max_discharge_current_base * temp_derating_factor
        
        # === 3. 功率约束 ===
        max_charge_power_base, max_discharge_power_base = base_power_limits
        constraints.max_charge_power = max_charge_power_base * temp_derating_factor
        constraints.max_discharge_power = max_discharge_power_base * temp_derating_factor
        
        # === 4. 温升约束 ===
        constraints.max_temp_rise_rate = self._calculate_max_temp_rise_rate()
        constraints.max_temp_difference = self.warning_thresholds['temp_difference']
        
        # === 5. 时间约束 ===
        constraints.time_to_limit = self._calculate_time_to_temperature_limit()
        constraints.cooling_time_required = self._calculate_cooling_time_required()
        
        # 记录约束历史
        self.constraint_history.append(constraints)
        
        return constraints
    
    def _calculate_temperature_derating_factor(self) -> float:
        """计算温度降额因子"""
        max_temp = max(self.state.core_temperature, self.state.surface_temperature)
        
        # 温度阈值
        optimal_temp_max = self.params.OPTIMAL_TEMP_RANGE[1]  # 35℃
        warning_temp = self.warning_thresholds['high_temp']   # 50℃
        max_operating_temp = self.params.MAX_TEMP             # 60℃
        
        if max_temp <= optimal_temp_max:
            # 最佳温度范围，无降额
            return 1.0
        elif max_temp <= warning_temp:
            # 线性降额区间 35-50℃
            derating = 1.0 - 0.2 * (max_temp - optimal_temp_max) / (warning_temp - optimal_temp_max)
            return max(0.8, derating)
        elif max_temp <= max_operating_temp:
            # 严重降额区间 50-60℃
            derating = 0.8 - 0.6 * (max_temp - warning_temp) / (max_operating_temp - warning_temp)
            return max(0.2, derating)
        else:
            # 超过最大工作温度，严重限制
            return 0.1
    
    def _calculate_max_temp_rise_rate(self) -> float:
        """计算最大允许温升速率"""
        current_temp = max(self.state.core_temperature, self.state.surface_temperature)
        temp_margin = self.params.MAX_TEMP - current_temp
        
        if temp_margin > 20:
            return 5.0  # ℃/min
        elif temp_margin > 10:
            return 3.0  # ℃/min
        elif temp_margin > 5:
            return 1.0  # ℃/min
        else:
            return 0.5  # ℃/min
    
    def _calculate_time_to_temperature_limit(self) -> float:
        """计算到达温度限制的时间"""
        current_temp = max(self.state.core_temperature, self.state.surface_temperature)
        temp_margin = self.params.MAX_TEMP - current_temp
        
        if len(self.temperature_history) < 2:
            return float('inf')
        
        # 计算温升速率
        recent_temps = [record['core_temperature'] for record in self.temperature_history[-10:]]
        if len(recent_temps) >= 2:
            temp_rise_rate = (recent_temps[-1] - recent_temps[0]) / (len(recent_temps) - 1)  # ℃/step
            
            if temp_rise_rate > 0:
                time_steps_to_limit = temp_margin / temp_rise_rate
                return time_steps_to_limit * (self.config.SIMULATION_TIME_STEP if self.config else 1.0)
        
        return float('inf')
    
    def _calculate_cooling_time_required(self) -> float:
        """计算所需冷却时间"""
        current_temp = max(self.state.core_temperature, self.state.surface_temperature)
        target_temp = self.params.OPTIMAL_TEMP_RANGE[1]  # 35℃
        
        if current_temp <= target_temp:
            return 0.0
        
        temp_diff = current_temp - target_temp
        
        # 估算冷却时间 (基于散热能力)
        if self.cooling_mode == CoolingMode.LIQUID:
            cooling_rate = 0.5  # ℃/min
        elif self.cooling_mode == CoolingMode.FORCED_AIR:
            cooling_rate = 0.2  # ℃/min
        else:
            cooling_rate = 0.1  # ℃/min
        
        cooling_time = temp_diff / cooling_rate * 60  # 转换为秒
        return cooling_time
    
    def get_constraint_matrix_for_drl(self) -> np.ndarray:
        """
        为DRL上层提供约束矩阵 C_t
        格式化为标准矩阵形式
        
        Returns:
            约束矩阵 (n×m)
        """
        if not self.constraint_history:
            # 如果没有约束历史，使用当前状态计算
            constraints = self.calculate_thermal_constraints(
                (self.params.max_charge_current, self.params.max_discharge_current),
                (self.params.max_charge_power, self.params.max_discharge_power)
            )
        else:
            constraints = self.constraint_history[-1]
        
        # 构造约束矩阵 C_t
        # 行：约束类型，列：电池单体/电池组
        constraint_matrix = np.array([
            [constraints.max_charge_current],      # 最大充电电流约束
            [constraints.max_discharge_current],   # 最大放电电流约束
            [constraints.max_charge_power],        # 最大充电功率约束
            [constraints.max_discharge_power],     # 最大放电功率约束
            [constraints.max_temp_rise_rate],      # 最大温升速率约束
            [constraints.max_temp_difference]      # 最大温差约束
        ])
        
        return constraint_matrix
    
    def get_temperature_compensation_data(self) -> Dict[str, float]:
        """
        为下层温度补偿器提供数据
        
        Returns:
            温度补偿数据字典
        """
        return {
            'core_temperature': self.state.core_temperature,
            'surface_temperature': self.state.surface_temperature,
            'ambient_temperature': self.state.ambient_temperature,
            'thermal_gradient': self.state.core_surface_gradient,
            'temperature_derating_factor': self._calculate_temperature_derating_factor(),
            'cooling_efficiency': self.state.cooling_efficiency,
            'thermal_time_constant': self._calculate_thermal_time_constant(),
            'temperature_prediction': self._predict_future_temperature()
        }
    
    def _calculate_thermal_time_constant(self) -> float:
        """计算热时间常数"""
        thermal_resistance = 1.0 / (self.convection_coefficient * self.thermal_params['surface_area'])
        thermal_time_constant = self.state.thermal_capacity * thermal_resistance
        return thermal_time_constant
    
    def _predict_future_temperature(self, prediction_time: float = 60.0) -> float:
        """预测未来温度"""
        if not self.temperature_history:
            return self.state.core_temperature
        
        # 简化的线性预测
        recent_temps = [record['core_temperature'] for record in self.temperature_history[-5:]]
        if len(recent_temps) >= 2:
            temp_trend = (recent_temps[-1] - recent_temps[0]) / (len(recent_temps) - 1)
            predicted_temp = self.state.core_temperature + temp_trend * prediction_time
            return min(self.params.MAX_TEMP + 10, max(self.state.ambient_temperature, predicted_temp))
        
        return self.state.core_temperature
    
    def step(self, 
             current: float, 
             voltage: float, 
             soc: float,
             internal_resistance: float,
             delta_t: float = 1.0,
             ambient_temperature: Optional[float] = None) -> Dict:
        """
        执行一个热仿真步
        
        Args:
            current: 电流 (A)
            voltage: 电压 (V)
            soc: SOC (%)
            internal_resistance: 内阻 (Ω)
            delta_t: 时间步长 (s)
            ambient_temperature: 环境温度 (℃)
            
        Returns:
            热状态信息字典
        """
        # === 1. 计算产热 ===
        heat_sources = self.calculate_heat_generation(current, voltage, soc, internal_resistance)
        total_heat_generation = heat_sources['total']
        
        # === 2. 更新温度 ===
        temp_info = self.update_temperature(
            total_heat_generation, 
            delta_t, 
            current, 
            ambient_temperature
        )
        
        # === 3. 记录状态 ===
        thermal_record = {
            'timestamp': self.time_step_count,
            'simulation_time': self.total_time,
            'cell_id': self.cell_id,
            
            # 温度状态
            'core_temperature': self.state.core_temperature,
            'surface_temperature': self.state.surface_temperature,
            'ambient_temperature': self.state.ambient_temperature,
            'thermal_gradient': self.state.core_surface_gradient,
            
            # 热流状态
            'heat_generation': total_heat_generation,
            'heat_dissipation': self.state.heat_dissipation_rate,
            'net_heat_flow': self.state.net_heat_flow,
            'cooling_power': self.state.cooling_power,
            
            # 热源分解
            **heat_sources,
            
            # 安全状态
            'temperature_warning': self.state.temperature_warning,
            'temperature_alarm': self.state.temperature_alarm,
            'thermal_runaway_risk': self.state.thermal_runaway_risk,
            
            # 输入参数
            'current': current,
            'voltage': voltage,
            'soc': soc,
            'delta_t': delta_t
        }
        
        self.temperature_history.append(thermal_record)
        
        # === 4. 更新时间 ===
        self.time_step_count += 1
        self.total_time += delta_t
        
        # === 5. 维护历史长度 ===
        max_history = self.config.MAX_HISTORY_LENGTH if self.config else 1000
        if len(self.temperature_history) > max_history:
            self.temperature_history.pop(0)
        
        return thermal_record
    
    def reset(self, 
              initial_temp: Optional[float] = None,
              initial_ambient: Optional[float] = None,
              reset_history: bool = True) -> Dict:
        """
        重置热模型
        
        Args:
            initial_temp: 初始温度 (℃)
            initial_ambient: 初始环境温度 (℃)
            reset_history: 是否重置历史记录
            
        Returns:
            初始状态字典
        """
        # 设置初始温度
        if initial_temp is not None:
            self.state.core_temperature = initial_temp
            self.state.surface_temperature = initial_temp
        else:
            self.state.core_temperature = self.params.NOMINAL_TEMP
            self.state.surface_temperature = self.params.NOMINAL_TEMP
        
        if initial_ambient is not None:
            self.state.ambient_temperature = initial_ambient
        else:
            self.state.ambient_temperature = self.params.NOMINAL_TEMP
        
        # 重置其他状态
        self.state.heat_generation_rate = 0.0
        self.state.heat_dissipation_rate = 0.0
        self.state.net_heat_flow = 0.0
        self.state.core_surface_gradient = 0.0
        self.state.surface_ambient_gradient = 0.0
        self.state.cooling_power = 0.0
        self.state.temperature_warning = False
        self.state.temperature_alarm = False
        self.state.thermal_runaway_risk = 0.0
        
        # 重置时间
        self.time_step_count = 0
        self.total_time = 0.0
        
        # 重置历史
        if reset_history:
            self.temperature_history.clear()
            self.constraint_history.clear()
        
        initial_state = {
            'cell_id': self.cell_id,
            'core_temperature': self.state.core_temperature,
            'surface_temperature': self.state.surface_temperature,
            'ambient_temperature': self.state.ambient_temperature,
            'cooling_mode': self.cooling_mode.value,
            'reset_time': self.total_time
        }
        
        print(f"🔄 热模型 {self.cell_id} 已重置: T_core={self.state.core_temperature:.1f}℃, T_amb={self.state.ambient_temperature:.1f}℃")
        
        return initial_state
    
    def get_diagnostics(self) -> Dict:
        """获取热模型诊断信息"""
        if not self.temperature_history:
            return {'error': 'No thermal history available'}
        
        # 提取历史数据
        core_temps = [record['core_temperature'] for record in self.temperature_history]
        surface_temps = [record['surface_temperature'] for record in self.temperature_history]
        heat_gens = [record['heat_generation'] for record in self.temperature_history]
        
        diagnostics = {
            # 基本信息
            'cell_id': self.cell_id,
            'cooling_mode': self.cooling_mode.value,
            'simulation_steps': len(self.temperature_history),
            'total_time': self.total_time,
            
            # 温度统计
            'core_temp_range': (min(core_temps), max(core_temps)),
            'surface_temp_range': (min(surface_temps), max(surface_temps)),
            'avg_core_temperature': np.mean(core_temps),
            'avg_surface_temperature': np.mean(surface_temps),
            'max_thermal_gradient': max([record['thermal_gradient'] for record in self.temperature_history]),
            
            # 热性能
            'total_heat_generated': sum(heat_gens) * (self.config.SIMULATION_TIME_STEP if self.config else 1.0),
            'avg_heat_generation': np.mean(heat_gens),
            'peak_heat_generation': max(heat_gens),
            
            # 安全状态
            'warning_count': sum([record['temperature_warning'] for record in self.temperature_history]),
            'alarm_count': sum([record['temperature_alarm'] for record in self.temperature_history]),
            'max_thermal_runaway_risk': max([record['thermal_runaway_risk'] for record in self.temperature_history]),
            
            # 当前状态
            'current_core_temp': self.state.core_temperature,
            'current_thermal_gradient': self.state.core_surface_gradient,
            'thermal_health_status': self._get_thermal_health_status(),
            
            # 约束信息
            'current_derating_factor': self._calculate_temperature_derating_factor(),
            'cooling_efficiency': self.state.cooling_efficiency
        }
        
        return diagnostics
    
    def _get_thermal_health_status(self) -> str:
        """获取热健康状态"""
        max_temp = max(self.state.core_temperature, self.state.surface_temperature)
        
        if max_temp > self.params.MAX_TEMP:
            return 'Critical'
        elif max_temp > self.warning_thresholds['high_temp']:
            return 'Warning'
        elif max_temp > self.params.OPTIMAL_TEMP_RANGE[1]:
            return 'Elevated'
        else:
            return 'Normal'
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"ThermalModel({self.cell_id}): "
                f"T_core={self.state.core_temperature:.1f}℃, "
                f"T_surf={self.state.surface_temperature:.1f}℃, "
                f"ΔT={self.state.core_surface_gradient:.1f}℃, "
                f"Mode={self.cooling_mode.value}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"ThermalModel(cell_id='{self.cell_id}', "
                f"cooling_mode={self.cooling_mode.value}, "
                f"core_temp={self.state.core_temperature:.2f}℃, "
                f"thermal_gradient={self.state.core_surface_gradient:.2f}℃)")
