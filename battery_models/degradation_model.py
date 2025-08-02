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

class DegradationMode(Enum):
    """劣化模式枚举"""
    CALENDAR = "calendar"          # 日历老化
    CYCLE = "cycle"               # 循环老化
    COMBINED = "combined"         # 综合老化

@dataclass
class DegradationState:
    """劣化状态数据结构"""
    # 容量状态
    current_capacity: float = 280.0         # Ah, 当前容量
    initial_capacity: float = 280.0         # Ah, 初始容量
    capacity_retention: float = 100.0       # %, 容量保持率
    capacity_fade: float = 0.0              # Ah, 容量衰减量
    
    # 安时吞吐量
    amp_hour_throughput: float = 0.0        # Ah, 累积安时吞吐量 A'_t
    amp_hour_increment: float = 0.0         # Ah, 安时吞吐量增量 ΔA'_t
    
    # 劣化速率
    degradation_rate: float = 0.0           # Ah/s, 当前劣化速率
    capacity_loss_rate: float = 0.0         # %/cycle, 容量损失率
    
    # 成本相关
    degradation_cost: float = 0.0           # 元, 当前步劣化成本
    cumulative_cost: float = 0.0            # 元, 累积劣化成本
    
    # SOH相关 (为DRL提供)
    soh_current: float = 100.0              # %, 当前健康状态
    soh_change: float = 0.0                 # %, SOH变化量 ΔSOH
    soh_trend: float = 0.0                  # %/hour, SOH变化趋势
    
    # 老化因子
    aging_acceleration_factor: float = 1.0   # 老化加速因子
    temperature_factor: float = 1.0         # 温度老化因子
    current_factor: float = 1.0             # 电流老化因子

@dataclass
class DegradationParameters:
    """劣化模型参数"""
    # 核心物理参数
    activation_energy: float = -31700.0     # J, 活化能 E_a
    gas_constant: float = 8.314             # J/(mol·K), 气体常数 R
    exponent_z: float = 0.552               # 指数参数 z
    beta_coefficient: float = 370.3         # 系数 β
    
    # 经济参数
    battery_price: float = 0.486            # 元/Wh, 电池价格
    eol_capacity_threshold: float = 80.0    # %, 寿命终止容量阈值
    
    # 倍率系数多项式参数 (b_t = ax² + bx + c)
    rate_coeff_a: float = 448.96
    rate_coeff_b: float = -6301.1
    rate_coeff_c: float = 33840.0
    
    # 温度相关参数
    temp_coefficient: float = 1.421         # 温度系数 (℃/C²)
    temp_multiplier: float = 2.44           # 温度倍数因子
    
    # 单芯参数
    cell_capacity_ah: float = 280.0         # Ah, 单芯容量
    cell_voltage: float = 3.2               # V, 单芯电压
    cell_energy_kwh: float = 0.896          # kWh, 单芯能量

class BatteryDegradationModel:
    """
    电池劣化模型类
    基于安时吞吐量电池老化模型，实现动态容量衰减计算
    """
    
    def __init__(self, 
                 battery_params: BatteryParams,
                 system_config: Optional[SystemConfig] = None,
                 degradation_mode: DegradationMode = DegradationMode.COMBINED,
                 cell_id: str = "DegradationCell_001"):
        """
        初始化劣化模型
        
        Args:
            battery_params: 电池参数
            system_config: 系统配置
            degradation_mode: 劣化模式
            cell_id: 电池ID
        """
        self.battery_params = battery_params
        self.system_config = system_config
        self.degradation_mode = degradation_mode
        self.cell_id = cell_id
        
        # === 劣化参数 ===
        self.deg_params = DegradationParameters()
        self._initialize_degradation_parameters()
        
        # === 初始化状态 ===
        self.state = DegradationState()
        self.state.current_capacity = self.battery_params.CELL_CAPACITY
        self.state.initial_capacity = self.battery_params.CELL_CAPACITY
        
        # === 历史记录 ===
        self.degradation_history: List[Dict] = []
        self.soh_history: List[float] = []
        
        # === 仿真参数 ===
        self.time_step_count = 0
        self.total_simulation_time = 0.0
        
        # === 环境温度缓存 ===
        self.environmental_temperature = self.battery_params.NOMINAL_TEMP
        
        print(f"✅ 劣化模型初始化完成: {cell_id} ({degradation_mode.value})")
    
    def _initialize_degradation_parameters(self):
        """初始化劣化模型参数"""
        # 从电池参数更新劣化参数
        self.deg_params.cell_capacity_ah = self.battery_params.CELL_CAPACITY
        self.deg_params.cell_voltage = self.battery_params.NOMINAL_VOLTAGE
        self.deg_params.cell_energy_kwh = (self.battery_params.CELL_CAPACITY * 
                                         self.battery_params.NOMINAL_VOLTAGE / 1000.0)
        
        # 计算容量阈值 (sb)
        self.capacity_sb = self.deg_params.cell_capacity_ah  # 单个电芯容量作为sb
        
        # 根据电池组配置调整
        if hasattr(self.battery_params, 'PARALLEL_NUM'):
            self.capacity_sb *= self.battery_params.PARALLEL_NUM
    
    def calculate_c_rate(self, power: float, voltage: float) -> float:
        """
        计算充放电速率 c_t
        
        Args:
            power: 功率 P_t (W)
            voltage: 电压 V_t (V)
            
        Returns:
            c_rate: 充放电速率
        """
        if abs(voltage) < 1e-6:
            return 0.0
        
        # c_t = P_t / (V_t * s_b)
        c_rate = abs(power) / (voltage * self.capacity_sb)
        return c_rate
    
    def calculate_battery_temperature(self, 
                                    environmental_temp: float, 
                                    c_rate: float) -> float:
        """
        计算电池温度 T_t
        
        Args:
            environmental_temp: 环境温度 T_env (℃)
            c_rate: 充放电速率 c_t
            
        Returns:
            battery_temp: 电池温度 (℃)
        """
        # T_t = T_env + 1.421 * c_t²
        temperature_rise = self.deg_params.temp_coefficient * (c_rate ** 2)
        battery_temp = environmental_temp + temperature_rise
        
        return battery_temp
    
    def calculate_rate_coefficient(self, c_rate: float) -> float:
        """
        计算倍率系数 b_t
        
        Args:
            c_rate: 充放电速率 c_t
            
        Returns:
            b_coefficient: 倍率系数
        """
        # b_t = 448.96 * c_t² - 6301.1 * c_t + 33840
        b_coeff = (self.deg_params.rate_coeff_a * (c_rate ** 2) + 
                  self.deg_params.rate_coeff_b * c_rate + 
                  self.deg_params.rate_coeff_c)
        
        return max(0.1, b_coeff)  # 确保系数为正值
    
    def calculate_amp_hour_increment(self, 
                                   power: float, 
                                   voltage: float, 
                                   delta_t: float) -> float:
        """
        计算安时吞吐量增量 ΔA'_t
        
        Args:
            power: 功率 P_t (W)
            voltage: 电压 V_t (V)
            delta_t: 时间步长 Δt (s)
            
        Returns:
            amp_hour_increment: 安时吞吐量增量 (Ah)
        """
        if abs(voltage) < 1e-6:
            return 0.0
        
        # ΔA'_t = (1/3600) * (P_t/V_t) * Δt * (2.44/s_b)
        amp_hour_increment = (1.0 / 3600.0) * (abs(power) / voltage) * delta_t * (self.deg_params.temp_multiplier / self.capacity_sb)
        
        return amp_hour_increment
    
    def calculate_capacity_degradation(self, 
                                     power: float, 
                                     voltage: float, 
                                     delta_t: float,
                                     environmental_temp: Optional[float] = None) -> Dict[str, float]:
        """
        计算容量衰减 ΔQ_t
        
        Args:
            power: 功率 P_t (W)
            voltage: 电压 V_t (V)
            delta_t: 时间步长 Δt (s)
            environmental_temp: 环境温度 (℃)
            
        Returns:
            劣化计算结果字典
        """
        if environmental_temp is not None:
            self.environmental_temperature = environmental_temp
        
        # === 1. 计算基础参数 ===
        c_rate = self.calculate_c_rate(power, voltage)
        battery_temp = self.calculate_battery_temperature(self.environmental_temperature, c_rate)
        battery_temp_kelvin = battery_temp + 273.15  # 转换为开尔文
        
        amp_hour_increment = self.calculate_amp_hour_increment(power, voltage, delta_t)
        rate_coefficient = self.calculate_rate_coefficient(c_rate)
        
        # === 2. 计算指数项 ===
        # exp((-E_a + β * c_t) / (R * T_t))
        exponent_numerator = (-self.deg_params.activation_energy + 
                            self.deg_params.beta_coefficient * c_rate)
        exponent_denominator = self.deg_params.gas_constant * battery_temp_kelvin
        
        if abs(exponent_denominator) > 1e-10:
            exponential_term = math.exp(exponent_numerator / exponent_denominator)
        else:
            exponential_term = 1.0
        
        # === 3. 计算幂次项 ===
        # (A'_t)^(z-1)
        if self.state.amp_hour_throughput > 0:
            power_term = (self.state.amp_hour_throughput ** (self.deg_params.exponent_z - 1))
        else:
            power_term = 0.0
        
        # === 4. 计算容量衰减量 ===
        # ΔQ_t = b * exp(...) * (A'_t)^(z-1) * z * ΔA'_t
        capacity_degradation = (rate_coefficient * exponential_term * power_term * 
                              self.deg_params.exponent_z * amp_hour_increment)
        
        # === 5. 计算劣化成本 ===
        # f_ESS,t = (ΔQ_t / (100-80)) * price_ESS
        capacity_loss_percentage = capacity_degradation / self.state.initial_capacity * 100.0
        total_capacity_loss_percentage = 100.0 - self.deg_params.eol_capacity_threshold  # 20%
        
        if total_capacity_loss_percentage > 0:
            cost_ratio = capacity_loss_percentage / total_capacity_loss_percentage
        else:
            cost_ratio = 0.0
        
        battery_total_cost = (self.deg_params.cell_energy_kwh * 1000 * 
                            self.deg_params.battery_price)  # 元
        degradation_cost = cost_ratio * battery_total_cost
        
        # === 6. 返回结果 ===
        degradation_result = {
            'c_rate': c_rate,
            'battery_temperature': battery_temp,
            'battery_temperature_kelvin': battery_temp_kelvin,
            'amp_hour_increment': amp_hour_increment,
            'rate_coefficient': rate_coefficient,
            'exponential_term': exponential_term,
            'power_term': power_term,
            'capacity_degradation': capacity_degradation,
            'capacity_loss_percentage': capacity_loss_percentage,
            'degradation_cost': degradation_cost,
            'environmental_temp': self.environmental_temperature
        }
        
        return degradation_result
    
    def update_degradation_state(self, degradation_result: Dict[str, float]) -> Dict[str, float]:
        """
        更新劣化状态
        
        Args:
            degradation_result: 劣化计算结果
            
        Returns:
            状态更新信息
        """
        # === 1. 更新安时吞吐量 ===
        old_amp_hour_throughput = self.state.amp_hour_throughput
        self.state.amp_hour_increment = degradation_result['amp_hour_increment']
        self.state.amp_hour_throughput += self.state.amp_hour_increment
        
        # === 2. 更新容量 ===
        old_capacity = self.state.current_capacity
        old_capacity_fade = self.state.capacity_fade
        
        capacity_degradation = degradation_result['capacity_degradation']
        self.state.capacity_fade += capacity_degradation
        self.state.current_capacity = self.state.initial_capacity - self.state.capacity_fade
        
        # 确保容量不小于EOL阈值
        min_capacity = self.state.initial_capacity * self.deg_params.eol_capacity_threshold / 100.0
        self.state.current_capacity = max(min_capacity, self.state.current_capacity)
        
        # === 3. 更新容量保持率和SOH ===
        self.state.capacity_retention = (self.state.current_capacity / 
                                       self.state.initial_capacity * 100.0)
        
        old_soh = self.state.soh_current
        self.state.soh_current = self.state.capacity_retention
        self.state.soh_change = self.state.soh_current - old_soh
        
        # === 4. 更新劣化速率 ===
        if len(self.degradation_history) > 0:
            time_interval = (self.system_config.SIMULATION_TIME_STEP 
                           if self.system_config else 1.0)
            self.state.degradation_rate = capacity_degradation / time_interval
        
        # === 5. 更新成本 ===
        self.state.degradation_cost = degradation_result['degradation_cost']
        self.state.cumulative_cost += self.state.degradation_cost
        
        # === 6. 更新老化因子 ===
        self.state.aging_acceleration_factor = self._calculate_aging_acceleration_factor(
            degradation_result['c_rate'], degradation_result['battery_temperature']
        )
        self.state.temperature_factor = self._calculate_temperature_factor(
            degradation_result['battery_temperature']
        )
        self.state.current_factor = self._calculate_current_factor(
            degradation_result['c_rate']
        )
        
        # === 7. 更新SOH趋势 ===
        self._update_soh_trend()
        
        # === 8. 返回更新信息 ===
        update_info = {
            'capacity_change': self.state.current_capacity - old_capacity,
            'soh_change': self.state.soh_change,
            'amp_hour_increase': self.state.amp_hour_throughput - old_amp_hour_throughput,
            'degradation_cost': self.state.degradation_cost,
            'current_capacity': self.state.current_capacity,
            'current_soh': self.state.soh_current,
            'capacity_retention': self.state.capacity_retention
        }
        
        return update_info
    
    def _calculate_aging_acceleration_factor(self, c_rate: float, temperature: float) -> float:
        """计算老化加速因子"""
        # 基于C率的加速因子
        c_rate_factor = 1.0 + max(0, c_rate - 1.0) * 0.5  # C率超过1时加速老化
        
        # 基于温度的加速因子
        optimal_temp = self.battery_params.OPTIMAL_TEMP_RANGE[1]  # 35℃
        if temperature > optimal_temp:
            temp_factor = 1.0 + (temperature - optimal_temp) * 0.02  # 每度温升增加2%老化
        else:
            temp_factor = 1.0
        
        return c_rate_factor * temp_factor
    
    def _calculate_temperature_factor(self, temperature: float) -> float:
        """计算温度因子"""
        # 阿伦尼乌斯关系简化
        reference_temp = 25.0  # ℃
        activation_energy_simplified = 0.5  # eV (简化值)
        
        temp_factor = math.exp(activation_energy_simplified * 
                             (1/(reference_temp + 273.15) - 1/(temperature + 273.15)))
        
        return temp_factor
    
    def _calculate_current_factor(self, c_rate: float) -> float:
        """计算电流因子"""
        # 基于C率的非线性关系
        if c_rate <= 1.0:
            return 1.0
        else:
            return 1.0 + (c_rate - 1.0) ** 1.5 * 0.3
    
    def _update_soh_trend(self):
        """更新SOH趋势"""
        # 记录SOH历史
        self.soh_history.append(self.state.soh_current)
        
        # 保持历史窗口大小
        max_history_window = 100
        if len(self.soh_history) > max_history_window:
            self.soh_history.pop(0)
        
        # 计算SOH趋势 (线性回归斜率)
        if len(self.soh_history) >= 10:
            x = np.arange(len(self.soh_history))
            y = np.array(self.soh_history)
            
            # 简单线性回归
            n = len(x)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xy = np.sum(x * y)
            sum_x2 = np.sum(x * x)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # 转换为每小时SOH变化趋势
                time_step = self.system_config.SIMULATION_TIME_STEP if self.system_config else 1.0
                self.state.soh_trend = slope * 3600.0 / time_step  # %/hour
            else:
                self.state.soh_trend = 0.0
        else:
            self.state.soh_trend = 0.0
    
    def get_delta_soh_for_drl(self) -> float:
        """
        为DRL上层提供ΔSOH (老化趋势)
        
        Returns:
            delta_soh: SOH变化趋势 (%/hour)
        """
        return self.state.soh_trend
    
    def get_aging_statistics_for_drl(self) -> Dict[str, float]:
        """
        为DRL通信层提供老化统计信息
        
        Returns:
            老化统计字典
        """
        return {
            'current_soh': self.state.soh_current,
            'soh_change_rate': self.state.soh_trend,
            'capacity_retention': self.state.capacity_retention,
            'aging_acceleration_factor': self.state.aging_acceleration_factor,
            'cumulative_degradation_cost': self.state.cumulative_cost,
            'amp_hour_throughput': self.state.amp_hour_throughput,
            'equivalent_cycles': self.state.amp_hour_throughput / (2 * self.state.initial_capacity),
            'remaining_cycles_estimate': self._estimate_remaining_cycles()
        }
    
    def _estimate_remaining_cycles(self) -> float:
        """估算剩余循环次数"""
        current_cycles = self.state.amp_hour_throughput / (2 * self.state.initial_capacity)
        
        if self.state.soh_trend < 0 and abs(self.state.soh_trend) > 1e-6:
            # 基于当前趋势预测
            soh_remaining = self.state.soh_current - self.deg_params.eol_capacity_threshold
            time_to_eol_hours = soh_remaining / abs(self.state.soh_trend)
            
            # 假设平均C率，估算剩余循环
            avg_cycle_time = 4.0  # 小时 (假设2C率充放电)
            remaining_cycles = time_to_eol_hours / avg_cycle_time
            
            return max(0, remaining_cycles)
        else:
            # 基于设计寿命估算
            design_cycles = self.battery_params.CYCLE_LIFE
            return max(0, design_cycles - current_cycles)
    
    def step(self, 
             power: float, 
             voltage: float, 
             delta_t: float,
             environmental_temp: Optional[float] = None,
             thermal_model_temp: Optional[float] = None) -> Dict:
        """
        执行一个劣化仿真步
        
        Args:
            power: 功率 (W)
            voltage: 电压 (V)
            delta_t: 时间步长 (s)
            environmental_temp: 环境温度 (℃)
            thermal_model_temp: 热模型提供的温度 (℃)
            
        Returns:
            劣化信息字典
        """
        # === 1. 温度处理 ===
        if thermal_model_temp is not None:
            # 优先使用热模型温度
            effective_temp = thermal_model_temp
        elif environmental_temp is not None:
            effective_temp = environmental_temp
        else:
            effective_temp = self.environmental_temperature
        
        # === 2. 计算劣化 ===
        degradation_result = self.calculate_capacity_degradation(
            power, voltage, delta_t, effective_temp
        )
        
        # === 3. 更新状态 ===
        update_info = self.update_degradation_state(degradation_result)
        
        # === 4. 记录状态 ===
        degradation_record = {
            'timestamp': self.time_step_count,
            'simulation_time': self.total_simulation_time,
            'cell_id': self.cell_id,
            
            # 输入参数
            'power': power,
            'voltage': voltage,
            'delta_t': delta_t,
            'environmental_temp': effective_temp,
            
            # 计算中间量
            'c_rate': degradation_result['c_rate'],
            'battery_temperature': degradation_result['battery_temperature'],
            'amp_hour_increment': degradation_result['amp_hour_increment'],
            'capacity_degradation': degradation_result['capacity_degradation'],
            'degradation_cost': degradation_result['degradation_cost'],
            
            # 状态量
            'current_capacity': self.state.current_capacity,
            'capacity_retention': self.state.capacity_retention,
            'soh_current': self.state.soh_current,
            'soh_change': self.state.soh_change,
            'soh_trend': self.state.soh_trend,
            'amp_hour_throughput': self.state.amp_hour_throughput,
            'cumulative_cost': self.state.cumulative_cost,
            
            # 老化因子
            'aging_acceleration_factor': self.state.aging_acceleration_factor,
            'temperature_factor': self.state.temperature_factor,
            'current_factor': self.state.current_factor
        }
        
        self.degradation_history.append(degradation_record)
        
        # === 5. 更新时间 ===
        self.time_step_count += 1
        self.total_simulation_time += delta_t
        
        # === 6. 维护历史长度 ===
        max_history = self.system_config.MAX_HISTORY_LENGTH if self.system_config else 1000
        if len(self.degradation_history) > max_history:
            self.degradation_history.pop(0)
        
        return degradation_record
    
    def reset(self, 
              reset_to_new: bool = True,
              initial_soh: Optional[float] = None,
              reset_history: bool = True) -> Dict:
        """
        重置劣化模型
        
        Args:
            reset_to_new: 是否重置为全新电池
            initial_soh: 初始SOH (%)
            reset_history: 是否重置历史记录
            
        Returns:
            初始状态字典
        """
        if reset_to_new:
            # 重置为全新电池
            self.state.current_capacity = self.state.initial_capacity
            self.state.capacity_retention = 100.0
            self.state.capacity_fade = 0.0
            self.state.soh_current = 100.0
        elif initial_soh is not None:
            # 设置指定SOH
            self.state.soh_current = np.clip(initial_soh, 
                                           self.deg_params.eol_capacity_threshold, 100.0)
            self.state.capacity_retention = self.state.soh_current
            self.state.current_capacity = (self.state.initial_capacity * 
                                         self.state.capacity_retention / 100.0)
            self.state.capacity_fade = self.state.initial_capacity - self.state.current_capacity
        
        # 重置其他状态
        self.state.amp_hour_throughput = 0.0
        self.state.amp_hour_increment = 0.0
        self.state.degradation_rate = 0.0
        self.state.degradation_cost = 0.0
        self.state.cumulative_cost = 0.0
        self.state.soh_change = 0.0
        self.state.soh_trend = 0.0
        self.state.aging_acceleration_factor = 1.0
        self.state.temperature_factor = 1.0
        self.state.current_factor = 1.0
        
        # 重置时间
        self.time_step_count = 0
        self.total_simulation_time = 0.0
        
        # 重置历史
        if reset_history:
            self.degradation_history.clear()
            self.soh_history.clear()
        
        initial_state = {
            'cell_id': self.cell_id,
            'initial_capacity': self.state.initial_capacity,
            'current_capacity': self.state.current_capacity,
            'capacity_retention': self.state.capacity_retention,
            'soh_current': self.state.soh_current,
            'degradation_mode': self.degradation_mode.value,
            'reset_time': self.total_simulation_time
        }
        
        print(f"🔄 劣化模型 {self.cell_id} 已重置: SOH={self.state.soh_current:.1f}%, "
              f"容量={self.state.current_capacity:.1f}Ah")
        
        return initial_state
    
    def get_diagnostics(self) -> Dict:
        """获取劣化模型诊断信息"""
        if not self.degradation_history:
            return {'error': 'No degradation history available'}
        
        # 提取历史数据
        soh_values = [record['soh_current'] for record in self.degradation_history]
        capacity_values = [record['current_capacity'] for record in self.degradation_history]
        cost_values = [record['degradation_cost'] for record in self.degradation_history]
        
        diagnostics = {
            # 基本信息
            'cell_id': self.cell_id,
            'degradation_mode': self.degradation_mode.value,
            'simulation_steps': len(self.degradation_history),
            'total_time': self.total_simulation_time,
            
            # 容量统计
            'initial_capacity': self.state.initial_capacity,
            'current_capacity': self.state.current_capacity,
            'capacity_fade': self.state.capacity_fade,
            'capacity_retention': self.state.capacity_retention,
            'capacity_range': (min(capacity_values), max(capacity_values)),
            
            # SOH统计
            'current_soh': self.state.soh_current,
            'soh_range': (min(soh_values), max(soh_values)),
            'soh_trend': self.state.soh_trend,
            'total_soh_loss': 100.0 - self.state.soh_current,
            
            # 老化统计
            'total_amp_hour_throughput': self.state.amp_hour_throughput,
            'equivalent_cycles': self.state.amp_hour_throughput / (2 * self.state.initial_capacity),
            'aging_acceleration_factor': self.state.aging_acceleration_factor,
            
            # 成本统计
            'total_degradation_cost': self.state.cumulative_cost,
            'avg_step_cost': np.mean(cost_values),
            'peak_step_cost': max(cost_values),
            
            # 寿命预测
            'estimated_remaining_cycles': self._estimate_remaining_cycles(),
            'time_to_eol_estimate': self._estimate_time_to_eol(),
            
            # 健康状态
            'degradation_health_status': self._get_degradation_health_status()
        }
        
        return diagnostics
    
    def _estimate_time_to_eol(self) -> float:
        """估算到达寿命终止的时间"""
        if abs(self.state.soh_trend) < 1e-6:
            return float('inf')
        
        soh_remaining = self.state.soh_current - self.deg_params.eol_capacity_threshold
        if soh_remaining <= 0:
            return 0.0
        
        time_to_eol_hours = soh_remaining / abs(self.state.soh_trend)
        return time_to_eol_hours
    
    def _get_degradation_health_status(self) -> str:
        """获取劣化健康状态"""
        if self.state.soh_current <= self.deg_params.eol_capacity_threshold:
            return 'End of Life'
        elif self.state.soh_current <= 85:
            return 'Severely Degraded'
        elif self.state.soh_current <= 90:
            return 'Moderately Degraded'
        elif self.state.soh_current <= 95:
            return 'Mildly Degraded'
        else:
            return 'Healthy'
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"BatteryDegradationModel({self.cell_id}): "
                f"SOH={self.state.soh_current:.1f}%, "
                f"容量={self.state.current_capacity:.1f}Ah, "
                f"循环={self.state.amp_hour_throughput/(2*self.state.initial_capacity):.1f}, "
                f"成本={self.state.cumulative_cost:.2f}元")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"BatteryDegradationModel(cell_id='{self.cell_id}', "
                f"mode={self.degradation_mode.value}, "
                f"soh={self.state.soh_current:.2f}%, "
                f"capacity={self.state.current_capacity:.1f}Ah)")
