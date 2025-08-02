"""
单体电池模型
实现完整的电化学行为仿真，包括SOC计算、电压模型、功率限制等
"""

import numpy as np
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig

@dataclass
class CellState:
    """电池单体状态数据结构"""
    soc: float = 50.0              # %, 荷电状态
    voltage: float = 3.2           # V, 端电压
    current: float = 0.0           # A, 电流
    temperature: float = 25.0      # ℃, 温度
    energy_stored: float = 0.0     # Wh, 储存能量
    power: float = 0.0             # W, 功率
    
    # 内部状态
    ocv: float = 3.2               # V, 开路电压
    internal_resistance: float = 0.001  # Ω, 内阻
    capacity_remaining: float = 280.0    # Ah, 剩余容量
    
    # 累积统计
    cumulative_charge: float = 0.0      # Ah, 累积充放电量
    cycle_count: float = 0.0             # 等效循环次数
    aging_factor: float = 1.0            # 老化因子

class BatteryCellModel:
    """
    单体电池模型类
    实现磷酸铁锂电池的完整电化学行为
    """
    
    def __init__(self, battery_params: BatteryParams, 
                 system_config: Optional[SystemConfig] = None,
                 cell_id: str = "Cell_001"):
        """
        初始化电池模型
        
        Args:
            battery_params: 电池参数配置
            system_config: 系统配置 (可选)
            cell_id: 电池单体ID
        """
        self.params = battery_params
        self.config = system_config
        self.cell_id = cell_id
        
        # 验证参数
        if not battery_params.validate_params():
            raise ValueError(f"电池参数验证失败: {cell_id}")
        
        # === 初始化状态 ===
        self.state = CellState()
        self.state.soc = self.params.NOMINAL_SOC
        self.state.temperature = self.params.NOMINAL_TEMP
        self.state.capacity_remaining = self.params.CELL_CAPACITY
        self.state.internal_resistance = self.params.INTERNAL_RESISTANCE
        
        # === 历史记录 ===
        self.state_history: List[Dict] = []
        self.performance_metrics: Dict = {}
        
        # === 仿真参数 ===
        self.time_step_count = 0
        self.total_simulation_time = 0.0  # s
        
        # === 内部计算缓存 ===
        self._last_soc_change_time = 0.0
        self._soc_trend_window = []  # SOC变化趋势窗口
        
        # 初始化计算
        self._update_derived_states()
        
        print(f"✅ 电池单体模型初始化完成: {cell_id}")
    
    def _update_derived_states(self):
        """更新衍生状态量"""
        # 更新开路电压
        self.state.ocv = self.params.get_ocv_from_soc(self.state.soc)
        
        # 更新储存能量
        self.state.energy_stored = (self.state.soc / 100.0 * 
                                   self.state.capacity_remaining * 
                                   self.params.NOMINAL_VOLTAGE)
        
        # 更新功率
        self.state.power = self.state.voltage * self.state.current
    
    def calculate_terminal_voltage(self, current: float, 
                                  temperature: Optional[float] = None) -> float:
        """
        计算端电压
        考虑内阻、温度、SOC和电流方向的影响
        
        Args:
            current: 电流 (A, 正为充电，负为放电)
            temperature: 温度 (℃, 可选)
            
        Returns:
            terminal_voltage: 端电压 (V)
        """
        if temperature is None:
            temperature = self.state.temperature
        
        # 开路电压
        ocv = self.params.get_ocv_from_soc(self.state.soc)
        
        # 温度对内阻的影响
        temp_factor = self._get_temperature_resistance_factor(temperature)
        effective_resistance = self.state.internal_resistance * temp_factor
        
        # SOC对内阻的影响
        soc_factor = self._get_soc_resistance_factor(self.state.soc)
        effective_resistance *= soc_factor
        
        # 电流对内阻的影响 (非线性)
        current_factor = self._get_current_resistance_factor(abs(current))
        effective_resistance *= current_factor
        
        # 欧姆压降
        ohmic_drop = current * effective_resistance
        
        # 极化压降 (简化模型)
        polarization_drop = self._calculate_polarization(current, temperature)
        
        # 端电压计算
        terminal_voltage = ocv + ohmic_drop + polarization_drop
        
        # 电压限制
        return np.clip(terminal_voltage, 
                      self.params.MIN_VOLTAGE, 
                      self.params.MAX_VOLTAGE)
    
    def _get_temperature_resistance_factor(self, temperature: float) -> float:
        """获取温度对内阻的影响因子"""
        # 温度系数 (典型值: 每℃变化0.5%的内阻变化)
        temp_coeff = 0.005  # 1/℃
        reference_temp = 25.0  # ℃
        
        temp_factor = 1.0 + temp_coeff * (reference_temp - temperature)
        return max(0.5, min(3.0, temp_factor))  # 限制在0.5-3倍范围内
    
    def _get_soc_resistance_factor(self, soc: float) -> float:
        """获取SOC对内阻的影响因子"""
        # 在极端SOC下内阻增加
        if soc < 10:
            return 1.0 + (10 - soc) * 0.1  # 低SOC内阻增加
        elif soc > 90:
            return 1.0 + (soc - 90) * 0.05  # 高SOC内阻略增
        else:
            return 1.0
    
    def _get_current_resistance_factor(self, abs_current: float) -> float:
        """获取电流对内阻的影响因子"""
        # 大电流下内阻非线性增加
        c_rate = abs_current / self.state.capacity_remaining
        if c_rate > 1.0:
            return 1.0 + (c_rate - 1.0) * 0.2
        else:
            return 1.0
    
    def _calculate_polarization(self, current: float, temperature: float) -> float:
        """
        计算极化压降
        包括活化极化和浓差极化
        
        Args:
            current: 电流 (A)
            temperature: 温度 (℃)
            
        Returns:
            polarization_voltage: 极化电压 (V)
        """
        if abs(current) < 1e-6:
            return 0.0
        
        # 极化电阻 (简化模型)
        base_polarization_resistance = 0.0005  # Ω
        
        # 温度对极化的影响
        temp_factor = math.exp(-500 / (temperature + 273.15))  # 阿伦尼乌斯关系
        
        # SOC对极化的影响
        soc_factor = 1.0
        if self.state.soc < 20:
            soc_factor = 1.0 + (20 - self.state.soc) * 0.02
        elif self.state.soc > 80:
            soc_factor = 1.0 + (self.state.soc - 80) * 0.01
        
        # 电流相关的非线性项
        current_nonlinear = current * (1 + abs(current) / self.state.capacity_remaining * 0.1)
        
        # 总极化电压
        polarization_resistance = base_polarization_resistance * temp_factor * soc_factor
        polarization_voltage = current_nonlinear * polarization_resistance
        
        return polarization_voltage
    
    def calculate_current_limits(self, temperature: Optional[float] = None) -> Tuple[float, float]:
        """
        计算当前状态下的电流限制
        考虑电压、SOC、温度和C率限制
        
        Args:
            temperature: 温度 (℃, 可选)
            
        Returns:
            (max_charge_current, max_discharge_current): 最大充放电电流 (A)
        """
        if temperature is None:
            temperature = self.state.temperature
        
        # 1. C率限制
        max_charge_c, max_discharge_c = self.params.get_c_rate_limits(
            self.state.soc, temperature
        )
        charge_current_c = self.state.capacity_remaining * max_charge_c
        discharge_current_c = self.state.capacity_remaining * max_discharge_c
        
        # 2. 电压限制
        ocv = self.params.get_ocv_from_soc(self.state.soc)
        
        # 充电电压限制
        voltage_margin_charge = self.params.MAX_VOLTAGE - ocv
        if voltage_margin_charge > 0:
            # 考虑内阻和极化
            effective_resistance = (self.state.internal_resistance * 
                                   self._get_temperature_resistance_factor(temperature))
            charge_current_v = voltage_margin_charge / (effective_resistance + 0.0005)
        else:
            charge_current_v = 0.0
        
        # 放电电压限制  
        voltage_margin_discharge = ocv - self.params.MIN_VOLTAGE
        if voltage_margin_discharge > 0:
            effective_resistance = (self.state.internal_resistance * 
                                   self._get_temperature_resistance_factor(temperature))
            discharge_current_v = voltage_margin_discharge / (effective_resistance + 0.0005)
        else:
            discharge_current_v = 0.0
        
        # 3. SOC限制
        if self.state.soc >= self.params.MAX_SOC:
            charge_current_soc = 0.0
        else:
            # 接近上限时线性衰减
            soc_margin = max(0, self.params.MAX_SOC - self.state.soc)
            soc_factor = min(1.0, soc_margin / 5.0)  # 5%衰减区间
            charge_current_soc = charge_current_c * soc_factor
        
        if self.state.soc <= self.params.MIN_SOC:
            discharge_current_soc = 0.0
        else:
            # 接近下限时线性衰减
            soc_margin = max(0, self.state.soc - self.params.MIN_SOC)
            soc_factor = min(1.0, soc_margin / 5.0)  # 5%衰减区间
            discharge_current_soc = discharge_current_c * soc_factor
        
        # 4. 温度限制
        if temperature < self.params.MIN_TEMP + 5:
            temp_factor = max(0.1, (temperature - self.params.MIN_TEMP) / 10.0)
        elif temperature > self.params.MAX_TEMP - 5:
            temp_factor = max(0.1, (self.params.MAX_TEMP - temperature) / 10.0)
        else:
            temp_factor = 1.0
        
        # 取最严格的限制
        max_charge_current = min(charge_current_c, charge_current_v, 
                               charge_current_soc) * temp_factor
        max_discharge_current = min(discharge_current_c, discharge_current_v, 
                                  discharge_current_soc) * temp_factor
        
        # 安全裕度
        safety_factor = self.params.SAFETY_MARGINS.get('current_factor', 0.9)
        max_charge_current *= safety_factor
        max_discharge_current *= safety_factor
        
        return max(0.0, max_charge_current), max(0.0, max_discharge_current)
    
    def update_soc(self, current: float, delta_t: float, 
                   efficiency: Optional[float] = None) -> float:
        """
        更新SOC (改进的库仑计数法)
        考虑充放电效率和温度影响
        
        Args:
            current: 电流 (A, 正为充电，负为放电)
            delta_t: 时间步长 (s)
            efficiency: 充放电效率 (可选)
            
        Returns:
            soc_change: SOC变化量 (%)
        """
        if efficiency is None:
            if current > 0:  # 充电
                efficiency = self.params.CHARGE_EFFICIENCY
            else:  # 放电
                efficiency = self.params.DISCHARGE_EFFICIENCY
        
        # 温度对效率的影响
        temp_efficiency_factor = self._get_temperature_efficiency_factor(
            self.state.temperature
        )
        effective_efficiency = efficiency * temp_efficiency_factor
        
        # 电荷变化 (考虑效率)
        if current > 0:  # 充电
            delta_charge = current * delta_t / 3600.0 * effective_efficiency
        else:  # 放电  
            delta_charge = current * delta_t / 3600.0 / effective_efficiency
        
        # SOC变化
        if self.state.capacity_remaining > 0:
            delta_soc = (delta_charge / self.state.capacity_remaining) * 100.0
        else:
            delta_soc = 0.0
        
        # 更新SOC
        old_soc = self.state.soc
        self.state.soc = np.clip(self.state.soc + delta_soc, 0.0, 100.0)
        
        # 更新累积充放电量
        self.state.cumulative_charge += abs(delta_charge)
        
        # 更新等效循环次数 (简化计算)
        if abs(delta_charge) > 0:
            cycle_increment = abs(delta_charge) / self.state.capacity_remaining
            self.state.cycle_count += cycle_increment
        
        # 记录SOC变化趋势
        self._update_soc_trend(self.state.soc - old_soc)
        
        return self.state.soc - old_soc
    
    def _get_temperature_efficiency_factor(self, temperature: float) -> float:
        """获取温度对效率的影响因子"""
        # 最佳效率温度范围
        optimal_temp_range = self.params.OPTIMAL_TEMP_RANGE
        
        if optimal_temp_range[0] <= temperature <= optimal_temp_range[1]:
            return 1.0
        elif temperature < optimal_temp_range[0]:
            # 低温效率降低
            temp_diff = optimal_temp_range[0] - temperature
            return max(0.8, 1.0 - temp_diff * 0.01)
        else:
            # 高温效率降低
            temp_diff = temperature - optimal_temp_range[1]
            return max(0.85, 1.0 - temp_diff * 0.005)
    
    def _update_soc_trend(self, soc_change: float):
        """更新SOC变化趋势"""
        self._soc_trend_window.append(soc_change)
        
        # 保持窗口大小
        max_window_size = 60  # 记录最近60个时间步
        if len(self._soc_trend_window) > max_window_size:
            self._soc_trend_window.pop(0)
    
    def get_soc_trend(self) -> float:
        """获取SOC变化趋势 (%/h)"""
        if len(self._soc_trend_window) < 2:
            return 0.0
        
        # 计算平均变化率
        avg_change_per_step = np.mean(self._soc_trend_window)
        
        # 转换为每小时变化率
        if self.config:
            time_step = self.config.SIMULATION_TIME_STEP
        else:
            time_step = 1.0
        
        return avg_change_per_step * 3600.0 / time_step
    
    def step(self, power_command: float, delta_t: float = 1.0, 
             ambient_temperature: Optional[float] = None) -> Dict:
        """
        执行一个仿真步
        
        Args:
            power_command: 功率指令 (W, 正为充电，负为放电)
            delta_t: 时间步长 (s)
            ambient_temperature: 环境温度 (℃, 可选)
            
        Returns:
            状态信息字典
        """
        # === 1. 环境更新 ===
        if ambient_temperature is not None:
            # 简化的温度动态 (实际应该有热模型)
            temp_time_constant = 300.0  # s, 温度时间常数
            temp_change_rate = (ambient_temperature - self.state.temperature) / temp_time_constant
            self.state.temperature += temp_change_rate * delta_t
            self.state.temperature = np.clip(self.state.temperature,
                                           self.params.MIN_TEMP,
                                           self.params.MAX_TEMP)
        
        # === 2. 功率到电流转换 ===
        if abs(power_command) < 1e-6:
            target_current = 0.0
        else:
            # 迭代计算实际电流 (考虑电压随电流变化)
            target_current = self._solve_current_from_power(power_command)
        
        # === 3. 电流限制检查 ===
        max_charge_current, max_discharge_current = self.calculate_current_limits()
        
        if target_current > 0:  # 充电
            actual_current = min(target_current, max_charge_current)
        elif target_current < 0:  # 放电
            actual_current = max(target_current, -max_discharge_current)
        else:  # 静置
            actual_current = 0.0
        
        # === 4. 状态更新 ===
        # 更新SOC
        soc_change = self.update_soc(actual_current, delta_t)
        
        # 更新电压
        self.state.voltage = self.calculate_terminal_voltage(actual_current)
        
        # 更新电流
        self.state.current = actual_current
        
        # 更新衍生状态
        self._update_derived_states()
        
        # === 5. 性能计算 ===
        actual_power = self.state.voltage * actual_current
        
        if abs(power_command) > 1e-6:
            power_efficiency = abs(actual_power / power_command)
            power_error = abs(actual_power - power_command)
        else:
            power_efficiency = 1.0
            power_error = abs(actual_power)
        
        # === 6. 状态记录 ===
        current_state = {
            # 基本状态
            'timestamp': self.time_step_count,
            'simulation_time': self.total_simulation_time,
            'cell_id': self.cell_id,
            
            # 电气状态
            'soc': self.state.soc,
            'soc_change': soc_change,
            'voltage': self.state.voltage,
            'ocv': self.state.ocv,
            'current': actual_current,
            'power_command': power_command,
            'actual_power': actual_power,
            'power_efficiency': power_efficiency,
            'power_error': power_error,
            
            # 能量状态
            'energy_stored': self.state.energy_stored,
            'capacity_remaining': self.state.capacity_remaining,
            'cumulative_charge': self.state.cumulative_charge,
            'cycle_count': self.state.cycle_count,
            
            # 热状态
            'temperature': self.state.temperature,
            'ambient_temperature': ambient_temperature,
            
            # 限制信息
            'max_charge_current': max_charge_current,
            'max_discharge_current': max_discharge_current,
            
            # 内部状态
            'internal_resistance': self.state.internal_resistance,
            'aging_factor': self.state.aging_factor,
            
            # 趋势信息
            'soc_trend': self.get_soc_trend()
        }
        
        # 添加到历史记录
        self.state_history.append(current_state)
        
        # 更新时间
        self.time_step_count += 1
        self.total_simulation_time += delta_t
        
        return current_state
    
    def _solve_current_from_power(self, power_command: float, 
                                 max_iterations: int = 10, 
                                 tolerance: float = 1e-3) -> float:
        """
        从功率指令求解电流 (迭代法)
        考虑电压随电流的非线性变化
        
        Args:
            power_command: 功率指令 (W)
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            current: 计算得到的电流 (A)
        """
        # 初始估计
        estimated_voltage = self.state.ocv
        current_estimate = power_command / estimated_voltage if estimated_voltage > 0 else 0.0
        
        for i in range(max_iterations):
            # 计算当前估计电流下的电压
            voltage_estimate = self.calculate_terminal_voltage(current_estimate)
            
            # 计算功率误差
            power_estimate = voltage_estimate * current_estimate
            power_error = power_command - power_estimate
            
            # 检查收敛
            if abs(power_error) < tolerance:
                break
            
            # 更新电流估计 (牛顿法)
            if abs(voltage_estimate) > 1e-6:
                # 计算电压对电流的导数 (数值导数)
                delta_i = 0.01
                v_plus = self.calculate_terminal_voltage(current_estimate + delta_i)
                v_minus = self.calculate_terminal_voltage(current_estimate - delta_i)
                dv_di = (v_plus - v_minus) / (2 * delta_i)
                
                # 功率对电流的导数
                dp_di = voltage_estimate + current_estimate * dv_di
                
                if abs(dp_di) > 1e-6:
                    current_estimate += power_error / dp_di
                else:
                    # 回退到简单方法
                    current_estimate = power_command / voltage_estimate
            else:
                break
        
        return current_estimate
    
    def get_state_vector(self, normalize: bool = True) -> np.ndarray:
        """
        获取状态向量 (用于DRL)
        
        Args:
            normalize: 是否归一化
            
        Returns:
            状态向量
        """
        if normalize:
            # 归一化状态向量
            state_vector = np.array([
                self.state.soc / 100.0,  # SOC [0,1]
                (self.state.voltage - self.params.MIN_VOLTAGE) / 
                (self.params.MAX_VOLTAGE - self.params.MIN_VOLTAGE),  # 电压 [0,1]
                self.state.current / (self.state.capacity_remaining * 
                                    self.params.MAX_DISCHARGE_C_RATE),  # 电流 [-1,1]
                (self.state.temperature - self.params.MIN_TEMP) / 
                (self.params.MAX_TEMP - self.params.MIN_TEMP),  # 温度 [0,1]
                self.state.energy_stored / (self.state.capacity_remaining * 
                                          self.params.NOMINAL_VOLTAGE),  # 能量比例 [0,1]
                self.state.aging_factor,  # 老化因子 [0,1]
                np.tanh(self.get_soc_trend() / 10.0),  # SOC趋势 [-1,1]
                self.state.cycle_count / 1000.0  # 归一化循环次数
            ])
        else:
            # 原始状态向量
            state_vector = np.array([
                self.state.soc,
                self.state.voltage,
                self.state.current,
                self.state.temperature,
                self.state.energy_stored,
                self.state.aging_factor,
                self.get_soc_trend(),
                self.state.cycle_count
            ])
        
        return state_vector
    
    def reset(self, initial_soc: Optional[float] = None,
              initial_temp: Optional[float] = None,
              reset_aging: bool = True,
              random_variation: bool = False) -> Dict:
        """
        重置电池状态
        
        Args:
            initial_soc: 初始SOC (%)
            initial_temp: 初始温度 (℃)
            reset_aging: 是否重置老化状态
            random_variation: 是否添加随机变异
            
        Returns:
            初始状态字典
        """
        # 设置初始SOC
        if initial_soc is not None:
            self.state.soc = np.clip(initial_soc, 0.0, 100.0)
        else:
            self.state.soc = self.params.NOMINAL_SOC
        
        # 设置初始温度
        if initial_temp is not None:
            self.state.temperature = np.clip(initial_temp, 
                                           self.params.MIN_TEMP, 
                                           self.params.MAX_TEMP)
        else:
            self.state.temperature = self.params.NOMINAL_TEMP
        
        # 添加随机变异 (模拟电池个体差异)
        if random_variation:
            soc_variation = np.random.normal(0, 2.0)  # ±2% SOC变异
            temp_variation = np.random.normal(0, 1.0)  # ±1℃ 温度变异
            capacity_variation = np.random.normal(1.0, 0.02)  # ±2% 容量变异
            
            self.state.soc = np.clip(self.state.soc + soc_variation, 0.0, 100.0)
            self.state.temperature = np.clip(self.state.temperature + temp_variation,
                                           self.params.MIN_TEMP, self.params.MAX_TEMP)
            self.state.capacity_remaining = (self.params.CELL_CAPACITY * 
                                           np.clip(capacity_variation, 0.9, 1.1))
        else:
            self.state.capacity_remaining = self.params.CELL_CAPACITY
        
        # 重置其他状态
        self.state.current = 0.0
        self.state.power = 0.0
        
        if reset_aging:
            self.state.cumulative_charge = 0.0
            self.state.cycle_count = 0.0
            self.state.aging_factor = 1.0
            self.state.internal_resistance = self.params.INTERNAL_RESISTANCE
        
        # 重置仿真参数
        self.time_step_count = 0
        self.total_simulation_time = 0.0
        
        # 清空历史
        self.state_history.clear()
        self._soc_trend_window.clear()
        
        # 更新衍生状态
        self._update_derived_states()
        
        # 返回初始状态
        initial_state = {
            'cell_id': self.cell_id,
            'soc': self.state.soc,
            'voltage': self.state.voltage,
            'current': self.state.current,
            'temperature': self.state.temperature,
            'energy_stored': self.state.energy_stored,
            'capacity_remaining': self.state.capacity_remaining,
            'reset_time': self.total_simulation_time
        }
        
        print(f"🔄 电池 {self.cell_id} 已重置: SOC={self.state.soc:.1f}%, T={self.state.temperature:.1f}℃")
        
        return initial_state
    
    def get_diagnostics(self) -> Dict:
        """
        获取诊断信息
        
        Returns:
            诊断数据字典
        """
        if len(self.state_history) == 0:
            return {'error': 'No simulation history available'}
        
        # 提取历史数据
        soc_values = [state['soc'] for state in self.state_history]
        voltage_values = [state['voltage'] for state in self.state_history]
        current_values = [state['current'] for state in self.state_history]
        power_values = [state['actual_power'] for state in self.state_history]
        temp_values = [state['temperature'] for state in self.state_history]
        efficiency_values = [state['power_efficiency'] for state in self.state_history]
        
        # 计算统计信息
        diagnostics = {
            # 基本信息
            'cell_id': self.cell_id,
            'simulation_steps': len(self.state_history),
            'total_time': self.total_simulation_time,
            
            # 状态范围
            'soc_range': (min(soc_values), max(soc_values)),
            'voltage_range': (min(voltage_values), max(voltage_values)),
            'current_range': (min(current_values), max(current_values)),
            'power_range': (min(power_values), max(power_values)),
            'temperature_range': (min(temp_values), max(temp_values)),
            
            # 平均值
            'avg_soc': np.mean(soc_values),
            'avg_voltage': np.mean(voltage_values),
            'avg_temperature': np.mean(temp_values),
            'avg_efficiency': np.mean(efficiency_values),
            
            # 能量统计
            'total_energy_throughput': self.state.cumulative_charge * self.params.NOMINAL_VOLTAGE / 1000,  # kWh
            'equivalent_cycles': self.state.cycle_count,
            'capacity_utilization': (max(soc_values) - min(soc_values)) / 100.0,
            
            # 健康状态
            'capacity_remaining_ratio': self.state.capacity_remaining / self.params.CELL_CAPACITY,
            'aging_factor': self.state.aging_factor,
            'resistance_increase': self.state.internal_resistance / self.params.INTERNAL_RESISTANCE,
            
            # 运行状态
            'current_soc': self.state.soc,
            'soc_trend': self.get_soc_trend(),
            'health_status': self._get_health_status(),
            
            # 性能指标
            'min_efficiency': min(efficiency_values),
            'max_efficiency': max(efficiency_values),
            'voltage_stability': np.std(voltage_values),
            'temperature_stability': np.std(temp_values)
        }
        
        return diagnostics
    
    def _get_health_status(self) -> str:
        """获取健康状态"""
        if self.state.capacity_remaining < self.params.CELL_CAPACITY * 0.8:
            return 'Critical'
        elif self.state.capacity_remaining < self.params.CELL_CAPACITY * 0.9:
            return 'Degraded'
        elif (self.state.soc < 5 or self.state.soc > 95 or
              self.state.temperature < self.params.MIN_TEMP + 5 or
              self.state.temperature > self.params.MAX_TEMP - 5):
            return 'Warning'
        else:
            return 'Normal'
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"BatteryCellModel({self.cell_id}): "
                f"SOC={self.state.soc:.1f}%, "
                f"V={self.state.voltage:.3f}V, "
                f"I={self.state.current:.2f}A, "
                f"T={self.state.temperature:.1f}℃")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"BatteryCellModel(cell_id='{self.cell_id}', "
                f"soc={self.state.soc:.2f}, "
                f"capacity={self.state.capacity_remaining:.1f}Ah, "
                f"cycles={self.state.cycle_count:.2f})")
