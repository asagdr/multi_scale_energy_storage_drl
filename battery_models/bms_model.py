"""
单个BMS模型
管理100个电池单体，实现BMS内SOC和温度均衡
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from battery_models.battery_cell_model import BatteryCellModel
from battery_models.thermal_model import ThermalModel
from battery_models.degradation_model import DegradationModel
from battery_models.intra_bms_balancer import IntraBMSBalancer

@dataclass
class BMSState:
    """BMS状态数据结构"""
    bms_id: str
    
    # BMS级汇总状态
    avg_soc: float = 50.0
    soc_std: float = 0.0          # BMS内SOC不平衡度 (关键指标)
    avg_temperature: float = 25.0
    temp_std: float = 0.0         # BMS内温度不平衡度 (关键指标)
    avg_soh: float = 100.0
    
    # 功率状态
    actual_power: float = 0.0
    power_command: float = 0.0
    power_efficiency: float = 1.0
    
    # 均衡状态
    balancing_active: bool = False
    balancing_power: float = 0.0
    
    # 成本状态
    bms_total_cost: float = 0.0
    cost_increase_rate: float = 0.0

class BMSModel:
    """
    单个BMS模型
    管理100个电池单体，实现BMS内均衡和成本计算
    """
    
    def __init__(self, 
                 bms_id: str,
                 cells_count: int,
                 battery_params: BatteryParams):
        """
        初始化BMS模型
        
        Args:
            bms_id: BMS标识
            cells_count: 电池单体数量 (100)
            battery_params: 电池参数
        """
        self.bms_id = bms_id
        self.cells_count = cells_count
        self.battery_params = battery_params
        
        # === 创建电池单体列表 ===
        self.cells: List[BatteryCellModel] = []
        for i in range(cells_count):
            cell = BatteryCellModel(
                cell_id=f"{bms_id}_Cell_{i+1:03d}",
                battery_params=battery_params
            )
            self.cells.append(cell)
        
        # === BMS内均衡器 ===
        self.balancer = IntraBMSBalancer(
            cells=self.cells,
            balancer_id=f"{bms_id}_Balancer"
        )
        
        # === BMS状态 ===
        self.state = BMSState(bms_id=bms_id)
        self.previous_total_cost = 0.0
        
        # === 仿真统计 ===
        self.step_count = 0
        self.total_time = 0.0
        
        print(f"✅ BMS模型初始化完成: {bms_id}, 单体数量: {cells_count}")
    
    def step(self, 
             bms_power_command: float, 
             delta_t: float,
             ambient_temperature: float = 25.0) -> Dict:
        """
        BMS仿真步
        
        Args:
            bms_power_command: BMS功率指令 (W)
            delta_t: 时间步长 (s)
            ambient_temperature: 环境温度 (℃)
            
        Returns:
            BMS仿真记录
        """
        
        # === 1. BMS内功率分配 ===
        cell_power_allocation = self._allocate_power_to_cells(bms_power_command)
        
        # === 2. 单体仿真 ===
        cell_records = []
        for i, cell in enumerate(self.cells):
            cell_power = cell_power_allocation[i]
            cell_record = cell.step(
                power_command=cell_power,
                delta_t=delta_t,
                ambient_temperature=ambient_temperature
            )
            cell_records.append(cell_record)
        
        # === 3. BMS内均衡 ===
        balancing_result = self.balancer.balance_cells(cell_records, delta_t)
        
        # === 4. 更新BMS状态 ===
        self._update_bms_state(cell_records, balancing_result, bms_power_command)
        
        # === 5. 计算BMS成本 ===
        bms_cost = self._calculate_bms_cost(cell_records, balancing_result)
        
        # === 6. 构建BMS记录 ===
        bms_record = {
            'bms_id': self.bms_id,
            'step_count': self.step_count,
            'simulation_time': self.total_time,
            
            # 单体记录
            'cells': cell_records,
            'cell_count': len(cell_records),
            
            # BMS状态 (关键指标)
            'avg_soc': self.state.avg_soc,
            'soc_std': self.state.soc_std,                # BMS内SOC不平衡度
            'avg_temperature': self.state.avg_temperature,
            'temp_std': self.state.temp_std,              # BMS内温度不平衡度
            'avg_soh': self.state.avg_soh,
            
            # 功率状态
            'actual_power': self.state.actual_power,
            'power_command': bms_power_command,
            'power_efficiency': self.state.power_efficiency,
            'power_tracking_error': abs(self.state.actual_power - bms_power_command),
            
            # 均衡状态
            'balancing_active': self.state.balancing_active,
            'balancing_power': self.state.balancing_power,
            'balancing_efficiency': balancing_result.get('efficiency', 1.0),
            
            # 成本状态 (关键指标)
            'bms_total_cost': self.state.bms_total_cost,
            'cost_increase_rate': self.state.cost_increase_rate,
            'cost_breakdown': bms_cost,
            
            # 约束状态
            'max_charge_power': self._get_max_charge_power(),
            'max_discharge_power': self._get_max_discharge_power(),
            'thermal_constraints_active': self._check_thermal_constraints(),
            'voltage_constraints_active': self._check_voltage_constraints(),
            
            # 健康状态
            'health_status': self._calculate_health_status(),
            'warning_flags': self._get_warning_flags(),
            'alarm_flags': self._get_alarm_flags()
        }
        
        # === 7. 更新计数器 ===
        self.step_count += 1
        self.total_time += delta_t
        
        return bms_record
    
    def _allocate_power_to_cells(self, bms_power_command: float) -> List[float]:
        """
        BMS内功率分配 - 基于SOC均衡的智能分配
        
        Args:
            bms_power_command: BMS总功率指令 (W)
            
        Returns:
            各单体功率分配列表
        """
        cell_power_allocation = []
        
        # 获取单体SOC
        soc_values = [cell.soc for cell in self.cells]
        soc_mean = np.mean(soc_values)
        
        # 计算分配权重
        allocation_weights = []
        for soc in soc_values:
            if bms_power_command > 0:  # 充电
                # SOC低的单体获得更多充电功率
                weight = 1.0 + (soc_mean - soc) * 0.02  # 每1%SOC差异对应2%功率差异
            else:  # 放电
                # SOC高的单体提供更多放电功率
                weight = 1.0 + (soc - soc_mean) * 0.02
            
            # 温度约束
            cell_temp = self.cells[soc_values.index(soc)].temperature
            if cell_temp > 45.0:
                weight *= 0.8  # 高温单体减少功率
            elif cell_temp < 10.0:
                weight *= 0.8  # 低温单体减少功率
            
            allocation_weights.append(max(0.1, weight))  # 最小10%权重
        
        # 归一化权重
        total_weight = sum(allocation_weights)
        normalized_weights = [w / total_weight for w in allocation_weights]
        
        # 分配功率
        for i, weight in enumerate(normalized_weights):
            cell_power = bms_power_command * weight
            
            # 单体功率约束
            max_cell_power = self.battery_params.max_charge_power / self.cells_count
            min_cell_power = -self.battery_params.max_discharge_power / self.cells_count
            
            cell_power = np.clip(cell_power, min_cell_power, max_cell_power)
            cell_power_allocation.append(cell_power)
        
        return cell_power_allocation
    
    def _update_bms_state(self, 
                         cell_records: List[Dict], 
                         balancing_result: Dict,
                         bms_power_command: float):
        """更新BMS状态"""
        
        # 提取单体数据
        soc_values = [cell['soc'] for cell in cell_records]
        temp_values = [cell['temperature'] for cell in cell_records]
        soh_values = [cell['soh'] for cell in cell_records]
        power_values = [cell['actual_power'] for cell in cell_records]
        
        # 更新BMS状态
        self.state.avg_soc = float(np.mean(soc_values))
        self.state.soc_std = float(np.std(soc_values))              # 关键指标
        self.state.avg_temperature = float(np.mean(temp_values))
        self.state.temp_std = float(np.std(temp_values))            # 关键指标
        self.state.avg_soh = float(np.mean(soh_values))
        
        self.state.actual_power = float(np.sum(power_values))
        self.state.power_command = bms_power_command
        self.state.power_efficiency = self._calculate_power_efficiency(cell_records)
        
        self.state.balancing_active = balancing_result.get('active', False)
        self.state.balancing_power = balancing_result.get('total_power', 0.0)
    
    def _calculate_bms_cost(self, 
                           cell_records: List[Dict], 
                           balancing_result: Dict) -> Dict:
        """
        计算BMS级成本 - 100个单体成本相加 + 不平衡惩罚
        
        Returns:
            BMS成本详细分解
        """
        
        # === 1. 单体成本线性累加 ===
        cell_costs = [cell.get('degradation_cost', 0.0) for cell in cell_records]
        base_bms_cost = sum(cell_costs)
        
        # === 2. BMS内SOC不平衡惩罚 ===
        soc_imbalance_cost = 0.0
        if self.state.soc_std > 1.0:  # 1%以上SOC不平衡
            penalty_factor = min(1.5, self.state.soc_std / 1.0)
            soc_imbalance_cost = base_bms_cost * (penalty_factor - 1.0) * 0.05  # 最大5%惩罚
        
        # === 3. BMS内温度不平衡惩罚 ===
        temp_imbalance_cost = 0.0
        if self.state.temp_std > 3.0:  # 3℃以上温度不平衡
            penalty_factor = min(1.3, self.state.temp_std / 3.0)
            temp_imbalance_cost = base_bms_cost * (penalty_factor - 1.0) * 0.03  # 最大3%惩罚
        
        # === 4. 均衡功耗成本 ===
        balancing_energy_cost = self.state.balancing_power * 0.001  # 简化的能耗成本
        
        # === 5. BMS总成本 ===
        total_bms_cost = (base_bms_cost + 
                         soc_imbalance_cost + 
                         temp_imbalance_cost + 
                         balancing_energy_cost)
        
        # === 6. 成本增长率计算 ===
        cost_increase = total_bms_cost - self.previous_total_cost
        self.state.cost_increase_rate = cost_increase
        self.state.bms_total_cost = total_bms_cost
        self.previous_total_cost = total_bms_cost
        
        return {
            'base_cost': base_bms_cost,
            'soc_imbalance_cost': soc_imbalance_cost,
            'temp_imbalance_cost': temp_imbalance_cost,
            'balancing_cost': balancing_energy_cost,
            'total_cost': total_bms_cost,
            'cost_increase': cost_increase,
            'cost_per_cell': total_bms_cost / self.cells_count
        }
    
    def _calculate_power_efficiency(self, cell_records: List[Dict]) -> float:
        """计算功率效率"""
        total_input_power = abs(sum(cell.get('power_input', 0.0) for cell in cell_records))
        total_output_power = abs(sum(cell.get('actual_power', 0.0) for cell in cell_records))
        
        if total_input_power > 0:
            return total_output_power / total_input_power
        else:
            return 1.0
    
    def _get_max_charge_power(self) -> float:
        """获取BMS最大充电功率"""
        cell_max_powers = []
        for cell in self.cells:
            cell_max_power = cell.get_power_limits()[0]  # 最大充电功率
            cell_max_powers.append(cell_max_power)
        
        return sum(cell_max_powers)
    
    def _get_max_discharge_power(self) -> float:
        """获取BMS最大放电功率"""
        cell_max_powers = []
        for cell in self.cells:
            cell_max_power = cell.get_power_limits()[1]  # 最大放电功率
            cell_max_powers.append(cell_max_power)
        
        return sum(cell_max_powers)
    
    def _check_thermal_constraints(self) -> bool:
        """检查热约束是否激活"""
        for cell in self.cells:
            if (cell.temperature > self.battery_params.MAX_TEMP - 5 or
                cell.temperature < self.battery_params.MIN_TEMP + 5):
                return True
        return False
    
    def _check_voltage_constraints(self) -> bool:
        """检查电压约束是否激活"""
        for cell in self.cells:
            if (cell.voltage > self.battery_params.MAX_VOLTAGE - 0.1 or
                cell.voltage < self.battery_params.MIN_VOLTAGE + 0.1):
                return True
        return False
    
    def _calculate_health_status(self) -> str:
        """计算BMS健康状态"""
        if self.state.avg_soh < 70:
            return "Critical"
        elif self.state.avg_soh < 80:
            return "Poor"
        elif self.state.avg_soh < 90:
            return "Fair"
        else:
            return "Good"
    
    def _get_warning_flags(self) -> List[str]:
        """获取警告标志"""
        warnings = []
        
        if self.state.soc_std > 2.0:
            warnings.append("SOC_IMBALANCE")
        
        if self.state.temp_std > 5.0:
            warnings.append("TEMP_IMBALANCE")
        
        if self.state.avg_temperature > 50.0:
            warnings.append("HIGH_TEMPERATURE")
        
        if self.state.avg_soc < 10.0 or self.state.avg_soc > 90.0:
            warnings.append("SOC_EXTREME")
        
        return warnings
    
    def _get_alarm_flags(self) -> List[str]:
        """获取报警标志"""
        alarms = []
        
        if self.state.soc_std > 5.0:
            alarms.append("CRITICAL_SOC_IMBALANCE")
        
        if self.state.temp_std > 10.0:
            alarms.append("CRITICAL_TEMP_IMBALANCE")
        
        if self.state.avg_temperature > self.battery_params.MAX_TEMP:
            alarms.append("OVER_TEMPERATURE")
        
        if self.state.avg_soh < 70.0:
            alarms.append("LOW_SOH")
        
        return alarms
    
    def reset(self, 
              target_soc: float = 50.0,
              target_temp: float = 25.0,
              add_variation: bool = True) -> Dict:
        """
        重置BMS
        
        Args:
            target_soc: 目标SOC (%)
            target_temp: 目标温度 (℃)
            add_variation: 是否添加随机变化
            
        Returns:
            重置后的状态
        """
        
        # 重置所有单体
        for i, cell in enumerate(self.cells):
            if add_variation:
                # 添加小幅随机变化模拟现实不一致性
                cell_soc = target_soc + np.random.normal(0, 1.0)  # ±1%变化
                cell_temp = target_temp + np.random.normal(0, 2.0)  # ±2℃变化
            else:
                cell_soc = target_soc
                cell_temp = target_temp
            
            cell.reset(
                initial_soc=np.clip(cell_soc, 5.0, 95.0),
                initial_temp=np.clip(cell_temp, 15.0, 35.0)
            )
        
        # 重置BMS状态
        self.state = BMSState(bms_id=self.bms_id)
        self.previous_total_cost = 0.0
        self.step_count = 0
        self.total_time = 0.0
        
        # 重置均衡器
        self.balancer.reset()
        
        print(f"🔄 BMS {self.bms_id} 已重置: 目标SOC={target_soc:.1f}%, 目标温度={target_temp:.1f}℃")
        
        return {
            'bms_id': self.bms_id,
            'target_soc': target_soc,
            'target_temp': target_temp,
            'cells_count': self.cells_count,
            'reset_complete': True
        }
    
    def get_bms_summary(self) -> Dict:
        """获取BMS摘要信息"""
        return {
            'bms_id': self.bms_id,
            'cells_count': self.cells_count,
            'avg_soc': self.state.avg_soc,
            'soc_std': self.state.soc_std,
            'avg_temperature': self.state.avg_temperature,
            'temp_std': self.state.temp_std,
            'avg_soh': self.state.avg_soh,
            'total_cost': self.state.bms_total_cost,
            'health_status': self._calculate_health_status(),
            'balancing_active': self.state.balancing_active
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"BMS({self.bms_id}): "
                f"SOC={self.state.avg_soc:.1f}±{self.state.soc_std:.2f}%, "
                f"Temp={self.state.avg_temperature:.1f}±{self.state.temp_std:.1f}℃, "
                f"SOH={self.state.avg_soh:.1f}%, "
                f"Cost={self.state.bms_total_cost:.2f}元")
