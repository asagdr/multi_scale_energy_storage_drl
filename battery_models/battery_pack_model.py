"""
电池组模型 - 兼容接口版本
为了向后兼容，保留原有接口，内部使用BMS模型
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig
from battery_models.bms_model import BMSModel

# 保留原有枚举
class PackTopology(Enum):
    """电池组拓扑枚举"""
    SERIES_PARALLEL = "series_parallel"
    PARALLEL_SERIES = "parallel_series"
    MATRIX = "matrix"

class BalancingStrategy(Enum):
    """均衡策略枚举"""
    PASSIVE = "passive"
    ACTIVE = "active"
    HYBRID = "hybrid"
    DISABLED = "disabled"

class BatteryPackModel:
    """
    电池组模型 - 兼容接口
    内部使用单个BMS模型，对外提供原有接口
    """
    
    def __init__(self,
                 battery_params: BatteryParams,
                 system_config: SystemConfig,
                 pack_topology: PackTopology = PackTopology.SERIES_PARALLEL,
                 balancing_strategy: BalancingStrategy = BalancingStrategy.ACTIVE,
                 pack_id: str = "BatteryPack_001"):
        """
        初始化电池组模型
        
        Args:
            battery_params: 电池参数
            system_config: 系统配置
            pack_topology: 电池组拓扑
            balancing_strategy: 均衡策略
            pack_id: 电池组ID
        """
        self.battery_params = battery_params
        self.system_config = system_config
        self.pack_topology = pack_topology
        self.balancing_strategy = balancing_strategy
        self.pack_id = pack_id
        
        # === 核心：使用单个BMS模型代表整个电池组 ===
        self.bms_model = BMSModel(
            bms_id=f"BMS_{pack_id}",
            cells_count=battery_params.total_cells,  # 所有单体
            battery_params=battery_params
        )
        
        # === 兼容性参数 ===
        self.series_num = battery_params.SERIES_NUM
        self.parallel_num = battery_params.PARALLEL_NUM
        self.total_cells = battery_params.total_cells
        
        # === 电池组状态 ===
        self.pack_voltage = 0.0
        self.pack_current = 0.0
        self.pack_power = 0.0
        self.pack_soc = 50.0
        self.pack_temperature = 25.0
        self.pack_soh = 100.0
        
        # === 历史记录 ===
        self.pack_history: List[Dict] = []
        
        print(f"✅ 电池组模型初始化完成: {pack_id} (单BMS兼容模式)")
        print(f"   拓扑: {pack_topology.value}, 均衡: {balancing_strategy.value}")
        print(f"   单体总数: {self.total_cells} ({self.series_num}S{self.parallel_num}P)")
    
    def step(self,
             pack_power_command: float,
             delta_t: float,
             ambient_temperature: float = 25.0,
             enable_balancing: bool = True) -> Dict:
        """
        电池组仿真步 - 兼容接口
        
        Args:
            pack_power_command: 电池组功率指令 (W)
            delta_t: 时间步长 (s)
            ambient_temperature: 环境温度 (℃)
            enable_balancing: 是否启用均衡
            
        Returns:
            电池组仿真记录
        """
        
        # === 调用BMS模型执行仿真 ===
        bms_record = self.bms_model.step(
            bms_power_command=pack_power_command,
            delta_t=delta_t,
            ambient_temperature=ambient_temperature
        )
        
        # === 更新电池组状态 ===
        self._update_pack_state_from_bms(bms_record)
        
        # === 转换为兼容格式 ===
        pack_record = self._convert_bms_to_pack_record(bms_record, pack_power_command, delta_t)
        
        # === 记录历史 ===
        self.pack_history.append(pack_record)
        
        # 维护历史长度
        max_history = getattr(self.system_config, 'MAX_HISTORY_LENGTH', 1000)
        if len(self.pack_history) > max_history:
            self.pack_history.pop(0)
        
        return pack_record
    
    def _update_pack_state_from_bms(self, bms_record: Dict):
        """从BMS记录更新电池组状态"""
        
        self.pack_soc = bms_record['avg_soc']
        self.pack_temperature = bms_record['avg_temperature']
        self.pack_soh = bms_record['avg_soh']
        self.pack_power = bms_record['actual_power']
        
        # 计算电压和电流
        ocv = self.battery_params.get_ocv_from_soc(self.pack_soc)
        self.pack_voltage = ocv * self.series_num
        
        if self.pack_voltage > 0:
            self.pack_current = self.pack_power / self.pack_voltage
        else:
            self.pack_current = 0.0
    
    def _convert_bms_to_pack_record(self, bms_record: Dict, pack_power_command: float, delta_t: float) -> Dict:
        """将BMS记录转换为电池组记录格式"""
        
        pack_record = {
            # === 基础信息 ===
            'pack_id': self.pack_id,
            'topology': self.pack_topology.value,
            'balancing_strategy': self.balancing_strategy.value,
            'timestamp': bms_record.get('step_count', 0),
            'simulation_time': bms_record.get('simulation_time', 0.0),
            'delta_t': delta_t,
            
            # === 电池组状态 ===
            'pack_soc': self.pack_soc,
            'pack_voltage': self.pack_voltage,
            'pack_current': self.pack_current,
            'pack_power': self.pack_power,
            'pack_temperature': self.pack_temperature,
            'pack_soh': self.pack_soh,
            
            # === 功率和控制 ===
            'power_command': pack_power_command,
            'power_tracking_error': abs(self.pack_power - pack_power_command),
            'power_efficiency': bms_record.get('power_efficiency', 1.0),
            
            # === 均衡状态 ===
            'soc_std': bms_record.get('soc_std', 0.0),
            'temp_std': bms_record.get('temp_std', 0.0),
            'soc_range': self._calculate_soc_range(bms_record),
            'temp_range': self._calculate_temp_range(bms_record),
            
            'balancing_active': bms_record.get('balancing_active', False),
            'balancing_power': bms_record.get('balancing_power', 0.0),
            'balancing_efficiency': bms_record.get('balancing_efficiency', 1.0),
            
            # === 成本和劣化 ===
            'degradation_cost': bms_record.get('bms_total_cost', 0.0),
            'cost_increase_rate': bms_record.get('cost_increase_rate', 0.0),
            'cost_breakdown': bms_record.get('cost_breakdown', {}),
            
            # === 约束和安全 ===
            'thermal_constraints_active': bms_record.get('thermal_constraints_active', False),
            'voltage_constraints_active': bms_record.get('voltage_constraints_active', False),
            'safety_status': self._assess_safety_status(bms_record),
            
            # === 健康状态 ===
            'health_status': bms_record.get('health_status', 'Good'),
            'warning_flags': bms_record.get('warning_flags', []),
            'alarm_flags': bms_record.get('alarm_flags', []),
            
            # === 配置信息 ===
            'series_num': self.series_num,
            'parallel_num': self.parallel_num,
            'total_cells': self.total_cells,
            
            # === 扩展信息（保留BMS数据） ===
            'bms_data': bms_record,
            'cell_count': bms_record.get('cell_count', self.total_cells)
        }
        
        return pack_record
    
    def _calculate_soc_range(self, bms_record: Dict) -> float:
        """计算SOC极差"""
        cells = bms_record.get('cells', [])
        if not cells:
            return 0.0
        
        soc_values = [cell.get('soc', 50.0) for cell in cells]
        return max(soc_values) - min(soc_values)
    
    def _calculate_temp_range(self, bms_record: Dict) -> float:
        """计算温度极差"""
        cells = bms_record.get('cells', [])
        if not cells:
            return 0.0
        
        temp_values = [cell.get('temperature', 25.0) for cell in cells]
        return max(temp_values) - min(temp_values)
    
    def _assess_safety_status(self, bms_record: Dict) -> str:
        """评估安全状态"""
        
        warning_count = len(bms_record.get('warning_flags', []))
        alarm_count = len(bms_record.get('alarm_flags', []))
        
        if alarm_count > 0:
            return "Critical"
        elif warning_count > 2:
            return "Warning"
        elif warning_count > 0:
            return "Caution"
        else:
            return "Normal"
    
    def get_cell_states(self) -> List[Dict]:
        """
        获取所有单体状态 - 兼容接口
        
        Returns:
            单体状态列表
        """
        
        # 从BMS模型获取单体状态
        if hasattr(self.bms_model, 'cells') and self.bms_model.cells:
            cell_states = []
            for i, cell in enumerate(self.bms_model.cells):
                cell_state = {
                    'cell_id': f"Cell_{i+1:03d}",
                    'soc': getattr(cell, 'soc', 50.0),
                    'voltage': getattr(cell, 'voltage', 3.2),
                    'temperature': getattr(cell, 'temperature', 25.0),
                    'current': getattr(cell, 'current', 0.0),
                    'soh': getattr(cell, 'soh', 100.0),
                    'degradation_cost': getattr(cell, 'degradation_cost', 0.0),
                    'balancing_active': getattr(cell, 'balancing_active', False)
                }
                cell_states.append(cell_state)
            
            return cell_states
        else:
            # 如果没有单体数据，生成模拟数据
            return self._generate_simulated_cell_states()
    
    def _generate_simulated_cell_states(self) -> List[Dict]:
        """生成模拟的单体状态"""
        
        cell_states = []
        base_soc = self.pack_soc
        base_temp = self.pack_temperature
        base_soh = self.pack_soh
        
        for i in range(self.total_cells):
            # 添加小幅随机变化
            cell_soc = base_soc + np.random.normal(0, 1.0)
            cell_temp = base_temp + np.random.normal(0, 2.0)
            cell_soh = base_soh + np.random.normal(0, 1.0)
            
            cell_state = {
                'cell_id': f"Cell_{i+1:03d}",
                'soc': np.clip(cell_soc, 0.0, 100.0),
                'voltage': self.battery_params.get_ocv_from_soc(cell_soc),
                'temperature': np.clip(cell_temp, -20.0, 60.0),
                'current': self.pack_current / self.parallel_num,
                'soh': np.clip(cell_soh, 50.0, 100.0),
                'degradation_cost': 0.01,
                'balancing_active': False
            }
            cell_states.append(cell_state)
        
        return cell_states
    
    def get_pack_state_vector(self, normalize: bool = True) -> np.ndarray:
        """
        获取电池组状态向量 - 兼容接口
        
        Args:
            normalize: 是否归一化
            
        Returns:
            状态向量
        """
        
        if not self.pack_history:
            # 返回默认状态
            return np.array([
                0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)
        
        latest_record = self.pack_history[-1]
        
        # 构建状态向量
        state_vector = np.array([
            latest_record['pack_soc'] / 100.0 if normalize else latest_record['pack_soc'],
            (latest_record['pack_temperature'] - 15.0) / 30.0 if normalize else latest_record['pack_temperature'],
            latest_record['soc_std'] / 10.0 if normalize else latest_record['soc_std'],
            latest_record['temp_std'] / 15.0 if normalize else latest_record['temp_std'],
            latest_record['pack_soh'] / 100.0 if normalize else latest_record['pack_soh'],
            abs(latest_record['pack_power']) / self.battery_params.max_discharge_power if normalize else latest_record['pack_power'],
            latest_record['power_efficiency'] if normalize else latest_record['power_efficiency'],
            1.0 if latest_record['thermal_constraints_active'] else 0.0,
            1.0 if latest_record['voltage_constraints_active'] else 0.0,
            latest_record['power_tracking_error'] / 1000.0 if normalize else latest_record['power_tracking_error'],
            1.0 if latest_record['balancing_active'] else 0.0,
            latest_record['balancing_power'] / 1000.0 if normalize else latest_record['balancing_power'],
            latest_record['degradation_cost'] / 100.0 if normalize else latest_record['degradation_cost'],
            latest_record['cost_increase_rate'] if normalize else latest_record['cost_increase_rate']
        ], dtype=np.float32)
        
        if normalize:
            state_vector = np.clip(state_vector, 0.0, 1.0)
        
        return state_vector
    
    def get_balance_metrics(self) -> Dict[str, float]:
        """
        获取均衡指标 - 兼容接口
        
        Returns:
            均衡指标字典
        """
        
        if not self.pack_history:
            return {
                'soc_std': 0.0,
                'temp_std': 0.0,
                'soc_range': 0.0,
                'temp_range': 0.0,
                'balance_score': 1.0,
                'balancing_efficiency': 1.0
            }
        
        latest_record = self.pack_history[-1]
        
        # 计算均衡评分
        soc_balance_score = max(0.0, 1.0 - latest_record['soc_std'] / 5.0)
        temp_balance_score = max(0.0, 1.0 - latest_record['temp_std'] / 10.0)
        overall_balance_score = 0.7 * soc_balance_score + 0.3 * temp_balance_score
        
        return {
            'soc_std': latest_record['soc_std'],
            'temp_std': latest_record['temp_std'],
            'soc_range': latest_record['soc_range'],
            'temp_range': latest_record['temp_range'],
            'balance_score': overall_balance_score,
            'balancing_efficiency': latest_record.get('balancing_efficiency', 1.0),
            'soc_balance_score': soc_balance_score,
            'temp_balance_score': temp_balance_score
        }
    
    def get_degradation_metrics(self) -> Dict[str, float]:
        """
        获取劣化指标 - 兼容接口
        
        Returns:
            劣化指标字典
        """
        
        if not self.pack_history:
            return {
                'total_cost': 0.0,
                'cost_rate': 0.0,
                'avg_soh': 100.0,
                'soh_std': 0.0,
                'lifetime_remaining': 1.0
            }
        
        latest_record = self.pack_history[-1]
        
        # 计算剩余寿命估算
        current_soh = latest_record['pack_soh']
        eol_threshold = self.battery_params.EOL_CAPACITY
        lifetime_remaining = max(0.0, (current_soh - eol_threshold) / (100.0 - eol_threshold))
        
        return {
            'total_cost': latest_record['degradation_cost'],
            'cost_rate': latest_record['cost_increase_rate'],
            'avg_soh': current_soh,
            'soh_std': 0.0,  # 单个BMS模式下SOH标准差为0
            'lifetime_remaining': lifetime_remaining,
            'eol_threshold': eol_threshold
        }
    
    def set_balancing_strategy(self, new_strategy: BalancingStrategy) -> bool:
        """
        设置均衡策略 - 兼容接口
        
        Args:
            new_strategy: 新的均衡策略
            
        Returns:
            是否成功设置
        """
        try:
            old_strategy = self.balancing_strategy
            self.balancing_strategy = new_strategy
            
            # 如果BMS模型有均衡器，更新其策略
            if hasattr(self.bms_model, 'balancer'):
                from battery_models.intra_bms_balancer import BalancingMode
                
                # 映射均衡策略
                strategy_mapping = {
                    BalancingStrategy.PASSIVE: BalancingMode.PASSIVE,
                    BalancingStrategy.ACTIVE: BalancingMode.ACTIVE,
                    BalancingStrategy.HYBRID: BalancingMode.HYBRID,
                    BalancingStrategy.DISABLED: BalancingMode.DISABLED
                }
                
                bms_mode = strategy_mapping.get(new_strategy, BalancingMode.ACTIVE)
                self.bms_model.balancer.update_balancing_mode(bms_mode)
            
            print(f"🔄 电池组 {self.pack_id} 均衡策略更新: {old_strategy.value} -> {new_strategy.value}")
            return True
            
        except Exception as e:
            print(f"❌ 均衡策略更新失败: {str(e)}")
            return False
    
    def reset(self, 
              target_soc: float = 50.0,
              target_temp: float = 25.0,
              random_variation: bool = False,
              reset_degradation: bool = False) -> Dict:
        """
        重置电池组模型 - 兼容接口
        
        Args:
            target_soc: 目标SOC (%)
            target_temp: 目标温度 (℃)
            random_variation: 是否添加随机变化
            reset_degradation: 是否重置劣化状态
            
        Returns:
            重置结果
        """
        
        # 重置BMS模型
        bms_reset_result = self.bms_model.reset(
            target_soc=target_soc,
            target_temp=target_temp,
            add_variation=random_variation
        )
        
        # 重置电池组状态
        self.pack_soc = target_soc
        self.pack_temperature = target_temp
        self.pack_soh = 100.0 if reset_degradation else self.pack_soh
        self.pack_power = 0.0
        self.pack_current = 0.0
        
        # 重新计算电压
        ocv = self.battery_params.get_ocv_from_soc(self.pack_soc)
        self.pack_voltage = ocv * self.series_num
        
        # 清空历史
        self.pack_history.clear()
        
        reset_result = {
            'pack_id': self.pack_id,
            'reset_complete': True,
            'target_soc': target_soc,
            'target_temp': target_temp,
            'random_variation': random_variation,
            'reset_degradation': reset_degradation,
            'bms_reset_result': bms_reset_result
        }
        
        print(f"🔄 电池组模型 {self.pack_id} 已重置")
        
        return reset_result
    
    def get_pack_summary(self) -> Dict:
        """获取电池组摘要 - 兼容接口"""
        
        bms_summary = self.bms_model.get_bms_summary()
        
        pack_summary = {
            'pack_id': self.pack_id,
            'topology': self.pack_topology.value,
            'balancing_strategy': self.balancing_strategy.value,
            'configuration': {
                'series_num': self.series_num,
                'parallel_num': self.parallel_num,
                'total_cells': self.total_cells
            },
            'current_state': {
                'pack_soc': self.pack_soc,
                'pack_voltage': self.pack_voltage,
                'pack_current': self.pack_current,
                'pack_power': self.pack_power,
                'pack_temperature': self.pack_temperature,
                'pack_soh': self.pack_soh
            },
            'balance_metrics': self.get_balance_metrics(),
            'degradation_metrics': self.get_degradation_metrics(),
            'bms_summary': bms_summary,
            'simulation_steps': len(self.pack_history)
        }
        
        return pack_summary
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"BatteryPackModel({self.pack_id}): "
                f"拓扑={self.pack_topology.value}, "
                f"均衡={self.balancing_strategy.value}, "
                f"配置={self.series_num}S{self.parallel_num}P, "
                f"SOC={self.pack_soc:.1f}%, "
                f"温度={self.pack_temperature:.1f}℃")
