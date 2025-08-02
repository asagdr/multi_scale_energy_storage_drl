"""
多层级成本模型
正确计算单体级、BMS级、系统级的协同劣化成本
解决简单相加的问题
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams

@dataclass
class CostBreakdown:
    """成本分解数据结构"""
    # 单体级成本 (线性累加基础)
    total_cell_cost: float = 0.0
    
    # BMS级不平衡惩罚
    bms_soc_imbalance_cost: float = 0.0
    bms_temp_imbalance_cost: float = 0.0
    bms_balancing_cost: float = 0.0
    
    # 系统级协同效应
    inter_bms_imbalance_penalty: float = 0.0
    system_coordination_penalty: float = 0.0
    bottleneck_penalty: float = 0.0  # 木桶效应惩罚
    
    # 总成本
    total_system_cost: float = 0.0

class MultiLevelCostModel:
    """
    多层级劣化成本模型
    正确处理单体、BMS、系统三级成本的非线性关系
    """
    
    def __init__(self, 
                 bms_list: List,
                 battery_params: BatteryParams,
                 cost_model_id: str = "MultiLevelCostModel_001"):
        """
        初始化多层级成本模型
        
        Args:
            bms_list: BMS列表
            battery_params: 电池参数
            cost_model_id: 成本模型标识
        """
        self.bms_list = bms_list
        self.battery_params = battery_params
        self.cost_model_id = cost_model_id
        
        # === 成本模型参数 ===
        self.cost_params = {
            # BMS级不平衡惩罚系数
            'bms_soc_imbalance_factor': 0.05,      # 每1%SOC不平衡增加5%成本
            'bms_temp_imbalance_factor': 0.03,     # 每1℃温度不平衡增加3%成本
            
            # 系统级协同惩罚系数
            'inter_bms_soc_penalty_factor': 0.08,  # BMS间SOC不平衡惩罚
            'inter_bms_temp_penalty_factor': 0.05, # BMS间温度不平衡惩罚
            'system_coordination_factor': 0.10,    # 系统协调效应
            
            # 木桶效应系数
            'bottleneck_threshold_soh': 80.0,      # SOH阈值
            'bottleneck_penalty_factor': 0.20,     # 木桶效应惩罚系数
            
            # 替换策略系数
            'replacement_threshold_soh': 70.0,     # 替换阈值SOH
            'replacement_cost_factor': 0.30        # 整体替换成本增加
        }
        
        # === 历史成本记录 ===
        self.cost_history: List[CostBreakdown] = []
        self.previous_total_cost = 0.0
        
        print(f"✅ 多层级成本模型初始化完成: {cost_model_id}")
    
    def calculate_total_system_cost(self, bms_records: List[Dict]) -> Dict[str, float]:
        """
        计算总体系统成本
        考虑单体级、BMS级、系统级的协同效应
        
        Args:
            bms_records: BMS记录列表
            
        Returns:
            详细成本分解字典
        """
        
        cost_breakdown = CostBreakdown()
        
        # === 1. 单体级成本累加 (基础成本) ===
        cost_breakdown.total_cell_cost = self._calculate_cell_level_cost(bms_records)
        
        # === 2. BMS级不平衡惩罚 ===
        bms_penalties = self._calculate_bms_level_penalties(bms_records)
        cost_breakdown.bms_soc_imbalance_cost = bms_penalties['soc_imbalance']
        cost_breakdown.bms_temp_imbalance_cost = bms_penalties['temp_imbalance']
        cost_breakdown.bms_balancing_cost = bms_penalties['balancing_cost']
        
        # === 3. 系统级协同效应 ===
        system_penalties = self._calculate_system_level_penalties(bms_records)
        cost_breakdown.inter_bms_imbalance_penalty = system_penalties['inter_bms_imbalance']
        cost_breakdown.system_coordination_penalty = system_penalties['coordination_penalty']
        cost_breakdown.bottleneck_penalty = system_penalties['bottleneck_penalty']
        
        # === 4. 总成本计算 ===
        cost_breakdown.total_system_cost = (
            cost_breakdown.total_cell_cost +
            cost_breakdown.bms_soc_imbalance_cost +
            cost_breakdown.bms_temp_imbalance_cost +
            cost_breakdown.bms_balancing_cost +
            cost_breakdown.inter_bms_imbalance_penalty +
            cost_breakdown.system_coordination_penalty +
            cost_breakdown.bottleneck_penalty
        )
        
        # === 5. 记录历史 ===
        self.cost_history.append(cost_breakdown)
        
        # === 6. 计算成本增长率 ===
        cost_increase = cost_breakdown.total_system_cost - self.previous_total_cost
        cost_increase_rate = cost_increase
        self.previous_total_cost = cost_breakdown.total_system_cost
        
        # === 7. 构建返回字典 ===
        return {
            # 基础成本
            'total_cell_cost': cost_breakdown.total_cell_cost,
            
            # BMS级惩罚
            'bms_soc_imbalance_cost': cost_breakdown.bms_soc_imbalance_cost,
            'bms_temp_imbalance_cost': cost_breakdown.bms_temp_imbalance_cost,
            'bms_balancing_cost': cost_breakdown.bms_balancing_cost,
            'total_bms_penalty': (cost_breakdown.bms_soc_imbalance_cost + 
                                cost_breakdown.bms_temp_imbalance_cost + 
                                cost_breakdown.bms_balancing_cost),
            
            # 系统级惩罚
            'inter_bms_imbalance_penalty': cost_breakdown.inter_bms_imbalance_penalty,
            'system_coordination_penalty': cost_breakdown.system_coordination_penalty,
            'bottleneck_penalty': cost_breakdown.bottleneck_penalty,
            'total_system_penalty': (cost_breakdown.inter_bms_imbalance_penalty + 
                                   cost_breakdown.system_coordination_penalty + 
                                   cost_breakdown.bottleneck_penalty),
            
            # 总成本
            'total_system_cost': cost_breakdown.total_system_cost,
            'cost_increase': cost_increase,
            'cost_increase_rate': cost_increase_rate,
            
            # 成本占比分析
            'cell_cost_ratio': cost_breakdown.total_cell_cost / cost_breakdown.total_system_cost if cost_breakdown.total_system_cost > 0 else 0,
            'bms_penalty_ratio': (cost_breakdown.bms_soc_imbalance_cost + cost_breakdown.bms_temp_imbalance_cost + cost_breakdown.bms_balancing_cost) / cost_breakdown.total_system_cost if cost_breakdown.total_system_cost > 0 else 0,
            'system_penalty_ratio': (cost_breakdown.inter_bms_imbalance_penalty + cost_breakdown.system_coordination_penalty + cost_breakdown.bottleneck_penalty) / cost_breakdown.total_system_cost if cost_breakdown.total_system_cost > 0 else 0
        }
    
    def _calculate_cell_level_cost(self, bms_records: List[Dict]) -> float:
        """计算单体级成本 (线性累加)"""
        total_cell_cost = 0.0
        
        for bms_record in bms_records:
            # 每个BMS的成本已经是其100个单体成本的累加
            bms_base_cost = bms_record.get('cost_breakdown', {}).get('base_cost', 0.0)
            total_cell_cost += bms_base_cost
        
        return total_cell_cost
    
    def _calculate_bms_level_penalties(self, bms_records: List[Dict]) -> Dict[str, float]:
        """计算BMS级不平衡惩罚"""
        penalties = {
            'soc_imbalance': 0.0,
            'temp_imbalance': 0.0,
            'balancing_cost': 0.0
        }
        
        for bms_record in bms_records:
            bms_base_cost = bms_record.get('cost_breakdown', {}).get('base_cost', 0.0)
            
            # SOC不平衡惩罚
            soc_std = bms_record.get('soc_std', 0.0)
            if soc_std > 1.0:  # 1%以上不平衡
                soc_penalty_factor = min(1.5, soc_std / 1.0)  # 最大1.5倍惩罚
                soc_penalty = bms_base_cost * (soc_penalty_factor - 1.0) * self.cost_params['bms_soc_imbalance_factor']
                penalties['soc_imbalance'] += soc_penalty
            
            # 温度不平衡惩罚
            temp_std = bms_record.get('temp_std', 0.0)
            if temp_std > 3.0:  # 3℃以上不平衡
                temp_penalty_factor = min(1.3, temp_std / 3.0)  # 最大1.3倍惩罚
                temp_penalty = bms_base_cost * (temp_penalty_factor - 1.0) * self.cost_params['bms_temp_imbalance_factor']
                penalties['temp_imbalance'] += temp_penalty
            
            # 均衡功耗成本
            balancing_power = bms_record.get('balancing_power', 0.0)
            balancing_cost = balancing_power * 0.001  # 简化的能耗成本计算
            penalties['balancing_cost'] += balancing_cost
        
        return penalties
    
    def _calculate_system_level_penalties(self, bms_records: List[Dict]) -> Dict[str, float]:
        """计算系统级协同效应惩罚"""
        penalties = {
            'inter_bms_imbalance': 0.0,
            'coordination_penalty': 0.0,
            'bottleneck_penalty': 0.0
        }
        
        # 提取BMS级数据
        bms_socs = [record['avg_soc'] for record in bms_records]
        bms_temps = [record['avg_temperature'] for record in bms_records]
        bms_sohs = [record['avg_soh'] for record in bms_records]
        bms_costs = [record.get('cost_breakdown', {}).get('base_cost', 0.0) for record in bms_records]
        
        total_base_cost = sum(bms_costs)
        
        # === 1. BMS间不平衡惩罚 ===
        inter_bms_soc_std = np.std(bms_socs)
        inter_bms_temp_std = np.std(bms_temps)
        
        # SOC不平衡惩罚
        if inter_bms_soc_std > 5.0:  # BMS间SOC差异超过5%
            soc_penalty_factor = min(2.0, inter_bms_soc_std / 5.0)  # 最大2倍惩罚
            soc_penalty = total_base_cost * (soc_penalty_factor - 1.0) * self.cost_params['inter_bms_soc_penalty_factor']
            penalties['inter_bms_imbalance'] += soc_penalty
        
        # 温度不平衡惩罚
        if inter_bms_temp_std > 10.0:  # BMS间温差超过10℃
            temp_penalty_factor = min(1.5, inter_bms_temp_std / 10.0)  # 最大1.5倍惩罚
            temp_penalty = total_base_cost * (temp_penalty_factor - 1.0) * self.cost_params['inter_bms_temp_penalty_factor']
            penalties['inter_bms_imbalance'] += temp_penalty
        
        # === 2. 系统协调效应惩罚 ===
        # 基于BMS间的相互影响
        coordination_penalty = self._calculate_coordination_penalty(bms_records, total_base_cost)
        penalties['coordination_penalty'] = coordination_penalty
        
        # === 3. 木桶效应惩罚 (最差BMS决定系统寿命) ===
        min_soh = min(bms_sohs)
        bottleneck_threshold = self.cost_params['bottleneck_threshold_soh']
        
        if min_soh < bottleneck_threshold:
            # 最差BMS健康度低于阈值，触发木桶效应
            bottleneck_factor = (bottleneck_threshold - min_soh) / bottleneck_threshold
            bottleneck_penalty = total_base_cost * bottleneck_factor * self.cost_params['bottleneck_penalty_factor']
            penalties['bottleneck_penalty'] = bottleneck_penalty
            
            # 如果需要整体替换
            replacement_threshold = self.cost_params['replacement_threshold_soh']
            if min_soh < replacement_threshold:
                replacement_penalty = total_base_cost * self.cost_params['replacement_cost_factor']
                penalties['bottleneck_penalty'] += replacement_penalty
        
        return penalties
    
    def _calculate_coordination_penalty(self, bms_records: List[Dict], total_base_cost: float) -> float:
        """计算系统协调效应惩罚"""
        
        # 计算BMS间的相互影响
        coordination_penalty = 0.0
        
        # 1. 负载不均衡导致的加速老化
        bms_powers = [abs(record['actual_power']) for record in bms_records]
        power_std = np.std(bms_powers)
        power_mean = np.mean(bms_powers)
        
        if power_mean > 0:
            power_cv = power_std / power_mean  # 变异系数
            if power_cv > 0.2:  # 20%以上功率不均衡
                load_imbalance_penalty = total_base_cost * power_cv * 0.05
                coordination_penalty += load_imbalance_penalty
        
        # 2. 热耦合效应
        thermal_coupling_penalty = self._calculate_thermal_coupling_penalty(bms_records, total_base_cost)
        coordination_penalty += thermal_coupling_penalty
        
        # 3. 电气耦合效应
        electrical_coupling_penalty = self._calculate_electrical_coupling_penalty(bms_records, total_base_cost)
        coordination_penalty += electrical_coupling_penalty
        
        return coordination_penalty
    
    def _calculate_thermal_coupling_penalty(self, bms_records: List[Dict], total_base_cost: float) -> float:
        """计算热耦合效应惩罚"""
        
        bms_temps = [record['avg_temperature'] for record in bms_records]
        temp_max = max(bms_temps)
        temp_min = min(bms_temps)
        temp_range = temp_max - temp_min
        
        # 温差过大导致的热耦合效应
        if temp_range > 15.0:  # 15℃以上温差
            coupling_factor = min(1.2, temp_range / 15.0)
            thermal_penalty = total_base_cost * (coupling_factor - 1.0) * 0.03
            return thermal_penalty
        
        return 0.0
    
    def _calculate_electrical_coupling_penalty(self, bms_records: List[Dict], total_base_cost: float) -> float:
        """计算电气耦合效应惩罚"""
        
        # 基于SOC差异导致的环流和电气应力
        bms_socs = [record['avg_soc'] for record in bms_records]
        soc_range = max(bms_socs) - min(bms_socs)
        
        if soc_range > 10.0:  # 10%以上SOC差异
            coupling_factor = min(1.3, soc_range / 10.0)
            electrical_penalty = total_base_cost * (coupling_factor - 1.0) * 0.02
            return electrical_penalty
        
        return 0.0
    
    def get_cost_trends(self, window_size: int = 50) -> Dict[str, float]:
        """获取成本趋势分析"""
        
        if len(self.cost_history) < window_size:
            return {'error': 'Insufficient cost history'}
        
        recent_costs = self.cost_history[-window_size:]
        
        # 提取各类成本
        cell_costs = [cost.total_cell_cost for cost in recent_costs]
        bms_penalties = [cost.bms_soc_imbalance_cost + cost.bms_temp_imbalance_cost + cost.bms_balancing_cost for cost in recent_costs]
        system_penalties = [cost.inter_bms_imbalance_penalty + cost.system_coordination_penalty + cost.bottleneck_penalty for cost in recent_costs]
        total_costs = [cost.total_system_cost for cost in recent_costs]
        
        return {
            'cell_cost_trend': self._calculate_trend(cell_costs),
            'bms_penalty_trend': self._calculate_trend(bms_penalties),
            'system_penalty_trend': self._calculate_trend(system_penalties),
            'total_cost_trend': self._calculate_trend(total_costs),
            'avg_cost_increase_rate': np.mean(np.diff(total_costs)),
            'cost_volatility': np.std(total_costs),
            'latest_total_cost': total_costs[-1]
        }
    
    def _calculate_trend(self, data: List[float]) -> str:
        """计算趋势方向"""
        if len(data) < 2:
            return "insufficient_data"
        
        # 简单线性趋势
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        slope = coeffs[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def reset(self):
        """重置成本模型"""
        self.cost_history.clear()
        self.previous_total_cost = 0.0
        print(f"🔄 多层级成本模型 {self.cost_model_id} 已重置")
    
    def get_cost_model_summary(self) -> Dict:
        """获取成本模型摘要"""
        
        if not self.cost_history:
            return {'error': 'No cost history available'}
        
        latest_cost = self.cost_history[-1]
        
        return {
            'cost_model_id': self.cost_model_id,
            'total_calculations': len(self.cost_history),
            
            'latest_breakdown': {
                'cell_cost': latest_cost.total_cell_cost,
                'bms_penalties': (latest_cost.bms_soc_imbalance_cost + 
                                latest_cost.bms_temp_imbalance_cost + 
                                latest_cost.bms_balancing_cost),
                'system_penalties': (latest_cost.inter_bms_imbalance_penalty + 
                                   latest_cost.system_coordination_penalty + 
                                   latest_cost.bottleneck_penalty),
                'total_cost': latest_cost.total_system_cost
            },
            
            'cost_composition': {
                'cell_cost_ratio': latest_cost.total_cell_cost / latest_cost.total_system_cost if latest_cost.total_system_cost > 0 else 0,
                'bms_penalty_ratio': (latest_cost.bms_soc_imbalance_cost + latest_cost.bms_temp_imbalance_cost + latest_cost.bms_balancing_cost) / latest_cost.total_system_cost if latest_cost.total_system_cost > 0 else 0,
                'system_penalty_ratio': (latest_cost.inter_bms_imbalance_penalty + latest_cost.system_coordination_penalty + latest_cost.bottleneck_penalty) / latest_cost.total_system_cost if latest_cost.total_system_cost > 0 else 0
            }
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        if self.cost_history:
            latest_cost = self.cost_history[-1].total_system_cost
            return f"MultiLevelCostModel({self.cost_model_id}): 最新成本={latest_cost:.2f}元"
        else:
            return f"MultiLevelCostModel({self.cost_model_id}): 未计算"
