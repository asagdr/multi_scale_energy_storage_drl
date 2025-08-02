"""
BMS内均衡器
实现100个单体间的SOC和温度均衡
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class BalancingMode(Enum):
    """均衡模式枚举"""
    PASSIVE = "passive"      # 被动均衡 (放电电阻)
    ACTIVE = "active"        # 主动均衡 (电容/电感)
    HYBRID = "hybrid"        # 混合均衡
    DISABLED = "disabled"    # 禁用均衡

@dataclass
class BalancingResult:
    """均衡结果数据结构"""
    active: bool = False
    mode: BalancingMode = BalancingMode.DISABLED
    
    # 均衡功率
    total_balancing_power: float = 0.0
    cell_balancing_powers: List[float] = None
    
    # 均衡效果
    soc_improvement: float = 0.0      # SOC均衡改善程度
    temp_improvement: float = 0.0     # 温度均衡改善程度
    
    # 均衡状态
    balancing_cells_count: int = 0
    balancing_efficiency: float = 1.0
    estimated_balancing_time: float = 0.0  # 预计均衡时间 (s)

class IntraBMSBalancer:
    """
    BMS内均衡器
    实现100个单体间的智能均衡控制
    """
    
    def __init__(self, 
                 cells: List,
                 balancing_mode: BalancingMode = BalancingMode.ACTIVE,
                 balancer_id: str = "IntraBMSBalancer_001"):
        """
        初始化BMS内均衡器
        
        Args:
            cells: 电池单体列表
            balancing_mode: 均衡模式
            balancer_id: 均衡器ID
        """
        self.cells = cells
        self.balancing_mode = balancing_mode
        self.balancer_id = balancer_id
        self.cells_count = len(cells)
        
        # === 均衡参数 ===
        self.balancing_params = {
            # 启动阈值
            'soc_threshold': 1.0,           # 1% SOC差异启动均衡
            'temp_threshold': 3.0,          # 3℃温差启动热管理
            
            # 均衡功率限制
            'max_balancing_current': 0.5,   # A, 最大均衡电流
            'max_balancing_power_per_cell': 5.0,  # W, 单体最大均衡功率
            
            # 均衡策略
            'target_soc_tolerance': 0.5,    # 0.5% SOC目标容差
            'target_temp_tolerance': 2.0,   # 2℃温度目标容差
            
            # 效率参数
            'passive_efficiency': 0.0,      # 被动均衡效率 (纯消耗)
            'active_efficiency': 0.85,      # 主动均衡效率
            'hybrid_efficiency': 0.75       # 混合均衡效率
        }
        
        # === 均衡状态 ===
        self.is_balancing = False
        self.balancing_start_time = 0.0
        self.total_balancing_time = 0.0
        
        # === 均衡历史 ===
        self.balancing_history: List[BalancingResult] = []
        
        print(f"✅ BMS内均衡器初始化完成: {balancer_id}")
        print(f"   单体数量: {self.cells_count}, 均衡模式: {balancing_mode.value}")
    
    def balance_cells(self, 
                     cell_records: List[Dict], 
                     delta_t: float) -> BalancingResult:
        """
        执行单体均衡
        
        Args:
            cell_records: 单体记录列表
            delta_t: 时间步长 (s)
            
        Returns:
            均衡结果
        """
        
        result = BalancingResult()
        result.mode = self.balancing_mode
        
        if self.balancing_mode == BalancingMode.DISABLED:
            return result
        
        # === 1. 评估均衡需求 ===
        balance_assessment = self._assess_balancing_need(cell_records)
        
        if not balance_assessment['need_balancing']:
            result.active = False
            self.is_balancing = False
            return result
        
        # === 2. 制定均衡策略 ===
        balancing_strategy = self._generate_balancing_strategy(
            cell_records, balance_assessment
        )
        
        # === 3. 执行均衡控制 ===
        balancing_actions = self._execute_balancing(
            balancing_strategy, cell_records, delta_t
        )
        
        # === 4. 计算均衡效果 ===
        result = self._calculate_balancing_result(
            balancing_actions, balance_assessment, delta_t
        )
        
        # === 5. 更新均衡状态 ===
        self._update_balancing_state(result, delta_t)
        
        # === 6. 记录历史 ===
        self.balancing_history.append(result)
        
        return result
    
    def _assess_balancing_need(self, cell_records: List[Dict]) -> Dict:
        """评估均衡需求"""
        
        # 提取单体数据
        soc_values = [cell['soc'] for cell in cell_records]
        temp_values = [cell['temperature'] for cell in cell_records]
        voltage_values = [cell['voltage'] for cell in cell_records]
        
        # 计算统计量
        soc_std = np.std(soc_values)
        soc_range = max(soc_values) - min(soc_values)
        temp_std = np.std(temp_values)
        temp_range = max(temp_values) - min(temp_values)
        voltage_std = np.std(voltage_values)
        
        # 判断均衡需求
        need_soc_balancing = soc_std > self.balancing_params['soc_threshold']
        need_temp_management = temp_std > self.balancing_params['temp_threshold']
        
        assessment = {
            'need_balancing': need_soc_balancing or need_temp_management,
            'need_soc_balancing': need_soc_balancing,
            'need_temp_management': need_temp_management,
            
            'soc_stats': {
                'mean': np.mean(soc_values),
                'std': soc_std,
                'range': soc_range,
                'max_index': np.argmax(soc_values),
                'min_index': np.argmin(soc_values)
            },
            
            'temp_stats': {
                'mean': np.mean(temp_values),
                'std': temp_std,
                'range': temp_range,
                'max_index': np.argmax(temp_values),
                'min_index': np.argmin(temp_values)
            },
            
            'voltage_stats': {
                'mean': np.mean(voltage_values),
                'std': voltage_std
            },
            
            'priority_cells': self._identify_priority_cells(cell_records)
        }
        
        return assessment
    
    def _identify_priority_cells(self, cell_records: List[Dict]) -> Dict:
        """识别优先处理的单体"""
        
        soc_values = [cell['soc'] for cell in cell_records]
        temp_values = [cell['temperature'] for cell in cell_records]
        
        soc_mean = np.mean(soc_values)
        temp_mean = np.mean(temp_values)
        
        priority_cells = {
            'high_soc_cells': [],     # 高SOC单体 (需要放电均衡)
            'low_soc_cells': [],      # 低SOC单体 (需要充电均衡)
            'hot_cells': [],          # 高温单体 (需要冷却)
            'cold_cells': []          # 低温单体 (需要加热)
        }
        
        for i, (soc, temp) in enumerate(zip(soc_values, temp_values)):
            # SOC偏差超过阈值的单体
            if soc > soc_mean + self.balancing_params['soc_threshold']:
                priority_cells['high_soc_cells'].append(i)
            elif soc < soc_mean - self.balancing_params['soc_threshold']:
                priority_cells['low_soc_cells'].append(i)
            
            # 温度偏差超过阈值的单体
            if temp > temp_mean + self.balancing_params['temp_threshold']:
                priority_cells['hot_cells'].append(i)
            elif temp < temp_mean - self.balancing_params['temp_threshold']:
                priority_cells['cold_cells'].append(i)
        
        return priority_cells
    
    def _generate_balancing_strategy(self, 
                                   cell_records: List[Dict], 
                                   assessment: Dict) -> Dict:
        """生成均衡策略"""
        
        strategy = {
            'mode': self.balancing_mode,
            'cell_actions': [],
            'total_estimated_power': 0.0,
            'estimated_duration': 0.0
        }
        
        priority_cells = assessment['priority_cells']
        soc_stats = assessment['soc_stats']
        
        # === SOC均衡策略 ===
        if assessment['need_soc_balancing']:
            
            # 高SOC单体均衡策略
            for cell_index in priority_cells['high_soc_cells']:
                cell_soc = cell_records[cell_index]['soc']
                soc_excess = cell_soc - soc_stats['mean']
                
                # 计算所需均衡功率
                if self.balancing_mode == BalancingMode.PASSIVE:
                    # 被动均衡：通过电阻放电
                    balancing_power = min(
                        self.balancing_params['max_balancing_power_per_cell'],
                        soc_excess * 2.0  # 简化计算
                    )
                    action_type = 'discharge'
                    
                elif self.balancing_mode == BalancingMode.ACTIVE:
                    # 主动均衡：能量转移
                    balancing_power = min(
                        self.balancing_params['max_balancing_power_per_cell'],
                        soc_excess * 1.5
                    )
                    action_type = 'transfer_out'
                
                else:  # HYBRID
                    balancing_power = min(
                        self.balancing_params['max_balancing_power_per_cell'],
                        soc_excess * 1.0
                    )
                    action_type = 'hybrid_balance'
                
                action = {
                    'cell_index': cell_index,
                    'action_type': action_type,
                    'target_power': balancing_power,
                    'target_soc': soc_stats['mean'],
                    'priority': 'high'
                }
                
                strategy['cell_actions'].append(action)
                strategy['total_estimated_power'] += balancing_power
            
            # 低SOC单体均衡策略 (仅主动均衡)
            if self.balancing_mode == BalancingMode.ACTIVE:
                for cell_index in priority_cells['low_soc_cells']:
                    cell_soc = cell_records[cell_index]['soc']
                    soc_deficit = soc_stats['mean'] - cell_soc
                    
                    balancing_power = min(
                        self.balancing_params['max_balancing_power_per_cell'],
                        soc_deficit * 1.5
                    )
                    
                    action = {
                        'cell_index': cell_index,
                        'action_type': 'transfer_in',
                        'target_power': balancing_power,
                        'target_soc': soc_stats['mean'],
                        'priority': 'high'
                    }
                    
                    strategy['cell_actions'].append(action)
                    strategy['total_estimated_power'] += balancing_power
        
        # === 估算均衡时间 ===
        if strategy['total_estimated_power'] > 0:
            # 基于SOC范围和均衡功率估算时间
            soc_range = assessment['soc_stats']['range']
            avg_balancing_power = strategy['total_estimated_power'] / len(strategy['cell_actions']) if strategy['cell_actions'] else 1.0
            
            # 简化时间估算 (实际需要考虑电池容量)
            strategy['estimated_duration'] = (soc_range * 10.0) / (avg_balancing_power / 100.0)  # 简化公式
        
        return strategy
    
    def _execute_balancing(self, 
                          strategy: Dict, 
                          cell_records: List[Dict], 
                          delta_t: float) -> Dict:
        """执行均衡控制"""
        
        balancing_actions = {
            'executed_actions': [],
            'total_power_consumed': 0.0,
            'total_energy_transferred': 0.0,
            'efficiency': 1.0
        }
        
        # 获取效率
        if strategy['mode'] == BalancingMode.PASSIVE:
            efficiency = self.balancing_params['passive_efficiency']
        elif strategy['mode'] == BalancingMode.ACTIVE:
            efficiency = self.balancing_params['active_efficiency']
        else:
            efficiency = self.balancing_params['hybrid_efficiency']
        
        balancing_actions['efficiency'] = efficiency
        
        # 执行各单体均衡动作
        for action in strategy['cell_actions']:
            cell_index = action['cell_index']
            target_power = action['target_power']
            action_type = action['action_type']
            
            # 实际执行的功率 (考虑约束)
            actual_power = min(target_power, 
                             self.balancing_params['max_balancing_power_per_cell'])
            
            # 记录执行的动作
            executed_action = {
                'cell_index': cell_index,
                'action_type': action_type,
                'target_power': target_power,
                'actual_power': actual_power,
                'energy_delta_t': actual_power * delta_t,
                'efficiency': efficiency
            }
            
            balancing_actions['executed_actions'].append(executed_action)
            balancing_actions['total_power_consumed'] += actual_power
            
            # 能量转移计算
            if action_type in ['transfer_in', 'transfer_out']:
                energy_transferred = actual_power * delta_t * efficiency
                balancing_actions['total_energy_transferred'] += energy_transferred
        
        return balancing_actions
    
    def _calculate_balancing_result(self, 
                                  balancing_actions: Dict, 
                                  assessment: Dict, 
                                  delta_t: float) -> BalancingResult:
        """计算均衡结果"""
        
        result = BalancingResult()
        result.mode = self.balancing_mode
        result.active = len(balancing_actions['executed_actions']) > 0
        
        if not result.active:
            return result
        
        # 均衡功率
        result.total_balancing_power = balancing_actions['total_power_consumed']
        result.cell_balancing_powers = [0.0] * self.cells_count
        
        for action in balancing_actions['executed_actions']:
            cell_index = action['cell_index']
            result.cell_balancing_powers[cell_index] = action['actual_power']
        
        # 均衡状态
        result.balancing_cells_count = len(balancing_actions['executed_actions'])
        result.balancing_efficiency = balancing_actions['efficiency']
        
        # 均衡效果评估 (简化计算)
        # 实际应该基于均衡前后的SOC/温度分布变化
        initial_soc_std = assessment['soc_stats']['std']
        initial_temp_std = assessment['temp_stats']['std']
        
        # 估算改善程度 (基于均衡功率和时间)
        balancing_intensity = result.total_balancing_power / self.cells_count
        
        # SOC改善估算
        if initial_soc_std > 0:
            soc_improvement_rate = min(0.1, balancing_intensity * delta_t / 1000.0)  # 简化公式
            result.soc_improvement = soc_improvement_rate * initial_soc_std
        
        # 温度改善估算 (主要通过功率分配优化实现)
        if initial_temp_std > 0:
            temp_improvement_rate = min(0.05, balancing_intensity * delta_t / 2000.0)
            result.temp_improvement = temp_improvement_rate * initial_temp_std
        
        # 估算剩余均衡时间
        remaining_soc_imbalance = initial_soc_std - result.soc_improvement
        if remaining_soc_imbalance > self.balancing_params['target_soc_tolerance'] and balancing_intensity > 0:
            result.estimated_balancing_time = remaining_soc_imbalance * 100.0 / balancing_intensity
        else:
            result.estimated_balancing_time = 0.0
        
        return result
    
    def _update_balancing_state(self, result: BalancingResult, delta_t: float):
        """更新均衡状态"""
        
        if result.active:
            if not self.is_balancing:
                self.balancing_start_time = 0.0  # 重置开始时间
                self.is_balancing = True
            
            self.total_balancing_time += delta_t
        else:
            if self.is_balancing:
                self.is_balancing = False
    
    def get_balancing_status(self) -> Dict:
        """获取均衡状态"""
        
        recent_results = self.balancing_history[-10:] if len(self.balancing_history) >= 10 else self.balancing_history
        
        status = {
            'balancer_id': self.balancer_id,
            'balancing_mode': self.balancing_mode.value,
            'is_active': self.is_balancing,
            'total_balancing_time': self.total_balancing_time,
            'cells_count': self.cells_count,
            
            'recent_performance': {
                'avg_balancing_power': np.mean([r.total_balancing_power for r in recent_results]) if recent_results else 0.0,
                'avg_efficiency': np.mean([r.balancing_efficiency for r in recent_results]) if recent_results else 1.0,
                'avg_soc_improvement': np.mean([r.soc_improvement for r in recent_results]) if recent_results else 0.0,
                'avg_active_cells': np.mean([r.balancing_cells_count for r in recent_results]) if recent_results else 0.0
            },
            
            'balancing_parameters': self.balancing_params.copy()
        }
        
        return status
    
    def update_balancing_mode(self, new_mode: BalancingMode) -> bool:
        """更新均衡模式"""
        try:
            old_mode = self.balancing_mode
            self.balancing_mode = new_mode
            
            print(f"🔄 均衡器 {self.balancer_id} 模式更新: {old_mode.value} -> {new_mode.value}")
            return True
        except Exception as e:
            print(f"❌ 均衡模式更新失败: {str(e)}")
            return False
    
    def reset(self):
        """重置均衡器"""
        self.is_balancing = False
        self.balancing_start_time = 0.0
        self.total_balancing_time = 0.0
        self.balancing_history.clear()
        
        print(f"🔄 BMS内均衡器 {self.balancer_id} 已重置")
    
    def __str__(self) -> str:
        """字符串表示"""
        status = "运行中" if self.is_balancing else "待机"
        return (f"IntraBMSBalancer({self.balancer_id}): "
                f"模式={self.balancing_mode.value}, "
                f"状态={status}, "
                f"单体数={self.cells_count}")
