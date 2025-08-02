"""
中央功率分配器
智能分配系统级功率指令到10个BMS
基于SOC均衡、温度均衡、寿命优化的多目标分配
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class AllocationStrategy(Enum):
    """分配策略枚举"""
    EQUAL_POWER = "equal_power"                    # 均等功率分配
    SOC_WEIGHTED = "soc_weighted"                  # SOC加权分配
    MULTI_OBJECTIVE = "multi_objective"            # 多目标优化分配
    ADAPTIVE = "adaptive"                          # 自适应分配

@dataclass
class AllocationWeights:
    """分配权重数据结构"""
    soc_balance_weight: float = 0.3               # SOC均衡权重
    temp_balance_weight: float = 0.2              # 温度均衡权重
    lifetime_weight: float = 0.3                  # 寿命优化权重
    efficiency_weight: float = 0.2                # 效率权重

@dataclass
class AllocationConstraints:
    """分配约束数据结构"""
    min_power_ratio: float = 0.0                  # 最小功率比例
    max_power_ratio: float = 1.0                  # 最大功率比例
    power_balance_tolerance: float = 0.01         # 功率平衡容差
    constraint_violation_penalty: float = 10.0    # 约束违反惩罚

class CentralPowerAllocator:
    """
    中央功率分配器
    实现系统级功率到BMS级的智能分配
    """
    
    def __init__(self, 
                 bms_list: List,
                 allocation_strategy: AllocationStrategy = AllocationStrategy.MULTI_OBJECTIVE,
                 allocator_id: str = "CentralPowerAllocator_001"):
        """
        初始化中央功率分配器
        
        Args:
            bms_list: BMS列表
            allocation_strategy: 分配策略
            allocator_id: 分配器ID
        """
        self.bms_list = bms_list
        self.num_bms = len(bms_list)
        self.allocation_strategy = allocation_strategy
        self.allocator_id = allocator_id
        
        # === 分配参数 ===
        self.default_weights = AllocationWeights()
        self.constraints = AllocationConstraints()
        
        # === 分配历史 ===
        self.allocation_history: List[Dict] = []
        
        # === 自适应参数 ===
        self.adaptation_enabled = True
        self.adaptation_rate = 0.05
        self.performance_window = 20  # 性能评估窗口
        
        # === 统计信息 ===
        self.total_allocations = 0
        self.allocation_efficiency_history: List[float] = []
        
        print(f"✅ 中央功率分配器初始化完成: {allocator_id}")
        print(f"   BMS数量: {self.num_bms}, 分配策略: {allocation_strategy.value}")
    
    def allocate_power(self, 
                      total_power_command: float,
                      upper_layer_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        执行功率分配
        
        Args:
            total_power_command: 系统总功率指令 (W)
            upper_layer_weights: 上层权重指令
            
        Returns:
            各BMS功率分配字典 {"BMS_01": 150000.0, "BMS_02": 120000.0, ...}
        """
        
        # === 1. 准备权重 ===
        allocation_weights = self._prepare_allocation_weights(upper_layer_weights)
        
        # === 2. 收集BMS状态 ===
        bms_states = self._collect_bms_states()
        
        # === 3. 计算功率约束 ===
        power_constraints = self._calculate_power_constraints(bms_states)
        
        # === 4. 执行分配算法 ===
        power_allocation = self._execute_allocation_algorithm(
            total_power_command, 
            allocation_weights, 
            bms_states, 
            power_constraints
        )
        
        # === 5. 验证和调整分配结果 ===
        validated_allocation = self._validate_and_adjust_allocation(
            power_allocation, 
            total_power_command, 
            power_constraints
        )
        
        # === 6. 记录分配历史 ===
        allocation_record = {
            'timestamp': self.total_allocations,
            'total_power_command': total_power_command,
            'allocation_weights': allocation_weights.__dict__,
            'power_allocation': validated_allocation,
            'bms_states': bms_states,
            'allocation_efficiency': self._calculate_allocation_efficiency(validated_allocation, bms_states)
        }
        
        self.allocation_history.append(allocation_record)
        self.total_allocations += 1
        
        # === 7. 自适应权重调整 ===
        if self.adaptation_enabled:
            self._adapt_allocation_weights(allocation_record)
        
        return validated_allocation
    
    def _prepare_allocation_weights(self, upper_layer_weights: Optional[Dict[str, float]]) -> AllocationWeights:
        """准备分配权重"""
        
        if upper_layer_weights is None:
            return AllocationWeights()
        
        # 从上层权重映射到分配权重
        weights = AllocationWeights()
        
        weights.soc_balance_weight = upper_layer_weights.get('soc_balance', 0.3)
        weights.temp_balance_weight = upper_layer_weights.get('temp_balance', 0.2)
        weights.lifetime_weight = upper_layer_weights.get('lifetime', 0.3)
        weights.efficiency_weight = upper_layer_weights.get('efficiency', 0.2)
        
        # 归一化权重
        total_weight = (weights.soc_balance_weight + weights.temp_balance_weight + 
                       weights.lifetime_weight + weights.efficiency_weight)
        
        if total_weight > 0:
            weights.soc_balance_weight /= total_weight
            weights.temp_balance_weight /= total_weight
            weights.lifetime_weight /= total_weight
            weights.efficiency_weight /= total_weight
        
        return weights
    
    def _collect_bms_states(self) -> List[Dict]:
        """收集各BMS状态"""
        
        bms_states = []
        
        for bms in self.bms_list:
            # 获取BMS摘要状态
            bms_summary = bms.get_bms_summary()
            
            # 计算额外的分配相关指标
            bms_state = {
                'bms_id': bms_summary['bms_id'],
                'avg_soc': bms_summary['avg_soc'],
                'soc_std': bms_summary['soc_std'],
                'avg_temperature': bms_summary['avg_temperature'],
                'temp_std': bms_summary['temp_std'],
                'avg_soh': bms_summary['avg_soh'],
                'total_cost': bms_summary['total_cost'],
                'health_status': bms_summary['health_status'],
                'balancing_active': bms_summary['balancing_active'],
                
                # 功率能力
                'max_charge_power': bms._get_max_charge_power(),
                'max_discharge_power': bms._get_max_discharge_power(),
                
                # 分配相关指标
                'soc_priority': self._calculate_soc_priority(bms_summary),
                'temp_priority': self._calculate_temp_priority(bms_summary),
                'lifetime_priority': self._calculate_lifetime_priority(bms_summary),
                'efficiency_factor': self._calculate_efficiency_factor(bms_summary)
            }
            
            bms_states.append(bms_state)
        
        return bms_states
    
    def _calculate_soc_priority(self, bms_summary: Dict) -> float:
        """计算SOC优先级 (0-1, 越高越需要功率)"""
        
        avg_soc = bms_summary['avg_soc']
        soc_std = bms_summary['soc_std']
        
        # 基于SOC水平的优先级
        if avg_soc < 30.0:
            soc_level_priority = 1.0  # 低SOC需要充电
        elif avg_soc > 70.0:
            soc_level_priority = 0.2  # 高SOC减少充电
        else:
            soc_level_priority = 0.5  # 中等SOC
        
        # 基于SOC不平衡的优先级调整
        if soc_std > 2.0:
            imbalance_adjustment = -0.2  # 不平衡的BMS降低优先级
        else:
            imbalance_adjustment = 0.1   # 平衡良好的BMS提高优先级
        
        priority = np.clip(soc_level_priority + imbalance_adjustment, 0.0, 1.0)
        return priority
    
    def _calculate_temp_priority(self, bms_summary: Dict) -> float:
        """计算温度优先级 (0-1, 越高越适合接受功率)"""
        
        avg_temp = bms_summary['avg_temperature']
        temp_std = bms_summary['temp_std']
        
        # 基于温度水平的优先级
        optimal_temp = 25.0
        temp_deviation = abs(avg_temp - optimal_temp)
        
        if temp_deviation < 5.0:
            temp_level_priority = 1.0   # 温度最佳
        elif temp_deviation < 15.0:
            temp_level_priority = 0.7   # 温度良好
        else:
            temp_level_priority = 0.3   # 温度偏差较大
        
        # 基于温度不平衡的调整
        if temp_std > 5.0:
            imbalance_adjustment = -0.3  # 温度不平衡降低优先级
        else:
            imbalance_adjustment = 0.0
        
        priority = np.clip(temp_level_priority + imbalance_adjustment, 0.0, 1.0)
        return priority
    
    def _calculate_lifetime_priority(self, bms_summary: Dict) -> float:
        """计算寿命优先级 (0-1, 越高越适合接受功率)"""
        
        avg_soh = bms_summary['avg_soh']
        health_status = bms_summary['health_status']
        
        # 基于SOH的优先级
        if avg_soh > 95.0:
            soh_priority = 1.0      # 新电池
        elif avg_soh > 85.0:
            soh_priority = 0.8      # 良好状态
        elif avg_soh > 75.0:
            soh_priority = 0.5      # 中等状态
        else:
            soh_priority = 0.2      # 老化严重
        
        # 基于健康状态的调整
        health_adjustments = {
            "Good": 0.1,
            "Fair": 0.0,
            "Poor": -0.2,
            "Critical": -0.5
        }
        
        health_adjustment = health_adjustments.get(health_status, 0.0)
        priority = np.clip(soh_priority + health_adjustment, 0.0, 1.0)
        
        return priority
    
    def _calculate_efficiency_factor(self, bms_summary: Dict) -> float:
        """计算效率因子 (0-1, 越高效率越好)"""
        
        # 基于BMS内均衡状态的效率评估
        soc_std = bms_summary['soc_std']
        temp_std = bms_summary['temp_std']
        balancing_active = bms_summary['balancing_active']
        
        # SOC均匀性对效率的影响
        soc_efficiency = max(0.5, 1.0 - soc_std / 10.0)
        
        # 温度均匀性对效率的影响
        temp_efficiency = max(0.5, 1.0 - temp_std / 20.0)
        
        # 均衡状态对效率的影响
        balance_efficiency = 0.9 if balancing_active else 1.0
        
        overall_efficiency = soc_efficiency * temp_efficiency * balance_efficiency
        return np.clip(overall_efficiency, 0.0, 1.0)
    
    def _calculate_power_constraints(self, bms_states: List[Dict]) -> List[Tuple[float, float]]:
        """计算各BMS功率约束"""
        
        power_constraints = []
        
        for bms_state in bms_states:
            # 基础功率限制
            max_charge = bms_state['max_charge_power']
            max_discharge = bms_state['max_discharge_power']
            
            # 安全裕度
            safety_factor = 0.95
            max_charge *= safety_factor
            max_discharge *= safety_factor
            
            # 温度降额
            avg_temp = bms_state['avg_temperature']
            if avg_temp > 45.0:
                temp_derating = max(0.5, (60.0 - avg_temp) / 15.0)
                max_charge *= temp_derating
                max_discharge *= temp_derating
            elif avg_temp < 10.0:
                temp_derating = max(0.5, (avg_temp + 10.0) / 20.0)
                max_charge *= temp_derating
                max_discharge *= temp_derating
            
            # SOC降额
            avg_soc = bms_state['avg_soc']
            if avg_soc > 90.0:
                soc_derating = max(0.3, (95.0 - avg_soc) / 5.0)
                max_charge *= soc_derating
            elif avg_soc < 10.0:
                soc_derating = max(0.3, (avg_soc - 5.0) / 5.0)
                max_discharge *= soc_derating
            
            # 约束范围 [min_discharge_power, max_charge_power]
            constraints = (-max_discharge, max_charge)
            power_constraints.append(constraints)
        
        return power_constraints
    
    def _execute_allocation_algorithm(self, 
                                    total_power: float,
                                    weights: AllocationWeights,
                                    bms_states: List[Dict],
                                    power_constraints: List[Tuple[float, float]]) -> Dict[str, float]:
        """执行分配算法"""
        
        if self.allocation_strategy == AllocationStrategy.EQUAL_POWER:
            return self._equal_power_allocation(total_power, bms_states)
        
        elif self.allocation_strategy == AllocationStrategy.SOC_WEIGHTED:
            return self._soc_weighted_allocation(total_power, bms_states, power_constraints)
        
        elif self.allocation_strategy == AllocationStrategy.MULTI_OBJECTIVE:
            return self._multi_objective_allocation(total_power, weights, bms_states, power_constraints)
        
        else:  # ADAPTIVE
            return self._adaptive_allocation(total_power, weights, bms_states, power_constraints)
    
    def _equal_power_allocation(self, total_power: float, bms_states: List[Dict]) -> Dict[str, float]:
        """均等功率分配"""
        power_per_bms = total_power / self.num_bms
        
        allocation = {}
        for bms_state in bms_states:
            allocation[bms_state['bms_id']] = power_per_bms
        
        return allocation
    
    def _soc_weighted_allocation(self, 
                               total_power: float, 
                               bms_states: List[Dict],
                               power_constraints: List[Tuple[float, float]]) -> Dict[str, float]:
        """基于SOC的加权分配"""
        
        # 计算SOC权重
        soc_values = [state['avg_soc'] for state in bms_states]
        soc_mean = np.mean(soc_values)
        
        allocation_weights = []
        for soc in soc_values:
            if total_power > 0:  # 充电
                # SOC低的BMS获得更多功率
                weight = 1.0 + (soc_mean - soc) * 0.02
            else:  # 放电
                # SOC高的BMS提供更多功率
                weight = 1.0 + (soc - soc_mean) * 0.02
            
            allocation_weights.append(max(0.1, weight))
        
        # 归一化权重
        total_weight = sum(allocation_weights)
        normalized_weights = [w / total_weight for w in allocation_weights]
        
        # 分配功率
        allocation = {}
        for i, bms_state in enumerate(bms_states):
            allocated_power = total_power * normalized_weights[i]
            allocation[bms_state['bms_id']] = allocated_power
        
        return allocation
    
    def _multi_objective_allocation(self, 
                                  total_power: float,
                                  weights: AllocationWeights,
                                  bms_states: List[Dict],
                                  power_constraints: List[Tuple[float, float]]) -> Dict[str, float]:
        """多目标优化分配"""
        
        # 计算各BMS的综合评分
        composite_scores = []
        
        for bms_state in bms_states:
            # 各目标评分
            soc_score = bms_state['soc_priority']
            temp_score = bms_state['temp_priority']
            lifetime_score = bms_state['lifetime_priority']
            efficiency_score = bms_state['efficiency_factor']
            
            # 加权综合评分
            composite_score = (
                weights.soc_balance_weight * soc_score +
                weights.temp_balance_weight * temp_score +
                weights.lifetime_weight * lifetime_score +
                weights.efficiency_weight * efficiency_score
            )
            
            composite_scores.append(composite_score)
        
        # 归一化评分为分配权重
        total_score = sum(composite_scores)
        if total_score > 0:
            allocation_weights = [score / total_score for score in composite_scores]
        else:
            allocation_weights = [1.0 / self.num_bms] * self.num_bms
        
        # 初始分配
        allocation = {}
        for i, bms_state in enumerate(bms_states):
            allocated_power = total_power * allocation_weights[i]
            allocation[bms_state['bms_id']] = allocated_power
        
        return allocation
    
    def _adaptive_allocation(self, 
                           total_power: float,
                           weights: AllocationWeights,
                           bms_states: List[Dict],
                           power_constraints: List[Tuple[float, float]]) -> Dict[str, float]:
        """自适应分配算法"""
        
        # 基础多目标分配
        base_allocation = self._multi_objective_allocation(total_power, weights, bms_states, power_constraints)
        
        # 基于历史性能的自适应调整
        if len(self.allocation_history) >= self.performance_window:
            performance_adjustments = self._calculate_performance_adjustments()
            
            # 应用性能调整
            for i, bms_state in enumerate(bms_states):
                bms_id = bms_state['bms_id']
                if bms_id in performance_adjustments:
                    adjustment_factor = performance_adjustments[bms_id]
                    base_allocation[bms_id] *= adjustment_factor
        
        return base_allocation
    
    def _calculate_performance_adjustments(self) -> Dict[str, float]:
        """计算基于历史性能的调整因子"""
        
        performance_adjustments = {}
        
        # 分析最近的分配性能
        recent_records = self.allocation_history[-self.performance_window:]
        
        # 按BMS统计性能
        bms_performance = {}
        for record in recent_records:
            for bms_id, allocated_power in record['power_allocation'].items():
                if bms_id not in bms_performance:
                    bms_performance[bms_id] = []
                
                # 计算该BMS的分配效率
                efficiency = record['allocation_efficiency'].get(bms_id, 1.0)
                bms_performance[bms_id].append(efficiency)
        
        # 计算调整因子
        for bms_id, efficiencies in bms_performance.items():
            avg_efficiency = np.mean(efficiencies)
            
            # 效率高的BMS增加权重，效率低的减少权重
            if avg_efficiency > 0.9:
                adjustment_factor = 1.1
            elif avg_efficiency > 0.8:
                adjustment_factor = 1.0
            elif avg_efficiency > 0.7:
                adjustment_factor = 0.9
            else:
                adjustment_factor = 0.8
            
            performance_adjustments[bms_id] = adjustment_factor
        
        return performance_adjustments
    
    def _validate_and_adjust_allocation(self, 
                                      allocation: Dict[str, float],
                                      total_power: float,
                                      power_constraints: List[Tuple[float, float]]) -> Dict[str, float]:
        """验证和调整分配结果"""
        
        adjusted_allocation = allocation.copy()
        
        # === 1. 应用功率约束 ===
        for i, bms_id in enumerate(adjusted_allocation.keys()):
            min_power, max_power = power_constraints[i]
            
            # 约束到可行域
            original_power = adjusted_allocation[bms_id]
            constrained_power = np.clip(original_power, min_power, max_power)
            adjusted_allocation[bms_id] = constrained_power
        
        # === 2. 功率平衡调整 ===
        allocated_total = sum(adjusted_allocation.values())
        power_error = total_power - allocated_total
        
        if abs(power_error) > abs(total_power) * self.constraints.power_balance_tolerance:
            # 重新分配剩余功率
            adjusted_allocation = self._redistribute_power_error(
                adjusted_allocation, power_error, power_constraints
            )
        
        return adjusted_allocation
    
    def _redistribute_power_error(self, 
                                allocation: Dict[str, float],
                                power_error: float,
                                power_constraints: List[Tuple[float, float]]) -> Dict[str, float]:
        """重新分配功率误差"""
        
        redistributed_allocation = allocation.copy()
        remaining_error = power_error
        
        # 找到还有余量的BMS
        bms_ids = list(allocation.keys())
        
        for i, bms_id in enumerate(bms_ids):
            if abs(remaining_error) < 1.0:  # 1W容差
                break
            
            min_power, max_power = power_constraints[i]
            current_power = redistributed_allocation[bms_id]
            
            if power_error > 0:  # 需要增加功率
                available_capacity = max_power - current_power
                redistribution = min(available_capacity, remaining_error)
            else:  # 需要减少功率
                available_capacity = current_power - min_power
                redistribution = max(-available_capacity, remaining_error)
            
            redistributed_allocation[bms_id] += redistribution
            remaining_error -= redistribution
        
        return redistributed_allocation
    
    def _calculate_allocation_efficiency(self, 
                                       allocation: Dict[str, float],
                                       bms_states: List[Dict]) -> Dict[str, float]:
        """计算分配效率"""
        
        efficiency_scores = {}
        
        for bms_state in bms_states:
            bms_id = bms_state['bms_id']
            allocated_power = allocation.get(bms_id, 0.0)
            
            # 基于BMS状态计算预期效率
            base_efficiency = bms_state['efficiency_factor']
            
            # 功率利用率影响
            max_power_capacity = max(bms_state['max_charge_power'], 
                                   abs(bms_state['max_discharge_power']))
            
            if max_power_capacity > 0:
                power_utilization = abs(allocated_power) / max_power_capacity
                utilization_efficiency = 1.0 - abs(power_utilization - 0.7) * 0.3  # 70%利用率最优
            else:
                utilization_efficiency = 1.0
            
            # 综合效率
            overall_efficiency = base_efficiency * utilization_efficiency
            efficiency_scores[bms_id] = np.clip(overall_efficiency, 0.0, 1.0)
        
        return efficiency_scores
    
    def _adapt_allocation_weights(self, allocation_record: Dict):
        """自适应调整分配权重"""
        
        if len(self.allocation_history) < self.performance_window:
            return
        
        # 分析最近的分配效果
        recent_records = self.allocation_history[-self.performance_window:]
        
        # 计算各目标的达成情况
        soc_balance_scores = []
        temp_balance_scores = []
        lifetime_scores = []
        efficiency_scores = []
        
        for record in recent_records:
            avg_efficiency = np.mean(list(record['allocation_efficiency'].values()))
            efficiency_scores.append(avg_efficiency)
            
            # 简化的目标评分计算
            bms_states = record['bms_states']
            soc_values = [state['avg_soc'] for state in bms_states]
            temp_values = [state['avg_temperature'] for state in bms_states]
            soh_values = [state['avg_soh'] for state in bms_states]
            
            soc_balance_score = 1.0 - np.std(soc_values) / 20.0  # 标准化
            temp_balance_score = 1.0 - np.std(temp_values) / 30.0
            lifetime_score = np.mean(soh_values) / 100.0
            
            soc_balance_scores.append(max(0.0, soc_balance_score))
            temp_balance_scores.append(max(0.0, temp_balance_score))
            lifetime_scores.append(lifetime_score)
        
        # 基于目标达成情况调整权重
        avg_soc_score = np.mean(soc_balance_scores)
        avg_temp_score = np.mean(temp_balance_scores)
        avg_lifetime_score = np.mean(lifetime_scores)
        avg_efficiency_score = np.mean(efficiency_scores)
        
        # 表现差的目标增加权重
        if avg_soc_score < 0.7:
            self.default_weights.soc_balance_weight *= (1.0 + self.adaptation_rate)
        if avg_temp_score < 0.7:
            self.default_weights.temp_balance_weight *= (1.0 + self.adaptation_rate)
        if avg_lifetime_score < 0.85:
            self.default_weights.lifetime_weight *= (1.0 + self.adaptation_rate)
        if avg_efficiency_score < 0.8:
            self.default_weights.efficiency_weight *= (1.0 + self.adaptation_rate)
        
        # 重新归一化权重
        total_weight = (self.default_weights.soc_balance_weight + 
                       self.default_weights.temp_balance_weight +
                       self.default_weights.lifetime_weight + 
                       self.default_weights.efficiency_weight)
        
        if total_weight > 0:
            self.default_weights.soc_balance_weight /= total_weight
            self.default_weights.temp_balance_weight /= total_weight
            self.default_weights.lifetime_weight /= total_weight
            self.default_weights.efficiency_weight /= total_weight
    
    def get_allocation_statistics(self) -> Dict:
        """获取分配统计信息"""
        
        if not self.allocation_history:
            return {'error': 'No allocation history available'}
        
        recent_records = self.allocation_history[-20:] if len(self.allocation_history) >= 20 else self.allocation_history
        
        # 计算统计量
        total_powers = [record['total_power_command'] for record in recent_records]
        avg_efficiencies = [np.mean(list(record['allocation_efficiency'].values())) for record in recent_records]
        
        statistics = {
            'allocator_id': self.allocator_id,
            'allocation_strategy': self.allocation_strategy.value,
            'total_allocations': self.total_allocations,
            'num_bms': self.num_bms,
            
            'recent_performance': {
                'avg_total_power': np.mean(total_powers),
                'avg_allocation_efficiency': np.mean(avg_efficiencies),
                'power_std': np.std(total_powers),
                'efficiency_std': np.std(avg_efficiencies)
            },
            
            'current_weights': self.default_weights.__dict__,
            'adaptation_enabled': self.adaptation_enabled,
            
            'allocation_counts_by_range': self._analyze_allocation_distribution(recent_records)
        }
        
        return statistics
    
    def _analyze_allocation_distribution(self, records: List[Dict]) -> Dict:
        """分析分配分布"""
        
        power_ranges = {
            'low_power': 0,      # <100kW
            'medium_power': 0,   # 100-500kW
            'high_power': 0      # >500kW
        }
        
        for record in records:
            total_power = abs(record['total_power_command'])
            
            if total_power < 100000:
                power_ranges['low_power'] += 1
            elif total_power < 500000:
                power_ranges['medium_power'] += 1
            else:
                power_ranges['high_power'] += 1
        
        return power_ranges
    
    def reset(self):
        """重置分配器"""
        self.allocation_history.clear()
        self.total_allocations = 0
        self.allocation_efficiency_history.clear()
        
        # 重置权重为默认值
        self.default_weights = AllocationWeights()
        
        print(f"🔄 中央功率分配器 {self.allocator_id} 已重置")
    
    def update_allocation_strategy(self, new_strategy: AllocationStrategy) -> bool:
        """更新分配策略"""
        try:
            old_strategy = self.allocation_strategy
            self.allocation_strategy = new_strategy
            
            print(f"🔄 分配器 {self.allocator_id} 策略更新: {old_strategy.value} -> {new_strategy.value}")
            return True
        except Exception as e:
            print(f"❌ 分配策略更新失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"CentralPowerAllocator({self.allocator_id}): "
                f"策略={self.allocation_strategy.value}, "
                f"BMS数={self.num_bms}, "
                f"分配次数={self.total_allocations}")
