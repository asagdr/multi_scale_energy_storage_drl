import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.system_config import SystemConfig

class RewardType(Enum):
    """奖励类型枚举"""
    POWER_TRACKING = "power_tracking"       # 功率跟踪奖励
    SOC_BALANCE = "soc_balance"            # SOC均衡奖励
    TEMP_BALANCE = "temp_balance"          # 温度均衡奖励
    LIFETIME_COST = "lifetime_cost"        # 寿命成本奖励
    EFFICIENCY = "efficiency"              # 效率奖励
    SAFETY = "safety"                      # 安全奖励
    CONSTRAINT = "constraint"              # 约束满足奖励

@dataclass
class RewardComponent:
    """奖励组件数据结构"""
    reward_type: RewardType
    raw_value: float = 0.0          # 原始值
    normalized_value: float = 0.0    # 归一化值 [-1, 1]
    weight: float = 1.0             # 权重
    weighted_value: float = 0.0     # 加权值
    description: str = ""           # 描述

@dataclass
class RewardResult:
    """奖励计算结果"""
    total_reward: float = 0.0
    components: Dict[RewardType, RewardComponent] = field(default_factory=dict)
    bonus_rewards: List[Tuple[str, float]] = field(default_factory=list)
    penalty_rewards: List[Tuple[str, float]] = field(default_factory=list)
    
    def add_component(self, component: RewardComponent):
        """添加奖励组件"""
        self.components[component.reward_type] = component
        self.total_reward += component.weighted_value
    
    def add_bonus(self, description: str, value: float):
        """添加奖励加分"""
        self.bonus_rewards.append((description, value))
        self.total_reward += value
    
    def add_penalty(self, description: str, value: float):
        """添加奖励惩罚"""
        self.penalty_rewards.append((description, value))
        self.total_reward += value  # value应为负数

class RewardCalculator:
    """
    奖励计算器
    为双层DRL架构计算多目标奖励函数
    """
    
    def __init__(self, 
                 system_config: SystemConfig,
                 calculator_id: str = "RewardCalculator_001"):
        """
        初始化奖励计算器
        
        Args:
            system_config: 系统配置
            calculator_id: 计算器ID
        """
        self.system_config = system_config
        self.calculator_id = calculator_id
        
        # === 奖励权重配置 ===
        self.reward_weights = system_config.objective_weights.copy()
        
        # === 归一化参数 ===
        self.normalization_params = self._initialize_normalization_params()
        
        # === 奖励历史 ===
        self.reward_history: List[RewardResult] = []
        
        # === 统计信息 ===
        self.total_calculations = 0
        self.cumulative_reward = 0.0
        self.reward_stats = {
            reward_type: {'sum': 0.0, 'count': 0, 'avg': 0.0, 'std': 0.0}
            for reward_type in RewardType
        }
        
        # === 自适应权重 ===
        self.enable_adaptive_weights = True
        self.weight_adaptation_rate = 0.01
        
        print(f"✅ 奖励计算器初始化完成: {calculator_id}")
        print(f"   奖励权重: {self.reward_weights}")
    
    def _initialize_normalization_params(self) -> Dict[RewardType, Dict[str, float]]:
        """初始化归一化参数"""
        return {
            RewardType.POWER_TRACKING: {
                'max_error': 1000.0,    # W, 最大功率误差
                'target_error': 50.0    # W, 目标功率误差
            },
            RewardType.SOC_BALANCE: {
                'max_std': 20.0,        # %, 最大SOC标准差
                'target_std': 1.0       # %, 目标SOC标准差
            },
            RewardType.TEMP_BALANCE: {
                'max_std': 20.0,        # ℃, 最大温度标准差
                'target_std': 2.0       # ℃, 目标温度标准差
            },
            RewardType.LIFETIME_COST: {
                'max_cost_rate': 1.0,   # 元/s, 最大成本增长率
                'target_cost_rate': 0.01 # 元/s, 目标成本增长率
            },
            RewardType.EFFICIENCY: {
                'min_efficiency': 0.8,  # 最低效率
                'target_efficiency': 0.95 # 目标效率
            },
            RewardType.SAFETY: {
                'min_score': 0.0,       # 最低安全评分
                'target_score': 1.0     # 目标安全评分
            },
            RewardType.CONSTRAINT: {
                'max_violations': 10,   # 最大违约数
                'target_violations': 0  # 目标违约数
            }
        }
    
    def calculate_power_tracking_reward(self, 
                                      command_power: float, 
                                      actual_power: float) -> RewardComponent:
        """
        计算功率跟踪奖励
        
        Args:
            command_power: 命令功率 (W)
            actual_power: 实际功率 (W)
            
        Returns:
            功率跟踪奖励组件
        """
        # 计算功率误差
        power_error = abs(actual_power - command_power)
        
        # 归一化 (使用双曲正切函数)
        params = self.normalization_params[RewardType.POWER_TRACKING]
        max_error = params['max_error']
        target_error = params['target_error']
        
        if power_error <= target_error:
            normalized_reward = 1.0  # 完美跟踪
        else:
            # 指数衰减
            normalized_reward = np.exp(-(power_error - target_error) / (max_error - target_error))
            normalized_reward = max(0.0, min(1.0, normalized_reward))
        
        # 转换到 [-1, 1] 范围
        normalized_reward = 2 * normalized_reward - 1
        
        component = RewardComponent(
            reward_type=RewardType.POWER_TRACKING,
            raw_value=power_error,
            normalized_value=normalized_reward,
            weight=self.reward_weights.get('power_tracking', 0.3),
            description=f"功率误差: {power_error:.1f}W"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def calculate_soc_balance_reward(self, soc_std: float, soc_consistency: float) -> RewardComponent:
        """
        计算SOC均衡奖励
        
        Args:
            soc_std: SOC标准差 (%)
            soc_consistency: SOC一致性指数 [0, 1]
            
        Returns:
            SOC均衡奖励组件
        """
        # 基于SOC标准差的奖励
        params = self.normalization_params[RewardType.SOC_BALANCE]
        max_std = params['max_std']
        target_std = params['target_std']
        
        if soc_std <= target_std:
            std_reward = 1.0
        else:
            std_reward = max(0.0, 1.0 - (soc_std - target_std) / (max_std - target_std))
        
        # 结合一致性指数
        normalized_reward = 0.7 * std_reward + 0.3 * soc_consistency
        normalized_reward = 2 * normalized_reward - 1  # 转换到 [-1, 1]
        
        component = RewardComponent(
            reward_type=RewardType.SOC_BALANCE,
            raw_value=soc_std,
            normalized_value=normalized_reward,
            weight=self.reward_weights.get('soc_balance', 0.25),
            description=f"SOC标准差: {soc_std:.2f}%, 一致性: {soc_consistency:.3f}"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def calculate_temp_balance_reward(self, 
                                    temp_std: float, 
                                    temp_consistency: float,
                                    max_temp: float) -> RewardComponent:
        """
        计算温度均衡奖励
        
        Args:
            temp_std: 温度标准差 (℃)
            temp_consistency: 温度一致性指数 [0, 1]
            max_temp: 最高温度 (℃)
            
        Returns:
            温度均衡奖励组件
        """
        # 基于温度标准差的奖励
        params = self.normalization_params[RewardType.TEMP_BALANCE]
        max_std = params['max_std']
        target_std = params['target_std']
        
        if temp_std <= target_std:
            std_reward = 1.0
        else:
            std_reward = max(0.0, 1.0 - (temp_std - target_std) / (max_std - target_std))
        
        # 温度过高惩罚
        temp_penalty = 0.0
        if max_temp > 45.0:  # 45℃以上开始惩罚
            temp_penalty = min(0.5, (max_temp - 45.0) / 20.0)  # 最大惩罚50%
        
        # 综合奖励
        balance_reward = 0.6 * std_reward + 0.4 * temp_consistency
        final_reward = balance_reward * (1.0 - temp_penalty)
        normalized_reward = 2 * final_reward - 1  # 转换到 [-1, 1]
        
        component = RewardComponent(
            reward_type=RewardType.TEMP_BALANCE,
            raw_value=temp_std,
            normalized_value=normalized_reward,
            weight=self.reward_weights.get('thermal_balance', 0.2),
            description=f"温度标准差: {temp_std:.1f}℃, 最高温度: {max_temp:.1f}℃"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def calculate_lifetime_cost_reward(self, 
                                     current_cost: float, 
                                     previous_cost: float,
                                     delta_t: float) -> RewardComponent:
        """
        计算寿命成本奖励
        
        Args:
            current_cost: 当前累积成本 (元)
            previous_cost: 前一时刻累积成本 (元)
            delta_t: 时间间隔 (s)
            
        Returns:
            寿命成本奖励组件
        """
        # 计算成本增长率
        cost_increase_rate = (current_cost - previous_cost) / delta_t if delta_t > 0 else 0.0
        
        # 归一化
        params = self.normalization_params[RewardType.LIFETIME_COST]
        max_cost_rate = params['max_cost_rate']
        target_cost_rate = params['target_cost_rate']
        
        if cost_increase_rate <= target_cost_rate:
            normalized_reward = 1.0
        else:
            # 成本增长越快，奖励越低
            normalized_reward = max(0.0, 1.0 - (cost_increase_rate - target_cost_rate) / 
                                  (max_cost_rate - target_cost_rate))
        
        normalized_reward = 2 * normalized_reward - 1  # 转换到 [-1, 1]
        
        component = RewardComponent(
            reward_type=RewardType.LIFETIME_COST,
            raw_value=cost_increase_rate,
            normalized_value=normalized_reward,
            weight=self.reward_weights.get('lifetime_extension', 0.2),
            description=f"成本增长率: {cost_increase_rate:.4f}元/s"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def calculate_efficiency_reward(self, 
                                  power_efficiency: float, 
                                  energy_efficiency: float) -> RewardComponent:
        """
        计算效率奖励
        
        Args:
            power_efficiency: 功率效率 [0, 1]
            energy_efficiency: 能量效率 [0, 1]
            
        Returns:
            效率奖励组件
        """
        # 综合效率
        overall_efficiency = 0.6 * power_efficiency + 0.4 * energy_efficiency
        
        # 归一化
        params = self.normalization_params[RewardType.EFFICIENCY]
        min_eff = params['min_efficiency']
        target_eff = params['target_efficiency']
        
        if overall_efficiency >= target_eff:
            normalized_reward = 1.0
        elif overall_efficiency >= min_eff:
            normalized_reward = (overall_efficiency - min_eff) / (target_eff - min_eff)
        else:
            normalized_reward = 0.0
        
        normalized_reward = 2 * normalized_reward - 1  # 转换到 [-1, 1]
        
        component = RewardComponent(
            reward_type=RewardType.EFFICIENCY,
            raw_value=overall_efficiency,
            normalized_value=normalized_reward,
            weight=self.reward_weights.get('efficiency', 0.15),
            description=f"功率效率: {power_efficiency:.3f}, 能量效率: {energy_efficiency:.3f}"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def calculate_safety_reward(self, safety_score: float, violation_count: int) -> RewardComponent:
        """
        计算安全奖励
        
        Args:
            safety_score: 安全评分 [0, 1]
            violation_count: 违约次数
            
        Returns:
            安全奖励组件
        """
        # 基础安全奖励
        base_reward = safety_score
        
        # 违约惩罚
        violation_penalty = min(0.8, violation_count * 0.1)  # 每次违约惩罚10%，最大80%
        
        # 综合安全奖励
        final_reward = base_reward * (1.0 - violation_penalty)
        normalized_reward = 2 * final_reward - 1  # 转换到 [-1, 1]
        
        component = RewardComponent(
            reward_type=RewardType.SAFETY,
            raw_value=safety_score,
            normalized_value=normalized_reward,
            weight=self.reward_weights.get('safety', 0.1),
            description=f"安全评分: {safety_score:.3f}, 违约次数: {violation_count}"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def calculate_constraint_reward(self, 
                                  constraint_violations: int, 
                                  constraint_warnings: int) -> RewardComponent:
        """
        计算约束满足奖励
        
        Args:
            constraint_violations: 约束违反次数
            constraint_warnings: 约束警告次数
            
        Returns:
            约束奖励组件
        """
        # 基础约束奖励
        if constraint_violations == 0 and constraint_warnings == 0:
            base_reward = 1.0
        elif constraint_violations == 0:
            base_reward = max(0.5, 1.0 - constraint_warnings * 0.1)
        else:
            base_reward = max(0.0, 0.5 - constraint_violations * 0.2)
        
        normalized_reward = 2 * base_reward - 1  # 转换到 [-1, 1]
        
        component = RewardComponent(
            reward_type=RewardType.CONSTRAINT,
            raw_value=constraint_violations + constraint_warnings * 0.1,
            normalized_value=normalized_reward,
            weight=self.reward_weights.get('constraint_satisfaction', 0.1),
            description=f"违反: {constraint_violations}, 警告: {constraint_warnings}"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def calculate_comprehensive_reward(self, 
                                     system_state: Dict[str, Any],
                                     previous_state: Optional[Dict[str, Any]] = None,
                                     action: Optional[np.ndarray] = None,
                                     delta_t: float = 1.0) -> RewardResult:
        """
        计算综合奖励
        
        Args:
            system_state: 当前系统状态
            previous_state: 前一时刻系统状态
            action: 执行的动作
            delta_t: 时间间隔 (s)
            
        Returns:
            综合奖励结果
        """
        result = RewardResult()
        
        # === 1. 功率跟踪奖励 ===
        if 'power_command' in system_state and 'actual_power' in system_state:
            power_component = self.calculate_power_tracking_reward(
                system_state['power_command'],
                system_state['actual_power']
            )
            result.add_component(power_component)
        
        # === 2. SOC均衡奖励 ===
        if 'soc_std' in system_state:
            soc_consistency = system_state.get('soc_consistency', 0.8)
            soc_component = self.calculate_soc_balance_reward(
                system_state['soc_std'],
                soc_consistency
            )
            result.add_component(soc_component)
        
        # === 3. 温度均衡奖励 ===
        if 'temp_std' in system_state:
            temp_consistency = system_state.get('temp_consistency', 0.8)
            max_temp = system_state.get('max_temperature', 25.0)
            temp_component = self.calculate_temp_balance_reward(
                system_state['temp_std'],
                temp_consistency,
                max_temp
            )
            result.add_component(temp_component)
        
        # === 4. 寿命成本奖励 ===
        if ('current_degradation_cost' in system_state and 
            previous_state and 'current_degradation_cost' in previous_state):
            lifetime_component = self.calculate_lifetime_cost_reward(
                system_state['current_degradation_cost'],
                previous_state['current_degradation_cost'],
                delta_t
            )
            result.add_component(lifetime_component)
        
        # === 5. 效率奖励 ===
        if 'power_efficiency' in system_state and 'energy_efficiency' in system_state:
            efficiency_component = self.calculate_efficiency_reward(
                system_state['power_efficiency'],
                system_state['energy_efficiency']
            )
            result.add_component(efficiency_component)
        
        # === 6. 安全奖励 ===
        if 'safety_score' in system_state:
            violation_count = system_state.get('violation_count', 0)
            safety_component = self.calculate_safety_reward(
                system_state['safety_score'],
                violation_count
            )
            result.add_component(safety_component)
        
        # === 7. 约束奖励 ===
        if 'constraint_violations' in system_state:
            constraint_warnings = system_state.get('constraint_warnings', 0)
            constraint_component = self.calculate_constraint_reward(
                system_state['constraint_violations'],
                constraint_warnings
            )
            result.add_component(constraint_component)
        
        # === 8. 额外奖励和惩罚 ===
        self._apply_bonus_penalties(result, system_state, action)
        
        # === 9. 自适应权重调整 ===
        if self.enable_adaptive_weights:
            self._adapt_weights(result)
        
        # === 10. 记录历史和统计 ===
        self.reward_history.append(result)
        self.total_calculations += 1
        self.cumulative_reward += result.total_reward
        
        # 更新统计信息
        for reward_type, component in result.components.items():
            stats = self.reward_stats[reward_type]
            stats['sum'] += component.normalized_value
            stats['count'] += 1
            stats['avg'] = stats['sum'] / stats['count']
        
        # 维护历史长度
        max_history = 1000
        if len(self.reward_history) > max_history:
            self.reward_history.pop(0)
        
        return result

    def calculate_multi_level_cluster_reward(self, 
                                           cluster_record: Dict,
                                           upper_layer_weights: Dict[str, float],
                                           previous_cluster_record: Optional[Dict] = None,
                                           action: Optional[np.ndarray] = None,
                                           delta_t: float = 1.0) -> RewardResult:
        """
        计算BMS集群的多层级综合奖励
        
        Args:
            cluster_record: 集群记录
            upper_layer_weights: 上层权重
            previous_cluster_record: 前一时刻集群记录
            action: 执行的动作
            delta_t: 时间间隔 (s)
            
        Returns:
            多层级奖励结果
        """
        result = RewardResult()
        
        # === 1. 系统级功率跟踪奖励 ===
        if 'total_power_command' in cluster_record and 'total_actual_power' in cluster_record:
            power_component = self.calculate_power_tracking_reward(
                cluster_record['total_power_command'],
                cluster_record['total_actual_power']
            )
            result.add_component(power_component)
        
        # === 2. BMS间均衡奖励 ===
        inter_bms_soc_component = RewardComponent(
            reward_type=RewardType.SOC_BALANCE,
            raw_value=cluster_record.get('inter_bms_soc_std', 0.0),
            weight=upper_layer_weights.get('soc_balance', 0.3) * 0.6,  # 60%用于BMS间均衡
            description=f"BMS间SOC均衡: σ={cluster_record.get('inter_bms_soc_std', 0.0):.2f}%"
        )
        
        # 归一化BMS间SOC均衡奖励
        soc_std = cluster_record.get('inter_bms_soc_std', 0.0)
        inter_bms_soc_component.normalized_value = 1.0 - min(1.0, soc_std / 15.0)
        inter_bms_soc_component.normalized_value = 2 * inter_bms_soc_component.normalized_value - 1
        inter_bms_soc_component.weighted_value = (inter_bms_soc_component.normalized_value * 
                                                inter_bms_soc_component.weight)
        
        result.add_component(inter_bms_soc_component)
        
        # === 3. BMS内均衡奖励 ===
        intra_bms_soc_component = RewardComponent(
            reward_type=RewardType.SOC_BALANCE,
            raw_value=cluster_record.get('avg_intra_bms_soc_std', 0.0),
            weight=upper_layer_weights.get('soc_balance', 0.3) * 0.4,  # 40%用于BMS内均衡
            description=f"BMS内SOC均衡: 平均σ={cluster_record.get('avg_intra_bms_soc_std', 0.0):.2f}%"
        )
        
        # 归一化BMS内SOC均衡奖励
        intra_soc_std = cluster_record.get('avg_intra_bms_soc_std', 0.0)
        intra_bms_soc_component.normalized_value = 1.0 - min(1.0, intra_soc_std / 8.0)
        intra_bms_soc_component.normalized_value = 2 * intra_bms_soc_component.normalized_value - 1
        intra_bms_soc_component.weighted_value = (intra_bms_soc_component.normalized_value * 
                                                intra_bms_soc_component.weight)
        
        result.add_component(intra_bms_soc_component)
        
        # === 4. 温度均衡奖励（类似处理） ===
        temp_component = self._calculate_multi_level_temp_reward(cluster_record, upper_layer_weights)
        result.add_component(temp_component)
        
        # === 5. 多层级成本奖励 ===
        if ('cost_breakdown' in cluster_record and 
            previous_cluster_record and 'cost_breakdown' in previous_cluster_record):
            
            current_cost = cluster_record['cost_breakdown'].get('total_system_cost', 0.0)
            previous_cost = previous_cluster_record['cost_breakdown'].get('total_system_cost', 0.0)
            
            lifetime_component = self.calculate_lifetime_cost_reward(
                current_cost, previous_cost, delta_t
            )
            lifetime_component.weight = upper_layer_weights.get('lifetime', 0.3)
            lifetime_component.weighted_value = lifetime_component.normalized_value * lifetime_component.weight
            
            result.add_component(lifetime_component)
        
        # === 6. 效率奖励 ===
        if 'system_power_efficiency' in cluster_record:
            energy_efficiency = cluster_record.get('cluster_metrics', {}).get('energy_efficiency', 1.0)
            efficiency_component = self.calculate_efficiency_reward(
                cluster_record['system_power_efficiency'],
                energy_efficiency
            )
            efficiency_component.weight = upper_layer_weights.get('efficiency', 0.2)
            efficiency_component.weighted_value = efficiency_component.normalized_value * efficiency_component.weight
            
            result.add_component(efficiency_component)
        
        # === 7. 安全和约束奖励 ===
        system_constraints = cluster_record.get('system_constraints_active', {})
        constraint_violations = sum(1 for active in system_constraints.values() if active)
        system_warnings = cluster_record.get('system_warning_count', 0)
        
        safety_component = self.calculate_safety_reward(
            cluster_record.get('cluster_metrics', {}).get('overall_balance_score', 0.8),
            constraint_violations
        )
        result.add_component(safety_component)
        
        constraint_component = self.calculate_constraint_reward(constraint_violations, system_warnings)
        result.add_component(constraint_component)
        
        # === 8. 协调效率加分/惩罚 ===
        self._apply_coordination_bonus_penalties(result, cluster_record)
        
        return result
    
    def _calculate_multi_level_temp_reward(self, cluster_record: Dict, upper_layer_weights: Dict) -> RewardComponent:
        """计算多层级温度奖励"""
        
        # BMS间温度均衡
        inter_temp_std = cluster_record.get('inter_bms_temp_std', 0.0)
        inter_temp_score = 1.0 - min(1.0, inter_temp_std / 20.0)
        
        # BMS内温度均衡
        intra_temp_std = cluster_record.get('avg_intra_bms_temp_std', 0.0)
        intra_temp_score = 1.0 - min(1.0, intra_temp_std / 12.0)
        
        # 综合温度评分
        overall_temp_score = 0.6 * inter_temp_score + 0.4 * intra_temp_score
        
        component = RewardComponent(
            reward_type=RewardType.TEMP_BALANCE,
            raw_value=inter_temp_std,
            normalized_value=2 * overall_temp_score - 1,
            weight=upper_layer_weights.get('temp_balance', 0.2),
            description=f"多层级温度均衡: BMS间σ={inter_temp_std:.1f}℃, BMS内σ={intra_temp_std:.1f}℃"
        )
        component.weighted_value = component.normalized_value * component.weight
        
        return component
    
    def _apply_coordination_bonus_penalties(self, result: RewardResult, cluster_record: Dict):
        """应用协调相关的奖励和惩罚"""
        
        coordination_commands = cluster_record.get('coordination_commands', {})
        
        # === 协调合理性奖励 ===
        if coordination_commands:
            # 有协调指令时，评估协调的合理性
            total_bms = cluster_record.get('num_bms', 10)
            coordination_ratio = len(coordination_commands) / total_bms
            
            if coordination_ratio < 0.3:  # 少量精准协调
                result.add_bonus("精准协调", 0.05)
            elif coordination_ratio > 0.7:  # 过度协调
                result.add_penalty("过度协调", -0.05)
        
        # === 系统均衡稳定奖励 ===
        balance_score = cluster_record.get('cluster_metrics', {}).get('overall_balance_score', 0.5)
        if balance_score > 0.9:
            result.add_bonus("系统高度均衡", 0.03)
        elif balance_score < 0.3:
            result.add_penalty("系统严重不均衡", -0.1)
        
        # === BMS健康差异惩罚 ===
        inter_soh_std = cluster_record.get('inter_bms_soh_std', 0.0)
        if inter_soh_std > 10.0:  # BMS间SOH差异超过10%
            result.add_penalty("BMS健康差异过大", -0.08)
    
    def _apply_bonus_penalties(self, 
                             result: RewardResult, 
                             system_state: Dict[str, Any],
                             action: Optional[np.ndarray]):
        """应用额外的奖励和惩罚"""
        
        # === 连续优秀表现奖励 ===
        if len(self.reward_history) >= 5:
            recent_rewards = [r.total_reward for r in self.reward_history[-5:]]
            if all(r > 0.5 for r in recent_rewards):
                result.add_bonus("连续优秀表现", 0.1)
        
        # === SOC极端值惩罚 ===
        avg_soc = system_state.get('pack_soc', 50.0)
        if avg_soc < 10.0 or avg_soc > 90.0:
            result.add_penalty("SOC极端值", -0.2)
        
        # === 温度过高惩罚 ===
        max_temp = system_state.get('max_temperature', 25.0)
        if max_temp > 50.0:
            penalty = -min(0.5, (max_temp - 50.0) / 20.0)
            result.add_penalty("温度过高", penalty)
        
        # === 快速响应奖励 ===
        response_time = system_state.get('response_time', float('inf'))
        if response_time < 0.1:  # 100ms内响应
            result.add_bonus("快速响应", 0.05)
        
        # === 动作平滑性奖励 ===
        if action is not None and len(self.reward_history) > 0:
            # 检查动作变化的平滑性
            action_smoothness = self._calculate_action_smoothness(action)
            if action_smoothness > 0.8:
                result.add_bonus("动作平滑", 0.05)
    
    def _calculate_action_smoothness(self, current_action: np.ndarray) -> float:
        """计算动作平滑性"""
        # 简化的平滑性计算
        # 实际实现中应该存储历史动作
        action_variance = np.var(current_action)
        smoothness = 1.0 / (1.0 + action_variance)
        return smoothness
    
    def _adapt_weights(self, result: RewardResult):
        """自适应权重调整"""
        if len(self.reward_history) < 10:
            return
        
        # 基于最近表现调整权重
        recent_results = self.reward_history[-10:]
        
        for reward_type in RewardType:
            if reward_type in result.components:
                # 计算该奖励类型的表现
                recent_values = [r.components.get(reward_type, RewardComponent(reward_type)).normalized_value 
                               for r in recent_results if reward_type in r.components]
                
                if recent_values:
                    avg_performance = np.mean(recent_values)
                    
                    # 表现差的奖励类型增加权重，表现好的减少权重
                    if avg_performance < -0.2:
                        weight_key = self._get_weight_key(reward_type)
                        if weight_key in self.reward_weights:
                            self.reward_weights[weight_key] *= (1.0 + self.weight_adaptation_rate)
                    elif avg_performance > 0.5:
                        weight_key = self._get_weight_key(reward_type)
                        if weight_key in self.reward_weights:
                            self.reward_weights[weight_key] *= (1.0 - self.weight_adaptation_rate * 0.5)
    
    def _get_weight_key(self, reward_type: RewardType) -> str:
        """获取权重键名"""
        mapping = {
            RewardType.POWER_TRACKING: 'power_tracking',
            RewardType.SOC_BALANCE: 'soc_balance',
            RewardType.TEMP_BALANCE: 'thermal_balance',
            RewardType.LIFETIME_COST: 'lifetime_extension',
            RewardType.EFFICIENCY: 'efficiency',
            RewardType.SAFETY: 'safety',
            RewardType.CONSTRAINT: 'constraint_satisfaction'
        }
        return mapping.get(reward_type, 'default')
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """获取奖励统计信息"""
        if not self.reward_history:
            return {'error': 'No reward history available'}
        
        recent_rewards = [r.total_reward for r in self.reward_history[-100:]]
        
        return {
            'calculator_id': self.calculator_id,
            'total_calculations': self.total_calculations,
            'cumulative_reward': self.cumulative_reward,
            'average_reward': self.cumulative_reward / self.total_calculations if self.total_calculations > 0 else 0.0,
            
            'recent_performance': {
                'avg_reward': np.mean(recent_rewards),
                'std_reward': np.std(recent_rewards),
                'min_reward': min(recent_rewards),
                'max_reward': max(recent_rewards),
                'trend': self._calculate_reward_trend()
            },
            
            'component_stats': {
                reward_type.value: {
                    'avg': stats['avg'],
                    'count': stats['count']
                } for reward_type, stats in self.reward_stats.items() if stats['count'] > 0
            },
            
            'current_weights': self.reward_weights.copy(),
            'adaptive_weights_enabled': self.enable_adaptive_weights
        }
    
    def _calculate_reward_trend(self) -> str:
        """计算奖励趋势"""
        if len(self.reward_history) < 20:
            return "insufficient_data"
        
        recent_20 = [r.total_reward for r in self.reward_history[-20:]]
        first_half = np.mean(recent_20[:10])
        second_half = np.mean(recent_20[10:])
        
        if second_half > first_half + 0.1:
            return "improving"
        elif second_half < first_half - 0.1:
            return "declining"
        else:
            return "stable"
    
    def reset_statistics(self):
        """重置统计信息"""
        self.reward_history.clear()
        self.total_calculations = 0
        self.cumulative_reward = 0.0
        for stats in self.reward_stats.values():
            stats.update({'sum': 0.0, 'count': 0, 'avg': 0.0, 'std': 0.0})
        
        print(f"🔄 奖励统计已重置: {self.calculator_id}")
    
    def __str__(self) -> str:
        """字符串表示"""
        avg_reward = self.cumulative_reward / self.total_calculations if self.total_calculations > 0 else 0.0
        return (f"RewardCalculator({self.calculator_id}): "
                f"计算次数={self.total_calculations}, "
                f"平均奖励={avg_reward:.3f}")
