"""
BMS间协调器
实现10个BMS之间的协调优化
基于系统级均衡目标生成协调指令
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class CoordinationMode(Enum):
    """协调模式枚举"""
    DISABLED = "disabled"                # 禁用协调
    SOC_BALANCE = "soc_balance"         # SOC均衡协调
    THERMAL_BALANCE = "thermal_balance"  # 热均衡协调
    LIFETIME_OPTIMIZATION = "lifetime"   # 寿命优化协调
    COMPREHENSIVE = "comprehensive"      # 综合协调

class CoordinationPriority(Enum):
    """协调优先级枚举"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CoordinationCommand:
    """协调指令数据结构"""
    target_bms_id: str
    command_type: str                    # 指令类型
    priority_level: CoordinationPriority
    
    # 功率调整
    suggested_power_bias: float = 0.0    # 功率偏置 [-0.5, 0.5]
    power_limit_adjustment: float = 1.0  # 功率限制调整系数 [0.5, 1.0]
    
    # 均衡目标
    target_soc: Optional[float] = None
    target_temp: Optional[float] = None
    
    # 协调参数
    coordination_weight: float = 1.0     # 协调权重
    expected_duration: float = 0.0       # 预期协调时间 (s)
    
    # 描述信息
    description: str = ""
    reasoning: str = ""

@dataclass
class CoordinationMetrics:
    """协调指标数据结构"""
    # BMS间均衡指标
    inter_bms_soc_std: float = 0.0
    inter_bms_temp_std: float = 0.0
    inter_bms_soh_std: float = 0.0
    
    # 系统级指标
    system_balance_score: float = 1.0    # 系统均衡评分 [0-1]
    coordination_efficiency: float = 1.0 # 协调效率 [0-1]
    
    # 协调效果
    soc_convergence_rate: float = 0.0    # SOC收敛速率 (%/hour)
    temp_convergence_rate: float = 0.0   # 温度收敛速率 (℃/hour)

class InterBMSCoordinator:
    """
    BMS间协调器
    实现系统级BMS间的智能协调优化
    """
    
    def __init__(self, 
                 bms_list: List,
                 coordination_mode: CoordinationMode = CoordinationMode.COMPREHENSIVE,
                 coordinator_id: str = "InterBMSCoordinator_001"):
        """
        初始化BMS间协调器
        
        Args:
            bms_list: BMS列表
            coordination_mode: 协调模式
            coordinator_id: 协调器ID
        """
        self.bms_list = bms_list
        self.num_bms = len(bms_list)
        self.coordination_mode = coordination_mode
        self.coordinator_id = coordinator_id
        
        # === 协调参数 ===
        self.coordination_params = {
            # 协调阈值
            'soc_imbalance_threshold': 5.0,        # 5% BMS间SOC差异触发协调
            'temp_imbalance_threshold': 10.0,      # 10℃ BMS间温差触发协调
            'soh_imbalance_threshold': 5.0,        # 5% BMS间SOH差异触发协调
            
            # 协调强度
            'max_power_bias': 0.3,                 # 最大功率偏置30%
            'max_power_limit_reduction': 0.5,      # 最大功率限制减少50%
            
            # 协调目标
            'target_soc_tolerance': 2.0,           # SOC目标容差2%
            'target_temp_tolerance': 5.0,          # 温度目标容差5℃
            
            # 协调速度
            'soc_convergence_speed': 0.1,          # SOC收敛速度系数
            'temp_convergence_speed': 0.05,        # 温度收敛速度系数
            
            # 安全限制
            'min_coordination_interval': 10.0,     # 最小协调间隔10s
            'max_coordination_duration': 300.0     # 最大协调持续时间5min
        }
        
        # === 协调状态 ===
        self.active_commands: Dict[str, CoordinationCommand] = {}
        self.coordination_history: List[Dict] = []
        self.last_coordination_time = 0.0
        
        # === 性能追踪 ===
        self.coordination_count = 0
        self.successful_coordinations = 0
        self.coordination_metrics_history: List[CoordinationMetrics] = []
        
        print(f"✅ BMS间协调器初始化完成: {coordinator_id}")
        print(f"   BMS数量: {self.num_bms}, 协调模式: {coordination_mode.value}")
    
    def generate_coordination_commands(self, current_time: float = 0.0) -> Dict[str, Dict]:
        """
        生成BMS间协调指令
        
        Args:
            current_time: 当前时间 (s)
            
        Returns:
            协调指令字典 {"BMS_01": {...}, "BMS_02": {...}}
        """
        
        if self.coordination_mode == CoordinationMode.DISABLED:
            return {}
        
        # === 1. 检查协调间隔 ===
        if (current_time - self.last_coordination_time < 
            self.coordination_params['min_coordination_interval']):
            return self._convert_active_commands_to_dict()
        
        # === 2. 收集BMS状态 ===
        bms_states = self._collect_bms_states()
        
        # === 3. 计算协调指标 ===
        metrics = self._calculate_coordination_metrics(bms_states)
        
        # === 4. 评估协调需求 ===
        coordination_needs = self._assess_coordination_needs(metrics, bms_states)
        
        # === 5. 生成协调指令 ===
        if coordination_needs['need_coordination']:
            new_commands = self._generate_specific_commands(coordination_needs, bms_states, metrics)
            
            # 更新活跃指令
            self._update_active_commands(new_commands, current_time)
            
            # 记录协调历史
            self._record_coordination_event(new_commands, metrics, current_time)
            
            self.last_coordination_time = current_time
            self.coordination_count += 1
        
        return self._convert_active_commands_to_dict()
    
    def _collect_bms_states(self) -> List[Dict]:
        """收集BMS状态"""
        
        bms_states = []
        
        for bms in self.bms_list:
            bms_summary = bms.get_bms_summary()
            
            # 增强状态信息
            enhanced_state = {
                **bms_summary,
                
                # 功率状态
                'current_power': getattr(bms.state, 'actual_power', 0.0),
                'max_power_capacity': max(bms._get_max_charge_power(), 
                                        abs(bms._get_max_discharge_power())),
                'power_utilization': self._calculate_power_utilization(bms),
                
                # 协调相关指标
                'coordination_priority': self._calculate_coordination_priority(bms_summary),
                'coordination_capacity': self._calculate_coordination_capacity(bms_summary),
                'stability_score': self._calculate_stability_score(bms)
            }
            
            bms_states.append(enhanced_state)
        
        return bms_states
    
    def _calculate_power_utilization(self, bms) -> float:
        """计算功率利用率"""
        current_power = abs(getattr(bms.state, 'actual_power', 0.0))
        max_power = max(bms._get_max_charge_power(), abs(bms._get_max_discharge_power()))
        
        if max_power > 0:
            return current_power / max_power
        else:
            return 0.0
    
    def _calculate_coordination_priority(self, bms_summary: Dict) -> float:
        """计算协调优先级 (0-1, 越高越需要协调)"""
        
        # 基于不平衡程度的优先级
        soc_imbalance = bms_summary['soc_std'] / 5.0  # 归一化到[0,1]
        temp_imbalance = bms_summary['temp_std'] / 10.0
        
        # 基于健康状态的优先级
        health_priority = 0.0
        if bms_summary['health_status'] == 'Critical':
            health_priority = 1.0
        elif bms_summary['health_status'] == 'Poor':
            health_priority = 0.7
        elif bms_summary['health_status'] == 'Fair':
            health_priority = 0.3
        
        # 综合优先级
        priority = min(1.0, soc_imbalance + temp_imbalance + health_priority)
        return priority
    
    def _calculate_coordination_capacity(self, bms_summary: Dict) -> float:
        """计算协调容量 (0-1, 越高越能配合协调)"""
        
        # 基于SOC水平的协调容量
        avg_soc = bms_summary['avg_soc']
        if 20.0 <= avg_soc <= 80.0:
            soc_capacity = 1.0  # 中等SOC最适合协调
        elif 10.0 <= avg_soc <= 90.0:
            soc_capacity = 0.7
        else:
            soc_capacity = 0.3  # 极端SOC不适合大幅协调
        
        # 基于健康状态的协调容量
        soh = bms_summary['avg_soh']
        if soh > 90.0:
            health_capacity = 1.0
        elif soh > 80.0:
            health_capacity = 0.8
        elif soh > 70.0:
            health_capacity = 0.5
        else:
            health_capacity = 0.2  # 健康度低不适合大幅协调
        
        # 基于均衡状态的协调容量
        balance_capacity = 0.8 if bms_summary['balancing_active'] else 1.0
        
        capacity = soc_capacity * health_capacity * balance_capacity
        return capacity
    
    def _calculate_stability_score(self, bms) -> float:
        """计算稳定性评分 (0-1, 越高越稳定)"""
        
        # 简化计算，基于BMS内不平衡度
        bms_summary = bms.get_bms_summary()
        
        soc_stability = max(0.0, 1.0 - bms_summary['soc_std'] / 5.0)
        temp_stability = max(0.0, 1.0 - bms_summary['temp_std'] / 10.0)
        
        stability = 0.6 * soc_stability + 0.4 * temp_stability
        return stability
    
    def _calculate_coordination_metrics(self, bms_states: List[Dict]) -> CoordinationMetrics:
        """计算协调指标"""
        
        metrics = CoordinationMetrics()
        
        # 提取关键数据
        soc_values = [state['avg_soc'] for state in bms_states]
        temp_values = [state['avg_temperature'] for state in bms_states]
        soh_values = [state['avg_soh'] for state in bms_states]
        
        # BMS间均衡指标
        metrics.inter_bms_soc_std = float(np.std(soc_values))
        metrics.inter_bms_temp_std = float(np.std(temp_values))
        metrics.inter_bms_soh_std = float(np.std(soh_values))
        
        # 系统均衡评分
        soc_balance_score = max(0.0, 1.0 - metrics.inter_bms_soc_std / 20.0)
        temp_balance_score = max(0.0, 1.0 - metrics.inter_bms_temp_std / 30.0)
        soh_balance_score = max(0.0, 1.0 - metrics.inter_bms_soh_std / 20.0)
        
        metrics.system_balance_score = (0.5 * soc_balance_score + 
                                      0.3 * temp_balance_score + 
                                      0.2 * soh_balance_score)
        
        # 协调效率 (基于历史数据)
        metrics.coordination_efficiency = self._calculate_coordination_efficiency()
        
        # 收敛速率 (基于历史变化趋势)
        if len(self.coordination_metrics_history) >= 2:
            prev_metrics = self.coordination_metrics_history[-1]
            time_interval = 1.0  # 简化为1秒间隔
            
            soc_change = prev_metrics.inter_bms_soc_std - metrics.inter_bms_soc_std
            temp_change = prev_metrics.inter_bms_temp_std - metrics.inter_bms_temp_std
            
            metrics.soc_convergence_rate = soc_change * 3600.0 / time_interval  # %/hour
            metrics.temp_convergence_rate = temp_change * 3600.0 / time_interval  # ℃/hour
        
        # 记录指标历史
        self.coordination_metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_coordination_efficiency(self) -> float:
        """计算协调效率"""
        
        if self.coordination_count == 0:
            return 1.0
        
        success_rate = self.successful_coordinations / self.coordination_count
        
        # 基于成功率和最近协调效果的综合效率
        recent_effectiveness = 1.0
        if len(self.coordination_metrics_history) >= 10:
            recent_metrics = self.coordination_metrics_history[-10:]
            balance_scores = [m.system_balance_score for m in recent_metrics]
            recent_effectiveness = np.mean(balance_scores)
        
        efficiency = 0.6 * success_rate + 0.4 * recent_effectiveness
        return efficiency
    
    def _assess_coordination_needs(self, metrics: CoordinationMetrics, bms_states: List[Dict]) -> Dict:
        """评估协调需求"""
        
        needs = {
            'need_coordination': False,
            'coordination_types': [],
            'urgency_level': 'low',
            'target_metrics': {}
        }
        
        # === SOC协调需求 ===
        if metrics.inter_bms_soc_std > self.coordination_params['soc_imbalance_threshold']:
            needs['need_coordination'] = True
            needs['coordination_types'].append('soc_balance')
            
            # 设置SOC协调目标
            soc_values = [state['avg_soc'] for state in bms_states]
            target_soc = np.mean(soc_values)
            needs['target_metrics']['target_soc'] = target_soc
            
            # 评估紧急程度
            if metrics.inter_bms_soc_std > 15.0:
                needs['urgency_level'] = 'critical'
            elif metrics.inter_bms_soc_std > 10.0:
                needs['urgency_level'] = 'high'
            else:
                needs['urgency_level'] = 'normal'
        
        # === 温度协调需求 ===
        if metrics.inter_bms_temp_std > self.coordination_params['temp_imbalance_threshold']:
            needs['need_coordination'] = True
            needs['coordination_types'].append('thermal_balance')
            
            # 设置温度协调目标
            temp_values = [state['avg_temperature'] for state in bms_states]
            target_temp = np.mean(temp_values)
            needs['target_metrics']['target_temp'] = target_temp
            
            # 温度不平衡的紧急程度评估
            if metrics.inter_bms_temp_std > 20.0:
                needs['urgency_level'] = 'critical'
            elif metrics.inter_bms_temp_std > 15.0 and needs['urgency_level'] != 'critical':
                needs['urgency_level'] = 'high'
        
        # === SOH协调需求 ===
        if metrics.inter_bms_soh_std > self.coordination_params['soh_imbalance_threshold']:
            needs['need_coordination'] = True
            needs['coordination_types'].append('lifetime_optimization')
        
        # === 综合模式评估 ===
        if (self.coordination_mode == CoordinationMode.COMPREHENSIVE and 
            metrics.system_balance_score < 0.7):
            needs['need_coordination'] = True
            if 'comprehensive' not in needs['coordination_types']:
                needs['coordination_types'].append('comprehensive')
        
        return needs
    
    def _generate_specific_commands(self, 
                                  coordination_needs: Dict,
                                  bms_states: List[Dict],
                                  metrics: CoordinationMetrics) -> Dict[str, CoordinationCommand]:
        """生成具体协调指令"""
        
        commands = {}
        
        urgency_level = coordination_needs['urgency_level']
        priority_map = {
            'low': CoordinationPriority.LOW,
            'normal': CoordinationPriority.NORMAL,
            'high': CoordinationPriority.HIGH,
            'critical': CoordinationPriority.CRITICAL
        }
        priority = priority_map[urgency_level]
        
        # === SOC均衡协调指令 ===
        if 'soc_balance' in coordination_needs['coordination_types']:
            target_soc = coordination_needs['target_metrics']['target_soc']
            soc_commands = self._generate_soc_balance_commands(
                bms_states, target_soc, priority, metrics
            )
            commands.update(soc_commands)
        
        # === 温度均衡协调指令 ===
        if 'thermal_balance' in coordination_needs['coordination_types']:
            target_temp = coordination_needs['target_metrics']['target_temp']
            temp_commands = self._generate_thermal_balance_commands(
                bms_states, target_temp, priority, metrics
            )
            commands.update(temp_commands)
        
        # === 寿命优化协调指令 ===
        if 'lifetime_optimization' in coordination_needs['coordination_types']:
            lifetime_commands = self._generate_lifetime_optimization_commands(
                bms_states, priority, metrics
            )
            commands.update(lifetime_commands)
        
        # === 综合协调指令 ===
        if 'comprehensive' in coordination_needs['coordination_types']:
            comprehensive_commands = self._generate_comprehensive_commands(
                bms_states, priority, metrics
            )
            # 合并指令，避免冲突
            commands = self._merge_coordination_commands(commands, comprehensive_commands)
        
        return commands
    
    def _generate_soc_balance_commands(self, 
                                     bms_states: List[Dict],
                                     target_soc: float,
                                     priority: CoordinationPriority,
                                     metrics: CoordinationMetrics) -> Dict[str, CoordinationCommand]:
        """生成SOC均衡协调指令"""
        
        commands = {}
        
        for bms_state in bms_states:
            bms_id = bms_state['bms_id']
            current_soc = bms_state['avg_soc']
            soc_deviation = current_soc - target_soc
            
            # 仅对偏差较大的BMS生成指令
            if abs(soc_deviation) > self.coordination_params['target_soc_tolerance']:
                
                # 计算功率偏置
                power_bias = self._calculate_soc_power_bias(soc_deviation, bms_state)
                
                # 计算协调权重
                coordination_weight = min(1.0, abs(soc_deviation) / 10.0)
                
                # 估算协调时间
                expected_duration = abs(soc_deviation) * 3600.0 / 5.0  # 假设5%/hour收敛速度
                expected_duration = min(expected_duration, 
                                      self.coordination_params['max_coordination_duration'])
                
                command = CoordinationCommand(
                    target_bms_id=bms_id,
                    command_type='soc_balance',
                    priority_level=priority,
                    suggested_power_bias=power_bias,
                    target_soc=target_soc,
                    coordination_weight=coordination_weight,
                    expected_duration=expected_duration,
                    description=f"SOC均衡: 当前{current_soc:.1f}% -> 目标{target_soc:.1f}%",
                    reasoning=f"SOC偏差{soc_deviation:.1f}%超过容差{self.coordination_params['target_soc_tolerance']:.1f}%"
                )
                
                commands[bms_id] = command
        
        return commands
    
    def _calculate_soc_power_bias(self, soc_deviation: float, bms_state: Dict) -> float:
        """计算SOC协调的功率偏置"""
        
        # 基础功率偏置
        base_bias = np.tanh(soc_deviation / 10.0) * self.coordination_params['max_power_bias']
        
        # 根据BMS协调容量调整
        coordination_capacity = bms_state['coordination_capacity']
        adjusted_bias = base_bias * coordination_capacity
        
        # 安全限制
        max_bias = self.coordination_params['max_power_bias']
        final_bias = np.clip(adjusted_bias, -max_bias, max_bias)
        
        return final_bias
    
    def _generate_thermal_balance_commands(self, 
                                         bms_states: List[Dict],
                                         target_temp: float,
                                         priority: CoordinationPriority,
                                         metrics: CoordinationMetrics) -> Dict[str, CoordinationCommand]:
        """生成温度均衡协调指令"""
        
        commands = {}
        
        for bms_state in bms_states:
            bms_id = bms_state['bms_id']
            current_temp = bms_state['avg_temperature']
            temp_deviation = current_temp - target_temp
            
            # 仅对温度偏差较大的BMS生成指令
            if abs(temp_deviation) > self.coordination_params['target_temp_tolerance']:
                
                # 温度过高的BMS需要减少功率
                if temp_deviation > 0:
                    power_limit_adjustment = max(0.5, 1.0 - temp_deviation / 20.0)
                    power_bias = -min(0.2, temp_deviation / 30.0)
                else:
                    power_limit_adjustment = 1.0
                    power_bias = 0.0  # 温度低的BMS不增加功率，避免进一步升温
                
                coordination_weight = min(1.0, abs(temp_deviation) / 15.0)
                
                # 温度协调通常较慢
                expected_duration = abs(temp_deviation) * 3600.0 / 2.0  # 假设2℃/hour收敛速度
                expected_duration = min(expected_duration, 
                                      self.coordination_params['max_coordination_duration'])
                
                command = CoordinationCommand(
                    target_bms_id=bms_id,
                    command_type='thermal_balance',
                    priority_level=priority,
                    suggested_power_bias=power_bias,
                    power_limit_adjustment=power_limit_adjustment,
                    target_temp=target_temp,
                    coordination_weight=coordination_weight,
                    expected_duration=expected_duration,
                    description=f"温度均衡: 当前{current_temp:.1f}℃ -> 目标{target_temp:.1f}℃",
                    reasoning=f"温度偏差{temp_deviation:.1f}℃超过容差{self.coordination_params['target_temp_tolerance']:.1f}℃"
                )
                
                commands[bms_id] = command
        
        return commands
    
    def _generate_lifetime_optimization_commands(self, 
                                               bms_states: List[Dict],
                                               priority: CoordinationPriority,
                                               metrics: CoordinationMetrics) -> Dict[str, CoordinationCommand]:
        """生成寿命优化协调指令"""
        
        commands = {}
        
        # 计算SOH统计
        soh_values = [state['avg_soh'] for state in bms_states]
        soh_mean = np.mean(soh_values)
        soh_std = np.std(soh_values)
        
        for bms_state in bms_states:
            bms_id = bms_state['bms_id']
            current_soh = bms_state['avg_soh']
            soh_deviation = current_soh - soh_mean
            
            # 对健康度差异较大的BMS生成协调指令
            if abs(soh_deviation) > self.coordination_params['soh_imbalance_threshold']:
                
                # 健康度低的BMS需要保护
                if soh_deviation < -2.0:  # SOH低于平均值2%以上
                    power_limit_adjustment = max(0.6, 1.0 + soh_deviation / 20.0)
                    power_bias = max(-0.3, soh_deviation / 30.0)
                    description = f"寿命保护: SOH{current_soh:.1f}%较低，减少负荷"
                    reasoning = f"SOH低于平均值{abs(soh_deviation):.1f}%，需要保护性协调"
                
                # 健康度高的BMS可以承担更多负荷
                elif soh_deviation > 2.0:  # SOH高于平均值2%以上
                    power_limit_adjustment = min(1.0, 1.0 + soh_deviation / 50.0)
                    power_bias = min(0.2, soh_deviation / 40.0)
                    description = f"负荷均衡: SOH{current_soh:.1f}%较高，可增加负荷"
                    reasoning = f"SOH高于平均值{soh_deviation:.1f}%，可承担更多负荷"
                
                else:
                    continue  # 中等偏差不需要协调
                
                coordination_weight = min(1.0, abs(soh_deviation) / 10.0)
                
                # 寿命优化是长期过程
                expected_duration = self.coordination_params['max_coordination_duration']
                
                command = CoordinationCommand(
                    target_bms_id=bms_id,
                    command_type='lifetime_optimization',
                    priority_level=priority,
                    suggested_power_bias=power_bias,
                    power_limit_adjustment=power_limit_adjustment,
                    coordination_weight=coordination_weight,
                    expected_duration=expected_duration,
                    description=description,
                    reasoning=reasoning
                )
                
                commands[bms_id] = command
        
        return commands
    
    def _generate_comprehensive_commands(self, 
                                       bms_states: List[Dict],
                                       priority: CoordinationPriority,
                                       metrics: CoordinationMetrics) -> Dict[str, CoordinationCommand]:
        """生成综合协调指令"""
        
        commands = {}
        
        # 计算各BMS的综合协调需求
        for bms_state in bms_states:
            bms_id = bms_state['bms_id']
            
            # 综合评估该BMS的协调需求
            comprehensive_score = self._calculate_comprehensive_coordination_score(bms_state, bms_states)
            
            # 仅对需要协调的BMS生成指令
            if abs(comprehensive_score) > 0.3:  # 阈值调整
                
                # 基于综合评分计算协调参数
                power_bias = np.tanh(comprehensive_score) * self.coordination_params['max_power_bias']
                
                if comprehensive_score < 0:  # 需要减少负荷
                    power_limit_adjustment = max(0.7, 1.0 + comprehensive_score * 0.3)
                else:  # 可以增加负荷
                    power_limit_adjustment = min(1.0, 1.0 + comprehensive_score * 0.1)
                
                coordination_weight = min(1.0, abs(comprehensive_score))
                expected_duration = self.coordination_params['max_coordination_duration'] * 0.8
                
                command = CoordinationCommand(
                    target_bms_id=bms_id,
                    command_type='comprehensive',
                    priority_level=priority,
                    suggested_power_bias=power_bias,
                    power_limit_adjustment=power_limit_adjustment,
                    coordination_weight=coordination_weight,
                    expected_duration=expected_duration,
                    description=f"综合协调: 协调评分{comprehensive_score:.2f}",
                    reasoning=f"基于SOC、温度、SOH的综合评估需要协调"
                )
                
                commands[bms_id] = command
        
        return commands
    
    def _calculate_comprehensive_coordination_score(self, 
                                                  target_bms_state: Dict,
                                                  all_bms_states: List[Dict]) -> float:
        """计算综合协调评分 (-1到1, 负值需要减少负荷，正值可以增加负荷)"""
        
        # 提取系统平均值
        all_socs = [state['avg_soc'] for state in all_bms_states]
        all_temps = [state['avg_temperature'] for state in all_bms_states]
        all_sohs = [state['avg_soh'] for state in all_bms_states]
        
        system_avg_soc = np.mean(all_socs)
        system_avg_temp = np.mean(all_temps)
        system_avg_soh = np.mean(all_sohs)
        
        # 计算该BMS与系统平均值的偏差
        soc_deviation = target_bms_state['avg_soc'] - system_avg_soc
        temp_deviation = target_bms_state['avg_temperature'] - system_avg_temp
        soh_deviation = target_bms_state['avg_soh'] - system_avg_soh
        
        # 归一化偏差到[-1, 1]范围
        soc_score = np.tanh(soc_deviation / 10.0)  # SOC偏差影响
        temp_score = -np.tanh(temp_deviation / 15.0)  # 温度高的需要减少负荷
        soh_score = np.tanh(soh_deviation / 10.0)  # SOH高的可以增加负荷
        
        # 加权综合评分
        comprehensive_score = (0.4 * soc_score +     # SOC权重40%
                             0.3 * temp_score +      # 温度权重30%
                             0.3 * soh_score)        # SOH权重30%
        
        # 考虑BMS内部状态
        internal_balance_factor = 1.0
        if target_bms_state['soc_std'] > 3.0 or target_bms_state['temp_std'] > 8.0:
            internal_balance_factor = 0.7  # 内部不平衡的BMS降低协调强度
        
        final_score = comprehensive_score * internal_balance_factor
        return np.clip(final_score, -1.0, 1.0)
    
    def _merge_coordination_commands(self, 
                                   commands1: Dict[str, CoordinationCommand],
                                   commands2: Dict[str, CoordinationCommand]) -> Dict[str, CoordinationCommand]:
        """合并协调指令，避免冲突"""
        
        merged_commands = commands1.copy()
        
        for bms_id, command2 in commands2.items():
            if bms_id in merged_commands:
                # 合并指令
                command1 = merged_commands[bms_id]
                merged_command = self._combine_commands(command1, command2)
                merged_commands[bms_id] = merged_command
            else:
                merged_commands[bms_id] = command2
        
        return merged_commands
    
    def _combine_commands(self, 
                         command1: CoordinationCommand,
                         command2: CoordinationCommand) -> CoordinationCommand:
        """合并两个协调指令"""
        
        # 选择优先级更高的指令作为主指令
        if command1.priority_level.value >= command2.priority_level.value:
            primary_command = command1
            secondary_command = command2
        else:
            primary_command = command2
            secondary_command = command1
        
        # 合并功率偏置（取平均值，避免过度调整）
        combined_power_bias = (primary_command.suggested_power_bias + 
                             secondary_command.suggested_power_bias) / 2.0
        combined_power_bias = np.clip(combined_power_bias, -0.3, 0.3)
        
        # 合并功率限制调整（取更保守的值）
        combined_power_limit = min(primary_command.power_limit_adjustment,
                                 secondary_command.power_limit_adjustment)
        
        # 合并协调权重（取平均值）
        combined_weight = (primary_command.coordination_weight + 
                         secondary_command.coordination_weight) / 2.0
        
        # 创建合并后的指令
        combined_command = CoordinationCommand(
            target_bms_id=primary_command.target_bms_id,
            command_type=f"{primary_command.command_type}+{secondary_command.command_type}",
            priority_level=primary_command.priority_level,
            suggested_power_bias=combined_power_bias,
            power_limit_adjustment=combined_power_limit,
            target_soc=primary_command.target_soc,
            target_temp=primary_command.target_temp,
            coordination_weight=combined_weight,
            expected_duration=max(primary_command.expected_duration, 
                                secondary_command.expected_duration),
            description=f"合并协调: {primary_command.description}; {secondary_command.description}",
            reasoning=f"合并原因: {primary_command.reasoning}; {secondary_command.reasoning}"
        )
        
        return combined_command
    
    def _update_active_commands(self, 
                              new_commands: Dict[str, CoordinationCommand],
                              current_time: float):
        """更新活跃协调指令"""
        
        # 移除过期的指令
        expired_commands = []
        for bms_id, command in self.active_commands.items():
            if (current_time - self.last_coordination_time) > command.expected_duration:
                expired_commands.append(bms_id)
        
        for bms_id in expired_commands:
            del self.active_commands[bms_id]
        
        # 添加新指令
        self.active_commands.update(new_commands)
    
    def _convert_active_commands_to_dict(self) -> Dict[str, Dict]:
        """将活跃指令转换为字典格式"""
        
        command_dict = {}
        
        for bms_id, command in self.active_commands.items():
            command_dict[bms_id] = {
                'command_type': command.command_type,
                'priority_level': command.priority_level.value,
                'suggested_power_bias': command.suggested_power_bias,
                'power_limit_adjustment': command.power_limit_adjustment,
                'target_soc': command.target_soc,
                'target_temp': command.target_temp,
                'coordination_weight': command.coordination_weight,
                'expected_duration': command.expected_duration,
                'description': command.description,
                'reasoning': command.reasoning
            }
        
        return command_dict
    
    def _record_coordination_event(self, 
                                 commands: Dict[str, CoordinationCommand],
                                 metrics: CoordinationMetrics,
                                 current_time: float):
        """记录协调事件"""
        
        coordination_event = {
            'timestamp': current_time,
            'coordination_count': self.coordination_count,
            'commands_issued': len(commands),
            'commands': {bms_id: cmd.__dict__ for bms_id, cmd in commands.items()},
            'metrics_before': metrics.__dict__,
            'system_balance_score': metrics.system_balance_score
        }
        
        self.coordination_history.append(coordination_event)
        
        # 维护历史长度
        max_history = 100
        if len(self.coordination_history) > max_history:
            self.coordination_history.pop(0)
    
    def evaluate_coordination_effectiveness(self) -> Dict[str, float]:
        """评估协调效果"""
        
        if len(self.coordination_metrics_history) < 10:
            return {'error': 'Insufficient coordination history'}
        
        # 分析最近的协调效果
        recent_metrics = self.coordination_metrics_history[-10:]
        
        # 计算均衡改善趋势
        soc_std_trend = self._calculate_trend([m.inter_bms_soc_std for m in recent_metrics])
        temp_std_trend = self._calculate_trend([m.inter_bms_temp_std for m in recent_metrics])
        balance_score_trend = self._calculate_trend([m.system_balance_score for m in recent_metrics])
        
        # 计算协调效率
        avg_coordination_efficiency = np.mean([m.coordination_efficiency for m in recent_metrics])
        
        # 计算收敛速度
        avg_soc_convergence = np.mean([abs(m.soc_convergence_rate) for m in recent_metrics])
        avg_temp_convergence = np.mean([abs(m.temp_convergence_rate) for m in recent_metrics])
        
        effectiveness = {
            'overall_effectiveness': avg_coordination_efficiency,
            'soc_balance_trend': soc_std_trend,
            'temp_balance_trend': temp_std_trend,
            'balance_score_trend': balance_score_trend,
            'avg_soc_convergence_rate': avg_soc_convergence,
            'avg_temp_convergence_rate': avg_temp_convergence,
            'coordination_success_rate': self.successful_coordinations / self.coordination_count if self.coordination_count > 0 else 0.0,
            'total_coordinations': self.coordination_count,
            'active_commands_count': len(self.active_commands)
        }
        
        return effectiveness
    
    def _calculate_trend(self, data: List[float]) -> str:
        """计算数据趋势"""
        if len(data) < 3:
            return "insufficient_data"
        
        # 简单的线性趋势分析
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        slope = coeffs[0]
        
        if slope < -0.1:
            return "improving"  # 对于标准差，下降是改善
        elif slope > 0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def get_coordination_status(self) -> Dict:
        """获取协调状态"""
        
        current_metrics = self.coordination_metrics_history[-1] if self.coordination_metrics_history else None
        
        status = {
            'coordinator_id': self.coordinator_id,
            'coordination_mode': self.coordination_mode.value,
            'num_bms': self.num_bms,
            'total_coordinations': self.coordination_count,
            'successful_coordinations': self.successful_coordinations,
            'active_commands_count': len(self.active_commands),
            
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            
            'active_commands': {
                bms_id: {
                    'type': cmd.command_type,
                    'priority': cmd.priority_level.value,
                    'power_bias': cmd.suggested_power_bias,
                    'description': cmd.description
                } for bms_id, cmd in self.active_commands.items()
            },
            
            'coordination_parameters': self.coordination_params.copy()
        }
        
        return status
    
    def reset(self):
        """重置协调器"""
        self.active_commands.clear()
        self.coordination_history.clear()
        self.coordination_metrics_history.clear()
        
        self.coordination_count = 0
        self.successful_coordinations = 0
        self.last_coordination_time = 0.0
        
        print(f"🔄 BMS间协调器 {self.coordinator_id} 已重置")
    
    def update_coordination_mode(self, new_mode: CoordinationMode) -> bool:
        """更新协调模式"""
        try:
            old_mode = self.coordination_mode
            self.coordination_mode = new_mode
            
            # 清除不兼容的活跃指令
            if new_mode == CoordinationMode.DISABLED:
                self.active_commands.clear()
            
            print(f"🔄 协调器 {self.coordinator_id} 模式更新: {old_mode.value} -> {new_mode.value}")
            return True
        except Exception as e:
            print(f"❌ 协调模式更新失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"InterBMSCoordinator({self.coordinator_id}): "
                f"模式={self.coordination_mode.value}, "
                f"BMS数={self.num_bms}, "
                f"活跃指令={len(self.active_commands)}, "
                f"协调次数={self.coordination_count}")
