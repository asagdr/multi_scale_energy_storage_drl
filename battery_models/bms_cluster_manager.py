"""
BMS集群管理器
管理10个独立BMS，实现系统级功率分配和协调
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig
from battery_models.bms_model import BMSModel
from battery_models.central_power_allocator import CentralPowerAllocator
from battery_models.inter_bms_coordinator import InterBMSCoordinator
from battery_models.multi_level_cost_model import MultiLevelCostModel

@dataclass
class ClusterState:
    """BMS集群状态"""
    cluster_id: str
    
    # 系统级状态
    system_avg_soc: float = 50.0
    system_avg_temp: float = 25.0
    system_avg_soh: float = 100.0
    
    # BMS间均衡指标 (关键指标)
    inter_bms_soc_std: float = 0.0      # BMS间SOC不平衡度
    inter_bms_temp_std: float = 0.0     # BMS间温度不平衡度
    inter_bms_soh_std: float = 0.0      # BMS间SOH不平衡度
    
    # BMS内均衡指标 (关键指标)
    avg_intra_bms_soc_std: float = 0.0  # 平均BMS内SOC不平衡度
    avg_intra_bms_temp_std: float = 0.0 # 平均BMS内温度不平衡度
    
    # 功率状态
    total_actual_power: float = 0.0
    total_power_command: float = 0.0
    system_power_efficiency: float = 1.0
    
    # 成本状态 (关键指标)
    total_system_cost: float = 0.0
    system_cost_increase_rate: float = 0.0

class BMSClusterManager:
    """
    BMS集群管理器
    管理10个独立BMS，实现系统级优化
    """
    
    def __init__(self, 
                 battery_params: BatteryParams,
                 system_config: SystemConfig,
                 num_bms: int = 10,
                 cluster_id: str = "BMSCluster_001"):
        """
        初始化BMS集群管理器
        
        Args:
            battery_params: 电池参数
            system_config: 系统配置
            num_bms: BMS数量 (默认10个)
            cluster_id: 集群标识
        """
        self.battery_params = battery_params
        self.system_config = system_config
        self.num_bms = num_bms
        self.cluster_id = cluster_id
        self.cells_per_bms = battery_params.total_cells // num_bms  # 100单体/BMS
        
        # === 创建10个独立BMS ===
        self.bms_list: List[BMSModel] = []
        for i in range(num_bms):
            bms = BMSModel(
                bms_id=f"BMS_{i+1:02d}",
                cells_count=self.cells_per_bms,
                battery_params=battery_params
            )
            self.bms_list.append(bms)
        
        # === 中央功率分配器 ===
        self.power_allocator = CentralPowerAllocator(
            bms_list=self.bms_list,
            allocator_id=f"{cluster_id}_PowerAllocator"
        )
        
        # === BMS间协调器 ===
        self.inter_bms_coordinator = InterBMSCoordinator(
            bms_list=self.bms_list,
            coordinator_id=f"{cluster_id}_Coordinator"
        )
        
        # === 多层级成本模型 ===
        self.cost_model = MultiLevelCostModel(
            bms_list=self.bms_list,
            cost_model_id=f"{cluster_id}_CostModel"
        )
        
        # === 集群状态 ===
        self.state = ClusterState(cluster_id=cluster_id)
        
        # === 仿真统计 ===
        self.step_count = 0
        self.total_time = 0.0
        self.cluster_history: List[Dict] = []
        
        print(f"✅ BMS集群管理器初始化完成: {cluster_id}")
        print(f"   BMS数量: {num_bms}, 每BMS单体数: {self.cells_per_bms}")
        print(f"   总单体数: {battery_params.total_cells}")
    
    def step(self, 
             total_power_command: float,
             delta_t: float,
             upper_layer_weights: Optional[Dict[str, float]] = None,
             ambient_temperature: float = 25.0) -> Dict:
        """
        集群仿真步
        
        Args:
            total_power_command: 系统总功率指令 (W)
            delta_t: 时间步长 (s)
            upper_layer_weights: 上层权重 {'soc_balance': 0.3, 'temp_balance': 0.2, 'lifetime': 0.3}
            ambient_temperature: 环境温度 (℃)
            
        Returns:
            集群仿真记录
        """
        
        if upper_layer_weights is None:
            upper_layer_weights = {
                'soc_balance': 0.3,
                'temp_balance': 0.2,
                'lifetime': 0.3,
                'efficiency': 0.2
            }
        
        # === 1. 中央功率分配 ===
        power_allocation = self.power_allocator.allocate_power(
            total_power_command=total_power_command,
            upper_layer_weights=upper_layer_weights
        )
        
        # === 2. BMS间协调 ===
        coordination_commands = self.inter_bms_coordinator.generate_coordination_commands()
        
        # === 3. 各BMS并行仿真 ===
        bms_records = []
        for i, bms in enumerate(self.bms_list):
            # 获取分配的功率
            allocated_power = power_allocation[bms.bms_id]
            
            # 应用协调指令调整
            if bms.bms_id in coordination_commands:
                coord_cmd = coordination_commands[bms.bms_id]
                power_bias = coord_cmd.get('suggested_power_bias', 0.0)
                allocated_power *= (1.0 + power_bias)
            
            # 执行BMS仿真
            bms_record = bms.step(
                bms_power_command=allocated_power,
                delta_t=delta_t,
                ambient_temperature=ambient_temperature
            )
            
            bms_records.append(bms_record)
        
        # === 4. 更新集群状态 ===
        self._update_cluster_state(bms_records, total_power_command)
        
        # === 5. 多层级成本计算 ===
        system_cost_breakdown = self.cost_model.calculate_total_system_cost(bms_records)
        
        # === 6. 集群级指标计算 ===
        cluster_metrics = self._calculate_cluster_metrics(bms_records)
        
        # === 7. 构建集群记录 ===
        cluster_record = {
            'cluster_id': self.cluster_id,
            'step_count': self.step_count,
            'simulation_time': self.total_time,
            
            # BMS记录
            'bms_records': bms_records,
            'num_bms': self.num_bms,
            'total_cells': self.num_bms * self.cells_per_bms,
            
            # 系统级状态 (关键指标)
            'system_avg_soc': self.state.system_avg_soc,
            'system_avg_temp': self.state.system_avg_temp,
            'system_avg_soh': self.state.system_avg_soh,
            
            # BMS间均衡指标 (关键指标)
            'inter_bms_soc_std': self.state.inter_bms_soc_std,
            'inter_bms_temp_std': self.state.inter_bms_temp_std,
            'inter_bms_soh_std': self.state.inter_bms_soh_std,
            
            # BMS内均衡指标 (关键指标)
            'avg_intra_bms_soc_std': self.state.avg_intra_bms_soc_std,
            'avg_intra_bms_temp_std': self.state.avg_intra_bms_temp_std,
            
            # 功率状态
            'total_actual_power': self.state.total_actual_power,
            'total_power_command': total_power_command,
            'system_power_efficiency': self.state.system_power_efficiency,
            'power_tracking_error': abs(self.state.total_actual_power - total_power_command),
            
            # 功率分配结果
            'power_allocation': power_allocation,
            'coordination_commands': coordination_commands,
            
            # 多层级成本 (关键指标)
            'total_system_cost': system_cost_breakdown['total_system_cost'],
            'system_cost_increase_rate': self.state.system_cost_increase_rate,
            'cost_breakdown': system_cost_breakdown,
            
            # 集群指标
            'cluster_metrics': cluster_metrics,
            
            # 约束和安全状态
            'system_constraints_active': self._check_system_constraints(bms_records),
            'system_health_status': self._calculate_system_health_status(bms_records),
            'system_warning_count': self._count_system_warnings(bms_records),
            'system_alarm_count': self._count_system_alarms(bms_records)
        }
        
        # === 8. 记录历史 ===
        self.cluster_history.append(cluster_record)
        self.step_count += 1
        self.total_time += delta_t
        
        # 维护历史长度
        max_history = self.system_config.MAX_HISTORY_LENGTH
        if len(self.cluster_history) > max_history:
            self.cluster_history.pop(0)
        
        return cluster_record
    
    def _update_cluster_state(self, bms_records: List[Dict], total_power_command: float):
        """更新集群状态"""
        
        # 提取BMS级数据
        bms_socs = [record['avg_soc'] for record in bms_records]
        bms_temps = [record['avg_temperature'] for record in bms_records]
        bms_sohs = [record['avg_soh'] for record in bms_records]
        bms_powers = [record['actual_power'] for record in bms_records]
        
        # BMS内不平衡度
        intra_bms_soc_stds = [record['soc_std'] for record in bms_records]
        intra_bms_temp_stds = [record['temp_std'] for record in bms_records]
        
        # 更新系统级状态
        self.state.system_avg_soc = float(np.mean(bms_socs))
        self.state.system_avg_temp = float(np.mean(bms_temps))
        self.state.system_avg_soh = float(np.mean(bms_sohs))
        
        # 更新BMS间均衡指标 (关键指标)
        self.state.inter_bms_soc_std = float(np.std(bms_socs))      # BMS间SOC不平衡度
        self.state.inter_bms_temp_std = float(np.std(bms_temps))    # BMS间温度不平衡度
        self.state.inter_bms_soh_std = float(np.std(bms_sohs))      # BMS间SOH不平衡度
        
        # 更新BMS内均衡指标 (关键指标)
        self.state.avg_intra_bms_soc_std = float(np.mean(intra_bms_soc_stds))   # 平均BMS内SOC不平衡度
        self.state.avg_intra_bms_temp_std = float(np.mean(intra_bms_temp_stds)) # 平均BMS内温度不平衡度
        
        # 更新功率状态
        self.state.total_actual_power = float(np.sum(bms_powers))
        self.state.total_power_command = total_power_command
        
        # 计算系统功率效率
        if total_power_command != 0:
            self.state.system_power_efficiency = self.state.total_actual_power / total_power_command
        else:
            self.state.system_power_efficiency = 1.0
    
    def _calculate_cluster_metrics(self, bms_records: List[Dict]) -> Dict:
        """计算集群级指标"""
        
        # 收集统计数据
        all_cell_socs = []
        all_cell_temps = []
        all_bms_costs = []
        
        for bms_record in bms_records:
            for cell in bms_record['cells']:
                all_cell_socs.append(cell['soc'])
                all_cell_temps.append(cell['temperature'])
            
            all_bms_costs.append(bms_record['bms_total_cost'])
        
        return {
            # 全系统单体级统计
            'all_cells_soc_std': float(np.std(all_cell_socs)),
            'all_cells_temp_std': float(np.std(all_cell_temps)),
            'all_cells_soc_range': float(np.max(all_cell_socs) - np.min(all_cell_socs)),
            'all_cells_temp_range': float(np.max(all_cell_temps) - np.min(all_cell_temps)),
            
            # BMS级统计
            'bms_cost_std': float(np.std(all_bms_costs)),
            'bms_cost_range': float(np.max(all_bms_costs) - np.min(all_bms_costs)),
            
            # 系统均衡评分 (0-1, 1为完美均衡)
            'soc_balance_score': self._calculate_balance_score('soc', bms_records),
            'temp_balance_score': self._calculate_balance_score('temp', bms_records),
            'overall_balance_score': self._calculate_overall_balance_score(bms_records),
            
            # 系统效率指标
            'energy_efficiency': self._calculate_energy_efficiency(bms_records),
            'thermal_efficiency': self._calculate_thermal_efficiency(bms_records),
            
            # 安全指标
            'safety_margin_soc': self._calculate_safety_margin('soc', bms_records),
            'safety_margin_temp': self._calculate_safety_margin('temp', bms_records)
        }
    
    def _calculate_balance_score(self, metric_type: str, bms_records: List[Dict]) -> float:
        """计算均衡评分"""
        
        if metric_type == 'soc':
            # BMS间 + BMS内SOC均衡评分
            inter_std = self.state.inter_bms_soc_std
            intra_std = self.state.avg_intra_bms_soc_std
            
            inter_score = max(0.0, 1.0 - inter_std / 10.0)  # 10%为完全不平衡
            intra_score = max(0.0, 1.0 - intra_std / 5.0)   # 5%为完全不平衡
            
            return 0.6 * inter_score + 0.4 * intra_score
        
        elif metric_type == 'temp':
            # BMS间 + BMS内温度均衡评分
            inter_std = self.state.inter_bms_temp_std
            intra_std = self.state.avg_intra_bms_temp_std
            
            inter_score = max(0.0, 1.0 - inter_std / 15.0)  # 15℃为完全不平衡
            intra_score = max(0.0, 1.0 - intra_std / 8.0)   # 8℃为完全不平衡
            
            return 0.6 * inter_score + 0.4 * intra_score
        
        else:
            return 0.5  # 默认中等评分
    
    def _calculate_overall_balance_score(self, bms_records: List[Dict]) -> float:
        """计算总体均衡评分"""
        soc_score = self._calculate_balance_score('soc', bms_records)
        temp_score = self._calculate_balance_score('temp', bms_records)
        
        return 0.7 * soc_score + 0.3 * temp_score
    
    def _calculate_energy_efficiency(self, bms_records: List[Dict]) -> float:
        """计算能量效率"""
        total_efficiency = 0.0
        for record in bms_records:
            total_efficiency += record.get('power_efficiency', 1.0)
        
        return total_efficiency / len(bms_records)
    
    def _calculate_thermal_efficiency(self, bms_records: List[Dict]) -> float:
        """计算热效率"""
        # 简化计算：基于温度均匀性
        temp_balance_score = self._calculate_balance_score('temp', bms_records)
        
        # 考虑平均温度与最优温度的偏差
        optimal_temp = 25.0
        temp_deviation = abs(self.state.system_avg_temp - optimal_temp)
        temp_optimality = max(0.0, 1.0 - temp_deviation / 20.0)
        
        return 0.6 * temp_balance_score + 0.4 * temp_optimality
    
    def _calculate_safety_margin(self, metric_type: str, bms_records: List[Dict]) -> float:
        """计算安全裕度"""
        
        if metric_type == 'soc':
            min_soc = min(record['avg_soc'] for record in bms_records)
            max_soc = max(record['avg_soc'] for record in bms_records)
            
            lower_margin = (min_soc - self.battery_params.MIN_SOC) / self.battery_params.MIN_SOC
            upper_margin = (self.battery_params.MAX_SOC - max_soc) / self.battery_params.MAX_SOC
            
            return min(lower_margin, upper_margin)
        
        elif metric_type == 'temp':
            min_temp = min(record['avg_temperature'] for record in bms_records)
            max_temp = max(record['avg_temperature'] for record in bms_records)
            
            lower_margin = (min_temp - self.battery_params.MIN_TEMP) / abs(self.battery_params.MIN_TEMP)
            upper_margin = (self.battery_params.MAX_TEMP - max_temp) / self.battery_params.MAX_TEMP
            
            return min(lower_margin, upper_margin)
        
        else:
            return 0.5
    
    def _check_system_constraints(self, bms_records: List[Dict]) -> Dict[str, bool]:
        """检查系统级约束"""
        return {
            'thermal_constraints': any(record.get('thermal_constraints_active', False) for record in bms_records),
            'voltage_constraints': any(record.get('voltage_constraints_active', False) for record in bms_records),
            'soc_constraints': (self.state.system_avg_soc < 10.0 or self.state.system_avg_soc > 90.0),
            'balance_constraints': (self.state.inter_bms_soc_std > 10.0 or self.state.avg_intra_bms_soc_std > 5.0)
        }
    
    def _calculate_system_health_status(self, bms_records: List[Dict]) -> str:
        """计算系统健康状态"""
        if self.state.system_avg_soh < 70:
            return "Critical"
        elif self.state.system_avg_soh < 80:
            return "Poor"
        elif any(record['health_status'] == 'Critical' for record in bms_records):
            return "Poor"
        elif any(record['health_status'] == 'Poor' for record in bms_records):
            return "Fair"
        else:
            return "Good"
    
    def _count_system_warnings(self, bms_records: List[Dict]) -> int:
        """统计系统警告数量"""
        warning_count = 0
        for record in bms_records:
            warning_count += len(record.get('warning_flags', []))
        return warning_count
    
    def _count_system_alarms(self, bms_records: List[Dict]) -> int:
        """统计系统报警数量"""
        alarm_count = 0
        for record in bms_records:
            alarm_count += len(record.get('alarm_flags', []))
        return alarm_count
    
    def reset(self, 
              target_soc: float = 50.0,
              target_temp: float = 25.0,
              add_inter_bms_variation: bool = True,
              add_intra_bms_variation: bool = True) -> Dict:
        """
        重置BMS集群
        
        Args:
            target_soc: 目标SOC (%)
            target_temp: 目标温度 (℃)
            add_inter_bms_variation: 是否添加BMS间变化
            add_intra_bms_variation: 是否添加BMS内变化
            
        Returns:
            重置结果
        """
        
        reset_results = []
        
        for i, bms in enumerate(self.bms_list):
            if add_inter_bms_variation:
                # BMS间添加变化
                bms_target_soc = target_soc + np.random.normal(0, 2.0)  # ±2%变化
                bms_target_temp = target_temp + np.random.normal(0, 3.0)  # ±3℃变化
            else:
                bms_target_soc = target_soc
                bms_target_temp = target_temp
            
            bms_result = bms.reset(
                target_soc=np.clip(bms_target_soc, 10.0, 90.0),
                target_temp=np.clip(bms_target_temp, 15.0, 35.0),
                add_variation=add_intra_bms_variation
            )
            
            reset_results.append(bms_result)
        
        # 重置集群状态
        self.state = ClusterState(cluster_id=self.cluster_id)
        self.step_count = 0
        self.total_time = 0.0
        self.cluster_history.clear()
        
        # 重置各组件
        self.power_allocator.reset()
        self.inter_bms_coordinator.reset()
        self.cost_model.reset()
        
        print(f"🔄 BMS集群 {self.cluster_id} 已重置")
        print(f"   目标SOC: {target_soc:.1f}%, 目标温度: {target_temp:.1f}℃")
        print(f"   BMS间变化: {add_inter_bms_variation}, BMS内变化: {add_intra_bms_variation}")
        
        return {
            'cluster_id': self.cluster_id,
            'num_bms': self.num_bms,
            'total_cells': self.num_bms * self.cells_per_bms,
            'bms_reset_results': reset_results,
            'reset_complete': True
        }
    
    def get_cluster_summary(self) -> Dict:
        """获取集群摘要"""
        bms_summaries = [bms.get_bms_summary() for bms in self.bms_list]
        
        return {
            'cluster_id': self.cluster_id,
            'num_bms': self.num_bms,
            'total_cells': self.num_bms * self.cells_per_bms,
            'step_count': self.step_count,
            'simulation_time': self.total_time,
            
            # 系统级状态
            'system_avg_soc': self.state.system_avg_soc,
            'system_avg_temp': self.state.system_avg_temp,
            'system_avg_soh': self.state.system_avg_soh,
            
            # 均衡指标
            'inter_bms_soc_std': self.state.inter_bms_soc_std,
            'inter_bms_temp_std': self.state.inter_bms_temp_std,
            'avg_intra_bms_soc_std': self.state.avg_intra_bms_soc_std,
            'avg_intra_bms_temp_std': self.state.avg_intra_bms_temp_std,
            
            # 成本
            'total_system_cost': self.state.total_system_cost,
            
            # BMS详细信息
            'bms_summaries': bms_summaries
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"BMSCluster({self.cluster_id}): "
                f"{self.num_bms}xBMS, "
                f"SOC={self.state.system_avg_soc:.1f}%, "
                f"σ_inter={self.state.inter_bms_soc_std:.2f}%, "
                f"σ_intra={self.state.avg_intra_bms_soc_std:.2f}%")
