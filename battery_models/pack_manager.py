"""
电池组管理器 - 兼容接口版本
为了向后兼容，保留原有接口，内部使用BMS集群管理器
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig
from battery_models.bms_cluster_manager import BMSClusterManager

class PackManager:
    """
    电池组管理器 - 兼容接口
    内部使用BMS集群管理器，对外提供原有接口
    """
    
    def __init__(self, 
                 pack_model,
                 manager_id: str = "PackManager_001",
                 battery_params: Optional[BatteryParams] = None,
                 system_config: Optional[SystemConfig] = None):
        """
        初始化电池组管理器
        
        Args:
            pack_model: 电池组模型（为了兼容性保留）
            manager_id: 管理器ID
            battery_params: 电池参数
            system_config: 系统配置
        """
        self.manager_id = manager_id
        self.pack_model = pack_model  # 保留原有引用
        
        # 获取参数
        if battery_params is None:
            battery_params = BatteryParams()
        if system_config is None:
            system_config = SystemConfig()
        
        # === 核心：使用BMS集群管理器 ===
        self.bms_cluster = BMSClusterManager(
            battery_params=battery_params,
            system_config=system_config,
            num_bms=battery_params.NUM_BMS,
            cluster_id=f"Cluster_{manager_id}"
        )
        
        # === 兼容性参数 ===
        self.battery_params = battery_params
        self.system_config = system_config
        
        # === 管理状态 ===
        self.is_active = True
        self.management_mode = "bms_cluster"  # 标识使用BMS集群模式
        
        # === 历史记录（兼容性） ===
        self.pack_history: List[Dict] = []
        
        print(f"✅ 电池组管理器初始化完成: {manager_id} (BMS集群模式)")
    
    def step(self, 
             pack_power_command: float, 
             delta_t: float,
             ambient_temperature: float = 25.0,
             enable_balancing: bool = True,
             upper_layer_weights: Optional[Dict[str, float]] = None) -> Dict:
        """
        执行管理步骤 - 兼容接口
        
        Args:
            pack_power_command: 电池组功率指令 (W)
            delta_t: 时间步长 (s)
            ambient_temperature: 环境温度 (℃)
            enable_balancing: 是否启用均衡
            upper_layer_weights: 上层权重
            
        Returns:
            电池组记录（兼容格式）
        """
        
        # 设置默认权重
        if upper_layer_weights is None:
            upper_layer_weights = {
                'soc_balance': 0.3,
                'temp_balance': 0.2,
                'lifetime': 0.3,
                'efficiency': 0.2
            }
        
        # === 调用BMS集群管理器 ===
        cluster_record = self.bms_cluster.step(
            total_power_command=pack_power_command,
            delta_t=delta_t,
            upper_layer_weights=upper_layer_weights,
            ambient_temperature=ambient_temperature
        )
        
        # === 转换为兼容格式 ===
        pack_record = self._convert_cluster_to_pack_record(cluster_record)
        
        # === 记录历史 ===
        self.pack_history.append(pack_record)
        
        # 维护历史长度
        max_history = getattr(self.system_config, 'MAX_HISTORY_LENGTH', 1000)
        if len(self.pack_history) > max_history:
            self.pack_history.pop(0)
        
        return pack_record
    
    def _convert_cluster_to_pack_record(self, cluster_record: Dict) -> Dict:
        """将集群记录转换为电池组记录格式"""
        
        # 计算兼容的电池组级指标
        pack_record = {
            # === 基础信息 ===
            'manager_id': self.manager_id,
            'management_mode': self.management_mode,
            'timestamp': cluster_record.get('step_count', 0),
            'simulation_time': cluster_record.get('simulation_time', 0.0),
            
            # === 系统级状态（原pack级别） ===
            'pack_soc': cluster_record.get('system_avg_soc', 50.0),
            'pack_temperature': cluster_record.get('system_avg_temp', 25.0),
            'pack_soh': cluster_record.get('system_avg_soh', 100.0),
            'pack_voltage': self._calculate_pack_voltage(cluster_record),
            'pack_current': self._calculate_pack_current(cluster_record),
            'pack_power': cluster_record.get('total_actual_power', 0.0),
            
            # === 不平衡指标 ===
            'soc_std': self._calculate_effective_soc_std(cluster_record),
            'temp_std': self._calculate_effective_temp_std(cluster_record),
            'soh_std': cluster_record.get('inter_bms_soh_std', 0.0),
            
            # === 功率和效率 ===
            'power_command': cluster_record.get('total_power_command', 0.0),
            'power_efficiency': cluster_record.get('system_power_efficiency', 1.0),
            'energy_efficiency': cluster_record.get('cluster_metrics', {}).get('energy_efficiency', 1.0),
            'power_tracking_error': cluster_record.get('power_tracking_error', 0.0),
            
            # === 成本状态 ===
            'total_degradation_cost': cluster_record.get('cost_breakdown', {}).get('total_system_cost', 0.0),
            'degradation_cost_rate': cluster_record.get('cost_breakdown', {}).get('system_cost_increase_rate', 0.0),
            
            # === 均衡状态 ===
            'balancing_active': self._check_any_balancing_active(cluster_record),
            'balancing_power': self._calculate_total_balancing_power(cluster_record),
            
            # === 约束状态 ===
            'thermal_constraints_active': cluster_record.get('system_constraints_active', {}).get('thermal_constraints', False),
            'degradation_constraints_active': cluster_record.get('system_constraints_active', {}).get('balance_constraints', False),
            'constraint_severity': self._calculate_constraint_severity(cluster_record),
            
            # === 健康和安全 ===
            'health_status': cluster_record.get('system_health_status', 'Good'),
            'warning_count': cluster_record.get('system_warning_count', 0),
            'alarm_count': cluster_record.get('system_alarm_count', 0),
            
            # === 扩展信息（保留BMS集群数据） ===
            'bms_cluster_data': cluster_record,
            'num_bms': cluster_record.get('num_bms', 10),
            'inter_bms_soc_std': cluster_record.get('inter_bms_soc_std', 0.0),
            'inter_bms_temp_std': cluster_record.get('inter_bms_temp_std', 0.0),
            'avg_intra_bms_soc_std': cluster_record.get('avg_intra_bms_soc_std', 0.0),
            'coordination_commands_count': len(cluster_record.get('coordination_commands', {}))
        }
        
        return pack_record
    
    def _calculate_pack_voltage(self, cluster_record: Dict) -> float:
        """计算等效电池组电压"""
        system_avg_soc = cluster_record.get('system_avg_soc', 50.0)
        # 使用SOC-OCV关系计算
        ocv = self.battery_params.get_ocv_from_soc(system_avg_soc)
        pack_voltage = ocv * self.battery_params.SERIES_NUM
        return pack_voltage
    
    def _calculate_pack_current(self, cluster_record: Dict) -> float:
        """计算等效电池组电流"""
        pack_power = cluster_record.get('total_actual_power', 0.0)
        pack_voltage = self._calculate_pack_voltage(cluster_record)
        
        if pack_voltage > 0:
            pack_current = pack_power / pack_voltage
        else:
            pack_current = 0.0
        
        return pack_current
    
    def _calculate_effective_soc_std(self, cluster_record: Dict) -> float:
        """计算有效SOC标准差（结合BMS间和BMS内）"""
        inter_bms_soc_std = cluster_record.get('inter_bms_soc_std', 0.0)
        avg_intra_bms_soc_std = cluster_record.get('avg_intra_bms_soc_std', 0.0)
        
        # 加权组合，BMS间不平衡影响更大
        effective_soc_std = 0.7 * inter_bms_soc_std + 0.3 * avg_intra_bms_soc_std
        return effective_soc_std
    
    def _calculate_effective_temp_std(self, cluster_record: Dict) -> float:
        """计算有效温度标准差"""
        inter_bms_temp_std = cluster_record.get('inter_bms_temp_std', 0.0)
        avg_intra_bms_temp_std = cluster_record.get('avg_intra_bms_temp_std', 0.0)
        
        # 加权组合
        effective_temp_std = 0.6 * inter_bms_temp_std + 0.4 * avg_intra_bms_temp_std
        return effective_temp_std
    
    def _check_any_balancing_active(self, cluster_record: Dict) -> bool:
        """检查是否有任何均衡活动"""
        bms_records = cluster_record.get('bms_records', [])
        
        for bms_record in bms_records:
            if bms_record.get('balancing_active', False):
                return True
        
        return False
    
    def _calculate_total_balancing_power(self, cluster_record: Dict) -> float:
        """计算总均衡功率"""
        bms_records = cluster_record.get('bms_records', [])
        total_balancing_power = 0.0
        
        for bms_record in bms_records:
            total_balancing_power += bms_record.get('balancing_power', 0.0)
        
        return total_balancing_power
    
    def _calculate_constraint_severity(self, cluster_record: Dict) -> float:
        """计算约束严重程度"""
        constraints_active = cluster_record.get('system_constraints_active', {})
        active_count = sum(1 for active in constraints_active.values() if active)
        
        # 简化计算：基于激活的约束数量
        max_constraints = 4  # 假设最多4种约束
        severity = active_count / max_constraints
        
        return min(1.0, severity)
    
    def get_drl_state_vector(self, normalize: bool = True) -> np.ndarray:
        """
        获取DRL状态向量 - 兼容接口
        
        Args:
            normalize: 是否归一化
            
        Returns:
            状态向量
        """
        
        if not self.pack_history:
            # 如果没有历史，返回默认状态
            state_dim = 14  # 原有状态维度
            return np.full(state_dim, 0.5, dtype=np.float32)
        
        latest_record = self.pack_history[-1]
        
        # 构建状态向量（保持原有格式）
        state_vector = np.array([
            latest_record['pack_soc'] / 100.0 if normalize else latest_record['pack_soc'],
            (latest_record['pack_temperature'] - 15.0) / 30.0 if normalize else latest_record['pack_temperature'],
            latest_record['soc_std'] / 10.0 if normalize else latest_record['soc_std'],
            latest_record['temp_std'] / 15.0 if normalize else latest_record['temp_std'],
            latest_record['pack_soh'] / 100.0 if normalize else latest_record['pack_soh'],
            abs(latest_record['pack_power']) / self.battery_params.max_discharge_power if normalize else latest_record['pack_power'],
            latest_record['power_efficiency'] if normalize else latest_record['power_efficiency'],
            1.0 if latest_record['thermal_constraints_active'] else 0.0,
            1.0 if latest_record['degradation_constraints_active'] else 0.0,
            latest_record['constraint_severity'] if normalize else latest_record['constraint_severity'],
            latest_record['power_tracking_error'] / 1000.0 if normalize else latest_record['power_tracking_error'],
            1.0 if latest_record['balancing_active'] else 0.0,
            latest_record['balancing_power'] / 1000.0 if normalize else latest_record['balancing_power'],
            latest_record['degradation_cost_rate'] if normalize else latest_record['degradation_cost_rate']
        ], dtype=np.float32)
        
        if normalize:
            state_vector = np.clip(state_vector, 0.0, 1.0)
        
        return state_vector
    
    def get_constraint_matrix_for_drl(self) -> np.ndarray:
        """获取DRL约束矩阵 - 兼容接口"""
        # 委托给BMS集群管理器
        return self.bms_cluster.inter_bms_coordinator.generate_coordination_commands()
    
    def reset(self, 
              random_initialization: bool = False,
              target_soc: float = 50.0,
              target_temp: float = 25.0,
              reset_degradation: bool = False) -> Dict:
        """
        重置电池组管理器 - 兼容接口
        
        Args:
            random_initialization: 是否随机初始化
            target_soc: 目标SOC
            target_temp: 目标温度
            reset_degradation: 是否重置劣化
            
        Returns:
            重置结果
        """
        
        # 重置BMS集群
        cluster_reset_result = self.bms_cluster.reset(
            target_soc=target_soc,
            target_temp=target_temp,
            add_inter_bms_variation=random_initialization,
            add_intra_bms_variation=random_initialization
        )
        
        # 清空历史
        self.pack_history.clear()
        
        # 兼容格式的重置结果
        reset_result = {
            'manager_id': self.manager_id,
            'reset_complete': True,
            'target_soc': target_soc,
            'target_temp': target_temp,
            'random_initialization': random_initialization,
            'bms_cluster_reset': cluster_reset_result
        }
        
        print(f"🔄 电池组管理器 {self.manager_id} 已重置 (BMS集群模式)")
        
        return reset_result
    
    def get_pack_summary(self) -> Dict:
        """获取电池组摘要 - 兼容接口"""
        
        cluster_summary = self.bms_cluster.get_cluster_summary()
        
        # 转换为兼容格式
        pack_summary = {
            'manager_id': self.manager_id,
            'management_mode': self.management_mode,
            'total_cells': cluster_summary['total_cells'],
            'pack_soc': cluster_summary['system_avg_soc'],
            'pack_temperature': cluster_summary['system_avg_temp'],
            'pack_soh': cluster_summary['system_avg_soh'],
            'soc_std': self._calculate_effective_soc_std(cluster_summary),
            'temp_std': self._calculate_effective_temp_std(cluster_summary),
            'total_cost': cluster_summary['total_system_cost'],
            'num_bms': cluster_summary['num_bms'],
            'inter_bms_balance': {
                'soc_std': cluster_summary['inter_bms_soc_std'],
                'temp_std': cluster_summary['inter_bms_temp_std']
            },
            'intra_bms_balance': {
                'avg_soc_std': cluster_summary['avg_intra_bms_soc_std'],
                'avg_temp_std': cluster_summary['avg_intra_bms_temp_std']
            },
            'bms_details': cluster_summary['bms_summaries']
        }
        
        return pack_summary
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"PackManager({self.manager_id}): "
                f"模式={self.management_mode}, "
                f"BMS数={getattr(self.bms_cluster, 'num_bms', 'N/A')}")
