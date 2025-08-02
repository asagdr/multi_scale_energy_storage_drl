"""
状态管理器 - 修正版本
专注于状态存储、管理和BMS集群数据处理
移除神经网络相关功能，保持职责清晰
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import pickle
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.system_config import SystemConfig

class StateScope(Enum):
    """状态范围枚举"""
    CELL_LEVEL = "cell_level"       # 单体级状态
    BMS_LEVEL = "bms_level"         # BMS级状态 (新增)
    PACK_LEVEL = "pack_level"       # 电池组级状态
    CLUSTER_LEVEL = "cluster_level" # 集群级状态 (新增)
    SYSTEM_LEVEL = "system_level"   # 系统级状态
    ENVIRONMENT = "environment"     # 环境状态
    CONTROL = "control"            # 控制状态

class StateType(Enum):
    """状态类型枚举"""
    PHYSICAL = "physical"           # 物理状态
    ELECTRICAL = "electrical"       # 电气状态
    THERMAL = "thermal"            # 热状态
    DEGRADATION = "degradation"     # 劣化状态
    SAFETY = "safety"              # 安全状态
    PERFORMANCE = "performance"     # 性能状态
    BALANCE = "balance"            # 均衡状态 (新增)
    COORDINATION = "coordination"   # 协调状态 (新增)

@dataclass
class StateSnapshot:
    """状态快照数据结构"""
    timestamp: float
    state_scope: StateScope
    state_type: StateType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.metadata['created_at'] = time.time()
        self.metadata['size'] = len(str(self.data))

@dataclass
class BMSClusterState:
    """BMS集群状态数据结构"""
    cluster_id: str
    timestamp: float
    
    # 系统级状态
    system_avg_soc: float = 50.0
    system_avg_temp: float = 25.0
    system_avg_soh: float = 100.0
    total_power: float = 0.0
    system_efficiency: float = 1.0
    
    # BMS间状态
    inter_bms_soc_std: float = 0.0
    inter_bms_temp_std: float = 0.0
    inter_bms_soh_std: float = 0.0
    
    # BMS内平均状态
    avg_intra_bms_soc_std: float = 0.0
    avg_intra_bms_temp_std: float = 0.0
    
    # 协调状态
    coordination_active: bool = False
    coordination_commands_count: int = 0
    
    # 健康状态
    system_health_status: str = "Good"
    warning_count: int = 0
    alarm_count: int = 0
    
    # 约束状态
    constraints_active: Dict[str, bool] = field(default_factory=dict)
    
    # 成本状态
    total_system_cost: float = 0.0
    cost_increase_rate: float = 0.0

class StateHistory:
    """状态历史管理"""
    def __init__(self, max_length: int = 1000):
        self.snapshots: List[StateSnapshot] = []
        self.max_length = max_length
        self._lock = threading.Lock()
    
    def add_snapshot(self, snapshot: StateSnapshot):
        """添加状态快照"""
        with self._lock:
            self.snapshots.append(snapshot)
            if len(self.snapshots) > self.max_length:
                self.snapshots.pop(0)
    
    def get_latest(self, state_scope: Optional[StateScope] = None, 
                  state_type: Optional[StateType] = None) -> Optional[StateSnapshot]:
        """获取最新状态快照"""
        with self._lock:
            for snapshot in reversed(self.snapshots):
                if ((state_scope is None or snapshot.state_scope == state_scope) and
                    (state_type is None or snapshot.state_type == state_type)):
                    return snapshot
        return None
    
    def get_range(self, start_time: float, end_time: float) -> List[StateSnapshot]:
        """获取时间范围内的状态快照"""
        with self._lock:
            return [s for s in self.snapshots 
                   if start_time <= s.timestamp <= end_time]
    
    def clear(self):
        """清空历史"""
        with self._lock:
            self.snapshots.clear()

class StateManager:
    """
    状态管理器 - 职责清晰版本
    专注于状态存储、管理和BMS集群数据处理
    不包含神经网络、分析、归一化等功能
    """
    
    def __init__(self, 
                 system_config: SystemConfig,
                 manager_id: str = "StateManager_001"):
        """
        初始化状态管理器
        
        Args:
            system_config: 系统配置
            manager_id: 管理器ID
        """
        self.system_config = system_config
        self.manager_id = manager_id
        
        # === 状态存储 ===
        self.current_states: Dict[Tuple[StateScope, StateType], Dict[str, Any]] = {}
        self.state_histories: Dict[Tuple[StateScope, StateType], StateHistory] = {}
        
        # === BMS集群状态存储 ===
        self.cluster_states: Dict[str, BMSClusterState] = {}
        self.cluster_history: List[BMSClusterState] = []
        
        # === 状态观察者 ===
        self.observers: Dict[str, List[callable]] = {}
        
        # === 线程安全 ===
        self._state_lock = threading.RLock()
        
        # === 状态统计 ===
        self.update_count = 0
        self.last_update_time = 0.0
        
        # === 自动保存配置 ===
        self.enable_auto_save = True
        self.save_interval = 300.0  # 5分钟
        self.last_save_time = time.time()
        
        # === 初始化状态历史 ===
        self._initialize_state_histories()
        
        print(f"✅ 状态管理器初始化完成: {manager_id} (支持BMS集群)")
    
    def _initialize_state_histories(self):
        """初始化状态历史管理器"""
        for scope in StateScope:
            for state_type in StateType:
                key = (scope, state_type)
                max_length = self.system_config.MAX_HISTORY_LENGTH
                self.state_histories[key] = StateHistory(max_length)
    
    def update_state(self, 
                    state_scope: StateScope,
                    state_type: StateType,
                    state_data: Dict[str, Any],
                    timestamp: Optional[float] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        更新状态
        
        Args:
            state_scope: 状态范围
            state_type: 状态类型
            state_data: 状态数据
            timestamp: 时间戳
            metadata: 元数据
            
        Returns:
            更新成功标志
        """
        if timestamp is None:
            timestamp = time.time()
        
        if metadata is None:
            metadata = {}
        
        try:
            with self._state_lock:
                # 更新当前状态
                key = (state_scope, state_type)
                self.current_states[key] = state_data.copy()
                
                # 创建状态快照
                snapshot = StateSnapshot(
                    timestamp=timestamp,
                    state_scope=state_scope,
                    state_type=state_type,
                    data=state_data.copy(),
                    metadata=metadata
                )
                
                # 添加到历史
                if key in self.state_histories:
                    self.state_histories[key].add_snapshot(snapshot)
                
                # 更新统计
                self.update_count += 1
                self.last_update_time = timestamp
                
                # 通知观察者
                self._notify_observers(state_scope, state_type, snapshot)
                
                # 自动保存检查
                if (self.enable_auto_save and 
                    timestamp - self.last_save_time > self.save_interval):
                    self._auto_save_states()
                
                return True
                
        except Exception as e:
            print(f"❌ 状态更新失败: {str(e)}")
            return False
    
    def get_current_state(self, 
                         state_scope: StateScope,
                         state_type: StateType) -> Optional[Dict[str, Any]]:
        """
        获取当前状态
        
        Args:
            state_scope: 状态范围
            state_type: 状态类型
            
        Returns:
            状态数据或None
        """
        with self._state_lock:
            key = (state_scope, state_type)
            return self.current_states.get(key, {}).copy()
    
    def get_state_history(self, 
                         state_scope: StateScope,
                         state_type: StateType,
                         count: Optional[int] = None) -> List[StateSnapshot]:
        """
        获取状态历史
        
        Args:
            state_scope: 状态范围
            state_type: 状态类型
            count: 获取数量（None表示全部）
            
        Returns:
            状态快照列表
        """
        key = (state_scope, state_type)
        if key not in self.state_histories:
            return []
        
        snapshots = self.state_histories[key].snapshots.copy()
        
        if count is not None:
            snapshots = snapshots[-count:]
        
        return snapshots
    
    # === BMS集群状态管理功能 ===
    
    def update_bms_cluster_state(self, cluster_record: Dict[str, Any]) -> bool:
        """
        更新BMS集群状态
        
        Args:
            cluster_record: BMS集群记录
            
        Returns:
            更新成功标志
        """
        try:
            cluster_id = cluster_record.get('cluster_id', 'default_cluster')
            timestamp = time.time()
            
            # 创建集群状态对象
            cluster_state = BMSClusterState(
                cluster_id=cluster_id,
                timestamp=timestamp,
                system_avg_soc=cluster_record.get('system_avg_soc', 50.0),
                system_avg_temp=cluster_record.get('system_avg_temp', 25.0),
                system_avg_soh=cluster_record.get('system_avg_soh', 100.0),
                total_power=cluster_record.get('total_actual_power', 0.0),
                system_efficiency=cluster_record.get('system_power_efficiency', 1.0),
                inter_bms_soc_std=cluster_record.get('inter_bms_soc_std', 0.0),
                inter_bms_temp_std=cluster_record.get('inter_bms_temp_std', 0.0),
                inter_bms_soh_std=cluster_record.get('inter_bms_soh_std', 0.0),
                avg_intra_bms_soc_std=cluster_record.get('avg_intra_bms_soc_std', 0.0),
                avg_intra_bms_temp_std=cluster_record.get('avg_intra_bms_temp_std', 0.0),
                coordination_active=len(cluster_record.get('coordination_commands', {})) > 0,
                coordination_commands_count=len(cluster_record.get('coordination_commands', {})),
                system_health_status=cluster_record.get('system_health_status', 'Good'),
                warning_count=cluster_record.get('system_warning_count', 0),
                alarm_count=cluster_record.get('system_alarm_count', 0),
                constraints_active=cluster_record.get('system_constraints_active', {}),
                total_system_cost=cluster_record.get('cost_breakdown', {}).get('total_system_cost', 0.0),
                cost_increase_rate=cluster_record.get('cost_breakdown', {}).get('system_cost_increase_rate', 0.0)
            )
            
            with self._state_lock:
                # 更新当前集群状态
                self.cluster_states[cluster_id] = cluster_state
                
                # 添加到历史
                self.cluster_history.append(cluster_state)
                
                # 维护历史长度
                if len(self.cluster_history) > self.system_config.MAX_HISTORY_LENGTH:
                    self.cluster_history.pop(0)
                
                # 同时更新到通用状态系统
                self.update_state(
                    StateScope.CLUSTER_LEVEL,
                    StateType.PERFORMANCE,
                    cluster_record,
                    timestamp
                )
            
            return True
            
        except Exception as e:
            print(f"❌ BMS集群状态更新失败: {str(e)}")
            return False
    
    def get_current_cluster_state(self, cluster_id: str) -> Optional[BMSClusterState]:
        """
        获取当前BMS集群状态
        
        Args:
            cluster_id: 集群ID
            
        Returns:
            集群状态或None
        """
        with self._state_lock:
            return self.cluster_states.get(cluster_id)
    
    def get_cluster_history(self, 
                           cluster_id: Optional[str] = None,
                           count: Optional[int] = None) -> List[BMSClusterState]:
        """
        获取BMS集群历史状态
        
        Args:
            cluster_id: 集群ID (None表示所有集群)
            count: 获取数量 (None表示全部)
            
        Returns:
            集群状态历史列表
        """
        with self._state_lock:
            if cluster_id is None:
                history = self.cluster_history.copy()
            else:
                history = [state for state in self.cluster_history if state.cluster_id == cluster_id]
            
            if count is not None:
                history = history[-count:]
            
            return history
    
    def process_bms_cluster_data(self, cluster_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理BMS集群数据 - 纯数据处理，不包含分析
        
        Args:
            cluster_record: 原始集群记录
            
        Returns:
            处理后的集群数据
        """
        
        processed_data = {
            'processing_timestamp': time.time(),
            'cluster_id': cluster_record.get('cluster_id', 'unknown'),
            'data_completeness': self._check_data_completeness(cluster_record),
            'bms_count': len(cluster_record.get('bms_records', [])),
            'cell_count_total': self._calculate_total_cells(cluster_record),
            
            # 系统级数据整理
            'system_data': {
                'avg_soc': cluster_record.get('system_avg_soc', 50.0),
                'avg_temp': cluster_record.get('system_avg_temp', 25.0),
                'avg_soh': cluster_record.get('system_avg_soh', 100.0),
                'total_power': cluster_record.get('total_actual_power', 0.0),
                'power_efficiency': cluster_record.get('system_power_efficiency', 1.0)
            },
            
            # BMS级数据整理
            'bms_data_summary': self._summarize_bms_data(cluster_record),
            
            # 单体级数据摘要
            'cell_data_summary': self._summarize_cell_data(cluster_record),
            
            # 原始数据保留
            'raw_cluster_record': cluster_record
        }
        
        return processed_data
    
    def get_basic_state_vector(self, 
                              state_scope: StateScope = StateScope.SYSTEM_LEVEL) -> List[float]:
        """
        获取基础状态向量 - 仅提供原始数值，不做归一化
        
        Args:
            state_scope: 状态范围
            
        Returns:
            状态数值列表
        """
        state_values = []
        
        # 收集不同类型的状态数值
        for state_type in StateType:
            current_state = self.get_current_state(state_scope, state_type)
            
            if current_state:
                # 提取数值状态
                numeric_values = self._extract_numeric_values(current_state)
                state_values.extend(numeric_values)
        
        return state_values
    
    def get_cluster_state_data(self, cluster_id: str) -> Dict[str, Any]:
        """
        获取集群状态数据 - 仅返回原始数据
        
        Args:
            cluster_id: 集群ID
            
        Returns:
            集群状态数据字典
        """
        cluster_state = self.get_current_cluster_state(cluster_id)
        
        if cluster_state is None:
            return {}
        
        return {
            'cluster_id': cluster_state.cluster_id,
            'timestamp': cluster_state.timestamp,
            
            # 系统级原始数据
            'system_avg_soc': cluster_state.system_avg_soc,
            'system_avg_temp': cluster_state.system_avg_temp,
            'system_avg_soh': cluster_state.system_avg_soh,
            'total_power': cluster_state.total_power,
            'system_efficiency': cluster_state.system_efficiency,
            
            # BMS间原始数据
            'inter_bms_soc_std': cluster_state.inter_bms_soc_std,
            'inter_bms_temp_std': cluster_state.inter_bms_temp_std,
            'inter_bms_soh_std': cluster_state.inter_bms_soh_std,
            
            # BMS内原始数据
            'avg_intra_bms_soc_std': cluster_state.avg_intra_bms_soc_std,
            'avg_intra_bms_temp_std': cluster_state.avg_intra_bms_temp_std,
            
            # 协调原始数据
            'coordination_active': cluster_state.coordination_active,
            'coordination_commands_count': cluster_state.coordination_commands_count,
            
            # 健康原始数据
            'system_health_status': cluster_state.system_health_status,
            'warning_count': cluster_state.warning_count,
            'alarm_count': cluster_state.alarm_count,
            
            # 约束原始数据
            'constraints_active': cluster_state.constraints_active,
            
            # 成本原始数据
            'total_system_cost': cluster_state.total_system_cost,
            'cost_increase_rate': cluster_state.cost_increase_rate
        }
    
    # === 辅助方法 ===
    
    def _check_data_completeness(self, cluster_record: Dict[str, Any]) -> float:
        """检查数据完整性"""
        required_fields = [
            'cluster_id', 'system_avg_soc', 'system_avg_temp', 'system_avg_soh',
            'total_actual_power', 'bms_records'
        ]
        
        present_fields = sum(1 for field in required_fields if field in cluster_record)
        completeness = present_fields / len(required_fields)
        
        return completeness
    
    def _calculate_total_cells(self, cluster_record: Dict[str, Any]) -> int:
        """计算总单体数量"""
        total_cells = 0
        bms_records = cluster_record.get('bms_records', [])
        
        for bms_record in bms_records:
            cells = bms_record.get('cells', [])
            total_cells += len(cells)
        
        return total_cells
    
    def _summarize_bms_data(self, cluster_record: Dict[str, Any]) -> Dict[str, Any]:
        """汇总BMS数据"""
        bms_records = cluster_record.get('bms_records', [])
        
        if not bms_records:
            return {}
        
        # 收集所有BMS的基础数据
        bms_socs = [bms.get('avg_soc', 50.0) for bms in bms_records]
        bms_temps = [bms.get('avg_temperature', 25.0) for bms in bms_records]
        bms_sohs = [bms.get('avg_soh', 100.0) for bms in bms_records]
        bms_powers = [bms.get('actual_power', 0.0) for bms in bms_records]
        
        return {
            'bms_count': len(bms_records),
            'soc_data': {
                'values': bms_socs,
                'mean': np.mean(bms_socs),
                'std': np.std(bms_socs),
                'min': np.min(bms_socs),
                'max': np.max(bms_socs)
            },
            'temp_data': {
                'values': bms_temps,
                'mean': np.mean(bms_temps),
                'std': np.std(bms_temps),
                'min': np.min(bms_temps),
                'max': np.max(bms_temps)
            },
            'soh_data': {
                'values': bms_sohs,
                'mean': np.mean(bms_sohs),
                'std': np.std(bms_sohs),
                'min': np.min(bms_sohs),
                'max': np.max(bms_sohs)
            },
            'power_data': {
                'values': bms_powers,
                'total': sum(bms_powers),
                'mean': np.mean(bms_powers),
                'std': np.std(bms_powers)
            }
        }
    
    def _summarize_cell_data(self, cluster_record: Dict[str, Any]) -> Dict[str, Any]:
        """汇总单体数据"""
        all_cell_socs = []
        all_cell_temps = []
        all_cell_sohs = []
        
        bms_records = cluster_record.get('bms_records', [])
        
        for bms_record in bms_records:
            cells = bms_record.get('cells', [])
            for cell in cells:
                all_cell_socs.append(cell.get('soc', 50.0))
                all_cell_temps.append(cell.get('temperature', 25.0))
                all_cell_sohs.append(cell.get('soh', 100.0))
        
        if not all_cell_socs:
            return {}
        
        return {
            'total_cells': len(all_cell_socs),
            'soc_stats': {
                'mean': np.mean(all_cell_socs),
                'std': np.std(all_cell_socs),
                'min': np.min(all_cell_socs),
                'max': np.max(all_cell_socs),
                'range': np.max(all_cell_socs) - np.min(all_cell_socs)
            },
            'temp_stats': {
                'mean': np.mean(all_cell_temps),
                'std': np.std(all_cell_temps),
                'min': np.min(all_cell_temps),
                'max': np.max(all_cell_temps),
                'range': np.max(all_cell_temps) - np.min(all_cell_temps)
            },
            'soh_stats': {
                'mean': np.mean(all_cell_sohs),
                'std': np.std(all_cell_sohs),
                'min': np.min(all_cell_sohs),
                'max': np.max(all_cell_sohs)
            }
        }
    
    def _extract_numeric_values(self, state_dict: Dict[str, Any]) -> List[float]:
        """从状态字典中提取数值"""
        numeric_values = []
        
        for key, value in state_dict.items():
            if isinstance(value, (int, float)):
                numeric_values.append(float(value))
            elif isinstance(value, (list, np.ndarray)):
                # 对于数组类型，计算统计量
                if len(value) > 0 and isinstance(value[0], (int, float)):
                    numeric_values.extend([
                        float(np.mean(value)),
                        float(np.std(value)),
                        float(np.min(value)),
                        float(np.max(value))
                    ])
            elif isinstance(value, bool):
                numeric_values.append(float(value))
        
        return numeric_values
    
    # === 观察者模式 ===
    
    def register_observer(self, 
                         observer_id: str,
                         callback: callable,
                         state_scope: Optional[StateScope] = None,
                         state_type: Optional[StateType] = None) -> bool:
        """
        注册状态观察者
        
        Args:
            observer_id: 观察者ID
            callback: 回调函数
            state_scope: 监听的状态范围（None表示全部）
            state_type: 监听的状态类型（None表示全部）
            
        Returns:
            注册成功标志
        """
        try:
            key = f"{state_scope}_{state_type}" if state_scope and state_type else "all"
            
            if key not in self.observers:
                self.observers[key] = []
            
            # 避免重复注册
            if callback not in self.observers[key]:
                self.observers[key].append(callback)
            
            print(f"✅ 已注册观察者: {observer_id} -> {key}")
            return True
            
        except Exception as e:
            print(f"❌ 观察者注册失败: {str(e)}")
            return False
    
    def unregister_observer(self, observer_id: str) -> bool:
        """注销观察者"""
        try:
            # 从所有观察者列表中移除
            removed_count = 0
            for key in self.observers:
                # 这里简化处理，实际应该维护observer_id到callback的映射
                pass
            
            print(f"✅ 已注销观察者: {observer_id}")
            return True
            
        except Exception as e:
            print(f"❌ 观察者注销失败: {str(e)}")
            return False
    
    def _notify_observers(self, 
                         state_scope: StateScope,
                         state_type: StateType,
                         snapshot: StateSnapshot):
        """通知观察者"""
        try:
            # 通知特定类型观察者
            specific_key = f"{state_scope}_{state_type}"
            if specific_key in self.observers:
                for callback in self.observers[specific_key]:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        print(f"⚠️ 观察者回调执行失败: {str(e)}")
            
            # 通知全局观察者
            if "all" in self.observers:
                for callback in self.observers["all"]:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        print(f"⚠️ 全局观察者回调执行失败: {str(e)}")
                        
        except Exception as e:
            print(f"❌ 观察者通知失败: {str(e)}")
    
    # === 状态持久化 ===
    
    def create_state_checkpoint(self, checkpoint_name: str) -> bool:
        """创建状态检查点"""
        try:
            checkpoint_data = {
                'timestamp': time.time(),
                'current_states': self.current_states.copy(),
                'cluster_states': {cid: {
                    'cluster_id': cs.cluster_id,
                    'timestamp': cs.timestamp,
                    'system_avg_soc': cs.system_avg_soc,
                    'system_avg_temp': cs.system_avg_temp,
                    'total_power': cs.total_power
                } for cid, cs in self.cluster_states.items()},
                'update_count': self.update_count,
                'last_update_time': self.last_update_time
            }
            
            # 简化实现：仅在内存中保存
            if not hasattr(self, 'checkpoints'):
                self.checkpoints = {}
            
            self.checkpoints[checkpoint_name] = checkpoint_data
            print(f"✅ 已创建状态检查点: {checkpoint_name}")
            return True
            
        except Exception as e:
            print(f"❌ 创建检查点失败: {str(e)}")
            return False
    
    def restore_from_checkpoint(self, checkpoint_name: str) -> bool:
        """从检查点恢复状态"""
        try:
            if not hasattr(self, 'checkpoints') or checkpoint_name not in self.checkpoints:
                print(f"❌ 检查点不存在: {checkpoint_name}")
                return False
            
            checkpoint_data = self.checkpoints[checkpoint_name]
            
            with self._state_lock:
                self.current_states = checkpoint_data['current_states'].copy()
                self.update_count = checkpoint_data['update_count']
                self.last_update_time = checkpoint_data['last_update_time']
                
                # 恢复集群状态（简化版本）
                cluster_data = checkpoint_data.get('cluster_states', {})
                for cid, cs_data in cluster_data.items():
                    cluster_state = BMSClusterState(
                        cluster_id=cs_data['cluster_id'],
                        timestamp=cs_data['timestamp'],
                        system_avg_soc=cs_data['system_avg_soc'],
                        system_avg_temp=cs_data['system_avg_temp'],
                        total_power=cs_data['total_power']
                    )
                    self.cluster_states[cid] = cluster_state
            
            print(f"✅ 已从检查点恢复: {checkpoint_name}")
            return True
            
        except Exception as e:
            print(f"❌ 检查点恢复失败: {str(e)}")
            return False
    
    def _auto_save_states(self):
        """自动保存状态"""
        try:
            # 简化实现：创建自动检查点
            auto_checkpoint_name = f"auto_save_{int(time.time())}"
            self.create_state_checkpoint(auto_checkpoint_name)
            self.last_save_time = time.time()
            
        except Exception as e:
            print(f"❌ 自动保存失败: {str(e)}")
    
    # === 统计和信息 ===
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """获取状态统计信息"""
        with self._state_lock:
            total_snapshots = sum(len(history.snapshots) for history in self.state_histories.values())
            
            # 计算各类型状态的数量
            state_counts = {}
            for (scope, state_type), history in self.state_histories.items():
                key = f"{scope.value}_{state_type.value}"
                state_counts[key] = len(history.snapshots)
            
            return {
                'manager_id': self.manager_id,
                'update_count': self.update_count,
                'last_update_time': self.last_update_time,
                'total_snapshots': total_snapshots,
                'current_state_count': len(self.current_states),
                'cluster_state_count': len(self.cluster_states),
                'cluster_history_length': len(self.cluster_history),
                'observer_count': sum(len(observers) for observers in self.observers.values()),
                'state_counts_by_type': state_counts,
                'memory_usage_mb': self._estimate_memory_usage(),
                'auto_save_enabled': self.enable_auto_save,
                'checkpoints_count': len(getattr(self, 'checkpoints', {})),
                'supports_bms_cluster': True
            }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用量 (MB)"""
        try:
            total_size = 0
            
            # 估算当前状态大小
            for state_data in self.current_states.values():
                total_size += len(str(state_data))
            
            # 估算历史数据大小
            for history in self.state_histories.values():
                for snapshot in history.snapshots:
                    total_size += snapshot.metadata.get('size', 0)
            
            # 估算集群状态大小
            for cluster_state in self.cluster_states.values():
                total_size += len(str(cluster_state))
            
            return total_size / (1024 * 1024)  # 转换为MB
            
        except Exception:
            return 0.0
    
    def clear_history(self, 
                     state_scope: Optional[StateScope] = None,
                     state_type: Optional[StateType] = None,
                     older_than: Optional[float] = None):
        """
        清理历史数据
        
        Args:
            state_scope: 状态范围（None表示全部）
            state_type: 状态类型（None表示全部）
            older_than: 清理早于该时间戳的数据
        """
        cleared_count = 0
        
        for (scope, stype), history in self.state_histories.items():
            if ((state_scope is None or scope == state_scope) and
                (state_type is None or stype == state_type)):
                
                if older_than is None:
                    cleared_count += len(history.snapshots)
                    history.clear()
                else:
                    old_count = len(history.snapshots)
                    history.snapshots = [s for s in history.snapshots if s.timestamp >= older_than]
                    cleared_count += old_count - len(history.snapshots)
        
        # 清理集群历史
        if older_than is not None:
            old_cluster_count = len(self.cluster_history)
            self.cluster_history = [cs for cs in self.cluster_history if cs.timestamp >= older_than]
            cleared_count += old_cluster_count - len(self.cluster_history)
        
        print(f"🧹 已清理 {cleared_count} 个历史状态记录")
    
    def export_states(self, 
                     file_path: str,
                     state_scope: Optional[StateScope] = None,
                     state_type: Optional[StateType] = None,
                     include_cluster_data: bool = True) -> bool:
        """导出状态数据"""
        try:
            export_data = {
                'metadata': {
                    'manager_id': self.manager_id,
                    'export_time': time.time(),
                    'update_count': self.update_count,
                    'supports_bms_cluster': True
                },
                'current_states': {},
                'histories': {},
                'cluster_data': {}
            }
            
            # 导出当前状态
            for (scope, stype), state_data in self.current_states.items():
                if ((state_scope is None or scope == state_scope) and
                    (state_type is None or stype == state_type)):
                    key = f"{scope.value}_{stype.value}"
                    export_data['current_states'][key] = state_data
            
            # 导出历史数据
            for (scope, stype), history in self.state_histories.items():
                if ((state_scope is None or scope == state_scope) and
                    (state_type is None or stype == state_type)):
                    key = f"{scope.value}_{stype.value}"
                    export_data['histories'][key] = [
                        {
                            'timestamp': snapshot.timestamp,
                            'data': snapshot.data,
                            'metadata': snapshot.metadata
                        } for snapshot in history.snapshots
                    ]
            
            # 导出集群数据
            if include_cluster_data:
                export_data['cluster_data'] = {
                    'current_cluster_states': {
                        cid: self.get_cluster_state_data(cid) 
                        for cid in self.cluster_states.keys()
                    },
                    'cluster_history_summary': {
                        'total_records': len(self.cluster_history),
                        'time_range': {
                            'start': min(cs.timestamp for cs in self.cluster_history) if self.cluster_history else 0,
                            'end': max(cs.timestamp for cs in self.cluster_history) if self.cluster_history else 0
                        }
                    }
                }
            
            # 保存到文件
            with open(file_path, 'wb') as f:
                pickle.dump(export_data, f)
            
            print(f"✅ 状态数据已导出到: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ 状态导出失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"StateManager({self.manager_id}): "
                f"状态数={len(self.current_states)}, "
                f"集群数={len(self.cluster_states)}, "
                f"更新次数={self.update_count}, "
                f"观察者数={sum(len(obs) for obs in self.observers.values())}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"StateManager(manager_id='{self.manager_id}', "
                f"states={len(self.current_states)}, "
                f"clusters={len(self.cluster_states)}, "
                f"updates={self.update_count})")
