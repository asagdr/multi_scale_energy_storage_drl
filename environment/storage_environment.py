"""
储能系统环境类 - 职责清晰版
专注于环境仿真，不包含状态归一化功能
"""

import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Optional, Tuple, Union
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig
from battery_models.bms_cluster_manager import BMSClusterManager

class StorageEnvironment(gym.Env):
    """
    储能系统环境类 - BMS集群版本，职责清晰
    专注于环境仿真，状态归一化交给专门的转换器处理
    """
    
    def __init__(self, 
                 battery_params: BatteryParams,
                 system_config: SystemConfig,
                 num_bms: int = 10,
                 env_id: str = "StorageEnv_001"):
        """
        初始化储能环境
        
        Args:
            battery_params: 电池参数
            system_config: 系统配置
            num_bms: BMS数量
            env_id: 环境ID
        """
        super().__init__()
        
        self.battery_params = battery_params
        self.system_config = system_config
        self.num_bms = num_bms
        self.env_id = env_id
        
        # === 初始化BMS集群管理器 ===
        self.bms_cluster = BMSClusterManager(
            battery_params=battery_params,
            system_config=system_config,
            num_bms=num_bms,
            cluster_id=f"Cluster_{env_id}"
        )
        
        # === 定义状态空间（原始数据空间，不预设归一化） ===
        # 状态向量包含24维原始数据
        self.state_names = [
            'system_avg_soc',           # 0: 系统平均SOC (%)
            'system_avg_temp',          # 1: 系统平均温度 (℃)
            'system_avg_soh',           # 2: 系统平均SOH (%)
            'inter_bms_soc_std',        # 3: BMS间SOC标准差 (%)
            'inter_bms_temp_std',       # 4: BMS间温度标准差 (℃)
            'inter_bms_soh_std',        # 5: BMS间SOH标准差 (%)
            'avg_intra_bms_soc_std',    # 6: 平均BMS内SOC标准差 (%)
            'avg_intra_bms_temp_std',   # 7: 平均BMS内温度标准差 (℃)
            'total_actual_power',       # 8: 总实际功率 (W)
            'system_power_efficiency',  # 9: 系统功率效率 (0-1)
            'power_tracking_error',     # 10: 功率跟踪误差 (W)
            'overall_balance_score',    # 11: 总体均衡评分 (0-1)
            'energy_efficiency',        # 12: 能量效率 (0-1)
            'safety_margin_soc',        # 13: SOC安全裕度 (0-1)
            'system_cost_increase_rate', # 14: 成本增长率 (0-1)
            'penalty_cost_ratio',       # 15: 惩罚成本比例 (0-1)
            'thermal_constraints_active', # 16: 热约束激活 (0/1)
            'soc_constraints_active',   # 17: SOC约束激活 (0/1)
            'balance_constraints_active', # 18: 均衡约束激活 (0/1)
            'coordination_ratio',       # 19: 协调指令比例 (0-1)
            'avg_coordination_weight',  # 20: 平均协调权重 (0-1)
            'ambient_temperature',      # 21: 环境温度 (℃)
            'external_power_demand',    # 22: 外部功率需求 (W)
            'system_health_score'       # 23: 系统健康评分 (0-1)
        ]
        
        # 定义原始数据的合理范围（用于检查，不用于归一化）
        self.state_ranges = {
            'system_avg_soc': (0.0, 100.0),
            'system_avg_temp': (-20.0, 60.0),
            'system_avg_soh': (50.0, 100.0),
            'inter_bms_soc_std': (0.0, 30.0),
            'inter_bms_temp_std': (0.0, 40.0),
            'inter_bms_soh_std': (0.0, 20.0),
            'avg_intra_bms_soc_std': (0.0, 15.0),
            'avg_intra_bms_temp_std': (0.0, 20.0),
            'total_actual_power': (-2000000.0, 2000000.0),  # ±2MW
            'system_power_efficiency': (0.0, 1.0),
            'power_tracking_error': (0.0, 200000.0),  # 200kW
            'overall_balance_score': (0.0, 1.0),
            'energy_efficiency': (0.0, 1.0),
            'safety_margin_soc': (0.0, 1.0),
            'system_cost_increase_rate': (0.0, 1.0),
            'penalty_cost_ratio': (0.0, 1.0),
            'thermal_constraints_active': (0, 1),
            'soc_constraints_active': (0, 1),
            'balance_constraints_active': (0, 1),
            'coordination_ratio': (0.0, 1.0),
            'avg_coordination_weight': (0.0, 1.0),
            'ambient_temperature': (-10.0, 50.0),
            'external_power_demand': (-2000000.0, 2000000.0),
            'system_health_score': (0.0, 1.0)
        }
        
        # 定义观察空间为原始数据空间（不归一化）
        state_dim = len(self.state_names)
        low = np.array([self.state_ranges[name][0] for name in self.state_names], dtype=np.float32)
        high = np.array([self.state_ranges[name][1] for name in self.state_names], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # === 定义动作空间 ===
        # 上层动作：[功率指令归一化值, SOC均衡权重, 温度均衡权重, 寿命优化权重, 效率权重]
        self.upper_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # 下层动作：[实际功率执行值, 响应速度参数, 误差补偿参数]
        self.lower_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # 当前使用上层动作空间
        self.action_space = self.upper_action_space
        
        # === 环境状态 ===
        self.current_step = 0
        self.max_steps = 1000
        self.done = False
        
        # === 外部输入 ===
        self.external_power_demand = 0.0    # W, 外部功率需求
        self.ambient_temperature = 25.0     # ℃, 环境温度
        
        # === 奖励权重 ===
        self.reward_weights = system_config.objective_weights
        
        # === 历史记录 ===
        self.episode_history: List[Dict] = []
        
        # === 上一步状态（用于计算奖励） ===
        self.last_total_cost = 0.0
        
        print(f"✅ BMS集群储能环境初始化完成: {env_id}")
        print(f"   状态维度: {state_dim}, 动作维度: {self.action_space.shape[0]}")
        print(f"   BMS数量: {num_bms}, 状态返回原始数据格式")
    
    def reset(self, **kwargs) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始状态向量（原始数据）
        """
        # 重置BMS集群
        reset_result = self.bms_cluster.reset(
            target_soc=50.0,
            target_temp=25.0,
            add_inter_bms_variation=True,
            add_intra_bms_variation=True
        )
        
        # 重置环境状态
        self.current_step = 0
        self.done = False
        self.external_power_demand = 0.0
        self.ambient_temperature = 25.0
        self.last_total_cost = 0.0
        
        # 清空历史
        self.episode_history.clear()
        
        # 获取初始状态
        initial_cluster_record = self._get_initial_cluster_state()
        state = self._extract_raw_state_vector(initial_cluster_record)
        
        print(f"🔄 BMS集群环境 {self.env_id} 已重置")
        
        return state.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一个环境步
        
        Args:
            action: 动作向量
            
        Returns:
            (next_state, reward, done, info) - 状态为原始数据
        """
        # === 1. 解析动作 ===
        action = np.clip(action, -1.0, 1.0)
        
        # 上层动作解析
        power_command_norm = action[0]      # 功率指令 [-1,1]
        soc_balance_weight = (action[1] + 1) / 2      # SOC均衡权重 [-1,1] -> [0,1]
        temp_balance_weight = (action[2] + 1) / 2     # 温度均衡权重 [-1,1] -> [0,1]
        lifetime_weight = (action[3] + 1) / 2         # 寿命优化权重 [-1,1] -> [0,1]
        efficiency_weight = (action[4] + 1) / 2       # 效率权重 [-1,1] -> [0,1]
        
        # 归一化权重
        total_weight = soc_balance_weight + temp_balance_weight + lifetime_weight + efficiency_weight
        if total_weight > 0:
            soc_balance_weight /= total_weight
            temp_balance_weight /= total_weight
            lifetime_weight /= total_weight
            efficiency_weight /= total_weight
        else:
            # 默认权重
            soc_balance_weight = 0.3
            temp_balance_weight = 0.2
            lifetime_weight = 0.3
            efficiency_weight = 0.2
        
        # 转换为实际功率指令
        max_system_power = self.battery_params.max_charge_power
        power_command = power_command_norm * max_system_power  # W
        
        # 构建上层权重字典
        upper_layer_weights = {
            'soc_balance': soc_balance_weight,
            'temp_balance': temp_balance_weight,
            'lifetime': lifetime_weight,
            'efficiency': efficiency_weight
        }
        
        # === 2. 设置外部条件 ===
        self._update_external_conditions()
        
        # === 3. 执行BMS集群仿真 ===
        cluster_record = self.bms_cluster.step(
            total_power_command=power_command,
            delta_t=self.system_config.SIMULATION_TIME_STEP,
            upper_layer_weights=upper_layer_weights,
            ambient_temperature=self.ambient_temperature
        )
        
        # === 4. 计算奖励 ===
        reward = self._calculate_multi_level_reward(cluster_record, action, upper_layer_weights)
        
        # === 5. 获取下一状态（原始数据） ===
        next_state = self._extract_raw_state_vector(cluster_record)
        
        # === 6. 检查终止条件 ===
        self.current_step += 1
        self.done = self._check_done(cluster_record)
        
        # === 7. 构建信息字典 ===
        info = self._build_info_dict(cluster_record, action, upper_layer_weights, power_command)
        
        # === 8. 记录历史 ===
        episode_record = {
            'step': self.current_step,
            'action': action.copy(),
            'upper_layer_weights': upper_layer_weights.copy(),
            'reward': reward,
            'cluster_record': cluster_record,
            'info': info.copy()
        }
        self.episode_history.append(episode_record)
        
        return next_state.astype(np.float32), reward, self.done, info
    
    def _get_initial_cluster_state(self) -> Dict:
        """获取初始集群状态"""
        # 执行一次零功率仿真获取初始状态
        initial_record = self.bms_cluster.step(
            total_power_command=0.0,
            delta_t=0.0,  # 零时间步长
            upper_layer_weights={'soc_balance': 0.3, 'temp_balance': 0.2, 'lifetime': 0.3, 'efficiency': 0.2},
            ambient_temperature=self.ambient_temperature
        )
        return initial_record
    
    def _extract_raw_state_vector(self, cluster_record: Dict) -> np.ndarray:
        """
        从集群记录中提取原始状态向量 - 不做归一化
        
        Args:
            cluster_record: 集群仿真记录
            
        Returns:
            原始状态向量 [24维]
        """
        
        state_vector = np.zeros(24, dtype=np.float32)
        
        try:
            # === 基础系统状态 (0-2) ===
            state_vector[0] = cluster_record.get('system_avg_soc', 50.0)  # SOC (%)
            state_vector[1] = cluster_record.get('system_avg_temp', 25.0)  # 温度 (℃)
            state_vector[2] = cluster_record.get('system_avg_soh', 100.0)  # SOH (%)
            
            # === BMS间均衡指标 (3-5) ===
            state_vector[3] = cluster_record.get('inter_bms_soc_std', 0.0)  # BMS间SOC标准差 (%)
            state_vector[4] = cluster_record.get('inter_bms_temp_std', 0.0)  # BMS间温度标准差 (℃)
            state_vector[5] = cluster_record.get('inter_bms_soh_std', 0.0)  # BMS间SOH标准差 (%)
            
            # === BMS内均衡指标 (6-7) ===
            state_vector[6] = cluster_record.get('avg_intra_bms_soc_std', 0.0)  # 平均BMS内SOC标准差 (%)
            state_vector[7] = cluster_record.get('avg_intra_bms_temp_std', 0.0)  # 平均BMS内温度标准差 (℃)
            
            # === 功率和效率状态 (8-10) ===
            state_vector[8] = cluster_record.get('total_actual_power', 0.0)  # 总实际功率 (W)
            state_vector[9] = cluster_record.get('system_power_efficiency', 1.0)  # 系统功率效率 (0-1)
            state_vector[10] = cluster_record.get('power_tracking_error', 0.0)  # 功率跟踪误差 (W)
            
            # === 系统级指标 (11-13) ===
            cluster_metrics = cluster_record.get('cluster_metrics', {})
            state_vector[11] = cluster_metrics.get('overall_balance_score', 0.5)  # 总体均衡评分 (0-1)
            state_vector[12] = cluster_metrics.get('energy_efficiency', 1.0)  # 能量效率 (0-1)
            state_vector[13] = cluster_metrics.get('safety_margin_soc', 0.5)  # SOC安全裕度 (0-1)
            
            # === 成本状态 (14-15) ===
            cost_breakdown = cluster_record.get('cost_breakdown', {})
            state_vector[14] = cost_breakdown.get('system_cost_increase_rate', 0.0)  # 成本增长率 (0-1)
            
            # 成本结构比例
            total_cost = cost_breakdown.get('total_system_cost', 1.0)
            if total_cost > 0:
                penalty_ratio = (cost_breakdown.get('total_bms_penalty', 0.0) + 
                               cost_breakdown.get('total_system_penalty', 0.0)) / total_cost
                state_vector[15] = penalty_ratio  # 惩罚成本比例 (0-1)
            else:
                state_vector[15] = 0.0
            
            # === 约束和安全状态 (16-18) ===
            constraints_active = cluster_record.get('system_constraints_active', {})
            state_vector[16] = 1.0 if constraints_active.get('thermal_constraints', False) else 0.0
            state_vector[17] = 1.0 if constraints_active.get('soc_constraints', False) else 0.0
            state_vector[18] = 1.0 if constraints_active.get('balance_constraints', False) else 0.0
            
            # === 协调状态 (19-20) ===
            coordination_commands = cluster_record.get('coordination_commands', {})
            state_vector[19] = len(coordination_commands) / self.num_bms if self.num_bms > 0 else 0.0  # 协调指令比例
            
            # 协调权重
            if coordination_commands:
                avg_coordination_weight = np.mean([cmd.get('coordination_weight', 0.0) 
                                                 for cmd in coordination_commands.values()])
                state_vector[20] = avg_coordination_weight
            else:
                state_vector[20] = 0.0
            
            # === 环境状态 (21-22) ===
            state_vector[21] = self.ambient_temperature  # 环境温度 (℃)
            state_vector[22] = self.external_power_demand  # 外部功率需求 (W)
            
            # === 系统健康 (23) ===
            health_status = cluster_record.get('system_health_status', 'Good')
            health_scores = {'Critical': 0.0, 'Poor': 0.3, 'Fair': 0.6, 'Good': 1.0}
            state_vector[23] = health_scores.get(health_status, 0.5)  # 系统健康评分 (0-1)
            
        except Exception as e:
            print(f"⚠️ 状态向量提取失败: {str(e)}")
            # 返回安全的默认状态
            default_values = [50.0, 25.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 
                            0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 0.0, 1.0]
            state_vector = np.array(default_values, dtype=np.float32)
        
        return state_vector
    
    def get_state_info(self) -> Dict[str, Dict]:
        """
        获取状态信息 - 供外部归一化工具使用
        
        Returns:
            状态名称、范围和描述信息
        """
        state_info = {}
        
        for i, name in enumerate(self.state_names):
            state_info[name] = {
                'index': i,
                'range': self.state_ranges[name],
                'description': self._get_state_description(name)
            }
        
        return state_info
    
    def _get_state_description(self, state_name: str) -> str:
        """获取状态描述"""
        descriptions = {
            'system_avg_soc': '系统平均SOC百分比',
            'system_avg_temp': '系统平均温度（摄氏度）',
            'system_avg_soh': '系统平均SOH百分比',
            'inter_bms_soc_std': 'BMS间SOC标准差',
            'inter_bms_temp_std': 'BMS间温度标准差',
            'inter_bms_soh_std': 'BMS间SOH标准差',
            'avg_intra_bms_soc_std': '平均BMS内SOC标准差',
            'avg_intra_bms_temp_std': '平均BMS内温度标准差',
            'total_actual_power': '总实际功率（瓦特）',
            'system_power_efficiency': '系统功率效率',
            'power_tracking_error': '功率跟踪误差（瓦特）',
            'overall_balance_score': '总体均衡评分',
            'energy_efficiency': '能量效率',
            'safety_margin_soc': 'SOC安全裕度',
            'system_cost_increase_rate': '系统成本增长率',
            'penalty_cost_ratio': '惩罚成本比例',
            'thermal_constraints_active': '热约束激活状态',
            'soc_constraints_active': 'SOC约束激活状态',
            'balance_constraints_active': '均衡约束激活状态',
            'coordination_ratio': '协调指令比例',
            'avg_coordination_weight': '平均协调权重',
            'ambient_temperature': '环境温度（摄氏度）',
            'external_power_demand': '外部功率需求（瓦特）',
            'system_health_score': '系统健康评分'
        }
        
        return descriptions.get(state_name, f"状态变量: {state_name}")
    
    def _update_external_conditions(self):
        """更新外部条件"""
        # 模拟变化的功率需求（正弦波 + 噪声）
        time_hours = self.current_step * self.system_config.SIMULATION_TIME_STEP / 3600.0
        base_demand = np.sin(2 * np.pi * time_hours / 24.0) * 0.5  # 日周期
        noise_demand = np.random.normal(0, 0.1)
        max_system_power = self.battery_params.max_discharge_power
        self.external_power_demand = (base_demand + noise_demand) * max_system_power
        
        # 模拟变化的环境温度
        temp_variation = np.sin(2 * np.pi * time_hours / 24.0) * 5.0 + np.random.normal(0, 1.0)
        self.ambient_temperature = 25.0 + temp_variation
        self.ambient_temperature = np.clip(self.ambient_temperature, 15.0, 40.0)
    
    def _calculate_multi_level_reward(self, 
                                    cluster_record: Dict, 
                                    action: np.ndarray,
                                    upper_layer_weights: Dict[str, float]) -> float:
        """
        计算多层级奖励函数
        体现系统级、BMS间、BMS内的多层级优化目标
        
        Args:
            cluster_record: 集群记录
            action: 动作向量
            upper_layer_weights: 上层权重
            
        Returns:
            总奖励值
        """
        rewards = {}
        
        # === 1. 系统级功率跟踪奖励 ===
        power_command = action[0] * self.battery_params.max_charge_power
        power_error = cluster_record['power_tracking_error']
        max_power_error = abs(power_command) * 0.05 if power_command != 0 else 1000.0  # 5%容差
        power_tracking_reward = 1.0 - min(1.0, power_error / max_power_error)
        rewards['power_tracking'] = power_tracking_reward
        
        # === 2. BMS间均衡奖励 ===
        inter_bms_soc_std = cluster_record['inter_bms_soc_std']
        inter_bms_temp_std = cluster_record['inter_bms_temp_std']
        
        inter_soc_reward = 1.0 - min(1.0, inter_bms_soc_std / 15.0)  # 15%为完全不平衡
        inter_temp_reward = 1.0 - min(1.0, inter_bms_temp_std / 20.0)  # 20℃为完全不平衡
        
        inter_bms_balance_reward = 0.6 * inter_soc_reward + 0.4 * inter_temp_reward
        rewards['inter_bms_balance'] = inter_bms_balance_reward
        
        # === 3. BMS内均衡奖励 ===
        intra_bms_soc_std = cluster_record['avg_intra_bms_soc_std']
        intra_bms_temp_std = cluster_record['avg_intra_bms_temp_std']
        
        intra_soc_reward = 1.0 - min(1.0, intra_bms_soc_std / 8.0)  # 8%为完全不平衡
        intra_temp_reward = 1.0 - min(1.0, intra_bms_temp_std / 12.0)  # 12℃为完全不平衡
        
        intra_bms_balance_reward = 0.6 * intra_soc_reward + 0.4 * intra_temp_reward
        rewards['intra_bms_balance'] = intra_bms_balance_reward
        
        # === 4. 多层级成本奖励 ===
        cost_breakdown = cluster_record.get('cost_breakdown', {})
        current_total_cost = cost_breakdown.get('total_system_cost', 0.0)
        cost_increase = current_total_cost - self.last_total_cost
        self.last_total_cost = current_total_cost
        
        max_cost_increase = 1.0  # 元，单步最大可接受成本增加
        if cost_increase <= 0:
            lifetime_reward = 1.0  # 成本未增加或减少
        else:
            lifetime_reward = 1.0 - min(1.0, cost_increase / max_cost_increase)
        
        rewards['lifetime'] = lifetime_reward
        
        # === 5. 系统效率奖励 ===
        power_efficiency = cluster_record.get('system_power_efficiency', 1.0)
        energy_efficiency = cluster_record.get('cluster_metrics', {}).get('energy_efficiency', 1.0)
        
        overall_efficiency = 0.6 * power_efficiency + 0.4 * energy_efficiency
        efficiency_reward = 2 * overall_efficiency - 1  # [0.5,1] -> [0,1]
        rewards['efficiency'] = efficiency_reward
        
        # === 6. 安全约束惩罚 ===
        safety_penalty = 0.0
        
        # 系统级约束惩罚
        system_constraints = cluster_record.get('system_constraints_active', {})
        if system_constraints.get('thermal_constraints', False):
            safety_penalty += 0.3
        if system_constraints.get('soc_constraints', False):
            safety_penalty += 0.2
        if system_constraints.get('balance_constraints', False):
            safety_penalty += 0.2
        
        # 系统健康状态惩罚
        system_health = cluster_record.get('system_health_status', 'Good')
        health_penalties = {'Critical': 0.5, 'Poor': 0.3, 'Fair': 0.1, 'Good': 0.0}
        safety_penalty += health_penalties.get(system_health, 0.0)
        
        rewards['safety'] = -safety_penalty
        
        # === 7. 协调效率奖励 ===
        coordination_commands = cluster_record.get('coordination_commands', {})
        if coordination_commands:
            # 有协调指令时，评估协调合理性
            avg_coordination_weight = np.mean([cmd.get('coordination_weight', 0.0) 
                                             for cmd in coordination_commands.values()])
            coordination_reward = avg_coordination_weight * 0.1  # 小幅奖励
        else:
            # 无协调指令时，基于系统均衡情况评估
            balance_score = cluster_record.get('cluster_metrics', {}).get('overall_balance_score', 0.5)
            if balance_score > 0.8:
                coordination_reward = 0.05  # 系统均衡良好，无需协调
            else:
                coordination_reward = -0.05  # 系统不均衡但无协调指令
        
        rewards['coordination'] = coordination_reward
        
        # === 8. 加权总奖励 ===
        # 使用动态权重（来自上层DRL决策）
        total_reward = (
            0.25 * rewards['power_tracking'] +                                               # 功率跟踪（固定权重）
            upper_layer_weights['soc_balance'] * 0.3 * (0.6 * rewards['inter_bms_balance'] + 0.4 * rewards['intra_bms_balance']) +  # SOC均衡
            upper_layer_weights['temp_balance'] * 0.2 * rewards['inter_bms_balance'] +      # 温度均衡
            upper_layer_weights['lifetime'] * 0.25 * rewards['lifetime'] +                  # 寿命成本
            upper_layer_weights['efficiency'] * 0.15 * rewards['efficiency'] +              # 效率优化
            0.05 * rewards['safety'] +                                                      # 安全（固定权重）
            0.05 * rewards['coordination']                                                  # 协调效率（固定权重）
        )
        
        return total_reward
    
    def _check_done(self, cluster_record: Dict) -> bool:
        """检查终止条件"""
        # 最大步数
        if self.current_step >= self.max_steps:
            return True
        
        # 系统级安全终止条件
        if cluster_record['system_avg_soh'] < 70.0:  # 系统SOH过低
            return True
        
        if cluster_record['system_avg_temp'] > self.battery_params.MAX_TEMP:  # 系统过温
            return True
        
        # BMS间严重不平衡
        if cluster_record['inter_bms_soc_std'] > 25.0:  # BMS间SOC严重不平衡
            return True
        
        if cluster_record['inter_bms_temp_std'] > 25.0:  # BMS间温度严重不平衡
            return True
        
        # 系统级约束严重程度
        system_constraints = cluster_record.get('system_constraints_active', {})
        active_constraints = sum(1 for active in system_constraints.values() if active)
        if active_constraints >= 3:  # 多个约束同时激活
            return True
        
        # 系统健康状态
        if cluster_record.get('system_health_status') == 'Critical':
            return True
        
        return False
    
    def _build_info_dict(self, 
                        cluster_record: Dict, 
                        action: np.ndarray,
                        upper_layer_weights: Dict[str, float],
                        power_command: float) -> Dict:
        """构建信息字典"""
        
        info = {
            'step': self.current_step,
            
            # 系统级状态
            'system_avg_soc': cluster_record['system_avg_soc'],
            'system_avg_temp': cluster_record['system_avg_temp'],
            'system_avg_soh': cluster_record['system_avg_soh'],
            
            # BMS间均衡指标
            'inter_bms_soc_std': cluster_record['inter_bms_soc_std'],
            'inter_bms_temp_std': cluster_record['inter_bms_temp_std'],
            'inter_bms_soh_std': cluster_record['inter_bms_soh_std'],
            
            # BMS内均衡指标
            'avg_intra_bms_soc_std': cluster_record['avg_intra_bms_soc_std'],
            'avg_intra_bms_temp_std': cluster_record['avg_intra_bms_temp_std'],
            
            # 功率状态
            'total_actual_power': cluster_record['total_actual_power'],
            'power_command': power_command,
            'power_tracking_error': cluster_record['power_tracking_error'],
            'system_power_efficiency': cluster_record['system_power_efficiency'],
            
            # 成本状态
            'total_system_cost': cluster_record.get('cost_breakdown', {}).get('total_system_cost', 0.0),
            'system_cost_increase_rate': cluster_record.get('cost_breakdown', {}).get('system_cost_increase_rate', 0.0),
            
            # 控制状态
            'upper_layer_weights': upper_layer_weights.copy(),
            'coordination_commands_count': len(cluster_record.get('coordination_commands', {})),
            'power_allocation': cluster_record.get('power_allocation', {}),
            
            # 环境状态
            'ambient_temperature': self.ambient_temperature,
            'external_power_demand': self.external_power_demand,
            
            # 约束和安全
            'system_constraints_active': cluster_record.get('system_constraints_active', {}),
            'system_health_status': cluster_record.get('system_health_status', 'Unknown'),
            'system_warning_count': cluster_record.get('system_warning_count', 0),
            'system_alarm_count': cluster_record.get('system_alarm_count', 0),
            
            # 集群管理
            'num_bms': self.num_bms,
            'total_cells': cluster_record.get('total_cells', 0),
            'cluster_id': self.bms_cluster.cluster_id,
            
            # 状态信息（供外部使用）
            'state_names': self.state_names,
            'state_ranges': self.state_ranges
        }
        
        return info
    
    def get_cluster_summary(self) -> Dict:
        """获取集群摘要信息"""
        return self.bms_cluster.get_cluster_summary()
    
    def get_bms_details(self, bms_id: Optional[str] = None) -> Dict:
        """获取BMS详细信息"""
        if bms_id:
            # 返回特定BMS信息
            for bms in self.bms_cluster.bms_list:
                if bms.bms_id == bms_id:
                    return bms.get_bms_summary()
            return {'error': f'BMS {bms_id} not found'}
        else:
            # 返回所有BMS信息
            return {bms.bms_id: bms.get_bms_summary() for bms in self.bms_cluster.bms_list}
    
    def switch_to_lower_layer(self) -> bool:
        """切换到下层动作空间"""
        self.action_space = self.lower_action_space
        print(f"🔄 环境 {self.env_id} 已切换到下层动作空间")
        return True
    
    def switch_to_upper_layer(self) -> bool:
        """切换到上层动作空间"""
        self.action_space = self.upper_action_space
        print(f"🔄 环境 {self.env_id} 已切换到上层动作空间")
        return True
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"StorageEnvironment({self.env_id}): "
                f"BMS数={self.num_bms}, "
                f"步数={self.current_step}/{self.max_steps}, "
                f"状态={self.observation_space.shape}, "
                f"动作={self.action_space.shape}")
