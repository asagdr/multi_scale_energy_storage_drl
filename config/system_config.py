from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum
import time

class SystemMode(Enum):
    """系统运行模式"""
    SIMULATION = "simulation"      # 纯仿真模式
    HARDWARE_IN_LOOP = "hil"      # 硬件在环
    REAL_DEPLOYMENT = "real"      # 实际部署

class ControlStrategy(Enum):
    """控制策略"""
    POWER_TRACKING = "power_tracking"    # 功率跟踪
    PEAK_SHAVING = "peak_shaving"        # 削峰填谷
    FREQUENCY_REGULATION = "freq_reg"    # 频率调节
    VOLTAGE_SUPPORT = "voltage_support"  # 电压支撑
    ENERGY_ARBITRAGE = "arbitrage"       # 能量套利

@dataclass
class SystemConfig:
    """系统配置类"""
    
    # === 基本系统信息 ===
    system_name: str = "Multi-Scale Energy Storage DRL System"
    version: str = "0.1.0"
    mode: SystemMode = SystemMode.SIMULATION
    
    # === 时间尺度参数 ===
    UPPER_LAYER_INTERVAL: float = 300.0     # s, 上层决策间隔 (5分钟)
    LOWER_LAYER_INTERVAL: float = 0.01      # s, 下层控制间隔 (10ms)
    SIMULATION_TIME_STEP: float = 1.0       # s, 仿真基础时间步长
    
    # === 控制参数 ===
    control_strategy: ControlStrategy = ControlStrategy.POWER_TRACKING
    enable_degradation_optimization: bool = True
    enable_thermal_management: bool = True
    enable_load_balancing: bool = True
    
    # === 安全参数 ===
    EMERGENCY_SHUTDOWN_TEMP: float = 65.0   # ℃, 紧急停机温度
    EMERGENCY_SHUTDOWN_VOLTAGE: float = 2.0 # V, 紧急停机电压
    MAX_POWER_RAMP_RATE: float = 100.0      # kW/s, 最大功率变化率
    
    # === 通信参数 ===
    UPPER_LOWER_COMM_DELAY: float = 0.001   # s, 上下层通信延迟
    MAX_COMM_TIMEOUT: float = 1.0           # s, 最大通信超时
    
    # === 数据记录参数 ===
    DATA_LOGGING_INTERVAL: float = 1.0      # s, 数据记录间隔
    MAX_HISTORY_LENGTH: int = 86400          # 最大历史记录长度 (1天)
    ENABLE_DETAILED_LOGGING: bool = True     # 是否启用详细日志
    
    # === BMS集群配置 ===
    BMS_CLUSTER_ENABLED: bool = True           # 启用BMS集群管理
    BMS_COORDINATION_MODE: str = "comprehensive"  # 协调模式: disabled/soc_balance/thermal_balance/comprehensive
    
    # BMS间通信参数
    INTER_BMS_COMM_INTERVAL: float = 1.0       # BMS间通信间隔 (s)
    BMS_COORDINATION_INTERVAL: float = 10.0    # BMS协调间隔 (s)
    MAX_COORDINATION_DURATION: float = 300.0   # 最大协调持续时间 (s)
    
    # 功率分配参数
    POWER_ALLOCATION_UPDATE_FREQ: float = 1.0  # 功率分配更新频率 (Hz)
    ADAPTIVE_ALLOCATION_ENABLED: bool = True   # 启用自适应分配
    ALLOCATION_EFFICIENCY_THRESHOLD: float = 0.85  # 分配效率阈值
    
    # 多层级成本优化参数
    ENABLE_MULTI_LEVEL_COST: bool = True       # 启用多层级成本优化
    COST_OPTIMIZATION_WEIGHT: float = 0.3      # 成本优化权重
    BMS_IMBALANCE_PENALTY_FACTOR: float = 0.05 # BMS不平衡惩罚因子
    SYSTEM_COORDINATION_PENALTY_FACTOR: float = 0.1  # 系统协调惩罚因子
    
    # === 优化目标权重 ===
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'power_tracking': 0.4,      # 功率跟踪权重
        'lifetime_extension': 0.3,  # 寿命延长权重
        'thermal_balance': 0.2,     # 热平衡权重
        'efficiency': 0.1           # 效率权重
    })
    
    # === 约束参数 ===
    constraints: Dict[str, Dict] = field(default_factory=lambda: {
        'power': {
            'max_charge_power_ratio': 0.95,    # 最大充电功率比例
            'max_discharge_power_ratio': 0.95, # 最大放电功率比例
            'ramp_rate_limit': True             # 是否启用功率变化率限制
        },
        'thermal': {
            'max_temp_rise_rate': 2.0,          # ℃/min, 最大温升速率
            'temp_difference_limit': 10.0,      # ℃, 最大温差限制
            'cooling_strategy': 'active'        # 冷却策略
        },
        'lifetime': {
            'max_daily_cycles': 2.0,           # 每日最大等效循环次数
            'deep_discharge_limit': 0.9,       # 深度放电限制 (DOD)
            'aging_acceleration_threshold': 1.5 # 老化加速阈值
        }
    })
    
    # === 仿真特定参数 ===
    simulation_config: Dict = field(default_factory=lambda: {
        'enable_noise': True,               # 是否启用传感器噪声
        'noise_scale': 1.0,                # 噪声比例因子
        'enable_faults': False,            # 是否启用故障注入
        'random_seed': 42,                 # 随机种子
        'parallel_episodes': 1             # 并行仿真数量
    })
    
    # === DRL训练相关 ===
    drl_config: Dict = field(default_factory=lambda: {
        'state_normalization': True,       # 状态归一化
        'action_clipping': True,          # 动作裁剪
        'reward_shaping': True,           # 奖励塑形
        'curriculum_learning': False,     # 课程学习
        'transfer_learning': False        # 迁移学习
    })
    
    def __post_init__(self):
        """初始化后验证配置"""
        self._validate_config()
        self._calculate_derived_params()
        self._validate_bms_cluster_config()  # 新增：验证BMS集群配置
    
    def _validate_config(self):
        """验证配置参数"""
        # 时间参数验证
        if self.UPPER_LAYER_INTERVAL <= self.LOWER_LAYER_INTERVAL:
            raise ValueError("上层决策间隔必须大于下层控制间隔")
        
        if self.SIMULATION_TIME_STEP > self.LOWER_LAYER_INTERVAL:
            raise ValueError("仿真时间步长不能大于下层控制间隔")
        
        # 权重验证
        total_weight = sum(self.objective_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"目标权重总和必须为1.0，当前为{total_weight}")
        
        # 约束验证
        power_constraints = self.constraints['power']
        if not (0 < power_constraints['max_charge_power_ratio'] <= 1.0):
            raise ValueError("最大充电功率比例必须在(0,1]范围内")
    
    def _validate_bms_cluster_config(self):
        """验证BMS集群配置"""
        # BMS协调间隔验证
        if self.BMS_COORDINATION_INTERVAL < self.INTER_BMS_COMM_INTERVAL:
            raise ValueError("BMS协调间隔不能小于通信间隔")
        
        if self.BMS_COORDINATION_INTERVAL > self.MAX_COORDINATION_DURATION:
            raise ValueError("BMS协调间隔不能大于最大协调持续时间")
        
        # 功率分配频率验证
        if self.POWER_ALLOCATION_UPDATE_FREQ <= 0:
            raise ValueError("功率分配更新频率必须大于0")
        
        # 成本优化权重验证
        if not (0.0 <= self.COST_OPTIMIZATION_WEIGHT <= 1.0):
            raise ValueError("成本优化权重必须在[0,1]范围内")
    
    def _calculate_derived_params(self):
        """计算衍生参数"""
        # 计算时间比例
        self.time_scale_ratio = self.UPPER_LAYER_INTERVAL / self.LOWER_LAYER_INTERVAL
        
        # 计算每个上层周期的下层步数
        self.lower_steps_per_upper = int(self.UPPER_LAYER_INTERVAL / self.LOWER_LAYER_INTERVAL)
        
        # 计算仿真步数比例
        self.sim_steps_per_lower = int(self.LOWER_LAYER_INTERVAL / self.SIMULATION_TIME_STEP)
    
    def get_communication_config(self) -> Dict:
        """获取通信配置"""
        return {
            'upper_to_lower': {
                'interval': self.UPPER_LAYER_INTERVAL,
                'delay': self.UPPER_LOWER_COMM_DELAY,
                'timeout': self.MAX_COMM_TIMEOUT
            },
            'lower_to_upper': {
                'interval': self.UPPER_LAYER_INTERVAL,  # 统计信息上报间隔
                'delay': self.UPPER_LOWER_COMM_DELAY,
                'timeout': self.MAX_COMM_TIMEOUT
            }
        }
    
    def get_bms_cluster_config(self) -> Dict:
        """获取BMS集群配置"""
        return {
            'cluster_enabled': self.BMS_CLUSTER_ENABLED,
            'coordination_mode': self.BMS_COORDINATION_MODE,
            'inter_bms_comm_interval': self.INTER_BMS_COMM_INTERVAL,
            'coordination_interval': self.BMS_COORDINATION_INTERVAL,
            'max_coordination_duration': self.MAX_COORDINATION_DURATION,
            'power_allocation_update_freq': self.POWER_ALLOCATION_UPDATE_FREQ,
            'adaptive_allocation_enabled': self.ADAPTIVE_ALLOCATION_ENABLED,
            'allocation_efficiency_threshold': self.ALLOCATION_EFFICIENCY_THRESHOLD,
            'multi_level_cost_enabled': self.ENABLE_MULTI_LEVEL_COST,
            'cost_optimization_weight': self.COST_OPTIMIZATION_WEIGHT,
            'bms_imbalance_penalty_factor': self.BMS_IMBALANCE_PENALTY_FACTOR,
            'system_coordination_penalty_factor': self.SYSTEM_COORDINATION_PENALTY_FACTOR
        }
    
    def get_safety_limits(self) -> Dict:
        """获取安全限制"""
        return {
            'temperature': {
                'emergency_shutdown': self.EMERGENCY_SHUTDOWN_TEMP,
                'max_rise_rate': self.constraints['thermal']['max_temp_rise_rate']
            },
            'voltage': {
                'emergency_shutdown': self.EMERGENCY_SHUTDOWN_VOLTAGE
            },
            'power': {
                'max_ramp_rate': self.MAX_POWER_RAMP_RATE
            }
        }
    
    def update_objective_weights(self, new_weights: Dict[str, float]):
        """更新目标权重"""
        if abs(sum(new_weights.values()) - 1.0) > 1e-6:
            raise ValueError("新权重总和必须为1.0")
        
        self.objective_weights.update(new_weights)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'system_info': {
                'name': self.system_name,
                'version': self.version,
                'mode': self.mode.value
            },
            'time_scales': {
                'upper_interval': self.UPPER_LAYER_INTERVAL,
                'lower_interval': self.LOWER_LAYER_INTERVAL,
                'simulation_step': self.SIMULATION_TIME_STEP,
                'time_scale_ratio': self.time_scale_ratio
            },
            'objectives': self.objective_weights,
            'constraints': self.constraints,
            'safety_limits': self.get_safety_limits(),
            'bms_cluster': self.get_bms_cluster_config()
        }

# 预定义系统配置
class PresetSystemConfigs:
    """预定义系统配置"""
    
    @staticmethod
    def research_simulation() -> SystemConfig:
        """研究仿真配置"""
        return SystemConfig(
            mode=SystemMode.SIMULATION,
            SIMULATION_TIME_STEP=0.1,
            simulation_config={
                'enable_noise': False,
                'enable_faults': False,
                'parallel_episodes': 4
            }
        )
    
    @staticmethod
    def realistic_simulation() -> SystemConfig:
        """真实仿真配置"""
        return SystemConfig(
            mode=SystemMode.SIMULATION,
            simulation_config={
                'enable_noise': True,
                'noise_scale': 1.0,
                'enable_faults': True
            }
        )
    
    @staticmethod
    def power_plant_deployment() -> SystemConfig:
        """电站部署配置"""
        config = SystemConfig(
            mode=SystemMode.REAL_DEPLOYMENT,
            control_strategy=ControlStrategy.PEAK_SHAVING
        )
        # 更保守的安全参数
        config.EMERGENCY_SHUTDOWN_TEMP = 55.0
        config.constraints['power']['max_charge_power_ratio'] = 0.85
        config.constraints['power']['max_discharge_power_ratio'] = 0.85
        return config
    
    @staticmethod
    def bms_cluster_research() -> SystemConfig:
        """BMS集群研究配置"""
        config = SystemConfig(
            mode=SystemMode.SIMULATION,
            SIMULATION_TIME_STEP=0.1,
            BMS_CLUSTER_ENABLED=True,
            BMS_COORDINATION_MODE="comprehensive",
            ENABLE_MULTI_LEVEL_COST=True,
            simulation_config={
                'enable_noise': False,
                'enable_faults': False,
                'parallel_episodes': 4
            }
        )
        return config
    
    @staticmethod
    def bms_cluster_realistic() -> SystemConfig:
        """BMS集群真实仿真配置"""
        config = SystemConfig(
            mode=SystemMode.SIMULATION,
            BMS_CLUSTER_ENABLED=True,
            BMS_COORDINATION_MODE="comprehensive",
            ADAPTIVE_ALLOCATION_ENABLED=True,
            ENABLE_MULTI_LEVEL_COST=True,
            simulation_config={
                'enable_noise': True,
                'noise_scale': 1.0,
                'enable_faults': True
            }
        )
        return config
    
    @staticmethod
    def bms_cluster_deployment() -> SystemConfig:
        """BMS集群部署配置"""
        config = SystemConfig(
            mode=SystemMode.REAL_DEPLOYMENT,
            control_strategy=ControlStrategy.PEAK_SHAVING,
            BMS_CLUSTER_ENABLED=True,
            BMS_COORDINATION_MODE="soc_balance",  # 部署时更保守
            ADAPTIVE_ALLOCATION_ENABLED=True,
            ENABLE_MULTI_LEVEL_COST=True
        )
        # 更保守的安全参数
        config.EMERGENCY_SHUTDOWN_TEMP = 55.0
        config.constraints['power']['max_charge_power_ratio'] = 0.85
        config.constraints['power']['max_discharge_power_ratio'] = 0.85
        return config
