"""
下层DRL模块
实现10ms级的底层控制：功率跟踪、响应优化、约束处理
"""

from .ddpg_agent import DDPGAgent
from .power_tracker import PowerTracker
from .constraint_handler import ConstraintHandler
from .temperature_compensator import TemperatureCompensator
from .response_optimizer import ResponseOptimizer

__all__ = [
    'DDPGAgent',
    'PowerTracker',
    'ConstraintHandler',
    'TemperatureCompensator',
    'ResponseOptimizer'
]

__version__ = '0.1.0'

# 下层DRL架构说明
LOWER_LAYER_INFO = {
    'time_scale': '10_milliseconds',
    'objectives': [
        'power_tracking',       # 功率跟踪精度
        'response_speed',       # 响应速度
        'constraint_satisfaction',  # 约束满足
        'control_smoothness'    # 控制平滑性
    ],
    'key_inputs': [
        'upper_layer_commands',    # 上层指令
        'constraint_matrix',       # 约束矩阵 C_t
        'real_time_state',        # 实时状态
        'disturbances'            # 扰动信息
    ],
    'key_outputs': [
        'power_control_signal',   # 功率控制信号
        'balancing_commands',     # 均衡指令
        'safety_actions'          # 安全动作
    ]
}
