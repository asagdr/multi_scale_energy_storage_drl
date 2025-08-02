"""
DRL通信模块
实现上下层DRL之间的信息交换和协调
"""

from .message_protocol import MessageProtocol, MessageType, Priority
from .information_flow import InformationFlow, FlowDirection
from .data_exchange import DataExchange, ExchangeMode

__all__ = [
    'MessageProtocol',
    'MessageType',
    'Priority',
    'InformationFlow', 
    'FlowDirection',
    'DataExchange',
    'ExchangeMode'
]

__version__ = '0.1.0'

# 通信架构说明
COMMUNICATION_INFO = {
    'architecture': 'hierarchical_bidirectional',
    'upper_to_lower': {
        'constraint_matrix': 'C_t约束矩阵传递',
        'weight_vector': 'w_t权重向量传递', 
        'balance_targets': '均衡目标设定',
        'safety_commands': '安全指令下发'
    },
    'lower_to_upper': {
        'performance_feedback': '性能反馈上报',
        'constraint_violations': '约束违反上报',
        'system_status': '系统状态上报',
        'emergency_alerts': '紧急告警上报'
    },
    'synchronization': {
        'upper_layer_cycle': '5_minutes',
        'lower_layer_cycle': '10_milliseconds',
        'sync_mechanism': 'event_driven_with_periodic_sync'
    }
}
