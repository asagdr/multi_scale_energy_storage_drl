"""
DRL智能体模块
实现双层深度强化学习架构
"""

# 上层DRL组件
from .upper_layer.transformer_encoder import TransformerEncoder
from .upper_layer.balance_analyzer import BalanceAnalyzer
from .upper_layer.constraint_generator import ConstraintGenerator
from .upper_layer.multi_objective_agent import MultiObjectiveAgent
from .upper_layer.pareto_optimizer import ParetoOptimizer

# 下层DRL组件
from .lower_layer.ddpg_agent import DDPGAgent
from .lower_layer.power_tracker import PowerTracker
from .lower_layer.constraint_handler import ConstraintHandler
from .lower_layer.temperature_compensator import TemperatureCompensator
from .lower_layer.response_optimizer import ResponseOptimizer

# 通信组件
from .communication.message_protocol import MessageProtocol
from .communication.information_flow import InformationFlow
from .communication.data_exchange import DataExchange

__all__ = [
    # 上层DRL
    'TransformerEncoder',
    'BalanceAnalyzer', 
    'ConstraintGenerator',
    'MultiObjectiveAgent',
    'ParetoOptimizer',
    
    # 下层DRL
    'DDPGAgent',
    'PowerTracker',
    'ConstraintHandler', 
    'TemperatureCompensator',
    'ResponseOptimizer',
    
    # 通信
    'MessageProtocol',
    'InformationFlow',
    'DataExchange'
]

__version__ = '0.1.0'
