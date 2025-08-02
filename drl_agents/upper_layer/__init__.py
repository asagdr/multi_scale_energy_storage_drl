"""
上层DRL模块
实现5分钟级的高层决策：SOC均衡、温度均衡、寿命成本最小化
"""

from .transformer_encoder import TransformerEncoder
from .balance_analyzer import BalanceAnalyzer
from .constraint_generator import ConstraintGenerator  
from .multi_objective_agent import MultiObjectiveAgent
from .pareto_optimizer import ParetoOptimizer

__all__ = [
    'TransformerEncoder',
    'BalanceAnalyzer',
    'ConstraintGenerator',
    'MultiObjectiveAgent', 
    'ParetoOptimizer'
]

__version__ = '0.1.0'

# 上层DRL架构说明
UPPER_LAYER_INFO = {
    'time_scale': '5_minutes',
    'objectives': [
        'soc_balance',      # SOC均衡 (σ_SOC最小化)
        'temp_balance',     # 温度均衡
        'lifetime_cost',    # 寿命成本最小化
        'constraint_satisfaction'  # 约束满足
    ],
    'key_outputs': [
        'constraint_matrix',    # 约束矩阵 C_t
        'weight_vector',       # 权重向量 w_t
        'balance_targets'      # 均衡目标
    ]
}
