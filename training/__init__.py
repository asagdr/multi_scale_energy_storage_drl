"""
训练模块
实现分层DRL的专业化训练架构
"""

from .upper_trainer import UpperLayerTrainer
from .lower_trainer import LowerLayerTrainer
from .hierarchical_trainer import HierarchicalTrainer
from .pretraining_pipeline import PretrainingPipeline
from .evaluation_suite import EvaluationSuite

__all__ = [
    'UpperLayerTrainer',
    'LowerLayerTrainer',
    'HierarchicalTrainer',
    'PretrainingPipeline',
    'EvaluationSuite'
]

__version__ = '0.1.0'

# 训练架构重构说明
TRAINING_ARCHITECTURE = {
    'design_principles': {
        'separation_of_concerns': '上下层分离训练',
        'specialized_optimization': '针对性优化策略',
        'hierarchical_coordination': '分层协调机制',
        'modular_evaluation': '模块化评估体系'
    },
    
    'training_components': {
        'upper_trainer': {
            'focus': '5分钟级高层决策训练',
            'algorithm': 'Multi-Objective Transformer DRL',
            'objectives': ['SOC均衡', '温度均衡', '寿命成本', '约束生成'],
            'optimization': 'Pareto多目标优化'
        },
        
        'lower_trainer': {
            'focus': '10ms级底层控制训练',
            'algorithm': 'DDPG with Constraints',
            'objectives': ['功率跟踪', '响应优化', '约束满足', '控制平滑'],
            'optimization': 'Actor-Critic with Experience Replay'
        },
        
        'hierarchical_trainer': {
            'focus': '上下层联合训练协调',
            'coordination': '信息流同步 + 性能反馈',
            'sync_strategy': '自适应同步频率',
            'joint_optimization': '层级化联合优化'
        },
        
        'pretraining_pipeline': {
            'focus': '预训练流水线',
            'stages': ['上层预训练', '下层预训练', '联合微调'],
            'data_strategy': '渐进式数据增强',
            'transfer_learning': '跨层知识迁移'
        },
        
        'evaluation_suite': {
            'focus': '全面评估体系',
            'metrics': ['性能指标', '稳定性指标', '鲁棒性指标'],
            'scenarios': ['标准测试', '极限测试', '对抗测试'],
            'benchmarking': '基准对比分析'
        }
    }
}
