"""
工具模块
提供日志、指标、可视化等通用工具
"""

from .logger import Logger, LogLevel
from .metrics import MetricsCalculator, MetricType
from .visualization import Visualizer, PlotType
from .checkpoint_manager import CheckpointManager, CheckpointType
from .experiment_tracker import ExperimentTracker, ExperimentStatus

__all__ = [
    'Logger',
    'LogLevel', 
    'MetricsCalculator',
    'MetricType',
    'Visualizer',
    'PlotType',
    'CheckpointManager',
    'CheckpointType',
    'ExperimentTracker',
    'ExperimentStatus'
]

__version__ = '0.1.0'

# 工具模块信息
UTILS_INFO = {
    'description': 'Multi-scale Energy Storage DRL Utils',
    'components': {
        'logger': '统一日志系统',
        'metrics': '评估指标计算',
        'visualization': '数据可视化',
        'checkpoint_manager': '检查点管理',
        'experiment_tracker': '实验跟踪'
    },
    'features': [
        '多级日志记录',
        '实时指标监控',
        '交互式可视化',
        '自动检查点保存',
        '实验版本管理'
    ]
}
