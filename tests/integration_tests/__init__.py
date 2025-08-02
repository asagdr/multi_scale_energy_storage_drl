"""
集成测试模块
测试不同组件之间的集成功能
"""

from .test_training_pipeline import TestTrainingPipeline
from .test_experiment_workflow import TestExperimentWorkflow
from .test_data_flow import TestDataFlow
from .test_system_integration import TestSystemIntegration

__all__ = [
    'TestTrainingPipeline',
    'TestExperimentWorkflow', 
    'TestDataFlow',
    'TestSystemIntegration'
]

__version__ = '0.1.0'

# 集成测试信息
INTEGRATION_TESTS_INFO = {
    'description': 'Integration tests for multi-scale energy storage DRL system',
    'test_suites': {
        'training_pipeline': '训练流水线集成测试',
        'experiment_workflow': '实验工作流集成测试',
        'data_flow': '数据流集成测试',
        'system_integration': '系统集成测试'
    },
    'coverage_areas': [
        'component_interaction',
        'data_consistency',
        'workflow_integrity',
        'performance_validation'
    ]
}
