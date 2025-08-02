"""
基准测试模块
提供性能基准测试和对比分析
"""

from .performance_benchmarks import PerformanceBenchmark
from .algorithm_comparison import AlgorithmComparison
from .scalability_tests import ScalabilityTest

__all__ = [
    'PerformanceBenchmark',
    'AlgorithmComparison', 
    'ScalabilityTest'
]

__version__ = '0.1.0'

# 基准测试信息
BENCHMARK_INFO = {
    'description': 'Performance benchmarks for multi-scale energy storage DRL system',
    'benchmark_types': {
        'performance': '性能基准测试',
        'algorithm_comparison': '算法对比测试',
        'scalability': '可扩展性测试'
    },
    'metrics': [
        'training_speed',
        'convergence_rate',
        'memory_usage',
        'cpu_utilization',
        'final_performance',
        'sample_efficiency'
    ]
}
