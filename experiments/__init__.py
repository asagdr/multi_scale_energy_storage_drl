"""
实验模块
提供各种储能系统DRL实验的完整实现
"""

from .basic_experiments import BasicExperiment, ExperimentType
from .ablation_studies import AblationStudy, AblationComponent
from .sensitivity_analysis import SensitivityAnalysis, ParameterType
from .case_studies.peak_shaving import PeakShavingExperiment
from .case_studies.frequency_regulation import FrequencyRegulationExperiment
from .case_studies.energy_arbitrage import EnergyArbitrageExperiment

__all__ = [
    'BasicExperiment',
    'ExperimentType',
    'AblationStudy',
    'AblationComponent',
    'SensitivityAnalysis',
    'ParameterType',
    'PeakShavingExperiment',
    'FrequencyRegulationExperiment',
    'EnergyArbitrageExperiment'
]

__version__ = '0.1.0'

# 实验模块信息
EXPERIMENTS_INFO = {
    'description': 'Multi-scale Energy Storage DRL Experiments Suite',
    'components': {
        'basic_experiments': '基础实验框架',
        'ablation_studies': '消融实验',
        'sensitivity_analysis': '敏感性分析',
        'case_studies': '应用案例研究'
    },
    'experiment_types': [
        'single_objective_training',
        'multi_objective_training',
        'hierarchical_comparison',
        'benchmark_comparison',
        'robustness_testing',
        'generalization_testing'
    ],
    'case_studies': [
        'peak_shaving',        # 削峰填谷
        'frequency_regulation', # 频率调节
        'energy_arbitrage'     # 能量套利
    ]
}
