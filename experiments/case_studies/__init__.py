"""
案例研究模块
提供具体的储能系统应用案例研究
"""

from .peak_shaving import PeakShavingExperiment
from .frequency_regulation import FrequencyRegulationExperiment
from .energy_arbitrage import EnergyArbitrageExperiment

__all__ = [
    'PeakShavingExperiment',
    'FrequencyRegulationExperiment', 
    'EnergyArbitrageExperiment'
]

__version__ = '0.1.0'

# 案例研究信息
CASE_STUDIES_INFO = {
    'description': 'Real-world energy storage application case studies',
    'cases': {
        'peak_shaving': {
            'name': '削峰填谷',
            'objective': '降低电网峰值负荷，减少电费成本',
            'scenarios': ['commercial_building', 'industrial_facility', 'residential_complex'],
            'key_metrics': ['peak_reduction_ratio', 'cost_savings', 'load_factor_improvement']
        },
        'frequency_regulation': {
            'name': '频率调节',
            'objective': '维持电网频率稳定，提供辅助服务',
            'scenarios': ['primary_reserve', 'secondary_reserve', 'tertiary_reserve'],
            'key_metrics': ['frequency_response_time', 'regulation_accuracy', 'service_revenue']
        },
        'energy_arbitrage': {
            'name': '能量套利',
            'objective': '利用电价差异获得经济收益',
            'scenarios': ['time_of_use_pricing', 'real_time_pricing', 'renewable_integration'],
            'key_metrics': ['arbitrage_profit', 'round_trip_efficiency', 'market_participation']
        }
    }
}
