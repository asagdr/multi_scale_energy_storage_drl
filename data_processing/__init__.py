"""
数据处理模块
负责生成仿真数据和特征处理
"""

from .scenario_generator import ScenarioGenerator, ScenarioType
from .load_profile_generator import LoadProfileGenerator, LoadPattern
from .weather_simulator import WeatherSimulator, WeatherCondition
from .feature_extractor import FeatureExtractor, FeatureType
from .data_preprocessor import DataPreprocessor, PreprocessingMethod

__all__ = [
    'ScenarioGenerator',
    'ScenarioType',
    'LoadProfileGenerator',
    'LoadPattern',
    'WeatherSimulator',
    'WeatherCondition',
    'FeatureExtractor',
    'FeatureType',
    'DataPreprocessor',
    'PreprocessingMethod'
]

__version__ = '0.1.0'

# 数据处理架构说明
DATA_PROCESSING_INFO = {
    'architecture': 'simulation_based_data_generation',
    'components': {
        'scenario_generator': {
            'purpose': '生成多样化的仿真场景',
            'scenarios': ['daily_cycle', 'seasonal_variation', 'emergency', 'grid_support'],
            'parameters': ['duration', 'complexity', 'disturbance_level']
        },
        'load_profile_generator': {
            'purpose': '生成真实的负荷曲线',
            'patterns': ['residential', 'commercial', 'industrial', 'renewable'],
            'features': ['peak_shaving', 'load_following', 'frequency_regulation']
        },
        'weather_simulator': {
            'purpose': '模拟环境条件影响',
            'factors': ['temperature', 'humidity', 'solar_irradiance', 'wind_speed'],
            'impact': ['thermal_behavior', 'cooling_demand', 'renewable_generation']
        },
        'feature_extractor': {
            'purpose': '提取关键特征',
            'types': ['temporal', 'frequency', 'statistical', 'physical'],
            'methods': ['windowing', 'fourier_transform', 'wavelets', 'embedding']
        },
        'data_preprocessor': {
            'purpose': '数据预处理和增强',
            'methods': ['normalization', 'augmentation', 'filtering', 'resampling'],
            'quality': ['noise_reduction', 'outlier_detection', 'missing_data_handling']
        }
    }
}
