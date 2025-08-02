"""
电池模型模块
包含电池系统的各个组件模型
"""

from .battery_cell_model import BatteryCellModel
from .thermal_model import ThermalModel
from .degradation_model import BatteryDegradationModel, DegradationMode
from .battery_pack_model import BatteryPackModel, PackTopology, BalancingStrategy
from .pack_manager import PackManager

__all__ = [
    'BatteryCellModel',
    'ThermalModel', 
    'BatteryDegradationModel',
    'DegradationMode',
    'BatteryPackModel',
    'PackTopology',
    'BalancingStrategy',
    'PackManager'
]

__version__ = '0.1.0'
