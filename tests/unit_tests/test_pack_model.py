import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from battery_models.battery_pack_model import BatteryPackModel, PackTopology, BalancingStrategy
from config.battery_params import BatteryParams
from config.system_config import SystemConfig

class TestBatteryPackModel(unittest.TestCase):
    """电池组模型测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 使用小规模配置便于测试
        self.battery_params = BatteryParams()
        self.battery_params.SERIES_NUM = 4    # 4串
        self.battery_params.PARALLEL_NUM = 2  # 2并
        
        self.system_config = SystemConfig()
        self.pack_model = BatteryPackModel(
            battery_params=self.battery_params,
            system_config=self.system_config,
            pack_topology=PackTopology.SERIES_PARALLEL,
            balancing_strategy=BalancingStrategy.ACTIVE,
            pack_id="TestPack"
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.pack_model.pack_id, "TestPack")
        self.assertEqual(self.pack_model.pack_topology, PackTopology.SERIES_PARALLEL)
        self.assertEqual(self.pack_model.balancing_strategy, BalancingStrategy.ACTIVE)
        self.assertEqual(self.pack_model.total_cells, 8)  # 4S2P = 8个单体
        self.assertEqual(len(self.pack_model.cells), 8)
        self.assertEqual(len(self.pack_model.thermal_models), 8)
        self.assertEqual(len(self.pack_model.degradation_models), 8)
    
    def test_pack_electrical_calculation(self):
        """测试组电气状态计算"""
        electrical_state = self.pack_model.calculate_pack_electrical_state()
        
        # 验证返回字段
        required_keys = ['pack_voltage', 'pack_current', 'pack_power', 'pack_energy']
        for key in required_keys:
            self.assertIn(key, electrical_state)
        
        # 验证电压约为单体电压×串联数
        expected_voltage_range = (3.0 * self.battery_params.SERIES_NUM, 
                                3.8 * self.battery_params.SERIES_NUM)
        self.assertGreater(electrical_state['pack_voltage'], expected_voltage_range[0])
        self.assertLess(electrical_state['pack_voltage'], expected_voltage_range[1])
        
        # 验证能量为正值
        self.assertGreater(electrical_state['pack_energy'], 0)
    
    def test_soc_balance_metrics(self):
        """测试SOC均衡指标计算"""
        soc_balance = self.pack_model.calculate_soc_balance_metrics()
        
        # 验证关键指标
        required_keys = ['pack_soc', 'soc_variance', 'soc_std', 'soc_range', 
                        'max_soc_diff', 'soc_consistency', 'balancing_urgency']
        for key in required_keys:
            self.assertIn(key, soc_balance)
        
        # 验证SOC范围
        self.assertGreaterEqual(soc_balance['pack_soc'], 0.0)
        self.assertLessEqual(soc_balance['pack_soc'], 100.0)
        self.assertGreaterEqual(soc_balance['soc_std'], 0.0)  # σ_SOC应非负
        self.assertGreaterEqual(soc_balance['soc_consistency'], 0.0)
        self.assertLessEqual(soc_balance['soc_consistency'], 1.0)
    
    def test_temperature_balance_metrics(self):
        """测试温度均衡指标计算"""
        temp_balance = self.pack_model.calculate_temperature_balance_metrics()
        
        required_keys = ['pack_temperature', 'temp_variance', 'temp_std', 
                        'temp_range', 'max_temp_diff', 'temp_consistency', 'cooling_urgency']
        for key in required_keys:
            self.assertIn(key, temp_balance)
        
        # 验证温度范围
        self.assertGreater(temp_balance['pack_temperature'], 0.0)
        self.assertLess(temp_balance['pack_temperature'], 100.0)
        self.assertGreaterEqual(temp_balance['temp_std'], 0.0)
    
    def test_degradation_balance_metrics(self):
        """测试劣化均衡指标计算"""
        degradation_balance = self.pack_model.calculate_degradation_balance_metrics()
        
        required_keys = ['pack_soh', 'soh_variance', 'soh_std', 'soh_range',
                        'avg_degradation_rate', 'total_degradation_cost', 
                        'degradation_consistency', 'lifetime_urgency']
        for key in required_keys:
            self.assertIn(key, degradation_balance)
        
        # 验证SOH范围
        self.assertGreaterEqual(degradation_balance['pack_soh'], 0.0)
        self.assertLessEqual(degradation_balance['pack_soh'], 100.0)
        self.assertGreaterEqual(degradation_balance['total_degradation_cost'], 0.0)
    
    def test_pack_constraints_calculation(self):
        """测试组约束计算"""
        constraints = self.pack_model.calculate_pack_constraints()
        
        required_keys = ['pack_current_limits', 'pack_power_limits', 
                        'thermal_constraints_active', 'degradation_constraints_active',
                        'constraint_severity', 'response_time_limit']
        for key in required_keys:
            self.assertIn(key, constraints)
        
        # 验证电流限制
        charge_limit, discharge_limit = constraints['pack_current_limits']
        self.assertGreater(charge_limit, 0)
        self.assertGreater(discharge_limit, 0)
        
        # 验证功率限制
        charge_power_limit, discharge_power_limit = constraints['pack_power_limits']
        self.assertGreater(charge_power_limit, 0)
        self.assertGreater(discharge_power_limit, 0)
        
        # 验证约束严重程度
        self.assertGreaterEqual(constraints['constraint_severity'], 0.0)
        self.assertLessEqual(constraints['constraint_severity'], 1.0)
    
    def test_power_distribution(self):
        """测试功率分配"""
        pack_power_command = 10000.0  # W
        
        cell_powers = self.pack_model.distribute_pack_power(pack_power_command)
        
        # 验证分配结果
        self.assertEqual(len(cell_powers), self.pack_model.total_cells)
        
        # 验证功率守恒
        total_distributed_power = sum(cell_powers)
        self.assertAlmostEqual(total_distributed_power, pack_power_command, places=1)
        
        # 验证单体功率合理性
        for power in cell_powers:
            self.assertGreater(abs(power), 0)
            self.assertLess(abs(power), pack_power
