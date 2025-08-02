"""
BMS模型单元测试
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.battery_params import BatteryParams
from battery_models.bms_model import BMSModel

class TestBMSModel(unittest.TestCase):
    """BMS模型测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.battery_params = BatteryParams()
        self.bms = BMSModel(
            bms_id="TEST_BMS_01",
            cells_count=100,
            battery_params=self.battery_params
        )
    
    def test_bms_initialization(self):
        """测试BMS初始化"""
        self.assertEqual(self.bms.bms_id, "TEST_BMS_01")
        self.assertEqual(self.bms.cells_count, 100)
        self.assertEqual(len(self.bms.cells), 100)
        
        # 检查初始状态
        self.assertAlmostEqual(self.bms.state.avg_soc, 50.0, delta=10.0)
        self.assertAlmostEqual(self.bms.state.avg_temperature, 25.0, delta=5.0)
    
    def test_bms_step_simulation(self):
        """测试BMS仿真步"""
        # 执行充电仿真
        result = self.bms.step(
            bms_power_command=50000.0,  # 50kW充电
            delta_t=1.0,
            ambient_temperature=25.0
        )
        
        # 检查返回结果
        self.assertIn('bms_id', result)
        self.assertIn('avg_soc', result)
        self.assertIn('soc_std', result)
        self.assertIn('actual_power', result)
        self.assertIn('bms_total_cost', result)
        
        # 检查功率范围
        self.assertLessEqual(abs(result['actual_power']), 60000.0)  # 应在合理范围内
    
    def test_power_allocation_to_cells(self):
        """测试单体功率分配"""
        # 创建不平衡的SOC状态
        for i, cell in enumerate(self.bms.cells):
            if i < 20:
                cell.soc = 30.0  # 低SOC
            elif i < 40:
                cell.soc = 70.0  # 高SOC
            else:
                cell.soc = 50.0  # 中等SOC
        
        power_allocation = self.bms._allocate_power_to_cells(10000.0)  # 10kW充电
        
        # 检查分配结果
        self.assertEqual(len(power_allocation), 100)
        self.assertAlmostEqual(sum(power_allocation), 10000.0, delta=100.0)
        
        # 低SOC单体应该获得更多功率
        low_soc_avg_power = np.mean(power_allocation[:20])
        high_soc_avg_power = np.mean(power_allocation[20:40])
        self.assertGreater(low_soc_avg_power, high_soc_avg_power)
    
    def test_bms_cost_calculation(self):
        """测试BMS成本计算"""
        # 模拟单体记录
        cell_records = []
        for i in range(100):
            cell_record = {
                'soc': 50.0 + np.random.normal(0, 2.0),
                'temperature': 25.0 + np.random.normal(0, 3.0),
                'degradation_cost': 0.01 + np.random.uniform(0, 0.005)
            }
            cell_records.append(cell_record)
        
        balancing_result = {'active': True, 'total_power': 100.0}
        
        cost = self.bms._calculate_bms_cost(cell_records, balancing_result)
        
        # 检查成本结构
        self.assertIn('base_cost', cost)
        self.assertIn('soc_imbalance_cost', cost)
        self.assertIn('temp_imbalance_cost', cost)
        self.assertIn('total_cost', cost)
        
        # 成本应为正值
        self.assertGreaterEqual(cost['total_cost'], 0)
        self.assertGreaterEqual(cost['base_cost'], 1.0)  # 100个单体，每个至少0.01元
    
    def test_bms_reset(self):
        """测试BMS重置"""
        # 先执行一些仿真步
        for _ in range(5):
            self.bms.step(1000.0, 1.0)
        
        initial_step_count = self.bms.step_count
        self.assertGreater(initial_step_count, 0)
        
        # 重置BMS
        reset_result = self.bms.reset(target_soc=60.0, target_temp=30.0)
        
        # 检查重置结果
        self.assertEqual(self.bms.step_count, 0)
        self.assertEqual(self.bms.total_time, 0.0)
        self.assertIn('reset_complete', reset_result)
        self.assertTrue(reset_result['reset_complete'])
    
    def test_bms_summary(self):
        """测试BMS摘要"""
        summary = self.bms.get_bms_summary()
        
        required_keys = [
            'bms_id', 'cells_count', 'avg_soc', 'soc_std',
            'avg_temperature', 'temp_std', 'avg_soh', 'total_cost',
            'health_status', 'balancing_active'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)

if __name__ == '__main__':
    unittest.main()
