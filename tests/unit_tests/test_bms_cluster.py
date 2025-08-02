"""
BMS集群管理器单元测试
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig
from battery_models.bms_cluster_manager import BMSClusterManager

class TestBMSClusterManager(unittest.TestCase):
    """BMS集群管理器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.battery_params = BatteryParams()
        self.system_config = SystemConfig()
        self.cluster = BMSClusterManager(
            battery_params=self.battery_params,
            system_config=self.system_config,
            num_bms=5,  # 使用5个BMS进行测试
            cluster_id="TEST_CLUSTER"
        )
    
    def test_cluster_initialization(self):
        """测试集群初始化"""
        self.assertEqual(self.cluster.num_bms, 5)
        self.assertEqual(len(self.cluster.bms_list), 5)
        self.assertEqual(self.cluster.cluster_id, "TEST_CLUSTER")
        
        # 检查BMS ID
        expected_ids = [f"BMS_{i+1:02d}" for i in range(5)]
        actual_ids = [bms.bms_id for bms in self.cluster.bms_list]
        self.assertEqual(actual_ids, expected_ids)
    
    def test_cluster_step_simulation(self):
        """测试集群仿真步"""
        upper_weights = {
            'soc_balance': 0.3,
            'temp_balance': 0.2,
            'lifetime': 0.3,
            'efficiency': 0.2
        }
        
        result = self.cluster.step(
            total_power_command=100000.0,  # 100kW
            delta_t=1.0,
            upper_layer_weights=upper_weights,
            ambient_temperature=25.0
        )
        
        # 检查返回结果
        required_keys = [
            'cluster_id', 'bms_records', 'system_avg_soc',
            'inter_bms_soc_std', 'total_actual_power',
            'power_allocation', 'cost_breakdown'
        ]
        
        for key in required_keys:
            self.assertIn(key, result)
        
        # 检查BMS记录数量
        self.assertEqual(len(result['bms_records']), 5)
        
        # 检查功率分配
        power_allocation = result['power_allocation']
        self.assertEqual(len(power_allocation), 5)
        total_allocated = sum(power_allocation.values())
        self.assertAlmostEqual(total_allocated, 100000.0, delta=1000.0)
    
    def test_power_allocation(self):
        """测试功率分配"""
        # 创建不平衡状态
        for i, bms in enumerate(self.cluster.bms_list):
            if i < 2:
                # 前两个BMS设置低SOC
                for cell in bms.cells[:50]:  # 前50个单体
                    cell.soc = 30.0
            else:
                # 后三个BMS设置高SOC
                for cell in bms.cells[:50]:
                    cell.soc = 70.0
        
        power_allocation = self.cluster.power_allocator.allocate_power(
            total_power_command=50000.0,
            upper_layer_weights={'soc_balance': 1.0, 'temp_balance': 0.0, 'lifetime': 0.0, 'efficiency': 0.0}
        )
        
        # 检查分配结果
        self.assertEqual(len(power_allocation), 5)
        
        # 低SOC的BMS应该获得更多充电功率
        low_soc_power = sum([power_allocation[f"BMS_{i+1:02d}"] for i in range(2)])
        high_soc_power = sum([power_allocation[f"BMS_{i+1:02d}"] for i in range(2, 5)])
        
        if low_soc_power > 0 and high_soc_power > 0:  # 如果都是充电功率
            self.assertGreater(low_soc_power / 2, high_soc_power / 3)  # 平均功率比较
    
    def test_inter_bms_coordination(self):
        """测试BMS间协调"""
        # 创建需要协调的状态
        soc_values = [30.0, 40.0, 50.0, 60.0, 70.0]  # 大范围SOC差异
        
        for i, (bms, target_soc) in enumerate(zip(self.cluster.bms_list, soc_values)):
            for cell in bms.cells:
                cell.soc = target_soc + np.random.normal(0, 1.0)
        
        coordination_commands = self.cluster.inter_bms_coordinator.generate_coordination_commands()
        
        # 应该有协调指令
        self.assertGreater(len(coordination_commands), 0)
        
        # 检查协调指令结构
        for bms_id, command in coordination_commands.items():
            self.assertIn('command_type', command)
            self.assertIn('suggested_power_bias', command)
            self.assertIn('description', command)
    
    def test_multi_level_cost_calculation(self):
        """测试多层级成本计算"""
        # 执行仿真获取BMS记录
        result = self.cluster.step(
            total_power_command=50000.0,
            delta_t=1.0,
            upper_layer_weights={'soc_balance': 0.3, 'temp_balance': 0.2, 'lifetime': 0.3, 'efficiency': 0.2}
        )
        
        cost_breakdown = result['cost_breakdown']
        
        # 检查成本分解结构
        required_cost_keys = [
            'total_cell_cost', 'total_bms_penalty', 'total_system_penalty',
            'total_system_cost', 'cell_cost_ratio', 'bms_penalty_ratio'
        ]
        
        for key in required_cost_keys:
            self.assertIn(key, cost_breakdown)
        
        # 成本应为正值
        self.assertGreaterEqual(cost_breakdown['total_system_cost'], 0)
        
        # 比例检查
        ratios_sum = (cost_breakdown['cell_cost_ratio'] + 
                     cost_breakdown['bms_penalty_ratio'] + 
                     cost_breakdown['system_penalty_ratio'])
        self.assertAlmostEqual(ratios_sum, 1.0, delta=0.01)
    
    def test_cluster_reset(self):
        """测试集群重置"""
        # 先执行一些仿真
        for _ in range(3):
            self.cluster.step(10000.0, 1.0)
        
        initial_step_count = self.cluster.step_count
        self.assertGreater(initial_step_count, 0)
        
        # 重置集群
        reset_result = self.cluster.reset(
            target_soc=55.0,
            target_temp=28.0,
            add_inter_bms_variation=True
        )
        
        # 检查重置结果
        self.assertEqual(self.cluster.step_count, 0)
        self.assertEqual(self.cluster.total_time, 0.0)
        self.assertTrue(reset_result['reset_complete'])
        self.assertEqual(len(reset_result['bms_reset_results']), 5)
    
    def test_cluster_summary(self):
        """测试集群摘要"""
        summary = self.cluster.get_cluster_summary()
        
        required_keys = [
            'cluster_id', 'num_bms', 'total_cells',
            'system_avg_soc', 'inter_bms_soc_std',
            'total_system_cost', 'bms_summaries'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        # BMS摘要数量检查
        self.assertEqual(len(summary['bms_summaries']), 5)

if __name__ == '__main__':
    unittest.main()
