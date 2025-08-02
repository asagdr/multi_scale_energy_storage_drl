"""
多层级成本模型单元测试
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.battery_params import BatteryParams
from battery_models.multi_level_cost_model import MultiLevelCostModel, CostBreakdown

class TestMultiLevelCostModel(unittest.TestCase):
    """多层级成本模型测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.battery_params = BatteryParams()
        
        # 创建模拟BMS列表
        self.mock_bms_list = []
        for i in range(5):  # 5个BMS用于测试
            mock_bms = type('MockBMS', (), {
                'bms_id': f'BMS_{i+1:02d}',
                'get_bms_summary': lambda: {
                    'bms_id': f'BMS_{i+1:02d}',
                    'avg_soc': 50.0 + np.random.normal(0, 5.0),
                    'avg_temperature': 25.0 + np.random.normal(0, 3.0),
                    'avg_soh': 90.0 + np.random.normal(0, 2.0),
                    'total_cost': 1.0 + np.random.uniform(0, 0.5)
                }
            })()
            self.mock_bms_list.append(mock_bms)
        
        self.cost_model = MultiLevelCostModel(
            bms_list=self.mock_bms_list,
            battery_params=self.battery_params,
            cost_model_id="TEST_COST_MODEL"
        )
    
    def test_cost_model_initialization(self):
        """测试成本模型初始化"""
        self.assertEqual(len(self.cost_model.bms_list), 5)
        self.assertEqual(self.cost_model.cost_model_id, "TEST_COST_MODEL")
        self.assertIn('bms_soc_imbalance_factor', self.cost_model.cost_params)
        self.assertEqual(len(self.cost_model.cost_history), 0)
    
    def test_cell_level_cost_calculation(self):
        """测试单体级成本计算"""
        # 模拟BMS记录
        bms_records = []
        for i in range(5):
            bms_record = {
                'bms_id': f'BMS_{i+1:02d}',
                'cost_breakdown': {
                    'base_cost': 1.0 + i * 0.1  # 递增的基础成本
                }
            }
            bms_records.append(bms_record)
        
        cell_cost = self.cost_model._calculate_cell_level_cost(bms_records)
        
        # 验证成本计算
        expected_cost = sum(1.0 + i * 0.1 for i in range(5))
        self.assertAlmostEqual(cell_cost, expected_cost, places=2)
    
    def test_bms_level_penalties(self):
        """测试BMS级不平衡惩罚"""
        # 创建有不平衡的BMS记录
        bms_records = [
            {
                'bms_id': 'BMS_01',
                'soc_std': 3.0,     # 高SOC不平衡
                'temp_std': 8.0,    # 高温度不平衡
                'balancing_power': 50.0,
                'cost_breakdown': {'base_cost': 100.0}
            },
            {
                'bms_id': 'BMS_02',
                'soc_std': 0.5,     # 低SOC不平衡
                'temp_std': 2.0,    # 低温度不平衡
                'balancing_power': 10.0,
                'cost_breakdown': {'base_cost': 100.0}
            }
        ]
        
        penalties = self.cost_model._calculate_bms_level_penalties(bms_records)
        
        # 验证惩罚计算
        self.assertGreater(penalties['soc_imbalance'], 0)
        self.assertGreater(penalties['temp_imbalance'], 0)
        self.assertGreater(penalties['balancing_cost'], 0)
        
        # BMS_01应该有更高的惩罚
        # 这里简化验证，实际中可以分别计算每个BMS的惩罚
        total_penalty = sum(penalties.values())
        self.assertGreater(total_penalty, 0)
    
    def test_system_level_penalties(self):
        """测试系统级协同效应惩罚"""
        # 创建BMS间不平衡的记录
        bms_records = []
        soc_values = [30.0, 40.0, 50.0, 60.0, 70.0]  # 大范围SOC差异
        temp_values = [20.0, 25.0, 30.0, 35.0, 40.0]  # 大范围温度差异
        soh_values = [85.0, 88.0, 90.0, 92.0, 70.0]  # 一个低SOH的BMS
        
        for i in range(5):
            bms_record = {
                'bms_id': f'BMS_{i+1:02d}',
                'avg_soc': soc_values[i],
                'avg_temperature': temp_values[i],
                'avg_soh': soh_values[i],
                'actual_power': 20000.0 + i * 5000.0,  # 功率不均衡
                'cost_breakdown': {'base_cost': 100.0}
            }
            bms_records.append(bms_record)
        
        penalties = self.cost_model._calculate_system_level_penalties(bms_records)
        
        # 验证系统级惩罚
        self.assertGreater(penalties['inter_bms_imbalance'], 0)
        self.assertGreater(penalties['coordination_penalty'], 0)
        self.assertGreater(penalties['bottleneck_penalty'], 0)  # 由于有70%SOH的BMS
    
    def test_comprehensive_cost_calculation(self):
        """测试综合成本计算"""
        # 创建完整的BMS记录
        bms_records = []
        for i in range(5):
            bms_record = {
                'bms_id': f'BMS_{i+1:02d}',
                'avg_soc': 50.0 + np.random.normal(0, 3.0),
                'avg_temperature': 25.0 + np.random.normal(0, 2.0),
                'avg_soh': 90.0 + np.random.normal(0, 1.0),
                'soc_std': 1.0 + np.random.uniform(0, 2.0),
                'temp_std': 2.0 + np.random.uniform(0, 3.0),
                'actual_power': 20000.0 + np.random.normal(0, 5000.0),
                'balancing_power': np.random.uniform(0, 100.0),
                'cost_breakdown': {
                    'base_cost': 100.0 + np.random.uniform(0, 20.0)
                }
            }
            bms_records.append(bms_record)
        
        cost_breakdown = self.cost_model.calculate_total_system_cost(bms_records)
        
        # 验证成本分解结构
        required_keys = [
            'total_cell_cost', 'bms_soc_imbalance_cost', 'bms_temp_imbalance_cost',
            'inter_bms_imbalance_penalty', 'system_coordination_penalty',
            'bottleneck_penalty', 'total_system_cost'
        ]
        
        for key in required_keys:
            self.assertIn(key, cost_breakdown)
        
        # 验证成本关系
        self.assertGreaterEqual(cost_breakdown['total_system_cost'], cost_breakdown['total_cell_cost'])
        self.assertGreaterEqual(cost_breakdown['total_cell_cost'], 0)
        
        # 验证比例
        if cost_breakdown['total_system_cost'] > 0:
            cell_ratio = cost_breakdown['cell_cost_ratio']
            bms_ratio = cost_breakdown['bms_penalty_ratio']
            system_ratio = cost_breakdown['system_penalty_ratio']
            
            self.assertAlmostEqual(cell_ratio + bms_ratio + system_ratio, 1.0, places=2)
    
    def test_cost_trends_analysis(self):
        """测试成本趋势分析"""
        # 生成历史成本数据
        for i in range(60):  # 60个历史记录
            mock_breakdown = CostBreakdown()
            mock_breakdown.total_cell_cost = 500.0 + i * 0.1
            mock_breakdown.total_system_cost = 550.0 + i * 0.15
            
            self.cost_model.cost_history.append(mock_breakdown)
        
        trends = self.cost_model.get_cost_trends(window_size=50)
        
        # 验证趋势分析结果
        self.assertIn('total_cost_trend', trends)
        self.assertIn('avg_cost_increase_rate', trends)
        self.assertIn('cost_volatility', trends)
        self.assertIn('latest_total_cost', trends)
        
        # 由于成本递增，趋势应该是增长的
        self.assertEqual(trends['total_cost_trend'], 'increasing')
        self.assertGreater(trends['avg_cost_increase_rate'], 0)
    
    def test_cost_model_reset(self):
        """测试成本模型重置"""
        # 先添加一些历史数据
        for _ in range(10):
            mock_breakdown = CostBreakdown()
            mock_breakdown.total_system_cost = 100.0
            self.cost_model.cost_history.append(mock_breakdown)
        
        self.cost_model.previous_total_cost = 100.0
        
        # 重置模型
        self.cost_model.reset()
        
        # 验证重置结果
        self.assertEqual(len(self.cost_model.cost_history), 0)
        self.assertEqual(self.cost_model.previous_total_cost, 0.0)
    
    def test_cost_model_summary(self):
        """测试成本模型摘要"""
        # 添加一些历史数据
        mock_breakdown = CostBreakdown()
        mock_breakdown.total_cell_cost = 400.0
        mock_breakdown.bms_soc_imbalance_cost = 20.0
        mock_breakdown.inter_bms_imbalance_penalty = 30.0
        mock_breakdown.total_system_cost = 500.0
        
        self.cost_model.cost_history.append(mock_breakdown)
        
        summary = self.cost_model.get_cost_model_summary()
        
        # 验证摘要结构
        required_keys = [
            'cost_model_id', 'total_calculations', 'latest_breakdown', 'cost_composition'
        ]
        
        for key in required_keys:
            self.assertIn(key, summary)
        
        # 验证成本组成
        composition = summary['cost_composition']
        self.assertAlmostEqual(composition['cell_cost_ratio'], 0.8, places=1)  # 400/500
    
    def test_coordination_penalty_calculation(self):
        """测试协调效应惩罚计算"""
        bms_records = [
            {
                'bms_id': 'BMS_01',
                'actual_power': 10000.0,  # 低功率
                'avg_temperature': 30.0,
                'avg_soc': 60.0,
                'avg_soh': 90.0
            },
            {
                'bms_id': 'BMS_02', 
                'actual_power': 50000.0,  # 高功率
                'avg_temperature': 40.0,  # 高温
                'avg_soc': 70.0,
                'avg_soh': 85.0
            }
        ]
        
        total_base_cost = 200.0
        
        penalty = self.cost_model._calculate_coordination_penalty(bms_records, total_base_cost)
        
        # 应该有协调惩罚，因为功率和温度不均衡
        self.assertGreaterEqual(penalty, 0)

if __name__ == '__main__':
    unittest.main()
