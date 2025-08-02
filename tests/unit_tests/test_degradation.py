import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from battery_models.degradation_model import BatteryDegradationModel, DegradationMode, DegradationState
from config.battery_params import BatteryParams
from config.system_config import SystemConfig

class TestBatteryDegradationModel(unittest.TestCase):
    """电池劣化模型测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.battery_params = BatteryParams()
        self.system_config = SystemConfig()
        self.degradation_model = BatteryDegradationModel(
            battery_params=self.battery_params,
            system_config=self.system_config,
            degradation_mode=DegradationMode.COMBINED,
            cell_id="TestDegradationCell"
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.degradation_model.cell_id, "TestDegradationCell")
        self.assertEqual(self.degradation_model.degradation_mode, DegradationMode.COMBINED)
        self.assertIsInstance(self.degradation_model.state, DegradationState)
        self.assertEqual(self.degradation_model.state.current_capacity, self.battery_params.CELL_CAPACITY)
        self.assertEqual(self.degradation_model.state.soh_current, 100.0)
    
    def test_c_rate_calculation(self):
        """测试C率计算"""
        power = 896.0  # W (1C功率)
        voltage = 3.2   # V
        
        c_rate = self.degradation_model.calculate_c_rate(power, voltage)
        
        # 验证C率计算: P/(V*sb) = 896/(3.2*280) = 1.0
        expected_c_rate = power / (voltage * self.degradation_model.capacity_sb)
        self.assertAlmostEqual(c_rate, expected_c_rate, places=3)
        self.assertAlmostEqual(c_rate, 1.0, places=1)  # 应该接近1C
    
    def test_temperature_calculation(self):
        """测试温度计算"""
        env_temp = 25.0  # ℃
        c_rate = 1.0
        
        battery_temp = self.degradation_model.calculate_battery_temperature(env_temp, c_rate)
        
        # 验证温度计算: T = T_env + 1.421 * c_rate²
        expected_temp = env_temp + 1.421 * (c_rate ** 2)
        self.assertAlmostEqual(battery_temp, expected_temp, places=3)
        self.assertGreater(battery_temp, env_temp)  # 电池温度应高于环境温度
    
    def test_rate_coefficient_calculation(self):
        """测试倍率系数计算"""
        c_rate = 1.0
        
        b_coeff = self.degradation_model.calculate_rate_coefficient(c_rate)
        
        # 验证倍率系数计算: b = 448.96*c² - 6301.1*c + 33840
        expected_b = 448.96 * (c_rate**2) - 6301.1 * c_rate + 33840
        self.assertAlmostEqual(b_coeff, expected_b, places=3)
        self.assertGreater(b_coeff, 0)  # 系数应为正值
    
    def test_amp_hour_increment_calculation(self):
        """测试安时吞吐量增量计算"""
        power = 896.0   # W
        voltage = 3.2   # V
        delta_t = 1.0   # s
        
        amp_hour_increment = self.degradation_model.calculate_amp_hour_increment(
            power, voltage, delta_t
        )
        
        # 验证计算: ΔA' = (1/3600) * (P/V) * Δt * (2.44/sb)
        expected_increment = (1.0/3600.0) * (power/voltage) * delta_t * (2.44/self.degradation_model.capacity_sb)
        self.assertAlmostEqual(amp_hour_increment, expected_increment, places=6)
        self.assertGreater(amp_hour_increment, 0)
    
    def test_capacity_degradation_calculation(self):
        """测试容量衰减计算"""
        power = 896.0   # W (1C)
        voltage = 3.2   # V
        delta_t = 3600.0  # s (1小时)
        env_temp = 25.0 # ℃
        
        # 设置一些初始安时吞吐量以避免零幂次
        self.degradation_model.state.amp_hour_throughput = 100.0
        
        degradation_result = self.degradation_model.calculate_capacity_degradation(
            power, voltage, delta_t, env_temp
        )
        
        # 验证结果结构
        required_keys = [
            'c_rate', 'battery_temperature', 'amp_hour_increment',
            'capacity_degradation', 'degradation_cost'
        ]
        for key in required_keys:
            self.assertIn(key, degradation_result)
        
        # 验证数值合理性
        self.assertGreater(degradation_result['capacity_degradation'], 0)
        self.assertGreater(degradation_result['degradation_cost'], 0)
        self.assertAlmostEqual(degradation_result['c_rate'], 1.0, places=1)
    
    def test_degradation_state_update(self):
        """测试劣化状态更新"""
        # 模拟劣化计算结果
        degradation_result = {
            'c_rate': 1.0,
            'battery_temperature': 26.421,
            'amp_hour_increment': 0.1,
            'capacity_degradation': 0.01,  # 假设衰减0.01Ah
            'degradation_cost': 0.01
        }
        
        initial_capacity = self.degradation_model.state.current_capacity
        initial_soh = self.degradation_model.state.soh_current
        
        update_info = self.degradation_model.update_degradation_state(degradation_result)
        
        # 验证容量更新
        self.assertLess(self.degradation_model.state.current_capacity, initial_capacity)
        self.assertLess(self.degradation_model.state.soh_current, initial_soh)
        
        # 验证返回信息
        self.assertIn('capacity_change', update_info)
        self.assertIn('soh_change', update_info)
        self.assertLess(update_info['capacity_change'], 0)  # 容量应该减少
        self.assertLess(update_info['soh_change'], 0)       # SOH应该减少
    
    def test_delta_soh_for_drl(self):
        """测试为DRL提供的ΔSOH"""
        # 运行一些仿真步骤以建立趋势
        for i in range(20):
            power = 896.0 * (1 + i * 0.1)  # 逐渐增加功率
            self.degradation_model.step(power, 3.2, 180.0, 25.0)  # 3分钟步长
        
        delta_soh = self.degradation_model.get_delta_soh_for_drl()
        
        # ΔSOH应该是负值（SOH下降）
        self.assertLessEqual(delta_soh, 0)
        self.assertIsInstance(delta_soh, float)
    
    def test_aging_statistics_for_drl(self):
        """测试为DRL提供的老化统计"""
        aging_stats = self.degradation_model.get_aging_statistics_for_drl()
        
        # 验证统计信息完整性
        required_keys = [
            'current_soh', 'soh_change_rate', 'capacity_retention',
            'aging_acceleration_factor', 'cumulative_degradation_cost',
            'amp_hour_throughput', 'equivalent_cycles', 'remaining_cycles_estimate'
        ]
        
        for key in required_keys:
            self.assertIn(key, aging_stats)
        
        # 验证数值范围
        self.assertGreaterEqual(aging_stats['current_soh'], 0)
        self.assertLessEqual(aging_stats['current_soh'], 100)
        self.assertGreaterEqual(aging_stats['capacity_retention'], 0)
        self.assertGreaterEqual(aging_stats['equivalent_cycles'], 0)
    
    def test_degradation_simulation_step(self):
        """测试劣化仿真步"""
        power = 1344.0  # W (1.5C)
        voltage = 3.2   # V
        delta_t = 60.0  # s (1分钟)
        env_temp = 30.0 # ℃
        
        initial_soh = self.degradation_model.state.soh_current
        
        degradation_record = self.degradation_model.step(
            power, voltage, delta_t, env_temp
        )
        
        # 验证记录完整性
        self.assertIn('soh_current', degradation_record)
        self.assertIn('capacity_degradation', degradation_record)
        self.assertIn('degradation_cost', degradation_record)
        self.assertIn('c_rate', degradation_record)
        
        # 验证劣化效果
        self.assertLessEqual(self.degradation_model.state.soh_current, initial_soh)
        self.assertEqual(self.degradation_model.time_step_count, 1)
    
    def test_degradation_modes(self):
        """测试不同劣化模式"""
        modes = [DegradationMode.CALENDAR, DegradationMode.CYCLE, DegradationMode.COMBINED]
        
        for mode in modes:
            degradation_model = BatteryDegradationModel(
                battery_params=self.battery_params,
                system_config=self.system_config,
                degradation_mode=mode,
                cell_id=f"Test_{mode.value}"
            )
            
            # 运行仿真步骤
            degradation_record = degradation_model.step(896.0, 3.2, 60.0, 25.0)
            
            self.assertIn('soh_current', degradation_record)
            self.assertEqual(degradation_model.degradation_mode, mode)
    
    def test_extreme_operating_conditions(self):
        """测试极端操作条件"""
        # 极高C率
        extreme_power = 8960.0  # W (10C)
        voltage = 3.0   # V (较低电压)
        delta_t = 10.0  # s
        high_temp = 45.0  # ℃
        
        degradation_result = self.degradation_model.calculate_capacity_degradation(
            extreme_power, voltage, delta_t, high_temp
        )
        
        # 极端条件下应有显著劣化
        self.assertGreater(degradation_result['capacity_degradation'], 0.001)
        self.assertGreater(degradation_result['c_rate'], 5.0)
        self.assertGreater(degradation_result['degradation_cost'], 0.01)
    
    def test_degradation_model_reset(self):
        """测试劣化模型重置"""
        # 运行一些仿真步骤
        for i in range(10):
            self.degradation_model.step(896.0, 3.2, 60.0, 25.0)
        
        # 记录重置前状态
        old_soh = self.degradation_model.state.soh_current
        old_capacity = self.degradation_model.state.current_capacity
        
        # 重置为新电池
        initial_state = self.degradation_model.reset(
            reset_to_new=True, 
            reset_history=True
        )
        
        # 验证重置效果
        self.assertEqual(self.degradation_model.state.soh_current, 100.0)
        self.assertEqual(self.degradation_model.state.current_capacity, 
                        self.degradation_model.state.initial_capacity)
        self.assertEqual(self.degradation_model.time_step_count, 0)
        self.assertEqual(len(self.degradation_model.degradation_history), 0)
        
        # 重置为指定SOH
        target_soh = 85.0
        self.degradation_model.reset(reset_to_new=False, initial_soh=target_soh)
        self.assertAlmostEqual(self.degradation_model.state.soh_current, target_soh, places=1)
    
    def test_degradation_diagnostics(self):
        """测试劣化诊断"""
        # 运行长期仿真
        for i in range(50):
            power = 896.0 + i * 10  # 逐渐增加功率负荷
            self.degradation_model.step(power, 3.2, 120.0, 25.0)
        
        diagnostics = self.degradation_model.get_diagnostics()
        
        # 验证诊断信息
        self.assertIn('cell_id', diagnostics)
        self.assertIn('current_soh', diagnostics)
        self.assertIn('capacity_retention', diagnostics)
        self.assertIn('equivalent_cycles', diagnostics)
        self.assertIn('degradation_health_status', diagnostics)
        
        # 验证统计数据
        self.assertEqual(diagnostics['simulation_steps'], 50)
        self.assertGreater(diagnostics['total_degradation_cost'], 0)
        self.assertLess(diagnostics['current_soh'], 100.0)  # 应该有SOH损失
    
    def test_economic_model_integration(self):
        """测试经济模型集成"""
        # 高功率操作
        high_power = 2688.0  # W (3C)
        voltage = 3.2
        operation_time = 1800.0  # s (30分钟)
        
        initial_cost = self.degradation_model.state.cumulative_cost
        
        degradation_record = self.degradation_model.step(
            high_power, voltage, operation_time, 35.0
        )
        
        # 验证成本增加
        cost_increase = self.degradation_model.state.cumulative_cost - initial_cost
        self.assertGreater(cost_increase, 0)
        
        # 验证成本计算合理性
        expected_range = (0.001, 10.0)  # 元 (合理的单步成本范围)
        self.assertGreater(degradation_record['degradation_cost'], expected_range[0])
        self.assertLess(degradation_record['degradation_cost'], expected_range[1])
    
    def test_soh_trend_calculation(self):
        """测试SOH趋势计算"""
        # 持续运行以建立明显趋势
        for i in range(100):
            # 模拟逐渐加重的使用模式
            power = 896.0 * (1 + i * 0.02)
            self.degradation_model.step(power, 3.2, 60.0, 25.0)
        
        # 获取SOH趋势
        soh_trend = self.degradation_model.state.soh_trend
        
        # SOH趋势应该为负值（持续下降）
        self.assertLess(soh_trend, 0)
        self.assertIsInstance(soh_trend, float)
        
        # 趋势值应该在合理范围内 (%/hour)
        self.assertGreater(soh_trend, -10.0)  # 不应过于陡峭
        self.assertLess(soh_trend, 0.0)       # 应该是下降趋势

if __name__ == '__main__':
    unittest.main()
