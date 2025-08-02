import unittest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from battery_models.thermal_model import ThermalModel, CoolingMode, ThermalState
from config.battery_params import BatteryParams
from config.system_config import SystemConfig

class TestThermalModel(unittest.TestCase):
    """热模型测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.battery_params = BatteryParams()
        self.system_config = SystemConfig()
        self.thermal_model = ThermalModel(
            battery_params=self.battery_params,
            system_config=self.system_config,
            cooling_mode=CoolingMode.FORCED_AIR,
            cell_id="TestThermalCell"
        )
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.thermal_model.cell_id, "TestThermalCell")
        self.assertEqual(self.thermal_model.cooling_mode, CoolingMode.FORCED_AIR)
        self.assertIsInstance(self.thermal_model.state, ThermalState)
        self.assertEqual(self.thermal_model.state.core_temperature, self.battery_params.NOMINAL_TEMP)
    
    def test_heat_generation_calculation(self):
        """测试产热计算"""
        current = 100.0  # A
        voltage = 3.3    # V
        soc = 50.0      # %
        internal_resistance = 0.001  # Ω
        
        heat_sources = self.thermal_model.calculate_heat_generation(
            current, voltage, soc, internal_resistance
        )
        
        # 验证热源
        self.assertIn('joule', heat_sources)
        self.assertIn('polarization', heat_sources)
        self.assertIn('reaction', heat_sources)
        self.assertIn('external', heat_sources)
        self.assertIn('total', heat_sources)
        
        # 验证焦耳热计算
        expected_joule_heat = current**2 * internal_resistance
        self.assertAlmostEqual(heat_sources['joule'], expected_joule_heat, places=3)
        
        # 验证总热量为正值
        self.assertGreater(heat_sources['total'], 0)
    
    def test_heat_dissipation_calculation(self):
        """测试散热计算"""
        surface_temp = 40.0  # ℃
        ambient_temp = 25.0  # ℃
        
        heat_dissipation = self.thermal_model.calculate_heat_dissipation(
            surface_temp, ambient_temp
        )
        
        # 验证散热类型
        self.assertIn('convection', heat_dissipation)
        self.assertIn('radiation', heat_dissipation)
        self.assertIn('active_cooling', heat_dissipation)
        self.assertIn('total', heat_dissipation)
        
        # 验证散热量为正值
        self.assertGreater(heat_dissipation['total'], 0)
        self.assertGreater(heat_dissipation['convection'], 0)
    
    def test_temperature_update(self):
        """测试温度更新"""
        initial_temp = self.thermal_model.state.core_temperature
        heat_generation = 50.0  # W
        delta_t = 1.0  # s
        
        temp_info = self.thermal_model.update_temperature(
            heat_generation, delta_t
        )
        
        # 验证返回信息
        self.assertIn('core_temp_change', temp_info)
        self.assertIn('surface_temp_change', temp_info)
        self.assertIn('heat_generation', temp_info)
        self.assertIn('heat_dissipation', temp_info)
        
        # 验证温度变化
        self.assertNotEqual(self.thermal_model.state.core_temperature, initial_temp)
        self.assertEqual(temp_info['heat_generation'], heat_generation)
    
    def test_thermal_constraints_calculation(self):
        """测试热约束计算"""
        # 模拟高温状态
        self.thermal_model.state.core_temperature = 55.0  # ℃ (接近警告温度)
        
        base_current_limits = (280.0, 840.0)  # A (1C充电, 3C放电)
        base_power_limits = (89600.0, 268800.0)  # W
        
        constraints = self.thermal_model.calculate_thermal_constraints(
            base_current_limits, base_power_limits
        )
        
        # 验证约束降额
        self.assertLess(constraints.max_charge_current, base_current_limits[0])
        self.assertLess(constraints.max_discharge_current, base_current_limits[1])
        self.assertLess(constraints.max_charge_power, base_power_limits[0])
        self.assertLess(constraints.max_discharge_power, base_power_limits[1])
        
        # 验证约束值为正
        self.assertGreater(constraints.max_charge_current, 0)
        self.assertGreater(constraints.max_discharge_current, 0)
    
    def test_constraint_matrix_for_drl(self):
        """测试DRL约束矩阵生成"""
        constraint_matrix = self.thermal_model.get_constraint_matrix_for_drl()
        
        # 验证矩阵形状
        self.assertEqual(constraint_matrix.shape[0], 6)  # 6种约束类型
        self.assertEqual(constraint_matrix.shape[1], 1)  # 1个电池单体
        
        # 验证约束值
        self.assertTrue(np.all(constraint_matrix >= 0))  # 所有约束值非负
    
    def test_temperature_compensation_data(self):
        """测试温度补偿数据"""
        comp_data = self.thermal_model.get_temperature_compensation_data()
        
        # 验证数据完整性
        required_keys = [
            'core_temperature', 'surface_temperature', 'ambient_temperature',
            'thermal_gradient', 'temperature_derating_factor', 'cooling_efficiency',
            'thermal_time_constant', 'temperature_prediction'
        ]
        
        for key in required_keys:
            self.assertIn(key, comp_data)
        
        # 验证数据范围
        self.assertGreaterEqual(comp_data['temperature_derating_factor'], 0.0)
        self.assertLessEqual(comp_data['temperature_derating_factor'], 1.0)
    
    def test_thermal_simulation_step(self):
        """测试热仿真步"""
        current = 150.0    # A
        voltage = 3.25     # V
        soc = 60.0        # %
        internal_resistance = 0.0012  # Ω
        delta_t = 1.0     # s
        
        thermal_record = self.thermal_model.step(
            current, voltage, soc, internal_resistance, delta_t
        )
        
        # 验证记录完整性
        self.assertIn('core_temperature', thermal_record)
        self.assertIn('surface_temperature', thermal_record)
        self.assertIn('heat_generation', thermal_record)
        self.assertIn('heat_dissipation', thermal_record)
        self.assertIn('temperature_warning', thermal_record)
        
        # 验证时间更新
        self.assertEqual(self.thermal_model.time_step_count, 1)
        self.assertEqual(self.thermal_model.total_time, delta_t)
    
    def test_cooling_modes(self):
        """测试不同冷却模式"""
        cooling_modes = [CoolingMode.NATURAL, CoolingMode.FORCED_AIR, 
                        CoolingMode.LIQUID, CoolingMode.HYBRID]
        
        for mode in cooling_modes:
            thermal_model = ThermalModel(
                battery_params=self.battery_params,
                system_config=self.system_config,
                cooling_mode=mode,
                cell_id=f"Test_{mode.value}"
            )
            
            # 测试散热能力差异
            heat_dissipation = thermal_model.calculate_heat_dissipation(50.0, 25.0)
            
            self.assertGreater(heat_dissipation['total'], 0)
            
            # 液冷应该有最高的散热能力
            if mode == CoolingMode.LIQUID:
                self.assertGreater(thermal_model.convection_coefficient, 100)
    
    def test_safety_monitoring(self):
        """测试安全监控"""
        # 测试正常状态
        self.assertFalse(self.thermal_model.state.temperature_warning)
        self.assertFalse(self.thermal_model.state.temperature_alarm)
        self.assertEqual(self.thermal_model.state.thermal_runaway_risk, 0.0)
        
        # 模拟高温状态
        self.thermal_model.state.core_temperature = 85.0  # ℃
        self.thermal_model._update_safety_status()
        
        # 验证安全状态更新
        self.assertTrue(self.thermal_model.state.temperature_warning)
        self.assertTrue(self.thermal_model.state.temperature_alarm)
        self.assertGreater(self.thermal_model.state.thermal_runaway_risk, 0.0)
    
    def test_thermal_model_reset(self):
        """测试热模型重置"""
        # 运行几步仿真
        for i in range(5):
            self.thermal_model.step(100.0, 3.3, 50.0, 0.001, 1.0)
        
        # 记录重置前状态
        old_step_count = self.thermal_model.time_step_count
        old_history_length = len(self.thermal_model.temperature_history)
        
        # 重置
        initial_state = self.thermal_model.reset(
            initial_temp=30.0, 
            initial_ambient=20.0, 
            reset_history=True
        )
        
        # 验证重置效果
        self.assertEqual(self.thermal_model.time_step_count, 0)
        self.assertEqual(self.thermal_model.total_time, 0.0)
        self.assertEqual(len(self.thermal_model.temperature_history), 0)
        self.assertEqual(self.thermal_model.state.core_temperature, 30.0)
        self.assertEqual(self.thermal_model.state.ambient_temperature, 20.0)
        
        # 验证返回的初始状态
        self.assertEqual(initial_state['core_temperature'], 30.0)
        self.assertEqual(initial_state['ambient_temperature'], 20.0)
    
    def test_thermal_diagnostics(self):
        """测试热诊断"""
        # 运行一些仿真步骤
        for i in range(10):
            current = 100 + i * 10  # 逐渐增加电流
            self.thermal_model.step(current, 3.3, 50.0, 0.001, 1.0)
        
        diagnostics = self.thermal_model.get_diagnostics()
        
        # 验证诊断信息
        self.assertIn('cell_id', diagnostics)
        self.assertIn('cooling_mode', diagnostics)
        self.assertIn('core_temp_range', diagnostics)
        self.assertIn('avg_core_temperature', diagnostics)
        self.assertIn('thermal_health_status', diagnostics)
        
        # 验证统计数据
        self.assertEqual(diagnostics['simulation_steps'], 10)
        self.assertGreater(diagnostics['total_heat_generated'], 0)
    
    def test_extreme_conditions(self):
        """测试极端条件"""
        # 测试极高电流
        extreme_current = 1000.0  # A
        heat_sources = self.thermal_model.calculate_heat_generation(
            extreme_current, 3.0, 50.0, 0.002
        )
        self.assertGreater(heat_sources['total'], 1000)  # 应产生大量热
        
        # 测试极低环境温度
        temp_info = self.thermal_model.update_temperature(
            100.0, 1.0, 100.0, -10.0
        )
        self.assertGreaterEqual(self.thermal_model.state.core_temperature, -15.0)
        
        # 测试约束在极端温度下的表现
        self.thermal_model.state.core_temperature = 70.0  # 极高温度
        constraints = self.thermal_model.calculate_thermal_constraints(
            (280.0, 840.0), (89600.0, 268800.0)
        )
        self.assertLess(constraints.max_charge_current, 50.0)  # 应严重限制电流

if __name__ == '__main__':
    unittest.main()
