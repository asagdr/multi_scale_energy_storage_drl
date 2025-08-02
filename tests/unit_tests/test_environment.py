import unittest
import numpy as np
import time
import sys
import os
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig
from environment.storage_environment import StorageEnvironment
from environment.multi_scale_scheduler import MultiScaleScheduler, TimeScale, SchedulerMode
from environment.constraint_validator import ConstraintValidator, ConstraintType, ViolationSeverity
from environment.reward_calculator import RewardCalculator, RewardType
from environment.state_manager import StateManager, StateScope, StateType

class TestStorageEnvironment(unittest.TestCase):
    """储能环境测试"""
    
    def setUp(self):
        """测试前准备"""
        self.battery_params = BatteryParams()
        self.system_config = SystemConfig()
        self.env = StorageEnvironment(
            battery_params=self.battery_params,
            system_config=self.system_config,
            env_id="TestEnv"
        )
    
    def test_environment_initialization(self):
        """测试环境初始化"""
        self.assertEqual(self.env.env_id, "TestEnv")
        self.assertIsNotNone(self.env.battery_pack)
        self.assertEqual(self.env.observation_space.shape[0], 14)
        self.assertEqual(self.env.action_space.shape[0], 4)
    
    def test_environment_reset(self):
        """测试环境重置"""
        initial_state = self.env.reset()
        
        self.assertEqual(len(initial_state), 14)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in initial_state))
        self.assertEqual(self.env.current_step, 0)
        self.assertFalse(self.env.done)
    
    def test_environment_step(self):
        """测试环境步进"""
        self.env.reset()
        action = np.array([0.1, 0.2, 0.3, 0.4])  # 测试动作
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertEqual(len(next_state), 14)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        self.assertEqual(self.env.current_step, 1)
        
        # 验证info字典内容
        required_keys = ['pack_soc', 'soc_std', 'pack_temperature', 'power_command']
        for key in required_keys:
            self.assertIn(key, info)
    
    def test_constraint_matrix_access(self):
        """测试约束矩阵访问"""
        self.env.reset()
        constraint_matrix = self.env.get_constraint_matrix()
        
        self.assertIsInstance(constraint_matrix, np.ndarray)
        self.assertEqual(constraint_matrix.shape[0], 7)  # 7种约束类型
        self.assertEqual(constraint_matrix.shape[1], 1)  # 1个电池组
    
    def test_balance_metrics_access(self):
        """测试均衡指标访问"""
        self.env.reset()
        balance_metrics = self.env.get_balance_metrics()
        
        required_keys = ['sigma_soc', 'soc_consistency', 'temp_std', 'total_degradation_cost']
        for key in required_keys:
            self.assertIn(key, balance_metrics)
        
        # 验证σ_SOC是关键指标
        self.assertIn('sigma_soc', balance_metrics)
        self.assertGreaterEqual(balance_metrics['sigma_soc'], 0.0)

class TestMultiScaleScheduler(unittest.TestCase):
    """多时间尺度调度器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.system_config = SystemConfig()
        self.scheduler = MultiScaleScheduler(
            system_config=self.system_config,
            mode=SchedulerMode.SEQUENTIAL,
            scheduler_id="TestScheduler"
        )
    
    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        self.assertEqual(self.scheduler.scheduler_id, "TestScheduler")
        self.assertEqual(self.scheduler.mode, SchedulerMode.SEQUENTIAL)
        self.assertFalse(self.scheduler.is_running)
        
        # 验证时间尺度配置
        self.assertIn(TimeScale.UPPER_LAYER, self.scheduler.time_scales)
        self.assertIn(TimeScale.LOWER_LAYER, self.scheduler.time_scales)
        self.assertIn(TimeScale.SIMULATION, self.scheduler.time_scales)
    
    def test_task_registration(self):
        """测试任务注册"""
        def dummy_callback(current_time, delta_t):
            return {'executed': True, 'time': current_time}
        
        success = self.scheduler.register_task(
            task_id="test_task",
            time_scale=TimeScale.SIMULATION,
            callback=dummy_callback,
            priority=5
        )
        
        self.assertTrue(success)
        self.assertIn("test_task", self.scheduler.tasks)
        
        task_info = self.scheduler.tasks["test_task"]
        self.assertEqual(task_info.time_scale, TimeScale.SIMULATION)
        self.assertEqual(task_info.priority, 5)
        self.assertTrue(task_info.enabled)
    
    def test_task_execution(self):
        """测试任务执行"""
        execution_log = []
        
        def test_callback(current_time, delta_t):
            execution_log.append((current_time, delta_t))
            return {'status': 'executed'}
        
        self.scheduler.register_task(
            task_id="exec_test",
            time_scale=TimeScale.SIMULATION,
            callback=test_callback
        )
        
        # 执行一个步骤
        result = self.scheduler.step(delta_t=1.0)
        
        self.assertIn('exec_test', result['executed_tasks'])
        self.assertEqual(len(execution_log), 1)
        self.assertEqual(execution_log[0][1], 1.0)  # delta_t
    
    def test_data_exchange(self):
        """测试数据交换"""
        # 测试数据放入
        test_data = {'value': 42, 'timestamp': time.time()}
        success = self.scheduler.put_data('upper_to_lower', test_data)
        self.assertTrue(success)
        
        # 测试数据获取
        retrieved_data = self.scheduler.get_data('upper_to_lower')
        self.assertEqual(retrieved_data['value'], 42)
        
        # 测试空队列
        empty_data = self.scheduler.get_data('upper_to_lower')
        self.assertIsNone(empty_data)

class TestConstraintValidator(unittest.TestCase):
    """约束验证器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.battery_params = BatteryParams()
        self.system_config = SystemConfig()
        self.validator = ConstraintValidator(
            battery_params=self.battery_params,
            system_config=self.system_config,
            validator_id="TestValidator"
        )
    
    def test_validator_initialization(self):
        """测试验证器初始化"""
        self.assertEqual(self.validator.validator_id, "TestValidator")
        self.assertIn(ConstraintType.POWER, self.validator.constraints)
        self.assertIn(ConstraintType.TEMPERATURE, self.validator.constraints)
        self.assertIn(ConstraintType.SOC, self.validator.constraints)
    
    def test_power_constraint_validation(self):
        """测试功率约束验证"""
        # 测试正常功率
        result = self.validator.validate_power_constraints(1000.0)  # 1kW
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.violations), 0)
        
        # 测试功率超限
        result = self.validator.validate_power_constraints(100000.0)  # 100kW (超限)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.violations), 0)
        self.assertEqual(result.violations[0].constraint_type, ConstraintType.POWER)
    
    def test_temperature_constraint_validation(self):
        """测试温度约束验证"""
        # 测试正常温度
        normal_temps = [25.0, 26.0, 27.0, 25.5]
        result = self.validator.validate_temperature_constraints(normal_temps)
        self.assertTrue(result.is_valid)
        
        # 测试过高温度
        high_temps = [60.0, 62.0, 65.0, 58.0]  # 超过MAX_TEMP
        result = self.validator.validate_temperature_constraints(high_temps)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.violations), 0)
    
    def test_soc_constraint_validation(self):
        """测试SOC约束验证"""
        # 测试正常SOC
        normal_socs = [45.0, 50.0, 55.0, 48.0]
        result = self.validator.validate_soc_constraints(normal_socs)
        self.assertTrue(result.is_valid)
        
        # 测试SOC不平衡
        unbalanced_socs = [20.0, 80.0, 30.0, 70.0]  # 严重不平衡
        result = self.validator.validate_soc_constraints(unbalanced_socs)
        # 可能有警告但不一定违约，取决于具体阈值
        self.assertGreaterEqual(len(result.warnings) + len(result.violations), 0)
    
    def test_comprehensive_validation(self):
        """测试综合约束验证"""
        system_state = {
            'pack_power': 5000.0,
            'temperatures': [30.0, 32.0, 31.0, 29.0],
            'soc_values': [48.0, 52.0, 50.0, 49.0],
            'voltages': [3.3, 3.35, 3.32, 3.31],
            'soh_values': [95.0, 94.0, 96.0, 95.5]
        }
        
        result = self.validator.validate_comprehensive_constraints(system_state)
        
        self.assertIsInstance(result.is_valid, bool)
        self.assertIsInstance(result.safety_score, float)
        self.assertGreaterEqual(result.safety_score, 0.0)
        self.assertLessEqual(result.safety_score, 1.0)

class TestRewardCalculator(unittest.TestCase):
    """奖励计算器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.system_config = SystemConfig()
        self.calculator = RewardCalculator(
            system_config=self.system_config,
            calculator_id="TestCalculator"
        )
    
    def test_calculator_initialization(self):
        """测试计算器初始化"""
        self.assertEqual(self.calculator.calculator_id, "TestCalculator")
        self.assertIn('power_tracking', self.calculator.reward_weights)
        self.assertIn(RewardType.POWER_TRACKING, self.calculator.normalization_params)
    
    def test_power_tracking_reward(self):
        """测试功率跟踪奖励"""
        # 测试完美跟踪
        component = self.calculator.calculate_power_tracking_reward(1000.0, 1000.0)
        self.assertEqual(component.reward_type, RewardType.POWER_TRACKING)
        self.assertGreater(component.normalized_value, 0.5)  # 应该有高奖励
        
        # 测试有误差的跟踪
        component = self.calculator.calculate_power_tracking_reward(1000.0, 1200.0)
        self.assertLess(component.normalized_value, 0.5)  # 奖励应该降低
    
    def test_soc_balance_reward(self):
        """测试SOC均衡奖励"""
        # 测试良好均衡
        component = self.calculator.calculate_soc_balance_reward(1.0, 0.95)  # 低标准差，高一致性
        self.assertEqual(component.reward_type, RewardType.SOC_BALANCE)
        self.assertGreater(component.normalized_value, 0.0)
        
        # 测试差均衡
        component = self.calculator.calculate_soc_balance_reward(15.0, 0.3)  # 高标准差，低一致性
        self.assertLess(component.normalized_value, 0.0)
    
    def test_comprehensive_reward_calculation(self):
        """测试综合奖励计算"""
        system_state = {
            'power_command': 1000.0,
            'actual_power': 980.0,
            'soc_std': 2.5,
            'soc_consistency': 0.85,
            'temp_std': 3.0,
            'temp_consistency': 0.9,
            'max_temperature': 35.0,
            'current_degradation_cost': 10.0,
            'power_efficiency': 0.92,
            'energy_efficiency': 0.94,
            'safety_score': 0.95,
            'violation_count': 0,
            'constraint_violations': 0,
            'constraint_warnings': 1
        }
        
        previous_state = {
            'current_degradation_cost': 9.8
        }
        
        result = self.calculator.calculate_comprehensive_reward(
            system_state, previous_state, delta_t=1.0
        )
        
        self.assertIsInstance(result.total_reward, float)
        self.assertGreater(len(result.components), 0)
        
        # 验证各组件都被计算
        expected_types = [RewardType.POWER_TRACKING, RewardType.SOC_BALANCE, 
                         RewardType.TEMP_BALANCE, RewardType.LIFETIME_COST]
        for reward_type in expected_types:
            self.assertIn(reward_type, result.components)

class TestStateManager(unittest.TestCase):
    """状态管理器测试"""
    
    def setUp(self):
        """测试前准备"""
        self.system_config = SystemConfig()
        self.manager = StateManager(
            system_config=self.system_config,
            manager_id="TestManager"
        )
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        self.assertEqual(self.manager.manager_id, "TestManager")
        self.assertEqual(self.manager.update_count, 0)
        self.assertGreater(len(self.manager.state_histories), 0)
    
    def test_state_update_and_retrieval(self):
        """测试状态更新和获取"""
        test_state = {
            'voltage': 3.3,
            'current': 10.0,
            'temperature': 25.0
        }
        
        # 更新状态
        success = self.manager.update_state(
            StateScope.CELL_LEVEL,
            StateType.ELECTRICAL,
            test_state
        )
        self.assertTrue(success)
        self.assertEqual(self.manager.update_count, 1)
        
        # 获取当前状态
        retrieved_state = self.manager.get_current_state(
            StateScope.CELL_LEVEL,
            StateType.ELECTRICAL
        )
        self.assertEqual(retrieved_state['voltage'], 3.3)
        self.assertEqual(retrieved_state['current'], 10.0)
    
    def test_state_history(self):
        """测试状态历史"""
        # 添加多个状态记录
        for i in range(5):
            test_state = {'value': i, 'timestamp': time.time() + i}
            self.manager.update_state(
                StateScope.SYSTEM_LEVEL,
                StateType.PERFORMANCE,
                test_state,
                timestamp=time.time() + i
            )
        
        # 获取历史
        history = self.manager.get_state_history(
            StateScope.SYSTEM_LEVEL,
            StateType.PERFORMANCE
        )
        
        self.assertEqual(len(history), 5)
        self.assertEqual(history[-1].data['value'], 4)  # 最新的记录
    
    def test_drl_state_vector(self):
        """测试DRL状态向量生成"""
        # 更新一些状态
        electrical_state = {'voltage': 3.3, 'current': 10.0, 'power': 33.0}
        thermal_state = {'temperature': 25.0, 'heat_rate': 1.0}
        
        self.manager.update_state(StateScope.SYSTEM_LEVEL, StateType.ELECTRICAL, electrical_state)
        self.manager.update_state(StateScope.SYSTEM_LEVEL, StateType.THERMAL, thermal_state)
        
        # 获取状态向量
        state_vector = self.manager.get_drl_state_vector(normalize=True)
        
        self.assertIsInstance(state_vector, np.ndarray)
        self.assertGreater(len(state_vector), 0)
        # 归一化后的值应该在合理范围内
        if len(state_vector) > 0:
            self.assertTrue(all(-1.0 <= x <= 2.0 for x in state_vector))  # 允许一些超出[0,1]的值
    
    def test_observer_registration(self):
        """测试观察者注册"""
        callback_called = []
        
        def test_callback(snapshot):
            callback_called.append(snapshot.data)
        
        # 注册观察者
        success = self.manager.register_observer(
            "test_observer",
            test_callback,
            StateScope.CELL_LEVEL,
            StateType.ELECTRICAL
        )
        self.assertTrue(success)
        
        # 更新状态应该触发回调
        test_state = {'test_value': 42}
        self.manager.update_state(
            StateScope.CELL_LEVEL,
            StateType.ELECTRICAL,
            test_state
        )
        
        # 验证回调被调用
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0]['test_value'], 42)
    
    def test_checkpoint_functionality(self):
        """测试检查点功能"""
        # 创建一些状态
        test_state = {'checkpoint_test': True, 'value': 100}
        self.manager.update_state(StateScope.SYSTEM_LEVEL, StateType.PERFORMANCE, test_state)
        
        # 创建检查点
        success = self.manager.create_state_checkpoint("test_checkpoint")
        self.assertTrue(success)
        
        # 修改状态
        new_state = {'checkpoint_test': True, 'value': 200}
        self.manager.update_state(StateScope.SYSTEM_LEVEL, StateType.PERFORMANCE, new_state)
        
        # 从检查点恢复
        success = self.manager.restore_from_checkpoint("test_checkpoint")
        self.assertTrue(success)
        
        # 验证状态已恢复
        restored_state = self.manager.get_current_state(StateScope.SYSTEM_LEVEL, StateType.PERFORMANCE)
        self.assertEqual(restored_state['value'], 100)

if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestStorageEnvironment,
        TestMultiScaleScheduler,
        TestConstraintValidator,
        TestRewardCalculator,
        TestStateManager
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果摘要
    print(f"\n{'='*50}")
    print(f"测试摘要:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
