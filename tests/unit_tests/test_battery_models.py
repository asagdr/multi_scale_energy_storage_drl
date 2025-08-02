"""
电池模型单元测试
验证单体电池模型的各项功能
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Dict

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.battery_params import BatteryParams, PresetConfigs
from config.system_config import SystemConfig, PresetSystemConfigs
from battery_models.battery_cell_model import BatteryCellModel

class TestBatteryCellModel(unittest.TestCase):
    """电池单体模型测试类"""
    
    def setUp(self):
        """测试初始化"""
        print(f"\n{'='*60}")
        print(f"🧪 测试初始化 - {self._testMethodName}")
        print(f"{'='*60}")
        
        # 创建测试配置
        self.battery_params = PresetConfigs.medium_ess()
        self.system_config = PresetSystemConfigs.research_simulation()
        
        # 创建电池模型
        self.battery = BatteryCellModel(
            battery_params=self.battery_params,
            system_config=self.system_config,
            cell_id="TEST_CELL_001"
        )
        
        print(f"✅ 测试电池创建成功")
        print(f"📊 电池参数: {self.battery_params.CELL_CAPACITY}Ah, {self.battery_params.NOMINAL_VOLTAGE}V")
        print(f"🔧 电池组配置: {self.battery_params.SERIES_NUM}S{self.battery_params.PARALLEL_NUM}P")
    
    def tearDown(self):
        """测试清理"""
        print(f"🧹 测试清理完成 - {self._testMethodName}")
    
    def test_01_initialization(self):
        """测试1: 电池模型初始化"""
        print("\n📝 测试电池模型初始化...")
        
        # 验证初始状态
        self.assertAlmostEqual(self.battery.state.soc, 50.0, places=1)
        self.assertAlmostEqual(self.battery.state.temperature, 25.0, places=1)
        self.assertAlmostEqual(self.battery.state.voltage, 3.275, places=2)
        self.assertEqual(self.battery.state.current, 0.0)
        
        # 验证参数设置
        self.assertEqual(self.battery.params.CELL_CAPACITY, 100.0)  # medium_ess配置
        self.assertEqual(self.battery.cell_id, "TEST_CELL_001")
        
        # 验证衍生参数
        self.assertGreater(self.battery.state.energy_stored, 0)
        self.assertGreater(self.battery.state.capacity_remaining, 0)
        
        print(f"✅ 初始SOC: {self.battery.state.soc:.1f}%")
        print(f"✅ 初始电压: {self.battery.state.voltage:.3f}V")
        print(f"✅ 初始温度: {self.battery.state.temperature:.1f}℃")
        print(f"✅ 储存能量: {self.battery.state.energy_stored:.2f}Wh")
    
    def test_02_soc_ocv_relationship(self):
        """测试2: SOC-OCV关系"""
        print("\n📝 测试SOC-OCV关系...")
        
        # 测试关键点
        test_points = [
            (0, 2.8),     # 0% SOC
            (10, 3.2),    # 10% SOC (平台开始)
            (50, 3.275),  # 50% SOC (平台中点)
            (90, 3.35),   # 90% SOC (平台结束)
            (100, 3.65)   # 100% SOC
        ]
        
        print("🔍 验证SOC-OCV关键点:")
        for soc, expected_ocv in test_points:
            actual_ocv = self.battery.params.get_ocv_from_soc(soc)
            self.assertAlmostEqual(actual_ocv, expected_ocv, places=2,
                                 msg=f"SOC {soc}% 的OCV不匹配")
            print(f"  SOC {soc:3.0f}% -> OCV {actual_ocv:.3f}V (期望: {expected_ocv:.3f}V)")
        
        # 生成完整SOC-OCV曲线用于可视化
        soc_range = np.linspace(0, 100, 101)
        ocv_values = [self.battery.params.get_ocv_from_soc(soc) for soc in soc_range]
        
        # 绘制SOC-OCV曲线
        plt.figure(figsize=(10, 6))
        plt.plot(soc_range, ocv_values, 'b-', linewidth=2, label='SOC-OCV关系')
        plt.scatter([p[0] for p in test_points], [p[1] for p in test_points], 
                   color='red', s=50, zorder=5, label='验证点')
        plt.xlabel('SOC (%)')
        plt.ylabel('开路电压 (V)')
        plt.title('磷酸铁锂电池 SOC-OCV 特性曲线')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # 保存图片
        plt.savefig('test_soc_ocv_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ SOC-OCV关系验证通过")
        print("📊 SOC-OCV曲线已保存: test_soc_ocv_curve.png")
    
    def test_03_current_limits(self):
        """测试3: 电流限制计算"""
        print("\n📝 测试电流限制计算...")
        
        # 测试不同SOC和温度下的电流限制
        test_conditions = [
            (10, 25),   # 低SOC, 常温
            (50, 25),   # 中SOC, 常温
            (90, 25),   # 高SOC, 常温
            (50, 0),    # 中SOC, 低温
            (50, 50),   # 中SOC, 高温
        ]
        
        print("🔍 不同条件下的电流限制:")
        print(f"{'SOC(%)':>6} {'温度(℃)':>8} {'最大充电(A)':>12} {'最大放电(A)':>12}")
        print("-" * 50)
        
        for soc, temp in test_conditions:
            self.battery.reset(initial_soc=soc, initial_temp=temp)
            max_charge, max_discharge = self.battery.calculate_current_limits()
            
            # 验证电流限制合理性
            self.assertGreaterEqual(max_charge, 0, "充电电流不能为负")
            self.assertGreaterEqual(max_discharge, 0, "放电电流不能为负")
            self.assertLessEqual(max_charge, self.battery.params.CELL_CAPACITY * 2, "充电电流过大")
            self.assertLessEqual(max_discharge, self.battery.params.CELL_CAPACITY * 4, "放电电流过大")
            
            print(f"{soc:>6.0f} {temp:>8.0f} {max_charge:>12.2f} {max_discharge:>12.2f}")
        
        print("✅ 电流限制计算验证通过")
    
    def test_04_power_control(self):
        """测试4: 功率控制"""
        print("\n📝 测试功率控制...")
        
        # 重置到50% SOC
        self.battery.reset(initial_soc=50.0)
        
        # 测试不同功率指令
        power_commands = [0, 100, 320, -150, -500, 1000]  # W
        
        print("🔍 功率控制测试:")
        print(f"{'功率指令(W)':>12} {'实际功率(W)':>12} {'电流(A)':>10} {'电压(V)':>10} {'效率(%)':>10}")
        print("-" * 65)
        
        results = []
        for power_cmd in power_commands:
            result = self.battery.step(power_command=power_cmd, delta_t=1.0)
            
            # 验证功率跟踪精度
            actual_power = result['actual_power']
            efficiency = result['power_efficiency'] * 100
            
            # 对于合理功率范围，跟踪误差应该很小
            if abs(power_cmd) < 500:  # 合理功率范围
                power_error_ratio = abs(actual_power - power_cmd) / max(abs(power_cmd), 1)
                self.assertLess(power_error_ratio, 0.05, f"功率跟踪误差过大: {power_cmd}W")
            
            results.append(result)
            print(f"{power_cmd:>12.0f} {actual_power:>12.1f} {result['current']:>10.2f} "
                  f"{result['voltage']:>10.3f} {efficiency:>10.1f}")
        
        print("✅ 功率控制验证通过")
        
        return results
    
    def test_05_charge_discharge_cycle(self):
        """测试5: 充放电循环"""
        print("\n📝 测试充放电循环...")
        
        # 重置电池
        self.battery.reset(initial_soc=50.0)
        
        # 设计一个完整的充放电循环
        cycle_profile = [
            # 阶段1: 恒功率充电 (30分钟)
            {'power': 160, 'duration': 1800, 'description': '恒功率充电'},
            
            # 阶段2: 静置 (5分钟)
            {'power': 0, 'duration': 300, 'description': '静置'},
            
            # 阶段3: 恒功率放电 (40分钟)
            {'power': -120, 'duration': 2400, 'description': '恒功率放电'},
            
            # 阶段4: 静置 (5分钟)
            {'power': 0, 'duration': 300, 'description': '静置'},
        ]
        
        # 执行循环
        time_history = []
        soc_history = []
        voltage_history = []
        current_history = []
        power_history = []
        temp_history = []
        
        current_time = 0
        initial_soc = self.battery.state.soc
        
        print("🔄 执行充放电循环:")
        for stage in cycle_profile:
            print(f"  {stage['description']}: {stage['power']}W, {stage['duration']}s")
            
            for _ in range(stage['duration']):
                result = self.battery.step(
                    power_command=stage['power'],
                    delta_t=1.0
                )
                
                time_history.append(current_time)
                soc_history.append(result['soc'])
                voltage_history.append(result['voltage'])
                current_history.append(result['current'])
                power_history.append(result['actual_power'])
                temp_history.append(result['temperature'])
                
                current_time += 1
        
        final_soc = self.battery.state.soc
        soc_change = final_soc - initial_soc
        
        print(f"📊 循环结果:")
        print(f"  初始SOC: {initial_soc:.2f}%")
        print(f"  最终SOC: {final_soc:.2f}%")
        print(f"  SOC变化: {soc_change:.2f}%")
        print(f"  循环次数: {self.battery.state.cycle_count:.4f}")
        print(f"  累积充放电量: {self.battery.state.cumulative_charge:.2f}Ah")
        
        # 绘制充放电曲线
        time_hours = np.array(time_history) / 3600.0
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # SOC曲线
        axes[0, 0].plot(time_hours, soc_history, 'b-', linewidth=2)
        axes[0, 0].set_ylabel('SOC (%)')
        axes[0, 0].set_title('SOC变化')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 电压曲线
        axes[0, 1].plot(time_hours, voltage_history, 'g-', linewidth=2)
        axes[0, 1].set_ylabel('电压 (V)')
        axes[0, 1].set_title('端电压变化')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 电流曲线
        axes[1, 0].plot(time_hours, current_history, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('时间 (小时)')
        axes[1, 0].set_ylabel('电流 (A)')
        axes[1, 0].set_title('电流变化')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 功率曲线
        axes[1, 1].plot(time_hours, power_history, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('时间 (小时)')
        axes[1, 1].set_ylabel('功率 (W)')
        axes[1, 1].set_title('功率变化')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_charge_discharge_cycle.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 验证循环合理性
        self.assertGreater(max(soc_history), min(soc_history), "SOC应该有变化")
        self.assertLess(abs(soc_change), 20, "单次循环SOC变化不应过大")
        
        print("✅ 充放电循环验证通过")
        print("📊 充放电曲线已保存: test_charge_discharge_cycle.png")
    
    def test_06_temperature_effects(self):
        """测试6: 温度影响"""
        print("\n📝 测试温度影响...")
        
        # 测试不同温度下的性能
        test_temperatures = [-10, 0, 25, 40, 55]  # ℃
        test_power = 160  # W
        
        results = []
        
        print("🌡️ 不同温度下的性能:")
        print(f"{'温度(℃)':>8} {'最大充电(A)':>12} {'最大放电(A)':>12} {'内阻系数':>10} {'效率(%)':>10}")
        print("-" * 60)
        
        for temp in test_temperatures:
            self.battery.reset(initial_soc=50.0, initial_temp=temp)
            
            # 获取电流限制
            max_charge, max_discharge = self.battery.calculate_current_limits()
            
            # 测试温度对内阻的影响
            resistance_factor = self.battery._get_temperature_resistance_factor(temp)
            
            # 测试功率响应
            result = self.battery.step(power_command=test_power, delta_t=1.0)
            efficiency = result['power_efficiency'] * 100
            
            results.append({
                'temperature': temp,
                'max_charge': max_charge,
                'max_discharge': max_discharge,
                'resistance_factor': resistance_factor,
                'efficiency': efficiency
            })
            
            print(f"{temp:>8.0f} {max_charge:>12.2f} {max_discharge:>12.2f} "
                  f"{resistance_factor:>10.2f} {efficiency:>10.1f}")
        
        # 验证温度影响的合理性
        # 低温时电流限制应该降低
        low_temp_result = results[0]  # -10℃
        normal_temp_result = results[2]  # 25℃
        
        self.assertLess(low_temp_result['max_charge'], 
                       normal_temp_result['max_charge'],
                       "低温下充电电流应该降低")
        
        self.assertGreater(low_temp_result['resistance_factor'], 1.0,
                          "低温下内阻系数应该大于1")
        
        print("✅ 温度影响验证通过")
        
        return results
    
    def test_07_state_vector_generation(self):
        """测试7: 状态向量生成 (DRL接口)"""
        print("\n📝 测试状态向量生成...")
        
        # 重置到已知状态
        self.battery.reset(initial_soc=75.0, initial_temp=30.0)
        
        # 执行一些操作以建立历史
        for i in range(10):
            power = 100 * np.sin(i * 0.1)  # 变化的功率
            self.battery.step(power_command=power, delta_t=1.0)
        
        # 获取归一化状态向量
        state_vector_norm = self.battery.get_state_vector(normalize=True)
        state_vector_raw = self.battery.get_state_vector(normalize=False)
        
        print(f"🔢 状态向量维度: {len(state_vector_norm)}")
        print(f"📊 归一化状态向量: {state_vector_norm}")
        print(f"📊 原始状态向量: {state_vector_raw}")
        
        # 验证状态向量
        self.assertEqual(len(state_vector_norm), 8, "状态向量维度不正确")
        
        # 验证归一化范围
        for i, val in enumerate(state_vector_norm):
            if i == 2:  # 电流可以为负
                self.assertGreaterEqual(val, -1.0, f"状态向量第{i}维度超出范围")
                self.assertLessEqual(val, 1.0, f"状态向量第{i}维度超出范围")
            elif i == 6:  # SOC趋势可以为负
                self.assertGreaterEqual(val, -1.0, f"状态向量第{i}维度超出范围")
                self.assertLessEqual(val, 1.0, f"状态向量第{i}维度超出范围")
            else:  # 其他维度应在[0,1]范围内
                self.assertGreaterEqual(val, 0.0, f"状态向量第{i}维度超出范围")
                self.assertLessEqual(val, 1.0, f"状态向量第{i}维度超出范围")
        
        # 验证状态向量的物理意义
        soc_norm = state_vector_norm[0]
        expected_soc_norm = self.battery.state.soc / 100.0
        self.assertAlmostEqual(soc_norm, expected_soc_norm, places=3,
                              msg="SOC归一化不正确")
        
        print("✅ 状态向量生成验证通过")
    
    def test_08_diagnostics(self):
        """测试8: 诊断功能"""
        print("\n📝 测试诊断功能...")
        
        # 运行一段时间以积累诊断数据
        self.battery.reset(initial_soc=40.0)
        
        # 模拟一个复杂的运行场景
        for i in range(100):
            # 变化的功率模式
            t = i * 0.1
            power = 200 * np.sin(t) + 50 * np.cos(t * 3)
            temp = 25 + 5 * np.sin(t * 0.05)  # 缓慢变化的温度
            
            self.battery.step(power_command=power, delta_t=1.0, 
                            ambient_temperature=temp)
        
        # 获取诊断信息
        diagnostics = self.battery.get_diagnostics()
        
        print("🔍 诊断信息:")
        print(f"  电池ID: {diagnostics['cell_id']}")
        print(f"  仿真步数: {diagnostics['simulation_steps']}")
        print(f"  总仿真时间: {diagnostics['total_time']:.1f}s")
        print(f"  SOC范围: {diagnostics['soc_range']}")
        print(f"  电压范围: {diagnostics['voltage_range']}")
        print(f"  电流范围: {diagnostics['current_range']}")
        print(f"  功率范围: {diagnostics['power_range']}")
        print(f"  温度范围: {diagnostics['temperature_range']}")
        print(f"  平均效率: {diagnostics['avg_efficiency']:.3f}")
        print(f"  能量吞吐量: {diagnostics['total_energy_throughput']:.3f}kWh")
        print(f"  等效循环: {diagnostics['equivalent_cycles']:.4f}")
        print(f"  容量利用率: {diagnostics['capacity_utilization']:.2f}")
        print(f"  健康状态: {diagnostics['health_status']}")
        print(f"  SOC趋势: {diagnostics['soc_trend']:.2f}%/h")
        
        # 验证诊断数据的合理性
        self.assertGreater(diagnostics['simulation_steps'], 0)
        self.assertIn(diagnostics['health_status'], ['Normal', 'Warning', 'Degraded', 'Critical'])
        self.assertGreaterEqual(diagnostics['avg_efficiency'], 0.5)
        self.assertLessEqual(diagnostics['avg_efficiency'], 1.0)
        
        print("✅ 诊断功能验证通过")
        
        return diagnostics
    
    def test_09_reset_functionality(self):
        """测试9: 重置功能"""
        print("\n📝 测试重置功能...")
        
        # 运行一段时间改变状态
        initial_state = self.battery.reset(initial_soc=60.0)
        
        for i in range(50):
            self.battery.step(power_command=200, delta_t=1.0)
        
        # 记录运行后的状态
        state_after_run = {
            'soc': self.battery.state.soc,
            'cumulative_charge': self.battery.state.cumulative_charge,
            'cycle_count': self.battery.state.cycle_count,
            'history_length': len(self.battery.state_history)
        }
        
        print(f"🏃 运行后状态:")
        print(f"  SOC: {state_after_run['soc']:.2f}%")
        print(f"  累积充放电: {state_after_run['cumulative_charge']:.3f}Ah")
        print(f"  循环次数: {state_after_run['cycle_count']:.4f}")
        print(f"  历史记录长度: {state_after_run['history_length']}")
        
        # 重置电池
        reset_state = self.battery.reset(initial_soc=30.0, initial_temp=20.0, 
                                       reset_aging=True, random_variation=False)
        
        print(f"🔄 重置后状态:")
        print(f"  SOC: {self.battery.state.soc:.2f}%")
        print(f"  温度: {self.battery.state.temperature:.2f}℃")
        print(f"  累积充放电: {self.battery.state.cumulative_charge:.3f}Ah")
        print(f"  循环次数: {self.battery.state.cycle_count:.4f}")
        print(f"  历史记录长度: {len(self.battery.state_history)}")
        
        # 验证重置效果
        self.assertAlmostEqual(self.battery.state.soc, 30.0, places=1)
        self.assertAlmostEqual(self.battery.state.temperature, 20.0, places=1)
        self.assertEqual(self.battery.state.cumulative_charge, 0.0)
        self.assertEqual(self.battery.state.cycle_count, 0.0)
        self.assertEqual(len(self.battery.state_history), 0)
        
        # 测试随机变异重置
        reset_state_random = self.battery.reset(initial_soc=50.0, 
                                              random_variation=True)
        
        print(f"🎲 随机变异重置:")
        print(f"  SOC: {self.battery.state.soc:.2f}%")
        print(f"  容量: {self.battery.state.capacity_remaining:.2f}Ah")
        
        # 验证随机变异在合理范围内
        self.assertGreater(self.battery.state.soc, 45.0)
        self.assertLess(self.battery.state.soc, 55.0)
        
        print("✅ 重置功能验证通过")
    
    def test_10_performance_benchmark(self):
        """测试10: 性能基准测试"""
        print("\n📝 性能基准测试...")
        
        import time
        
        # 重置电池
        self.battery.reset()
        
        # 性能测试参数
        num_steps = 1000
        test_power = 160  # W
        
        # 计时开始
        start_time = time.time()
        
        # 执行大量仿真步
        for i in range(num_steps):
            power = test_power * np.sin(i * 0.01)  # 变化功率
            result = self.battery.step(power_command=power, delta_t=1.0)
        
        # 计时结束
        end_time = time.time()
        elapsed_time = end_time - start_time
        steps_per_second = num_steps / elapsed_time
        
        print(f"⏱️ 性能基准结果:")
        print(f"  总仿真步数: {num_steps}")
        print(f"  总耗时: {elapsed_time:.3f}s")
        print(f"  仿真速度: {steps_per_second:.1f} steps/s")
        print(f"  平均每步耗时: {elapsed_time/num_steps*1000:.3f}ms")
        
        # 验证性能要求 (应该能达到至少100 steps/s)
        self.assertGreater(steps_per_second, 100, 
                          f"仿真速度过慢: {steps_per_second:.1f} steps/s")
        
        # 获取最终诊断
        final_diagnostics = self.battery.get_diagnostics()
        print(f"  最终SOC: {final_diagnostics['current_soc']:.2f}%")
        print(f"  平均效率: {final_diagnostics['avg_efficiency']:.3f}")
        
        print("✅ 性能基准测试通过")
        
        return {
            'steps_per_second': steps_per_second,
            'total_time': elapsed_time,
            'final_diagnostics': final_diagnostics
        }

def run_comprehensive_test():
    """运行综合测试"""
    print(f"\n{'='*80}")
    print(f"🚀 开始电池模型综合测试")
    print(f"⏰ 测试时间: 2025-08-01 04:56:23")
    print(f"👤 测试用户: asagdr")
    print(f"{'='*80}")
    
    # 创建测试套件
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestBatteryCellModel)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # 输出测试总结
    print(f"\n{'='*80}")
    print(f"📊 测试结果总结")
    print(f"{'='*80}")
    print(f"✅ 成功测试: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ 失败测试: {len(result.failures)}")
    print(f"💥 错误测试: {len(result.errors)}")
    print(f"🏆 总体成功率: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print(f"\n{'='*80}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # 运行综合测试
    success = run_comprehensive_test()
    
    if success:
        print("🎉 所有测试通过！电池模型功能验证成功！")
        print("📊 生成的测试图片:")
        print("  - test_soc_ocv_curve.png: SOC-OCV特性曲线")
        print("  - test_charge_discharge_cycle.png: 充放电循环曲线")
    else:
        print("⚠️ 部分测试失败，请检查代码实现")
    
    print(f"\n🏁 测试完成！")
