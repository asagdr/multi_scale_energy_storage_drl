import unittest
import numpy as np
import time
import psutil
import sys
import os
from typing import Dict, List, Any, Tuple
import json
import threading
import multiprocessing

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from training.hierarchical_trainer import HierarchicalTrainer
from environment.multi_scale_env import MultiScaleEnvironment
from data_processing.scenario_generator import ScenarioGenerator, ScenarioType

class ScalabilityTest(unittest.TestCase):
    """可扩展性测试"""
    
    def setUp(self):
        """测试设置"""
        self.results = {}
        self.scenario_generator = ScenarioGenerator()
        
        # 获取系统信息
        self.cpu_count = os.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"系统信息: {self.cpu_count} CPUs, {self.memory_gb:.1f}GB RAM")
    
    def test_battery_count_scalability(self):
        """电池数量可扩展性测试"""
        print("🔋 开始电池数量可扩展性测试")
        
        # 不同的电池数量配置
        battery_counts = [1, 3, 5, 10, 20]
        scalability_results = {}
        
        for num_batteries in battery_counts:
            print(f"测试 {num_batteries} 个电池")
            
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                # 调整配置以适应电池数量
                training_config = TrainingConfig()
                training_config.upper_config.total_episodes = 10
                training_config.lower_config.total_episodes = 10
                
                model_config = ModelConfig()
                # 调整下层动作维度以适应电池数量
                model_config.lower_layer.action_dim = num_batteries * 2  # 每个电池2个控制维度
                model_config.lower_layer.state_dim = num_batteries * 4   # 每个电池4个状态维度
                
                # 创建场景
                scenario = self.scenario_generator.generate_scenario(
                    scenario_type=ScenarioType.DAILY_CYCLE,
                    scenario_id=f"scalability_battery_{num_batteries}"
                )
                
                # 创建环境
                env = MultiScaleEnvironment(
                    scenario=scenario,
                    config=training_config.environment_config
                )
                
                # 创建训练器
                trainer = HierarchicalTrainer(
                    config=training_config,
                    model_config=model_config,
                    trainer_id=f"scalability_trainer_{num_batteries}"
                )
                
                # 执行简化训练
                training_result = trainer.train(
                    environment=env,
                    num_episodes=10
                )
                
                # 测量性能
                execution_time = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_usage = memory_after - memory_before
                
                # 计算吞吐量
                total_steps = 10 * 100  # 假设每回合100步
                throughput = total_steps / execution_time
                
                scalability_results[num_batteries] = {
                    'execution_time': execution_time,
                    'memory_usage_mb': memory_usage,
                    'throughput_steps_per_sec': throughput,
                    'memory_per_battery_mb': memory_usage / num_batteries,
                    'time_per_battery_sec': execution_time / num_batteries,
                    'success': True
                }
                
                print(f"  成功 - 时间: {execution_time:.2f}s, 内存: {memory_usage:.1f}MB")
                
            except Exception as e:
                print(f"  失败 - {str(e)}")
                scalability_results[num_batteries] = {
                    'execution_time': time.time() - start_time,
                    'memory_usage_mb': 0,
                    'throughput_steps_per_sec': 0,
                    'memory_per_battery_mb': 0,
                    'time_per_battery_sec': 0,
                    'success': False,
                    'error': str(e)
                }
        
        # 分析可扩展性
        scalability_analysis = self._analyze_battery_scalability(scalability_results)
        
        # 保存结果
        self.results['battery_count_scalability'] = {
            'results': scalability_results,
            'analysis': scalability_analysis
        }
        
        print("✅ 电池数量可扩展性测试完成")
        self._print_battery_scalability_results(scalability_results, scalability_analysis)
    
    def test_episode_count_scalability(self):
        """训练回合数可扩展性测试"""
        print("📈 开始训练回合数可扩展性测试")
        
        episode_counts = [10, 50, 100, 200, 500]
        episode_scalability_results = {}
        
        for num_episodes in episode_counts:
            print(f"测试 {num_episodes} 个训练回合")
            
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                # 标准配置
                training_config = TrainingConfig()
                training_config.upper_config.total_episodes = num_episodes
                training_config.lower_config.total_episodes = num_episodes
                
                model_config = ModelConfig()
                model_config.upper_layer.hidden_size = 64  # 较小的网络以加速测试
                model_config.lower_layer.hidden_size = 64
                
                # 创建场景
                scenario = self.scenario_generator.generate_scenario(
                    scenario_type=ScenarioType.DAILY_CYCLE,
                    scenario_id=f"scalability_episode_{num_episodes}"
                )
                
                # 创建环境
                env = MultiScaleEnvironment(
                    scenario=scenario,
                    config=training_config.environment_config
                )
                
                # 创建训练器
                trainer = HierarchicalTrainer(
                    config=training_config,
                    model_config=model_config,
                    trainer_id=f"episode_scalability_trainer_{num_episodes}"
                )
                
                # 监控资源使用
                resource_monitor = ResourceMonitor()
                resource_monitor.start()
                
                # 执行训练
                training_result = trainer.train(
                    environment=env,
                    num_episodes=min(num_episodes, 100)  # 限制实际执行的回合数以控制测试时间
                )
                
                # 停止监控
                resource_monitor.stop()
                
                execution_time = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage = memory_after - memory_before
                
                # 获取资源使用统计
                cpu_stats = resource_monitor.get_cpu_stats()
                memory_stats = resource_monitor.get_memory_stats()
                
                episode_scalability_results[num_episodes] = {
                    'execution_time': execution_time,
                    'memory_usage_mb': memory_usage,
                    'episodes_per_second': min(num_episodes, 100) / execution_time,
                    'avg_cpu_percent': cpu_stats['avg_cpu_percent'],
                    'max_cpu_percent': cpu_stats['max_cpu_percent'],
                    'avg_memory_mb': memory_stats['avg_memory_mb'],
                    'max_memory_mb': memory_stats['max_memory_mb'],
                    'success': True
                }
                
                print(f"  完成 - 时间: {execution_time:.2f}s, CPU: {cpu_stats['avg_cpu_percent']:.1f}%")
                
            except Exception as e:
                print(f"  失败 - {str(e)}")
                episode_scalability_results[num_episodes] = {
                    'execution_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
        
        # 保存结果
        self.results['episode_count_scalability'] = episode_scalability_results
        
        print("✅ 训练回合数可扩展性测试完成")
        self._print_episode_scalability_results(episode_scalability_results)
    
    def test_parallel_processing_scalability(self):
        """并行处理可扩展性测试"""
        print("⚡ 开始并行处理可扩展性测试")
        
        # 测试不同的并行度
        worker_counts = [1, 2, 4, min(8, self.cpu_count)]
        parallel_results = {}
        
        for num_workers in worker_counts:
            print(f"测试 {num_workers} 个并行worker")
            
            start_time = time.time()
            
            try:
                # 创建多个独立的训练任务
                training_tasks = []
                
                for worker_id in range(num_workers):
                    task = self._create_training_task(
                        task_id=f"parallel_task_{worker_id}",
                        episodes=20
                    )
                    training_tasks.append(task)
                
                # 并行执行训练任务
                if num_workers == 1:
                    # 串行执行
                    results = [self._execute_training_task(task) for task in training_tasks]
                else:
                    # 并行执行
                    with multiprocessing.Pool(processes=num_workers) as pool:
                        results = pool.map(self._execute_training_task, training_tasks)
                
                execution_time = time.time() - start_time
                
                # 聚合结果
                successful_tasks = [r for r in results if r['success']]
                total_episodes = sum(r['episodes_completed'] for r in successful_tasks)
                avg_performance = np.mean([r['final_performance'] for r in successful_tasks]) if successful_tasks else 0
                
                parallel_results[num_workers] = {
                    'execution_time': execution_time,
                    'successful_tasks': len(successful_tasks),
                    'total_tasks': len(training_tasks),
                    'total_episodes': total_episodes,
                    'episodes_per_second': total_episodes / execution_time,
                    'avg_performance': avg_performance,
                    'speedup': parallel_results[1]['execution_time'] / execution_time if 1 in parallel_results else 1.0,
                    'efficiency': (parallel_results[1]['execution_time'] / execution_time) / num_workers if 1 in parallel_results else 1.0,
                    'success': True
                }
                
                print(f"  完成 - 时间: {execution_time:.2f}s, 成功率: {len(successful_tasks)}/{len(training_tasks)}")
                
            except Exception as e:
                print(f"  失败 - {str(e)}")
                parallel_results[num_workers] = {
                    'execution_time': time.time() - start_time,
                    'success': False,
                    'error': str(e)
                }
        
        # 保存结果
        self.results['parallel_processing_scalability'] = parallel_results
        
        print("✅ 并行处理可扩展性测试完成")
        self._print_parallel_scalability_results(parallel_results)
    
    def test_memory_scalability_limits(self):
        """内存可扩展性极限测试"""
        print("🧠 开始内存可扩展性极限测试")
        
        # 逐步增加模型大小直到内存限制
        hidden_sizes = [64, 128, 256, 512, 1024, 2048]
        memory_limit_results = {}
        
        for hidden_size in hidden_sizes:
            print(f"测试隐藏层大小: {hidden_size}")
            
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            
            try:
                # 创建大模型配置
                training_config = TrainingConfig()
                training_config.upper_config.total_episodes = 5  # 减少回合数以专注于内存测试
                training_config.lower_config.total_episodes = 5
                
                model_config = ModelConfig()
                model_config.upper_layer.hidden_size = hidden_size
                model_config.lower_layer.hidden_size = hidden_size
                model_config.upper_layer.num_layers = 3  # 增加层数
                model_config.lower_layer.num_layers = 3
                
                # 创建场景
                scenario = self.scenario_generator.generate_scenario(
                    scenario_type=ScenarioType.DAILY_CYCLE,
                    scenario_id=f"memory_limit_test_{hidden_size}"
                )
                
                # 创建环境
                env = MultiScaleEnvironment(
                    scenario=scenario,
                    config=training_config.environment_config
                )
                
                # 创建训练器
                trainer = HierarchicalTrainer(
                    config=training_config,
                    model_config=model_config,
                    trainer_id=f"memory_limit_trainer_{hidden_size}"
                )
                
                # 执行训练
                start_time = time.time()
                training_result = trainer.train(
                    environment=env,
                    num_episodes=5
                )
                execution_time = time.time() - start_time
                
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage = memory_after - memory_before
                
                # 估算模型参数数量
                estimated_params = self._estimate_model_parameters(model_config)
                
                memory_limit_results[hidden_size] = {
                    'memory_usage_mb': memory_usage,
                    'execution_time': execution_time,
                    'estimated_parameters': estimated_params,
                    'memory_per_parameter_bytes': (memory_usage * 1024 * 1024) / estimated_params if estimated_params > 0 else 0,
                    'available_memory_mb': available_memory,
                    'memory_utilization_percent': (memory_usage / available_memory) * 100,
                    'success': True
                }
                
                print(f"  成功 - 内存使用: {memory_usage:.1f}MB, 参数: {estimated_params:,}")
                
                # 如果内存使用超过80%，停止测试
                if memory_usage > available_memory * 0.8:
                    print(f"  达到内存限制，停止测试")
                    break
                    
            except Exception as e:
                print(f"  失败 - {str(e)}")
                memory_limit_results[hidden_size] = {
                    'memory_usage_mb': 0,
                    'success': False,
                    'error': str(e)
                }
                break  # 内存不足，停止测试
        
        # 保存结果
        self.results['memory_scalability_limits'] = memory_limit_results
        
        print("✅ 内存可扩展性极限测试完成")
        self._print_memory_limit_results(memory_limit_results)
    
    def _create_training_task(self, task_id: str, episodes: int) -> Dict[str, Any]:
        """创建训练任务"""
        return {
            'task_id': task_id,
            'episodes': episodes,
            'hidden_size': 64,
            'scenario_type': ScenarioType.DAILY_CYCLE
        }
    
    def _execute_training_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行训练任务"""
        try:
            # 创建配置
            training_config = TrainingConfig()
            training_config.upper_config.total_episodes = task['episodes']
            training_config.lower_config.total_episodes = task['episodes']
            
            model_config = ModelConfig()
            model_config.upper_layer.hidden_size = task['hidden_size']
            model_config.lower_layer.hidden_size = task['hidden_size']
            
            # 创建场景
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=task['scenario_type'],
                scenario_id=task['task_id']
            )
            
            # 创建环境
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=training_config.environment_config
            )
            
            # 创建训练器
            trainer = HierarchicalTrainer(
                config=training_config,
                model_config=model_config,
                trainer_id=task['task_id']
            )
            
            # 执行训练
            training_result = trainer.train(
                environment=env,
                num_episodes=task['episodes']
            )
            
            # 提取性能指标
            if 'training_history' in training_result:
                episode_rewards = [entry.get('episode_reward', 0) for entry in training_result['training_history']]
                final_performance = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else 0
            else:
                final_performance = 0
            
            return {
                'task_id': task['task_id'],
                'episodes_completed': task['episodes'],
                'final_performance': final_performance,
                'success': True
            }
            
        except Exception as e:
            return {
                'task_id': task['task_id'],
                'episodes_completed': 0,
                'final_performance': 0,
                'success': False,
                'error': str(e)
            }
    
    def _estimate_model_parameters(self, model_config: ModelConfig) -> int:
        """估算模型参数数量"""
        upper_params = (
            model_config.upper_layer.state_dim * model_config.upper_layer.hidden_size +
            model_config.upper_layer.hidden_size * model_config.upper_layer.hidden_size * (model_config.upper_layer.num_layers - 1) +
            model_config.upper_layer.hidden_size * model_config.upper_layer.action_dim
        )
        
        lower_params = (
            model_config.lower_layer.state_dim * model_config.lower_layer.hidden_size +
            model_config.lower_layer.hidden_size * model_config.lower_layer.hidden_size * (model_config.lower_layer.num_layers - 1) +
            model_config.lower_layer.hidden_size * model_config.lower_layer.action_dim
        )
        
        return upper_params + lower_params
    
    def _analyze_battery_scalability(self, results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """分析电池可扩展性"""
        successful_results = {k: v for k, v in results.items() if v['success']}
        
        if len(successful_results) < 2:
            return {'scalability_factor': 0, 'linear_scalability': False}
        
        battery_counts = list(successful_results.keys())
        execution_times = [successful_results[k]['execution_time'] for k in battery_counts]
        memory_usages = [successful_results[k]['memory_usage_mb'] for k in battery_counts]
        
        # 计算线性回归斜率
        time_slope = np.polyfit(battery_counts, execution_times, 1)[0]
        memory_slope = np.polyfit(battery_counts, memory_usages, 1)[0]
        
        # 评估线性可扩展性
        time_correlation = np.corrcoef(battery_counts, execution_times)[0, 1]
        memory_correlation = np.corrcoef(battery_counts, memory_usages)[0, 1]
        
        return {
            'time_slope': time_slope,
            'memory_slope': memory_slope,
            'time_correlation': time_correlation,
            'memory_correlation': memory_correlation,
            'linear_scalability': time_correlation > 0.8 and memory_correlation > 0.8,
            'max_tested_batteries': max(battery_counts),
            'scalability_factor': time_slope
        }
    
    def _print_battery_scalability_results(self, results: Dict[int, Dict[str, Any]], analysis: Dict[str, Any]):
        """打印电池可扩展性结果"""
        print("\n🔋 电池数量可扩展性测试结果:")
        print("=" * 80)
        for num_batteries, result in results.items():
            if result['success']:
                print(f"电池数 {num_batteries:>3}: 时间={result['execution_time']:>6.2f}s, "
                      f"内存={result['memory_usage_mb']:>6.1f}MB, "
                      f"吞吐量={result['throughput_steps_per_sec']:>6.1f}steps/s")
            else:
                print(f"电池数 {num_batteries:>3}: 失败 - {result.get('error', '未知错误')}")
        
        print(f"\n分析结果:")
        print(f"  线性可扩展性: {'是' if analysis['linear_scalability'] else '否'}")
        print(f"  时间相关性: {analysis['time_correlation']:.3f}")
        print(f"  内存相关性: {analysis['memory_correlation']:.3f}")
    
    def _print_episode_scalability_results(self, results: Dict[int, Dict[str, Any]]):
        """打印回合数可扩展性结果"""
        print("\n📈 训练回合数可扩展性测试结果:")
        print("=" * 80)
        for num_episodes, result in results.items():
            if result['success']:
                print(f"回合数 {num_episodes:>4}: 时间={result['execution_time']:>6.2f}s, "
                      f"内存={result['memory_usage_mb']:>6.1f}MB, "
                      f"CPU={result['avg_cpu_percent']:>5.1f}%")
            else:
                print(f"回合数 {num_episodes:>4}: 失败 - {result.get('error', '未知错误')}")
    
    def _print_parallel_scalability_results(self, results: Dict[int, Dict[str, Any]]):
        """打印并行可扩展性结果"""
        print("\n⚡ 并行处理可扩展性测试结果:")
        print("=" * 80)
        for num_workers, result in results.items():
            if result['success']:
                print(f"Worker数 {num_workers:>2}: 时间={result['execution_time']:>6.2f}s, "
                      f"加速比={result['speedup']:>5.2f}x, "
                      f"效率={result['efficiency']:>5.2f}")
            else:
                print(f"Worker数 {num_workers:>2}: 失败 - {result.get('error', '未知错误')}")
    
    def _print_memory_limit_results(self, results: Dict[int, Dict[str, Any]]):
        """打印内存极限结果"""
        print("\n🧠 内存可扩展性极限测试结果:")
        print("=" * 80)
        for hidden_size, result in results.items():
            if result['success']:
                print(f"隐藏层 {hidden_size:>4}: 内存={result['memory_usage_mb']:>6.1f}MB, "
                      f"参数={result['estimated_parameters']:>8,}, "
                      f"利用率={result['memory_utilization_percent']:>5.1f}%")
            else:
                print(f"隐藏层 {hidden_size:>4}: 失败 - {result.get('error', '未知错误')}")
    
    def save_scalability_results(self, filepath: str = "scalability_test_results.json"):
        """保存可扩展性测试结果"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 可扩展性测试结果已保存到: {filepath}")


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_data = []
        self.memory_data = []
        self.monitor_thread = None
    
    def start(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.cpu_data.append(cpu_percent)
            self.memory_data.append(memory_mb)
            
            time.sleep(0.5)
    
    def get_cpu_stats(self) -> Dict[str, float]:
        """获取CPU统计"""
        if not self.cpu_data:
            return {'avg_cpu_percent': 0, 'max_cpu_percent': 0}
        
        return {
            'avg_cpu_percent': np.mean(self.cpu_data),
            'max_cpu_percent': np.max(self.cpu_data)
        }
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        if not self.memory_data:
            return {'avg_memory_mb': 0, 'max_memory_mb': 0}
        
        return {
            'avg_memory_mb': np.mean(self.memory_data),
            'max_memory_mb': np.max(self.memory_data)
        }


if __name__ == '__main__':
    unittest.main()
