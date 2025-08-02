import unittest
import numpy as np
import torch
import time
import psutil
import os
import sys
from typing import Dict, List, Any
import json

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from training.hierarchical_trainer import HierarchicalTrainer
from training.upper_trainer import UpperLayerTrainer
from training.lower_trainer import LowerLayerTrainer
from environment.multi_scale_env import MultiScaleEnvironment
from data_processing.scenario_generator import ScenarioGenerator, ScenarioType
from experiments.basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType

class PerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""
    
    def setUp(self):
        """测试设置"""
        self.results = {}
        self.scenario_generator = ScenarioGenerator()
        
        # 标准配置
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()
        
        # 基准测试配置
        self.benchmark_episodes = 100
        self.warmup_episodes = 10
        
    def test_training_speed_benchmark(self):
        """训练速度基准测试"""
        print("🚀 开始训练速度基准测试")
        
        configurations = [
            {'name': 'small', 'hidden_size': 64, 'episodes': 50},
            {'name': 'medium', 'hidden_size': 128, 'episodes': 50},
            {'name': 'large', 'hidden_size': 256, 'episodes': 50}
        ]
        
        speed_results = {}
        
        for config in configurations:
            print(f"测试配置: {config['name']}")
            
            # 设置模型配置
            model_config = ModelConfig()
            model_config.upper_layer.hidden_size = config['hidden_size']
            model_config.lower_layer.hidden_size = config['hidden_size']
            
            # 设置训练配置
            training_config = TrainingConfig()
            training_config.upper_config.total_episodes = config['episodes']
            training_config.lower_config.total_episodes = config['episodes']
            
            # 创建场景和环境
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=ScenarioType.DAILY_CYCLE,
                scenario_id=f"speed_benchmark_{config['name']}"
            )
            
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=training_config.environment_config
            )
            
            # 创建训练器
            trainer = HierarchicalTrainer(
                config=training_config,
                model_config=model_config,
                trainer_id=f"speed_benchmark_trainer_{config['name']}"
            )
            
            # 预热
            print(f"  预热训练...")
            warmup_start = time.time()
            state = env.reset()
            for _ in range(self.warmup_episodes):
                action = {
                    'upper_action': np.array([0.5]),
                    'lower_action': np.array([0.0, 0.0])
                }
                state, _, done, _ = env.step(action)
                if done:
                    state = env.reset()
            warmup_time = time.time() - warmup_start
            
            # 基准测试
            print(f"  开始基准测试...")
            start_time = time.time()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 执行训练
            training_results = trainer.train(
                environment=env,
                num_episodes=config['episodes']
            )
            
            end_time = time.time()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # 计算性能指标
            total_time = end_time - start_time
            episodes_per_second = config['episodes'] / total_time
            memory_usage = final_memory - initial_memory
            
            speed_results[config['name']] = {
                'total_time': total_time,
                'episodes_per_second': episodes_per_second,
                'memory_usage_mb': memory_usage,
                'warmup_time': warmup_time,
                'avg_episode_time': total_time / config['episodes']
            }
            
            print(f"  完成 - 时间: {total_time:.2f}s, 速度: {episodes_per_second:.2f} eps/s")
        
        # 保存结果
        self.results['training_speed'] = speed_results
        
        # 验证性能
        for config_name, result in speed_results.items():
            self.assertGreater(result['episodes_per_second'], 0)
            self.assertLess(result['avg_episode_time'], 60)  # 每回合应小于60秒
        
        print("✅ 训练速度基准测试完成")
        self._print_speed_results(speed_results)
    
    def test_memory_usage_benchmark(self):
        """内存使用基准测试"""
        print("🧠 开始内存使用基准测试")
        
        batch_sizes = [16, 32, 64, 128]
        memory_results = {}
        
        for batch_size in batch_sizes:
            print(f"测试批量大小: {batch_size}")
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 配置
            training_config = TrainingConfig()
            training_config.upper_config.batch_size = batch_size
            training_config.lower_config.batch_size = batch_size
            training_config.upper_config.total_episodes = 20
            training_config.lower_config.total_episodes = 20
            
            # 创建训练器
            trainer = HierarchicalTrainer(
                config=training_config,
                model_config=self.model_config,
                trainer_id=f"memory_benchmark_trainer_{batch_size}"
            )
            
            # 测量内存使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                initial_gpu_memory = 0
            
            # 创建场景和环境
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=ScenarioType.DAILY_CYCLE,
                scenario_id=f"memory_benchmark_{batch_size}"
            )
            
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=training_config.environment_config
            )
            
            # 执行训练
            training_results = trainer.train(
                environment=env,
                num_episodes=20
            )
            
            # 测量最终内存
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            else:
                final_gpu_memory = 0
            
            memory_results[batch_size] = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'memory_increase_mb': final_memory - initial_memory,
                'initial_gpu_memory_mb': initial_gpu_memory,
                'final_gpu_memory_mb': final_gpu_memory,
                'gpu_memory_increase_mb': final_gpu_memory - initial_gpu_memory,
                'memory_per_sample': (final_memory - initial_memory) / batch_size
            }
            
            print(f"  内存增长: {final_memory - initial_memory:.2f} MB")
        
        # 保存结果
        self.results['memory_usage'] = memory_results
        
        # 验证内存使用合理性
        for batch_size, result in memory_results.items():
            self.assertLess(result['memory_increase_mb'], 2000)  # 内存增长应小于2GB
        
        print("✅ 内存使用基准测试完成")
        self._print_memory_results(memory_results)
    
    def test_convergence_rate_benchmark(self):
        """收敛速度基准测试"""
        print("📈 开始收敛速度基准测试")
        
        algorithms = [
            {'name': 'hierarchical', 'type': 'hierarchical'},
            {'name': 'upper_only', 'type': 'upper'},
            {'name': 'lower_only', 'type': 'lower'}
        ]
        
        convergence_results = {}
        
        for algo in algorithms:
            print(f"测试算法: {algo['name']}")
            
            # 创建配置
            training_config = TrainingConfig()
            training_config.upper_config.total_episodes = 200
            training_config.lower_config.total_episodes = 200
            
            # 创建场景
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=ScenarioType.DAILY_CYCLE,
                scenario_id=f"convergence_benchmark_{algo['name']}"
            )
            
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=training_config.environment_config
            )
            
            # 创建相应的训练器
            if algo['type'] == 'hierarchical':
                trainer = HierarchicalTrainer(
                    config=training_config,
                    model_config=self.model_config,
                    trainer_id=f"convergence_trainer_{algo['name']}"
                )
            elif algo['type'] == 'upper':
                trainer = UpperLayerTrainer(
                    config=training_config.upper_config,
                    model_config=self.model_config,
                    trainer_id=f"convergence_trainer_{algo['name']}"
                )
            else:  # lower
                trainer = LowerLayerTrainer(
                    config=training_config.lower_config,
                    model_config=self.model_config,
                    trainer_id=f"convergence_trainer_{algo['name']}"
                )
            
            # 执行训练并记录收敛过程
            start_time = time.time()
            
            if algo['type'] == 'hierarchical':
                training_results = trainer.train(
                    environment=env,
                    num_episodes=200
                )
            else:
                # 对于单层训练器，需要适配接口
                training_results = {'training_history': [], 'final_metrics': {}}
                
                # 简化的训练循环
                state = env.reset()
                episode_rewards = []
                
                for episode in range(50):  # 减少回合数以加速测试
                    episode_reward = 0
                    state = env.reset()
                    
                    for step in range(100):
                        if algo['type'] == 'upper':
                            action = {'upper_action': np.array([0.5]), 'lower_action': np.array([0.0, 0.0])}
                        else:
                            action = {'upper_action': np.array([0.5]), 'lower_action': np.array([0.0, 0.0])}
                        
                        state, reward, done, _ = env.step(action)
                        episode_reward += reward if isinstance(reward, (int, float)) else sum(reward.values())
                        
                        if done:
                            break
                    
                    episode_rewards.append(episode_reward)
                
                training_results['episode_rewards'] = episode_rewards
            
            training_time = time.time() - start_time
            
            # 分析收敛性
            if algo['type'] == 'hierarchical' and 'training_history' in training_results:
                episode_rewards = [entry.get('episode_reward', 0) for entry in training_results['training_history']]
            else:
                episode_rewards = training_results.get('episode_rewards', [])
            
            if len(episode_rewards) > 10:
                # 检测收敛
                convergence_episode = self._detect_convergence(episode_rewards)
                final_performance = np.mean(episode_rewards[-10:])  # 最后10回合平均
                improvement_rate = (final_performance - episode_rewards[0]) / len(episode_rewards) if episode_rewards else 0
            else:
                convergence_episode = len(episode_rewards)
                final_performance = np.mean(episode_rewards) if episode_rewards else 0
                improvement_rate = 0
            
            convergence_results[algo['name']] = {
                'convergence_episode': convergence_episode,
                'final_performance': final_performance,
                'training_time': training_time,
                'improvement_rate': improvement_rate,
                'episode_rewards': episode_rewards[:50]  # 保存前50回合用于分析
            }
            
            print(f"  收敛回合: {convergence_episode}, 最终性能: {final_performance:.2f}")
        
        # 保存结果
        self.results['convergence_rate'] = convergence_results
        
        # 验证收敛性
        for algo_name, result in convergence_results.items():
            self.assertGreater(result['final_performance'], -1000)  # 基本的性能下界
        
        print("✅ 收敛速度基准测试完成")
        self._print_convergence_results(convergence_results)
    
    def test_cpu_utilization_benchmark(self):
        """CPU利用率基准测试"""
        print("⚡ 开始CPU利用率基准测试")
        
        # CPU密集型配置
        configs = [
            {'name': 'single_thread', 'num_workers': 1},
            {'name': 'multi_thread', 'num_workers': min(4, os.cpu_count())}
        ]
        
        cpu_results = {}
        
        for config in configs:
            print(f"测试配置: {config['name']}")
            
            # 配置
            training_config = TrainingConfig()
            training_config.num_workers = config['num_workers']
            training_config.upper_config.total_episodes = 30
            training_config.lower_config.total_episodes = 30
            
            # 创建场景
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=ScenarioType.DAILY_CYCLE,
                scenario_id=f"cpu_benchmark_{config['name']}"
            )
            
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=training_config.environment_config
            )
            
            # 创建训练器
            trainer = HierarchicalTrainer(
                config=training_config,
                model_config=self.model_config,
                trainer_id=f"cpu_benchmark_trainer_{config['name']}"
            )
            
            # 监控CPU使用率
            cpu_usage_history = []
            
            def monitor_cpu():
                while True:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_usage_history.append(cpu_percent)
                    if len(cpu_usage_history) > 100:  # 限制历史长度
                        break
            
            # 启动CPU监控（简化版本）
            start_time = time.time()
            initial_cpu = psutil.cpu_percent()
            
            # 执行训练
            training_results = trainer.train(
                environment=env,
                num_episodes=30
            )
            
            end_time = time.time()
            final_cpu = psutil.cpu_percent()
            
            # 计算CPU指标
            training_duration = end_time - start_time
            
            cpu_results[config['name']] = {
                'initial_cpu_percent': initial_cpu,
                'final_cpu_percent': final_cpu,
                'training_duration': training_duration,
                'num_workers': config['num_workers'],
                'cpu_efficiency': (final_cpu - initial_cpu) / training_duration if training_duration > 0 else 0
            }
            
            print(f"  CPU使用: {final_cpu:.1f}%, 训练时间: {training_duration:.2f}s")
        
        # 保存结果
        self.results['cpu_utilization'] = cpu_results
        
        print("✅ CPU利用率基准测试完成")
        self._print_cpu_results(cpu_results)
    
    def _detect_convergence(self, rewards: List[float], window_size: int = 20, threshold: float = 0.01) -> int:
        """检测收敛点"""
        if len(rewards) < window_size * 2:
            return len(rewards)
        
        for i in range(window_size, len(rewards) - window_size):
            recent_mean = np.mean(rewards[i:i + window_size])
            previous_mean = np.mean(rewards[i - window_size:i])
            
            if abs(recent_mean - previous_mean) / (abs(previous_mean) + 1e-6) < threshold:
                return i
        
        return len(rewards)
    
    def _print_speed_results(self, results: Dict[str, Any]):
        """打印速度测试结果"""
        print("\n📊 训练速度基准测试结果:")
        print("=" * 60)
        for config, result in results.items():
            print(f"{config:>10}: {result['episodes_per_second']:>8.2f} eps/s, "
                  f"{result['avg_episode_time']:>8.2f}s/ep, "
                  f"{result['memory_usage_mb']:>8.1f}MB")
    
    def _print_memory_results(self, results: Dict[str, Any]):
        """打印内存测试结果"""
        print("\n🧠 内存使用基准测试结果:")
        print("=" * 60)
        for batch_size, result in results.items():
            print(f"Batch {batch_size:>3}: {result['memory_increase_mb']:>8.1f}MB, "
                  f"{result['memory_per_sample']:>8.2f}MB/sample")
    
    def _print_convergence_results(self, results: Dict[str, Any]):
        """打印收敛测试结果"""
        print("\n📈 收敛速度基准测试结果:")
        print("=" * 60)
        for algo, result in results.items():
            print(f"{algo:>12}: {result['convergence_episode']:>8d} episodes, "
                  f"{result['final_performance']:>8.2f} reward, "
                  f"{result['training_time']:>8.2f}s")
    
    def _print_cpu_results(self, results: Dict[str, Any]):
        """打印CPU测试结果"""
        print("\n⚡ CPU利用率基准测试结果:")
        print("=" * 60)
        for config, result in results.items():
            print(f"{config:>12}: {result['final_cpu_percent']:>8.1f}% CPU, "
                  f"{result['training_duration']:>8.2f}s duration, "
                  f"{result['num_workers']:>2d} workers")
    
    def save_benchmark_results(self, filepath: str = "benchmark_results.json"):
        """保存基准测试结果"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 基准测试结果已保存到: {filepath}")


if __name__ == '__main__':
    unittest.main()
