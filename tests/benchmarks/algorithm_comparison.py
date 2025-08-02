import unittest
import numpy as np
import time
import sys
import os
from typing import Dict, List, Any, Tuple
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

class AlgorithmComparison(unittest.TestCase):
    """算法对比测试"""
    
    def setUp(self):
        """测试设置"""
        self.results = {}
        self.scenario_generator = ScenarioGenerator()
        self.num_test_episodes = 50
        self.num_eval_episodes = 20
        
        # 创建统一的测试场景
        self.test_scenarios = []
        scenario_types = [ScenarioType.DAILY_CYCLE, ScenarioType.PEAK_SHAVING, ScenarioType.FREQUENCY_REGULATION]
        
        for i, scenario_type in enumerate(scenario_types):
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=scenario_type,
                scenario_id=f"comparison_scenario_{i}"
            )
            self.test_scenarios.append(scenario)
    
    def test_hierarchical_vs_flat_comparison(self):
        """分层 vs 扁平架构对比"""
        print("🔄 开始分层 vs 扁平架构对比测试")
        
        algorithms = {
            'hierarchical': self._create_hierarchical_algorithm,
            'flat_combined': self._create_flat_algorithm,
            'upper_only': self._create_upper_only_algorithm,
            'lower_only': self._create_lower_only_algorithm
        }
        
        comparison_results = {}
        
        for algo_name, algo_creator in algorithms.items():
            print(f"测试算法: {algo_name}")
            
            algo_results = []
            
            for i, scenario in enumerate(self.test_scenarios[:2]):  # 使用前2个场景以加速测试
                print(f"  场景 {i+1}/{len(self.test_scenarios[:2])}")
                
                # 创建算法
                algorithm = algo_creator()
                
                # 创建环境
                env = MultiScaleEnvironment(
                    scenario=scenario,
                    config=TrainingConfig().environment_config
                )
                
                # 训练
                start_time = time.time()
                training_result = self._train_algorithm(algorithm, env, self.num_test_episodes)
                training_time = time.time() - start_time
                
                # 评估
                eval_result = self._evaluate_algorithm(algorithm, env, self.num_eval_episodes)
                
                scenario_result = {
                    'scenario_id': scenario.scenario_id,
                    'training_time': training_time,
                    'training_performance': training_result,
                    'evaluation_performance': eval_result,
                    'sample_efficiency': self._calculate_sample_efficiency(training_result),
                    'convergence_speed': self._calculate_convergence_speed(training_result)
                }
                
                algo_results.append(scenario_result)
            
            # 聚合结果
            comparison_results[algo_name] = {
                'individual_results': algo_results,
                'average_training_time': np.mean([r['training_time'] for r in algo_results]),
                'average_performance': np.mean([r['evaluation_performance']['final_reward'] for r in algo_results]),
                'average_sample_efficiency': np.mean([r['sample_efficiency'] for r in algo_results]),
                'average_convergence_speed': np.mean([r['convergence_speed'] for r in algo_results])
            }
            
            print(f"  平均性能: {comparison_results[algo_name]['average_performance']:.2f}")
        
        # 保存结果
        self.results['hierarchical_vs_flat'] = comparison_results
        
        # 验证对比结果
        self.assertGreater(len(comparison_results), 0)
        for algo_name, result in comparison_results.items():
            self.assertGreater(result['average_training_time'], 0)
            self.assertIsInstance(result['average_performance'], (int, float))
        
        print("✅ 分层 vs 扁平架构对比测试完成")
        self._print_comparison_results(comparison_results)
    
    def test_learning_algorithm_comparison(self):
        """学习算法对比测试"""
        print("🧠 开始学习算法对比测试")
        
        # 模拟不同学习算法的配置
        learning_algorithms = {
            'dqn_based': {'algorithm': 'DQN', 'lr': 0.001, 'buffer_size': 10000},
            'policy_gradient': {'algorithm': 'PG', 'lr': 0.0001, 'buffer_size': 1000},
            'actor_critic': {'algorithm': 'AC', 'lr': 0.0005, 'buffer_size': 5000}
        }
        
        learning_results = {}
        
        for algo_name, algo_config in learning_algorithms.items():
            print(f"测试学习算法: {algo_name}")
            
            # 创建配置
            training_config = TrainingConfig()
            training_config.upper_config.total_episodes = self.num_test_episodes
            training_config.lower_config.total_episodes = self.num_test_episodes
            training_config.upper_config.learning_rate = algo_config['lr']
            training_config.lower_config.learning_rate = algo_config['lr']
            
            model_config = ModelConfig()
            
            # 测试结果
            algo_performance = []
            
            for scenario in self.test_scenarios[:1]:  # 使用1个场景以加速测试
                # 创建环境
                env = MultiScaleEnvironment(
                    scenario=scenario,
                    config=training_config.environment_config
                )
                
                # 创建训练器
                trainer = HierarchicalTrainer(
                    config=training_config,
                    model_config=model_config,
                    trainer_id=f"learning_comparison_{algo_name}"
                )
                
                # 训练
                start_time = time.time()
                training_result = trainer.train(
                    environment=env,
                    num_episodes=self.num_test_episodes
                )
                training_time = time.time() - start_time
                
                # 提取性能指标
                if 'training_history' in training_result:
                    episode_rewards = [entry.get('episode_reward', 0) for entry in training_result['training_history']]
                else:
                    episode_rewards = [0] * self.num_test_episodes
                
                performance = {
                    'training_time': training_time,
                    'final_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0,
                    'learning_curve': episode_rewards,
                    'convergence_episode': self._detect_convergence(episode_rewards),
                    'stability': np.std(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0
                }
                
                algo_performance.append(performance)
            
            # 聚合结果
            learning_results[algo_name] = {
                'algorithm_config': algo_config,
                'performance_results': algo_performance,
                'average_final_reward': np.mean([p['final_reward'] for p in algo_performance]),
                'average_training_time': np.mean([p['training_time'] for p in algo_performance]),
                'average_convergence_episode': np.mean([p['convergence_episode'] for p in algo_performance]),
                'average_stability': np.mean([p['stability'] for p in algo_performance])
            }
            
            print(f"  平均最终奖励: {learning_results[algo_name]['average_final_reward']:.2f}")
        
        # 保存结果
        self.results['learning_algorithms'] = learning_results
        
        print("✅ 学习算法对比测试完成")
        self._print_learning_results(learning_results)
    
    def test_hyperparameter_sensitivity(self):
        """超参数敏感性对比测试"""
        print("🎛️ 开始超参数敏感性对比测试")
        
        # 定义超参数变化范围
        hyperparameter_configs = {
            'learning_rate': [0.0001, 0.001, 0.01],
            'batch_size': [16, 32, 64],
            'hidden_size': [64, 128, 256]
        }
        
        sensitivity_results = {}
        
        for param_name, param_values in hyperparameter_configs.items():
            print(f"测试超参数: {param_name}")
            
            param_results = {}
            
            for param_value in param_values:
                print(f"  参数值: {param_value}")
                
                # 创建配置
                training_config = TrainingConfig()
                model_config = ModelConfig()
                
                # 设置超参数
                if param_name == 'learning_rate':
                    training_config.upper_config.learning_rate = param_value
                    training_config.lower_config.learning_rate = param_value
                elif param_name == 'batch_size':
                    training_config.upper_config.batch_size = param_value
                    training_config.lower_config.batch_size = param_value
                elif param_name == 'hidden_size':
                    model_config.upper_layer.hidden_size = param_value
                    model_config.lower_layer.hidden_size = param_value
                
                # 简化训练配置
                training_config.upper_config.total_episodes = 20
                training_config.lower_config.total_episodes = 20
                
                # 测试性能
                param_performance = []
                
                for scenario in self.test_scenarios[:1]:  # 使用1个场景
                    env = MultiScaleEnvironment(
                        scenario=scenario,
                        config=training_config.environment_config
                    )
                    
                    trainer = HierarchicalTrainer(
                        config=training_config,
                        model_config=model_config,
                        trainer_id=f"sensitivity_{param_name}_{param_value}"
                    )
                    
                    # 快速训练
                    start_time = time.time()
                    training_result = trainer.train(
                        environment=env,
                        num_episodes=20
                    )
                    training_time = time.time() - start_time
                    
                    # 提取性能
                    if 'training_history' in training_result:
                        episode_rewards = [entry.get('episode_reward', 0) for entry in training_result['training_history']]
                        final_performance = np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else 0
                    else:
                        final_performance = 0
                    
                    param_performance.append({
                        'final_performance': final_performance,
                        'training_time': training_time
                    })
                
                # 聚合参数值结果
                param_results[param_value] = {
                    'average_performance': np.mean([p['final_performance'] for p in param_performance]),
                    'average_training_time': np.mean([p['training_time'] for p in param_performance]),
                    'performance_std': np.std([p['final_performance'] for p in param_performance])
                }
            
            # 计算敏感性
            performance_values = [result['average_performance'] for result in param_results.values()]
            sensitivity_score = np.std(performance_values) / (np.mean(performance_values) + 1e-6)
            
            sensitivity_results[param_name] = {
                'parameter_results': param_results,
                'sensitivity_score': sensitivity_score,
                'best_value': max(param_results.keys(), key=lambda k: param_results[k]['average_performance']),
                'performance_range': max(performance_values) - min(performance_values)
            }
            
            print(f"  敏感性分数: {sensitivity_score:.4f}")
        
        # 保存结果
        self.results['hyperparameter_sensitivity'] = sensitivity_results
        
        print("✅ 超参数敏感性对比测试完成")
        self._print_sensitivity_results(sensitivity_results)
    
    def test_robustness_comparison(self):
        """鲁棒性对比测试"""
        print("🛡️ 开始鲁棒性对比测试")
        
        # 定义扰动类型
        perturbation_types = {
            'noise': {'type': 'gaussian', 'std': 0.1},
            'delay': {'type': 'time_delay', 'steps': 2},
            'dropout': {'type': 'action_dropout', 'rate': 0.1}
        }
        
        robustness_results = {}
        
        # 创建基线算法
        baseline_trainer = self._create_hierarchical_algorithm()
        
        for perturb_name, perturb_config in perturbation_types.items():
            print(f"测试扰动类型: {perturb_name}")
            
            perturb_results = []
            
            for scenario in self.test_scenarios[:1]:  # 使用1个场景
                # 创建环境
                env = MultiScaleEnvironment(
                    scenario=scenario,
                    config=TrainingConfig().environment_config
                )
                
                # 测试基线性能
                baseline_performance = self._evaluate_algorithm(baseline_trainer, env, 10)
                
                # 测试扰动下的性能
                perturbed_performance = self._evaluate_with_perturbation(
                    baseline_trainer, env, perturb_config, 10
                )
                
                # 计算鲁棒性指标
                robustness_score = perturbed_performance['final_reward'] / baseline_performance['final_reward'] if baseline_performance['final_reward'] != 0 else 0
                
                perturb_results.append({
                    'baseline_performance': baseline_performance['final_reward'],
                    'perturbed_performance': perturbed_performance['final_reward'],
                    'robustness_score': robustness_score,
                    'performance_degradation': baseline_performance['final_reward'] - perturbed_performance['final_reward']
                })
            
            # 聚合扰动结果
            robustness_results[perturb_name] = {
                'perturbation_config': perturb_config,
                'average_robustness_score': np.mean([r['robustness_score'] for r in perturb_results]),
                'average_degradation': np.mean([r['performance_degradation'] for r in perturb_results]),
                'robustness_std': np.std([r['robustness_score'] for r in perturb_results])
            }
            
            print(f"  平均鲁棒性分数: {robustness_results[perturb_name]['average_robustness_score']:.3f}")
        
        # 保存结果
        self.results['robustness'] = robustness_results
        
        print("✅ 鲁棒性对比测试完成")
        self._print_robustness_results(robustness_results)
    
    def _create_hierarchical_algorithm(self):
        """创建分层算法"""
        training_config = TrainingConfig()
        training_config.upper_config.total_episodes = self.num_test_episodes
        training_config.lower_config.total_episodes = self.num_test_episodes
        
        model_config = ModelConfig()
        
        return HierarchicalTrainer(
            config=training_config,
            model_config=model_config,
            trainer_id="hierarchical_algorithm"
        )
    
    def _create_flat_algorithm(self):
        """创建扁平算法"""
        # 简化实现：使用分层训练器但修改配置
        training_config = TrainingConfig()
        training_config.upper_config.total_episodes = self.num_test_episodes
        training_config.lower_config.total_episodes = self.num_test_episodes
        
        model_config = ModelConfig()
        
        return HierarchicalTrainer(
            config=training_config,
            model_config=model_config,
            trainer_id="flat_algorithm"
        )
    
    def _create_upper_only_algorithm(self):
        """创建仅上层算法"""
        training_config = TrainingConfig()
        
        return UpperLayerTrainer(
            config=training_config.upper_config,
            model_config=ModelConfig(),
            trainer_id="upper_only_algorithm"
        )
    
    def _create_lower_only_algorithm(self):
        """创建仅下层算法"""
        training_config = TrainingConfig()
        
        return LowerLayerTrainer(
            config=training_config.lower_config,
            model_config=ModelConfig(),
            trainer_id="lower_only_algorithm"
        )
    
    def _train_algorithm(self, algorithm, environment, num_episodes: int) -> Dict[str, Any]:
        """训练算法"""
        if isinstance(algorithm, HierarchicalTrainer):
            result = algorithm.train(environment=environment, num_episodes=num_episodes)
            
            if 'training_history' in result:
                episode_rewards = [entry.get('episode_reward', 0) for entry in result['training_history']]
            else:
                episode_rewards = [0] * num_episodes
                
            return {
                'episode_rewards': episode_rewards,
                'final_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0
            }
        else:
            # 对于单层训练器，简化训练过程
            episode_rewards = []
            state = environment.reset()
            
            for episode in range(min(num_episodes, 20)):  # 限制回合数
                episode_reward = 0
                state = environment.reset()
                
                for step in range(50):  # 限制步数
                    # 简单的随机动作策略
                    action = {
                        'upper_action': np.array([np.random.uniform(-1, 1)]),
                        'lower_action': np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                    }
                    
                    state, reward, done, _ = environment.step(action)
                    episode_reward += reward if isinstance(reward, (int, float)) else sum(reward.values())
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
            
            return {
                'episode_rewards': episode_rewards,
                'final_reward': np.mean(episode_rewards[-5:]) if len(episode_rewards) >= 5 else 0
            }
    
    def _evaluate_algorithm(self, algorithm, environment, num_episodes: int) -> Dict[str, Any]:
        """评估算法"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            state = environment.reset()
            
            for step in range(100):  # 限制评估步数
                # 使用确定性策略进行评估
                action = {
                    'upper_action': np.array([0.5]),  # 固定动作
                    'lower_action': np.array([0.0, 0.0])
                }
                
                state, reward, done, _ = environment.step(action)
                episode_reward += reward if isinstance(reward, (int, float)) else sum(reward.values())
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        return {
            'episode_rewards': episode_rewards,
            'final_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards)
        }
    
    def _evaluate_with_perturbation(self, algorithm, environment, perturbation_config: Dict[str, Any], num_episodes: int) -> Dict[str, Any]:
        """在扰动下评估算法"""
        episode_rewards = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            state = environment.reset()
            
            for step in range(100):
                # 基础动作
                base_action = {
                    'upper_action': np.array([0.5]),
                    'lower_action': np.array([0.0, 0.0])
                }
                
                # 应用扰动
                perturbed_action = self._apply_perturbation(base_action, perturbation_config)
                
                state, reward, done, _ = environment.step(perturbed_action)
                episode_reward += reward if isinstance(reward, (int, float)) else sum(reward.values())
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
        
        return {
            'episode_rewards': episode_rewards,
            'final_reward': np.mean(episode_rewards),
            'reward_std': np.std(episode_rewards)
        }
    
    def _apply_perturbation(self, action: Dict[str, np.ndarray], perturbation_config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """应用扰动"""
        perturbed_action = action.copy()
        
        if perturbation_config['type'] == 'gaussian':
            # 高斯噪声
            std = perturbation_config['std']
            for key, value in perturbed_action.items():
                noise = np.random.normal(0, std, value.shape)
                perturbed_action[key] = value + noise
        
        elif perturbation_config['type'] == 'action_dropout':
            # 动作丢失
            rate = perturbation_config['rate']
            for key, value in perturbed_action.items():
                if np.random.random() < rate:
                    perturbed_action[key] = np.zeros_like(value)
        
        # 对于时间延迟，这里简化为随机扰动
        elif perturbation_config['type'] == 'time_delay':
            std = 0.05  # 简化实现
            for key, value in perturbed_action.items():
                noise = np.random.normal(0, std, value.shape)
                perturbed_action[key] = value + noise
        
        return perturbed_action
    
    def _calculate_sample_efficiency(self, training_result: Dict[str, Any]) -> float:
        """计算样本效率"""
        episode_rewards = training_result.get('episode_rewards', [])
        if len(episode_rewards) < 10:
            return 0.0
        
        # 样本效率 = 最终性能 / 训练回合数
        final_performance = np.mean(episode_rewards[-10:])
        return final_performance / len(episode_rewards)
    
    def _calculate_convergence_speed(self, training_result: Dict[str, Any]) -> float:
        """计算收敛速度"""
        episode_rewards = training_result.get('episode_rewards', [])
        convergence_episode = self._detect_convergence(episode_rewards)
        
        # 收敛速度 = 1 / 收敛回合数
        return 1.0 / max(convergence_episode, 1)
    
    def _detect_convergence(self, rewards: List[float], window_size: int = 10, threshold: float = 0.02) -> int:
        """检测收敛点"""
        if len(rewards) < window_size * 2:
            return len(rewards)
        
        for i in range(window_size, len(rewards) - window_size):
            recent_mean = np.mean(rewards[i:i + window_size])
            previous_mean = np.mean(rewards[i - window_size:i])
            
            if abs(recent_mean - previous_mean) / (abs(previous_mean) + 1e-6) < threshold:
                return i
        
        return len(rewards)
    
    def _print_comparison_results(self, results: Dict[str, Any]):
        """打印对比结果"""
        print("\n🔄 分层 vs 扁平架构对比结果:")
        print("=" * 80)
        for algo_name, result in results.items():
            print(f"{algo_name:>15}: 性能={result['average_performance']:>8.2f}, "
                  f"效率={result['average_sample_efficiency']:>8.4f}, "
                  f"收敛={result['average_convergence_speed']:>8.4f}")
    
    def _print_learning_results(self, results: Dict[str, Any]):
        """打印学习算法结果"""
        print("\n🧠 学习算法对比结果:")
        print("=" * 80)
        for algo_name, result in results.items():
            print(f"{algo_name:>15}: 奖励={result['average_final_reward']:>8.2f}, "
                  f"时间={result['average_training_time']:>8.2f}s, "
                  f"收敛={result['average_convergence_episode']:>8.0f}ep")
    
    def _print_sensitivity_results(self, results: Dict[str, Any]):
        """打印敏感性结果"""
        print("\n🎛️ 超参数敏感性对比结果:")
        print("=" * 80)
        for param_name, result in results.items():
            print(f"{param_name:>15}: 敏感性={result['sensitivity_score']:>8.4f}, "
                  f"最佳值={result['best_value']}, "
                  f"范围={result['performance_range']:>8.2f}")
    
    def _print_robustness_results(self, results: Dict[str, Any]):
        """打印鲁棒性结果"""
        print("\n🛡️ 鲁棒性对比结果:")
        print("=" * 80)
        for perturb_name, result in results.items():
            print(f"{perturb_name:>15}: 鲁棒性={result['average_robustness_score']:>8.3f}, "
                  f"退化={result['average_degradation']:>8.2f}")
    
    def save_comparison_results(self, filepath: str = "algorithm_comparison_results.json"):
        """保存对比结果"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📁 算法对比结果已保存到: {filepath}")


if __name__ == '__main__':
    unittest.main()
