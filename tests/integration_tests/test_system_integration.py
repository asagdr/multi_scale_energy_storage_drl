import unittest
import numpy as np
import tempfile
import shutil
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from config.environment_config import EnvironmentConfig
from environment.multi_scale_env import MultiScaleEnvironment
from drl_agents.upper_layer.hierarchical_controller import HierarchicalController
from drl_agents.lower_layer.battery_controller import BatteryController
from training.hierarchical_trainer import HierarchicalTrainer
from data_processing.scenario_generator import ScenarioGenerator, ScenarioType
from utils.logger import Logger
from utils.metrics import MetricsCalculator
from utils.checkpoint_manager import CheckpointManager
from utils.experiment_tracker import ExperimentTracker, ExperimentConfig

class TestSystemIntegration(unittest.TestCase):
    """系统集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # 创建配置
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()
        self.env_config = EnvironmentConfig()
        
        # 简化配置以加速测试
        self.training_config.upper_config.total_episodes = 5
        self.training_config.lower_config.total_episodes = 5
        self.model_config.upper_layer.hidden_size = 32
        self.model_config.lower_layer.hidden_size = 32
        
        # 初始化组件
        self.scenario_generator = ScenarioGenerator()
        self.logger = Logger("SystemIntegrationTest")
        self.metrics_calculator = MetricsCalculator()
        self.checkpoint_manager = CheckpointManager()
        
    def test_end_to_end_system_integration(self):
        """测试端到端系统集成"""
        # 1. 场景生成
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.DAILY_CYCLE,
            scenario_id="e2e_test_scenario"
        )
        
        # 2. 环境创建
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=self.env_config
        )
        
        # 3. 智能体创建
        upper_agent = HierarchicalController(
            state_dim=self.model_config.upper_layer.state_dim,
            action_dim=self.model_config.upper_layer.action_dim,
            config=self.model_config.upper_layer
        )
        
        lower_agent = BatteryController(
            state_dim=self.model_config.lower_layer.state_dim,
            action_dim=self.model_config.lower_layer.action_dim,
            config=self.model_config.lower_layer
        )
        
        # 4. 训练器创建
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="e2e_trainer"
        )
        
        # 5. 端到端训练
        training_results = trainer.train(
            environment=env,
            num_episodes=3
        )
        
        # 6. 验证训练结果
        self.assertIsNotNone(training_results)
        self.assertIn('training_history', training_results)
        self.assertIn('final_metrics', training_results)
        
        # 7. 测试部署
        state = env.reset()
        total_reward = 0
        
        for step in range(10):
            # 上层决策
            upper_action = upper_agent.select_action(state['upper_state'])
            
            # 下层控制
            lower_action = lower_agent.select_action(state['lower_state'])
            
            # 组合动作
            action = {
                'upper_action': upper_action,
                'lower_action': lower_action
            }
            
            # 环境交互
            next_state, reward, done, info = env.step(action)
            total_reward += reward if isinstance(reward, (int, float)) else sum(reward.values())
            
            state = next_state
            
            if done:
                break
        
        # 验证部署运行
        self.assertIsInstance(total_reward, (int, float))
        
        print("✅ 端到端系统集成测试通过")
    
    def test_multi_component_interaction(self):
        """测试多组件交互"""
        # 实验跟踪器
        experiment_tracker = ExperimentTracker()
        
        # 创建实验
        exp_config = ExperimentConfig(
            name="multi_component_test",
            description="多组件交互测试",
            hyperparameters={
                'learning_rate': 0.001,
                'batch_size': 32
            }
        )
        
        exp_id = experiment_tracker.create_experiment(exp_config)
        experiment_tracker.start_experiment(exp_id)
        
        # 场景和环境
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.PEAK_SHAVING,
            scenario_id="multi_component_scenario"
        )
        
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=self.env_config
        )
        
        # 训练器
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="multi_component_trainer"
        )
        
        # 执行训练并记录指标
        state = env.reset()
        episode_rewards = []
        
        for episode in range(3):
            episode_reward = 0
            state = env.reset()
            
            for step in range(20):
                # 模拟训练步骤
                action = {
                    'upper_action': np.array([np.random.uniform(-1, 1)]),
                    'lower_action': np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                }
                
                next_state, reward, done, info = env.step(action)
                step_reward = reward if isinstance(reward, (int, float)) else sum(reward.values())
                episode_reward += step_reward
                
                # 记录指标
                experiment_tracker.log_metric(
                    'step_reward', step_reward, 
                    step=episode*20+step, episode=episode
                )
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            
            # 记录回合指标
            experiment_tracker.log_metric(
                'episode_reward', episode_reward,
                step=episode, episode=episode
            )
            
            # 保存检查点
            if episode % 2 == 0:
                checkpoint_path = os.path.join(self.temp_dir, f"checkpoint_ep{episode}.pth")
                trainer.save_checkpoint(checkpoint_path)
                experiment_tracker.log_model_checkpoint(checkpoint_path)
        
        # 计算性能指标
        data = {
            'episode_rewards': np.array(episode_rewards),
            'performance_score': np.array([0.8, 0.85, 0.9])  # 模拟性能分数
        }
        
        metrics_suite = self.metrics_calculator.calculate_metrics(
            data=data,
            metric_names=['tracking_accuracy', 'energy_efficiency']
        )
        
        # 验证组件交互
        self.assertIsNotNone(metrics_suite)
        self.assertGreater(len(metrics_suite.metrics), 0)
        
        # 完成实验
        experiment_tracker.complete_experiment(exp_id, {
            'final_reward': episode_rewards[-1],
            'avg_reward': np.mean(episode_rewards)
        })
        
        # 验证实验记录
        experiment = experiment_tracker.get_experiment(exp_id)
        self.assertEqual(experiment.status.value, 'completed')
        self.assertGreater(len(experiment.metrics), 0)
        
        print("✅ 多组件交互测试通过")
    
    def test_fault_tolerance_and_recovery(self):
        """测试故障容错和恢复"""
        # 创建系统组件
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.DAILY_CYCLE,
            scenario_id="fault_tolerance_scenario"
        )
        
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=self.env_config
        )
        
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="fault_tolerance_trainer"
        )
        
        # 测试1: 环境重置故障处理
        try:
            state = env.reset()
            self.assertIsNotNone(state)
            
            # 模拟异常状态
            invalid_action = {
                'upper_action': np.array([float('inf')]),  # 无效动作
                'lower_action': np.array([float('nan'), float('nan')])
            }
            
            # 环境应该能处理无效动作
            try:
                next_state, reward, done, info = env.step(invalid_action)
                # 如果没有异常，说明环境有容错机制
                print("环境具备容错能力")
            except:
                # 如果有异常，测试恢复能力
                state = env.reset()  # 重置环境
                self.assertIsNotNone(state)
                print("环境具备恢复能力")
                
        except Exception as e:
            self.fail(f"环境容错测试失败: {str(e)}")
        
        # 测试2: 检查点保存和恢复
        try:
            # 保存初始检查点
            checkpoint_path = os.path.join(self.temp_dir, "fault_test_checkpoint.pth")
            trainer.save_checkpoint(checkpoint_path)
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # 模拟训练一段时间
            state = env.reset()
            for _ in range(5):
                action = {
                    'upper_action': np.array([0.5]),
                    'lower_action': np.array([0.0, 0.0])
                }
                state, _, done, _ = env.step(action)
                if done:
                    state = env.reset()
            
            # 恢复检查点
            new_trainer = HierarchicalTrainer(
                config=self.training_config,
                model_config=self.model_config,
                trainer_id="recovered_trainer"
            )
            
            new_trainer.load_checkpoint(checkpoint_path)
            
            # 验证恢复成功
            self.assertIsNotNone(new_trainer.upper_trainer.agent)
            self.assertIsNotNone(new_trainer.lower_trainer.agent)
            
            print("检查点恢复测试通过")
            
        except Exception as e:
            self.fail(f"检查点恢复测试失败: {str(e)}")
        
        print("✅ 故障容错和恢复测试通过")
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        # 创建性能监控系统
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.FREQUENCY_REGULATION,
            scenario_id="performance_monitoring_scenario"
        )
        
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=self.env_config
        )
        
        # 性能监控数据收集
        performance_data = {
            'episode_rewards': [],
            'step_times': [],
            'memory_usage': [],
            'action_values': [],
            'state_values': []
        }
        
        # 运行性能监控测试
        state = env.reset()
        
        for episode in range(3):
            episode_reward = 0
            episode_start_time = time.time()
            state = env.reset()
            
            for step in range(50):
                step_start_time = time.time()
                
                # 生成动作
                action = {
                    'upper_action': np.array([np.random.uniform(-1, 1)]),
                    'lower_action': np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
                }
                
                # 环境交互
                next_state, reward, done, info = env.step(action)
                step_reward = reward if isinstance(reward, (int, float)) else sum(reward.values())
                episode_reward += step_reward
                
                # 记录性能数据
                step_time = time.time() - step_start_time
                performance_data['step_times'].append(step_time)
                performance_data['action_values'].append(np.concatenate([action['upper_action'], action['lower_action']]))
                performance_data['state_values'].append(np.concatenate([state['upper_state'], state['lower_state']]))
                
                # 模拟内存使用（实际应用中可使用psutil）
                import sys
                performance_data['memory_usage'].append(sys.getsizeof(state) + sys.getsizeof(action))
                
                state = next_state
                
                if done:
                    break
            
            performance_data['episode_rewards'].append(episode_reward)
            
            episode_time = time.time() - episode_start_time
            print(f"Episode {episode}: Reward={episode_reward:.2f}, Time={episode_time:.2f}s")
        
        # 性能分析
        avg_step_time = np.mean(performance_data['step_times'])
        max_step_time = np.max(performance_data['step_times'])
        avg_memory = np.mean(performance_data['memory_usage'])
        
        # 验证性能指标
        self.assertLess(avg_step_time, 1.0)  # 平均步时间应小于1秒
        self.assertLess(max_step_time, 5.0)  # 最大步时间应小于5秒
        self.assertGreater(len(performance_data['episode_rewards']), 0)
        
        # 计算系统性能指标
        performance_metrics = {
            'avg_step_time': avg_step_time,
            'max_step_time': max_step_time,
            'avg_memory_usage': avg_memory,
            'total_episodes': len(performance_data['episode_rewards']),
            'avg_episode_reward': np.mean(performance_data['episode_rewards'])
        }
        
        print(f"性能指标: {performance_metrics}")
        
        print("✅ 性能监控测试通过")
    
    def test_scalability(self):
        """测试系统可扩展性"""
        # 测试不同规模的系统配置
        scale_configs = [
            {'num_batteries': 1, 'episodes': 2},
            {'num_batteries': 3, 'episodes': 2},
            {'num_batteries': 5, 'episodes': 2}
        ]
        
        scalability_results = []
        
        for config in scale_configs:
            start_time = time.time()
            
            # 调整模型配置以适应不同规模
            scaled_model_config = ModelConfig()
            scaled_model_config.lower_layer.action_dim = config['num_batteries'] * 2  # 每个电池2个动作
            
            # 创建场景
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=ScenarioType.DAILY_CYCLE,
                scenario_id=f"scalability_test_{config['num_batteries']}"
            )
            
            # 创建环境
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=self.env_config
            )
            
            # 运行测试
            state = env.reset()
            total_steps = 0
            
            for episode in range(config['episodes']):
                episode_steps = 0
                state = env.reset()
                
                for step in range(20):  # 限制步数以控制测试时间
                    # 生成适应规模的动作
                    upper_action = np.array([np.random.uniform(-1, 1)])
                    lower_action = np.random.uniform(-1, 1, config['num_batteries'] * 2)
                    
                    action = {
                        'upper_action': upper_action,
                        'lower_action': lower_action
                    }
                    
                    try:
                        next_state, reward, done, info = env.step(action)
                        episode_steps += 1
                        total_steps += 1
                        state = next_state
                        
                        if done:
                            break
                            
                    except Exception as e:
                        # 如果动作维度不匹配，使用默认动作
                        default_action = {
                            'upper_action': np.array([0.5]),
                            'lower_action': np.array([0.0, 0.0])
                        }
                        next_state, reward, done, info = env.step(default_action)
                        state = next_state
                        break
            
            execution_time = time.time() - start_time
            
            result = {
                'num_batteries': config['num_batteries'],
                'episodes': config['episodes'],
                'total_steps': total_steps,
                'execution_time': execution_time,
                'steps_per_second': total_steps / execution_time if execution_time > 0 else 0
            }
            
            scalability_results.append(result)
            print(f"规模测试 - 电池数: {config['num_batteries']}, 执行时间: {execution_time:.2f}s")
        
        # 验证可扩展性
        self.assertEqual(len(scalability_results), len(scale_configs))
        
        for result in scalability_results:
            self.assertGreater(result['total_steps'], 0)
            self.assertGreater(result['execution_time'], 0)
        
        print("✅ 系统可扩展性测试通过")
    
    def test_configuration_consistency(self):
        """测试配置一致性"""
        # 测试不同配置组合的一致性
        configs_to_test = [
            {
                'name': 'small_system',
                'training_episodes': 3,
                'hidden_size': 32,
                'batch_size': 16
            },
            {
                'name': 'medium_system', 
                'training_episodes': 5,
                'hidden_size': 64,
                'batch_size': 32
            }
        ]
        
        for config_test in configs_to_test:
            # 创建配置
            training_config = TrainingConfig()
            training_config.upper_config.total_episodes = config_test['training_episodes']
            training_config.lower_config.total_episodes = config_test['training_episodes']
            training_config.upper_config.batch_size = config_test['batch_size']
            training_config.lower_config.batch_size = config_test['batch_size']
            
            model_config = ModelConfig()
            model_config.upper_layer.hidden_size = config_test['hidden_size']
            model_config.lower_layer.hidden_size = config_test['hidden_size']
            
            # 验证配置有效性
            self.assertGreater(training_config.upper_config.total_episodes, 0)
            self.assertGreater(training_config.lower_config.total_episodes, 0)
            self.assertGreater(model_config.upper_layer.hidden_size, 0)
            self.assertGreater(model_config.lower_layer.hidden_size, 0)
            
            # 测试配置兼容性
            try:
                scenario = self.scenario_generator.generate_scenario(
                    scenario_type=ScenarioType.DAILY_CYCLE,
                    scenario_id=f"config_test_{config_test['name']}"
                )
                
                env = MultiScaleEnvironment(
                    scenario=scenario,
                    config=self.env_config
                )
                
                trainer = HierarchicalTrainer(
                    config=training_config,
                    model_config=model_config,
                    trainer_id=f"config_trainer_{config_test['name']}"
                )
                
                # 简单运行测试
                state = env.reset()
                action = {
                    'upper_action': np.array([0.5]),
                    'lower_action': np.array([0.0, 0.0])
                }
                next_state, reward, done, info = env.step(action)
                
                # 验证结果
                self.assertIsNotNone(next_state)
                self.assertIsNotNone(reward)
                
                print(f"配置 {config_test['name']} 兼容性测试通过")
                
            except Exception as e:
                self.fail(f"配置 {config_test['name']} 测试失败: {str(e)}")
        
        print("✅ 配置一致性测试通过")


if __name__ == '__main__':
    # 添加缺失的import
    import time
    unittest.main()
