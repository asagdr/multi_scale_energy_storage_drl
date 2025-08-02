import unittest
import numpy as np
import torch
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from training.hierarchical_trainer import HierarchicalTrainer
from training.pretraining_pipeline import PretrainingPipeline
from training.evaluation_suite import EvaluationSuite
from environment.multi_scale_env import MultiScaleEnvironment
from data_processing.scenario_generator import ScenarioGenerator, ScenarioType
from utils.logger import Logger

class TestTrainingPipeline(unittest.TestCase):
    """训练流水线集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # 创建测试配置
        self.training_config = TrainingConfig()
        self.training_config.upper_config.total_episodes = 10  # 减少测试时间
        self.training_config.lower_config.total_episodes = 10
        
        self.model_config = ModelConfig()
        self.model_config.upper_layer.hidden_size = 64  # 减小模型以加速测试
        self.model_config.lower_layer.hidden_size = 64
        
        # 初始化组件
        self.scenario_generator = ScenarioGenerator()
        self.logger = Logger("TestTrainingPipeline")
        
    def test_complete_training_pipeline(self):
        """测试完整训练流水线"""
        # 1. 创建分层训练器
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="test_trainer"
        )
        
        # 2. 生成测试场景
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.DAILY_CYCLE,
            scenario_id="test_scenario"
        )
        
        # 3. 创建环境
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=self.training_config.environment_config
        )
        
        # 4. 执行训练
        training_results = trainer.train(
            environment=env,
            num_episodes=5  # 少量回合用于测试
        )
        
        # 验证训练结果
        self.assertIsNotNone(training_results)
        self.assertIn('training_history', training_results)
        self.assertIn('final_metrics', training_results)
        self.assertGreater(len(training_results['training_history']), 0)
        
        # 5. 验证模型保存
        self.assertTrue(hasattr(trainer.upper_trainer, 'agent'))
        self.assertTrue(hasattr(trainer.lower_trainer, 'agent'))
        
        print("✅ 完整训练流水线测试通过")
    
    def test_pretraining_integration(self):
        """测试预训练集成"""
        # 创建预训练流水线
        pretraining_pipeline = PretrainingPipeline(
            config=self.training_config,
            model_config=self.model_config,
            pipeline_id="test_pretraining"
        )
        
        # 执行预训练
        pretraining_results = pretraining_pipeline.run_pretraining()
        
        # 验证预训练结果
        self.assertIsNotNone(pretraining_results)
        self.assertIn('stage_results', pretraining_results)
        self.assertIn('transfer_results', pretraining_results)
        
        # 验证各阶段结果
        for stage_name, stage_result in pretraining_results['stage_results'].items():
            self.assertIsNotNone(stage_result)
            print(f"预训练阶段 {stage_name} 完成")
        
        print("✅ 预训练集成测试通过")
    
    def test_evaluation_integration(self):
        """测试评估集成"""
        # 创建训练器
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="test_eval_trainer"
        )
        
        # 创建评估套件
        evaluator = EvaluationSuite(
            config=self.training_config,
            model_config=self.model_config,
            suite_id="test_evaluator"
        )
        
        # 生成测试场景
        scenarios = []
        for i in range(3):
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=ScenarioType.DAILY_CYCLE,
                scenario_id=f"eval_scenario_{i}"
            )
            scenarios.append(scenario)
        
        # 模拟训练后的智能体
        with patch.object(trainer.upper_trainer, 'agent') as mock_upper_agent, \
             patch.object(trainer.lower_trainer, 'agent') as mock_lower_agent:
            
            # 配置模拟智能体
            mock_upper_agent.select_action.return_value = np.array([0.5])
            mock_lower_agent.select_action.return_value = np.array([0.0, 0.0])
            
            # 执行评估
            evaluation_results = evaluator.comprehensive_evaluation(
                agents={'upper': mock_upper_agent, 'lower': mock_lower_agent},
                scenarios=scenarios
            )
            
            # 验证评估结果
            self.assertIsNotNone(evaluation_results)
            self.assertIn('performance_metrics', evaluation_results)
            self.assertIn('detailed_results', evaluation_results)
            
        print("✅ 评估集成测试通过")
    
    def test_training_data_consistency(self):
        """测试训练数据一致性"""
        # 创建训练器
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="test_consistency_trainer"
        )
        
        # 生成多个场景
        scenarios = []
        for scenario_type in [ScenarioType.DAILY_CYCLE, ScenarioType.PEAK_SHAVING]:
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=scenario_type,
                scenario_id=f"consistency_test_{scenario_type.value}"
            )
            scenarios.append(scenario)
        
        # 测试每个场景的数据一致性
        for scenario in scenarios:
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=self.training_config.environment_config
            )
            
            # 验证环境初始化
            state = env.reset()
            self.assertIsNotNone(state)
            self.assertIn('upper_state', state)
            self.assertIn('lower_state', state)
            
            # 验证状态维度
            upper_state_dim = self.model_config.upper_layer.state_dim
            lower_state_dim = self.model_config.lower_layer.state_dim
            
            self.assertEqual(len(state['upper_state']), upper_state_dim)
            self.assertEqual(len(state['lower_state']), lower_state_dim)
            
            # 测试步进
            action = {
                'upper_action': np.array([0.5]),
                'lower_action': np.array([0.0, 0.0])
            }
            
            next_state, reward, done, info = env.step(action)
            
            # 验证返回值
            self.assertIsNotNone(next_state)
            self.assertIsInstance(reward, (int, float, dict))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
        
        print("✅ 训练数据一致性测试通过")
    
    def test_model_checkpoint_integration(self):
        """测试模型检查点集成"""
        # 创建训练器
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="test_checkpoint_trainer"
        )
        
        # 生成场景
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.DAILY_CYCLE,
            scenario_id="checkpoint_test_scenario"
        )
        
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=self.training_config.environment_config
        )
        
        # 短期训练
        training_results = trainer.train(
            environment=env,
            num_episodes=3
        )
        
        # 保存检查点
        checkpoint_path = os.path.join(self.temp_dir, "test_checkpoint.pth")
        trainer.save_checkpoint(checkpoint_path)
        
        # 验证检查点文件存在
        self.assertTrue(os.path.exists(checkpoint_path))
        
        # 创建新训练器并加载检查点
        new_trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="test_loaded_trainer"
        )
        
        new_trainer.load_checkpoint(checkpoint_path)
        
        # 验证加载成功
        self.assertIsNotNone(new_trainer.upper_trainer.agent)
        self.assertIsNotNone(new_trainer.lower_trainer.agent)
        
        print("✅ 模型检查点集成测试通过")
    
    def test_multi_environment_training(self):
        """测试多环境训练"""
        # 创建训练器
        trainer = HierarchicalTrainer(
            config=self.training_config,
            model_config=self.model_config,
            trainer_id="test_multi_env_trainer"
        )
        
        # 创建多个不同的环境
        environments = []
        scenario_types = [ScenarioType.DAILY_CYCLE, ScenarioType.PEAK_SHAVING, ScenarioType.FREQUENCY_REGULATION]
        
        for i, scenario_type in enumerate(scenario_types):
            scenario = self.scenario_generator.generate_scenario(
                scenario_type=scenario_type,
                scenario_id=f"multi_env_scenario_{i}"
            )
            
            env = MultiScaleEnvironment(
                scenario=scenario,
                config=self.training_config.environment_config
            )
            environments.append(env)
        
        # 在每个环境中进行少量训练
        all_results = []
        for i, env in enumerate(environments):
            training_results = trainer.train(
                environment=env,
                num_episodes=2
            )
            all_results.append(training_results)
            print(f"环境 {i+1} 训练完成")
        
        # 验证所有环境都成功训练
        self.assertEqual(len(all_results), len(environments))
        for result in all_results:
            self.assertIsNotNone(result)
            self.assertIn('training_history', result)
        
        print("✅ 多环境训练集成测试通过")


class TestCommunicationIntegration(unittest.TestCase):
    """通信集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()
    
    def test_hierarchical_communication(self):
        """测试分层通信"""
        from drl_agents.communication.hierarchical_communication import HierarchicalCommunication
        
        # 创建通信模块
        comm = HierarchicalCommunication()
        
        # 模拟上层决策
        upper_decision = {
            'power_allocation': np.array([100.0, 50.0, 25.0]),
            'priority_weights': np.array([0.6, 0.3, 0.1]),
            'constraints': {'max_power': 200.0, 'min_soc': 0.2}
        }
        
        # 发送上层决策
        comm.send_upper_decision(upper_decision)
        
        # 接收上层决策
        received_decision = comm.receive_upper_decision()
        
        # 验证通信
        self.assertIsNotNone(received_decision)
        self.assertIn('power_allocation', received_decision)
        np.testing.assert_array_equal(
            received_decision['power_allocation'], 
            upper_decision['power_allocation']
        )
        
        # 模拟下层反馈
        lower_feedback = {
            'execution_status': np.array([1.0, 0.8, 0.9]),
            'actual_power': np.array([98.0, 45.0, 23.0]),
            'system_state': {'avg_soc': 0.65, 'temperature': 25.0}
        }
        
        # 发送下层反馈
        comm.send_lower_feedback(lower_feedback)
        
        # 接收下层反馈
        received_feedback = comm.receive_lower_feedback()
        
        # 验证反馈
        self.assertIsNotNone(received_feedback)
        self.assertIn('execution_status', received_feedback)
        np.testing.assert_array_equal(
            received_feedback['actual_power'],
            lower_feedback['actual_power']
        )
        
        print("✅ 分层通信集成测试通过")


if __name__ == '__main__':
    unittest.main()
