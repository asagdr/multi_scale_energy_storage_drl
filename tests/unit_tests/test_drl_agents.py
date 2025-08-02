import unittest
import torch
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import UpperLayerConfig, LowerLayerConfig
from config.model_config import ModelConfig
from drl_agents.upper_layer.transformer_encoder import TransformerEncoder
from drl_agents.upper_layer.balance_analyzer import BalanceAnalyzer
from drl_agents.upper_layer.constraint_generator import ConstraintGenerator
from drl_agents.upper_layer.multi_objective_agent import MultiObjectiveAgent
from drl_agents.upper_layer.pareto_optimizer import ParetoOptimizer
from drl_agents.lower_layer.ddpg_agent import DDPGAgent
from drl_agents.lower_layer.power_tracker import PowerTracker
from drl_agents.lower_layer.constraint_handler import ConstraintHandler
from drl_agents.lower_layer.temperature_compensator import TemperatureCompensator
from drl_agents.lower_layer.response_optimizer import ResponseOptimizer
from drl_agents.communication.message_protocol import MessageProtocol, MessageType, Priority
from drl_agents.communication.information_flow import InformationFlow
from drl_agents.communication.data_exchange import DataExchange

class TestUpperLayerDRL(unittest.TestCase):
    """上层DRL组件测试"""
    
    def setUp(self):
        """测试初始化"""
        self.upper_config = UpperLayerConfig()
        self.model_config = ModelConfig()
        
    def test_transformer_encoder_initialization(self):
        """测试Transformer编码器初始化"""
        encoder = TransformerEncoder(
            config=self.upper_config,
            model_config=self.model_config,
            encoder_id="test_encoder"
        )
        
        self.assertEqual(encoder.encoder_id, "test_encoder")
        self.assertEqual(encoder.d_model, self.upper_config.hidden_dim)
        self.assertEqual(encoder.num_heads, self.upper_config.attention_heads)
        self.assertEqual(encoder.num_layers, self.upper_config.transformer_layers)
        
    def test_transformer_encoder_forward(self):
        """测试Transformer编码器前向传播"""
        encoder = TransformerEncoder(
            config=self.upper_config,
            model_config=self.model_config
        )
        
        # 测试输入
        batch_size = 2
        seq_len = 10
        input_dim = self.model_config.upper_state_dim
        
        test_input = torch.randn(batch_size, seq_len, input_dim)
        
        # 前向传播
        output = encoder.forward(test_input, return_attention=True)
        
        # 检查输出结构
        self.assertIn('encoded_sequence', output)
        self.assertIn('global_features', output)
        self.assertIn('balance_features', output)
        self.assertIn('temporal_features', output)
        self.assertIn('attention_weights', output)
        
        # 检查输出维度
        self.assertEqual(output['encoded_sequence'].shape, 
                        (batch_size, seq_len, encoder.d_model))
        self.assertEqual(output['global_features'].shape, 
                        (batch_size, encoder.d_model))
        
    def test_balance_analyzer_initialization(self):
        """测试均衡分析器初始化"""
        analyzer = BalanceAnalyzer(
            config=self.upper_config,
            model_config=self.model_config,
            analyzer_id="test_analyzer"
        )
        
        self.assertEqual(analyzer.analyzer_id, "test_analyzer")
        self.assertEqual(analyzer.input_dim, self.model_config.upper_state_dim)
        
    def test_balance_analyzer_forward(self):
        """测试均衡分析器前向传播"""
        analyzer = BalanceAnalyzer(
            config=self.upper_config,
            model_config=self.model_config
        )
        
        batch_size = 3
        state_dim = self.model_config.upper_state_dim
        test_state = torch.randn(batch_size, state_dim)
        
        # 前向分析
        result = analyzer.forward(test_state, return_detailed=True)
        
        # 检查输出
        self.assertIn('overall_balance_score', result)
        self.assertIn('balance_priorities', result)
        self.assertIn('soc_urgency', result)
        self.assertIn('thermal_urgency', result)
        self.assertIn('degradation_urgency', result)
        
        # 检查维度
        self.assertEqual(result['overall_balance_score'].shape, (batch_size,))
        self.assertEqual(result['balance_priorities'].shape, (batch_size, 3))
        
    def test_constraint_generator_initialization(self):
        """测试约束生成器初始化"""
        generator = ConstraintGenerator(
            config=self.upper_config,
            model_config=self.model_config,
            generator_id="test_generator"
        )
        
        self.assertEqual(generator.generator_id, "test_generator")
        
    def test_constraint_generator_forward(self):
        """测试约束生成器前向传播"""
        generator = ConstraintGenerator(
            config=self.upper_config,
            model_config=self.model_config
        )
        
        batch_size = 2
        state_dim = self.model_config.upper_state_dim
        test_state = torch.randn(batch_size, state_dim)
        
        # 生成约束矩阵
        constraint_matrix = generator.forward(test_state)
        
        # 检查约束矩阵
        self.assertIsNotNone(constraint_matrix.max_charge_power)
        self.assertIsNotNone(constraint_matrix.max_discharge_power)
        self.assertIsNotNone(constraint_matrix.max_temperature)
        
        # 检查维度
        self.assertEqual(constraint_matrix.max_charge_power.shape, (batch_size,))
        
    def test_multi_objective_agent_initialization(self):
        """测试多目标智能体初始化"""
        agent = MultiObjectiveAgent(
            config=self.upper_config,
            model_config=self.model_config,
            agent_id="test_agent"
        )
        
        self.assertEqual(agent.agent_id, "test_agent")
        self.assertEqual(agent.state_dim, self.model_config.upper_state_dim)
        self.assertEqual(agent.action_dim, self.model_config.upper_action_dim)
        self.assertEqual(agent.num_objectives, 4)
        
    def test_multi_objective_agent_action_selection(self):
        """测试多目标智能体动作选择"""
        agent = MultiObjectiveAgent(
            config=self.upper_config,
            model_config=self.model_config
        )
        
        batch_size = 1
        state_dim = self.model_config.upper_state_dim
        test_state = torch.randn(batch_size, state_dim)
        
        # 选择动作
        action = agent.select_action(test_state, add_noise=False)
        
        # 检查动作维度和范围
        self.assertEqual(action.shape, (batch_size, self.model_config.upper_action_dim))
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))
        
    def test_pareto_optimizer_initialization(self):
        """测试帕累托优化器初始化"""
        optimizer = ParetoOptimizer(
            config=self.upper_config,
            n_objectives=4,
            optimizer_id="test_optimizer"
        )
        
        self.assertEqual(optimizer.optimizer_id, "test_optimizer")
        self.assertEqual(optimizer.n_objectives, 4)
        
    def test_pareto_optimizer_add_solution(self):
        """测试帕累托优化器添加解"""
        optimizer = ParetoOptimizer(
            config=self.upper_config,
            n_objectives=4
        )
        
        # 添加测试解
        objectives = np.array([0.8, 0.7, 0.9, 0.6])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        action = np.array([0.1, 0.2, 0.3, 0.4])
        
        success = optimizer.add_solution(objectives, weights, action)
        
        self.assertTrue(success)
        self.assertEqual(len(optimizer.pareto_front.solutions), 1)

class TestLowerLayerDRL(unittest.TestCase):
    """下层DRL组件测试"""
    
    def setUp(self):
        """测试初始化"""
        self.lower_config = LowerLayerConfig()
        self.model_config = ModelConfig()
        
    def test_ddpg_agent_initialization(self):
        """测试DDPG智能体初始化"""
        agent = DDPGAgent(
            config=self.lower_config,
            model_config=self.model_config,
            agent_id="test_ddpg"
        )
        
        self.assertEqual(agent.agent_id, "test_ddpg")
        self.assertEqual(agent.state_dim, self.model_config.lower_state_dim)
        self.assertEqual(agent.action_dim, self.model_config.lower_action_dim)
        
    def test_ddpg_agent_action_selection(self):
        """测试DDPG智能体动作选择"""
        agent = DDPGAgent(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        batch_size = 1
        state_dim = self.model_config.lower_state_dim
        test_state = torch.randn(batch_size, state_dim)
        
        # 选择动作
        action = agent.select_action(test_state, add_noise=False)
        
        # 检查动作维度和范围
        self.assertEqual(action.shape, (batch_size, self.model_config.lower_action_dim))
        self.assertTrue(torch.all(action >= -agent.max_action))
        self.assertTrue(torch.all(action <= agent.max_action))
        
    def test_ddpg_agent_experience_replay(self):
        """测试DDPG智能体经验回放"""
        agent = DDPGAgent(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        # 添加经验
        state = torch.randn(1, self.model_config.lower_state_dim)
        action = torch.randn(1, self.model_config.lower_action_dim)
        reward = 1.0
        next_state = torch.randn(1, self.model_config.lower_state_dim)
        done = False
        
        agent.add_experience(state, action, reward, next_state, done)
        
        self.assertEqual(len(agent.replay_buffer), 1)
        
    def test_power_tracker_initialization(self):
        """测试功率跟踪器初始化"""
        tracker = PowerTracker(
            config=self.lower_config,
            model_config=self.model_config,
            tracker_id="test_tracker"
        )
        
        self.assertEqual(tracker.tracker_id, "test_tracker")
        self.assertEqual(tracker.dt, 0.01)  # 10ms
        
    def test_power_tracker_tracking(self):
        """测试功率跟踪控制"""
        tracker = PowerTracker(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        # 模拟跟踪场景
        power_reference = 10000.0  # 10kW
        current_power = 9500.0     # 9.5kW
        system_state = {
            'soc': 50.0,
            'temperature': 25.0,
            'voltage': 3.4,
            'current': 100.0
        }
        
        # 执行跟踪
        result = tracker.track_power(
            power_reference, current_power, system_state
        )
        
        # 检查结果
        self.assertIn('control_signal', result)
        self.assertIn('response_speed', result)
        self.assertIn('control_confidence', result)
        
    def test_constraint_handler_initialization(self):
        """测试约束处理器初始化"""
        handler = ConstraintHandler(
            config=self.lower_config,
            model_config=self.model_config,
            handler_id="test_handler"
        )
        
        self.assertEqual(handler.handler_id, "test_handler")
        self.assertGreater(len(handler.constraints), 0)
        
    def test_constraint_handler_constraint_handling(self):
        """测试约束处理"""
        handler = ConstraintHandler(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        # 测试动作
        action = torch.tensor([0.5, 0.3, -0.2])
        system_state = {
            'soc': 50.0,
            'temperature': 30.0,
            'voltage': 3.4,
            'current_power': 5000.0
        }
        
        # 处理约束
        result = handler.handle_constraints(action, system_state)
        
        # 检查结果
        self.assertIn('constrained_action', result)
        self.assertIn('constraint_penalty', result)
        self.assertIn('violations', result)
        
    def test_temperature_compensator_initialization(self):
        """测试温度补偿器初始化"""
        compensator = TemperatureCompensator(
            config=self.lower_config,
            model_config=self.model_config,
            compensator_id="test_compensator"
        )
        
        self.assertEqual(compensator.compensator_id, "test_compensator")
        self.assertEqual(compensator.num_cells, 10)
        
    def test_temperature_compensator_analysis(self):
        """测试温度补偿器分析"""
        compensator = TemperatureCompensator(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        # 模拟温度数据
        temperatures = np.array([25.0, 26.0, 27.0, 28.0, 29.0, 
                               30.0, 31.0, 32.0, 33.0, 34.0])
        
        # 分析温度分布
        profile = compensator.analyze_temperature_profile(temperatures)
        
        # 检查分析结果
        self.assertEqual(len(profile.temperatures), 10)
        self.assertGreater(profile.avg_temperature, 0)
        self.assertGreater(profile.max_temperature, profile.min_temperature)
        
    def test_response_optimizer_initialization(self):
        """测试响应优化器初始化"""
        optimizer = ResponseOptimizer(
            config=self.lower_config,
            model_config=self.model_config,
            optimizer_id="test_optimizer"
        )
        
        self.assertEqual(optimizer.optimizer_id, "test_optimizer")
        self.assertIsNotNone(optimizer.optimization_target)

class TestCommunicationSystem(unittest.TestCase):
    """通信系统测试"""
    
    def setUp(self):
        """测试初始化"""
        self.node_id = "test_node"
        
    def test_message_protocol_initialization(self):
        """测试消息协议初始化"""
        protocol = MessageProtocol(
            node_id=self.node_id,
            protocol_id="test_protocol"
        )
        
        self.assertEqual(protocol.node_id, self.node_id)
        self.assertEqual(protocol.protocol_id, "test_protocol")
        
    def test_message_protocol_send_receive(self):
        """测试消息协议收发"""
        protocol = MessageProtocol(node_id=self.node_id)
        
        # 发送消息
        success = protocol.send_message(
            message_type=MessageType.HEARTBEAT,
            payload={'status': 'active'},
            receiver_id="target_node",
            priority=Priority.NORMAL
        )
        
        self.assertTrue(success)
        self.assertGreater(protocol.outgoing_queue.size(), 0)
        
    def test_information_flow_initialization(self):
        """测试信息流管理器初始化"""
        protocol = MessageProtocol(node_id=self.node_id)
        flow = InformationFlow(
            flow_id="test_flow",
            message_protocol=protocol
        )
        
        self.assertEqual(flow.flow_id, "test_flow")
        self.assertEqual(flow.message_protocol, protocol)
        
    def test_information_flow_constraint_matrix_sending(self):
        """测试信息流约束矩阵发送"""
        protocol = MessageProtocol(node_id=self.node_id)
        flow = InformationFlow(flow_id="test_flow", message_protocol=protocol)
        
        # 发送约束矩阵
        constraint_matrix = torch.randn(3, 4)
        success = flow.send_constraint_matrix(
            constraint_matrix=constraint_matrix,
            target_layer="lower_layer",
            priority=Priority.HIGH
        )
        
        self.assertTrue(success)
        
    def test_data_exchange_initialization(self):
        """测试数据交换器初始化"""
        protocol = MessageProtocol(node_id=self.node_id)
        flow = InformationFlow(flow_id="test_flow", message_protocol=protocol)
        exchange = DataExchange(
            exchange_id="test_exchange",
            message_protocol=protocol,
            information_flow=flow
        )
        
        self.assertEqual(exchange.exchange_id, "test_exchange")
        
    def test_data_exchange_constraint_matrix_exchange(self):
        """测试数据交换器约束矩阵交换"""
        protocol = MessageProtocol(node_id=self.node_id)
        flow = InformationFlow(flow_id="test_flow", message_protocol=protocol)
        exchange = DataExchange(
            exchange_id="test_exchange",
            message_protocol=protocol,
            information_flow=flow
        )
        
        # 交换约束矩阵
        constraint_matrix = torch.randn(2, 3)
        transaction_id = exchange.exchange_constraint_matrix(
            constraint_matrix=constraint_matrix,
            target_node="target_node"
        )
        
        self.assertIsNotNone(transaction_id)
        self.assertNotEqual(transaction_id, "")

class TestIntegrationScenarios(unittest.TestCase):
    """集成场景测试"""
    
    def setUp(self):
        """测试初始化"""
        self.upper_config = UpperLayerConfig()
        self.lower_config = LowerLayerConfig()
        self.model_config = ModelConfig()
        
    def test_upper_lower_communication(self):
        """测试上下层通信"""
        # 初始化上层智能体
        upper_agent = MultiObjectiveAgent(
            config=self.upper_config,
            model_config=self.model_config,
            agent_id="upper_test"
        )
        
        # 初始化下层智能体
        lower_agent = DDPGAgent(
            config=self.lower_config,
            model_config=self.model_config,
            agent_id="lower_test"
        )
        
        # 初始化通信系统
        protocol = MessageProtocol(node_id="integration_test")
        flow = InformationFlow(flow_id="integration_flow", message_protocol=protocol)
        
        # 模拟上层决策
        state = torch.randn(1, self.model_config.upper_state_dim)
        system_state = {'soc': 50.0, 'temperature': 25.0}
        
        upper_decision = upper_agent.generate_high_level_decision(state, system_state)
        
        # 检查上层决策
        self.assertIn('constraint_matrix', upper_decision)
        self.assertIn('objective_weights', upper_decision)
        
        # 模拟下层控制
        lower_state = torch.randn(1, self.model_config.lower_state_dim)
        lower_action = lower_agent.select_action(lower_state, add_noise=False)
        
        # 检查下层动作
        self.assertEqual(lower_action.shape, (1, self.model_config.lower_action_dim))
        
    def test_end_to_end_scenario(self):
        """测试端到端场景"""
        # 创建完整的系统组件
        upper_agent = MultiObjectiveAgent(
            config=self.upper_config,
            model_config=self.model_config
        )
        
        lower_agent = DDPGAgent(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        power_tracker = PowerTracker(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        constraint_handler = ConstraintHandler(
            config=self.lower_config,
            model_config=self.model_config
        )
        
        # 模拟一个完整的控制周期
        upper_state = torch.randn(1, self.model_config.upper_state_dim)
        system_state = {
            'soc': 50.0,
            'temperature': 25.0,
            'voltage': 3.4,
            'current_power': 5000.0
        }
        
        # 1. 上层决策
        upper_decision = upper_agent.generate_high_level_decision(upper_state, system_state)
        
        # 2. 下层控制
        lower_state = torch.randn(1, self.model_config.lower_state_dim)
        lower_action = lower_agent.select_action(lower_state)
        
        # 3. 约束处理
        constraint_result = constraint_handler.handle_constraints(
            lower_action, system_state
        )
        
        # 4. 功率跟踪
        power_reference = 8000.0
        current_power = 7500.0
        tracking_result = power_tracker.track_power(
            power_reference, current_power, system_state
        )
        
        # 验证结果
        self.assertIn('balance_score', upper_decision)
        self.assertIn('constrained_action', constraint_result)
        self.assertIn('control_signal', tracking_result)
        
        print("✅ 端到端场景测试通过")

class TestModelPersistence(unittest.TestCase):
    """模型持久化测试"""
    
    def setUp(self):
        """测试初始化"""
        self.upper_config = UpperLayerConfig()
        self.lower_config = LowerLayerConfig()
        self.model_config = ModelConfig()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """测试清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_upper_agent_save_load(self):
        """测试上层智能体保存加载"""
        agent = MultiObjectiveAgent(
            config=self.upper_config,
            model_config=self.model_config,
            agent_id="save_test"
        )
        
        # 保存检查点
        save_path = os.path.join(self.temp_dir, "upper_checkpoint.pth")
        success = agent.save_checkpoint(save_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(save_path))
        
        # 创建新的智能体并加载
        new_agent = MultiObjectiveAgent(
            config=self.upper_config,
            model_config=self.model_config,
            agent_id="load_test"
        )
        
        success = new_agent.load_checkpoint(save_path)
        self.assertTrue(success)
        
    def test_lower_agent_save_load(self):
        """测试下层智能体保存加载"""
        agent = DDPGAgent(
            config=self.lower_config,
            model_config=self.model_config,
            agent_id="save_test"
        )
        
        # 保存检查点
        save_path = os.path.join(self.temp_dir, "lower_checkpoint.pth")
        success = agent.save_checkpoint(save_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(save_path))
        
        # 创建新的智能体并加载
        new_agent = DDPGAgent(
            config=self.lower_config,
            model_config=self.model_config,
            agent_id="load_test"
        )
        
        success = new_agent.load_checkpoint(save_path)
        self.assertTrue(success)

if __name__ == '__main__':
    # 设置测试环境
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.makeSuite(TestUpperLayerDRL))
    test_suite.addTest(unittest.makeSuite(TestLowerLayerDRL))
    test_suite.addTest(unittest.makeSuite(TestCommunicationSystem))
    test_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))
    test_suite.addTest(unittest.makeSuite(TestModelPersistence))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    print(f"\n{'='*60}")
    print(f"DRL智能体测试完成")
    print(f"运行测试: {result.testsRun}")
    print(f"失败测试: {len(result.failures)}")
    print(f"错误测试: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
