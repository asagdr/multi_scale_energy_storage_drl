import unittest
import numpy as np
import tempfile
import shutil
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from experiments.basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType
from experiments.ablation_studies import AblationStudy, AblationConfig, AblationComponent
from experiments.sensitivity_analysis import SensitivityAnalysis, SensitivityConfig, ParameterRange, ParameterType
from experiments.case_studies.peak_shaving import PeakShavingExperiment, PeakShavingConfig, PeakShavingScenario
from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from data_processing.scenario_generator import ScenarioType

class TestExperimentWorkflow(unittest.TestCase):
    """实验工作流集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # 创建基础配置
        self.training_config = TrainingConfig()
        self.model_config = ModelConfig()
        
        # 简化配置以加速测试
        self.training_config.upper_config.total_episodes = 5
        self.training_config.lower_config.total_episodes = 5
    
    def test_basic_experiment_workflow(self):
        """测试基础实验工作流"""
        # 创建实验设置
        experiment_settings = ExperimentSettings(
            experiment_name="test_basic_experiment",
            experiment_type=ExperimentType.HIERARCHICAL,
            description="基础实验工作流测试",
            total_episodes=5,
            evaluation_frequency=2,
            save_frequency=3,
            scenario_types=[ScenarioType.DAILY_CYCLE],
            environment_variations=2,
            use_pretraining=False,  # 禁用预训练以加速测试
            enable_hierarchical=True,
            evaluation_episodes=2,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        # 创建并运行实验
        experiment = BasicExperiment(
            settings=experiment_settings,
            experiment_id="test_basic_exp_001"
        )
        
        results = experiment.run_experiment()
        
        # 验证实验结果
        self.assertIsNotNone(results)
        self.assertEqual(results.experiment_id, "test_basic_exp_001")
        self.assertEqual(results.settings.experiment_name, "test_basic_experiment")
        self.assertTrue(results.total_time > 0)
        
        # 验证训练指标
        self.assertIsInstance(results.training_metrics, dict)
        self.assertGreater(len(results.training_metrics), 0)
        
        print("✅ 基础实验工作流测试通过")
    
    def test_ablation_study_workflow(self):
        """测试消融实验工作流"""
        # 创建基线配置
        baseline_config = ExperimentSettings(
            experiment_name="ablation_baseline",
            experiment_type=ExperimentType.HIERARCHICAL,
            total_episodes=3,
            scenario_types=[ScenarioType.DAILY_CYCLE],
            environment_variations=1,
            use_pretraining=False,
            enable_hierarchical=True,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        # 创建消融配置
        ablation_config = AblationConfig(
            study_name="test_ablation_study",
            description="消融实验工作流测试",
            components_to_ablate=[
                AblationComponent.HIERARCHICAL_STRUCTURE,
                AblationComponent.PRETRAINING
            ],
            baseline_config=baseline_config,
            num_repetitions=1,  # 减少重复次数以加速测试
            combination_ablation=False,
            primary_metrics=['episode_reward', 'tracking_accuracy']
        )
        
        # 运行消融实验
        ablation_study = AblationStudy(ablation_config)
        results = ablation_study.run_study()
        
        # 验证消融结果
        self.assertIsNotNone(results)
        self.assertIsInstance(results, dict)
        
        # 验证基线结果
        self.assertIsNotNone(ablation_study.baseline_result)
        
        # 验证消融配置结果
        expected_configs = ['ablate_hierarchical_structure', 'ablate_pretraining']
        for config_name in expected_configs:
            self.assertIn(config_name, results)
        
        print("✅ 消融实验工作流测试通过")
    
    def test_sensitivity_analysis_workflow(self):
        """测试敏感性分析工作流"""
        # 创建基线配置
        baseline_config = ExperimentSettings(
            experiment_name="sensitivity_baseline",
            experiment_type=ExperimentType.HIERARCHICAL,
            total_episodes=3,
            scenario_types=[ScenarioType.DAILY_CYCLE],
            environment_variations=1,
            use_pretraining=False,
            enable_hierarchical=True,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        # 定义参数范围
        parameter_ranges = [
            ParameterRange(
                param_type=ParameterType.LEARNING_RATE,
                min_value=0.0001,
                max_value=0.01,
                num_samples=3,
                scale="log"
            ),
            ParameterRange(
                param_type=ParameterType.BATCH_SIZE,
                min_value=16,
                max_value=64,
                num_samples=3,
                scale="linear"
            )
        ]
        
        # 创建敏感性分析配置
        sensitivity_config = SensitivityConfig(
            study_name="test_sensitivity_analysis",
            description="敏感性分析工作流测试",
            parameters_to_analyze=parameter_ranges,
            baseline_config=baseline_config,
            analysis_type="one_at_a_time",
            num_repetitions=1,
            primary_metrics=['episode_reward', 'tracking_accuracy'],
            sensitivity_methods=['local_sensitivity']
        )
        
        # 运行敏感性分析
        sensitivity_analysis = SensitivityAnalysis(sensitivity_config)
        results = sensitivity_analysis.run_analysis()
        
        # 验证分析结果
        self.assertIsNotNone(results)
        self.assertIn('study_info', results)
        self.assertIn('parameter_sensitivity', results)
        
        # 验证参数敏感性结果
        for param_range in parameter_ranges:
            param_name = param_range.param_type.value
            self.assertIn(param_name, results['parameter_sensitivity'])
        
        print("✅ 敏感性分析工作流测试通过")
    
    def test_case_study_workflow(self):
        """测试案例研究工作流"""
        # 创建削峰填谷配置
        peak_shaving_config = PeakShavingConfig(
            scenario_type=PeakShavingScenario.COMMERCIAL_BUILDING,
            base_load_kw=100.0,
            peak_load_kw=200.0,
            load_profile_days=2,  # 减少天数以加速测试
            battery_capacity_kwh=100.0,
            max_power_kw=50.0,
            round_trip_efficiency=0.9,
            peak_price=1.0,
            valley_price=0.4,
            normal_price=0.7,
            demand_charge=50.0,
            target_peak_reduction=0.2
        )
        
        # 运行削峰填谷案例研究
        peak_shaving_experiment = PeakShavingExperiment(
            config=peak_shaving_config,
            experiment_id="test_peak_shaving_001"
        )
        
        results = peak_shaving_experiment.run_case_study()
        
        # 验证案例研究结果
        self.assertIsNotNone(results)
        self.assertEqual(results.experiment_id, "test_peak_shaving_001")
        self.assertEqual(results.config.scenario_type, PeakShavingScenario.COMMERCIAL_BUILDING)
        
        # 验证关键指标
        self.assertGreaterEqual(results.original_peak_load, 0)
        self.assertGreaterEqual(results.reduced_peak_load, 0)
        self.assertGreaterEqual(results.peak_reduction_ratio, 0)
        
        # 验证时间序列数据
        self.assertGreater(len(results.load_profile), 0)
        self.assertGreater(len(results.battery_soc), 0)
        
        print("✅ 案例研究工作流测试通过")
    
    def test_experiment_reproducibility(self):
        """测试实验可重现性"""
        # 创建相同的实验设置
        experiment_settings = ExperimentSettings(
            experiment_name="reproducibility_test",
            experiment_type=ExperimentType.HIERARCHICAL,
            total_episodes=3,
            scenario_types=[ScenarioType.DAILY_CYCLE],
            environment_variations=1,
            use_pretraining=False,
            enable_hierarchical=True,
            enable_visualization=False,
            device="cpu",
            random_seed=123  # 固定随机种子
        )
        
        # 运行第一次实验
        experiment1 = BasicExperiment(
            settings=experiment_settings,
            experiment_id="repro_test_001"
        )
        results1 = experiment1.run_experiment()
        
        # 运行第二次实验（相同设置）
        experiment2 = BasicExperiment(
            settings=experiment_settings,
            experiment_id="repro_test_002"
        )
        results2 = experiment2.run_experiment()
        
        # 验证结果的一致性（在合理误差范围内）
        self.assertEqual(len(results1.training_metrics), len(results2.training_metrics))
        
        # 检查关键指标的一致性
        for metric_name in results1.training_metrics:
            if metric_name in results2.training_metrics:
                values1 = results1.training_metrics[metric_name]
                values2 = results2.training_metrics[metric_name]
                
                # 由于随机性，检查趋势而不是精确值
                if len(values1) > 1 and len(values2) > 1:
                    trend1 = values1[-1] - values1[0]
                    trend2 = values2[-1] - values2[0]
                    
                    # 趋势方向应该一致（符号相同）
                    if abs(trend1) > 1e-6 and abs(trend2) > 1e-6:
                        self.assertEqual(np.sign(trend1), np.sign(trend2),
                                       f"指标 {metric_name} 的趋势不一致")
        
        print("✅ 实验可重现性测试通过")
    
    def test_experiment_state_management(self):
        """测试实验状态管理"""
        # 创建实验
        experiment_settings = ExperimentSettings(
            experiment_name="state_management_test",
            experiment_type=ExperimentType.HIERARCHICAL,
            total_episodes=5,
            scenario_types=[ScenarioType.DAILY_CYCLE],
            environment_variations=1,
            use_pretraining=False,
            enable_hierarchical=True,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        experiment = BasicExperiment(
            settings=experiment_settings,
            experiment_id="state_mgmt_test_001"
        )
        
        # 测试进度跟踪
        initial_progress = experiment.get_progress()
        self.assertEqual(initial_progress['current_episode'], 0)
        self.assertFalse(initial_progress['is_running'])
        self.assertFalse(initial_progress['is_completed'])
        
        # 保存初始状态
        state_file = os.path.join(self.temp_dir, "experiment_state.json")
        experiment.save_experiment_state(state_file)
        self.assertTrue(os.path.exists(state_file))
        
        # 加载状态
        new_experiment = BasicExperiment(
            settings=experiment_settings,
            experiment_id="state_mgmt_test_002"
        )
        new_experiment.load_experiment_state(state_file)
        
        # 验证状态一致性
        self.assertEqual(new_experiment.experiment_id, experiment.experiment_id)
        self.assertEqual(new_experiment.current_episode, experiment.current_episode)
        
        print("✅ 实验状态管理测试通过")


class TestWorkflowIntegration(unittest.TestCase):
    """工作流集成测试"""
    
    def test_multi_experiment_workflow(self):
        """测试多实验工作流"""
        # 创建多个实验类型
        experiments = []
        
        # 1. 基础实验
        basic_settings = ExperimentSettings(
            experiment_name="multi_workflow_basic",
            experiment_type=ExperimentType.HIERARCHICAL,
            total_episodes=2,
            scenario_types=[ScenarioType.DAILY_CYCLE],
            environment_variations=1,
            use_pretraining=False,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        basic_experiment = BasicExperiment(
            settings=basic_settings,
            experiment_id="multi_basic_001"
        )
        experiments.append(('basic', basic_experiment))
        
        # 2. 案例研究
        case_config = PeakShavingConfig(
            scenario_type=PeakShavingScenario.COMMERCIAL_BUILDING,
            load_profile_days=1,  # 最小化测试时间
            battery_capacity_kwh=50.0,
            max_power_kw=25.0
        )
        
        case_experiment = PeakShavingExperiment(
            config=case_config,
            experiment_id="multi_case_001"
        )
        experiments.append(('case_study', case_experiment))
        
        # 执行所有实验
        results = {}
        for exp_type, experiment in experiments:
            if exp_type == 'basic':
                result = experiment.run_experiment()
            elif exp_type == 'case_study':
                result = experiment.run_case_study()
            
            results[exp_type] = result
            print(f"✅ {exp_type} 实验完成")
        
        # 验证所有实验都成功完成
        self.assertEqual(len(results), len(experiments))
        for exp_type, result in results.items():
            self.assertIsNotNone(result)
        
        print("✅ 多实验工作流测试通过")


if __name__ == '__main__':
    unittest.main()
