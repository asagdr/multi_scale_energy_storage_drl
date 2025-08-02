import unittest
import numpy as np
import tempfile
import shutil
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from data_processing.scenario_generator import ScenarioGenerator, ScenarioType
from data_processing.load_profile_generator import LoadProfileGenerator, LoadPattern
from data_processing.weather_simulator import WeatherSimulator, ClimateZone
from data_processing.feature_extractor import FeatureExtractor, FeatureType
from data_processing.data_preprocessor import DataPreprocessor, PreprocessingMethod
from environment.multi_scale_env import MultiScaleEnvironment
from config.environment_config import EnvironmentConfig

class TestDataFlow(unittest.TestCase):
    """数据流集成测试"""
    
    def setUp(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        
        # 初始化数据处理组件
        self.scenario_generator = ScenarioGenerator()
        self.load_generator = LoadProfileGenerator()
        self.weather_simulator = WeatherSimulator()
        self.feature_extractor = FeatureExtractor()
        self.data_preprocessor = DataPreprocessor()
    
    def test_complete_data_pipeline(self):
        """测试完整数据流水线"""
        # 1. 生成场景数据
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.DAILY_CYCLE,
            scenario_id="data_flow_test"
        )
        
        self.assertIsNotNone(scenario)
        self.assertEqual(scenario.scenario_id, "data_flow_test")
        
        # 2. 生成负荷数据
        load_profile = self.load_generator.generate_load_profile(
            load_pattern=LoadPattern.COMMERCIAL,
            duration_hours=24,
            time_resolution_minutes=15
        )
        
        self.assertIsNotNone(load_profile)
        self.assertGreater(len(load_profile.load_values), 0)
        
        # 3. 生成天气数据
        weather_data = self.weather_simulator.simulate_weather(
            climate_zone=ClimateZone.TEMPERATE,
            duration_hours=24,
            time_resolution_hours=1
        )
        
        self.assertIsNotNone(weather_data)
        self.assertGreater(len(weather_data.temperature), 0)
        
        # 4. 数据预处理
        raw_data = {
            'load': load_profile.load_values[:24],  # 取24个小时数据
            'temperature': weather_data.temperature[:24],
            'humidity': weather_data.humidity[:24]
        }
        
        preprocessed_data = self.data_preprocessor.preprocess_data(
            data=raw_data,
            methods=[PreprocessingMethod.NORMALIZATION, PreprocessingMethod.FILTERING]
        )
        
        self.assertIsNotNone(preprocessed_data)
        self.assertEqual(len(preprocessed_data.processed_data), len(raw_data))
        
        # 5. 特征提取
        features = self.feature_extractor.extract_features(
            data=preprocessed_data.processed_data,
            feature_types=[FeatureType.TEMPORAL, FeatureType.STATISTICAL]
        )
        
        self.assertIsNotNone(features)
        self.assertGreater(len(features.temporal_features), 0)
        self.assertGreater(len(features.statistical_features), 0)
        
        # 6. 环境集成
        env_config = EnvironmentConfig()
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=env_config
        )
        
        # 验证环境可以正常初始化和运行
        state = env.reset()
        self.assertIsNotNone(state)
        
        action = {
            'upper_action': np.array([0.5]),
            'lower_action': np.array([0.0, 0.0])
        }
        
        next_state, reward, done, info = env.step(action)
        self.assertIsNotNone(next_state)
        
        print("✅ 完整数据流水线测试通过")
    
    def test_data_consistency_across_components(self):
        """测试组件间数据一致性"""
        # 生成同一场景的多种数据
        scenario_id = "consistency_test"
        
        # 场景数据
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.PEAK_SHAVING,
            scenario_id=scenario_id
        )
        
        # 负荷数据
        load_profile = self.load_generator.generate_load_profile(
            load_pattern=LoadPattern.INDUSTRIAL,
            duration_hours=48,  # 2天
            time_resolution_minutes=60  # 1小时分辨率
        )
        
        # 天气数据
        weather_data = self.weather_simulator.simulate_weather(
            climate_zone=ClimateZone.SUBTROPICAL,
            duration_hours=48,
            time_resolution_hours=1
        )
        
        # 验证时间维度一致性
        self.assertEqual(len(load_profile.timestamps), len(load_profile.load_values))
        self.assertEqual(len(weather_data.timestamps), len(weather_data.temperature))
        
        # 验证数据范围合理性
        self.assertTrue(np.all(load_profile.load_values >= 0))
        self.assertTrue(np.all(weather_data.temperature > -50))
        self.assertTrue(np.all(weather_data.temperature < 60))
        self.assertTrue(np.all(weather_data.humidity >= 0))
        self.assertTrue(np.all(weather_data.humidity <= 100))
        
        print("✅ 组件间数据一致性测试通过")
    
    def test_data_format_compatibility(self):
        """测试数据格式兼容性"""
        # 生成测试数据
        load_profile = self.load_generator.generate_load_profile(
            load_pattern=LoadPattern.RESIDENTIAL,
            duration_hours=24,
            time_resolution_minutes=15
        )
        
        weather_data = self.weather_simulator.simulate_weather(
            climate_zone=ClimateZone.TROPICAL,
            duration_hours=24,
            time_resolution_hours=0.25  # 15分钟分辨率
        )
        
        # 测试数据格式
        self.assertIsInstance(load_profile.load_values, np.ndarray)
        self.assertIsInstance(load_profile.timestamps, np.ndarray)
        self.assertIsInstance(weather_data.temperature, np.ndarray)
        self.assertIsInstance(weather_data.humidity, np.ndarray)
        
        # 测试数据可以合并
        combined_data = {
            'load_values': load_profile.load_values,
            'temperature': weather_data.temperature,
            'humidity': weather_data.humidity,
            'solar_irradiance': weather_data.solar_irradiance
        }
        
        # 验证特征提取器可以处理合并数据
        features = self.feature_extractor.extract_features(
            data=combined_data,
            feature_types=[FeatureType.TEMPORAL, FeatureType.CORRELATION]
        )
        
        self.assertIsNotNone(features)
        self.assertGreater(len(features.temporal_features), 0)
        
        print("✅ 数据格式兼容性测试通过")
    
    def test_data_preprocessing_pipeline(self):
        """测试数据预处理流水线"""
        # 生成包含噪声和异常值的测试数据
        base_data = np.random.normal(100, 20, 1000)
        
        # 添加异常值
        outlier_indices = np.random.choice(1000, 50, replace=False)
        base_data[outlier_indices] += np.random.normal(0, 100, 50)
        
        # 添加缺失值
        missing_indices = np.random.choice(1000, 20, replace=False)
        base_data[missing_indices] = np.nan
        
        test_data = {
            'signal_1': base_data,
            'signal_2': base_data * 0.8 + np.random.normal(0, 5, 1000),
            'signal_3': np.sin(np.linspace(0, 10*np.pi, 1000)) * 20 + base_data * 0.1
        }
        
        # 应用完整预处理流水线
        preprocessing_methods = [
            PreprocessingMethod.IMPUTATION,
            PreprocessingMethod.OUTLIER_REMOVAL,
            PreprocessingMethod.NORMALIZATION,
            PreprocessingMethod.FILTERING,
            PreprocessingMethod.DENOISING
        ]
        
        preprocessed_data = self.data_preprocessor.preprocess_data(
            data=test_data,
            methods=preprocessing_methods
        )
        
        # 验证预处理效果
        self.assertIsNotNone(preprocessed_data)
        
        for signal_name, processed_signal in preprocessed_data.processed_data.items():
            # 验证无缺失值
            self.assertFalse(np.any(np.isnan(processed_signal)))
            
            # 验证无无穷值
            self.assertTrue(np.all(np.isfinite(processed_signal)))
            
            # 验证数据长度保持不变
            self.assertEqual(len(processed_signal), len(test_data[signal_name]))
        
        # 验证数据质量改善
        for signal_name in test_data.keys():
            if signal_name in preprocessed_data.quality_improvement:
                improvement = preprocessed_data.quality_improvement[signal_name]
                self.assertGreaterEqual(improvement, 0)  # 质量应该改善或保持不变
        
        print("✅ 数据预处理流水线测试通过")
    
    def test_feature_extraction_pipeline(self):
        """测试特征提取流水线"""
        # 生成多种类型的测试信号
        time_length = 1000
        t = np.linspace(0, 10, time_length)
        
        test_signals = {
            'periodic_signal': np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t),
            'trend_signal': 0.1 * t + np.random.normal(0, 0.1, time_length),
            'noise_signal': np.random.normal(0, 1, time_length),
            'step_signal': np.concatenate([np.zeros(500), np.ones(500)]),
            'power_signal': 100 + 50 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5, time_length)
        }
        
        # 提取所有类型特征
        all_feature_types = [
            FeatureType.TEMPORAL,
            FeatureType.FREQUENCY,
            FeatureType.STATISTICAL,
            FeatureType.PATTERN,
            FeatureType.CORRELATION,
            FeatureType.TREND
        ]
        
        features = self.feature_extractor.extract_features(
            data=test_signals,
            feature_types=all_feature_types
        )
        
        # 验证特征提取结果
        self.assertIsNotNone(features)
        
        # 验证每种类型的特征都被提取
        self.assertGreater(len(features.temporal_features), 0)
        self.assertGreater(len(features.frequency_features), 0)
        self.assertGreater(len(features.statistical_features), 0)
        self.assertGreater(len(features.pattern_features), 0)
        self.assertGreater(len(features.correlation_features), 0)
        self.assertGreater(len(features.trend_features), 0)
        
        # 验证特征重要性计算
        self.assertGreater(len(features.feature_importance), 0)
        self.assertGreater(features.quality_score, 0)
        
        # 测试特征分析
        importance_analysis = self.feature_extractor.analyze_feature_importance(features)
        self.assertIsNotNone(importance_analysis)
        self.assertIn('top_features', importance_analysis)
        
        print("✅ 特征提取流水线测试通过")
    
    def test_data_export_import(self):
        """测试数据导出导入"""
        # 生成测试数据
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.FREQUENCY_REGULATION,
            scenario_id="export_import_test"
        )
        
        weather_data = self.weather_simulator.simulate_weather(
            climate_zone=ClimateZone.DESERT,
            duration_hours=12,
            time_resolution_hours=1
        )
        
        # 测试天气数据导出
        export_path = os.path.join(self.temp_dir, "weather_export.json")
        self.weather_simulator.export_weather_data(weather_data, export_path, format='json')
        self.assertTrue(os.path.exists(export_path))
        
        # 测试CSV导出
        csv_export_path = os.path.join(self.temp_dir, "weather_export.csv")
        self.weather_simulator.export_weather_data(weather_data, csv_export_path, format='csv')
        self.assertTrue(os.path.exists(csv_export_path))
        
        # 验证导出文件不为空
        self.assertGreater(os.path.getsize(export_path), 0)
        self.assertGreater(os.path.getsize(csv_export_path), 0)
        
        # 测试特征数据导出
        test_data = {
            'signal_1': np.random.normal(0, 1, 100),
            'signal_2': np.random.normal(5, 2, 100)
        }
        
        features = self.feature_extractor.extract_features(
            data=test_data,
            feature_types=[FeatureType.TEMPORAL, FeatureType.STATISTICAL]
        )
        
        features_export_path = os.path.join(self.temp_dir, "features_export.json")
        self.feature_extractor.export_features(features, features_export_path, format='json')
        self.assertTrue(os.path.exists(features_export_path))
        
        print("✅ 数据导出导入测试通过")
    
    def test_real_time_data_flow(self):
        """测试实时数据流"""
        # 模拟实时数据生成和处理
        scenario = self.scenario_generator.generate_scenario(
            scenario_type=ScenarioType.REAL_TIME_CONTROL,
            scenario_id="realtime_test"
        )
        
        env_config = EnvironmentConfig()
        env = MultiScaleEnvironment(
            scenario=scenario,
            config=env_config
        )
        
        # 模拟实时数据流
        state = env.reset()
        data_history = []
        
        for step in range(50):  # 模拟50个时间步
            # 生成动作
            action = {
                'upper_action': np.array([np.random.uniform(-1, 1)]),
                'lower_action': np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
            }
            
            # 环境步进
            next_state, reward, done, info = env.step(action)
            
            # 收集数据
            step_data = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'info': info,
                'step': step
            }
            data_history.append(step_data)
            
            state = next_state
            
            if done:
                state = env.reset()
        
        # 验证数据流完整性
        self.assertEqual(len(data_history), 50)
        
        for i, step_data in enumerate(data_history):
            self.assertEqual(step_data['step'], i)
            self.assertIsNotNone(step_data['state'])
            self.assertIsNotNone(step_data['action'])
            self.assertIsNotNone(step_data['next_state'])
        
        print("✅ 实时数据流测试通过")


if __name__ == '__main__':
    unittest.main()
