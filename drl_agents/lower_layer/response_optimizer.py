import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import LowerLayerConfig
from config.model_config import ModelConfig

@dataclass
class ResponseMetrics:
    """响应性能指标"""
    response_time: float = 0.0          # 响应时间 (s)
    settling_time: float = 0.0          # 稳定时间 (s)
    overshoot: float = 0.0              # 超调量 (%)
    steady_state_error: float = 0.0     # 稳态误差 (%)
    rise_time: float = 0.0              # 上升时间 (s)
    bandwidth: float = 0.0              # 带宽 (Hz)
    phase_margin: float = 0.0           # 相位裕度 (度)
    gain_margin: float = 0.0            # 增益裕度 (dB)

@dataclass
class OptimizationTarget:
    """优化目标"""
    target_response_time: float = 0.01  # 目标响应时间 (s)
    target_overshoot: float = 5.0       # 目标超调量 (%)
    target_settling_time: float = 0.05  # 目标稳定时间 (s)
    target_bandwidth: float = 100.0     # 目标带宽 (Hz)
    stability_requirement: float = 0.8  # 稳定性要求 [0,1]

class ResponsePredictor(nn.Module):
    """响应性能预测器"""
    
    def __init__(self, input_dim: int = 15, hidden_dim: int = 64):
        super(ResponsePredictor, self).__init__()
        
        # 响应时间预测
        self.response_time_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1，表示归一化的响应时间
        )
        
        # 稳定性预测
        self.stability_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # [overshoot, settling_time, steady_state_error]
        )
        
        # 性能综合评估
        self.performance_evaluator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 综合性能评分 [0,1]
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        response_time = self.response_time_predictor(x) * 0.1  # 0-100ms
        stability_metrics = self.stability_predictor(x)
        performance_score = self.performance_evaluator(x)
        
        return {
            'predicted_response_time': response_time,
            'predicted_overshoot': torch.abs(stability_metrics[:, 0:1]) * 20,  # 0-20%
            'predicted_settling_time': torch.sigmoid(stability_metrics[:, 1:2]) * 0.2,  # 0-200ms
            'predicted_steady_error': torch.sigmoid(stability_metrics[:, 2:3]) * 5,  # 0-5%
            'performance_score': performance_score
        }

class AdaptiveController:
    """自适应控制器参数调节"""
    
    def __init__(self):
        # 控制器参数范围
        self.param_ranges = {
            'kp': (0.1, 10.0),
            'ki': (0.01, 2.0),
            'kd': (0.001, 0.5),
            'filter_freq': (10.0, 1000.0),    # Hz
            'damping_ratio': (0.3, 1.5)
        }
        
        # 当前参数
        self.current_params = {
            'kp': 2.0,
            'ki': 0.5,
            'kd': 0.1,
            'filter_freq': 100.0,
            'damping_ratio': 0.707
        }
        
        # 参数调整历史
        self.param_history = deque(maxlen=100)
        
        # 性能历史
        self.performance_history = deque(maxlen=100)
    
    def adapt_parameters(self, 
                        current_metrics: ResponseMetrics,
                        target: OptimizationTarget) -> Dict[str, float]:
        """自适应调整控制器参数"""
        # 计算性能偏差
        response_error = current_metrics.response_time - target.target_response_time
        overshoot_error = current_metrics.overshoot - target.target_overshoot
        settling_error = current_metrics.settling_time - target.target_settling_time
        
        # 参数调整策略
        param_adjustments = {}
        
        # 比例增益调整
        if abs(response_error) > 0.01:  # 响应时间误差超过10ms
            if response_error > 0:  # 响应太慢
                param_adjustments['kp'] = min(self.param_ranges['kp'][1], 
                                            self.current_params['kp'] * 1.1)
            else:  # 响应太快，可能不稳定
                param_adjustments['kp'] = max(self.param_ranges['kp'][0], 
                                            self.current_params['kp'] * 0.95)
        
        # 积分增益调整
        if abs(current_metrics.steady_state_error) > 1.0:  # 稳态误差超过1%
            param_adjustments['ki'] = min(self.param_ranges['ki'][1], 
                                        self.current_params['ki'] * 1.05)
        elif abs(current_metrics.steady_state_error) < 0.1:  # 稳态误差很小
            param_adjustments['ki'] = max(self.param_ranges['ki'][0], 
                                        self.current_params['ki'] * 0.98)
        
        # 微分增益调整
        if overshoot_error > 2.0:  # 超调过大
            param_adjustments['kd'] = min(self.param_ranges['kd'][1], 
                                        self.current_params['kd'] * 1.1)
        elif overshoot_error < -2.0:  # 超调过小，可能响应慢
            param_adjustments['kd'] = max(self.param_ranges['kd'][0], 
                                        self.current_params['kd'] * 0.95)
        
        # 更新参数
        for param, new_value in param_adjustments.items():
            self.current_params[param] = new_value
        
        # 记录历史
        self.param_history.append(self.current_params.copy())
        self.performance_history.append({
            'response_time': current_metrics.response_time,
            'overshoot': current_metrics.overshoot,
            'settling_time': current_metrics.settling_time,
            'steady_state_error': current_metrics.steady_state_error
        })
        
        return self.current_params.copy()
    
    def get_optimization_trend(self) -> str:
        """获取优化趋势"""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent_performance = list(self.performance_history)[-10:]
        
        # 计算响应时间趋势
        response_times = [p['response_time'] for p in recent_performance]
        response_trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
        
        # 计算超调量趋势
        overshoots = [p['overshoot'] for p in recent_performance]
        overshoot_trend = np.polyfit(range(len(overshoots)), overshoots, 1)[0]
        
        if response_trend < -0.001 and overshoot_trend < 0.5:
            return "improving"
        elif response_trend > 0.001 or overshoot_trend > 1.0:
            return "degrading"
        else:
            return "stable"

class ResponseOptimizer(nn.Module):
    """
    响应优化器
    优化控制系统的动态响应性能：响应时间、超调量、稳定性
    """
    
    def __init__(self,
                 config: LowerLayerConfig,
                 model_config: ModelConfig,
                 optimizer_id: str = "ResponseOptimizer_001"):
        """
        初始化响应优化器
        
        Args:
            config: 下层配置
            model_config: 模型配置
            optimizer_id: 优化器ID
        """
        super(ResponseOptimizer, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.optimizer_id = optimizer_id
        
        # === 神经网络预测器 ===
        self.response_predictor = ResponsePredictor(
            input_dim=15,  # 系统状态 + 控制参数
            hidden_dim=64
        )
        
        # === 自适应控制器 ===
        self.adaptive_controller = AdaptiveController()
        
        # === 优化目标 ===
        self.optimization_target = OptimizationTarget()
        
        # === 响应特性分析 ===
        self.step_response_data = deque(maxlen=1000)
        self.frequency_response_data = deque(maxlen=100)
        
        # === 性能基准 ===
        self.performance_baseline = {
            'response_time': 0.05,      # 50ms基准
            'overshoot': 10.0,          # 10%基准
            'settling_time': 0.1,       # 100ms基准
            'bandwidth': 50.0           # 50Hz基准
        }
        
        # === 优化历史 ===
        self.optimization_history: List[Dict] = []
        self.performance_improvements = []
        
        # === 统计信息 ===
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.response_measurements = 0
        
        print(f"✅ 响应优化器初始化完成: {optimizer_id}")
        print(f"   目标响应时间: {self.optimization_target.target_response_time*1000:.1f}ms")
        print(f"   目标超调量: {self.optimization_target.target_overshoot:.1f}%")
    
    def measure_step_response(self, 
                            input_signal: np.ndarray,
                            output_signal: np.ndarray,
                            time_vector: np.ndarray) -> ResponseMetrics:
        """
        测量阶跃响应特性
        
        Args:
            input_signal: 输入信号
            output_signal: 输出信号
            time_vector: 时间向量
            
        Returns:
            响应性能指标
        """
        self.response_measurements += 1
        
        # 查找阶跃开始点
        step_start_idx = self._find_step_start(input_signal)
        if step_start_idx == -1:
            return ResponseMetrics()  # 返回默认值
        
        # 提取阶跃响应部分
        step_input = input_signal[step_start_idx:]
        step_output = output_signal[step_start_idx:]
        step_time = time_vector[step_start_idx:] - time_vector[step_start_idx]
        
        # 计算稳态值
        steady_state_value = np.mean(step_output[-50:]) if len(step_output) > 50 else step_output[-1]
        initial_value = step_output[0]
        step_magnitude = steady_state_value - initial_value
        
        if abs(step_magnitude) < 1e-6:
            return ResponseMetrics()
        
        # === 1. 响应时间（10%-90%上升时间） ===
        response_time = self._calculate_response_time(step_output, step_time, 
                                                    initial_value, steady_state_value)
        
        # === 2. 上升时间（0%-100%） ===
        rise_time = self._calculate_rise_time(step_output, step_time, 
                                            initial_value, steady_state_value)
        
        # === 3. 超调量 ===
        overshoot = self._calculate_overshoot(step_output, steady_state_value, step_magnitude)
        
        # === 4. 稳定时间（±2%误差带） ===
        settling_time = self._calculate_settling_time(step_output, step_time, 
                                                    steady_state_value, step_magnitude)
        
        # === 5. 稳态误差 ===
        target_value = np.mean(step_input[-50:]) if len(step_input) > 50 else step_input[-1]
        steady_state_error = abs(steady_state_value - target_value) / abs(target_value) * 100.0 if target_value != 0 else 0.0
        
        # === 6. 频域特性（简化估算） ===
        bandwidth = self._estimate_bandwidth(step_output, step_time)
        
        metrics = ResponseMetrics(
            response_time=response_time,
            rise_time=rise_time,
            overshoot=overshoot,
            settling_time=settling_time,
            steady_state_error=steady_state_error,
            bandwidth=bandwidth,
            phase_margin=60.0,  # 简化假设
            gain_margin=10.0    # 简化假设
        )
        
        # 记录数据
        self.step_response_data.append({
            'time': step_time.copy(),
            'output': step_output.copy(),
            'metrics': metrics,
            'timestamp': time.time()
        })
        
        return metrics
    
    def predict_response_performance(self, 
                                   system_state: Dict[str, Any],
                                   control_params: Dict[str, float]) -> Dict[str, float]:
        """
        预测响应性能
        
        Args:
            system_state: 系统状态
            control_params: 控制参数
            
        Returns:
            预测的性能指标
        """
        # 准备输入特征
        input_features = self._prepare_prediction_input(system_state, control_params)
        
        # 神经网络预测
        self.response_predictor.eval()
        with torch.no_grad():
            predictions = self.response_predictor(input_features)
        
        # 解析预测结果
        predicted_performance = {
            'response_time': predictions['predicted_response_time'].item(),
            'overshoot': predictions['predicted_overshoot'].item(),
            'settling_time': predictions['predicted_settling_time'].item(),
            'steady_state_error': predictions['predicted_steady_error'].item(),
            'performance_score': predictions['performance_score'].item()
        }
        
        return predicted_performance
    
    def optimize_response(self, 
                         current_metrics: ResponseMetrics,
                         system_state: Dict[str, Any],
                         constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        优化响应性能
        
        Args:
            current_metrics: 当前性能指标
            system_state: 系统状态
            constraints: 优化约束
            
        Returns:
            优化结果
        """
        self.total_optimizations += 1
        
        # === 1. 性能评估 ===
        performance_score = self._evaluate_performance(current_metrics)
        
        # === 2. 自适应参数调整 ===
        optimized_params = self.adaptive_controller.adapt_parameters(
            current_metrics, self.optimization_target
        )
        
        # === 3. 预测优化效果 ===
        predicted_performance = self.predict_response_performance(
            system_state, optimized_params
        )
        
        # === 4. 约束检查 ===
        if constraints:
            optimized_params = self._apply_optimization_constraints(optimized_params, constraints)
        
        # === 5. 优化验证 ===
        optimization_success = self._validate_optimization(
            current_metrics, predicted_performance
        )
        
        if optimization_success:
            self.successful_optimizations += 1
        
        # === 6. 记录优化历史 ===
        optimization_record = {
            'timestamp': time.time(),
            'current_performance': {
                'response_time': current_metrics.response_time,
                'overshoot': current_metrics.overshoot,
                'settling_time': current_metrics.settling_time,
                'performance_score': performance_score
            },
            'optimized_params': optimized_params.copy(),
            'predicted_performance': predicted_performance.copy(),
            'optimization_success': optimization_success,
            'improvement_ratio': self._calculate_improvement_ratio(current_metrics, predicted_performance)
        }
        
        self.optimization_history.append(optimization_record)
        
        # 维护历史长度
        if len(self.optimization_history) > 1000:
            self.optimization_history.pop(0)
        
        # === 7. 构建结果 ===
        result = {
            'optimized_parameters': optimized_params,
            'predicted_performance': predicted_performance,
            'current_performance_score': performance_score,
            'optimization_success': optimization_success,
            'performance_improvement': optimization_record['improvement_ratio'],
            'optimization_trend': self.adaptive_controller.get_optimization_trend(),
            'recommendations': self._generate_optimization_recommendations(current_metrics)
        }
        
        return result
    
    def _find_step_start(self, input_signal: np.ndarray) -> int:
        """查找阶跃开始点"""
        # 简化实现：查找信号变化最大的点
        if len(input_signal) < 10:
            return -1
        
        signal_diff = np.diff(input_signal)
        max_change_idx = np.argmax(np.abs(signal_diff))
        
        # 验证是否为有效阶跃
        if abs(signal_diff[max_change_idx]) > 0.1 * np.std(input_signal):
            return max_change_idx
        else:
            return 0  # 默认从开始
    
    def _calculate_response_time(self, 
                               output: np.ndarray, 
                               time: np.ndarray,
                               initial_value: float, 
                               final_value: float) -> float:
        """计算10%-90%响应时间"""
        step_magnitude = final_value - initial_value
        if abs(step_magnitude) < 1e-6:
            return 0.0
        
        # 10%和90%阈值
        threshold_10 = initial_value + 0.1 * step_magnitude
        threshold_90 = initial_value + 0.9 * step_magnitude
        
        # 查找交越点
        time_10 = None
        time_90 = None
        
        for i, val in enumerate(output):
            if time_10 is None and ((step_magnitude > 0 and val >= threshold_10) or 
                                   (step_magnitude < 0 and val <= threshold_10)):
                time_10 = time[i]
            
            if time_90 is None and ((step_magnitude > 0 and val >= threshold_90) or 
                                   (step_magnitude < 0 and val <= threshold_90)):
                time_90 = time[i]
                break
        
        if time_10 is not None and time_90 is not None:
            return time_90 - time_10
        else:
            return time[-1] if len(time) > 0 else 0.0
    
    def _calculate_rise_time(self, 
                           output: np.ndarray, 
                           time: np.ndarray,
                           initial_value: float, 
                           final_value: float) -> float:
        """计算上升时间"""
        step_magnitude = final_value - initial_value
        if abs(step_magnitude) < 1e-6:
            return 0.0
        
        threshold_100 = initial_value + 0.95 * step_magnitude  # 95%阈值
        
        for i, val in enumerate(output):
            if ((step_magnitude > 0 and val >= threshold_100) or 
                (step_magnitude < 0 and val <= threshold_100)):
                return time[i] - time[0]
        
        return time[-1] - time[0] if len(time) > 1 else 0.0
    
    def _calculate_overshoot(self, 
                           output: np.ndarray, 
                           steady_state: float, 
                           step_magnitude: float) -> float:
        """计算超调量"""
        if abs(step_magnitude) < 1e-6:
            return 0.0
        
        if step_magnitude > 0:
            max_value = np.max(output)
            overshoot = (max_value - steady_state) / step_magnitude * 100.0
        else:
            min_value = np.min(output)
            overshoot = (steady_state - min_value) / abs(step_magnitude) * 100.0
        
        return max(0.0, overshoot)
    
    def _calculate_settling_time(self, 
                               output: np.ndarray, 
                               time: np.ndarray,
                               steady_state: float, 
                               step_magnitude: float) -> float:
        """计算±2%稳定时间"""
        if abs(step_magnitude) < 1e-6 or len(output) < 10:
            return 0.0
        
        tolerance = 0.02 * abs(step_magnitude)  # ±2%误差带
        
        # 从后往前搜索，找到最后一次超出误差带的时间
        for i in range(len(output) - 1, -1, -1):
            if abs(output[i] - steady_state) > tolerance:
                if i < len(time) - 1:
                    return time[i + 1] - time[0]
                else:
                    return time[-1] - time[0]
        
        return time[min(10, len(time) - 1)] - time[0]  # 最少10个采样点
    
    def _estimate_bandwidth(self, output: np.ndarray, time: np.ndarray) -> float:
        """估算带宽"""
        if len(output) < 10 or len(time) < 10:
            return 0.0
        
        # 简化方法：基于响应速度估算
        dt = time[1] - time[0] if len(time) > 1 else 0.01
        
        # 计算信号变化率
        signal_diff = np.diff(output)
        max_rate = np.max(np.abs(signal_diff)) / dt
        
        # 估算带宽（经验公式）
        bandwidth = max_rate / (2 * np.pi * np.std(output)) if np.std(output) > 1e-6 else 0.0
        
        return min(1000.0, max(1.0, bandwidth))  # 限制在合理范围
    
    def _prepare_prediction_input(self, 
                                system_state: Dict[str, Any],
                                control_params: Dict[str, float]) -> torch.Tensor:
        """准备预测输入特征"""
        features = []
        
        # 系统状态特征
        features.extend([
            system_state.get('soc', 50.0) / 100.0,
            system_state.get('temperature', 25.0) / 60.0,
            system_state.get('voltage', 3.4) / 4.2,
            system_state.get('current', 0.0) / 200.0,
            system_state.get('power', 0.0) / 50000.0,
            system_state.get('load_disturbance', 0.0) / 1000.0
        ])
        
        # 控制参数特征
        features.extend([
            control_params.get('kp', 1.0) / 10.0,
            control_params.get('ki', 0.1) / 2.0,
            control_params.get('kd', 0.01) / 0.5,
            control_params.get('filter_freq', 100.0) / 1000.0,
            control_params.get('damping_ratio', 0.707) / 2.0
        ])
        
        # 历史性能特征
        if len(self.step_response_data) > 0:
            recent_data = list(self.step_response_data)[-5:]
            avg_response_time = np.mean([data['metrics'].response_time for data in recent_data])
            avg_overshoot = np.mean([data['metrics'].overshoot for data in recent_data])
            features.extend([
                avg_response_time / 0.1,
                avg_overshoot / 20.0
            ])
        else:
            features.extend([0.5, 0.5])
        
        # 优化趋势特征
        trend = self.adaptive_controller.get_optimization_trend()
        trend_encoding = {'improving': 1.0, 'stable': 0.5, 'degrading': 0.0, 'insufficient_data': 0.5}
        features.append(trend_encoding.get(trend, 0.5))
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _evaluate_performance(self, metrics: ResponseMetrics) -> float:
        """评估性能得分"""
        # 响应时间得分
        response_score = max(0.0, 1.0 - metrics.response_time / 0.1)  # 100ms为0分
        
        # 超调量得分
        overshoot_score = max(0.0, 1.0 - metrics.overshoot / 20.0)  # 20%为0分
        
        # 稳定时间得分
        settling_score = max(0.0, 1.0 - metrics.settling_time / 0.2)  # 200ms为0分
        
        # 稳态误差得分
        error_score = max(0.0, 1.0 - metrics.steady_state_error / 5.0)  # 5%为0分
        
        # 加权综合得分
        total_score = (0.3 * response_score + 0.3 * overshoot_score + 
                      0.25 * settling_score + 0.15 * error_score)
        
        return total_score
    
    def _apply_optimization_constraints(self, 
                                      params: Dict[str, float],
                                      constraints: Dict[str, float]) -> Dict[str, float]:
        """应用优化约束"""
        constrained_params = params.copy()
        
        # 响应时间约束
        max_response_time = constraints.get('max_response_time', 0.1)
        if params.get('kp', 0) < 0.5:  # Kp太小可能导致响应慢
            constrained_params['kp'] = max(0.5, params['kp'])
        
        # 稳定性约束
        max_kp = constraints.get('max_kp', 10.0)
        constrained_params['kp'] = min(max_kp, constrained_params['kp'])
        
        # 积分饱和约束
        max_ki = constraints.get('max_ki', 2.0)
        constrained_params['ki'] = min(max_ki, constrained_params['ki'])
        
        # 噪声敏感性约束
        max_kd = constraints.get('max_kd', 0.5)
        constrained_params['kd'] = min(max_kd, constrained_params['kd'])
        
        return constrained_params
    
    def _validate_optimization(self, 
                             current_metrics: ResponseMetrics,
                             predicted_performance: Dict[str, float]) -> bool:
        """验证优化效果"""
        # 性能改善验证
        current_score = self._evaluate_performance(current_metrics)
        predicted_score = predicted_performance.get('performance_score', 0.0)
        
        # 响应时间改善
        response_improvement = current_metrics.response_time - predicted_performance.get('response_time', current_metrics.response_time)
        
        # 超调量改善
        overshoot_improvement = current_metrics.overshoot - predicted_performance.get('overshoot', current_metrics.overshoot)
        
        # 验证条件
        conditions = [
            predicted_score > current_score + 0.05,  # 性能得分至少提升5%
            response_improvement > -0.01,             # 响应时间不能恶化超过10ms
            overshoot_improvement > -2.0,             # 超调量不能恶化超过2%
            predicted_performance.get('response_time', 0.1) < 0.1,  # 响应时间在100ms内
            predicted_performance.get('overshoot', 20.0) < 15.0     # 超调量在15%内
        ]
        
        return all(conditions)
    
    def _calculate_improvement_ratio(self, 
                                   current_metrics: ResponseMetrics,
                                   predicted_performance: Dict[str, float]) -> float:
        """计算改善比例"""
        current_score = self._evaluate_performance(current_metrics)
        predicted_score = predicted_performance.get('performance_score', current_score)
        
        if current_score > 0:
            improvement = (predicted_score - current_score) / current_score
        else:
            improvement = predicted_score
        
        return improvement
    
    def _generate_optimization_recommendations(self, metrics: ResponseMetrics) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 响应时间建议
        if metrics.response_time > self.optimization_target.target_response_time * 2:
            recommendations.append("增加比例增益以提高响应速度")
        elif metrics.response_time < self.optimization_target.target_response_time * 0.5:
            recommendations.append("适度降低比例增益以避免过快响应")
        
        # 超调量建议
        if metrics.overshoot > self.optimization_target.target_overshoot * 1.5:
            recommendations.append("增加微分增益或降低比例增益以减少超调")
        elif metrics.overshoot < self.optimization_target.target_overshoot * 0.3:
            recommendations.append("可适度增加比例增益以改善响应")
        
        # 稳定时间建议
        if metrics.settling_time > self.optimization_target.target_settling_time * 2:
            recommendations.append("调整阻尼比或增加带宽以加快稳定")
        
        # 稳态误差建议
        if metrics.steady_state_error > 2.0:
            recommendations.append("增加积分增益以减少稳态误差")
        
        # 系统稳定性建议
        if metrics.phase_margin < 30.0:
            recommendations.append("降低增益或增加相位补偿以提高稳定裕度")
        
        if not recommendations:
            recommendations.append("当前性能良好，维持现有参数")
        
        return recommendations
    
    def tune_controller_online(self, 
                             performance_data: List[ResponseMetrics],
                             adaptation_rate: float = 0.1) -> Dict[str, float]:
        """在线控制器调节"""
        if len(performance_data) < 5:
            return self.adaptive_controller.current_params.copy()
        
        # 计算性能趋势
        recent_scores = [self._evaluate_performance(metrics) for metrics in performance_data[-10:]]
        performance_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
        
        # 计算平均性能指标
        avg_response_time = np.mean([m.response_time for m in performance_data[-5:]])
        avg_overshoot = np.mean([m.overshoot for m in performance_data[-5:]])
        avg_settling_time = np.mean([m.settling_time for m in performance_data[-5:]])
        
        # 自适应调节
        current_params = self.adaptive_controller.current_params.copy()
        
        if performance_trend < -0.05:  # 性能下降
            # 更保守的调节
            if avg_overshoot > self.optimization_target.target_overshoot:
                current_params['kp'] *= (1.0 - adaptation_rate)
                current_params['kd'] *= (1.0 + adaptation_rate)
            
            if avg_response_time > self.optimization_target.target_response_time:
                current_params['kp'] *= (1.0 + adaptation_rate * 0.5)
        
        elif performance_trend > 0.05:  # 性能提升
            # 继续优化
            if avg_response_time > self.optimization_target.target_response_time:
                current_params['kp'] *= (1.0 + adaptation_rate)
            
            if avg_settling_time > self.optimization_target.target_settling_time:
                current_params['ki'] *= (1.0 + adaptation_rate * 0.5)
        
        # 应用参数范围限制
        for param, value in current_params.items():
            if param in self.adaptive_controller.param_ranges:
                min_val, max_val = self.adaptive_controller.param_ranges[param]
                current_params[param] = np.clip(value, min_val, max_val)
        
        # 更新控制器参数
        self.adaptive_controller.current_params = current_params
        
        return current_params
    
    def analyze_frequency_response(self, 
                                 frequencies: np.ndarray,
                                 magnitude_response: np.ndarray,
                                 phase_response: np.ndarray) -> Dict[str, float]:
        """分析频率响应特性"""
        try:
            # 计算带宽（-3dB点）
            mag_db = 20 * np.log10(np.abs(magnitude_response) + 1e-10)
            dc_gain_db = mag_db[0]
            bandwidth_idx = np.where(mag_db <= dc_gain_db - 3.0)[0]
            bandwidth = frequencies[bandwidth_idx[0]] if len(bandwidth_idx) > 0 else frequencies[-1]
            
            # 计算相位裕度
            # 查找增益交越频率（|H(jw)| = 1）
            gain_crossover_idx = np.argmin(np.abs(magnitude_response - 1.0))
            phase_margin = 180.0 + phase_response[gain_crossover_idx]
            
            # 计算增益裕度
            # 查找相位交越频率（phase = -180°）
            phase_crossover_idx = np.argmin(np.abs(phase_response + 180.0))
            gain_margin_db = -mag_db[phase_crossover_idx]
            
            # 估算阻尼比
            # 基于频率响应的峰值
            peak_magnitude = np.max(magnitude_response)
            if peak_magnitude > 1.0:
                damping_ratio = 1.0 / (2.0 * peak_magnitude)
            else:
                damping_ratio = 0.707  # 默认值
            
            frequency_analysis = {
                'bandwidth': bandwidth,
                'phase_margin': phase_margin,
                'gain_margin': gain_margin_db,
                'damping_ratio': damping_ratio,
                'peak_magnitude': peak_magnitude,
                'dc_gain': magnitude_response[0],
                'stability_measure': min(phase_margin / 60.0, gain_margin_db / 10.0)  # 归一化稳定性指标
            }
            
            # 记录频率响应数据
            self.frequency_response_data.append({
                'frequencies': frequencies.copy(),
                'magnitude': magnitude_response.copy(),
                'phase': phase_response.copy(),
                'analysis': frequency_analysis,
                'timestamp': time.time()
            })
            
            return frequency_analysis
            
        except Exception as e:
            print(f"⚠️ 频率响应分析失败: {str(e)}")
            return {
                'bandwidth': 50.0,
                'phase_margin': 45.0,
                'gain_margin': 10.0,
                'damping_ratio': 0.707,
                'peak_magnitude': 1.0,
                'dc_gain': 1.0,
                'stability_measure': 0.75
            }
    
    def generate_optimal_trajectory(self, 
                                  start_state: np.ndarray,
                                  target_state: np.ndarray,
                                  time_horizon: float) -> Dict[str, np.ndarray]:
        """生成最优轨迹"""
        # 简化的轨迹生成（实际应用中可使用MPC或轨迹优化）
        num_points = int(time_horizon / 0.01)  # 10ms采样
        time_vector = np.linspace(0, time_horizon, num_points)
        
        # 使用5次多项式生成平滑轨迹
        # 确保位置、速度、加速度连续
        trajectory = np.zeros((num_points, len(start_state)))
        
        for i in range(len(start_state)):
            # 5次多项式系数计算
            # 边界条件：起始和终止的位置、速度、加速度
            a0 = start_state[i]
            a1 = 0.0  # 起始速度为0
            a2 = 0.0  # 起始加速度为0
            
            # 终止条件
            a3 = 10 * (target_state[i] - start_state[i]) / (time_horizon ** 3)
            a4 = -15 * (target_state[i] - start_state[i]) / (time_horizon ** 4)
            a5 = 6 * (target_state[i] - start_state[i]) / (time_horizon ** 5)
            
            # 生成轨迹
            for j, t in enumerate(time_vector):
                trajectory[j, i] = (a0 + a1 * t + a2 * t**2 + 
                                  a3 * t**3 + a4 * t**4 + a5 * t**5)
        
        # 计算速度和加速度轨迹
        velocity = np.gradient(trajectory, axis=0) / 0.01
        acceleration = np.gradient(velocity, axis=0) / 0.01
        
        return {
            'time': time_vector,
            'position': trajectory,
            'velocity': velocity,
            'acceleration': acceleration,
            'jerk': np.gradient(acceleration, axis=0) / 0.01
        }
    
    def evaluate_optimization_effectiveness(self, window_size: int = 100) -> Dict[str, float]:
        """评估优化效果"""
        if len(self.optimization_history) < window_size:
            recent_history = self.optimization_history
        else:
            recent_history = self.optimization_history[-window_size:]
        
        if not recent_history:
            return {'error': 'No optimization history available'}
        
        # 提取性能数据
        improvement_ratios = [record['improvement_ratio'] for record in recent_history]
        success_rate = np.mean([record['optimization_success'] for record in recent_history])
        
        # 计算各性能指标的改善
        response_times = [record['current_performance']['response_time'] for record in recent_history]
        overshoots = [record['current_performance']['overshoot'] for record in recent_history]
        performance_scores = [record['current_performance']['performance_score'] for record in recent_history]
        
        # 计算趋势
        if len(performance_scores) > 10:
            score_trend = np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]
            trend_direction = "improving" if score_trend > 0.01 else ("declining" if score_trend < -0.01 else "stable")
        else:
            trend_direction = "insufficient_data"
        
        effectiveness = {
            'optimization_success_rate': success_rate,
            'average_improvement_ratio': np.mean(improvement_ratios),
            'performance_trend': trend_direction,
            
            'response_time_performance': {
                'average': np.mean(response_times),
                'std': np.std(response_times),
                'target_achievement_rate': np.mean([1 if rt <= self.optimization_target.target_response_time else 0 
                                                  for rt in response_times])
            },
            
            'overshoot_performance': {
                'average': np.mean(overshoots),
                'std': np.std(overshoots),
                'target_achievement_rate': np.mean([1 if os <= self.optimization_target.target_overshoot else 0 
                                                  for os in overshoots])
            },
            
            'overall_performance': {
                'average_score': np.mean(performance_scores),
                'score_improvement': score_trend if len(performance_scores) > 10 else 0.0,
                'consistency': 1.0 - np.std(performance_scores) / max(np.mean(performance_scores), 0.1)
            }
        }
        
        return effectiveness
    
    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        effectiveness = self.evaluate_optimization_effectiveness()
        
        stats = {
            'optimizer_id': self.optimizer_id,
            'total_optimizations': self.total_optimizations,
            'successful_optimizations': self.successful_optimizations,
            'response_measurements': self.response_measurements,
            'optimization_success_rate': self.successful_optimizations / max(self.total_optimizations, 1),
            
            'optimization_targets': {
                'target_response_time': self.optimization_target.target_response_time,
                'target_overshoot': self.optimization_target.target_overshoot,
                'target_settling_time': self.optimization_target.target_settling_time,
                'target_bandwidth': self.optimization_target.target_bandwidth
            },
            
            'current_parameters': self.adaptive_controller.current_params.copy(),
            'parameter_ranges': self.adaptive_controller.param_ranges.copy(),
            
            'effectiveness_metrics': effectiveness,
            
            'data_sizes': {
                'step_response_data': len(self.step_response_data),
                'frequency_response_data': len(self.frequency_response_data),
                'optimization_history': len(self.optimization_history),
                'parameter_history': len(self.adaptive_controller.param_history)
            },
            
            'model_info': {
                'predictor_parameters': sum(p.numel() for p in self.response_predictor.parameters()),
                'model_size_mb': sum(p.numel() for p in self.response_predictor.parameters()) * 4 / (1024 * 1024)
            }
        }
        
        return stats
    
    def reset_optimizer(self):
        """重置优化器状态"""
        self.step_response_data.clear()
        self.frequency_response_data.clear()
        self.optimization_history.clear()
        self.performance_improvements.clear()
        
        self.adaptive_controller.param_history.clear()
        self.adaptive_controller.performance_history.clear()
        
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.response_measurements = 0
        
        print(f"🔄 响应优化器已重置: {self.optimizer_id}")
    
    def __str__(self) -> str:
        """字符串表示"""
        success_rate = self.successful_optimizations / max(self.total_optimizations, 1)
        return (f"ResponseOptimizer({self.optimizer_id}): "
                f"optimizations={self.total_optimizations}, "
                f"success_rate={success_rate:.3f}, "
                f"measurements={self.response_measurements}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"ResponseOptimizer(optimizer_id='{self.optimizer_id}', "
                f"optimizations={self.total_optimizations}, "
                f"target_response_time={self.optimization_target.target_response_time})")
