import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import LowerLayerConfig
from config.model_config import ModelConfig

@dataclass
class TrackingError:
    """跟踪误差数据结构"""
    instant_error: float = 0.0          # 瞬时误差 (W)
    integral_error: float = 0.0         # 积分误差 (W·s)
    derivative_error: float = 0.0       # 微分误差 (W/s)
    rms_error: float = 0.0              # 均方根误差 (W)
    relative_error: float = 0.0         # 相对误差 (%)
    settling_time: float = 0.0          # 稳定时间 (s)
    overshoot: float = 0.0              # 超调量 (%)

@dataclass
class ControlPerformance:
    """控制性能指标"""
    tracking_accuracy: float = 0.0      # 跟踪精度 [0,1]
    response_speed: float = 0.0         # 响应速度 [0,1]
    stability_margin: float = 0.0       # 稳定裕度 [0,1]
    control_effort: float = 0.0         # 控制努力 [0,1]
    robustness: float = 0.0             # 鲁棒性 [0,1]

class PIDController:
    """PID控制器（作为基线对比）"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.dt = 0.01  # 10ms
    
    def update(self, error: float) -> float:
        """PID控制更新"""
        # 积分项
        self.integral += error * self.dt
        
        # 微分项
        derivative = (error - self.previous_error) / self.dt
        
        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        
        return output
    
    def reset(self):
        """重置PID状态"""
        self.integral = 0.0
        self.previous_error = 0.0

class AdaptivePIDController:
    """自适应PID控制器"""
    
    def __init__(self, initial_gains: Tuple[float, float, float] = (1.0, 0.1, 0.01)):
        self.kp, self.ki, self.kd = initial_gains
        self.initial_gains = initial_gains
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.dt = 0.01
        
        # 自适应参数
        self.error_history = deque(maxlen=100)
        self.adaptation_rate = 0.01
    
    def update(self, error: float, adaptation_enabled: bool = True) -> float:
        """自适应PID控制更新"""
        # 记录误差历史
        self.error_history.append(abs(error))
        
        # 自适应调整增益
        if adaptation_enabled and len(self.error_history) >= 10:
            self._adapt_gains()
        
        # 积分项
        self.integral += error * self.dt
        
        # 积分饱和限制
        max_integral = 1000.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        
        # 微分项
        derivative = (error - self.previous_error) / self.dt
        
        # PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.previous_error = error
        
        return output
    
    def _adapt_gains(self):
        """自适应调整PID增益"""
        recent_errors = list(self.error_history)[-10:]
        avg_error = np.mean(recent_errors)
        error_trend = np.polyfit(range(len(recent_errors)), recent_errors, 1)[0]
        
        # 基于误差大小调整比例增益
        if avg_error > 100.0:  # 大误差
            self.kp += self.adaptation_rate
        elif avg_error < 10.0:  # 小误差
            self.kp -= self.adaptation_rate * 0.5
        
        # 基于误差趋势调整积分增益
        if error_trend > 0:  # 误差增加
            self.ki += self.adaptation_rate * 0.1
        else:  # 误差减少
            self.ki -= self.adaptation_rate * 0.05
        
        # 基于误差变化调整微分增益
        error_variance = np.var(recent_errors)
        if error_variance > 50.0:  # 误差变化大
            self.kd += self.adaptation_rate * 0.01
        
        # 限制增益范围
        self.kp = np.clip(self.kp, 0.1, 10.0)
        self.ki = np.clip(self.ki, 0.01, 1.0)
        self.kd = np.clip(self.kd, 0.001, 0.1)
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.error_history.clear()
        self.kp, self.ki, self.kd = self.initial_gains

class NeuralPowerTracker(nn.Module):
    """神经网络功率跟踪器"""
    
    def __init__(self, 
                 input_dim: int = 10, 
                 hidden_dim: int = 128,
                 output_dim: int = 1):
        super(NeuralPowerTracker, self).__init__()
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1)
        )
        
        # 控制信号生成
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Tanh()
        )
        
        # 误差预测头
        self.error_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # 稳定时间预测头
        self.settling_time_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        features = self.feature_extractor(x)
        
        control_signal = self.control_head(features)
        predicted_error = self.error_predictor(features)
        settling_time = self.settling_time_predictor(features) * 2.0  # 0-2s
        
        return {
            'control_signal': control_signal,
            'predicted_error': predicted_error,
            'settling_time': settling_time,
            'features': features
        }

class PowerTracker(nn.Module):
    """
    功率跟踪控制器
    实现高精度、快响应的功率跟踪控制
    """
    
    def __init__(self,
                 config: LowerLayerConfig,
                 model_config: ModelConfig,
                 tracker_id: str = "PowerTracker_001"):
        """
        初始化功率跟踪器
        
        Args:
            config: 下层配置
            model_config: 模型配置
            tracker_id: 跟踪器ID
        """
        super(PowerTracker, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.tracker_id = tracker_id
        
        # === 控制器参数 ===
        self.dt = 0.01  # 10ms时间步
        self.response_time_target = config.response_time
        
        # === 神经网络跟踪器 ===
        self.neural_tracker = NeuralPowerTracker(
            input_dim=15,  # [power_error, error_derivative, error_integral, reference, current_power, ...]
            hidden_dim=128,
            output_dim=3   # [control_signal, feed_forward, compensation]
        )
        
        # === 传统控制器（用于对比和备份） ===
        self.pid_controller = AdaptivePIDController((2.0, 0.5, 0.1))
        self.feedforward_gain = 0.8
        
        # === 跟踪历史 ===
        self.tracking_history: List[Dict] = []
        self.error_history = deque(maxlen=1000)
        
        # === 性能统计 ===
        self.performance_metrics = ControlPerformance()
        self.total_tracking_steps = 0
        
        # === 自适应参数 ===
        self.adaptation_enabled = True
        self.control_mode = "neural"  # "neural", "pid", "hybrid"
        
        print(f"✅ 功率跟踪器初始化完成: {tracker_id}")
        print(f"   目标响应时间: {self.response_time_target}s")
        print(f"   控制模式: {self.control_mode}")
    
    def track_power(self, 
                   power_reference: float,
                   current_power: float,
                   system_state: Dict[str, Any],
                   constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        功率跟踪控制
        
        Args:
            power_reference: 功率参考值 (W)
            current_power: 当前功率 (W)
            system_state: 系统状态
            constraints: 控制约束
            
        Returns:
            控制结果
        """
        # === 1. 计算跟踪误差 ===
        tracking_error = self._calculate_tracking_error(power_reference, current_power)
        
        # === 2. 准备控制输入 ===
        control_input = self._prepare_control_input(
            tracking_error, power_reference, current_power, system_state
        )
        
        # === 3. 生成控制信号 ===
        if self.control_mode == "neural":
            control_result = self._neural_control(control_input, tracking_error)
        elif self.control_mode == "pid":
            control_result = self._pid_control(tracking_error)
        else:  # hybrid
            control_result = self._hybrid_control(control_input, tracking_error)
        
        # === 4. 应用约束 ===
        if constraints:
            control_result = self._apply_constraints(control_result, constraints)
        
        # === 5. 记录跟踪历史 ===
        self._record_tracking_step(
            power_reference, current_power, tracking_error, control_result, system_state
        )
        
        # === 6. 更新性能指标 ===
        self._update_performance_metrics(tracking_error, control_result)
        
        self.total_tracking_steps += 1
        
        return control_result
    
    def _calculate_tracking_error(self, reference: float, current: float) -> TrackingError:
        """计算跟踪误差"""
        instant_error = reference - current
        
        # 计算积分误差
        if len(self.error_history) > 0:
            integral_error = sum(self.error_history) * self.dt + instant_error * self.dt
        else:
            integral_error = instant_error * self.dt
        
        # 计算微分误差
        if len(self.error_history) > 0:
            derivative_error = (instant_error - self.error_history[-1]) / self.dt
        else:
            derivative_error = 0.0
        
        # 计算RMS误差
        recent_errors = list(self.error_history)[-50:] + [instant_error]
        rms_error = np.sqrt(np.mean([e**2 for e in recent_errors]))
        
        # 计算相对误差
        relative_error = abs(instant_error) / max(abs(reference), 1.0) * 100.0
        
        # 估算稳定时间和超调量
        settling_time, overshoot = self._estimate_settling_metrics(recent_errors)
        
        error = TrackingError(
            instant_error=instant_error,
            integral_error=integral_error,
            derivative_error=derivative_error,
            rms_error=rms_error,
            relative_error=relative_error,
            settling_time=settling_time,
            overshoot=overshoot
        )
        
        # 更新误差历史
        self.error_history.append(instant_error)
        
        return error
    
    def _estimate_settling_metrics(self, errors: List[float]) -> Tuple[float, float]:
        """估算稳定时间和超调量"""
        if len(errors) < 10:
            return 0.0, 0.0
        
        # 稳定时间：误差进入±2%范围的时间
        settling_threshold = abs(errors[0]) * 0.02 if errors[0] != 0 else 1.0
        settling_time = 0.0
        
        for i, error in enumerate(reversed(errors)):
            if abs(error) > settling_threshold:
                settling_time = i * self.dt
                break
        
        # 超调量：最大偏差相对于稳态值的百分比
        max_error = max(abs(e) for e in errors)
        steady_state_error = abs(np.mean(errors[-5:]))
        overshoot = ((max_error - steady_state_error) / max(steady_state_error, 1.0)) * 100.0
        
        return settling_time, overshoot
    
    def _prepare_control_input(self, 
                             error: TrackingError,
                             reference: float,
                             current: float,
                             system_state: Dict[str, Any]) -> torch.Tensor:
        """准备神经网络控制输入"""
        # 归一化输入特征
        max_power = 100000.0  # 100kW归一化基准
        
        input_features = [
            error.instant_error / max_power,
            error.derivative_error / (max_power / self.dt),
            error.integral_error / (max_power * self.dt),
            reference / max_power,
            current / max_power,
            error.relative_error / 100.0,
            error.rms_error / max_power,
            
            # 系统状态特征
            system_state.get('soc', 50.0) / 100.0,
            system_state.get('temperature', 25.0) / 60.0,
            system_state.get('voltage', 3.4) / 4.2,
            system_state.get('soh', 100.0) / 100.0,
            
            # 时间特征
            (self.total_tracking_steps % 100) / 100.0,  # 周期性特征
            min(self.total_tracking_steps / 10000.0, 1.0),  # 经验特征
            
            # 约束特征
            system_state.get('constraint_severity', 0.0),
            system_state.get('thermal_constraint_active', 0.0)
        ]
        
        return torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
    
    def _neural_control(self, 
                       control_input: torch.Tensor, 
                       error: TrackingError) -> Dict[str, Any]:
        """神经网络控制"""
        self.neural_tracker.eval()
        
        with torch.no_grad():
            neural_output = self.neural_tracker(control_input)
        
        # 解析神经网络输出
        control_signals = neural_output['control_signal'].squeeze(0)
        
        primary_control = control_signals[0].item() * 10000.0  # W
        feedforward_control = control_signals[1].item() * 5000.0  # W
        compensation_control = control_signals[2].item() * 2000.0  # W
        
        # 组合控制信号
        total_control = primary_control + feedforward_control + compensation_control
        
        # 预测性能
        predicted_error = neural_output['predicted_error'].item()
        predicted_settling_time = neural_output['settling_time'].item()
        
        return {
            'control_signal': total_control,
            'primary_control': primary_control,
            'feedforward_control': feedforward_control,
            'compensation_control': compensation_control,
            'predicted_error': predicted_error,
            'predicted_settling_time': predicted_settling_time,
            'control_confidence': min(1.0, 1.0 / (abs(predicted_error) + 0.1)),
            'control_type': 'neural'
        }
    
    def _pid_control(self, error: TrackingError) -> Dict[str, Any]:
        """PID控制"""
        pid_output = self.pid_controller.update(error.instant_error)
        
        # 前馈控制
        feedforward = 0.0
        if len(self.tracking_history) > 0:
            recent_reference = self.tracking_history[-1]['power_reference']
            feedforward = recent_reference * self.feedforward_gain
        
        total_control = pid_output + feedforward
        
        return {
            'control_signal': total_control,
            'primary_control': pid_output,
            'feedforward_control': feedforward,
            'compensation_control': 0.0,
            'predicted_error': abs(error.instant_error) * 0.9,  # 简单预测
            'predicted_settling_time': self.response_time_target,
            'control_confidence': min(1.0, 100.0 / (abs(error.instant_error) + 1.0)),
            'control_type': 'pid',
            'pid_gains': [self.pid_controller.kp, self.pid_controller.ki, self.pid_controller.kd]
        }
    
    def _hybrid_control(self, 
                       control_input: torch.Tensor, 
                       error: TrackingError) -> Dict[str, Any]:
        """混合控制（神经网络+PID）"""
        # 神经网络控制
        neural_result = self._neural_control(control_input, error)
        
        # PID控制
        pid_result = self._pid_control(error)
        
        # 动态权重分配
        error_magnitude = abs(error.instant_error)
        if error_magnitude > 1000.0:  # 大误差时偏向PID
            neural_weight = 0.3
            pid_weight = 0.7
        elif error_magnitude < 100.0:  # 小误差时偏向神经网络
            neural_weight = 0.8
            pid_weight = 0.2
        else:  # 中等误差时平衡
            neural_weight = 0.6
            pid_weight = 0.4
        
        # 加权组合
        combined_control = (neural_weight * neural_result['control_signal'] + 
                          pid_weight * pid_result['control_signal'])
        
        return {
            'control_signal': combined_control,
            'primary_control': combined_control,
            'feedforward_control': neural_result['feedforward_control'],
            'compensation_control': neural_result['compensation_control'],
            'predicted_error': (neural_weight * neural_result['predicted_error'] + 
                              pid_weight * pid_result['predicted_error']),
            'predicted_settling_time': min(neural_result['predicted_settling_time'],
                                         pid_result['predicted_settling_time']),
            'control_confidence': max(neural_result['control_confidence'],
                                    pid_result['control_confidence']),
            'control_type': 'hybrid',
            'neural_weight': neural_weight,
            'pid_weight': pid_weight
        }
    
    def _apply_constraints(self, 
                          control_result: Dict[str, Any], 
                          constraints: Dict[str, float]) -> Dict[str, Any]:
        """应用控制约束"""
        control_signal = control_result['control_signal']
        
        # 功率变化率约束
        max_power_change_rate = constraints.get('max_power_change_rate', 10000.0)  # W/s
        max_change_per_step = max_power_change_rate * self.dt
        
        if len(self.tracking_history) > 0:
            last_control = self.tracking_history[-1]['control_result']['control_signal']
            control_change = control_signal - last_control
            
            if abs(control_change) > max_change_per_step:
                control_signal = last_control + np.sign(control_change) * max_change_per_step
        
        # 功率幅值约束
        max_power = constraints.get('max_power', 50000.0)  # W
        control_signal = np.clip(control_signal, -max_power, max_power)
        
        # 更新控制结果
        control_result = control_result.copy()
        control_result['control_signal'] = control_signal
        control_result['constraint_applied'] = True
        
        return control_result
    
    def _record_tracking_step(self, 
                             reference: float,
                             current: float,
                             error: TrackingError,
                             control_result: Dict[str, Any],
                             system_state: Dict[str, Any]):
        """记录跟踪步骤"""
        record = {
            'step': self.total_tracking_steps,
            'timestamp': self.total_tracking_steps * self.dt,
            'power_reference': reference,
            'current_power': current,
            'tracking_error': error,
            'control_result': control_result,
            'system_state': system_state.copy()
        }
        
        self.tracking_history.append(record)
        
        # 维护历史长度
        if len(self.tracking_history) > 10000:
            self.tracking_history.pop(0)
    
    def _update_performance_metrics(self, 
                                   error: TrackingError, 
                                   control_result: Dict[str, Any]):
        """更新性能指标"""
        # 跟踪精度
        self.performance_metrics.tracking_accuracy = 1.0 - min(1.0, error.rms_error / 1000.0)
        
        # 响应速度
        self.performance_metrics.response_speed = max(0.0, 1.0 - error.settling_time / (self.response_time_target * 2))
        
        # 稳定裕度
        self.performance_metrics.stability_margin = max(0.0, 1.0 - error.overshoot / 50.0)
        
        # 控制努力
        control_effort = abs(control_result['control_signal']) / 50000.0  # 归一化到50kW
        self.performance_metrics.control_effort = min(1.0, control_effort)
        
        # 鲁棒性（基于最近的性能一致性）
        if len(self.tracking_history) >= 100:
            recent_errors = [record['tracking_error'].rms_error for record in self.tracking_history[-100:]]
            error_consistency = 1.0 - np.std(recent_errors) / max(np.mean(recent_errors), 1.0)
            self.performance_metrics.robustness = max(0.0, error_consistency)
    
    def evaluate_tracking_performance(self, window_size: int = 1000) -> Dict[str, float]:
        """评估跟踪性能"""
        if len(self.tracking_history) < window_size:
            recent_history = self.tracking_history
        else:
            recent_history = self.tracking_history[-window_size:]
        
        if not recent_history:
            return {'error': 'No tracking history available'}
        
        # 提取性能数据
        instant_errors = [record['tracking_error'].instant_error for record in recent_history]
        rms_errors = [record['tracking_error'].rms_error for record in recent_history]
        settling_times = [record['tracking_error'].settling_time for record in recent_history]
        overshoots = [record['tracking_error'].overshoot for record in recent_history]
        relative_errors = [record['tracking_error'].relative_error for record in recent_history]
        
        # 计算统计指标
        performance = {
            'avg_instant_error': np.mean(np.abs(instant_errors)),
            'max_instant_error': max(np.abs(instant_errors)),
            'avg_rms_error': np.mean(rms_errors),
            'avg_settling_time': np.mean(settling_times),
            'max_settling_time': max(settling_times),
            'avg_overshoot': np.mean(overshoots),
            'max_overshoot': max(overshoots),
            'avg_relative_error': np.mean(relative_errors),
            
            # 性能指标
            'tracking_accuracy': self.performance_metrics.tracking_accuracy,
            'response_speed': self.performance_metrics.response_speed,
            'stability_margin': self.performance_metrics.stability_margin,
            'control_effort': self.performance_metrics.control_effort,
            'robustness': self.performance_metrics.robustness,
            
            # 响应时间性能
            'response_time_compliance': np.mean([1.0 if t <= self.response_time_target else 0.0 
                                               for t in settling_times]),
            
            # 误差分布
            'error_percentiles': {
                '50th': np.percentile(np.abs(instant_errors), 50),
                '90th': np.percentile(np.abs(instant_errors), 90),
                '95th': np.percentile(np.abs(instant_errors), 95),
                '99th': np.percentile(np.abs(instant_errors), 99)
            }
        }
        
        return performance
    
    def adapt_control_parameters(self, performance_feedback: Dict[str, float]) -> bool:
        """根据性能反馈自适应调整控制参数"""
        if not self.adaptation_enabled:
            return False
        
        try:
            # 获取当前性能指标
            tracking_accuracy = performance_feedback.get('tracking_accuracy', 0.8)
            response_speed = performance_feedback.get('response_speed', 0.8)
            stability_margin = performance_feedback.get('stability_margin', 0.8)
            
            # 根据性能调整控制模式
            if tracking_accuracy < 0.7 and response_speed < 0.7:
                # 性能不佳，切换到混合模式
                self.control_mode = "hybrid"
                print(f"🔄 切换到混合控制模式")
            elif tracking_accuracy > 0.9 and response_speed > 0.9:
                # 性能优秀，使用神经网络模式
                self.control_mode = "neural"
            elif stability_margin < 0.6:
                # 稳定性不佳，使用PID模式
                self.control_mode = "pid"
                print(f"🔄 切换到PID控制模式")
            
            # 调整PID参数
            avg_error = performance_feedback.get('avg_instant_error', 0.0)
            if avg_error > 500.0:  # 大误差
                self.pid_controller.kp = min(5.0, self.pid_controller.kp * 1.1)
            elif avg_error < 50.0:  # 小误差
                self.pid_controller.kp = max(0.5, self.pid_controller.kp * 0.95)
            
            # 调整前馈增益
            response_time_compliance = performance_feedback.get('response_time_compliance', 1.0)
            if response_time_compliance < 0.8:
                self.feedforward_gain = min(1.0, self.feedforward_gain * 1.05)
            
            return True
            
        except Exception as e:
            print(f"❌ 控制参数自适应失败: {str(e)}")
            return False
    
    def reset_tracker(self):
        """重置跟踪器状态"""
        self.error_history.clear()
        self.tracking_history.clear()
        self.pid_controller.reset()
        self.total_tracking_steps = 0
        self.performance_metrics = ControlPerformance()
        
        print(f"🔄 功率跟踪器已重置: {self.tracker_id}")
    
    def get_tracker_statistics(self) -> Dict[str, Any]:
        """获取跟踪器统计信息"""
        performance = self.evaluate_tracking_performance()
        
        stats = {
            'tracker_id': self.tracker_id,
            'total_tracking_steps': self.total_tracking_steps,
            'control_mode': self.control_mode,
            'adaptation_enabled': self.adaptation_enabled,
            
            'performance_metrics': performance,
            
            'current_parameters': {
                'pid_gains': [self.pid_controller.kp, self.pid_controller.ki, self.pid_controller.kd],
                'feedforward_gain': self.feedforward_gain,
                'response_time_target': self.response_time_target
            },
            
            'model_info': {
                'neural_tracker_parameters': sum(p.numel() for p in self.neural_tracker.parameters()),
                'model_size_mb': sum(p.numel() for p in self.neural_tracker.parameters()) * 4 / (1024 * 1024)
            },
            
            'tracking_history_size': len(self.tracking_history),
            'error_history_size': len(self.error_history)
        }
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"PowerTracker({self.tracker_id}): "
                f"mode={self.control_mode}, steps={self.total_tracking_steps}, "
                f"accuracy={self.performance_metrics.tracking_accuracy:.3f}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"PowerTracker(tracker_id='{self.tracker_id}', "
                f"control_mode='{self.control_mode}', "
                f"tracking_steps={self.total_tracking_steps})")
