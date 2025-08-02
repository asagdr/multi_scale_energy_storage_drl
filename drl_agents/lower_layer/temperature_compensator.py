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
class TemperatureProfile:
    """温度分布数据结构"""
    temperatures: np.ndarray               # 各单体温度 (℃)
    avg_temperature: float = 0.0           # 平均温度 (℃)
    max_temperature: float = 0.0           # 最高温度 (℃)
    min_temperature: float = 0.0           # 最低温度 (℃)
    temp_std: float = 0.0                  # 温度标准差 (℃)
    temp_gradient: np.ndarray = None       # 温度梯度 (℃/位置)
    hotspot_indices: List[int] = field(default_factory=list)  # 热点位置
    coldspot_indices: List[int] = field(default_factory=list) # 冷点位置

@dataclass
class CompensationAction:
    """温度补偿动作"""
    power_derating: float = 0.0            # 功率降额 (%)
    cooling_enhancement: float = 0.0       # 冷却增强 (%)
    balancing_adjustment: float = 0.0      # 均衡调整 (%)
    thermal_redistribution: np.ndarray = None  # 热量重分布策略
    urgency_level: float = 0.0             # 紧急程度 [0,1]

class ThermalModel(nn.Module):
    """简化热模型预测器"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 64):
        super(ThermalModel, self).__init__()
        
        # 温度预测网络
        self.temp_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 10)  # 预测10个单体的温度变化
        )
        
        # 热点检测网络
        self.hotspot_detector = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),  # 10个单体的热点概率
            nn.Sigmoid()
        )
        
        # 冷却需求预测
        self.cooling_predictor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        temp_changes = self.temp_predictor(x)
        hotspot_probs = self.hotspot_detector(x)
        cooling_demand = self.cooling_predictor(x)
        
        return {
            'temperature_changes': temp_changes,
            'hotspot_probabilities': hotspot_probs,
            'cooling_demand': cooling_demand
        }

class AdaptiveThermalController:
    """自适应热管理控制器"""
    
    def __init__(self, num_cells: int = 10):
        self.num_cells = num_cells
        
        # 控制参数
        self.temp_threshold_high = 45.0      # ℃, 高温阈值
        self.temp_threshold_critical = 55.0  # ℃, 危险温度
        self.temp_diff_threshold = 10.0      # ℃, 温差阈值
        
        # PID参数（用于温度控制）
        self.temp_kp = 0.5
        self.temp_ki = 0.1
        self.temp_kd = 0.05
        
        # 积分项
        self.temp_integral = np.zeros(num_cells)
        self.prev_temp_error = np.zeros(num_cells)
        
        self.dt = 0.01  # 10ms
    
    def update_thermal_control(self, 
                             temperatures: np.ndarray,
                             target_temp: float = 35.0) -> Dict[str, np.ndarray]:
        """更新热管理控制"""
        temp_errors = target_temp - temperatures
        
        # PID控制
        self.temp_integral += temp_errors * self.dt
        temp_derivative = (temp_errors - self.prev_temp_error) / self.dt
        
        # 计算控制输出
        control_output = (self.temp_kp * temp_errors + 
                         self.temp_ki * self.temp_integral + 
                         self.temp_kd * temp_derivative)
        
        # 冷却控制（正值表示需要更多冷却）
        cooling_control = np.clip(control_output, 0, 1)
        
        # 功率调节（负值表示需要降低功率）
        power_adjustment = np.clip(-control_output, -1, 0)
        
        self.prev_temp_error = temp_errors
        
        return {
            'cooling_control': cooling_control,
            'power_adjustment': power_adjustment,
            'temp_errors': temp_errors
        }
    
    def reset(self):
        """重置控制器状态"""
        self.temp_integral.fill(0)
        self.prev_temp_error.fill(0)

class TemperatureCompensator(nn.Module):
    """
    温度补偿器
    实时监测温度分布，预测热风险，生成温度补偿策略
    """
    
    def __init__(self,
                 config: LowerLayerConfig,
                 model_config: ModelConfig,
                 compensator_id: str = "TempCompensator_001",
                 num_cells: int = 10):
        """
        初始化温度补偿器
        
        Args:
            config: 下层配置
            model_config: 模型配置
            compensator_id: 补偿器ID
            num_cells: 电池单体数量
        """
        super(TemperatureCompensator, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.compensator_id = compensator_id
        self.num_cells = num_cells
        
        # === 热模型 ===
        self.thermal_model = ThermalModel(
            input_dim=num_cells + 10,  # 温度 + 系统状态
            hidden_dim=64
        )
        
        # === 自适应控制器 ===
        self.adaptive_controller = AdaptiveThermalController(num_cells)
        
        # === 温度阈值 ===
        self.temp_thresholds = {
            'normal_max': 40.0,      # ℃, 正常最高温度
            'warning': 45.0,         # ℃, 预警温度
            'alarm': 50.0,           # ℃, 报警温度
            'critical': 55.0,        # ℃, 危险温度
            'emergency': 60.0,       # ℃, 紧急停机温度
            'temp_diff_max': 15.0    # ℃, 最大温差
        }
        
        # === 补偿策略参数 ===
        self.compensation_gains = {
            'power_derating': 0.02,     # 每度温差的功率降额比例
            'cooling_boost': 0.05,      # 每度超温的冷却增强比例
            'balancing_factor': 0.1     # 温度不均时的均衡调整
        }
        
        # === 历史记录 ===
        self.temperature_history: List[TemperatureProfile] = []
        self.compensation_history: List[CompensationAction] = []
        
        # === 统计信息 ===
        self.total_compensations = 0
        self.hotspot_detections = 0
        self.emergency_interventions = 0
        
        # === 预测缓存 ===
        self.prediction_cache = deque(maxlen=100)
        
        print(f"✅ 温度补偿器初始化完成: {compensator_id}")
        print(f"   监测单体数: {num_cells}")
        print(f"   温度阈值: 正常≤{self.temp_thresholds['normal_max']}℃, 危险≥{self.temp_thresholds['critical']}℃")
    
    def analyze_temperature_profile(self, 
                                  temperatures: np.ndarray,
                                  ambient_temp: float = 25.0) -> TemperatureProfile:
        """
        分析温度分布
        
        Args:
            temperatures: 各单体温度数组 (℃)
            ambient_temp: 环境温度 (℃)
            
        Returns:
            温度分布分析结果
        """
        # 基本统计
        avg_temp = np.mean(temperatures)
        max_temp = np.max(temperatures)
        min_temp = np.min(temperatures)
        temp_std = np.std(temperatures)
        
        # 温度梯度（简化为相邻单体温差）
        temp_gradient = np.gradient(temperatures)
        
        # 热点检测（高于平均温度+1个标准差）
        hotspot_threshold = avg_temp + temp_std
        hotspot_indices = np.where(temperatures > hotspot_threshold)[0].tolist()
        
        # 冷点检测（低于平均温度-1个标准差）
        coldspot_threshold = avg_temp - temp_std
        coldspot_indices = np.where(temperatures < coldspot_threshold)[0].tolist()
        
        profile = TemperatureProfile(
            temperatures=temperatures.copy(),
            avg_temperature=avg_temp,
            max_temperature=max_temp,
            min_temperature=min_temp,
            temp_std=temp_std,
            temp_gradient=temp_gradient,
            hotspot_indices=hotspot_indices,
            coldspot_indices=coldspot_indices
        )
        
        # 记录历史
        self.temperature_history.append(profile)
        
        # 维护历史长度
        if len(self.temperature_history) > 1000:
            self.temperature_history.pop(0)
        
        return profile
    
    def predict_thermal_behavior(self, 
                                temp_profile: TemperatureProfile,
                                system_state: Dict[str, Any],
                                power_command: float) -> Dict[str, Any]:
        """
        预测热行为
        
        Args:
            temp_profile: 当前温度分布
            system_state: 系统状态
            power_command: 功率指令 (W)
            
        Returns:
            热行为预测结果
        """
        # 准备输入特征
        input_features = self._prepare_thermal_input(temp_profile, system_state, power_command)
        
        # 神经网络预测
        self.thermal_model.eval()
        with torch.no_grad():
            prediction = self.thermal_model(input_features)
        
        # 解析预测结果
        temp_changes = prediction['temperature_changes'].squeeze().numpy()
        hotspot_probs = prediction['hotspot_probabilities'].squeeze().numpy()
        cooling_demand = prediction['cooling_demand'].item()
        
        # 预测未来温度
        future_temps = temp_profile.temperatures + temp_changes
        
        # 热风险评估
        thermal_risk = self._assess_thermal_risk(future_temps, hotspot_probs)
        
        # 冷却需求评估
        cooling_urgency = self._evaluate_cooling_urgency(
            temp_profile, future_temps, cooling_demand
        )
        
        prediction_result = {
            'predicted_temperatures': future_temps,
            'temperature_changes': temp_changes,
            'hotspot_probabilities': hotspot_probs,
            'cooling_demand': cooling_demand,
            'thermal_risk': thermal_risk,
            'cooling_urgency': cooling_urgency,
            'max_predicted_temp': np.max(future_temps),
            'temp_rise_rate': np.max(temp_changes) / 0.01,  # ℃/s
            'hotspot_count': np.sum(hotspot_probs > 0.7)
        }
        
        # 缓存预测结果
        self.prediction_cache.append(prediction_result)
        
        return prediction_result
    
    def generate_compensation_action(self, 
                                   temp_profile: TemperatureProfile,
                                   thermal_prediction: Dict[str, Any],
                                   system_constraints: Dict[str, float]) -> CompensationAction:
        """
        生成温度补偿动作
        
        Args:
            temp_profile: 温度分布
            thermal_prediction: 热预测结果
            system_constraints: 系统约束
            
        Returns:
            补偿动作
        """
        # === 1. 评估补偿紧急程度 ===
        urgency = self._calculate_compensation_urgency(temp_profile, thermal_prediction)
        
        # === 2. 功率降额计算 ===
        power_derating = self._calculate_power_derating(
            temp_profile.max_temperature, 
            thermal_prediction['max_predicted_temp'],
            urgency
        )
        
        # === 3. 冷却增强计算 ===
        cooling_enhancement = self._calculate_cooling_enhancement(
            thermal_prediction['cooling_demand'],
            thermal_prediction['cooling_urgency'],
            urgency
        )
        
        # === 4. 均衡调整计算 ===
        balancing_adjustment = self._calculate_balancing_adjustment(
            temp_profile.temp_std,
            len(temp_profile.hotspot_indices),
            urgency
        )
        
        # === 5. 热量重分布策略 ===
        thermal_redistribution = self._generate_thermal_redistribution(
            temp_profile, thermal_prediction
        )
        
        # === 6. 应用系统约束 ===
        power_derating = min(power_derating, system_constraints.get('max_power_derating', 0.5))
        cooling_enhancement = min(cooling_enhancement, system_constraints.get('max_cooling_boost', 1.0))
        
        compensation = CompensationAction(
            power_derating=power_derating,
            cooling_enhancement=cooling_enhancement,
            balancing_adjustment=balancing_adjustment,
            thermal_redistribution=thermal_redistribution,
            urgency_level=urgency
        )
        
        # 记录补偿历史
        self.compensation_history.append(compensation)
        self.total_compensations += 1
        
        # 检测热点和紧急情况
        if len(temp_profile.hotspot_indices) > 0:
            self.hotspot_detections += 1
        
        if urgency > 0.8:
            self.emergency_interventions += 1
        
        # 维护历史长度
        if len(self.compensation_history) > 1000:
            self.compensation_history.pop(0)
        
        return compensation
    
    def _prepare_thermal_input(self, 
                              temp_profile: TemperatureProfile,
                              system_state: Dict[str, Any],
                              power_command: float) -> torch.Tensor:
        """准备热模型输入"""
        input_features = []
        
        # 温度特征
        input_features.extend(temp_profile.temperatures.tolist())
        
        # 系统状态特征
        input_features.extend([
            system_state.get('soc', 50.0) / 100.0,
            system_state.get('voltage', 3.4) / 4.2,
            system_state.get('current', 0.0) / 200.0,
            power_command / 50000.0,
            system_state.get('ambient_temperature', 25.0) / 60.0,
            
            # 温度分布特征
            temp_profile.avg_temperature / 60.0,
            temp_profile.max_temperature / 60.0,
            temp_profile.min_temperature / 60.0,
            temp_profile.temp_std / 20.0,
            len(temp_profile.hotspot_indices) / self.num_cells
        ])
        
        return torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
    
    def _assess_thermal_risk(self, 
                           future_temps: np.ndarray, 
                           hotspot_probs: np.ndarray) -> float:
        """评估热风险"""
        # 基于温度的风险
        temp_risk = 0.0
        for temp in future_temps:
            if temp > self.temp_thresholds['critical']:
                temp_risk += 1.0
            elif temp > self.temp_thresholds['alarm']:
                temp_risk += 0.7
            elif temp > self.temp_thresholds['warning']:
                temp_risk += 0.3
        
        temp_risk /= len(future_temps)
        
        # 基于热点概率的风险
        hotspot_risk = np.mean(hotspot_probs)
        
        # 基于温度分布的风险
        temp_std_risk = min(1.0, np.std(future_temps) / 20.0)
        
        # 综合风险
        overall_risk = 0.5 * temp_risk + 0.3 * hotspot_risk + 0.2 * temp_std_risk
        
        return min(1.0, overall_risk)
    
    def _evaluate_cooling_urgency(self, 
                                current_profile: TemperatureProfile,
                                future_temps: np.ndarray,
                                cooling_demand: float) -> float:
        """评估冷却紧急程度"""
        # 当前温度紧急程度
        current_urgency = 0.0
        if current_profile.max_temperature > self.temp_thresholds['critical']:
            current_urgency = 1.0
        elif current_profile.max_temperature > self.temp_thresholds['alarm']:
            current_urgency = 0.7
        elif current_profile.max_temperature > self.temp_thresholds['warning']:
            current_urgency = 0.3
        
        # 预测温度紧急程度
        future_urgency = 0.0
        max_future_temp = np.max(future_temps)
        if max_future_temp > self.temp_thresholds['critical']:
            future_urgency = 1.0
        elif max_future_temp > self.temp_thresholds['alarm']:
            future_urgency = 0.7
        
        # 温升速率紧急程度
        temp_rise_rate = (max_future_temp - current_profile.max_temperature) / 0.01  # ℃/s
        rate_urgency = min(1.0, temp_rise_rate / 10.0)  # 10℃/s为最高紧急程度
        
        # 综合紧急程度
        urgency = max(current_urgency, future_urgency) + 0.3 * rate_urgency + 0.2 * cooling_demand
        
        return min(1.0, urgency)
    
    def _calculate_compensation_urgency(self, 
                                      temp_profile: TemperatureProfile,
                                      thermal_prediction: Dict[str, Any]) -> float:
        """计算补偿紧急程度"""
        # 温度超限紧急程度
        temp_urgency = 0.0
        if temp_profile.max_temperature > self.temp_thresholds['emergency']:
            temp_urgency = 1.0
        elif temp_profile.max_temperature > self.temp_thresholds['critical']:
            temp_urgency = 0.9
        elif temp_profile.max_temperature > self.temp_thresholds['alarm']:
            temp_urgency = 0.6
        elif temp_profile.max_temperature > self.temp_thresholds['warning']:
            temp_urgency = 0.3
        
        # 温差紧急程度
        temp_diff = temp_profile.max_temperature - temp_profile.min_temperature
        diff_urgency = min(1.0, temp_diff / self.temp_thresholds['temp_diff_max'])
        
        # 热风险紧急程度
        risk_urgency = thermal_prediction['thermal_risk']
        
        # 热点数量紧急程度
        hotspot_urgency = min(1.0, len(temp_profile.hotspot_indices) / (self.num_cells * 0.3))
        
        # 综合紧急程度
        urgency = max(temp_urgency, 0.7 * risk_urgency) + 0.2 * diff_urgency + 0.1 * hotspot_urgency
        
        return min(1.0, urgency)
    
    def _calculate_power_derating(self, 
                                current_max_temp: float,
                                predicted_max_temp: float,
                                urgency: float) -> float:
        """计算功率降额"""
        # 基于当前温度的降额
        if current_max_temp > self.temp_thresholds['critical']:
            base_derating = 0.5  # 50%降额
        elif current_max_temp > self.temp_thresholds['alarm']:
            base_derating = 0.3  # 30%降额
        elif current_max_temp > self.temp_thresholds['warning']:
            base_derating = 0.1  # 10%降额
        else:
            base_derating = 0.0
        
        # 基于预测温度的预防性降额
        if predicted_max_temp > self.temp_thresholds['alarm']:
            predictive_derating = 0.2
        elif predicted_max_temp > self.temp_thresholds['warning']:
            predictive_derating = 0.1
        else:
            predictive_derating = 0.0
        
        # 基于紧急程度的调整
        urgency_multiplier = 1.0 + urgency * 0.5
        
        total_derating = (base_derating + predictive_derating) * urgency_multiplier
        
        return min(0.8, total_derating)  # 最大80%降额
    
    def _calculate_cooling_enhancement(self, 
                                     cooling_demand: float,
                                     cooling_urgency: float,
                                     urgency: float) -> float:
        """计算冷却增强"""
        # 基于冷却需求
        base_enhancement = cooling_demand * 0.5
        
        # 基于冷却紧急程度
        urgency_enhancement = cooling_urgency * 0.8
        
        # 基于总体紧急程度
        overall_enhancement = urgency * 0.3
        
        total_enhancement = base_enhancement + urgency_enhancement + overall_enhancement
        
        return min(2.0, total_enhancement)  # 最大200%冷却增强
    
    def _calculate_balancing_adjustment(self, 
                                      temp_std: float,
                                      hotspot_count: int,
                                      urgency: float) -> float:
        """计算均衡调整"""
        # 基于温度标准差
        std_adjustment = min(1.0, temp_std / 10.0) * 0.3
        
        # 基于热点数量
        hotspot_adjustment = min(1.0, hotspot_count / (self.num_cells * 0.5)) * 0.5
        
        # 基于紧急程度
        urgency_adjustment = urgency * 0.2
        
        total_adjustment = std_adjustment + hotspot_adjustment + urgency_adjustment
        
        return min(1.0, total_adjustment)
    
    def _generate_thermal_redistribution(self, 
                                       temp_profile: TemperatureProfile,
                                       thermal_prediction: Dict[str, Any]) -> np.ndarray:
        """生成热量重分布策略"""
        redistribution = np.zeros(self.num_cells)
        
        # 对热点进行负调整（减少功率分配）
        for hotspot_idx in temp_profile.hotspot_indices:
            if hotspot_idx < len(redistribution):
                redistribution[hotspot_idx] = -0.2  # 减少20%功率分配
        
        # 对冷点进行正调整（增加功率分配）
        for coldspot_idx in temp_profile.coldspot_indices:
            if coldspot_idx < len(redistribution):
                redistribution[coldspot_idx] = 0.1   # 增加10%功率分配
        
        # 基于热点概率进行细调
        hotspot_probs = thermal_prediction['hotspot_probabilities']
        for i, prob in enumerate(hotspot_probs):
            if i < len(redistribution) and prob > 0.5:
                redistribution[i] -= prob * 0.1
        
        return redistribution
    
    def apply_temperature_compensation(self, 
                                     control_action: torch.Tensor,
                                     compensation: CompensationAction,
                                     system_state: Dict[str, Any]) -> torch.Tensor:
        """
        应用温度补偿到控制动作
        
        Args:
            control_action: 原始控制动作
            compensation: 温度补偿动作
            system_state: 系统状态
            
        Returns:
            补偿后的控制动作
        """
        compensated_action = control_action.clone()
        
        # === 1. 功率降额 ===
        if compensation.power_derating > 0:
            # 对功率控制信号进行降额
            power_scale = 1.0 - compensation.power_derating
            compensated_action[0] *= power_scale
        
        # === 2. 响应速度调整 ===
        if compensation.urgency_level > 0.5:
            # 高紧急程度时加快响应
            response_boost = 1.0 + compensation.urgency_level * 0.2
            if len(compensated_action) > 1:
                compensated_action[1] = torch.clamp(
                    compensated_action[1] * response_boost, -1.0, 1.0
                )
        
        # === 3. 热补偿调整 ===
        if len(compensated_action) > 2:
            # 第三个动作维度用于热补偿
            thermal_compensation = compensation.cooling_enhancement * 0.1
            compensated_action[2] = torch.clamp(
                compensated_action[2] + thermal_compensation, -1.0, 1.0
            )
        
        return compensated_action
    
    def evaluate_compensation_effectiveness(self, window_size: int = 100) -> Dict[str, float]:
        """评估补偿效果"""
        if len(self.temperature_history) < window_size:
            recent_temps = self.temperature_history
            recent_comps = self.compensation_history
        else:
            recent_temps = self.temperature_history[-window_size:]
            recent_comps = self.compensation_history[-window_size:]
        
        if not recent_temps or not recent_comps:
            return {'error': 'Insufficient history for evaluation'}
        
        # 温度控制效果
        max_temps = [profile.max_temperature for profile in recent_temps]
        avg_temps = [profile.avg_temperature for profile in recent_temps]
        temp_stds = [profile.temp_std for profile in recent_temps]
        
        # 补偿响应效果
        compensations = [comp.power_derating for comp in recent_comps]
        urgencies = [comp.urgency_level for comp in recent_comps]
        
        effectiveness = {
            'temperature_control': {
                'avg_max_temp': np.mean(max_temps),
                'max_temp_variance': np.var(max_temps),
                'avg_temp_std': np.mean(temp_stds),
                'temperature_stability': 1.0 - np.std(avg_temps) / max(np.mean(avg_temps), 1.0)
            },
            
            'compensation_performance': {
                'avg_power_derating': np.mean(compensations),
                'compensation_frequency': np.mean([1 if c > 0 else 0 for c in compensations]),
                'avg_urgency': np.mean(urgencies),
                'response_consistency': 1.0 - np.std(urgencies)
            },
            
            'thermal_management': {
                'hotspot_detection_rate': self.hotspot_detections / max(self.total_compensations, 1),
                'emergency_intervention_rate': self.emergency_interventions / max(self.total_compensations, 1),
                'overheating_prevention': np.mean([1 if temp < self.temp_thresholds['critical'] 
                                                 else 0 for temp in max_temps])
            }
        }
        
        return effectiveness
    
    def update_thermal_model(self, 
                           training_data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """更新热模型（简化的在线学习）"""
        if not training_data:
            return {'error': 'No training data provided'}
        
        self.thermal_model.train()
        optimizer = torch.optim.Adam(self.thermal_model.parameters(), lr=0.001)
        
        total_loss = 0.0
        for inputs, targets in training_data:
            optimizer.zero_grad()
            
            predictions = self.thermal_model(inputs)
            loss = F.mse_loss(predictions['temperature_changes'], targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(training_data)
        
        return {
            'training_loss': avg_loss,
            'training_samples': len(training_data),
            'model_updated': True
        }
    
    def get_compensator_statistics(self) -> Dict[str, Any]:
        """获取补偿器统计信息"""
        effectiveness = self.evaluate_compensation_effectiveness()
        
        stats = {
            'compensator_id': self.compensator_id,
            'total_compensations': self.total_compensations,
            'hotspot_detections': self.hotspot_detections,
            'emergency_interventions': self.emergency_interventions,
            
            'temperature_thresholds': self.temp_thresholds,
            'compensation_gains': self.compensation_gains,
            
            'effectiveness_metrics': effectiveness,
            
            'current_status': {
                'latest_max_temp': self.temperature_history[-1].max_temperature if self.temperature_history else 0.0,
                'latest_temp_std': self.temperature_history[-1].temp_std if self.temperature_history else 0.0,
                'latest_hotspot_count': len(self.temperature_history[-1].hotspot_indices) if self.temperature_history else 0,
                'latest_urgency': self.compensation_history[-1].urgency_level if self.compensation_history else 0.0
            },
            
            'model_info': {
                'thermal_model_parameters': sum(p.numel() for p in self.thermal_model.parameters()),
                'prediction_cache_size': len(self.prediction_cache),
                'history_sizes': {
                    'temperature_history': len(self.temperature_history),
                    'compensation_history': len(self.compensation_history)
                }
            }
        }
        
        return stats
    
    def reset_compensator(self):
        """重置补偿器状态"""
        self.temperature_history.clear()
        self.compensation_history.clear()
        self.prediction_cache.clear()
        self.adaptive_controller.reset()
        
        self.total_compensations = 0
        self.hotspot_detections = 0
        self.emergency_interventions = 0
        
        print(f"🔄 温度补偿器已重置: {self.compensator_id}")
    
    def __str__(self) -> str:
        """字符串表示"""
        latest_temp = self.temperature_history[-1].max_temperature if self.temperature_history else 0.0
        return (f"TemperatureCompensator({self.compensator_id}): "
                f"compensations={self.total_compensations}, "
                f"latest_max_temp={latest_temp:.1f}℃, "
                f"emergencies={self.emergency_interventions}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"TemperatureCompensator(compensator_id='{self.compensator_id}', "
                f"num_cells={self.num_cells}, "
                f"compensations={self.total_compensations})")
