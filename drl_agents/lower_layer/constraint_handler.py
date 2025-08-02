import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import LowerLayerConfig
from config.model_config import ModelConfig

class ConstraintType(Enum):
    """约束类型枚举"""
    HARD = "hard"           # 硬约束（不可违反）
    SOFT = "soft"           # 软约束（可适度违反）
    ADAPTIVE = "adaptive"   # 自适应约束（根据情况调整）

class ViolationSeverity(Enum):
    """违反严重程度枚举"""
    NONE = 0       # 无违反
    MINOR = 1      # 轻微违反
    MODERATE = 2   # 中等违反
    SEVERE = 3     # 严重违反
    CRITICAL = 4   # 危急违反

@dataclass
class Constraint:
    """约束数据结构"""
    name: str
    constraint_type: ConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    penalty_weight: float = 1.0
    tolerance: float = 0.0
    enabled: bool = True
    priority: int = 1  # 1-10, 10最高优先级

@dataclass
class ConstraintViolation:
    """约束违反记录"""
    constraint_name: str
    violation_type: str  # "lower_bound", "upper_bound"
    current_value: float
    limit_value: float
    violation_amount: float
    severity: ViolationSeverity
    penalty: float
    timestamp: float = 0.0

class ConstraintProjector(nn.Module):
    """约束投影器"""
    
    def __init__(self, action_dim: int = 3, hidden_dim: int = 64):
        super(ConstraintProjector, self).__init__()
        
        self.action_dim = action_dim
        
        # 约束感知网络
        self.constraint_encoder = nn.Sequential(
            nn.Linear(action_dim + 10, hidden_dim),  # action + constraint_info
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # 投影网络
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, action_dim),
            nn.Tanh()
        )
        
        # 约束违反预测器
        self.violation_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),  # 5个严重程度等级
            nn.Softmax(dim=-1)
        )
    
    def forward(self, 
                action: torch.Tensor, 
                constraint_info: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        约束感知的动作投影
        
        Args:
            action: 原始动作 [batch_size, action_dim]
            constraint_info: 约束信息 [batch_size, constraint_dim]
        """
        # 组合输入
        combined_input = torch.cat([action, constraint_info], dim=-1)
        
        # 约束编码
        constraint_features = self.constraint_encoder(combined_input)
        
        # 投影动作
        projected_action = self.projector(constraint_features)
        
        # 预测违反概率
        violation_probs = self.violation_predictor(constraint_features)
        
        return {
            'projected_action': projected_action,
            'violation_probabilities': violation_probs,
            'constraint_features': constraint_features
        }

class BarrierFunction:
    """障碍函数（用于约束处理）"""
    
    def __init__(self, barrier_type: str = "logarithmic"):
        self.barrier_type = barrier_type
    
    def evaluate(self, 
                value: float, 
                min_bound: Optional[float] = None,
                max_bound: Optional[float] = None) -> float:
        """计算障碍函数值"""
        barrier_value = 0.0
        
        if self.barrier_type == "logarithmic":
            # 对数障碍函数
            if min_bound is not None and value > min_bound:
                barrier_value += -np.log(value - min_bound + 1e-8)
            
            if max_bound is not None and value < max_bound:
                barrier_value += -np.log(max_bound - value + 1e-8)
                
        elif self.barrier_type == "quadratic":
            # 二次障碍函数
            if min_bound is not None and value <= min_bound:
                barrier_value += (min_bound - value) ** 2
            
            if max_bound is not None and value >= max_bound:
                barrier_value += (value - max_bound) ** 2
                
        elif self.barrier_type == "exponential":
            # 指数障碍函数
            if min_bound is not None and value <= min_bound:
                barrier_value += np.exp(-(value - min_bound))
            
            if max_bound is not None and value >= max_bound:
                barrier_value += np.exp(value - max_bound)
        
        return barrier_value
    
    def gradient(self, 
                value: float, 
                min_bound: Optional[float] = None,
                max_bound: Optional[float] = None) -> float:
        """计算障碍函数梯度"""
        gradient = 0.0
        
        if self.barrier_type == "logarithmic":
            if min_bound is not None and value > min_bound:
                gradient += -1.0 / (value - min_bound + 1e-8)
            
            if max_bound is not None and value < max_bound:
                gradient += 1.0 / (max_bound - value + 1e-8)
                
        elif self.barrier_type == "quadratic":
            if min_bound is not None and value <= min_bound:
                gradient += -2 * (min_bound - value)
            
            if max_bound is not None and value >= max_bound:
                gradient += 2 * (value - max_bound)
                
        elif self.barrier_type == "exponential":
            if min_bound is not None and value <= min_bound:
                gradient += np.exp(-(value - min_bound))
            
            if max_bound is not None and value >= max_bound:
                gradient += np.exp(value - max_bound)
        
        return gradient

class ConstraintHandler(nn.Module):
    """
    约束处理器
    处理下层控制的各种约束，确保控制动作满足安全要求
    """
    
    def __init__(self,
                 config: LowerLayerConfig,
                 model_config: ModelConfig,
                 handler_id: str = "ConstraintHandler_001"):
        """
        初始化约束处理器
        
        Args:
            config: 下层配置
            model_config: 模型配置
            handler_id: 处理器ID
        """
        super(ConstraintHandler, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.handler_id = handler_id
        
        # === 约束定义 ===
        self.constraints = self._initialize_constraints()
        
        # === 神经网络组件 ===
        self.constraint_projector = ConstraintProjector(
            action_dim=model_config.lower_action_dim,
            hidden_dim=64
        )
        
        # === 障碍函数 ===
        self.barrier_function = BarrierFunction("logarithmic")
        
        # === 约束处理参数 ===
        self.penalty_coefficient = 100.0
        self.barrier_coefficient = 10.0
        self.projection_enabled = True
        self.adaptive_penalties = True
        
        # === 违反记录 ===
        self.violation_history: List[ConstraintViolation] = []
        self.total_violations = 0
        
        # === 统计信息 ===
        self.constraint_checks = 0
        self.successful_projections = 0
        
        print(f"✅ 约束处理器初始化完成: {handler_id}")
        print(f"   约束数量: {len(self.constraints)}")
    
    def _initialize_constraints(self) -> Dict[str, Constraint]:
        """初始化约束定义"""
        constraints = {}
        
        # === 功率约束 ===
        constraints['power_limit'] = Constraint(
            name='power_limit',
            constraint_type=ConstraintType.HARD,
            min_value=-50000.0,  # -50kW
            max_value=50000.0,   # 50kW
            penalty_weight=10.0,
            tolerance=1000.0,    # 1kW容差
            priority=10
        )
        
        # === 功率变化率约束 ===
        constraints['power_ramp_rate'] = Constraint(
            name='power_ramp_rate',
            constraint_type=ConstraintType.HARD,
            min_value=-10000.0,  # -10kW/s
            max_value=10000.0,   # 10kW/s
            penalty_weight=8.0,
            tolerance=500.0,     # 0.5kW/s容差
            priority=9
        )
        
        # === 响应时间约束 ===
        constraints['response_time'] = Constraint(
            name='response_time',
            constraint_type=ConstraintType.SOFT,
            min_value=0.001,     # 1ms最小响应时间
            max_value=0.1,       # 100ms最大响应时间
            penalty_weight=5.0,
            tolerance=0.01,      # 10ms容差
            priority=7
        )
        
        # === 控制平滑性约束 ===
        constraints['control_smoothness'] = Constraint(
            name='control_smoothness',
            constraint_type=ConstraintType.SOFT,
            min_value=0.0,
            max_value=1.0,       # 控制变化率归一化
            penalty_weight=3.0,
            tolerance=0.1,
            priority=5
        )
        
        # === 电流约束 ===
        constraints['current_limit'] = Constraint(
            name='current_limit',
            constraint_type=ConstraintType.HARD,
            min_value=-200.0,    # -200A
            max_value=200.0,     # 200A
            penalty_weight=9.0,
            tolerance=5.0,       # 5A容差
            priority=8
        )
        
        # === 温度约束 ===
        constraints['temperature_limit'] = Constraint(
            name='temperature_limit',
            constraint_type=ConstraintType.ADAPTIVE,
            min_value=-10.0,     # -10℃
            max_value=60.0,      # 60℃
            penalty_weight=7.0,
            tolerance=2.0,       # 2℃容差
            priority=6
        )
        
        return constraints
    
    def handle_constraints(self, 
                          action: torch.Tensor,
                          system_state: Dict[str, Any],
                          constraint_matrix: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        处理约束
        
        Args:
            action: 原始动作
            system_state: 系统状态
            constraint_matrix: 上层约束矩阵
            
        Returns:
            约束处理结果
        """
        self.constraint_checks += 1
        
        # === 1. 检查约束违反 ===
        violations = self._check_constraint_violations(action, system_state, constraint_matrix)
        
        # === 2. 计算约束信息 ===
        constraint_info = self._compute_constraint_info(system_state, constraint_matrix)
        
        # === 3. 动作投影 ===
        if self.projection_enabled and violations:
            projected_result = self._project_action(action, constraint_info, violations)
            final_action = projected_result['projected_action']
            projection_applied = True
        else:
            final_action = action
            projection_applied = False
        
        # === 4. 计算约束惩罚 ===
        constraint_penalty = self._calculate_constraint_penalty(violations)
        
        # === 5. 计算障碍函数值 ===
        barrier_value = self._calculate_barrier_function(final_action, system_state)
        
        # === 6. 记录违反历史 ===
        if violations:
            self._record_violations(violations)
        
        # === 7. 自适应调整 ===
        if self.adaptive_penalties:
            self._adapt_constraint_parameters(violations)
        
        result = {
            'constrained_action': final_action,
            'original_action': action,
            'constraint_penalty': constraint_penalty,
            'barrier_value': barrier_value,
            'violations': violations,
            'violation_count': len(violations),
            'projection_applied': projection_applied,
            'constraint_satisfaction_rate': self._calculate_satisfaction_rate(),
            'constraint_info': constraint_info,
            'safety_margin': self._calculate_safety_margin(final_action, system_state)
        }
        
        return result
    
    def _check_constraint_violations(self, 
                                   action: torch.Tensor,
                                   system_state: Dict[str, Any],
                                   constraint_matrix: Optional[torch.Tensor]) -> List[ConstraintViolation]:
        """检查约束违反"""
        violations = []
        
        # 转换动作为实际值
        action_values = self._action_to_physical_values(action, system_state)
        
        # 检查每个约束
        for constraint_name, constraint in self.constraints.items():
            if not constraint.enabled:
                continue
            
            # 获取当前值
            current_value = self._get_constraint_value(constraint_name, action_values, system_state)
            
            # 检查下界违反
            if constraint.min_value is not None:
                violation_amount = constraint.min_value - current_value
                if violation_amount > constraint.tolerance:
                    severity = self._determine_violation_severity(violation_amount, constraint)
                    penalty = self._calculate_violation_penalty(violation_amount, constraint)
                    
                    violation = ConstraintViolation(
                        constraint_name=constraint_name,
                        violation_type="lower_bound",
                        current_value=current_value,
                        limit_value=constraint.min_value,
                        violation_amount=violation_amount,
                        severity=severity,
                        penalty=penalty,
                        timestamp=self.constraint_checks
                    )
                    violations.append(violation)
            
            # 检查上界违反
            if constraint.max_value is not None:
                violation_amount = current_value - constraint.max_value
                if violation_amount > constraint.tolerance:
                    severity = self._determine_violation_severity(violation_amount, constraint)
                    penalty = self._calculate_violation_penalty(violation_amount, constraint)
                    
                    violation = ConstraintViolation(
                        constraint_name=constraint_name,
                        violation_type="upper_bound",
                        current_value=current_value,
                        limit_value=constraint.max_value,
                        violation_amount=violation_amount,
                        severity=severity,
                        penalty=penalty,
                        timestamp=self.constraint_checks
                    )
                    violations.append(violation)
        
        return violations
    
    def _action_to_physical_values(self, 
                                  action: torch.Tensor, 
                                  system_state: Dict[str, Any]) -> Dict[str, float]:
        """将动作转换为物理值"""
        # 假设动作为 [power_control, response_factor, compensation]
        action_np = action.detach().cpu().numpy()
        
        # 功率控制信号转换
        max_power_change = 10000.0  # 10kW/s
        power_change = action_np[0] * max_power_change * 0.01  # 10ms
        
        # 响应时间转换
        response_time = (action_np[1] + 1.0) / 2.0 * 0.1  # 0-100ms
        
        # 控制平滑性
        control_smoothness = abs(action_np[2])
        
        # 当前功率
        current_power = system_state.get('current_power', 0.0)
        new_power = current_power + power_change
        
        # 当前电流（简化计算）
        voltage = system_state.get('voltage', 3.4)
        current = new_power / (voltage * 100) if voltage > 0 else 0.0  # 假设100串联
        
        return {
            'power': new_power,
            'power_change_rate': power_change / 0.01,  # W/s
            'response_time': response_time,
            'control_smoothness': control_smoothness,
            'current': current,
            'temperature': system_state.get('temperature', 25.0)
        }
    
    def _get_constraint_value(self, 
                            constraint_name: str, 
                            action_values: Dict[str, float], 
                            system_state: Dict[str, Any]) -> float:
        """获取约束对应的当前值"""
        if constraint_name == 'power_limit':
            return action_values['power']
        elif constraint_name == 'power_ramp_rate':
            return action_values['power_change_rate']
        elif constraint_name == 'response_time':
            return action_values['response_time']
        elif constraint_name == 'control_smoothness':
            return action_values['control_smoothness']
        elif constraint_name == 'current_limit':
            return action_values['current']
        elif constraint_name == 'temperature_limit':
            return action_values['temperature']
        else:
            return 0.0
    
    def _determine_violation_severity(self, 
                                    violation_amount: float, 
                                    constraint: Constraint) -> ViolationSeverity:
        """确定违反严重程度"""
        if constraint.constraint_type == ConstraintType.HARD:
            if violation_amount > constraint.tolerance * 5:
                return ViolationSeverity.CRITICAL
            elif violation_amount > constraint.tolerance * 2:
                return ViolationSeverity.SEVERE
            else:
                return ViolationSeverity.MODERATE
        else:  # SOFT or ADAPTIVE
            if violation_amount > constraint.tolerance * 3:
                return ViolationSeverity.MODERATE
            else:
                return ViolationSeverity.MINOR
    
    def _calculate_violation_penalty(self, 
                                   violation_amount: float, 
                                   constraint: Constraint) -> float:
        """计算违反惩罚"""
        base_penalty = constraint.penalty_weight * (violation_amount ** 2)
        priority_multiplier = constraint.priority / 10.0
        
        return base_penalty * priority_multiplier
    
    def _compute_constraint_info(self, 
                               system_state: Dict[str, Any],
                               constraint_matrix: Optional[torch.Tensor]) -> torch.Tensor:
        """计算约束信息向量"""
        info = []
        
        # 系统状态信息
        info.extend([
            system_state.get('soc', 50.0) / 100.0,
            system_state.get('temperature', 25.0) / 60.0,
            system_state.get('voltage', 3.4) / 4.2,
            system_state.get('current_power', 0.0) / 50000.0,
            system_state.get('constraint_severity', 0.0)
        ])
        
        # 约束矩阵信息（如果提供）
        if constraint_matrix is not None:
            # 取前5个约束值
            matrix_values = constraint_matrix.flatten()[:5].tolist()
            info.extend(matrix_values)
        else:
            info.extend([0.5] * 5)  # 默认中等约束
        
        return torch.tensor(info, dtype=torch.float32).unsqueeze(0)
    
    def _project_action(self, 
                       action: torch.Tensor,
                       constraint_info: torch.Tensor,
                       violations: List[ConstraintViolation]) -> Dict[str, torch.Tensor]:
        """投影动作到可行域"""
        try:
            # 使用神经网络投影
            projection_result = self.constraint_projector(action, constraint_info)
            
            self.successful_projections += 1
            
            return projection_result
            
        except Exception as e:
            print(f"⚠️ 神经网络投影失败，使用数值投影: {str(e)}")
            return self._numerical_projection(action, violations)
    
    def _numerical_projection(self, 
                            action: torch.Tensor,
                            violations: List[ConstraintViolation]) -> Dict[str, torch.Tensor]:
        """数值投影方法"""
        projected_action = action.clone()
        
        # 简单的梯度投影
        for violation in violations:
            if violation.severity in [ViolationSeverity.SEVERE, ViolationSeverity.CRITICAL]:
                # 对严重违反进行强制投影
                constraint_gradient = self._compute_constraint_gradient(violation)
                projection_step = 0.1 * constraint_gradient
                projected_action = projected_action - projection_step
        
        # 确保动作在[-1, 1]范围内
        projected_action = torch.clamp(projected_action, -1.0, 1.0)
        
        return {
            'projected_action': projected_action,
            'violation_probabilities': torch.zeros(5),
            'constraint_features': torch.zeros(32)
        }
    
    def _compute_constraint_gradient(self, violation: ConstraintViolation) -> torch.Tensor:
        """计算约束梯度"""
        # 简化的梯度计算
        gradient = torch.zeros(3)  # 假设3维动作
        
        if violation.constraint_name == 'power_limit':
            gradient[0] = 0.1 if violation.violation_type == "upper_bound" else -0.1
        elif violation.constraint_name == 'response_time':
            gradient[1] = 0.1 if violation.violation_type == "upper_bound" else -0.1
        elif violation.constraint_name == 'control_smoothness':
            gradient[2] = 0.1 if violation.violation_type == "upper_bound" else -0.1
        
        return gradient
    
    def _calculate_constraint_penalty(self, violations: List[ConstraintViolation]) -> float:
        """计算总约束惩罚"""
        total_penalty = 0.0
        
        for violation in violations:
            penalty = violation.penalty
            
            # 严重程度加权
            severity_multiplier = {
                ViolationSeverity.MINOR: 1.0,
                ViolationSeverity.MODERATE: 2.0,
                ViolationSeverity.SEVERE: 5.0,
                ViolationSeverity.CRITICAL: 10.0
            }.get(violation.severity, 1.0)
            
            total_penalty += penalty * severity_multiplier
        
        return total_penalty * self.penalty_coefficient
    
    def _calculate_barrier_function(self, 
                                   action: torch.Tensor, 
                                   system_state: Dict[str, Any]) -> float:
        """计算障碍函数值"""
        total_barrier = 0.0
        
        action_values = self._action_to_physical_values(action, system_state)
        
        for constraint_name, constraint in self.constraints.items():
            if not constraint.enabled or constraint.constraint_type != ConstraintType.HARD:
                continue
            
            current_value = self._get_constraint_value(constraint_name, action_values, system_state)
            
            barrier_value = self.barrier_function.evaluate(
                current_value, constraint.min_value, constraint.max_value
            )
            
            total_barrier += barrier_value * constraint.penalty_weight
        
        return total_barrier * self.barrier_coefficient
    
    def _record_violations(self, violations: List[ConstraintViolation]):
        """记录违反历史"""
        self.violation_history.extend(violations)
        self.total_violations += len(violations)
        
        # 维护历史长度
        if len(self.violation_history) > 10000:
            excess = len(self.violation_history) - 10000
            self.violation_history = self.violation_history[excess:]
    
    def _adapt_constraint_parameters(self, violations: List[ConstraintViolation]):
        """自适应调整约束参数"""
        if not violations:
            return
        
        # 基于违反频率调整惩罚权重
        violation_counts = {}
        for violation in violations:
            name = violation.constraint_name
            violation_counts[name] = violation_counts.get(name, 0) + 1
        
        for constraint_name, count in violation_counts.items():
            if constraint_name in self.constraints:
                constraint = self.constraints[constraint_name]
                if constraint.constraint_type == ConstraintType.ADAPTIVE:
                    # 增加频繁违反约束的权重
                    constraint.penalty_weight = min(20.0, constraint.penalty_weight * 1.05)
    
    def _calculate_satisfaction_rate(self) -> float:
        """计算约束满足率"""
        if self.constraint_checks == 0:
            return 1.0
        
        recent_violations = len([v for v in self.violation_history[-100:] 
                               if v.severity in [ViolationSeverity.SEVERE, ViolationSeverity.CRITICAL]])
        recent_checks = min(100, self.constraint_checks)
        
        return max(0.0, 1.0 - recent_violations / recent_checks)
    
    def _calculate_safety_margin(self, 
                                action: torch.Tensor, 
                                system_state: Dict[str, Any]) -> float:
        """计算安全裕度"""
        action_values = self._action_to_physical_values(action, system_state)
        
        min_margin = float('inf')
        
        for constraint_name, constraint in self.constraints.items():
            if not constraint.enabled:
                continue
            
            current_value = self._get_constraint_value(constraint_name, action_values, system_state)
            
            # 计算到约束边界的距离
            if constraint.min_value is not None:
                margin = (current_value - constraint.min_value) / abs(constraint.min_value + 1e-8)
                min_margin = min(min_margin, margin)
            
            if constraint.max_value is not None:
                margin = (constraint.max_value - current_value) / abs(constraint.max_value + 1e-8)
                min_margin = min(min_margin, margin)
        
        return max(0.0, min_margin) if min_margin != float('inf') else 1.0
    
    def update_constraints_from_upper_layer(self, constraint_matrix: torch.Tensor) -> bool:
        """从上层更新约束"""
        try:
            # 解析约束矩阵
            if constraint_matrix.numel() >= 6:
                constraints_values = constraint_matrix.flatten()
                
                # 更新功率约束
                if len(constraints_values) > 0:
                    self.constraints['power_limit'].max_value = constraints_values[0].item() * 50000.0
                
                # 更新功率变化率约束
                if len(constraints_values) > 1:
                    self.constraints['power_ramp_rate'].max_value = constraints_values[1].item() * 10000.0
                
                # 更新响应时间约束
                if len(constraints_values) > 2:
                    self.constraints['response_time'].max_value = constraints_values[2].item() * 0.1
                
                # 更新温度约束
                if len(constraints_values) > 3:
                    self.constraints['temperature_limit'].max_value = 40.0 + constraints_values[3].item() * 20.0
                
                print(f"🔄 已从上层更新约束")
                return True
            
        except Exception as e:
            print(f"❌ 从上层更新约束失败: {str(e)}")
            return False
        
        return False
    
    def get_constraint_status(self) -> Dict[str, Any]:
        """获取约束状态"""
        active_constraints = {name: constraint for name, constraint in self.constraints.items() 
                            if constraint.enabled}
        
        recent_violations = self.violation_history[-100:] if len(self.violation_history) >= 100 else self.violation_history
        
        status = {
            'handler_id': self.handler_id,
            'total_constraint_checks': self.constraint_checks,
            'total_violations': self.total_violations,
            'successful_projections': self.successful_projections,
            'constraint_satisfaction_rate': self._calculate_satisfaction_rate(),
            
            'active_constraints_count': len(active_constraints),
            'constraint_types': {
                'hard': len([c for c in active_constraints.values() if c.constraint_type == ConstraintType.HARD]),
                'soft': len([c for c in active_constraints.values() if c.constraint_type == ConstraintType.SOFT]),
                'adaptive': len([c for c in active_constraints.values() if c.constraint_type == ConstraintType.ADAPTIVE])
            },
            
            'recent_violations': {
                'total': len(recent_violations),
                'by_severity': {
                    'minor': len([v for v in recent_violations if v.severity == ViolationSeverity.MINOR]),
                    'moderate': len([v for v in recent_violations if v.severity == ViolationSeverity.MODERATE]),
                    'severe': len([v for v in recent_violations if v.severity == ViolationSeverity.SEVERE]),
                    'critical': len([v for v in recent_violations if v.severity == ViolationSeverity.CRITICAL])
                }
            },
            
            'constraint_parameters': {
                'penalty_coefficient': self.penalty_coefficient,
                'barrier_coefficient': self.barrier_coefficient,
                'projection_enabled': self.projection_enabled,
                'adaptive_penalties': self.adaptive_penalties
            }
        }
        
        return status
    
    def __str__(self) -> str:
        """字符串表示"""
        satisfaction_rate = self._calculate_satisfaction_rate()
        return (f"ConstraintHandler({self.handler_id}): "
                f"checks={self.constraint_checks}, violations={self.total_violations}, "
                f"satisfaction={satisfaction_rate:.3f}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"ConstraintHandler(handler_id='{self.handler_id}', "
                f"constraints={len(self.constraints)}, "
                f"checks={self.constraint_checks})")
