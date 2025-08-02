import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.battery_params import BatteryParams
from config.system_config import SystemConfig

class ConstraintType(Enum):
    """约束类型枚举"""
    POWER = "power"                 # 功率约束
    CURRENT = "current"             # 电流约束
    VOLTAGE = "voltage"             # 电压约束
    TEMPERATURE = "temperature"     # 温度约束
    SOC = "soc"                    # SOC约束
    SOH = "soh"                    # SOH约束
    BALANCE = "balance"            # 均衡约束
    SAFETY = "safety"              # 安全约束
    RATE = "rate"                  # 变化率约束

class ViolationSeverity(Enum):
    """违约严重程度枚举"""
    NONE = "none"           # 无违约
    MINOR = "minor"         # 轻微违约
    MODERATE = "moderate"   # 中等违约
    SEVERE = "severe"       # 严重违约
    CRITICAL = "critical"   # 危急违约

@dataclass
class ConstraintLimit:
    """约束限制数据结构"""
    min_value: float = -float('inf')
    max_value: float = float('inf')
    soft_min: Optional[float] = None    # 软下限（预警）
    soft_max: Optional[float] = None    # 软上限（预警）
    rate_limit: Optional[float] = None  # 变化率限制
    
class ConstraintViolation:
    """约束违反记录"""
    def __init__(self, 
                 constraint_type: ConstraintType,
                 current_value: float,
                 limit_value: float,
                 severity: ViolationSeverity,
                 description: str,
                 timestamp: float = 0.0):
        self.constraint_type = constraint_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.severity = severity
        self.description = description
        self.timestamp = timestamp
        
    def __str__(self) -> str:
        return (f"{self.constraint_type.value}约束违反: "
                f"当前值{self.current_value:.3f} vs 限制{self.limit_value:.3f} "
                f"({self.severity.value})")

@dataclass
class ValidationResult:
    """验证结果数据结构"""
    is_valid: bool = True
    violations: List[ConstraintViolation] = field(default_factory=list)
    warnings: List[ConstraintViolation] = field(default_factory=list)
    overall_severity: ViolationSeverity = ViolationSeverity.NONE
    violation_count: int = 0
    warning_count: int = 0
    safety_score: float = 1.0
    
    def add_violation(self, violation: ConstraintViolation):
        """添加违约记录"""
        if violation.severity in [ViolationSeverity.SEVERE, ViolationSeverity.CRITICAL]:
            self.violations.append(violation)
            self.is_valid = False
            self.violation_count += 1
        else:
            self.warnings.append(violation)
            self.warning_count += 1
        
        # 更新总体严重程度
        if violation.severity.value == "critical":
            self.overall_severity = ViolationSeverity.CRITICAL
        elif violation.severity.value == "severe" and self.overall_severity.value != "critical":
            self.overall_severity = ViolationSeverity.SEVERE
        elif violation.severity.value == "moderate" and self.overall_severity.value in ["none", "minor"]:
            self.overall_severity = ViolationSeverity.MODERATE
        elif violation.severity.value == "minor" and self.overall_severity.value == "none":
            self.overall_severity = ViolationSeverity.MINOR

class ConstraintValidator:
    """
    约束验证器
    验证电池系统的各种物理和安全约束
    """
    
    def __init__(self, 
                 battery_params: BatteryParams,
                 system_config: SystemConfig,
                 validator_id: str = "ConstraintValidator_001"):
        """
        初始化约束验证器
        
        Args:
            battery_params: 电池参数
            system_config: 系统配置
            validator_id: 验证器ID
        """
        self.battery_params = battery_params
        self.system_config = system_config
        self.validator_id = validator_id
        
        # === 初始化约束限制 ===
        self.constraints = self._initialize_constraints()
        
        # === 验证历史 ===
        self.validation_history: List[ValidationResult] = []
        
        # === 统计信息 ===
        self.total_validations = 0
        self.total_violations = 0
        self.violation_by_type = {constraint_type: 0 for constraint_type in ConstraintType}
        
        # === 配置参数 ===
        self.enable_soft_constraints = True
        self.enable_rate_constraints = True
        self.safety_margin = 0.05  # 5%安全裕度
        
        print(f"✅ 约束验证器初始化完成: {validator_id}")
    
    def _initialize_constraints(self) -> Dict[ConstraintType, ConstraintLimit]:
        """初始化约束限制"""
        constraints = {}
        
        # === 功率约束 ===
        max_charge_power = self.battery_params.max_charge_power
        max_discharge_power = self.battery_params.max_discharge_power
        
        constraints[ConstraintType.POWER] = ConstraintLimit(
            min_value=-max_discharge_power,
            max_value=max_charge_power,
            soft_min=-max_discharge_power * 0.9,
            soft_max=max_charge_power * 0.9,
            rate_limit=max_charge_power * 0.1  # 10%/s功率变化率限制
        )
        
        # === 电流约束 ===
        max_charge_current = self.battery_params.max_charge_current
        max_discharge_current = self.battery_params.max_discharge_current
        
        constraints[ConstraintType.CURRENT] = ConstraintLimit(
            min_value=-max_discharge_current,
            max_value=max_charge_current,
            soft_min=-max_discharge_current * 0.9,
            soft_max=max_charge_current * 0.9,
            rate_limit=max_charge_current * 0.2  # 20%/s电流变化率限制
        )
        
        # === 电压约束 ===
        constraints[ConstraintType.VOLTAGE] = ConstraintLimit(
            min_value=self.battery_params.MIN_VOLTAGE,
            max_value=self.battery_params.MAX_VOLTAGE,
            soft_min=self.battery_params.MIN_VOLTAGE + 0.1,
            soft_max=self.battery_params.MAX_VOLTAGE - 0.1,
            rate_limit=0.1  # 0.1V/s电压变化率限制
        )
        
        # === 温度约束 ===
        constraints[ConstraintType.TEMPERATURE] = ConstraintLimit(
            min_value=self.battery_params.MIN_TEMP,
            max_value=self.battery_params.MAX_TEMP,
            soft_min=self.battery_params.OPTIMAL_TEMP_RANGE[0],
            soft_max=self.battery_params.OPTIMAL_TEMP_RANGE[1],
            rate_limit=5.0  # 5℃/min温度变化率限制
        )
        
        # === SOC约束 ===
        constraints[ConstraintType.SOC] = ConstraintLimit(
            min_value=self.battery_params.MIN_SOC,
            max_value=self.battery_params.MAX_SOC,
            soft_min=self.battery_params.MIN_SOC + 5.0,
            soft_max=self.battery_params.MAX_SOC - 5.0,
            rate_limit=10.0  # 10%/hour SOC变化率限制
        )
        
        # === SOH约束 ===
        constraints[ConstraintType.SOH] = ConstraintLimit(
            min_value=70.0,  # 70% EOL
            max_value=100.0,
            soft_min=80.0,   # 80%预警
            soft_max=100.0,
            rate_limit=1.0   # 1%/day SOH变化率限制
        )
        
        # === 均衡约束 ===
        constraints[ConstraintType.BALANCE] = ConstraintLimit(
            min_value=0.0,
            max_value=20.0,  # 20%最大不平衡度
            soft_min=0.0,
            soft_max=5.0,    # 5%预警阈值
            rate_limit=2.0   # 2%/min均衡速率限制
        )
        
        # === 安全约束 ===
        constraints[ConstraintType.SAFETY] = ConstraintLimit(
            min_value=0.0,   # 安全评分范围0-1
            max_value=1.0,
            soft_min=0.8,    # 0.8以下预警
            soft_max=1.0,
            rate_limit=0.1   # 0.1/min安全评分变化率
        )
        
        return constraints
    
    def validate_power_constraints(self, 
                                 current_power: float, 
                                 previous_power: Optional[float] = None,
                                 delta_t: float = 1.0) -> ValidationResult:
        """
        验证功率约束
        
        Args:
            current_power: 当前功率 (W)
            previous_power: 前一时刻功率 (W)
            delta_t: 时间间隔 (s)
            
        Returns:
            验证结果
        """
        result = ValidationResult()
        constraint = self.constraints[ConstraintType.POWER]
        
        # === 1. 功率范围约束 ===
        if current_power > constraint.max_value:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.POWER,
                current_value=current_power,
                limit_value=constraint.max_value,
                severity=ViolationSeverity.SEVERE,
                description=f"充电功率超限: {current_power:.1f}W > {constraint.max_value:.1f}W"
            )
            result.add_violation(violation)
        elif current_power > constraint.soft_max:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.POWER,
                current_value=current_power,
                limit_value=constraint.soft_max,
                severity=ViolationSeverity.MINOR,
                description=f"充电功率接近上限: {current_power:.1f}W"
            )
            result.add_violation(violation)
        
        if current_power < constraint.min_value:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.POWER,
                current_value=current_power,
                limit_value=constraint.min_value,
                severity=ViolationSeverity.SEVERE,
                description=f"放电功率超限: {current_power:.1f}W < {constraint.min_value:.1f}W"
            )
            result.add_violation(violation)
        elif current_power < constraint.soft_min:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.POWER,
                current_value=current_power,
                limit_value=constraint.soft_min,
                severity=ViolationSeverity.MINOR,
                description=f"放电功率接近下限: {current_power:.1f}W"
            )
            result.add_violation(violation)
        
        # === 2. 功率变化率约束 ===
        if (self.enable_rate_constraints and 
            previous_power is not None and 
            delta_t > 0):
            
            power_change_rate = abs(current_power - previous_power) / delta_t
            max_rate = constraint.rate_limit
            
            if power_change_rate > max_rate:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.RATE,
                    current_value=power_change_rate,
                    limit_value=max_rate,
                    severity=ViolationSeverity.MODERATE,
                    description=f"功率变化率过快: {power_change_rate:.1f}W/s > {max_rate:.1f}W/s"
                )
                result.add_violation(violation)
        
        return result
    
    def validate_temperature_constraints(self, 
                                       temperatures: List[float],
                                       previous_temps: Optional[List[float]] = None,
                                       delta_t: float = 1.0) -> ValidationResult:
        """
        验证温度约束
        
        Args:
            temperatures: 当前温度列表 (℃)
            previous_temps: 前一时刻温度列表 (℃)
            delta_t: 时间间隔 (s)
            
        Returns:
            验证结果
        """
        result = ValidationResult()
        constraint = self.constraints[ConstraintType.TEMPERATURE]
        
        max_temp = max(temperatures)
        min_temp = min(temperatures)
        avg_temp = np.mean(temperatures)
        temp_std = np.std(temperatures)
        
        # === 1. 温度范围约束 ===
        for i, temp in enumerate(temperatures):
            if temp > constraint.max_value:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.TEMPERATURE,
                    current_value=temp,
                    limit_value=constraint.max_value,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"单体{i}温度过高: {temp:.1f}℃ > {constraint.max_value:.1f}℃"
                )
                result.add_violation(violation)
            elif temp > constraint.soft_max:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.TEMPERATURE,
                    current_value=temp,
                    limit_value=constraint.soft_max,
                    severity=ViolationSeverity.MINOR,
                    description=f"单体{i}温度偏高: {temp:.1f}℃"
                )
                result.add_violation(violation)
            
            if temp < constraint.min_value:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.TEMPERATURE,
                    current_value=temp,
                    limit_value=constraint.min_value,
                    severity=ViolationSeverity.SEVERE,
                    description=f"单体{i}温度过低: {temp:.1f}℃ < {constraint.min_value:.1f}℃"
                )
                result.add_violation(violation)
        
        # === 2. 温度均匀性约束 ===
        max_temp_diff = max_temp - min_temp
        if max_temp_diff > 15.0:  # 15℃温差限制
            violation = ConstraintViolation(
                constraint_type=ConstraintType.BALANCE,
                current_value=max_temp_diff,
                limit_value=15.0,
                severity=ViolationSeverity.MODERATE,
                description=f"温度不均匀: 最大温差{max_temp_diff:.1f}℃"
            )
            result.add_violation(violation)
        
        # === 3. 温度变化率约束 ===
        if (self.enable_rate_constraints and 
            previous_temps is not None and 
            len(previous_temps) == len(temperatures) and
            delta_t > 0):
            
            for i, (curr_temp, prev_temp) in enumerate(zip(temperatures, previous_temps)):
                temp_change_rate = abs(curr_temp - prev_temp) / (delta_t / 60.0)  # ℃/min
                max_rate = constraint.rate_limit
                
                if temp_change_rate > max_rate:
                    violation = ConstraintViolation(
                        constraint_type=ConstraintType.RATE,
                        current_value=temp_change_rate,
                        limit_value=max_rate,
                        severity=ViolationSeverity.MODERATE,
                        description=f"单体{i}温升过快: {temp_change_rate:.1f}℃/min"
                    )
                    result.add_violation(violation)
        
        return result
    
    def validate_soc_constraints(self, 
                               soc_values: List[float],
                               previous_socs: Optional[List[float]] = None,
                               delta_t: float = 1.0) -> ValidationResult:
        """
        验证SOC约束
        
        Args:
            soc_values: SOC值列表 (%)
            previous_socs: 前一时刻SOC列表 (%)
            delta_t: 时间间隔 (s)
            
        Returns:
            验证结果
        """
        result = ValidationResult()
        constraint = self.constraints[ConstraintType.SOC]
        
        max_soc = max(soc_values)
        min_soc = min(soc_values)
        avg_soc = np.mean(soc_values)
        soc_std = np.std(soc_values)
        
        # === 1. SOC范围约束 ===
        for i, soc in enumerate(soc_values):
            if soc > constraint.max_value:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.SOC,
                    current_value=soc,
                    limit_value=constraint.max_value,
                    severity=ViolationSeverity.SEVERE,
                    description=f"单体{i} SOC过高: {soc:.1f}% > {constraint.max_value:.1f}%"
                )
                result.add_violation(violation)
            elif soc > constraint.soft_max:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.SOC,
                    current_value=soc,
                    limit_value=constraint.soft_max,
                    severity=ViolationSeverity.MINOR,
                    description=f"单体{i} SOC偏高: {soc:.1f}%"
                )
                result.add_violation(violation)
            
            if soc < constraint.min_value:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.SOC,
                    current_value=soc,
                    limit_value=constraint.min_value,
                    severity=ViolationSeverity.SEVERE,
                    description=f"单体{i} SOC过低: {soc:.1f}% < {constraint.min_value:.1f}%"
                )
                result.add_violation(violation)
            elif soc < constraint.soft_min:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.SOC,
                    current_value=soc,
                    limit_value=constraint.soft_min,
                    severity=ViolationSeverity.MINOR,
                    description=f"单体{i} SOC偏低: {soc:.1f}%"
                )
                result.add_violation(violation)
        
        # === 2. SOC均衡约束 ===
        balance_constraint = self.constraints[ConstraintType.BALANCE]
        
        if soc_std > balance_constraint.max_value:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.BALANCE,
                current_value=soc_std,
                limit_value=balance_constraint.max_value,
                severity=ViolationSeverity.SEVERE,
                description=f"SOC严重不平衡: σ={soc_std:.2f}% > {balance_constraint.max_value:.1f}%"
            )
            result.add_violation(violation)
        elif soc_std > balance_constraint.soft_max:
            violation = ConstraintViolation(
                constraint_type=ConstraintType.BALANCE,
                current_value=soc_std,
                limit_value=balance_constraint.soft_max,
                severity=ViolationSeverity.MINOR,
                description=f"SOC不平衡: σ={soc_std:.2f}%"
            )
            result.add_violation(violation)
        
        return result
    
    def validate_comprehensive_constraints(self, 
                                         system_state: Dict[str, Any],
                                         previous_state: Optional[Dict[str, Any]] = None,
                                         delta_t: float = 1.0) -> ValidationResult:
        """
        综合约束验证
        
        Args:
            system_state: 系统状态字典
            previous_state: 前一时刻系统状态
            delta_t: 时间间隔 (s)
            
        Returns:
            综合验证结果
        """
        comprehensive_result = ValidationResult()
        
        # === 1. 功率约束验证 ===
        if 'pack_power' in system_state:
            prev_power = previous_state.get('pack_power') if previous_state else None
            power_result = self.validate_power_constraints(
                system_state['pack_power'], prev_power, delta_t
            )
            comprehensive_result.violations.extend(power_result.violations)
            comprehensive_result.warnings.extend(power_result.warnings)
        
        # === 2. 温度约束验证 ===
        if 'temperatures' in system_state:
            prev_temps = previous_state.get('temperatures') if previous_state else None
            temp_result = self.validate_temperature_constraints(
                system_state['temperatures'], prev_temps, delta_t
            )
            comprehensive_result.violations.extend(temp_result.violations)
            comprehensive_result.warnings.extend(temp_result.warnings)
        
        # === 3. SOC约束验证 ===
        if 'soc_values' in system_state:
            prev_socs = previous_state.get('soc_values') if previous_state else None
            soc_result = self.validate_soc_constraints(
                system_state['soc_values'], prev_socs, delta_t
            )
            comprehensive_result.violations.extend(soc_result.violations)
            comprehensive_result.warnings.extend(soc_result.warnings)
        
        # === 4. 电压约束验证 ===
        if 'voltages' in system_state:
            voltage_result = self._validate_voltage_constraints(system_state['voltages'])
            comprehensive_result.violations.extend(voltage_result.violations)
            comprehensive_result.warnings.extend(voltage_result.warnings)
        
        # === 5. SOH约束验证 ===
        if 'soh_values' in system_state:
            soh_result = self._validate_soh_constraints(system_state['soh_values'])
            comprehensive_result.violations.extend(soh_result.violations)
            comprehensive_result.warnings.extend(soh_result.warnings)
        
        # === 6. 更新综合结果 ===
        comprehensive_result.violation_count = len(comprehensive_result.violations)
        comprehensive_result.warning_count = len(comprehensive_result.warnings)
        comprehensive_result.is_valid = comprehensive_result.violation_count == 0
        
        # 计算安全评分
        comprehensive_result.safety_score = self._calculate_safety_score(comprehensive_result)
        
        # 确定总体严重程度
        if comprehensive_result.violations:
            max_severity = max(v.severity for v in comprehensive_result.violations)
            comprehensive_result.overall_severity = max_severity
        elif comprehensive_result.warnings:
            max_severity = max(w.severity for w in comprehensive_result.warnings)
            comprehensive_result.overall_severity = max_severity
        
        # === 7. 记录历史和统计 ===
        self.validation_history.append(comprehensive_result)
        self.total_validations += 1
        self.total_violations += comprehensive_result.violation_count
        
        # 按类型统计违约
        for violation in comprehensive_result.violations:
            self.violation_by_type[violation.constraint_type] += 1
        
        # 维护历史长度
        max_history = 1000
        if len(self.validation_history) > max_history:
            self.validation_history.pop(0)
        
        return comprehensive_result
    
    def _validate_voltage_constraints(self, voltages: List[float]) -> ValidationResult:
        """验证电压约束"""
        result = ValidationResult()
        constraint = self.constraints[ConstraintType.VOLTAGE]
        
        for i, voltage in enumerate(voltages):
            if voltage > constraint.max_value:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.VOLTAGE,
                    current_value=voltage,
                    limit_value=constraint.max_value,
                    severity=ViolationSeverity.SEVERE,
                    description=f"单体{i}电压过高: {voltage:.3f}V > {constraint.max_value:.3f}V"
                )
                result.add_violation(violation)
            elif voltage > constraint.soft_max:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.VOLTAGE,
                    current_value=voltage,
                    limit_value=constraint.soft_max,
                    severity=ViolationSeverity.MINOR,
                    description=f"单体{i}电压偏高: {voltage:.3f}V"
                )
                result.add_violation(violation)
            
            if voltage < constraint.min_value:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.VOLTAGE,
                    current_value=voltage,
                    limit_value=constraint.min_value,
                    severity=ViolationSeverity.SEVERE,
                    description=f"单体{i}电压过低: {voltage:.3f}V < {constraint.min_value:.3f}V"
                )
                result.add_violation(violation)
            elif voltage < constraint.soft_min:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.VOLTAGE,
                    current_value=voltage,
                    limit_value=constraint.soft_min,
                    severity=ViolationSeverity.MINOR,
                    description=f"单体{i}电压偏低: {voltage:.3f}V"
                )
                result.add_violation(violation)
        
        return result
    
    def _validate_soh_constraints(self, soh_values: List[float]) -> ValidationResult:
        """验证SOH约束"""
        result = ValidationResult()
        constraint = self.constraints[ConstraintType.SOH]
        
        min_soh = min(soh_values)
        avg_soh = np.mean(soh_values)
        soh_std = np.std(soh_values)
        
        # SOH范围检查
        for i, soh in enumerate(soh_values):
            if soh < constraint.min_value:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.SOH,
                    current_value=soh,
                    limit_value=constraint.min_value,
                    severity=ViolationSeverity.CRITICAL,
                    description=f"单体{i} SOH过低: {soh:.1f}% < {constraint.min_value:.1f}%"
                )
                result.add_violation(violation)
            elif soh < constraint.soft_min:
                violation = ConstraintViolation(
                    constraint_type=ConstraintType.SOH,
                    current_value=soh,
                    limit_value=constraint.soft_min,
                    severity=ViolationSeverity.MINOR,
                    description=f"单体{i} SOH偏低: {soh:.1f}%"
                )
                result.add_violation(violation)
        
        # SOH一致性检查
        if soh_std > 10.0:  # 10% SOH差异限制
            violation = ConstraintViolation(
                constraint_type=ConstraintType.BALANCE,
                current_value=soh_std,
                limit_value=10.0,
                severity=ViolationSeverity.MODERATE,
                description=f"SOH不一致: σ={soh_std:.2f}%"
            )
            result.add_violation(violation)
        
        return result
    
    def _calculate_safety_score(self, result: ValidationResult) -> float:
        """计算安全评分 (0-1)"""
        base_score = 1.0
        
        # 违约惩罚
        for violation in result.violations:
            if violation.severity == ViolationSeverity.CRITICAL:
                base_score -= 0.3
            elif violation.severity == ViolationSeverity.SEVERE:
                base_score -= 0.2
            elif violation.severity == ViolationSeverity.MODERATE:
                base_score -= 0.1
        
        # 预警惩罚
        for warning in result.warnings:
            base_score -= 0.05
        
        return max(0.0, base_score)
    
    def get_constraint_status_summary(self) -> Dict[str, Any]:
        """获取约束状态摘要"""
        if not self.validation_history:
            return {'error': 'No validation history available'}
        
        latest_result = self.validation_history[-1]
        
        # 统计最近违约情况
        recent_results = self.validation_history[-10:] if len(self.validation_history) >= 10 else self.validation_history
        recent_violation_rate = sum(1 for r in recent_results if not r.is_valid) / len(recent_results)
        
        return {
            'validator_id': self.validator_id,
            'total_validations': self.total_validations,
            'total_violations': self.total_violations,
            'violation_rate': self.total_violations / self.total_validations if self.total_validations > 0 else 0.0,
            
            'latest_status': {
                'is_valid': latest_result.is_valid,
                'violation_count': latest_result.violation_count,
                'warning_count': latest_result.warning_count,
                'overall_severity': latest_result.overall_severity.value,
                'safety_score': latest_result.safety_score
            },
            
            'recent_performance': {
                'recent_violation_rate': recent_violation_rate,
                'avg_safety_score': np.mean([r.safety_score for r in recent_results])
            },
            
            'violation_by_type': {
                constraint_type.value: count 
                for constraint_type, count in self.violation_by_type.items()
            },
            
            'constraint_limits': {
                constraint_type.value: {
                    'min_value': limit.min_value,
                    'max_value': limit.max_value,
                    'soft_min': limit.soft_min,
                    'soft_max': limit.soft_max
                } for constraint_type, limit in self.constraints.items()
            }
        }
    
    def update_constraint_limits(self, 
                               constraint_type: ConstraintType, 
                               new_limits: ConstraintLimit) -> bool:
        """更新约束限制"""
        try:
            self.constraints[constraint_type] = new_limits
            print(f"✅ 已更新 {constraint_type.value} 约束限制")
            return True
        except Exception as e:
            print(f"❌ 更新约束限制失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        latest_status = "未知"
        if self.validation_history:
            latest_status = "有效" if self.validation_history[-1].is_valid else "违约"
        
        return (f"ConstraintValidator({self.validator_id}): "
                f"验证次数={self.total_validations}, "
                f"违约次数={self.total_violations}, "
                f"最新状态={latest_status}")
