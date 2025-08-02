import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import UpperLayerConfig
from config.model_config import ModelConfig

class ConstraintLevel(Enum):
    """约束等级枚举"""
    RELAXED = "relaxed"      # 宽松约束
    NORMAL = "normal"        # 正常约束
    STRICT = "strict"        # 严格约束
    EMERGENCY = "emergency"  # 紧急约束

@dataclass
class ConstraintMatrix:
    """约束矩阵数据结构"""
    # 功率约束
    max_charge_power: torch.Tensor = None      # 最大充电功率 (W)
    max_discharge_power: torch.Tensor = None   # 最大放电功率 (W)
    power_ramp_rate: torch.Tensor = None       # 功率变化率限制 (W/s)
    
    # 电流约束
    max_charge_current: torch.Tensor = None    # 最大充电电流 (A)
    max_discharge_current: torch.Tensor = None # 最大放电电流 (A)
    current_ramp_rate: torch.Tensor = None     # 电流变化率限制 (A/s)
    
    # 电压约束
    max_voltage: torch.Tensor = None           # 最大电压 (V)
    min_voltage: torch.Tensor = None           # 最小电压 (V)
    
    # 温度约束
    max_temperature: torch.Tensor = None       # 最大温度 (℃)
    min_temperature: torch.Tensor = None       # 最小温度 (℃)
    temp_change_rate: torch.Tensor = None      # 温度变化率限制 (℃/min)
    
    # SOC约束
    max_soc: torch.Tensor = None               # 最大SOC (%)
    min_soc: torch.Tensor = None               # 最小SOC (%)
    soc_balance_tolerance: torch.Tensor = None # SOC不平衡容忍度 (%)
    
    # 响应时间约束
    response_time_limit: torch.Tensor = None   # 响应时间限制 (s)
    
    # 安全裕度
    safety_margin: torch.Tensor = None         # 安全裕度 [0,1]
    
    # 约束等级
    constraint_level: ConstraintLevel = ConstraintLevel.NORMAL
    
    def to_matrix(self) -> torch.Tensor:
        """转换为矩阵形式"""
        constraints = []
        
        # 按顺序添加约束
        if self.max_charge_power is not None:
            constraints.append(self.max_charge_power.unsqueeze(-1))
        if self.max_discharge_power is not None:
            constraints.append(self.max_discharge_power.unsqueeze(-1))
        if self.max_charge_current is not None:
            constraints.append(self.max_charge_current.unsqueeze(-1))
        if self.max_discharge_current is not None:
            constraints.append(self.max_discharge_current.unsqueeze(-1))
        if self.max_temperature is not None:
            constraints.append(self.max_temperature.unsqueeze(-1))
        if self.min_temperature is not None:
            constraints.append(self.min_temperature.unsqueeze(-1))
        if self.max_soc is not None:
            constraints.append(self.max_soc.unsqueeze(-1))
        if self.min_soc is not None:
            constraints.append(self.min_soc.unsqueeze(-1))
        if self.soc_balance_tolerance is not None:
            constraints.append(self.soc_balance_tolerance.unsqueeze(-1))
        if self.response_time_limit is not None:
            constraints.append(self.response_time_limit.unsqueeze(-1))
        
        if constraints:
            return torch.cat(constraints, dim=-1)
        else:
            return torch.zeros(1, 10)  # 默认10个约束

class PowerConstraintGenerator(nn.Module):
    """功率约束生成器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(PowerConstraintGenerator, self).__init__()
        
        self.power_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 32)
        )
        
        # 功率限制预测
        self.power_limits = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   # [charge_power, discharge_power]
            nn.Sigmoid()
        )
        
        # 功率变化率限制
        self.ramp_rate_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, base_power: float = 50000.0) -> Dict[str, torch.Tensor]:
        """
        生成功率约束
        
        Args:
            x: 输入特征
            base_power: 基础功率 (W)
        """
        features = self.power_network(x)
        
        power_limits = self.power_limits(features)
        ramp_rate = self.ramp_rate_predictor(features)
        
        # 转换为实际功率值
        max_charge_power = power_limits[:, 0] * base_power
        max_discharge_power = power_limits[:, 1] * base_power
        power_ramp_rate = ramp_rate.squeeze(-1) * base_power * 0.1  # 10%/s最大变化率
        
        return {
            'max_charge_power': max_charge_power,
            'max_discharge_power': max_discharge_power,
            'power_ramp_rate': power_ramp_rate
        }

class ThermalConstraintGenerator(nn.Module):
    """温度约束生成器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ThermalConstraintGenerator, self).__init__()
        
        self.thermal_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 32)
        )
        
        # 温度限制预测
        self.temp_limits = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   # [max_temp, min_temp]
        )
        
        # 温度变化率限制
        self.temp_change_rate_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成温度约束"""
        features = self.thermal_network(x)
        
        temp_limits = self.temp_limits(features)
        temp_change_rate = self.temp_change_rate_predictor(features)
        
        # 转换为实际温度值
        max_temperature = torch.sigmoid(temp_limits[:, 0]) * 30.0 + 35.0  # 35-65℃
        min_temperature = torch.sigmoid(temp_limits[:, 1]) * 30.0 - 10.0  # -10-20℃
        temp_change_rate_limit = temp_change_rate.squeeze(-1) * 10.0  # 最大10℃/min
        
        return {
            'max_temperature': max_temperature,
            'min_temperature': min_temperature,
            'temp_change_rate': temp_change_rate_limit
        }

class SOCConstraintGenerator(nn.Module):
    """SOC约束生成器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(SOCConstraintGenerator, self).__init__()
        
        self.soc_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 32)
        )
        
        # SOC限制预测
        self.soc_limits = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   # [max_soc, min_soc]
            nn.Sigmoid()
        )
        
        # SOC平衡容忍度
        self.balance_tolerance = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成SOC约束"""
        features = self.soc_network(x)
        
        soc_limits = self.soc_limits(features)
        balance_tolerance = self.balance_tolerance(features)
        
        # 转换为实际SOC值
        max_soc = soc_limits[:, 0] * 20.0 + 80.0     # 80-100%
        min_soc = soc_limits[:, 1] * 20.0            # 0-20%
        soc_balance_tolerance = balance_tolerance.squeeze(-1) * 10.0  # 0-10%
        
        return {
            'max_soc': max_soc,
            'min_soc': min_soc,
            'soc_balance_tolerance': soc_balance_tolerance
        }

class ResponseTimeConstraintGenerator(nn.Module):
    """响应时间约束生成器"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ResponseTimeConstraintGenerator, self).__init__()
        
        self.response_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 16)
        )
        
        # 响应时间预测
        self.response_time_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 安全裕度预测
        self.safety_margin_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """生成响应时间约束"""
        features = self.response_network(x)
        
        response_time = self.response_time_predictor(features)
        safety_margin = self.safety_margin_predictor(features)
        
        # 转换为实际时间值
        response_time_limit = response_time.squeeze(-1) * 30.0 + 0.1  # 0.1-30.1s
        safety_margin_value = safety_margin.squeeze(-1) * 0.3 + 0.05  # 5%-35%
        
        return {
            'response_time_limit': response_time_limit,
            'safety_margin': safety_margin_value
        }

class ConstraintGenerator(nn.Module):
    """
    约束矩阵生成器
    根据系统状态动态生成约束矩阵C_t，为下层DRL提供约束边界
    """
    
    def __init__(self,
                 config: UpperLayerConfig,
                 model_config: ModelConfig,
                 generator_id: str = "ConstraintGenerator_001"):
        """
        初始化约束生成器
        
        Args:
            config: 上层配置
            model_config: 模型配置
            generator_id: 生成器ID
        """
        super(ConstraintGenerator, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.generator_id = generator_id
        
        # === 模型参数 ===
        self.input_dim = model_config.upper_state_dim
        self.hidden_dim = config.hidden_dim
        
        # === 特征提取器 ===
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate)
        )
        
        # === 约束生成器 ===
        self.power_generator = PowerConstraintGenerator(self.hidden_dim)
        self.thermal_generator = ThermalConstraintGenerator(self.hidden_dim)
        self.soc_generator = SOCConstraintGenerator(self.hidden_dim)
        self.response_generator = ResponseTimeConstraintGenerator(self.hidden_dim)
        
        # === 约束级别分类器 ===
        self.constraint_level_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4个约束等级
            nn.Softmax(dim=-1)
        )
        
        # === 约束适应器 ===
        self.constraint_adapter = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, 64),  # hidden_dim + constraint_level_one_hot
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)  # 10个约束调整因子
        )
        
        # === 历史记录 ===
        self.constraint_history: List[Dict] = []
        
        print(f"✅ 约束生成器初始化完成: {generator_id}")
    
    def forward(self, 
                state: torch.Tensor,
                base_constraints: Optional[Dict[str, float]] = None,
                constraint_level: Optional[ConstraintLevel] = None) -> ConstraintMatrix:
        """
        生成约束矩阵
        
        Args:
            state: 输入状态 [batch_size, state_dim]
            base_constraints: 基础约束参数
            constraint_level: 指定约束等级
            
        Returns:
            约束矩阵
        """
        # === 1. 特征提取 ===
        features = self.feature_extractor(state)
        
        # === 2. 约束等级分类 ===
        if constraint_level is None:
            level_probs = self.constraint_level_classifier(features)
            level_indices = torch.argmax(level_probs, dim=-1)
            
            # 转换为约束等级
            level_map = [ConstraintLevel.RELAXED, ConstraintLevel.NORMAL, 
                        ConstraintLevel.STRICT, ConstraintLevel.EMERGENCY]
            constraint_level = level_map[level_indices[0].item()]  # 取第一个样本的等级
        
        # === 3. 生成基础约束 ===
        if base_constraints is None:
            base_constraints = {
                'base_power': 50000.0,    # 50kW
                'base_current': 100.0,    # 100A
                'base_voltage': 3.4,      # 3.4V
                'base_temp': 45.0,        # 45℃
                'base_soc': 50.0          # 50%
            }
        
        # 功率约束
        power_constraints = self.power_generator(features, base_constraints['base_power'])
        
        # 温度约束
        thermal_constraints = self.thermal_generator(features)
        
        # SOC约束
        soc_constraints = self.soc_generator(features)
        
        # 响应时间约束
        response_constraints = self.response_generator(features)
        
        # === 4. 约束等级调整 ===
        level_one_hot = torch.zeros(features.shape[0], 4, device=features.device)
        level_one_hot[:, level_indices] = 1.0
        
        adapter_input = torch.cat([features, level_one_hot], dim=-1)
        adjustment_factors = torch.sigmoid(self.constraint_adapter(adapter_input))
        
        # === 5. 应用调整因子 ===
        adjusted_constraints = self._apply_adjustment_factors(
            power_constraints, thermal_constraints, soc_constraints, 
            response_constraints, adjustment_factors, constraint_level
        )
        
        # === 6. 构建约束矩阵 ===
        constraint_matrix = ConstraintMatrix(
            max_charge_power=adjusted_constraints['max_charge_power'],
            max_discharge_power=adjusted_constraints['max_discharge_power'],
            power_ramp_rate=adjusted_constraints['power_ramp_rate'],
            max_temperature=adjusted_constraints['max_temperature'],
            min_temperature=adjusted_constraints['min_temperature'],
            temp_change_rate=adjusted_constraints['temp_change_rate'],
            max_soc=adjusted_constraints['max_soc'],
            min_soc=adjusted_constraints['min_soc'],
            soc_balance_tolerance=adjusted_constraints['soc_balance_tolerance'],
            response_time_limit=adjusted_constraints['response_time_limit'],
            safety_margin=adjusted_constraints['safety_margin'],
            constraint_level=constraint_level
        )
        
        return constraint_matrix

    def generate_bms_cluster_constraints(self, cluster_record: Dict) -> Dict[str, Any]:
        """
        生成BMS集群约束矩阵
        
        Args:
            cluster_record: BMS集群记录
            
        Returns:
            BMS集群约束矩阵和相关信息
        """
        
        constraint_result = {
            'generation_timestamp': time.time(),
            'cluster_id': cluster_record.get('cluster_id', 'unknown'),
            
            # === 系统级约束矩阵 ===
            'system_constraints': self._generate_system_level_constraints(cluster_record),
            
            # === BMS级约束矩阵 ===
            'bms_constraints': self._generate_bms_level_constraints(cluster_record),
            
            # === 协调约束 ===
            'coordination_constraints': self._generate_coordination_constraints(cluster_record),
            
            # === 动态约束调整 ===
            'adaptive_constraints': self._generate_adaptive_constraints(cluster_record)
        }
        
        return constraint_result
    
    def _generate_system_level_constraints(self, cluster_record: Dict) -> Dict[str, Any]:
        """生成系统级约束"""
        
        system_constraints = {
            # 系统功率约束
            'power_constraints': {
                'max_total_charge_power': self._calculate_system_max_charge_power(cluster_record),
                'max_total_discharge_power': self._calculate_system_max_discharge_power(cluster_record),
                'power_ramp_rate_limit': self._calculate_power_ramp_limits(cluster_record)
            },
            
            # 系统均衡约束
            'balance_constraints': {
                'max_inter_bms_soc_std': 12.0,     # BMS间SOC标准差限制
                'max_inter_bms_temp_std': 18.0,    # BMS间温度标准差限制
                'min_system_efficiency': 0.85      # 最小系统效率
            },
            
            # 系统安全约束
            'safety_constraints': {
                'max_system_temp': cluster_record.get('system_avg_temp', 25.0) + 20.0,
                'min_system_soc': 10.0,
                'max_system_soc': 90.0,
                'max_concurrent_alarms': 2         # 最大同时报警BMS数量
            }
        }
        
        return system_constraints
    
    def _generate_bms_level_constraints(self, cluster_record: Dict) -> List[Dict]:
        """生成各BMS级约束"""
        
        bms_records = cluster_record.get('bms_records', [])
        bms_constraints = []
        
        for bms_record in bms_records:
            bms_id = bms_record.get('bms_id', 'unknown')
            
            # 单个BMS约束
            bms_constraint = {
                'bms_id': bms_id,
                
                # 功率约束
                'power_constraints': {
                    'max_charge_power': self._calculate_bms_max_charge_power(bms_record),
                    'max_discharge_power': self._calculate_bms_max_discharge_power(bms_record),
                    'power_derating_factor': self._calculate_power_derating_factor(bms_record)
                },
                
                # SOC约束
                'soc_constraints': {
                    'min_soc': max(5.0, bms_record.get('avg_soc', 50.0) - 30.0),
                    'max_soc': min(95.0, bms_record.get('avg_soc', 50.0) + 30.0),
                    'max_soc_std': 5.0  # BMS内SOC标准差限制
                },
                
                # 温度约束
                'temp_constraints': {
                    'max_temp': 55.0,
                    'min_temp': -10.0,
                    'max_temp_rise_rate': 3.0,  # ℃/min
                    'max_temp_std': 8.0         # BMS内温度标准差限制
                },
                
                # 均衡约束
                'balancing_constraints': {
                    'max_balancing_power': bms_record.get('bms_max_charge_power', 100000) * 0.05,  # 5%的BMS功率
                    'max_balancing_duration': 3600.0,  # 1小时
                    'balancing_efficiency_threshold': 0.8
                }
            }
            
            bms_constraints.append(bms_constraint)
        
        return bms_constraints
    
    def _generate_coordination_constraints(self, cluster_record: Dict) -> Dict[str, Any]:
        """生成协调约束"""
        
        coordination_constraints = {
            # 协调频率约束
            'coordination_frequency': {
                'min_coordination_interval': 10.0,      # 最小协调间隔(s)
                'max_coordination_frequency': 6,        # 每分钟最大协调次数
                'coordination_timeout': 300.0           # 协调超时时间(s)
            },
            
            # 功率分配约束
            'power_allocation': {
                'max_power_bias': 0.3,                  # 最大功率偏置
                'min_bms_power_ratio': 0.05,            # BMS最小功率分配比例
                'allocation_balance_tolerance': 0.02    # 分配平衡容差
            },
            
            # 协调强度约束
            'coordination_intensity': {
                'max_simultaneous_coordinations': 5,    # 最大同时协调BMS数量
                'coordination_priority_levels': ['low', 'medium', 'high', 'critical'],
                'max_coordination_bias_per_bms': 0.25   # 单个BMS最大协调偏置
            }
        }
        
        return coordination_constraints
    
    def _generate_adaptive_constraints(self, cluster_record: Dict) -> Dict[str, Any]:
        """生成自适应约束"""
        
        # 基于当前系统状态动态调整约束
        system_health = cluster_record.get('system_health_status', 'Good')
        overall_balance = cluster_record.get('cluster_metrics', {}).get('overall_balance_score', 0.8)
        
        adaptive_constraints = {
            'constraint_adaptation_enabled': True,
            'adaptation_factors': {},
            'dynamic_limits': {}
        }
        
        # 基于系统健康状态调整
        if system_health == 'Critical':
            adaptive_constraints['adaptation_factors']['power_derating'] = 0.5
            adaptive_constraints['adaptation_factors']['coordination_intensity'] = 0.3
        elif system_health == 'Poor':
            adaptive_constraints['adaptation_factors']['power_derating'] = 0.7
            adaptive_constraints['adaptation_factors']['coordination_intensity'] = 0.6
        elif system_health == 'Fair':
            adaptive_constraints['adaptation_factors']['power_derating'] = 0.85
            adaptive_constraints['adaptation_factors']['coordination_intensity'] = 0.8
        else:  # Good
            adaptive_constraints['adaptation_factors']['power_derating'] = 1.0
            adaptive_constraints['adaptation_factors']['coordination_intensity'] = 1.0
        
        # 基于系统均衡状态调整
        if overall_balance < 0.5:
            adaptive_constraints['dynamic_limits']['enhanced_coordination'] = True
            adaptive_constraints['dynamic_limits']['relaxed_power_limits'] = True
        elif overall_balance > 0.9:
            adaptive_constraints['dynamic_limits']['reduced_coordination'] = True
            adaptive_constraints['dynamic_limits']['standard_power_limits'] = True
        
        return adaptive_constraints
    
    def get_constraint_matrix_for_upper_drl(self, cluster_record: Dict) -> np.ndarray:
        """
        为上层DRL生成约束矩阵 C_t
        
        Args:
            cluster_record: BMS集群记录
            
        Returns:
            约束矩阵 (n×m)
        """
        
        # 生成完整约束
        constraints = self.generate_bms_cluster_constraints(cluster_record)
        
        # 构造约束矩阵
        # 行：约束类型，列：BMS编号
        num_bms = len(cluster_record.get('bms_records', []))
        
        # 初始化约束矩阵 (约束类型 × BMS数量)
        constraint_matrix = np.zeros((8, num_bms))  # 8种主要约束类型
        
        bms_constraints = constraints['bms_constraints']
        
        for i, bms_constraint in enumerate(bms_constraints):
            if i >= num_bms:
                break
                
            # 第0行：最大充电功率约束 (归一化)
            constraint_matrix[0, i] = bms_constraint['power_constraints']['max_charge_power'] / 200000.0
            
            # 第1行：最大放电功率约束 (归一化)
            constraint_matrix[1, i] = bms_constraint['power_constraints']['max_discharge_power'] / 600000.0
            
            # 第2行：SOC上限约束
            constraint_matrix[2, i] = bms_constraint['soc_constraints']['max_soc'] / 100.0
            
            # 第3行：SOC下限约束
            constraint_matrix[3, i] = bms_constraint['soc_constraints']['min_soc'] / 100.0
            
            # 第4行：温度上限约束
            constraint_matrix[4, i] = bms_constraint['temp_constraints']['max_temp'] / 70.0
            
            # 第5行：功率降额因子
            constraint_matrix[5, i] = bms_constraint['power_constraints']['power_derating_factor']
            
            # 第6行：BMS内SOC不平衡约束
            constraint_matrix[6, i] = bms_constraint['soc_constraints']['max_soc_std'] / 10.0
            
            # 第7行：BMS内温度不平衡约束
            constraint_matrix[7, i] = bms_constraint['temp_constraints']['max_temp_std'] / 15.0
        
        return constraint_matrix
    
    # 辅助计算方法
    def _calculate_bms_max_charge_power(self, bms_record: Dict) -> float:
        """计算BMS最大充电功率"""
        base_power = bms_record.get('bms_max_charge_power', 100000.0)
        
        # 温度降额
        avg_temp = bms_record.get('avg_temperature', 25.0)
        if avg_temp > 45.0:
            temp_derating = max(0.5, (60.0 - avg_temp) / 15.0)
        elif avg_temp < 10.0:
            temp_derating = max(0.5, (avg_temp + 10.0) / 20.0)
        else:
            temp_derating = 1.0
        
        # SOC降额
        avg_soc = bms_record.get('avg_soc', 50.0)
        if avg_soc > 85.0:
            soc_derating = max(0.3, (95.0 - avg_soc) / 10.0)
        else:
            soc_derating = 1.0
        
        # 健康状态降额
        health_status = bms_record.get('health_status', 'Good')
        health_derating = {'Critical': 0.3, 'Poor': 0.6, 'Fair': 0.8, 'Good': 1.0}.get(health_status, 1.0)
        
        return base_power * temp_derating * soc_derating * health_derating
    
    def _calculate_power_derating_factor(self, bms_record: Dict) -> float:
        """计算功率降额因子"""
        temp_factor = self._get_temperature_derating_factor(bms_record.get('avg_temperature', 25.0))
        soc_factor = self._get_soc_derating_factor(bms_record.get('avg_soc', 50.0))
        health_factor = self._get_health_derating_factor(bms_record.get('health_status', 'Good'))
        
        return min(temp_factor, soc_factor, health_factor)
    
    def _apply_adjustment_factors(self, 
                                power_constraints: Dict[str, torch.Tensor],
                                thermal_constraints: Dict[str, torch.Tensor],
                                soc_constraints: Dict[str, torch.Tensor],
                                response_constraints: Dict[str, torch.Tensor],
                                adjustment_factors: torch.Tensor,
                                constraint_level: ConstraintLevel) -> Dict[str, torch.Tensor]:
        """应用约束等级调整因子"""
        
        # 等级调整系数
        level_multipliers = {
            ConstraintLevel.RELAXED: 1.2,    # 放松20%
            ConstraintLevel.NORMAL: 1.0,     # 无调整
            ConstraintLevel.STRICT: 0.8,     # 收紧20%
            ConstraintLevel.EMERGENCY: 0.6   # 收紧40%
        }
        
        level_mult = level_multipliers[constraint_level]
        
        adjusted = {}
        
        # 功率约束调整
        adjusted['max_charge_power'] = power_constraints['max_charge_power'] * level_mult * adjustment_factors[:, 0]
        adjusted['max_discharge_power'] = power_constraints['max_discharge_power'] * level_mult * adjustment_factors[:, 1]
        adjusted['power_ramp_rate'] = power_constraints['power_ramp_rate'] * level_mult * adjustment_factors[:, 2]
        
        # 温度约束调整
        adjusted['max_temperature'] = thermal_constraints['max_temperature'] - (1.0 - level_mult) * 10.0 * adjustment_factors[:, 3]
        adjusted['min_temperature'] = thermal_constraints['min_temperature'] + (1.0 - level_mult) * 5.0 * adjustment_factors[:, 4]
        adjusted['temp_change_rate'] = thermal_constraints['temp_change_rate'] * level_mult * adjustment_factors[:, 5]
        
        # SOC约束调整
        adjusted['max_soc'] = soc_constraints['max_soc'] - (1.0 - level_mult) * 5.0 * adjustment_factors[:, 6]
        adjusted['min_soc'] = soc_constraints['min_soc'] + (1.0 - level_mult) * 5.0 * adjustment_factors[:, 7]
        adjusted['soc_balance_tolerance'] = soc_constraints['soc_balance_tolerance'] * level_mult * adjustment_factors[:, 8]
        
        # 响应时间约束调整
        adjusted['response_time_limit'] = response_constraints['response_time_limit'] * (2.0 - level_mult) * adjustment_factors[:, 9]
        adjusted['safety_margin'] = response_constraints['safety_margin'] * (2.0 - level_mult)
        
        return adjusted
    
    def generate_adaptive_constraints(self, 
                                    state: torch.Tensor,
                                    safety_status: Dict[str, Any],
                                    performance_metrics: Dict[str, float]) -> ConstraintMatrix:
        """
        生成自适应约束
        
        Args:
            state: 系统状态
            safety_status: 安全状态
            performance_metrics: 性能指标
            
        Returns:
            自适应约束矩阵
        """
        # 根据安全状态确定约束等级
        constraint_level = ConstraintLevel.NORMAL
        
        if safety_status.get('critical_violations', 0) > 0:
            constraint_level = ConstraintLevel.EMERGENCY
        elif safety_status.get('safety_score', 1.0) < 0.7:
            constraint_level = ConstraintLevel.STRICT
        elif performance_metrics.get('efficiency', 1.0) > 0.95:
            constraint_level = ConstraintLevel.RELAXED
        
        # 生成约束矩阵
        constraints = self.forward(state, constraint_level=constraint_level)
        
        # 记录历史
        self._record_constraint_generation(constraints, safety_status, performance_metrics)
        
        return constraints
    
    def _record_constraint_generation(self, 
                                    constraints: ConstraintMatrix,
                                    safety_status: Dict[str, Any],
                                    performance_metrics: Dict[str, float]):
        """记录约束生成历史"""
        record = {
            'timestamp': len(self.constraint_history),
            'constraint_level': constraints.constraint_level.value,
            'max_charge_power': constraints.max_charge_power.item() if constraints.max_charge_power is not None else 0.0,
            'max_discharge_power': constraints.max_discharge_power.item() if constraints.max_discharge_power is not None else 0.0,
            'max_temperature': constraints.max_temperature.item() if constraints.max_temperature is not None else 0.0,
            'response_time_limit': constraints.response_time_limit.item() if constraints.response_time_limit is not None else 0.0,
            'safety_score': safety_status.get('safety_score', 1.0),
            'efficiency': performance_metrics.get('efficiency', 1.0)
        }
        
        self.constraint_history.append(record)
        
        # 维护历史长度
        if len(self.constraint_history) > 1000:
            self.constraint_history.pop(0)
    
    def analyze_constraint_evolution(self) -> Dict[str, Any]:
        """分析约束演化趋势"""
        if len(self.constraint_history) < 10:
            return {'error': 'Insufficient constraint history'}
        
        recent_history = self.constraint_history[-50:]
        
        # 计算约束严格程度趋势
        level_severity = {
            'relaxed': 1, 'normal': 2, 'strict': 3, 'emergency': 4
        }
        
        severity_values = [level_severity[record['constraint_level']] for record in recent_history]
        avg_severity = np.mean(severity_values)
        
        # 计算约束变化频率
        level_changes = sum(1 for i in range(1, len(recent_history)) 
                           if recent_history[i]['constraint_level'] != recent_history[i-1]['constraint_level'])
        
        change_frequency = level_changes / len(recent_history)
        
        # 分析功率约束趋势
        power_values = [record['max_charge_power'] for record in recent_history]
        power_trend = 'increasing' if np.polyfit(range(len(power_values)), power_values, 1)[0] > 0 else 'decreasing'
        
        return {
            'avg_constraint_severity': avg_severity,
            'constraint_change_frequency': change_frequency,
            'power_constraint_trend': power_trend,
            'recent_constraint_levels': [record['constraint_level'] for record in recent_history[-10:]],
            'constraint_stability': 1.0 - change_frequency
        }
    
    def get_constraint_recommendations(self, 
                                     current_constraints: ConstraintMatrix,
                                     system_performance: Dict[str, float]) -> List[str]:
        """获取约束优化建议"""
        recommendations = []
        
        # 基于性能指标给出建议
        efficiency = system_performance.get('efficiency', 1.0)
        safety_score = system_performance.get('safety_score', 1.0)
        balance_score = system_performance.get('balance_score', 1.0)
        
        if efficiency < 0.85 and current_constraints.constraint_level == ConstraintLevel.STRICT:
            recommendations.append("考虑放松功率约束以提高效率")
        
        if safety_score < 0.8:
            recommendations.append("建议收紧安全相关约束")
        
        if balance_score < 0.7:
            recommendations.append("调整SOC平衡容忍度以改善均衡性")
        
        if current_constraints.response_time_limit is not None and current_constraints.response_time_limit.item() > 10.0:
            recommendations.append("响应时间约束可能过于宽松")
        
        if not recommendations:
            recommendations.append("当前约束配置合理")
        
        return recommendations
    
    def get_generator_statistics(self) -> Dict[str, Any]:
        """获取生成器统计信息"""
        if not self.constraint_history:
            return {'error': 'No constraint history available'}
        
        # 统计约束等级分布
        level_counts = {}
        for record in self.constraint_history:
            level = record['constraint_level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # 计算平均约束值
        avg_charge_power = np.mean([r['max_charge_power'] for r in self.constraint_history])
        avg_response_time = np.mean([r['response_time_limit'] for r in self.constraint_history])
        
        return {
            'generator_id': self.generator_id,
            'total_generations': len(self.constraint_history),
            'constraint_level_distribution': level_counts,
            'avg_max_charge_power': avg_charge_power,
            'avg_response_time_limit': avg_response_time,
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'constraint_evolution_analysis': self.analyze_constraint_evolution()
        }

    def _calculate_system_max_charge_power(self, cluster_record: Dict) -> float:
        """计算系统最大充电功率"""
        bms_records = cluster_record.get('bms_records', [])
        total_power = 0.0
        
        for bms_record in bms_records:
            bms_power = self._calculate_bms_max_charge_power(bms_record)
            total_power += bms_power
        
        # 系统级降额
        system_health = cluster_record.get('system_health_status', 'Good')
        system_derating = {'Critical': 0.5, 'Poor': 0.7, 'Fair': 0.9, 'Good': 1.0}.get(system_health, 1.0)
        
        return total_power * system_derating
    
    def _calculate_system_max_discharge_power(self, cluster_record: Dict) -> float:
        """计算系统最大放电功率"""
        bms_records = cluster_record.get('bms_records', [])
        total_power = 0.0
        
        for bms_record in bms_records:
            bms_power = self._calculate_bms_max_discharge_power(bms_record)
            total_power += bms_power
        
        # 系统级降额
        system_health = cluster_record.get('system_health_status', 'Good')
        system_derating = {'Critical': 0.5, 'Poor': 0.7, 'Fair': 0.9, 'Good': 1.0}.get(system_health, 1.0)
        
        return total_power * system_derating
    
    def _calculate_power_ramp_limits(self, cluster_record: Dict) -> float:
        """计算功率变化率限制"""
        total_power = cluster_record.get('total_actual_power', 100000.0)
        
        # 基于系统状态动态调整
        system_balance = cluster_record.get('cluster_metrics', {}).get('overall_balance_score', 0.8)
        
        if system_balance > 0.9:
            ramp_factor = 1.0  # 均衡好时允许快速变化
        elif system_balance > 0.7:
            ramp_factor = 0.8
        else:
            ramp_factor = 0.5  # 不均衡时限制快速变化
        
        # 基础变化率：每秒10%
        base_ramp_rate = abs(total_power) * 0.1
        return base_ramp_rate * ramp_factor
    
    def _calculate_bms_max_discharge_power(self, bms_record: Dict) -> float:
        """计算BMS最大放电功率"""
        base_power = bms_record.get('bms_max_discharge_power', 300000.0)  # 默认300kW
        
        # 温度降额
        avg_temp = bms_record.get('avg_temperature', 25.0)
        if avg_temp > 45.0:
            temp_derating = max(0.6, (55.0 - avg_temp) / 10.0)
        elif avg_temp < 0.0:
            temp_derating = max(0.5, (avg_temp + 20.0) / 20.0)
        else:
            temp_derating = 1.0
        
        # SOC降额
        avg_soc = bms_record.get('avg_soc', 50.0)
        if avg_soc < 15.0:
            soc_derating = max(0.3, (avg_soc - 5.0) / 10.0)
        else:
            soc_derating = 1.0
        
        # 健康状态降额
        health_status = bms_record.get('health_status', 'Good')
        health_derating = {'Critical': 0.3, 'Poor': 0.6, 'Fair': 0.8, 'Good': 1.0}.get(health_status, 1.0)
        
        return base_power * temp_derating * soc_derating * health_derating
    
    def _get_temperature_derating_factor(self, temperature: float) -> float:
        """获取温度降额因子"""
        if temperature > 50.0:
            return max(0.5, (60.0 - temperature) / 10.0)
        elif temperature < 10.0:
            return max(0.6, (temperature + 10.0) / 20.0)
        else:
            return 1.0
    
    def _get_soc_derating_factor(self, soc: float) -> float:
        """获取SOC降额因子"""
        if soc > 90.0:
            return max(0.7, (95.0 - soc) / 5.0)
        elif soc < 10.0:
            return max(0.5, (soc - 5.0) / 5.0)
        else:
            return 1.0
    
    def _get_health_derating_factor(self, health_status: str) -> float:
        """获取健康状态降额因子"""
        health_factors = {
            'Critical': 0.3,
            'Poor': 0.6, 
            'Fair': 0.8,
            'Good': 1.0
        }
        return health_factors.get(health_status, 1.0)
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"ConstraintGenerator({self.generator_id}): "
                f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}")
