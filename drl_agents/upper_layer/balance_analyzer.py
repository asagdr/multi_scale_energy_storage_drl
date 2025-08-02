"""
均衡状态分析器 - 神经网络模块
专门为上层DRL提供均衡状态的特征提取和决策支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

try:
    from config.training_config import UpperLayerConfig
except ImportError:
    @dataclass
    class UpperLayerConfig:
        hidden_dim: int = 64
        learning_rate: float = 3e-4

try:
    from config.model_config import ModelConfig
except ImportError:
    @dataclass
    class ModelConfig:
        upper_state_dim: int = 24  # 扩展为BMS集群状态维度
        upper_action_dim: int = 5

@dataclass
class BalanceMetrics:
    """均衡指标数据结构"""
    # SOC均衡指标
    soc_std: float = 0.0                    # σ_SOC (关键指标)
    soc_variance: float = 0.0               # SOC方差
    soc_range: float = 0.0                  # SOC极差
    soc_consistency: float = 1.0            # SOC一致性 [0,1]
    soc_balance_urgency: float = 0.0        # SOC均衡紧迫性 [0,1]
    
    # 温度均衡指标
    temp_std: float = 0.0                   # 温度标准差
    temp_variance: float = 0.0              # 温度方差
    temp_range: float = 0.0                 # 温度极差
    temp_consistency: float = 1.0           # 温度一致性 [0,1]
    temp_balance_urgency: float = 0.0       # 温度均衡紧迫性 [0,1]
    
    # 劣化均衡指标
    soh_std: float = 0.0                    # SOH标准差
    soh_variance: float = 0.0               # SOH方差
    soh_range: float = 0.0                  # SOH极差
    degradation_consistency: float = 1.0    # 劣化一致性 [0,1]
    lifetime_urgency: float = 0.0           # 寿命优化紧迫性 [0,1]
    
    # 综合指标
    overall_balance_score: float = 1.0      # 综合均衡评分 [0,1]
    critical_balance_type: str = "none"     # 最需要均衡的类型

@dataclass
class BalanceTargets:
    """均衡目标数据结构"""
    target_soc_std: float = 1.0            # 目标SOC标准差
    target_temp_std: float = 2.0           # 目标温度标准差
    target_soh_std: float = 3.0            # 目标SOH标准差
    
    soc_balance_weight: float = 0.33       # SOC均衡权重
    temp_balance_weight: float = 0.33      # 温度均衡权重
    lifetime_balance_weight: float = 0.34  # 寿命均衡权重
    
    balance_time_horizon: float = 300.0    # 均衡时间窗口 (s)

class SOCBalanceAnalyzer(nn.Module):
    """SOC均衡分析器 - 支持BMS集群"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(SOCBalanceAnalyzer, self).__init__()
        
        # 特征提取网络 - 处理BMS集群SOC状态
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # BMS集群SOC特征分析
        self.cluster_soc_analyzer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # BMS间SOC均衡紧迫性预测
        self.inter_bms_urgency_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出[0,1]
        )
        
        # BMS内SOC均衡紧迫性预测
        self.intra_bms_urgency_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出[0,1]
        )
        
        # SOC一致性评估
        self.consistency_evaluator = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 输出[0,1]
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 处理BMS集群SOC状态
        
        Args:
            x: BMS集群状态向量 [batch_size, input_dim]
               包含: [系统平均SOC, BMS间SOC不平衡, BMS内SOC不平衡, ...]
            
        Returns:
            BMS集群SOC均衡分析结果
        """
        features = self.feature_extractor(x)
        soc_features = self.cluster_soc_analyzer(features)
        
        inter_bms_urgency = self.inter_bms_urgency_predictor(soc_features)
        intra_bms_urgency = self.intra_bms_urgency_predictor(soc_features)
        consistency = self.consistency_evaluator(soc_features)
        
        return {
            'soc_features': soc_features,
            'inter_bms_urgency': inter_bms_urgency.squeeze(-1),
            'intra_bms_urgency': intra_bms_urgency.squeeze(-1),
            'consistency': consistency.squeeze(-1)
        }

class ThermalBalanceAnalyzer(nn.Module):
    """温度均衡分析器 - 支持BMS集群"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ThermalBalanceAnalyzer, self).__init__()
        
        # 特征提取网络 - 处理BMS集群温度状态
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # BMS集群温度特征分析
        self.cluster_thermal_analyzer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # BMS间热管理需求预测
        self.inter_bms_thermal_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # BMS内热管理需求预测
        self.intra_bms_thermal_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 温度一致性评估
        self.thermal_consistency_evaluator = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 处理BMS集群温度状态
        """
        features = self.feature_extractor(x)
        thermal_features = self.cluster_thermal_analyzer(features)
        
        inter_bms_thermal_urgency = self.inter_bms_thermal_predictor(thermal_features)
        intra_bms_thermal_urgency = self.intra_bms_thermal_predictor(thermal_features)
        consistency = self.thermal_consistency_evaluator(thermal_features)
        
        return {
            'thermal_features': thermal_features,
            'inter_bms_thermal_urgency': inter_bms_thermal_urgency.squeeze(-1),
            'intra_bms_thermal_urgency': intra_bms_thermal_urgency.squeeze(-1),
            'consistency': consistency.squeeze(-1)
        }

class DegradationBalanceAnalyzer(nn.Module):
    """劣化均衡分析器 - 支持BMS集群"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(DegradationBalanceAnalyzer, self).__init__()
        
        # 特征提取网络 - 处理BMS集群劣化状态
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # BMS集群劣化特征分析
        self.cluster_degradation_analyzer = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # BMS间寿命优化紧迫性
        self.inter_bms_lifetime_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # BMS内寿命优化紧迫性
        self.intra_bms_lifetime_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 劣化一致性评估
        self.degradation_consistency_evaluator = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 多层级成本趋势预测
        self.multi_level_cost_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 处理BMS集群劣化状态
        """
        features = self.feature_extractor(x)
        degradation_features = self.cluster_degradation_analyzer(features)
        
        inter_bms_lifetime_urgency = self.inter_bms_lifetime_predictor(degradation_features)
        intra_bms_lifetime_urgency = self.intra_bms_lifetime_predictor(degradation_features)
        consistency = self.degradation_consistency_evaluator(degradation_features)
        cost_trend = self.multi_level_cost_predictor(degradation_features)
        
        return {
            'degradation_features': degradation_features,
            'inter_bms_lifetime_urgency': inter_bms_lifetime_urgency.squeeze(-1),
            'intra_bms_lifetime_urgency': intra_bms_lifetime_urgency.squeeze(-1),
            'consistency': consistency.squeeze(-1),
            'multi_level_cost_trend': cost_trend.squeeze(-1)
        }

class BalanceAnalyzer(nn.Module):
    """
    BMS集群均衡状态分析器
    专门为上层DRL提供BMS集群的均衡特征提取和决策支持
    """
    
    def __init__(self,
                 config: UpperLayerConfig,
                 model_config: ModelConfig,
                 analyzer_id: str = "BalanceAnalyzer_001"):
        """
        初始化BMS集群均衡分析器
        
        Args:
            config: 上层配置
            model_config: 模型配置
            analyzer_id: 分析器ID
        """
        super(BalanceAnalyzer, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.analyzer_id = analyzer_id
        
        # === 模型参数 ===
        self.input_dim = model_config.upper_state_dim  # 24维BMS集群状态
        self.hidden_dim = config.hidden_dim
        
        # === BMS集群子分析器 ===
        self.soc_analyzer = SOCBalanceAnalyzer(self.input_dim, self.hidden_dim)
        self.thermal_analyzer = ThermalBalanceAnalyzer(self.input_dim, self.hidden_dim)
        self.degradation_analyzer = DegradationBalanceAnalyzer(self.input_dim, self.hidden_dim)
        
        # === BMS集群特征融合网络 ===
        fusion_input_dim = 16 + 16 + 16  # 各子分析器的特征维度
        self.cluster_fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 20)  # 扩展输出维度支持BMS集群
        )
        
        # === BMS集群输出头 ===
        self.cluster_balance_score_head = nn.Sequential(
            nn.Linear(20, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # 集群总体均衡评分 [0,1]
        )
        
        self.multi_level_priority_head = nn.Sequential(
            nn.Linear(20, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.Softmax(dim=-1)  # [BMS间SOC, BMS内SOC, BMS间温度, BMS内温度, BMS间寿命, BMS内寿命]
        )
        
        # === BMS集群协调策略预测 ===
        self.coordination_strategy_head = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.Softmax(dim=-1)  # [disabled, soc_balance, thermal_balance, comprehensive]
        )
        
        # === BMS集群目标生成器 ===
        self.cluster_target_generator = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 9)  # [inter_soc_std, intra_soc_std, inter_temp_std, intra_temp_std, inter_soh_std, intra_soh_std, weight1, weight2, weight3]
        )
        
        # === 历史记录 ===
        self.analysis_history: List[Dict] = []
        
        print(f"✅ BMS集群均衡分析器初始化完成: {analyzer_id}")
        print(f"   输入维度: {self.input_dim} (BMS集群状态)")
        print(f"   隐藏维度: {self.hidden_dim}")
    
    def forward(self, 
                cluster_state: torch.Tensor,
                return_detailed: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向分析 - 处理BMS集群状态
        
        Args:
            cluster_state: BMS集群状态向量 [batch_size, 24]
                         [0-2]: 系统级状态 (平均SOC, 平均温度, 平均SOH)
                         [3-5]: BMS间不平衡 (SOC标准差, 温度标准差, SOH标准差)
                         [6-7]: BMS内不平衡 (平均SOC标准差, 平均温度标准差)
                         [8-11]: 功率状态 (总功率, 功率效率, 跟踪误差, 功率利用率)
                         [12-15]: 成本状态 (成本增长率, 惩罚比例, BMS级成本, 系统级成本)
                         [16-18]: 约束状态 (热约束, SOC约束, 均衡约束)
                         [19-20]: 协调状态 (协调指令比例, 协调权重)
                         [21-23]: 环境状态 (环境温度, 功率需求, 预留)
            return_detailed: 是否返回详细分析结果
            
        Returns:
            BMS集群均衡分析结果
        """
        # === 1. BMS集群子分析器分析 ===
        soc_analysis = self.soc_analyzer(cluster_state)
        thermal_analysis = self.thermal_analyzer(cluster_state)
        degradation_analysis = self.degradation_analyzer(cluster_state)
        
        # === 2. BMS集群特征融合 ===
        cluster_fused_features = torch.cat([
            soc_analysis['soc_features'],
            thermal_analysis['thermal_features'],
            degradation_analysis['degradation_features']
        ], dim=-1)
        
        cluster_fusion_output = self.cluster_fusion_network(cluster_fused_features)
        
        # === 3. BMS集群综合分析 ===
        cluster_balance_score = self.cluster_balance_score_head(cluster_fusion_output)
        multi_level_priorities = self.multi_level_priority_head(cluster_fusion_output)
        coordination_strategy = self.coordination_strategy_head(cluster_fusion_output)
        
        # === 4. BMS集群目标生成 ===
        cluster_targets = self.cluster_target_generator(cluster_fusion_output)
        
        # 分解目标
        target_inter_soc_std = torch.relu(cluster_targets[:, 0]) + 0.5   # 最小0.5%
        target_intra_soc_std = torch.relu(cluster_targets[:, 1]) + 0.3   # 最小0.3%
        target_inter_temp_std = torch.relu(cluster_targets[:, 2]) + 1.0  # 最小1℃
        target_intra_temp_std = torch.relu(cluster_targets[:, 3]) + 0.5  # 最小0.5℃
        target_inter_soh_std = torch.relu(cluster_targets[:, 4]) + 2.0   # 最小2%
        target_intra_soh_std = torch.relu(cluster_targets[:, 5]) + 1.0   # 最小1%
        
        # 多层级权重归一化
        multi_level_weights = torch.softmax(cluster_targets[:, 6:9], dim=-1)
        
        # === 5. 构建BMS集群输出 ===
        output = {
            # 集群整体评分
            'cluster_balance_score': cluster_balance_score.squeeze(-1),
            
            # 多层级优先级 [BMS间SOC, BMS内SOC, BMS间温度, BMS内温度, BMS间寿命, BMS内寿命]
            'multi_level_priorities': multi_level_priorities,
            
            # 协调策略建议 [disabled, soc_balance, thermal_balance, comprehensive]
            'coordination_strategy': coordination_strategy,
            
            # BMS间分析结果
            'inter_bms_soc_urgency': soc_analysis['inter_bms_urgency'],
            'inter_bms_thermal_urgency': thermal_analysis['inter_bms_thermal_urgency'],
            'inter_bms_lifetime_urgency': degradation_analysis['inter_bms_lifetime_urgency'],
            
            # BMS内分析结果
            'intra_bms_soc_urgency': soc_analysis['intra_bms_urgency'],
            'intra_bms_thermal_urgency': thermal_analysis['intra_bms_thermal_urgency'], 
            'intra_bms_lifetime_urgency': degradation_analysis['intra_bms_lifetime_urgency'],
            
            # 一致性评估
            'soc_consistency': soc_analysis['consistency'],
            'thermal_consistency': thermal_analysis['consistency'],
            'degradation_consistency': degradation_analysis['consistency'],
            
            # 成本趋势
            'multi_level_cost_trend': degradation_analysis['multi_level_cost_trend'],
            
            # 多层级目标设定
            'target_inter_bms_soc_std': target_inter_soc_std,
            'target_intra_bms_soc_std': target_intra_soc_std,
            'target_inter_bms_temp_std': target_inter_temp_std,
            'target_intra_bms_temp_std': target_intra_temp_std,
            'target_inter_bms_soh_std': target_inter_soh_std,
            'target_intra_bms_soh_std': target_intra_soh_std,
            
            # 多层级权重
            'multi_level_balance_weights': multi_level_weights
        }
        
        if return_detailed:
            output.update({
                'cluster_fusion_features': cluster_fusion_output,
                'soc_features': soc_analysis['soc_features'],
                'thermal_features': thermal_analysis['thermal_features'],
                'degradation_features': degradation_analysis['degradation_features']
            })
        
        return output
    
    def analyze_balance_state(self, 
                            system_state: Dict[str, Any]) -> BalanceMetrics:
        """
        分析均衡状态 - 兼容接口（将字典转换为tensor输入）
        
        Args:
            system_state: 系统状态字典
            
        Returns:
            均衡指标
        """
        # 将字典状态转换为tensor
        cluster_state_tensor = self._dict_to_tensor(system_state)
        
        # 神经网络推理
        with torch.no_grad():
            output = self.forward(cluster_state_tensor.unsqueeze(0))
        
        # 转换为BalanceMetrics
        metrics = self._tensor_to_metrics(output, system_state)
        
        return metrics
    
    def _dict_to_tensor(self, system_state: Dict[str, Any]) -> torch.Tensor:
        """将系统状态字典转换为tensor"""
        state_vector = torch.zeros(24, dtype=torch.float32)
        
        # 系统级状态 (0-2)
        state_vector[0] = system_state.get('system_avg_soc', 50.0) / 100.0
        state_vector[1] = (system_state.get('system_avg_temp', 25.0) - 15.0) / 30.0
        state_vector[2] = system_state.get('system_avg_soh', 100.0) / 100.0
        
        # BMS间不平衡 (3-5)
        state_vector[3] = system_state.get('inter_bms_soc_std', 0.0) / 20.0
        state_vector[4] = system_state.get('inter_bms_temp_std', 0.0) / 30.0
        state_vector[5] = system_state.get('inter_bms_soh_std', 0.0) / 20.0
        
        # BMS内不平衡 (6-7)
        state_vector[6] = system_state.get('avg_intra_bms_soc_std', 0.0) / 10.0
        state_vector[7] = system_state.get('avg_intra_bms_temp_std', 0.0) / 15.0
        
        # 其他状态按需填充...
        
        return torch.clamp(state_vector, 0.0, 1.0)
    
    def _tensor_to_metrics(self, output: Dict[str, torch.Tensor], system_state: Dict[str, Any]) -> BalanceMetrics:
        """将tensor输出转换为BalanceMetrics"""
        metrics = BalanceMetrics()
        
        # 从神经网络输出提取
        metrics.overall_balance_score = output['cluster_balance_score'].item()
        metrics.soc_consistency = output['soc_consistency'].item()
        metrics.temp_consistency = output['thermal_consistency'].item()
        metrics.degradation_consistency = output['degradation_consistency'].item()
        
        # 从原始状态提取
        metrics.soc_std = system_state.get('inter_bms_soc_std', 0.0)
        metrics.temp_std = system_state.get('inter_bms_temp_std', 0.0)
        metrics.soh_std = system_state.get('inter_bms_soh_std', 0.0)
        
        # 计算紧迫性
        metrics.soc_balance_urgency = output['inter_bms_soc_urgency'].item()
        metrics.temp_balance_urgency = output['inter_bms_thermal_urgency'].item()
        metrics.lifetime_urgency = output['inter_bms_lifetime_urgency'].item()
        
        # 确定关键均衡类型
        urgencies = {
            'soc': metrics.soc_balance_urgency,
            'thermal': metrics.temp_balance_urgency,
            'degradation': metrics.lifetime_urgency
        }
        metrics.critical_balance_type = max(urgencies.items(), key=lambda x: x[1])[0]
        
        return metrics
    
    def generate_balance_targets(self, 
                               current_metrics: BalanceMetrics,
                               time_horizon: float = 300.0) -> BalanceTargets:
        """
        生成均衡目标 - 兼容接口
        """
        targets = BalanceTargets()
        targets.balance_time_horizon = time_horizon
        
        # 简化目标生成（实际应该通过神经网络）
        targets.target_soc_std = max(0.5, current_metrics.soc_std * 0.7)
        targets.target_temp_std = max(1.0, current_metrics.temp_std * 0.8)
        targets.target_soh_std = max(2.0, current_metrics.soh_std * 0.9)
        
        # 基于紧迫性分配权重
        urgencies = np.array([
            current_metrics.soc_balance_urgency,
            current_metrics.temp_balance_urgency,
            current_metrics.lifetime_urgency
        ])
        
        total_urgency = np.sum(urgencies)
        if total_urgency > 0:
            weights = urgencies / total_urgency
        else:
            weights = np.array([0.33, 0.33, 0.34])
        
        targets.soc_balance_weight = weights[0]
        targets.temp_balance_weight = weights[1]
        targets.lifetime_balance_weight = weights[2]
        
        return targets
    
    def predict_balance_evolution(self, 
                                current_state: torch.Tensor,
                                prediction_steps: int = 10) -> Dict[str, torch.Tensor]:
        """
        预测BMS集群均衡状态演化
        """
        state = current_state
        predictions = []
        
        for step in range(prediction_steps):
            analysis = self.forward(state)
            predictions.append({
                'step': step,
                'cluster_balance_score': analysis['cluster_balance_score'],
                'inter_bms_soc_urgency': analysis['inter_bms_soc_urgency'],
                'intra_bms_soc_urgency': analysis['intra_bms_soc_urgency'],
                'coordination_strategy': analysis['coordination_strategy']
            })
            
            # 简单的状态演化（实际应该基于物理模型）
            noise = torch.randn_like(state) * 0.005  # 减小噪声
            state = state + noise
            state = torch.clamp(state, 0.0, 1.0)
        
        # 整理预测结果
        prediction_result = {}
        for key in predictions[0].keys():
            if key != 'step':
                values = [pred[key] for pred in predictions]
                prediction_result[f'predicted_{key}'] = torch.stack(values, dim=1)
        
        return prediction_result
    
    def get_balance_insights(self, 
                           analysis_result: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """
        获取BMS集群均衡洞察
        """
        insights = {}
        
        # 集群整体状态
        cluster_balance_score = analysis_result['cluster_balance_score'].item()
        if cluster_balance_score > 0.8:
            insights['cluster_overall'] = "BMS集群均衡状态良好"
        elif cluster_balance_score > 0.6:
            insights['cluster_overall'] = "BMS集群均衡状态一般，需要关注"
        else:
            insights['cluster_overall'] = "BMS集群均衡状态差，需要立即优化"
        
        # BMS间均衡洞察
        inter_soc_urgency = analysis_result['inter_bms_soc_urgency'].item()
        if inter_soc_urgency > 0.7:
            insights['inter_bms_soc'] = "BMS间SOC严重不平衡，需要强化协调"
        elif inter_soc_urgency > 0.3:
            insights['inter_bms_soc'] = "BMS间SOC存在不平衡，建议适度协调"
        else:
            insights['inter_bms_soc'] = "BMS间SOC均衡状态良好"
        
        # BMS内均衡洞察
        intra_soc_urgency = analysis_result['intra_bms_soc_urgency'].item()
        if intra_soc_urgency > 0.7:
            insights['intra_bms_soc'] = "BMS内SOC严重不平衡，需要强化内部均衡"
        elif intra_soc_urgency > 0.3:
            insights['intra_bms_soc'] = "BMS内SOC存在不平衡，建议优化均衡策略"
        else:
            insights['intra_bms_soc'] = "BMS内SOC均衡状态良好"
        
        # 协调策略建议
        coordination_strategy = analysis_result['coordination_strategy'][0]  # 取第一个batch
        strategy_names = ['禁用协调', 'SOC均衡协调', '热管理协调', '综合协调']
        max_strategy_idx = torch.argmax(coordination_strategy).item()
        insights['coordination_strategy'] = f"建议采用: {strategy_names[max_strategy_idx]}"
        
        # 多层级优先级建议
        priorities = analysis_result['multi_level_priorities'][0]  # 取第一个batch
        priority_names = ['BMS间SOC', 'BMS内SOC', 'BMS间温度', 'BMS内温度', 'BMS间寿命', 'BMS内寿命']
        max_priority_idx = torch.argmax(priorities).item()
        insights['priority'] = f"当前应优先关注: {priority_names[max_priority_idx]}"
        
        return insights
    
    def save_analysis_history(self, analysis_result: Dict[str, Any]):
        """保存分析历史"""
        record = {
            'timestamp': len(self.analysis_history),
            'cluster_balance_score': analysis_result.get('cluster_balance_score', 0.0),
            'inter_bms_soc_urgency': analysis_result.get('inter_bms_soc_urgency', 0.0),
            'intra_bms_soc_urgency': analysis_result.get('intra_bms_soc_urgency', 0.0),
            'coordination_strategy': analysis_result.get('coordination_strategy', 'unknown')
        }
        
        self.analysis_history.append(record)
        
        # 维护历史长度
        if len(self.analysis_history) > 1000:
            self.analysis_history.pop(0)
    
    def get_analyzer_statistics(self) -> Dict[str, Any]:
        """获取分析器统计信息"""
        if not self.analysis_history:
            return {'error': 'No analysis history available'}
        
        cluster_balance_scores = [record['cluster_balance_score'] for record in self.analysis_history]
        inter_soc_urgencies = [record['inter_bms_soc_urgency'] for record in self.analysis_history]
        
        return {
            'analyzer_id': self.analyzer_id,
            'analyzer_type': 'BMS_Cluster_Balance_Analyzer',
            'total_analyses': len(self.analysis_history),
            'avg_cluster_balance_score': np.mean(cluster_balance_scores),
            'avg_inter_bms_soc_urgency': np.mean(inter_soc_urgencies),
            'balance_trend': 'improving' if len(cluster_balance_scores) > 10 and 
                           np.mean(cluster_balance_scores[-10:]) > np.mean(cluster_balance_scores[-20:-10]) 
                           else 'stable',
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'model_size_mb': sum(p.numel() for p in self.parameters()) * 4 / (1024 * 1024),
            'input_dim': self.input_dim,
            'supports_bms_cluster': True
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"BalanceAnalyzer({self.analyzer_id}): "
                f"BMS集群支持, input_dim={self.input_dim}, hidden_dim={self.hidden_dim}")
