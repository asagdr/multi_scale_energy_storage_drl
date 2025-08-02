"""
状态归一化转换器
专门处理状态向量的归一化，从环境层分离出来
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class NormalizationConfig:
    """归一化配置"""
    method: str = "min_max"  # "min_max" | "z_score" | "robust" | "none"
    clip_outliers: bool = True
    outlier_threshold: float = 3.0  # 标准差倍数
    
class StateNormalizer:
    """
    状态归一化器
    负责将环境的原始状态转换为神经网络适用的归一化状态
    """
    
    def __init__(self, 
                 state_info: Dict[str, Dict],
                 config: NormalizationConfig = None):
        """
        初始化状态归一化器
        
        Args:
            state_info: 状态信息字典（来自环境）
            config: 归一化配置
        """
        self.state_info = state_info
        self.config = config or NormalizationConfig()
        
        # 提取状态范围
        self.state_ranges = {}
        self.state_names = []
        
        for state_name, info in state_info.items():
            self.state_names.append(state_name)
            self.state_ranges[state_name] = info['range']
        
        # 构建归一化参数
        self._build_normalization_params()
        
        print(f"✅ 状态归一化器初始化完成: {len(self.state_names)}维状态")
    
    def _build_normalization_params(self):
        """构建归一化参数"""
        
        self.normalization_params = {}
        
        for state_name in self.state_names:
            min_val, max_val = self.state_ranges[state_name]
            
            if self.config.method == "min_max":
                # Min-Max归一化参数
                self.normalization_params[state_name] = {
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val
                }
            elif self.config.method == "z_score":
                # Z-score归一化参数（需要运行时统计）
                mid_val = (min_val + max_val) / 2
                std_val = (max_val - min_val) / 6  # 假设6σ覆盖整个范围
                self.normalization_params[state_name] = {
                    'mean': mid_val,
                    'std': std_val
                }
            elif self.config.method == "robust":
                # 鲁棒归一化参数
                self.normalization_params[state_name] = {
                    'median': (min_val + max_val) / 2,
                    'iqr': (max_val - min_val) / 2
                }
    
    def normalize_state(self, raw_state: np.ndarray) -> np.ndarray:
        """
        归一化状态向量
        
        Args:
            raw_state: 原始状态向量
            
        Returns:
            归一化状态向量 [0,1]
        """
        
        if self.config.method == "none":
            return raw_state.copy()
        
        normalized_state = np.zeros_like(raw_state, dtype=np.float32)
        
        for i, state_name in enumerate(self.state_names):
            if i >= len(raw_state):
                break
                
            raw_value = raw_state[i]
            params = self.normalization_params[state_name]
            
            if self.config.method == "min_max":
                # Min-Max归一化 [min,max] -> [0,1]
                if params['range'] > 1e-8:
                    normalized_value = (raw_value - params['min']) / params['range']
                else:
                    normalized_value = 0.5  # 常数值的情况
                    
            elif self.config.method == "z_score":
                # Z-score归一化，然后映射到[0,1]
                if params['std'] > 1e-8:
                    z_score = (raw_value - params['mean']) / params['std']
                    # 将z_score映射到[0,1]，假设±3σ覆盖99.7%的数据
                    normalized_value = np.clip((z_score + 3) / 6, 0.0, 1.0)
                else:
                    normalized_value = 0.5
                    
            elif self.config.method == "robust":
                # 鲁棒归一化
                if params['iqr'] > 1e-8:
                    robust_score = (raw_value - params['median']) / params['iqr']
                    normalized_value = np.clip((robust_score + 2) / 4, 0.0, 1.0)
                else:
                    normalized_value = 0.5
            
            # 异常值处理
            if self.config.clip_outliers:
                normalized_value = np.clip(normalized_value, 0.0, 1.0)
            
            normalized_state[i] = normalized_value
        
        return normalized_state
    
    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        反归一化状态向量
        
        Args:
            normalized_state: 归一化状态向量 [0,1]
            
        Returns:
            原始状态向量
        """
        
        if self.config.method == "none":
            return normalized_state.copy()
        
        raw_state = np.zeros_like(normalized_state, dtype=np.float32)
        
        for i, state_name in enumerate(self.state_names):
            if i >= len(normalized_state):
                break
                
            normalized_value = np.clip(normalized_state[i], 0.0, 1.0)
            params = self.normalization_params[state_name]
            
            if self.config.method == "min_max":
                # Min-Max反归一化 [0,1] -> [min,max]
                raw_value = normalized_value * params['range'] + params['min']
                
            elif self.config.method == "z_score":
                # Z-score反归一化
                z_score = normalized_value * 6 - 3
                raw_value = z_score * params['std'] + params['mean']
                
            elif self.config.method == "robust":
                # 鲁棒反归一化
                robust_score = normalized_value * 4 - 2
                raw_value = robust_score * params['iqr'] + params['median']
            
            raw_state[i] = raw_value
        
        return raw_state
    
    def get_normalization_info(self) -> Dict:
        """获取归一化信息"""
        return {
            'method': self.config.method,
            'state_count': len(self.state_names),
            'state_names': self.state_names.copy(),
            'parameters': self.normalization_params.copy(),
            'config': self.config
        }
    
    def update_statistics(self, raw_states: np.ndarray):
        """
        更新归一化统计量（用于在线学习）
        
        Args:
            raw_states: 原始状态批次 [batch_size, state_dim]
        """
        
        if self.config.method == "z_score":
            # 更新均值和标准差
            for i, state_name in enumerate(self.state_names):
                if i >= raw_states.shape[1]:
                    break
                
                state_values = raw_states[:, i]
                current_mean = np.mean(state_values)
                current_std = np.std(state_values)
                
                # 指数移动平均更新
                alpha = 0.1
                params = self.normalization_params[state_name]
                params['mean'] = (1 - alpha) * params['mean'] + alpha * current_mean
                params['std'] = (1 - alpha) * params['std'] + alpha * current_std
        
        elif self.config.method == "robust":
            # 更新中位数和四分位距
            for i, state_name in enumerate(self.state_names):
                if i >= raw_states.shape[1]:
                    break
                
                state_values = raw_states[:, i]
                current_median = np.median(state_values)
                current_iqr = np.percentile(state_values, 75) - np.percentile(state_values, 25)
                
                # 指数移动平均更新
                alpha = 0.1
                params = self.normalization_params[state_name]
                params['median'] = (1 - alpha) * params['median'] + alpha * current_median
                params['iqr'] = (1 - alpha) * params['iqr'] + alpha * current_iqr
