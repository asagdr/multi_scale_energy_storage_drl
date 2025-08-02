from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    TRANSFORMER = "transformer"
    LSTM = "lstm" 
    GRU = "gru"
    MLP = "mlp"

@dataclass
class ModelConfig:
    """DRL模型配置"""
    
    # 上层模型配置
    upper_model_type: ModelType = ModelType.TRANSFORMER
    upper_state_dim: int = 14      # 与storage_environment对应
    upper_action_dim: int = 4
    upper_hidden_dim: int = 256
    
    # 下层模型配置  
    lower_model_type: ModelType = ModelType.MLP
    lower_state_dim: int = 20      # 更详细的状态
    lower_action_dim: int = 3
    lower_hidden_dim: int = 128
    
    # 特征提取配置
    enable_attention: bool = True
    dropout_rate: float = 0.1
    batch_norm: bool = True
    
    # 约束处理
    constraint_penalty_weight: float = 10.0
    safety_margin: float = 0.1
