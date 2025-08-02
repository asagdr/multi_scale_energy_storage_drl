"""
DRL模型配置 - BMS集群版本
维度与环境状态匹配
"""

from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    TRANSFORMER = "transformer"
    LSTM = "lstm" 
    GRU = "gru"
    MLP = "mlp"

class LayerType(Enum):
    UPPER = "upper"
    LOWER = "lower"

@dataclass
class ModelConfig:
    """DRL模型配置 - 支持BMS集群"""
    
    # === 上层模型配置 ===
    upper_model_type: ModelType = ModelType.TRANSFORMER
    upper_state_dim: int = 24      # 与storage_environment的24维状态匹配
    upper_action_dim: int = 5      # [功率指令, SOC权重, 温度权重, 寿命权重, 效率权重]
    upper_hidden_dim: int = 256
    
    # === 下层模型配置 ===  
    lower_model_type: ModelType = ModelType.MLP
    lower_state_dim: int = 32      # 下层接收更详细的状态信息
    lower_action_dim: int = 3      # [实际功率执行, 响应速度, 误差补偿]
    lower_hidden_dim: int = 128
    
    # === BMS集群特定配置 ===
    bms_cluster_features: bool = True
    num_bms: int = 10              # BMS数量
    inter_bms_attention: bool = True   # BMS间注意力机制
    intra_bms_attention: bool = True   # BMS内注意力机制
    
    # === 状态处理配置 ===
    state_normalization: str = "external"  # "internal" | "external" | "none"
    state_history_length: int = 10        # 状态历史长度
    enable_state_filtering: bool = True   # 启用状态滤波
    
    # === 特征提取配置 ===
    enable_attention: bool = True
    attention_heads: int = 8
    dropout_rate: float = 0.1
    batch_norm: bool = True
    layer_norm: bool = True
    
    # === 多层级架构配置 ===
    hierarchical_levels: int = 2          # 层级数量
    cross_level_communication: bool = True # 跨层级通信
    level_coordination_dim: int = 64      # 层级协调维度
    
    # === 约束处理配置 ===
    constraint_handling_method: str = "penalty"  # "penalty" | "barrier" | "projection"
    constraint_penalty_weight: float = 10.0
    safety_margin: float = 0.1
    adaptive_constraints: bool = True     # 自适应约束
    
    # === 训练配置 ===
    learning_rate_upper: float = 3e-4
    learning_rate_lower: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # === 经验回放配置 ===
    buffer_size: int = 100000
    batch_size: int = 64
    prioritized_replay: bool = True
    replay_alpha: float = 0.6
    replay_beta: float = 0.4
    
    # === 探索配置 ===
    exploration_method: str = "noise"  # "noise" | "epsilon" | "entropy"
    initial_noise_scale: float = 0.2
    noise_decay_rate: float = 0.995
    min_noise_scale: float = 0.01
    
    # === 网络架构详细配置 ===
    
    # 编码器配置
    encoder_layers: int = 3
    encoder_hidden_dims: list = None  # [512, 256, 128]
    
    # 解码器配置  
    decoder_layers: int = 2
    decoder_hidden_dims: list = None  # [128, 64]
    
    # 注意力机制配置
    attention_temperature: float = 1.0
    attention_dropout: float = 0.1
    
    # 激活函数配置
    activation_function: str = "gelu"  # "relu" | "gelu" | "swish" | "leaky_relu"
    
    def __post_init__(self):
        """初始化后处理"""
        
        # 设置默认的隐藏层维度
        if self.encoder_hidden_dims is None:
            self.encoder_hidden_dims = [self.upper_hidden_dim * 2, self.upper_hidden_dim, self.upper_hidden_dim // 2]
        
        if self.decoder_hidden_dims is None:
            self.decoder_hidden_dims = [self.upper_hidden_dim // 2, self.upper_hidden_dim // 4]
        
        # 验证配置的一致性
        self._validate_config()
    
    def _validate_config(self):
        """验证配置一致性"""
        
        # 检查状态维度匹配
        if self.upper_state_dim != 24:
            print(f"⚠️ 上层状态维度 {self.upper_state_dim} 与环境状态维度 24 不匹配")
        
        # 检查动作维度
        if self.upper_action_dim != 5:
            print(f"⚠️ 上层动作维度 {self.upper_action_dim} 与环境动作维度 5 不匹配")
        
        # 检查BMS数量
        if self.num_bms <= 0:
            raise ValueError("BMS数量必须大于0")
        
        # 检查隐藏层维度
        if len(self.encoder_hidden_dims) != self.encoder_layers:
            print(f"⚠️ 编码器层数与隐藏维度列表长度不匹配")
        
        # 检查学习率
        if self.learning_rate_upper <= 0 or self.learning_rate_lower <= 0:
            raise ValueError("学习率必须大于0")
        
        print(f"✅ 模型配置验证通过: 上层{self.upper_state_dim}→{self.upper_action_dim}, 下层{self.lower_state_dim}→{self.lower_action_dim}")
    
    def get_upper_layer_config(self) -> dict:
        """获取上层配置字典"""
        return {
            'model_type': self.upper_model_type,
            'state_dim': self.upper_state_dim,
            'action_dim': self.upper_action_dim,
            'hidden_dim': self.upper_hidden_dim,
            'attention_enabled': self.enable_attention,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'layer_norm': self.layer_norm,
            'learning_rate': self.learning_rate_upper,
            'bms_cluster_features': self.bms_cluster_features,
            'num_bms': self.num_bms,
            'inter_bms_attention': self.inter_bms_attention
        }
    
    def get_lower_layer_config(self) -> dict:
        """获取下层配置字典"""
        return {
            'model_type': self.lower_model_type,
            'state_dim': self.lower_state_dim,
            'action_dim': self.lower_action_dim,
            'hidden_dim': self.lower_hidden_dim,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'learning_rate': self.learning_rate_lower,
            'intra_bms_attention': self.intra_bms_attention
        }
    
    def get_constraint_config(self) -> dict:
        """获取约束处理配置"""
        return {
            'method': self.constraint_handling_method,
            'penalty_weight': self.constraint_penalty_weight,
            'safety_margin': self.safety_margin,
            'adaptive': self.adaptive_constraints
        }
    
    def get_training_config(self) -> dict:
        """获取训练配置"""
        return {
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'prioritized_replay': self.prioritized_replay,
            'replay_alpha': self.replay_alpha,
            'replay_beta': self.replay_beta,
            'weight_decay': self.weight_decay,
            'gradient_clip_norm': self.gradient_clip_norm,
            'exploration_method': self.exploration_method,
            'initial_noise_scale': self.initial_noise_scale,
            'noise_decay_rate': self.noise_decay_rate,
            'min_noise_scale': self.min_noise_scale
        }
    
    def update_for_environment(self, env_info: dict):
        """根据环境信息更新配置"""
        
        if 'state_dim' in env_info:
            self.upper_state_dim = env_info['state_dim']
        
        if 'action_dim' in env_info:
            self.upper_action_dim = env_info['action_dim']
        
        if 'num_bms' in env_info:
            self.num_bms = env_info['num_bms']
        
        # 重新验证配置
        self._validate_config()
        
        print(f"🔄 模型配置已根据环境信息更新")
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"ModelConfig: 上层({self.upper_model_type.value}): "
                f"{self.upper_state_dim}→{self.upper_action_dim}, "
                f"下层({self.lower_model_type.value}): "
                f"{self.lower_state_dim}→{self.lower_action_dim}, "
                f"BMS集群: {self.num_bms}个")
