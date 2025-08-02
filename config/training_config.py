from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class UpperLayerConfig:
    """上层DRL配置"""
    # 网络结构
    transformer_layers: int = 4
    attention_heads: int = 8
    hidden_dim: int = 256
    sequence_length: int = 100
    
    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_episodes: int = 1000
    update_frequency: int = 5  # 5分钟更新一次
    
    # 多目标权重
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'soc_balance': 0.3,      # SOC均衡
        'temp_balance': 0.2,     # 温度均衡  
        'lifetime_cost': 0.3,    # 寿命成本
        'efficiency': 0.2        # 效率
    })

@dataclass
class LowerLayerConfig:
    """下层DRL配置"""
    # DDPG参数
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    tau: float = 0.001
    gamma: float = 0.99
    
    # 网络结构
    actor_hidden: List[int] = field(default_factory=lambda: [400, 300])
    critic_hidden: List[int] = field(default_factory=lambda: [400, 300])
    
    # 控制参数
    action_noise: float = 0.1
    update_frequency: int = 1    # 每个时间步更新
    response_time: float = 0.01  # 10ms响应时间

@dataclass
class TrainingConfig:
    """训练总配置"""
    upper_config: UpperLayerConfig = field(default_factory=UpperLayerConfig)
    lower_config: LowerLayerConfig = field(default_factory=LowerLayerConfig)
    
    # 联合训练
    joint_training: bool = True
    pretraining_episodes: int = 200
    
    # 实验设置
    random_seed: int = 42
    save_frequency: int = 100
    eval_frequency: int = 50
    
    # 数据配置
    replay_buffer_size: int = 100000
    min_replay_size: int = 1000
