import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import UpperLayerConfig
from config.model_config import ModelConfig

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """多头注意力模块"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, 
                                   query: torch.Tensor, 
                                   key: torch.Tensor, 
                                   value: torch.Tensor,
                                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """缩放点积注意力"""
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        return output, attention_weights
    
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换和reshape
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        attention_output, self.attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 拼接多头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 输出投影
        output = self.w_o(attention_output)
        return output

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int, 
                 dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力子层
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer编码器
    用于处理电池系统的时序状态数据，提取长期依赖关系
    """
    
    def __init__(self, 
                 config: UpperLayerConfig,
                 model_config: ModelConfig,
                 encoder_id: str = "TransformerEncoder_001"):
        """
        初始化Transformer编码器
        
        Args:
            config: 上层配置
            model_config: 模型配置
            encoder_id: 编码器ID
        """
        super(TransformerEncoder, self).__init__()
        
        self.config = config
        self.model_config = model_config
        self.encoder_id = encoder_id
        
        # === 模型参数 ===
        self.d_model = config.hidden_dim
        self.num_heads = config.attention_heads
        self.num_layers = config.transformer_layers
        self.sequence_length = config.sequence_length
        self.input_dim = model_config.upper_state_dim
        
        # === 输入处理 ===
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.sequence_length)
        
        # === Transformer层 ===
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_model * 4,
                dropout=model_config.dropout_rate
            ) for _ in range(self.num_layers)
        ])
        
        # === 输出处理 ===
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(model_config.dropout_rate),
            nn.Linear(self.d_model // 2, self.d_model)
        )
        
        # === 特征提取头 ===
        self.feature_heads = nn.ModuleDict({
            'balance_features': nn.Linear(self.d_model, 64),      # 均衡特征
            'temporal_features': nn.Linear(self.d_model, 64),    # 时序特征
            'degradation_features': nn.Linear(self.d_model, 32), # 劣化特征
            'constraint_features': nn.Linear(self.d_model, 32)   # 约束特征
        })
        
        # === 状态缓存 ===
        self.state_buffer = []
        self.attention_maps = []
        
        print(f"✅ Transformer编码器初始化完成: {encoder_id}")
        print(f"   模型维度: {self.d_model}, 注意力头数: {self.num_heads}, 层数: {self.num_layers}")
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入状态序列 [batch_size, seq_len, input_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
            return_attention: 是否返回注意力权重
            
        Returns:
            编码结果字典
        """
        batch_size, seq_len, _ = x.shape
        
        # === 1. 输入投影 ===
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # === 2. 位置编码 ===
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # === 3. Transformer编码 ===
        attention_weights = []
        
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x, mask)
            
            # 收集注意力权重
            if return_attention and hasattr(encoder_layer.self_attention, 'attention_weights'):
                attention_weights.append(encoder_layer.self_attention.attention_weights)
        
        # === 4. 全局特征提取 ===
        # 池化操作
        pooled_features = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # [batch_size, d_model]
        
        # 输出投影
        global_features = self.output_projection(pooled_features)  # [batch_size, d_model]
        
        # === 5. 专用特征提取 ===
        specialized_features = {}
        for head_name, head_layer in self.feature_heads.items():
            specialized_features[head_name] = head_layer(global_features)
        
        # === 6. 构建输出 ===
        output = {
            'encoded_sequence': x,                        # [batch_size, seq_len, d_model]
            'global_features': global_features,           # [batch_size, d_model]
            'balance_features': specialized_features['balance_features'],
            'temporal_features': specialized_features['temporal_features'],
            'degradation_features': specialized_features['degradation_features'],
            'constraint_features': specialized_features['constraint_features']
        }
        
        if return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def encode_single_step(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        单步编码（用于在线推理）
        
        Args:
            state: 当前状态 [batch_size, input_dim]
            
        Returns:
            编码结果
        """
        # 添加到状态缓存
        self.state_buffer.append(state)
        
        # 维护序列长度
        if len(self.state_buffer) > self.sequence_length:
            self.state_buffer.pop(0)
        
        # 如果缓存不足，用零填充
        if len(self.state_buffer) < self.sequence_length:
            padding_length = self.sequence_length - len(self.state_buffer)
            padding = torch.zeros(state.shape[0], padding_length, state.shape[1], 
                                device=state.device, dtype=state.dtype)
            sequence = torch.cat([padding] + self.state_buffer, dim=1)
        else:
            sequence = torch.stack(self.state_buffer, dim=1)
        
        # 前向编码
        return self.forward(sequence)
    
    def extract_aging_trends(self, encoded_features: torch.Tensor) -> torch.Tensor:
        """
        提取老化趋势特征
        
        Args:
            encoded_features: 编码特征 [batch_size, d_model]
            
        Returns:
            老化趋势特征 [batch_size, aging_feature_dim]
        """
        # 使用专门的劣化特征
        degradation_features = self.feature_heads['degradation_features'](encoded_features)
        
        # 进一步处理提取老化趋势
        aging_trends = torch.sigmoid(degradation_features)  # 归一化到[0,1]
        
        return aging_trends
    
    def extract_balance_patterns(self, encoded_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        提取均衡模式特征
        
        Args:
            encoded_features: 编码特征 [batch_size, d_model]
            
        Returns:
            均衡模式特征字典
        """
        balance_features = self.feature_heads['balance_features'](encoded_features)
        
        # 分解为不同类型的均衡特征
        feature_dim = balance_features.shape[-1] // 3
        
        patterns = {
            'soc_balance_pattern': balance_features[:, :feature_dim],
            'temp_balance_pattern': balance_features[:, feature_dim:2*feature_dim],
            'mixed_balance_pattern': balance_features[:, 2*feature_dim:]
        }
        
        return patterns
    
    def analyze_attention_patterns(self) -> Dict[str, Any]:
        """分析注意力模式"""
        if not self.attention_maps:
            return {'error': 'No attention maps available'}
        
        # 获取最新的注意力权重
        latest_attention = self.attention_maps[-1] if self.attention_maps else None
        
        if latest_attention is None:
            return {'error': 'No recent attention data'}
        
        # 分析注意力分布
        attention_stats = {}
        
        for layer_idx, attention_weights in enumerate(latest_attention):
            # attention_weights: [batch_size, num_heads, seq_len, seq_len]
            
            # 计算注意力集中度
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), 
                dim=-1
            ).mean()
            
            # 计算长程依赖强度
            seq_len = attention_weights.shape[-1]
            distance_weights = torch.zeros_like(attention_weights)
            for i in range(seq_len):
                for j in range(seq_len):
                    distance_weights[:, :, i, j] = abs(i - j)
            
            long_range_strength = torch.sum(
                attention_weights * distance_weights
            ).item() / (seq_len ** 2)
            
            attention_stats[f'layer_{layer_idx}'] = {
                'entropy': attention_entropy.item(),
                'long_range_strength': long_range_strength
            }
        
        return attention_stats
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'encoder_id': self.encoder_id,
            'model_size': {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
            },
            'architecture': {
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'sequence_length': self.sequence_length,
                'input_dim': self.input_dim
            },
            'feature_heads': list(self.feature_heads.keys()),
            'state_buffer_length': len(self.state_buffer)
        }
    
    def reset_state_buffer(self):
        """重置状态缓存"""
        self.state_buffer.clear()
        self.attention_maps.clear()
        print(f"🔄 编码器状态缓存已重置: {self.encoder_id}")
    
    def save_checkpoint(self, filepath: str) -> bool:
        """保存检查点"""
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'model_config': self.model_config,
                'encoder_id': self.encoder_id,
                'state_buffer': self.state_buffer,
                'model_stats': self.get_model_statistics()
            }
            
            torch.save(checkpoint, filepath)
            print(f"✅ 编码器检查点已保存: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 保存检查点失败: {str(e)}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """加载检查点"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.load_state_dict(checkpoint['model_state_dict'])
            self.state_buffer = checkpoint.get('state_buffer', [])
            
            print(f"✅ 编码器检查点已加载: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ 加载检查点失败: {str(e)}")
            return False
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"TransformerEncoder({self.encoder_id}): "
                f"d_model={self.d_model}, heads={self.num_heads}, "
                f"layers={self.num_layers}, seq_len={self.sequence_length}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"TransformerEncoder(encoder_id='{self.encoder_id}', "
                f"d_model={self.d_model}, num_heads={self.num_heads}, "
                f"num_layers={self.num_layers})")
