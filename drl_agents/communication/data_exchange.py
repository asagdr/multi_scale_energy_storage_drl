import numpy as np
import torch
import json
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue
import pickle
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .message_protocol import MessageProtocol, Priority
from .information_flow import InformationFlow, InformationType

class ExchangeMode(Enum):
    """数据交换模式枚举"""
    SYNCHRONOUS = "synchronous"         # 同步模式
    ASYNCHRONOUS = "asynchronous"       # 异步模式
    BUFFERED = "buffered"              # 缓冲模式
    STREAMING = "streaming"            # 流式模式

class DataFormat(Enum):
    """数据格式枚举"""
    JSON = "json"                      # JSON格式
    BINARY = "binary"                  # 二进制格式
    PICKLE = "pickle"                  # Pickle序列化
    NUMPY = "numpy"                    # NumPy格式
    TORCH = "torch"                    # PyTorch格式

@dataclass
class DataSchema:
    """数据模式定义"""
    schema_id: str
    data_type: str
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    validation_rules: Dict[str, callable] = field(default_factory=dict)
    format_specification: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExchangeTransaction:
    """数据交换事务"""
    transaction_id: str
    exchange_type: str
    source_node: str
    target_node: str
    data_schema: DataSchema
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"  # pending, completed, failed, timeout
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class DataValidator:
    """数据验证器"""
    
    def __init__(self):
        self.registered_schemas: Dict[str, DataSchema] = {}
        self.validation_errors = []
    
    def register_schema(self, schema: DataSchema) -> bool:
        """注册数据模式"""
        try:
            self.registered_schemas[schema.schema_id] = schema
            print(f"✅ 已注册数据模式: {schema.schema_id}")
            return True
        except Exception as e:
            print(f"❌ 注册数据模式失败: {str(e)}")
            return False
    
    def validate_data(self, data: Dict[str, Any], schema_id: str) -> Tuple[bool, List[str]]:
        """验证数据"""
        if schema_id not in self.registered_schemas:
            return False, [f"未找到数据模式: {schema_id}"]
        
        schema = self.registered_schemas[schema_id]
        errors = []
        
        # 检查必需字段
        for field in schema.required_fields:
            if field not in data:
                errors.append(f"缺少必需字段: {field}")
        
        # 应用验证规则
        for field, rule in schema.validation_rules.items():
            if field in data:
                try:
                    if not rule(data[field]):
                        errors.append(f"字段验证失败: {field}")
                except Exception as e:
                    errors.append(f"验证规则执行错误: {field} - {str(e)}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            self.validation_errors.extend(errors)
        
        return is_valid, errors

class DataSerializer:
    """数据序列化器"""
    
    def __init__(self):
        self.serializers = {
            DataFormat.JSON: self._json_serialize,
            DataFormat.BINARY: self._binary_serialize,
            DataFormat.PICKLE: self._pickle_serialize,
            DataFormat.NUMPY: self._numpy_serialize,
            DataFormat.TORCH: self._torch_serialize
        }
        
        self.deserializers = {
            DataFormat.JSON: self._json_deserialize,
            DataFormat.BINARY: self._binary_deserialize,
            DataFormat.PICKLE: self._pickle_deserialize,
            DataFormat.NUMPY: self._numpy_deserialize,
            DataFormat.TORCH: self._torch_deserialize
        }
    
    def serialize(self, data: Any, format_type: DataFormat) -> bytes:
        """序列化数据"""
        try:
            serializer = self.serializers.get(format_type)
            if serializer:
                return serializer(data)
            else:
                raise ValueError(f"不支持的序列化格式: {format_type}")
        except Exception as e:
            print(f"❌ 数据序列化失败: {str(e)}")
            raise
    
    def deserialize(self, data: bytes, format_type: DataFormat) -> Any:
        """反序列化数据"""
        try:
            deserializer = self.deserializers.get(format_type)
            if deserializer:
                return deserializer(data)
            else:
                raise ValueError(f"不支持的反序列化格式: {format_type}")
        except Exception as e:
            print(f"❌ 数据反序列化失败: {str(e)}")
            raise
    
    def _json_serialize(self, data: Any) -> bytes:
        """JSON序列化"""
        # 处理NumPy数组和PyTorch张量
        if isinstance(data, np.ndarray):
            data = {'type': 'numpy', 'data': data.tolist(), 'shape': data.shape, 'dtype': str(data.dtype)}
        elif isinstance(data, torch.Tensor):
            data = {'type': 'torch', 'data': data.detach().cpu().numpy().tolist(), 'shape': list(data.shape)}
        
        return json.dumps(data, default=str).encode('utf-8')
    
    def _json_deserialize(self, data: bytes) -> Any:
        """JSON反序列化"""
        decoded_data = json.loads(data.decode('utf-8'))
        
        # 恢复NumPy数组和PyTorch张量
        if isinstance(decoded_data, dict):
            if decoded_data.get('type') == 'numpy':
                return np.array(decoded_data['data']).reshape(decoded_data['shape']).astype(decoded_data['dtype'])
            elif decoded_data.get('type') == 'torch':
                return torch.tensor(decoded_data['data']).reshape(decoded_data['shape'])
        
        return decoded_data
    
    def _binary_serialize(self, data: Any) -> bytes:
        """二进制序列化"""
        return pickle.dumps(data)
    
    def _binary_deserialize(self, data: bytes) -> Any:
        """二进制反序列化"""
        return pickle.loads(data)
    
    def _pickle_serialize(self, data: Any) -> bytes:
        """Pickle序列化"""
        return pickle.dumps(data)
    
    def _pickle_deserialize(self, data: bytes) -> Any:
        """Pickle反序列化"""
        return pickle.loads(data)
    
    def _numpy_serialize(self, data: np.ndarray) -> bytes:
        """NumPy序列化"""
        import io
        buffer = io.BytesIO()
        np.save(buffer, data)
        return buffer.getvalue()
    
    def _numpy_deserialize(self, data: bytes) -> np.ndarray:
        """NumPy反序列化"""
        import io
        buffer = io.BytesIO(data)
        return np.load(buffer)
    
    def _torch_serialize(self, data: torch.Tensor) -> bytes:
        """PyTorch序列化"""
        import io
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()
    
    def _torch_deserialize(self, data: bytes) -> torch.Tensor:
        """PyTorch反序列化"""
        import io
        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location='cpu')

class DataExchange:
    """
    数据交换器
    提供上下层DRL之间的高级数据交换服务
    """
    
    def __init__(self, 
                 exchange_id: str,
                 message_protocol: MessageProtocol,
                 information_flow: InformationFlow):
        """
        初始化数据交换器
        
        Args:
            exchange_id: 交换器ID
            message_protocol: 消息协议
            information_flow: 信息流管理器
        """
        self.exchange_id = exchange_id
        self.message_protocol = message_protocol
        self.information_flow = information_flow
        
        # === 核心组件 ===
        self.data_validator = DataValidator()
        self.data_serializer = DataSerializer()
        
        # === 交换配置 ===
        self.exchange_config = {
            'default_mode': ExchangeMode.ASYNCHRONOUS,
            'default_format': DataFormat.JSON,
            'timeout': 30.0,
            'max_retries': 3,
            'buffer_size': 1000,
            'compression_enabled': True,
            'encryption_enabled': False
        }
        
        # === 事务管理 ===
        self.active_transactions: Dict[str, ExchangeTransaction] = {}
        self.completed_transactions: List[ExchangeTransaction] = []
        self.transaction_lock = threading.Lock()
        
        # === 数据缓冲区 ===
        self.data_buffers: Dict[str, queue.Queue] = {
            'constraint_matrix': queue.Queue(maxsize=10),
            'weight_vector': queue.Queue(maxsize=10),
            'performance_feedback': queue.Queue(maxsize=100),
            'system_state': queue.Queue(maxsize=100),
            'alarm_data': queue.Queue(maxsize=50)
        }
        
        # === 回调函数 ===
        self.data_callbacks: Dict[str, List[Callable]] = {}
        
        # === 统计信息 ===
        self.exchange_statistics = {
            'total_exchanges': 0,
            'successful_exchanges': 0,
            'failed_exchanges': 0,
            'timeout_exchanges': 0,
            'total_bytes_exchanged': 0,
            'average_exchange_time': 0.0
        }
        
        # === 注册默认数据模式 ===
        self._register_default_schemas()
        
        print(f"✅ 数据交换器初始化完成: {exchange_id}")
    
    def _register_default_schemas(self):
        """注册默认数据模式"""
        # 约束矩阵模式
        constraint_schema = DataSchema(
            schema_id="constraint_matrix",
            data_type="tensor",
            required_fields=["constraint_matrix", "matrix_shape", "constraint_type"],
            optional_fields=["validity_period", "priority_level"],
            validation_rules={
                "constraint_matrix": lambda x: isinstance(x, (list, np.ndarray, torch.Tensor)),
                "matrix_shape": lambda x: isinstance(x, (list, tuple)) and len(x) >= 2
            }
        )
        self.data_validator.register_schema(constraint_schema)
        
        # 权重向量模式
        weight_schema = DataSchema(
            schema_id="weight_vector",
            data_type="vector",
            required_fields=["weight_vector", "vector_length", "weight_type"],
            optional_fields=["normalization"],
            validation_rules={
                "weight_vector": lambda x: isinstance(x, (list, np.ndarray, torch.Tensor)),
                "vector_length": lambda x: isinstance(x, int) and x > 0,
                "weight_type": lambda x: isinstance(x, str)
            }
        )
        self.data_validator.register_schema(weight_schema)
        
        # 性能反馈模式
        performance_schema = DataSchema(
            schema_id="performance_feedback",
            data_type="metrics",
            required_fields=["performance_metrics", "measurement_time"],
            optional_fields=["metrics_type", "confidence_level"],
            validation_rules={
                "performance_metrics": lambda x: isinstance(x, dict),
                "measurement_time": lambda x: isinstance(x, (int, float))
            }
        )
        self.data_validator.register_schema(performance_schema)
        
        # 系统状态模式
        state_schema = DataSchema(
            schema_id="system_state",
            data_type="state",
            required_fields=["system_state", "state_timestamp"],
            optional_fields=["state_type", "sensor_data"],
            validation_rules={
                "system_state": lambda x: isinstance(x, dict),
                "state_timestamp": lambda x: isinstance(x, (int, float))
            }
        )
        self.data_validator.register_schema(state_schema)
        
        # 告警数据模式
        alarm_schema = DataSchema(
            schema_id="alarm_data",
            data_type="alarm",
            required_fields=["alarm_info", "alarm_timestamp", "severity"],
            optional_fields=["alarm_source", "recommended_actions"],
            validation_rules={
                "alarm_info": lambda x: isinstance(x, dict),
                "alarm_timestamp": lambda x: isinstance(x, (int, float)),
                "severity": lambda x: isinstance(x, str) and x in ["low", "medium", "high", "critical"]
            }
        )
        self.data_validator.register_schema(alarm_schema)
    
    def exchange_constraint_matrix(self, 
                                 constraint_matrix: torch.Tensor,
                                 target_node: str,
                                 mode: ExchangeMode = ExchangeMode.ASYNCHRONOUS,
                                 priority: Priority = Priority.HIGH) -> str:
        """
        交换约束矩阵
        
        Args:
            constraint_matrix: 约束矩阵
            target_node: 目标节点
            mode: 交换模式
            priority: 优先级
            
        Returns:
            事务ID
        """
        try:
            # 创建事务
            transaction_id = f"constraint_{int(time.time()*1000)}"
            transaction = ExchangeTransaction(
                transaction_id=transaction_id,
                exchange_type="constraint_matrix",
                source_node=self.exchange_id,
                target_node=target_node,
                data_schema=self.data_validator.registered_schemas["constraint_matrix"],
                start_time=time.time()
            )
            
            # 准备数据
            data = {
                "constraint_matrix": constraint_matrix.detach().cpu().numpy().tolist(),
                "matrix_shape": list(constraint_matrix.shape),
                "constraint_type": "dynamic_operational_constraints",
                "validity_period": 300.0,
                "priority_level": priority.value
            }
            
            # 验证数据
            is_valid, errors = self.data_validator.validate_data(data, "constraint_matrix")
            if not is_valid:
                transaction.status = "failed"
                transaction.error_message = f"数据验证失败: {errors}"
                return transaction_id
            
            # 执行交换
            success = self._execute_exchange(transaction, data, mode)
            
            if success:
                transaction.status = "completed"
                self.exchange_statistics['successful_exchanges'] += 1
            else:
                transaction.status = "failed"
                self.exchange_statistics['failed_exchanges'] += 1
            
            transaction.end_time = time.time()
            
            # 更新统计
            self.exchange_statistics['total_exchanges'] += 1
            exchange_time = transaction.end_time - transaction.start_time
            self.exchange_statistics['average_exchange_time'] = (
                self.exchange_statistics['average_exchange_time'] * 0.9 + exchange_time * 0.1
            )
            
            # 管理事务
            with self.transaction_lock:
                if transaction_id in self.active_transactions:
                    del self.active_transactions[transaction_id]
                self.completed_transactions.append(transaction)
                
                # 维护历史长度
                if len(self.completed_transactions) > 1000:
                    self.completed_transactions.pop(0)
            
            return transaction_id
            
        except Exception as e:
            print(f"❌ 约束矩阵交换失败: {str(e)}")
            self.exchange_statistics['failed_exchanges'] += 1
            return ""
    
    def exchange_weight_vector(self, 
                             weight_vector: torch.Tensor,
                             target_node: str,
                             mode: ExchangeMode = ExchangeMode.ASYNCHRONOUS,
                             priority: Priority = Priority.HIGH) -> str:
        """
        交换权重向量
        
        Args:
            weight_vector: 权重向量
            target_node: 目标节点
            mode: 交换模式
            priority: 优先级
            
        Returns:
            事务ID
        """
        try:
            transaction_id = f"weight_{int(time.time()*1000)}"
            transaction = ExchangeTransaction(
                transaction_id=transaction_id,
                exchange_type="weight_vector",
                source_node=self.exchange_id,
                target_node=target_node,
                data_schema=self.data_validator.registered_schemas["weight_vector"],
                start_time=time.time()
            )
            
            data = {
                "weight_vector": weight_vector.detach().cpu().numpy().tolist(),
                "vector_length": len(weight_vector),
                "weight_type": "multi_objective_weights",
                "normalization": "sum_to_one"
            }
            
            is_valid, errors = self.data_validator.validate_data(data, "weight_vector")
            if not is_valid:
                transaction.status = "failed"
                transaction.error_message = f"数据验证失败: {errors}"
                return transaction_id
            
            success = self._execute_exchange(transaction, data, mode)
            
            transaction.status = "completed" if success else "failed"
            transaction.end_time = time.time()
            
            self._update_transaction_statistics(transaction, success)
            
            return transaction_id
            
        except Exception as e:
            print(f"❌ 权重向量交换失败: {str(e)}")
            self.exchange_statistics['failed_exchanges'] += 1
            return ""
    
    def exchange_performance_feedback(self, 
                                    performance_data: Dict[str, Any],
                                    target_node: str,
                                    mode: ExchangeMode = ExchangeMode.BUFFERED) -> str:
        """
        交换性能反馈
        
        Args:
            performance_data: 性能数据
            target_node: 目标节点
            mode: 交换模式
            
        Returns:
            事务ID
        """
        try:
            transaction_id = f"perf_{int(time.time()*1000)}"
            transaction = ExchangeTransaction(
                transaction_id=transaction_id,
                exchange_type="performance_feedback",
                source_node=self.exchange_id,
                target_node=target_node,
                data_schema=self.data_validator.registered_schemas["performance_feedback"],
                start_time=time.time()
            )
            
            data = {
                "performance_metrics": performance_data,
                "measurement_time": time.time(),
                "metrics_type": "real_time_performance"
            }
            
            is_valid, errors = self.data_validator.validate_data(data, "performance_feedback")
            if not is_valid:
                transaction.status = "failed"
                transaction.error_message = f"数据验证失败: {errors}"
                return transaction_id
            
            success = self._execute_exchange(transaction, data, mode)
            
            transaction.status = "completed" if success else "failed"
            transaction.end_time = time.time()
            
            self._update_transaction_statistics(transaction, success)
            
            return transaction_id
            
        except Exception as e:
            print(f"❌ 性能反馈交换失败: {str(e)}")
            self.exchange_statistics['failed_exchanges'] += 1
            return ""
    
    def _execute_exchange(self, 
                         transaction: ExchangeTransaction,
                         data: Dict[str, Any],
                         mode: ExchangeMode) -> bool:
        """执行数据交换"""
        try:
            with self.transaction_lock:
                self.active_transactions[transaction.transaction_id] = transaction
            
            if mode == ExchangeMode.SYNCHRONOUS:
                return self._synchronous_exchange(transaction, data)
            elif mode == ExchangeMode.ASYNCHRONOUS:
                return self._asynchronous_exchange(transaction, data)
            elif mode == ExchangeMode.BUFFERED:
                return self._buffered_exchange(transaction, data)
            elif mode == ExchangeMode.STREAMING:
                return self._streaming_exchange(transaction, data)
            else:
                print(f"⚠️ 不支持的交换模式: {mode}")
                return False
                
        except Exception as e:
            print(f"❌ 执行数据交换失败: {str(e)}")
            return False
    
    def _synchronous_exchange(self, 
                            transaction: ExchangeTransaction,
                            data: Dict[str, Any]) -> bool:
        """同步交换"""
        try:
            # 序列化数据
            serialized_data = self.data_serializer.serialize(data, self.exchange_config['default_format'])
            
            # 通过信息流发送
            if transaction.exchange_type == "constraint_matrix":
                success = self.information_flow.send_constraint_matrix(
                    torch.tensor(data["constraint_matrix"]),
                    transaction.target_node,
                    Priority.HIGH
                )
            elif transaction.exchange_type == "weight_vector":
                success = self.information_flow.send_weight_vector(
                    torch.tensor(data["weight_vector"]),
                    transaction.target_node,
                    Priority.HIGH
                )
            elif transaction.exchange_type == "performance_feedback":
                success = self.information_flow.send_performance_feedback(
                    data["performance_metrics"],
                    transaction.target_node,
                    Priority.NORMAL
                )
            else:
                success = False
            
            if success:
                self.exchange_statistics['total_bytes_exchanged'] += len(serialized_data)
            
            return success
            
        except Exception as e:
            print(f"❌ 同步交换失败: {str(e)}")
            return False
    
    def _asynchronous_exchange(self, 
                             transaction: ExchangeTransaction,
                             data: Dict[str, Any]) -> bool:
        """异步交换"""
        def async_worker():
            try:
                return self._synchronous_exchange(transaction, data)
            except Exception as e:
                print(f"❌ 异步交换工作线程失败: {str(e)}")
                return False
        
        # 启动异步线程
        thread = threading.Thread(target=async_worker, daemon=True)
        thread.start()
        
        return True  # 异步模式立即返回成功
    
    def _buffered_exchange(self, 
                         transaction: ExchangeTransaction,
                         data: Dict[str, Any]) -> bool:
        """缓冲交换"""
        try:
            exchange_type = transaction.exchange_type
            
            if exchange_type in self.data_buffers:
                buffer = self.data_buffers[exchange_type]
                
                try:
                    buffer.put({
                        'transaction_id': transaction.transaction_id,
                        'data': data,
                        'timestamp': time.time()
                    }, block=False)
                    
                    return True
                    
                except queue.Full:
                    print(f"⚠️ 缓冲区满，数据被丢弃: {exchange_type}")
                    return False
            else:
                print(f"⚠️ 未找到对应缓冲区: {exchange_type}")
                return False
                
        except Exception as e:
            print(f"❌ 缓冲交换失败: {str(e)}")
            return False
    
    def _streaming_exchange(self, 
                          transaction: ExchangeTransaction,
                          data: Dict[str, Any]) -> bool:
        """流式交换"""
        try:
            # 简化的流式处理：分块发送大数据
            chunk_size = 1024  # 1KB块
            serialized_data = self.data_serializer.serialize(data, DataFormat.BINARY)
            
            total_chunks = len(serialized_data) // chunk_size + (1 if len(serialized_data) % chunk_size > 0 else 0)
            
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(serialized_data))
                chunk = serialized_data[start_idx:end_idx]
                
                chunk_data = {
                    'transaction_id': transaction.transaction_id,
                    'chunk_index': i,
                    'total_chunks': total_chunks,
                    'chunk_data': chunk.hex(),
                    'chunk_size': len(chunk)
                }
                
                # 发送块
                success = self.information_flow.send_system_state(
                    chunk_data,
                    transaction.target_node,
                    Priority.NORMAL
                )
                
                if not success:
                    print(f"❌ 流式交换块发送失败: {i}/{total_chunks}")
                    return False
                
                time.sleep(0.001)  # 1ms间隔
            
            return True
            
        except Exception as e:
            print(f"❌ 流式交换失败: {str(e)}")
            return False
    
    def _update_transaction_statistics(self, 
                                     transaction: ExchangeTransaction,
                                     success: bool):
        """更新事务统计"""
        self.exchange_statistics['total_exchanges'] += 1
        
        if success:
            self.exchange_statistics['successful_exchanges'] += 1
        else:
            self.exchange_statistics['failed_exchanges'] += 1
        
        if transaction.end_time:
            exchange_time = transaction.end_time - transaction.start_time
            self.exchange_statistics['average_exchange_time'] = (
                self.exchange_statistics['average_exchange_time'] * 0.9 + exchange_time * 0.1
            )
        
        # 管理事务
        with self.transaction_lock:
            if transaction.transaction_id in self.active_transactions:
                del self.active_transactions[transaction.transaction_id]
            self.completed_transactions.append(transaction)
    
    def register_callback(self, data_type: str, callback: Callable):
        """注册数据回调"""
        if data_type not in self.data_callbacks:
            self.data_callbacks[data_type] = []
        
        self.data_callbacks[data_type].append(callback)
        print(f"📝 已注册 {data_type} 数据回调")
    
    def process_buffered_data(self):
        """处理缓冲区数据"""
        for exchange_type, buffer in self.data_buffers.items():
            try:
                while not buffer.empty():
                    buffered_item = buffer.get(block=False)
                    
                    # 触发回调
                    if exchange_type in self.data_callbacks:
                        for callback in self.data_callbacks[exchange_type]:
                            try:
                                callback(buffered_item['data'])
                            except Exception as e:
                                print(f"⚠️ 回调执行失败: {str(e)}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 处理缓冲数据失败: {str(e)}")
    
    def get_transaction_status(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """获取事务状态"""
        with self.transaction_lock:
            # 检查活跃事务
            if transaction_id in self.active_transactions:
                transaction = self.active_transactions[transaction_id]
                return {
                    'transaction_id': transaction.transaction_id,
                    'status': transaction.status,
                    'exchange_type': transaction.exchange_type,
                    'start_time': transaction.start_time,
                    'elapsed_time': time.time() - transaction.start_time,
                    'retry_count': transaction.retry_count
                }
            
            # 检查完成事务
            for transaction in self.completed_transactions:
                if transaction.transaction_id == transaction_id:
                    return {
                        'transaction_id': transaction.transaction_id,
                        'status': transaction.status,
                        'exchange_type': transaction.exchange_type,
                        'start_time': transaction.start_time,
                        'end_time': transaction.end_time,
                        'total_time': transaction.end_time - transaction.start_time if transaction.end_time else None,
                        'error_message': transaction.error_message
                    }
        
        return None
    def get_exchange_statistics(self) -> Dict[str, Any]:
        """获取交换统计信息"""
        with self.transaction_lock:
            active_count = len(self.active_transactions)
            completed_count = len(self.completed_transactions)
        
        # 计算成功率
        total_exchanges = self.exchange_statistics['total_exchanges']
        success_rate = (self.exchange_statistics['successful_exchanges'] / total_exchanges 
                       if total_exchanges > 0 else 0.0)
        
        # 缓冲区状态
        buffer_status = {}
        for name, buffer in self.data_buffers.items():
            buffer_status[name] = {
                'size': buffer.qsize(),
                'maxsize': buffer.maxsize,
                'usage_ratio': buffer.qsize() / buffer.maxsize if buffer.maxsize > 0 else 0.0
            }
        
        stats = {
            'exchange_id': self.exchange_id,
            'exchange_statistics': self.exchange_statistics.copy(),
            'success_rate': success_rate,
            
            'transaction_status': {
                'active_transactions': active_count,
                'completed_transactions': completed_count,
                'total_transactions': active_count + completed_count
            },
            
            'buffer_status': buffer_status,
            
            'validator_status': {
                'registered_schemas': len(self.data_validator.registered_schemas),
                'validation_errors': len(self.data_validator.validation_errors)
            },
            
            'callback_status': {
                'registered_callbacks': sum(len(callbacks) for callbacks in self.data_callbacks.values()),
                'callback_types': list(self.data_callbacks.keys())
            },
            
            'configuration': self.exchange_config.copy()
        }
        
        return stats
    
    def cleanup_completed_transactions(self, max_age: float = 3600.0):
        """清理完成的事务"""
        current_time = time.time()
        
        with self.transaction_lock:
            # 清理过期的完成事务
            self.completed_transactions = [
                t for t in self.completed_transactions 
                if t.end_time and (current_time - t.end_time) < max_age
            ]
            
            # 清理超时的活跃事务
            timeout_transactions = []
            for tid, transaction in self.active_transactions.items():
                if (current_time - transaction.start_time) > self.exchange_config['timeout']:
                    transaction.status = "timeout"
                    transaction.end_time = current_time
                    timeout_transactions.append(tid)
                    self.exchange_statistics['timeout_exchanges'] += 1
            
            for tid in timeout_transactions:
                completed_transaction = self.active_transactions.pop(tid)
                self.completed_transactions.append(completed_transaction)
        
        if timeout_transactions:
            print(f"🧹 已清理 {len(timeout_transactions)} 个超时事务")
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"DataExchange({self.exchange_id}): "
                f"total={self.exchange_statistics['total_exchanges']}, "
                f"success={self.exchange_statistics['successful_exchanges']}, "
                f"active={len(self.active_transactions)}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"DataExchange(exchange_id='{self.exchange_id}', "
                f"total_exchanges={self.exchange_statistics['total_exchanges']}, "
                f"active_transactions={len(self.active_transactions)})")
