import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import queue
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class MessageType(Enum):
    """消息类型枚举"""
    # 控制类消息
    CONSTRAINT_UPDATE = "constraint_update"      # 约束更新
    WEIGHT_UPDATE = "weight_update"              # 权重更新
    BALANCE_TARGET = "balance_target"            # 均衡目标
    POWER_COMMAND = "power_command"              # 功率指令
    SAFETY_COMMAND = "safety_command"            # 安全指令
    
    # 反馈类消息
    PERFORMANCE_REPORT = "performance_report"    # 性能报告
    STATUS_UPDATE = "status_update"              # 状态更新
    CONSTRAINT_VIOLATION = "constraint_violation" # 约束违反
    ALARM_NOTIFICATION = "alarm_notification"    # 告警通知
    
    # 系统类消息
    HEARTBEAT = "heartbeat"                      # 心跳
    SYNC_REQUEST = "sync_request"                # 同步请求
    SYNC_RESPONSE = "sync_response"              # 同步响应
    SHUTDOWN = "shutdown"                        # 关机指令

class Priority(Enum):
    """消息优先级枚举"""
    LOW = 1          # 低优先级
    NORMAL = 2       # 普通优先级
    HIGH = 3         # 高优先级
    URGENT = 4       # 紧急优先级
    CRITICAL = 5     # 关键优先级

@dataclass
class MessageHeader:
    """消息头部"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.HEARTBEAT
    priority: Priority = Priority.NORMAL
    sender_id: str = ""
    receiver_id: str = ""
    timestamp: float = field(default_factory=time.time)
    sequence_number: int = 0
    session_id: str = ""
    ttl: float = 30.0  # 消息生存时间 (s)
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        return time.time() > self.timestamp + self.ttl

@dataclass
class Message:
    """通信消息"""
    header: MessageHeader
    payload: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """计算校验和"""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算消息校验和"""
        import hashlib
        
        # 将消息内容序列化
        content = {
            'header': asdict(self.header),
            'payload': self.payload
        }
        
        message_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(message_str.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """验证消息完整性"""
        expected_checksum = self._calculate_checksum()
        return self.checksum == expected_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'header': asdict(self.header),
            'payload': self.payload,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建消息"""
        header_data = data['header']
        header = MessageHeader(
            message_id=header_data['message_id'],
            message_type=MessageType(header_data['message_type']),
            priority=Priority(header_data['priority']),
            sender_id=header_data['sender_id'],
            receiver_id=header_data['receiver_id'],
            timestamp=header_data['timestamp'],
            sequence_number=header_data['sequence_number'],
            session_id=header_data['session_id'],
            ttl=header_data['ttl']
        )
        
        return cls(
            header=header,
            payload=data['payload'],
            checksum=data['checksum']
        )

class MessageQueue:
    """优先级消息队列"""
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.queues = {priority: queue.Queue() for priority in Priority}
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self.total_size = 0
    
    def put(self, message: Message, block: bool = True, timeout: Optional[float] = None) -> bool:
        """放入消息"""
        with self._condition:
            # 检查队列容量
            if self.total_size >= self.maxsize and not block:
                return False
            
            # 等待空间
            while self.total_size >= self.maxsize and block:
                if timeout is not None:
                    if not self._condition.wait(timeout):
                        return False
                else:
                    self._condition.wait()
            
            # 检查消息是否过期
            if message.header.is_expired():
                return False
            
            # 放入对应优先级队列
            self.queues[message.header.priority].put(message)
            self.total_size += 1
            self._condition.notify_all()
            
            return True
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Message]:
        """获取消息（按优先级）"""
        with self._condition:
            # 等待消息
            while self.total_size == 0 and block:
                if timeout is not None:
                    if not self._condition.wait(timeout):
                        return None
                else:
                    self._condition.wait()
            
            if self.total_size == 0:
                return None
            
            # 按优先级顺序获取消息
            for priority in sorted(Priority, key=lambda p: p.value, reverse=True):
                if not self.queues[priority].empty():
                    message = self.queues[priority].get()
                    self.total_size -= 1
                    self._condition.notify_all()
                    
                    # 检查消息是否过期
                    if message.header.is_expired():
                        continue  # 跳过过期消息
                    
                    return message
            
            return None
    
    def size(self) -> int:
        """获取队列大小"""
        return self.total_size
    
    def is_empty(self) -> bool:
        """检查队列是否为空"""
        return self.total_size == 0
    
    def clear(self):
        """清空队列"""
        with self._condition:
            for q in self.queues.values():
                while not q.empty():
                    q.get()
            self.total_size = 0

class MessageProtocol:
    """
    DRL消息协议
    定义上下层DRL之间的标准化通信协议
    """
    
    def __init__(self, 
                 node_id: str,
                 protocol_id: str = "DRLProtocol_001"):
        """
        初始化消息协议
        
        Args:
            node_id: 节点ID
            protocol_id: 协议ID
        """
        self.node_id = node_id
        self.protocol_id = protocol_id
        
        # === 消息队列 ===
        self.incoming_queue = MessageQueue(maxsize=1000)
        self.outgoing_queue = MessageQueue(maxsize=1000)
        
        # === 会话管理 ===
        self.active_sessions: Dict[str, Dict] = {}
        self.sequence_numbers: Dict[str, int] = {}
        
        # === 统计信息 ===
        self.message_stats = {
            'sent': 0,
            'received': 0,
            'dropped': 0,
            'corrupted': 0,
            'expired': 0
        }
        
        # === 协议配置 ===
        self.config = {
            'heartbeat_interval': 5.0,     # 心跳间隔 (s)
            'max_retries': 3,              # 最大重试次数
            'ack_timeout': 1.0,            # ACK超时 (s)
            'session_timeout': 300.0,      # 会话超时 (s)
            'enable_compression': False,    # 是否启用压缩
            'enable_encryption': False      # 是否启用加密
        }
        
        # === 消息处理器 ===
        self.message_handlers: Dict[MessageType, callable] = {}
        
        # === 线程控制 ===
        self.running = False
        self.heartbeat_thread = None
        self.cleanup_thread = None
        
        print(f"✅ DRL消息协议初始化完成: {protocol_id}")
        print(f"   节点ID: {node_id}")
    
    def register_handler(self, message_type: MessageType, handler: callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
        print(f"📝 已注册 {message_type.value} 消息处理器")
    
    def create_session(self, peer_id: str) -> str:
        """创建通信会话"""
        session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            'peer_id': peer_id,
            'created_time': time.time(),
            'last_activity': time.time(),
            'status': 'active'
        }
        
        self.sequence_numbers[session_id] = 0
        
        print(f"🔗 已创建会话: {session_id} -> {peer_id}")
        return session_id
    
    def send_message(self, 
                    message_type: MessageType,
                    payload: Dict[str, Any],
                    receiver_id: str,
                    priority: Priority = Priority.NORMAL,
                    session_id: Optional[str] = None) -> bool:
        """
        发送消息
        
        Args:
            message_type: 消息类型
            payload: 消息负载
            receiver_id: 接收者ID
            priority: 消息优先级
            session_id: 会话ID
            
        Returns:
            发送成功标志
        """
        try:
            # 创建消息头
            if session_id and session_id in self.sequence_numbers:
                seq_num = self.sequence_numbers[session_id]
                self.sequence_numbers[session_id] += 1
            else:
                seq_num = 0
            
            header = MessageHeader(
                message_type=message_type,
                priority=priority,
                sender_id=self.node_id,
                receiver_id=receiver_id,
                sequence_number=seq_num,
                session_id=session_id or ""
            )
            
            # 创建消息
            message = Message(header=header, payload=payload)
            
            # 加入发送队列
            success = self.outgoing_queue.put(message, block=False)
            
            if success:
                self.message_stats['sent'] += 1
                
                # 更新会话活动时间
                if session_id and session_id in self.active_sessions:
                    self.active_sessions[session_id]['last_activity'] = time.time()
                
                return True
            else:
                self.message_stats['dropped'] += 1
                return False
                
        except Exception as e:
            print(f"❌ 发送消息失败: {str(e)}")
            self.message_stats['dropped'] += 1
            return False
    
    def receive_message(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        接收消息
        
        Args:
            timeout: 超时时间 (s)
            
        Returns:
            接收到的消息或None
        """
        try:
            message = self.incoming_queue.get(block=True, timeout=timeout)
            
            if message:
                # 验证消息完整性
                if not message.verify_integrity():
                    self.message_stats['corrupted'] += 1
                    return None
                
                # 检查消息是否过期
                if message.header.is_expired():
                    self.message_stats['expired'] += 1
                    return None
                
                self.message_stats['received'] += 1
                
                # 更新会话活动时间
                session_id = message.header.session_id
                if session_id and session_id in self.active_sessions:
                    self.active_sessions[session_id]['last_activity'] = time.time()
                
                return message
            
            return None
            
        except Exception as e:
            print(f"❌ 接收消息失败: {str(e)}")
            return None
    
    def process_message(self, message: Message) -> bool:
        """
        处理接收到的消息
        
        Args:
            message: 消息对象
            
        Returns:
            处理成功标志
        """
        try:
            message_type = message.header.message_type
            
            # 检查是否有注册的处理器
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                result = handler(message)
                return result
            else:
                print(f"⚠️ 未找到 {message_type.value} 消息处理器")
                return False
                
        except Exception as e:
            print(f"❌ 处理消息失败: {str(e)}")
            return False
    
    def send_constraint_update(self, 
                             constraint_matrix: Any,
                             receiver_id: str,
                             session_id: Optional[str] = None) -> bool:
        """发送约束更新消息"""
        payload = {
            'constraint_matrix': constraint_matrix.tolist() if hasattr(constraint_matrix, 'tolist') else constraint_matrix,
            'update_time': time.time(),
            'constraint_type': 'dynamic_constraints'
        }
        
        return self.send_message(
            MessageType.CONSTRAINT_UPDATE,
            payload,
            receiver_id,
            Priority.HIGH,
            session_id
        )
    
    def send_weight_update(self, 
                          weight_vector: Any,
                          receiver_id: str,
                          session_id: Optional[str] = None) -> bool:
        """发送权重更新消息"""
        payload = {
            'weight_vector': weight_vector.tolist() if hasattr(weight_vector, 'tolist') else weight_vector,
            'update_time': time.time(),
            'weight_type': 'multi_objective_weights'
        }
        
        return self.send_message(
            MessageType.WEIGHT_UPDATE,
            payload,
            receiver_id,
            Priority.HIGH,
            session_id
        )
    
    def send_performance_report(self, 
                              performance_data: Dict[str, Any],
                              receiver_id: str,
                              session_id: Optional[str] = None) -> bool:
        """发送性能报告"""
        payload = {
            'performance_metrics': performance_data,
            'report_time': time.time(),
            'reporter_id': self.node_id
        }
        
        return self.send_message(
            MessageType.PERFORMANCE_REPORT,
            payload,
            receiver_id,
            Priority.NORMAL,
            session_id
        )
    
    def send_alarm_notification(self, 
                              alarm_data: Dict[str, Any],
                              receiver_id: str,
                              session_id: Optional[str] = None) -> bool:
        """发送告警通知"""
        payload = {
            'alarm_type': alarm_data.get('type', 'unknown'),
            'severity': alarm_data.get('severity', 'low'),
            'description': alarm_data.get('description', ''),
            'alarm_time': time.time(),
            'alarm_data': alarm_data
        }
        
        return self.send_message(
            MessageType.ALARM_NOTIFICATION,
            payload,
            receiver_id,
            Priority.URGENT,
            session_id
        )
    
    def send_heartbeat(self, receiver_id: str, session_id: Optional[str] = None) -> bool:
        """发送心跳消息"""
        payload = {
            'node_status': 'active',
            'uptime': time.time(),
            'queue_sizes': {
                'incoming': self.incoming_queue.size(),
                'outgoing': self.outgoing_queue.size()
            }
        }
        
        return self.send_message(
            MessageType.HEARTBEAT,
            payload,
            receiver_id,
            Priority.LOW,
            session_id
        )
    
    def start_heartbeat(self, peer_id: str, session_id: Optional[str] = None):
        """启动心跳发送"""
        def heartbeat_worker():
            while self.running:
                self.send_heartbeat(peer_id, session_id)
                time.sleep(self.config['heartbeat_interval'])
        
        if not self.running:
            self.running = True
            self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
            self.heartbeat_thread.start()
            print(f"💓 心跳线程已启动")
    
    def start_cleanup_service(self):
        """启动清理服务"""
        def cleanup_worker():
            while self.running:
                current_time = time.time()
                
                # 清理过期会话
                expired_sessions = []
                for session_id, session_info in self.active_sessions.items():
                    if current_time - session_info['last_activity'] > self.config['session_timeout']:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                    if session_id in self.sequence_numbers:
                        del self.sequence_numbers[session_id]
                    print(f"🧹 已清理过期会话: {session_id}")
                
                time.sleep(60.0)  # 每分钟清理一次
        
        if not self.running:
            self.running = True
            self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
            self.cleanup_thread.start()
            print(f"🧹 清理服务已启动")
    
    def stop(self):
        """停止协议服务"""
        self.running = False
        
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join(timeout=1.0)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=1.0)
        
        print(f"⏹️ 消息协议已停止: {self.protocol_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取协议统计信息"""
        stats = {
            'protocol_id': self.protocol_id,
            'node_id': self.node_id,
            'running': self.running,
            
            'message_statistics': self.message_stats.copy(),
            'message_rates': {
                'send_rate': self.message_stats['sent'] / max(time.time() - getattr(self, 'start_time', time.time()), 1),
                'receive_rate': self.message_stats['received'] / max(time.time() - getattr(self, 'start_time', time.time()), 1)
            },
            
            'queue_status': {
                'incoming_size': self.incoming_queue.size(),
                'outgoing_size': self.outgoing_queue.size(),
                'incoming_empty': self.incoming_queue.is_empty(),
                'outgoing_empty': self.outgoing_queue.is_empty()
            },
            
            'session_status': {
                'active_sessions': len(self.active_sessions),
                'session_ids': list(self.active_sessions.keys())
            },
            
            'configuration': self.config.copy()
        }
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"MessageProtocol({self.protocol_id}): "
                f"node={self.node_id}, sessions={len(self.active_sessions)}, "
                f"sent={self.message_stats['sent']}, received={self.message_stats['received']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"MessageProtocol(protocol_id='{self.protocol_id}', "
                f"node_id='{self.node_id}', "
                f"active_sessions={len(self.active_sessions)})")
