import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import queue
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .message_protocol import MessageProtocol, MessageType, Priority

class FlowDirection(Enum):
    """信息流方向枚举"""
    UPWARD = "upward"           # 下层→上层
    DOWNWARD = "downward"       # 上层→下层
    BIDIRECTIONAL = "bidirectional"  # 双向

class InformationType(Enum):
    """信息类型枚举"""
    CONTROL_COMMAND = "control_command"     # 控制指令
    STATE_FEEDBACK = "state_feedback"      # 状态反馈
    CONSTRAINT_DATA = "constraint_data"    # 约束数据
    PERFORMANCE_DATA = "performance_data"  # 性能数据
    ALARM_DATA = "alarm_data"              # 告警数据
    DIAGNOSTIC_DATA = "diagnostic_data"    # 诊断数据

@dataclass
class InformationPacket:
    """信息数据包"""
    packet_id: str
    info_type: InformationType
    flow_direction: FlowDirection
    source_layer: str           # 源层级
    target_layer: str          # 目标层级
    timestamp: float
    data: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    reliability_required: bool = True
    sequence_number: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

class FlowFilter:
    """信息流过滤器"""
    
    def __init__(self, filter_id: str):
        self.filter_id = filter_id
        self.rules: List[Dict[str, Any]] = []
        self.statistics = {
            'total_packets': 0,
            'passed_packets': 0,
            'filtered_packets': 0
        }
    
    def add_rule(self, 
                rule_type: str,
                condition: callable,
                action: str = "pass") -> bool:
        """
        添加过滤规则
        
        Args:
            rule_type: 规则类型
            condition: 条件函数
            action: 动作 ("pass", "block", "modify")
        """
        try:
            rule = {
                'rule_type': rule_type,
                'condition': condition,
                'action': action,
                'created_time': time.time(),
                'match_count': 0
            }
            
            self.rules.append(rule)
            print(f"✅ 已添加过滤规则: {rule_type}")
            return True
            
        except Exception as e:
            print(f"❌ 添加过滤规则失败: {str(e)}")
            return False
    
    def filter_packet(self, packet: InformationPacket) -> Tuple[bool, Optional[InformationPacket]]:
        """
        过滤信息包
        
        Args:
            packet: 信息包
            
        Returns:
            (是否通过, 修改后的包)
        """
        self.statistics['total_packets'] += 1
        
        for rule in self.rules:
            try:
                if rule['condition'](packet):
                    rule['match_count'] += 1
                    
                    if rule['action'] == "block":
                        self.statistics['filtered_packets'] += 1
                        return False, None
                    elif rule['action'] == "modify":
                        # 这里可以添加修改逻辑
                        self.statistics['passed_packets'] += 1
                        return True, packet
                    # "pass" action继续处理
            except Exception as e:
                print(f"⚠️ 过滤规则执行失败: {str(e)}")
        
        self.statistics['passed_packets'] += 1
        return True, packet

class QualityOfService:
    """服务质量管理"""
    
    def __init__(self):
        self.bandwidth_limits = {
            InformationType.CONTROL_COMMAND: 1000,      # 包/秒
            InformationType.STATE_FEEDBACK: 100,        # 包/秒
            InformationType.CONSTRAINT_DATA: 50,        # 包/秒
            InformationType.PERFORMANCE_DATA: 20,       # 包/秒
            InformationType.ALARM_DATA: 100,            # 包/秒
            InformationType.DIAGNOSTIC_DATA: 10         # 包/秒
        }
        
        self.traffic_counters = {info_type: 0 for info_type in InformationType}
        self.last_reset_time = time.time()
        self.qos_violations = 0
    
    def check_bandwidth(self, packet: InformationPacket) -> bool:
        """检查带宽限制"""
        current_time = time.time()
        
        # 每秒重置计数器
        if current_time - self.last_reset_time >= 1.0:
            self.traffic_counters = {info_type: 0 for info_type in InformationType}
            self.last_reset_time = current_time
        
        # 检查当前信息类型的带宽
        info_type = packet.info_type
        current_count = self.traffic_counters[info_type]
        limit = self.bandwidth_limits.get(info_type, 100)
        
        if current_count >= limit:
            self.qos_violations += 1
            return False
        
        self.traffic_counters[info_type] += 1
        return True
    
    def get_qos_status(self) -> Dict[str, Any]:
        """获取QoS状态"""
        return {
            'bandwidth_limits': self.bandwidth_limits.copy(),
            'current_traffic': self.traffic_counters.copy(),
            'qos_violations': self.qos_violations,
            'last_reset_time': self.last_reset_time
        }

class InformationFlow:
    """
    信息流管理器
    管理上下层DRL之间的信息交换和流量控制
    """
    
    def __init__(self, 
                 flow_id: str,
                 message_protocol: MessageProtocol):
        """
        初始化信息流管理器
        
        Args:
            flow_id: 流管理器ID
            message_protocol: 消息协议实例
        """
        self.flow_id = flow_id
        self.message_protocol = message_protocol
        
        # === 信息流配置 ===
        self.flow_config = {
            'max_packet_size': 1024 * 1024,    # 1MB
            'buffer_size': 10000,               # 缓冲区大小
            'timeout': 30.0,                    # 超时时间 (s)
            'retry_attempts': 3,                # 重试次数
            'enable_compression': True,          # 启用压缩
            'enable_encryption': False           # 启用加密
        }
        
        # === 信息流缓冲区 ===
        self.upward_buffer = queue.Queue(maxsize=self.flow_config['buffer_size'])
        self.downward_buffer = queue.Queue(maxsize=self.flow_config['buffer_size'])
        
        # === 过滤器和QoS ===
        self.upward_filter = FlowFilter("UpwardFilter")
        self.downward_filter = FlowFilter("DownwardFilter")
        self.qos_manager = QualityOfService()
        
        # === 流量统计 ===
        self.flow_statistics = {
            'upward_packets': 0,
            'downward_packets': 0,
            'dropped_packets': 0,
            'error_packets': 0,
            'total_bytes': 0,
            'compression_ratio': 1.0
        }
        
        # === 同步机制 ===
        self.synchronization_enabled = True
        self.sync_interval = 5.0  # 5秒同步一次
        self.last_sync_time = time.time()
        
        # === 线程控制 ===
        self.running = False
        self.flow_threads: List[threading.Thread] = []
        
        # === 注册默认过滤规则 ===
        self._setup_default_filters()
        
        print(f"✅ 信息流管理器初始化完成: {flow_id}")
    
    def _setup_default_filters(self):
        """设置默认过滤规则"""
        # 上行过滤规则
        self.upward_filter.add_rule(
            "size_limit",
            lambda p: len(str(p.data)) > self.flow_config['max_packet_size'],
            "block"
        )
        
        self.upward_filter.add_rule(
            "priority_filter",
            lambda p: p.priority == Priority.LOW and len(str(p.data)) > 1024,
            "block"
        )
        
        # 下行过滤规则
        self.downward_filter.add_rule(
            "size_limit",
            lambda p: len(str(p.data)) > self.flow_config['max_packet_size'],
            "block"
        )
        
        self.downward_filter.add_rule(
            "stale_data",
            lambda p: time.time() - p.timestamp > 60.0,  # 60秒过期
            "block"
        )
    
    def send_constraint_matrix(self, 
                             constraint_matrix: torch.Tensor,
                             target_layer: str,
                             priority: Priority = Priority.HIGH) -> bool:
        """
        发送约束矩阵（上层→下层）
        
        Args:
            constraint_matrix: 约束矩阵 C_t
            target_layer: 目标层级
            priority: 优先级
        """
        try:
            # 创建信息包
            packet = InformationPacket(
                packet_id=f"constraint_{int(time.time()*1000)}",
                info_type=InformationType.CONSTRAINT_DATA,
                flow_direction=FlowDirection.DOWNWARD,
                source_layer="upper_layer",
                target_layer=target_layer,
                timestamp=time.time(),
                data={
                    'constraint_matrix': constraint_matrix.detach().cpu().numpy().tolist(),
                    'matrix_shape': list(constraint_matrix.shape),
                    'constraint_type': 'dynamic_operational_constraints',
                    'validity_period': 300.0  # 5分钟有效期
                },
                priority=priority,
                reliability_required=True
            )
            
            return self._send_packet(packet)
            
        except Exception as e:
            print(f"❌ 发送约束矩阵失败: {str(e)}")
            return False
    
    def send_weight_vector(self, 
                          weight_vector: torch.Tensor,
                          target_layer: str,
                          priority: Priority = Priority.HIGH) -> bool:
        """
        发送权重向量（上层→下层）
        
        Args:
            weight_vector: 权重向量 w_t
            target_layer: 目标层级
            priority: 优先级
        """
        try:
            packet = InformationPacket(
                packet_id=f"weights_{int(time.time()*1000)}",
                info_type=InformationType.CONTROL_COMMAND,
                flow_direction=FlowDirection.DOWNWARD,
                source_layer="upper_layer",
                target_layer=target_layer,
                timestamp=time.time(),
                data={
                    'weight_vector': weight_vector.detach().cpu().numpy().tolist(),
                    'vector_length': len(weight_vector),
                    'weight_type': 'multi_objective_weights',
                    'normalization': 'sum_to_one'
                },
                priority=priority,
                reliability_required=True
            )
            
            return self._send_packet(packet)
            
        except Exception as e:
            print(f"❌ 发送权重向量失败: {str(e)}")
            return False
    
    def send_balance_targets(self, 
                           targets: Dict[str, float],
                           target_layer: str,
                           priority: Priority = Priority.NORMAL) -> bool:
        """
        发送均衡目标（上层→下层）
        
        Args:
            targets: 均衡目标字典
            target_layer: 目标层级
            priority: 优先级
        """
        try:
            packet = InformationPacket(
                packet_id=f"targets_{int(time.time()*1000)}",
                info_type=InformationType.CONTROL_COMMAND,
                flow_direction=FlowDirection.DOWNWARD,
                source_layer="upper_layer",
                target_layer=target_layer,
                timestamp=time.time(),
                data={
                    'balance_targets': targets,
                    'target_type': 'dynamic_balance_objectives',
                    'update_frequency': '5_minutes'
                },
                priority=priority
            )
            
            return self._send_packet(packet)
            
        except Exception as e:
            print(f"❌ 发送均衡目标失败: {str(e)}")
            return False
    
    def send_performance_feedback(self, 
                                performance_data: Dict[str, Any],
                                target_layer: str,
                                priority: Priority = Priority.NORMAL) -> bool:
        """
        发送性能反馈（下层→上层）
        
        Args:
            performance_data: 性能数据
            target_layer: 目标层级
            priority: 优先级
        """
        try:
            packet = InformationPacket(
                packet_id=f"perf_{int(time.time()*1000)}",
                info_type=InformationType.PERFORMANCE_DATA,
                flow_direction=FlowDirection.UPWARD,
                source_layer="lower_layer",
                target_layer=target_layer,
                timestamp=time.time(),
                data={
                    'performance_metrics': performance_data,
                    'measurement_time': time.time(),
                    'metrics_type': 'real_time_performance'
                },
                priority=priority
            )
            
            return self._send_packet(packet)
            
        except Exception as e:
            print(f"❌ 发送性能反馈失败: {str(e)}")
            return False
    
    def send_system_state(self, 
                         state_data: Dict[str, Any],
                         target_layer: str,
                         priority: Priority = Priority.NORMAL) -> bool:
        """
        发送系统状态（下层→上层）
        
        Args:
            state_data: 状态数据
            target_layer: 目标层级
            priority: 优先级
        """
        try:
            packet = InformationPacket(
                packet_id=f"state_{int(time.time()*1000)}",
                info_type=InformationType.STATE_FEEDBACK,
                flow_direction=FlowDirection.UPWARD,
                source_layer="lower_layer",
                target_layer=target_layer,
                timestamp=time.time(),
                data={
                    'system_state': state_data,
                    'state_timestamp': time.time(),
                    'state_type': 'real_time_system_status'
                },
                priority=priority
            )
            
            return self._send_packet(packet)
            
        except Exception as e:
            print(f"❌ 发送系统状态失败: {str(e)}")
            return False
    
    def send_alarm(self, 
                  alarm_data: Dict[str, Any],
                  target_layer: str,
                  priority: Priority = Priority.URGENT) -> bool:
        """
        发送告警信息（下层→上层）
        
        Args:
            alarm_data: 告警数据
            target_layer: 目标层级
            priority: 优先级
        """
        try:
            packet = InformationPacket(
                packet_id=f"alarm_{int(time.time()*1000)}",
                info_type=InformationType.ALARM_DATA,
                flow_direction=FlowDirection.UPWARD,
                source_layer="lower_layer",
                target_layer=target_layer,
                timestamp=time.time(),
                data={
                    'alarm_info': alarm_data,
                    'alarm_timestamp': time.time(),
                    'severity': alarm_data.get('severity', 'medium')
                },
                priority=priority,
                reliability_required=True
            )
            
            return self._send_packet(packet)
            
        except Exception as e:
            print(f"❌ 发送告警信息失败: {str(e)}")
            return False
    
    def _send_packet(self, packet: InformationPacket) -> bool:
        """发送信息包"""
        try:
            # QoS检查
            if not self.qos_manager.check_bandwidth(packet):
                self.flow_statistics['dropped_packets'] += 1
                print(f"⚠️ 包被QoS限制丢弃: {packet.packet_id}")
                return False
            
            # 过滤检查
            if packet.flow_direction == FlowDirection.UPWARD:
                passed, modified_packet = self.upward_filter.filter_packet(packet)
                target_buffer = self.upward_buffer
            else:
                passed, modified_packet = self.downward_filter.filter_packet(packet)
                target_buffer = self.downward_buffer
            
            if not passed:
                self.flow_statistics['dropped_packets'] += 1
                print(f"⚠️ 包被过滤器丢弃: {packet.packet_id}")
                return False
            
            packet = modified_packet or packet
            
            # 数据压缩
            if self.flow_config['enable_compression']:
                packet = self._compress_packet(packet)
            
            # 放入缓冲区
            try:
                target_buffer.put(packet, block=False)
                
                # 更新统计
                if packet.flow_direction == FlowDirection.UPWARD:
                    self.flow_statistics['upward_packets'] += 1
                else:
                    self.flow_statistics['downward_packets'] += 1
                
                self.flow_statistics['total_bytes'] += len(str(packet.data))
                
                return True
                
            except queue.Full:
                self.flow_statistics['dropped_packets'] += 1
                print(f"⚠️ 缓冲区满，包被丢弃: {packet.packet_id}")
                return False
            
        except Exception as e:
            self.flow_statistics['error_packets'] += 1
            print(f"❌ 发送包失败: {str(e)}")
            return False
    
    def receive_packet(self, 
                      flow_direction: FlowDirection,
                      timeout: float = 1.0) -> Optional[InformationPacket]:
        """
        接收信息包
        
        Args:
            flow_direction: 流向
            timeout: 超时时间
            
        Returns:
            接收到的包或None
        """
        try:
            if flow_direction == FlowDirection.UPWARD:
                source_buffer = self.upward_buffer
            else:
                source_buffer = self.downward_buffer
            
            packet = source_buffer.get(block=True, timeout=timeout)
            
            # 数据解压缩
            if self.flow_config['enable_compression']:
                packet = self._decompress_packet(packet)
            
            return packet
            
        except queue.Empty:
            return None
        except Exception as e:
            print(f"❌ 接收包失败: {str(e)}")
            return None
    
    def _compress_packet(self, packet: InformationPacket) -> InformationPacket:
        """压缩数据包（简化实现）"""
        try:
            import gzip
            import json
            
            # 序列化数据
            data_str = json.dumps(packet.data)
            
            # 压缩
            compressed_data = gzip.compress(data_str.encode())
            
            # 计算压缩比
            original_size = len(data_str)
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # 更新压缩比统计
            self.flow_statistics['compression_ratio'] = (
                self.flow_statistics['compression_ratio'] * 0.9 + compression_ratio * 0.1
            )
            
            # 创建压缩包
            compressed_packet = InformationPacket(
                packet_id=packet.packet_id,
                info_type=packet.info_type,
                flow_direction=packet.flow_direction,
                source_layer=packet.source_layer,
                target_layer=packet.target_layer,
                timestamp=packet.timestamp,
                data={
                    'compressed': True,
                    'original_size': original_size,
                    'compressed_data': compressed_data.hex(),
                    'compression_algorithm': 'gzip'
                },
                priority=packet.priority,
                reliability_required=packet.reliability_required,
                sequence_number=packet.sequence_number
            )
            
            return compressed_packet
            
        except Exception as e:
            print(f"⚠️ 数据压缩失败: {str(e)}")
            return packet
    
    def _decompress_packet(self, packet: InformationPacket) -> InformationPacket:
        """解压缩数据包"""
        try:
            if not packet.data.get('compressed', False):
                return packet
            
            import gzip
            import json
            
            # 解压缩
            compressed_data = bytes.fromhex(packet.data['compressed_data'])
            decompressed_str = gzip.decompress(compressed_data).decode()
            
            # 反序列化
            original_data = json.loads(decompressed_str)
            
            # 创建解压缩包
            decompressed_packet = InformationPacket(
                packet_id=packet.packet_id,
                info_type=packet.info_type,
                flow_direction=packet.flow_direction,
                source_layer=packet.source_layer,
                target_layer=packet.target_layer,
                timestamp=packet.timestamp,
                data=original_data,
                priority=packet.priority,
                reliability_required=packet.reliability_required,
                sequence_number=packet.sequence_number
            )
            
            return decompressed_packet
            
        except Exception as e:
            print(f"⚠️ 数据解压缩失败: {str(e)}")
            return packet
    
    def start_flow_processing(self):
        """启动信息流处理"""
        if self.running:
            return
        
        self.running = True
        
        # 启动上行处理线程
        upward_thread = threading.Thread(
            target=self._process_upward_flow,
            name=f"UpwardFlow_{self.flow_id}",
            daemon=True
        )
        upward_thread.start()
        self.flow_threads.append(upward_thread)
        
        # 启动下行处理线程
        downward_thread = threading.Thread(
            target=self._process_downward_flow,
            name=f"DownwardFlow_{self.flow_id}",
            daemon=True
        )
        downward_thread.start()
        self.flow_threads.append(downward_thread)
        
        # 启动同步线程
        if self.synchronization_enabled:
            sync_thread = threading.Thread(
                target=self._synchronization_worker,
                name=f"SyncWorker_{self.flow_id}",
                daemon=True
            )
            sync_thread.start()
            self.flow_threads.append(sync_thread)
        
        print(f"🚀 信息流处理已启动: {self.flow_id}")
    
    def _process_upward_flow(self):
        """处理上行信息流"""
        while self.running:
            try:
                packet = self.receive_packet(FlowDirection.UPWARD, timeout=1.0)
                if packet:
                    # 转换为消息协议格式并发送
                    self._forward_to_message_protocol(packet)
                    
            except Exception as e:
                print(f"❌ 上行流处理错误: {str(e)}")
                time.sleep(0.1)
    
    def _process_downward_flow(self):
        """处理下行信息流"""
        while self.running:
            try:
                packet = self.receive_packet(FlowDirection.DOWNWARD, timeout=1.0)
                if packet:
                    # 转换为消息协议格式并发送
                    self._forward_to_message_protocol(packet)
                    
            except Exception as e:
                print(f"❌ 下行流处理错误: {str(e)}")
                time.sleep(0.1)
    
    def _synchronization_worker(self):
        """同步工作线程"""
        while self.running:
            try:
                current_time = time.time()
                
                if current_time - self.last_sync_time >= self.sync_interval:
                    self._perform_synchronization()
                    self.last_sync_time = current_time
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"❌ 同步工作错误: {str(e)}")
                time.sleep(1.0)
    
    def _forward_to_message_protocol(self, packet: InformationPacket):
        """转发到消息协议"""
        try:
            # 根据信息类型转换为消息类型
            message_type_mapping = {
                InformationType.CONTROL_COMMAND: MessageType.POWER_COMMAND,
                InformationType.STATE_FEEDBACK: MessageType.STATUS_UPDATE,
                InformationType.CONSTRAINT_DATA: MessageType.CONSTRAINT_UPDATE,
                InformationType.PERFORMANCE_DATA: MessageType.PERFORMANCE_REPORT,
                InformationType.ALARM_DATA: MessageType.ALARM_NOTIFICATION,
                InformationType.DIAGNOSTIC_DATA: MessageType.STATUS_UPDATE
            }
            
            message_type = message_type_mapping.get(packet.info_type, MessageType.STATUS_UPDATE)
            
            # 发送消息
            success = self.message_protocol.send_message(
                message_type=message_type,
                payload=packet.data,
                receiver_id=packet.target_layer,
                priority=packet.priority
            )
            
            if not success:
                print(f"⚠️ 转发消息失败: {packet.packet_id}")
                
        except Exception as e:
            print(f"❌ 转发到消息协议失败: {str(e)}")
    
    def _perform_synchronization(self):
        """执行同步操作"""
        try:
            # 发送同步请求
            sync_data = {
                'sync_type': 'periodic_sync',
                'flow_statistics': self.flow_statistics.copy(),
                'qos_status': self.qos_manager.get_qos_status(),
                'buffer_status': {
                    'upward_size': self.upward_buffer.qsize(),
                    'downward_size': self.downward_buffer.qsize()
                }
            }
            
            # 这里可以添加具体的同步逻辑
            print(f"🔄 执行周期性同步: {self.flow_id}")
            
        except Exception as e:
            print(f"❌ 同步操作失败: {str(e)}")
    
    def stop_flow_processing(self):
        """停止信息流处理"""
        self.running = False
        
        # 等待所有线程结束
        for thread in self.flow_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        self.flow_threads.clear()
        print(f"⏹️ 信息流处理已停止: {self.flow_id}")
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """获取信息流统计"""
        qos_status = self.qos_manager.get_qos_status()
        
        stats = {
            'flow_id': self.flow_id,
            'running': self.running,
            'flow_statistics': self.flow_statistics.copy(),
            'qos_status': qos_status,
            
            'buffer_status': {
                'upward_buffer_size': self.upward_buffer.qsize(),
                'downward_buffer_size': self.downward_buffer.qsize(),
                'total_buffer_usage': (self.upward_buffer.qsize() + self.downward_buffer.qsize()) / (2 * self.flow_config['buffer_size'])
            },
            
            'filter_statistics': {
                'upward_filter': self.upward_filter.statistics.copy(),
                'downward_filter': self.downward_filter.statistics.copy()
            },
            
            'configuration': self.flow_config.copy(),
            'active_threads': len([t for t in self.flow_threads if t.is_alive()])
        }
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"InformationFlow({self.flow_id}): "
                f"upward={self.flow_statistics['upward_packets']}, "
                f"downward={self.flow_statistics['downward_packets']}, "
                f"running={self.running}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"InformationFlow(flow_id='{self.flow_id}', "
                f"running={self.running}, "
                f"total_packets={self.flow_statistics['upward_packets'] + self.flow_statistics['downward_packets']})")
