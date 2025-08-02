import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import queue
from abc import ABC, abstractmethod
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.system_config import SystemConfig

class TimeScale(Enum):
    """时间尺度枚举"""
    UPPER_LAYER = "upper_layer"      # 上层 - 5分钟级
    LOWER_LAYER = "lower_layer"      # 下层 - 10ms级
    SIMULATION = "simulation"        # 仿真 - 1s级

class SchedulerMode(Enum):
    """调度器模式枚举"""
    SEQUENTIAL = "sequential"        # 顺序执行
    PARALLEL = "parallel"           # 并行执行
    REAL_TIME = "real_time"         # 实时调度

@dataclass
class TaskInfo:
    """任务信息数据结构"""
    task_id: str
    time_scale: TimeScale
    interval: float                  # s, 执行间隔
    priority: int = 1               # 优先级 (1-10, 10最高)
    callback: Optional[Callable] = None
    last_execution: float = 0.0
    next_execution: float = 0.0
    execution_count: int = 0
    total_execution_time: float = 0.0
    enabled: bool = True

class MultiScaleScheduler:
    """
    多时间尺度调度器
    协调上层DRL(5分钟)、下层DRL(10ms)和仿真环境(1s)的执行节拍
    """
    
    def __init__(self, 
                 system_config: SystemConfig,
                 mode: SchedulerMode = SchedulerMode.SEQUENTIAL,
                 scheduler_id: str = "MultiScaleScheduler_001"):
        """
        初始化多时间尺度调度器
        
        Args:
            system_config: 系统配置
            mode: 调度模式
            scheduler_id: 调度器ID
        """
        self.system_config = system_config
        self.mode = mode
        self.scheduler_id = scheduler_id
        
        # === 时间尺度配置 ===
        self.time_scales = {
            TimeScale.UPPER_LAYER: system_config.UPPER_LAYER_INTERVAL,    # 300s (5分钟)
            TimeScale.LOWER_LAYER: system_config.LOWER_LAYER_INTERVAL,    # 0.01s (10ms)
            TimeScale.SIMULATION: system_config.SIMULATION_TIME_STEP       # 1s
        }
        
        # === 任务管理 ===
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue = queue.PriorityQueue()
        
        # === 执行状态 ===
        self.is_running = False
        self.current_time = 0.0
        self.total_steps = 0
        
        # === 性能统计 ===
        self.execution_stats = {
            TimeScale.UPPER_LAYER: {'count': 0, 'total_time': 0.0, 'avg_time': 0.0},
            TimeScale.LOWER_LAYER: {'count': 0, 'total_time': 0.0, 'avg_time': 0.0},
            TimeScale.SIMULATION: {'count': 0, 'total_time': 0.0, 'avg_time': 0.0}
        }
        
        # === 线程管理 (并行模式) ===
        if mode == SchedulerMode.PARALLEL:
            self.thread_pool = {}
            self.thread_locks = {
                TimeScale.UPPER_LAYER: threading.Lock(),
                TimeScale.LOWER_LAYER: threading.Lock(),
                TimeScale.SIMULATION: threading.Lock()
            }
        
        # === 同步机制 ===
        self.sync_barriers = {}
        self.data_exchange_buffers = {
            'upper_to_lower': queue.Queue(maxsize=10),
            'lower_to_upper': queue.Queue(maxsize=100),
            'simulation_data': queue.Queue(maxsize=1000)
        }
        
        print(f"✅ 多时间尺度调度器初始化完成: {scheduler_id}")
        print(f"   模式: {mode.value}")
        print(f"   时间尺度: 上层{self.time_scales[TimeScale.UPPER_LAYER]}s, "
              f"下层{self.time_scales[TimeScale.LOWER_LAYER]}s, "
              f"仿真{self.time_scales[TimeScale.SIMULATION]}s")
    
    def register_task(self, 
                     task_id: str,
                     time_scale: TimeScale,
                     callback: Callable,
                     priority: int = 1,
                     enabled: bool = True) -> bool:
        """
        注册调度任务
        
        Args:
            task_id: 任务ID
            time_scale: 时间尺度
            callback: 回调函数
            priority: 优先级
            enabled: 是否启用
            
        Returns:
            注册成功标志
        """
        if task_id in self.tasks:
            print(f"⚠️ 任务 {task_id} 已存在，将被覆盖")
        
        interval = self.time_scales[time_scale]
        
        task_info = TaskInfo(
            task_id=task_id,
            time_scale=time_scale,
            interval=interval,
            priority=priority,
            callback=callback,
            next_execution=self.current_time + interval,
            enabled=enabled
        )
        
        self.tasks[task_id] = task_info
        
        print(f"✅ 已注册任务: {task_id} ({time_scale.value}, {interval}s间隔)")
        return True
    
    def unregister_task(self, task_id: str) -> bool:
        """注销任务"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            print(f"🗑️ 已注销任务: {task_id}")
            return True
        else:
            print(f"⚠️ 任务 {task_id} 不存在")
            return False
    
    def enable_task(self, task_id: str) -> bool:
        """启用任务"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True
            return True
        return False
    
    def disable_task(self, task_id: str) -> bool:
        """禁用任务"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False
            return True
        return False
    
    def start(self) -> bool:
        """启动调度器"""
        if self.is_running:
            print("⚠️ 调度器已在运行")
            return False
        
        self.is_running = True
        print(f"🚀 调度器启动: {self.scheduler_id}")
        
        if self.mode == SchedulerMode.PARALLEL:
            return self._start_parallel_mode()
        else:
            return self._start_sequential_mode()
    
    def stop(self) -> bool:
        """停止调度器"""
        if not self.is_running:
            return True
        
        self.is_running = False
        
        if self.mode == SchedulerMode.PARALLEL:
            self._stop_parallel_mode()
        
        print(f"⏹️ 调度器已停止: {self.scheduler_id}")
        return True
    
    def step(self, delta_t: Optional[float] = None) -> Dict[str, Any]:
        """
        执行一个调度步 (主要用于顺序模式)
        
        Args:
            delta_t: 时间步长 (s)
            
        Returns:
            执行结果字典
        """
        if delta_t is None:
            delta_t = self.time_scales[TimeScale.SIMULATION]
        
        self.current_time += delta_t
        self.total_steps += 1
        
        # 收集需要执行的任务
        tasks_to_execute = []
        
        for task_id, task_info in self.tasks.items():
            if (task_info.enabled and 
                self.current_time >= task_info.next_execution):
                tasks_to_execute.append((task_info.priority, task_id, task_info))
        
        # 按优先级排序
        tasks_to_execute.sort(key=lambda x: x[0], reverse=True)
        
        # 执行任务
        execution_results = {}
        
        for _, task_id, task_info in tasks_to_execute:
            start_time = time.time()
            
            try:
                # 执行任务回调
                if task_info.callback:
                    result = task_info.callback(self.current_time, delta_t)
                    execution_results[task_id] = {
                        'success': True,
                        'result': result,
                        'execution_time': time.time() - start_time
                    }
                
                # 更新任务信息
                task_info.last_execution = self.current_time
                task_info.next_execution = self.current_time + task_info.interval
                task_info.execution_count += 1
                task_info.total_execution_time += time.time() - start_time
                
                # 更新统计
                self.execution_stats[task_info.time_scale]['count'] += 1
                self.execution_stats[task_info.time_scale]['total_time'] += time.time() - start_time
                
            except Exception as e:
                execution_results[task_id] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
                print(f"❌ 任务执行失败: {task_id}, 错误: {str(e)}")
        
        # 更新平均执行时间
        for time_scale in self.execution_stats:
            stats = self.execution_stats[time_scale]
            if stats['count'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['count']
        
        return {
            'current_time': self.current_time,
            'executed_tasks': list(execution_results.keys()),
            'execution_results': execution_results,
            'next_execution_times': {
                task_id: task_info.next_execution 
                for task_id, task_info in self.tasks.items()
            }
        }
    
    def _start_sequential_mode(self) -> bool:
        """启动顺序执行模式"""
        print("📋 启动顺序执行模式")
        # 顺序模式通过外部调用step()方法驱动
        return True
    
    def _start_parallel_mode(self) -> bool:
        """启动并行执行模式"""
        print("🔀 启动并行执行模式")
        
        # 为每个时间尺度创建执行线程
        for time_scale in [TimeScale.UPPER_LAYER, TimeScale.LOWER_LAYER, TimeScale.SIMULATION]:
            thread = threading.Thread(
                target=self._parallel_executor,
                args=(time_scale,),
                name=f"Scheduler_{time_scale.value}"
            )
            thread.daemon = True
            thread.start()
            self.thread_pool[time_scale] = thread
        
        return True
    
    def _stop_parallel_mode(self):
        """停止并行执行模式"""
        print("⏸️ 停止并行执行模式")
        
        # 等待所有线程结束
        for time_scale, thread in self.thread_pool.items():
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.thread_pool.clear()
    
    def _parallel_executor(self, time_scale: TimeScale):
        """并行执行器 (线程函数)"""
        interval = self.time_scales[time_scale]
        
        while self.is_running:
            start_time = time.time()
            
            # 执行该时间尺度的所有任务
            for task_id, task_info in self.tasks.items():
                if (task_info.time_scale == time_scale and 
                    task_info.enabled and 
                    task_info.callback):
                    
                    with self.thread_locks[time_scale]:
                        try:
                            result = task_info.callback(self.current_time, interval)
                            
                            # 更新任务统计
                            task_info.execution_count += 1
                            task_info.last_execution = self.current_time
                            
                        except Exception as e:
                            print(f"❌ 并行任务执行失败: {task_id}, 错误: {str(e)}")
            
            # 精确控制执行间隔
            execution_time = time.time() - start_time
            sleep_time = max(0, interval - execution_time)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif execution_time > interval * 1.1:  # 超时10%以上
                print(f"⚠️ {time_scale.value} 任务执行超时: {execution_time:.3f}s > {interval:.3f}s")
    
    def put_data(self, channel: str, data: Any) -> bool:
        """
        向数据交换缓冲区放入数据
        
        Args:
            channel: 通道名称
            data: 数据
            
        Returns:
            成功标志
        """
        if channel not in self.data_exchange_buffers:
            return False
        
        try:
            self.data_exchange_buffers[channel].put_nowait(data)
            return True
        except queue.Full:
            # 缓冲区满，移除最老的数据
            try:
                self.data_exchange_buffers[channel].get_nowait()
                self.data_exchange_buffers[channel].put_nowait(data)
                return True
            except queue.Empty:
                return False
    
    def get_data(self, channel: str, timeout: float = 0.0) -> Optional[Any]:
        """
        从数据交换缓冲区获取数据
        
        Args:
            channel: 通道名称
            timeout: 超时时间 (s)
            
        Returns:
            数据或None
        """
        if channel not in self.data_exchange_buffers:
            return None
        
        try:
            if timeout > 0:
                return self.data_exchange_buffers[channel].get(timeout=timeout)
            else:
                return self.data_exchange_buffers[channel].get_nowait()
        except (queue.Empty, queue.Queue.timeout):
            return None
    
    def get_scheduling_status(self) -> Dict[str, Any]:
        """获取调度状态"""
        return {
            'scheduler_id': self.scheduler_id,
            'mode': self.mode.value,
            'is_running': self.is_running,
            'current_time': self.current_time,
            'total_steps': self.total_steps,
            
            'time_scales': {
                scale.value: interval for scale, interval in self.time_scales.items()
            },
            
            'task_count': len(self.tasks),
            'enabled_tasks': sum(1 for task in self.tasks.values() if task.enabled),
            
            'execution_stats': {
                scale.value: stats for scale, stats in self.execution_stats.items()
            },
            
            'buffer_status': {
                channel: {
                    'size': buffer.qsize(),
                    'maxsize': buffer.maxsize
                } for channel, buffer in self.data_exchange_buffers.items()
            }
        }
    
    def get_task_diagnostics(self) -> Dict[str, Dict]:
        """获取任务诊断信息"""
        diagnostics = {}
        
        for task_id, task_info in self.tasks.items():
            avg_execution_time = (task_info.total_execution_time / task_info.execution_count 
                                if task_info.execution_count > 0 else 0.0)
            
            diagnostics[task_id] = {
                'time_scale': task_info.time_scale.value,
                'interval': task_info.interval,
                'priority': task_info.priority,
                'enabled': task_info.enabled,
                'execution_count': task_info.execution_count,
                'avg_execution_time': avg_execution_time,
                'last_execution': task_info.last_execution,
                'next_execution': task_info.next_execution,
                'utilization': avg_execution_time / task_info.interval if task_info.interval > 0 else 0.0
            }
        
        return diagnostics
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"MultiScaleScheduler({self.scheduler_id}): "
                f"模式={self.mode.value}, "
                f"运行={self.is_running}, "
                f"任务数={len(self.tasks)}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"MultiScaleScheduler(scheduler_id='{self.scheduler_id}', "
                f"mode={self.mode.value}, "
                f"tasks={len(self.tasks)}, "
                f"time={self.current_time:.2f}s)")
