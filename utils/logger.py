import logging
import os
import sys
import time
import json
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import threading
from datetime import datetime
import traceback

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogEntry:
    """日志条目"""
    timestamp: float
    level: LogLevel
    logger_name: str
    message: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    exception_info: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

class Logger:
    """
    统一日志系统
    提供多级别、多输出的日志功能
    """
    
    _instances = {}
    _lock = threading.Lock()
    
    def __new__(cls, name: str = "DRL_Logger"):
        """单例模式确保同名logger唯一"""
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = super().__new__(cls)
            return cls._instances[name]
    
    def __init__(self, name: str = "DRL_Logger"):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
        """
        if hasattr(self, '_initialized'):
            return
            
        self.name = name
        self.log_entries: List[LogEntry] = []
        self.handlers = {}
        self.filters = []
        self.min_level = LogLevel.INFO
        
        # 创建基础logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # 防止重复添加handler
        if not self.logger.handlers:
            self._setup_default_handlers()
        
        # 实验和性能统计
        self.stats = {
            'total_logs': 0,
            'logs_by_level': {level: 0 for level in LogLevel},
            'errors_count': 0,
            'start_time': time.time()
        }
        
        self._initialized = True
        print(f"✅ 日志系统初始化完成: {name}")
    
    def _setup_default_handlers(self):
        """设置默认处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # 文件处理器
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"),
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.handlers['console'] = console_handler
        self.handlers['file'] = file_handler
    
    def add_file_handler(self, file_path: str, level: LogLevel = LogLevel.DEBUG):
        """添加文件处理器"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        file_handler = logging.FileHandler(file_path, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.value))
        
        self.logger.addHandler(file_handler)
        self.handlers[f'file_{len(self.handlers)}'] = file_handler
    
    def add_json_handler(self, file_path: str):
        """添加JSON格式处理器"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': record.created,
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                if record.exc_info:
                    log_data['exception'] = self.formatException(record.exc_info)
                return json.dumps(log_data, ensure_ascii=False)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        json_handler = logging.FileHandler(file_path, encoding='utf-8')
        json_handler.setFormatter(JSONFormatter())
        
        self.logger.addHandler(json_handler)
        self.handlers['json'] = json_handler
    
    def set_level(self, level: LogLevel):
        """设置日志级别"""
        self.min_level = level
        self.logger.setLevel(getattr(logging, level.value))
        
        # 更新所有处理器级别
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, level.value))
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """异常日志（自动包含异常堆栈）"""
        kwargs['exc_info'] = True
        self._log(LogLevel.ERROR, message, **kwargs)
    
    def _log(self, level: LogLevel, message: str, **kwargs):
        """内部日志方法"""
        # 获取调用栈信息
        frame = sys._getframe(2)
        module_name = frame.f_globals.get('__name__', 'unknown')
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # 创建日志条目
        log_entry = LogEntry(
            timestamp=time.time(),
            level=level,
            logger_name=self.name,
            message=message,
            module=module_name,
            function=function_name,
            line_number=line_number,
            extra_data=kwargs.get('extra', {})
        )
        
        # 处理异常信息
        if kwargs.get('exc_info'):
            log_entry.exception_info = traceback.format_exc()
        
        # 存储日志条目
        self.log_entries.append(log_entry)
        
        # 更新统计
        self.stats['total_logs'] += 1
        self.stats['logs_by_level'][level] += 1
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self.stats['errors_count'] += 1
        
        # 发送到标准logging
        log_level = getattr(logging, level.value)
        extra_info = kwargs.get('extra', {})
        
        self.logger.log(
            log_level, 
            message, 
            extra=extra_info, 
            exc_info=kwargs.get('exc_info', False)
        )
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """记录实验开始"""
        self.info(f"🚀 实验开始: {experiment_name}", 
                 extra={'experiment_config': config, 'experiment_phase': 'start'})
    
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any]):
        """记录实验结束"""
        self.info(f"✅ 实验完成: {experiment_name}", 
                 extra={'experiment_results': results, 'experiment_phase': 'end'})
    
    def log_training_progress(self, episode: int, metrics: Dict[str, float]):
        """记录训练进度"""
        self.info(f"训练进度 - 回合 {episode}: {metrics}", 
                 extra={'training_metrics': metrics, 'episode': episode})
    
    def log_performance_metrics(self, metrics: Dict[str, float], context: str = ""):
        """记录性能指标"""
        self.info(f"性能指标 {context}: {metrics}", 
                 extra={'performance_metrics': metrics, 'context': context})
    
    def log_system_resource(self, cpu_usage: float, memory_usage: float, gpu_usage: Optional[float] = None):
        """记录系统资源使用"""
        resource_info = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        }
        if gpu_usage is not None:
            resource_info['gpu_usage'] = gpu_usage
        
        self.debug("系统资源使用情况", extra={'resource_usage': resource_info})
    
    def log_model_checkpoint(self, checkpoint_path: str, episode: int, performance: float):
        """记录模型检查点"""
        self.info(f"模型检查点已保存: {checkpoint_path}", 
                 extra={
                     'checkpoint_info': {
                         'path': checkpoint_path,
                         'episode': episode,
                         'performance': performance,
                         'timestamp': time.time()
                     }
                 })
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """记录带上下文的错误"""
        self.error(f"错误发生: {str(error)}", 
                  extra={'error_context': context, 'error_type': type(error).__name__})
    
    def get_recent_logs(self, count: int = 100, level: Optional[LogLevel] = None) -> List[LogEntry]:
        """获取最近的日志"""
        logs = self.log_entries[-count:] if count > 0 else self.log_entries
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        return logs
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        error_logs = [log for log in self.log_entries 
                     if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        
        error_summary = {
            'total_errors': len(error_logs),
            'error_rate': len(error_logs) / max(len(self.log_entries), 1),
            'recent_errors': error_logs[-10:],  # 最近10个错误
            'error_types': {},
            'error_timeline': []
        }
        
        # 统计错误类型
        for log in error_logs:
            error_type = log.extra_data.get('error_type', 'Unknown')
            error_summary['error_types'][error_type] = error_summary['error_types'].get(error_type, 0) + 1
        
        # 错误时间线
        for log in error_logs[-20:]:  # 最近20个错误的时间线
            error_summary['error_timeline'].append({
                'timestamp': log.timestamp,
                'message': log.message,
                'module': log.module,
                'function': log.function
            })
        
        return error_summary
    
    def export_logs(self, file_path: str, format: str = 'json', 
                   start_time: Optional[float] = None, end_time: Optional[float] = None):
        """导出日志"""
        # 筛选时间范围
        logs = self.log_entries
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format.lower() == 'json':
            log_data = []
            for log in logs:
                log_dict = {
                    'timestamp': log.timestamp,
                    'datetime': datetime.fromtimestamp(log.timestamp).isoformat(),
                    'level': log.level.value,
                    'logger': log.logger_name,
                    'message': log.message,
                    'module': log.module,
                    'function': log.function,
                    'line': log.line_number
                }
                
                if log.exception_info:
                    log_dict['exception'] = log.exception_info
                
                if log.extra_data:
                    log_dict['extra'] = log.extra_data
                
                log_data.append(log_dict)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'csv':
            import pandas as pd
            
            log_data = []
            for log in logs:
                log_data.append({
                    'timestamp': log.timestamp,
                    'datetime': datetime.fromtimestamp(log.timestamp).isoformat(),
                    'level': log.level.value,
                    'logger': log.logger_name,
                    'message': log.message,
                    'module': log.module,
                    'function': log.function,
                    'line': log.line_number,
                    'has_exception': bool(log.exception_info),
                    'has_extra': bool(log.extra_data)
                })
            
            df = pd.DataFrame(log_data)
            df.to_csv(file_path, index=False, encoding='utf-8')
        
        elif format.lower() == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                for log in logs:
                    timestamp_str = datetime.fromtimestamp(log.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{timestamp_str}] {log.level.value} - {log.logger_name} - "
                           f"{log.module}:{log.function}:{log.line_number} - {log.message}\n")
                    
                    if log.exception_info:
                        f.write(f"Exception:\n{log.exception_info}\n")
                    
                    if log.extra_data:
                        f.write(f"Extra: {log.extra_data}\n")
                    
                    f.write("\n")
        
        print(f"✅ 日志已导出: {file_path} ({len(logs)} 条记录)")
    
    def clear_logs(self):
        """清空日志历史"""
        self.log_entries.clear()
        self.stats = {
            'total_logs': 0,
            'logs_by_level': {level: 0 for level in LogLevel},
            'errors_count': 0,
            'start_time': time.time()
        }
        
        self.info("日志历史已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        current_time = time.time()
        runtime = current_time - self.stats['start_time']
        
        stats = self.stats.copy()
        stats.update({
            'runtime_seconds': runtime,
            'runtime_hours': runtime / 3600,
            'logs_per_hour': self.stats['total_logs'] / max(runtime / 3600, 0.001),
            'error_rate': self.stats['errors_count'] / max(self.stats['total_logs'], 1),
            'recent_log_count': len(self.get_recent_logs(100)),
            'log_levels_distribution': {
                level.value: count / max(self.stats['total_logs'], 1) 
                for level, count in self.stats['logs_by_level'].items()
            }
        })
        
        return stats
    
    @classmethod
    def get_logger(cls, name: str = "DRL_Logger") -> 'Logger':
        """获取日志器实例（类方法）"""
        return cls(name)
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"Logger({self.name}): {self.stats['total_logs']} logs, {self.stats['errors_count']} errors"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"Logger(name='{self.name}', total_logs={self.stats['total_logs']}, "
                f"handlers={len(self.handlers)}, level={self.min_level.value})")
