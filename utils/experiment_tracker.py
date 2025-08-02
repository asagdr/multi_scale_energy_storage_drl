import json
import os
import time
import uuid
import pickle
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import threading
import sys
import shutil
import hashlib

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class ExperimentStatus(Enum):
    """实验状态枚举"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MetricType(Enum):
    """指标类型枚举"""
    SCALAR = "scalar"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    VIDEO = "video"

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # 模型配置
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    
    # 超参数
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # 系统配置
    random_seed: Optional[int] = None
    device: str = "cpu"
    num_workers: int = 1
    
    # 实验元数据
    author: str = ""
    project: str = ""
    version: str = "1.0"

@dataclass
class MetricEntry:
    """指标条目"""
    name: str
    value: Union[float, int, str, List, Dict]
    metric_type: MetricType
    step: int
    timestamp: float
    episode: Optional[int] = None
    epoch: Optional[int] = None
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class ExperimentRun:
    """实验运行"""
    run_id: str
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    
    # 时间信息
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # 指标数据
    metrics: List[MetricEntry] = field(default_factory=list)
    
    # 模型检查点
    checkpoints: List[str] = field(default_factory=list)
    best_checkpoint: Optional[str] = None
    
    # 日志信息
    logs: List[str] = field(default_factory=list)
    
    # 系统信息
    system_info: Dict[str, Any] = field(default_factory=dict)
    
    # 结果
    final_results: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息
    error_message: Optional[str] = None
    
    # 资源使用
    resource_usage: Dict[str, Any] = field(default_factory=dict)

class ExperimentTracker:
    """
    实验跟踪器
    提供完整的实验管理和追踪功能
    """
    
    def __init__(self, 
                 tracker_id: str = "ExperimentTracker_001",
                 base_dir: str = "experiments"):
        """
        初始化实验跟踪器
        
        Args:
            tracker_id: 跟踪器ID
            base_dir: 基础目录
        """
        self.tracker_id = tracker_id
        self.base_dir = base_dir
        
        # === 创建目录结构 ===
        self._create_directory_structure()
        
        # === 实验存储 ===
        self.experiments: Dict[str, ExperimentRun] = {}
        self.current_run: Optional[ExperimentRun] = None
        
        # === 线程锁 ===
        self._lock = threading.Lock()
        
        # === 自动保存配置 ===
        self.auto_save = True
        self.save_frequency = 100  # 每100个指标保存一次
        self._metric_count = 0
        
        # === 统计信息 ===
        self.stats = {
            'total_experiments': 0,
            'completed_experiments': 0,
            'failed_experiments': 0,
            'total_metrics': 0,
            'total_runtime': 0.0
        }
        
        # === 加载现有实验 ===
        self._load_existing_experiments()
        
        print(f"✅ 实验跟踪器初始化完成: {tracker_id}")
        print(f"   基础目录: {self.base_dir}")
        print(f"   现有实验: {len(self.experiments)} 个")
    
    def _create_directory_structure(self):
        """创建目录结构"""
        directories = [
            self.base_dir,
            os.path.join(self.base_dir, "runs"),
            os.path.join(self.base_dir, "configs"),
            os.path.join(self.base_dir, "metrics"),
            os.path.join(self.base_dir, "checkpoints"),
            os.path.join(self.base_dir, "logs"),
            os.path.join(self.base_dir, "artifacts"),
            os.path.join(self.base_dir, "exports")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        创建新实验
        
        Args:
            config: 实验配置
            
        Returns:
            实验ID
        """
        experiment_id = str(uuid.uuid4())
        
        # 创建实验运行
        run = ExperimentRun(
            run_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            config=config,
            status=ExperimentStatus.CREATED,
            start_time=time.time(),
            system_info=self._get_system_info()
        )
        
        with self._lock:
            self.experiments[experiment_id] = run
            self.current_run = run
            self.stats['total_experiments'] += 1
        
        # 保存实验配置
        self._save_experiment_config(run)
        
        print(f"✅ 实验创建完成: {config.name} ({experiment_id})")
        
        return experiment_id
    
    def start_experiment(self, experiment_id: Optional[str] = None):
        """
        开始实验
        
        Args:
            experiment_id: 实验ID，如果为None则使用当前实验
        """
        if experiment_id is None:
            if self.current_run is None:
                raise ValueError("没有当前实验，请先创建实验")
            run = self.current_run
        else:
            if experiment_id not in self.experiments:
                raise ValueError(f"实验 {experiment_id} 不存在")
            run = self.experiments[experiment_id]
            self.current_run = run
        
        run.status = ExperimentStatus.RUNNING
        run.start_time = time.time()
        
        self._save_experiment_run(run)
        
        print(f"🚀 实验开始: {run.config.name}")
    
    def log_metric(self,
                   name: str,
                   value: Union[float, int, str, List, Dict],
                   step: Optional[int] = None,
                   episode: Optional[int] = None,
                   epoch: Optional[int] = None,
                   metric_type: MetricType = MetricType.SCALAR,
                   tags: Optional[Dict[str, str]] = None):
        """
        记录指标
        
        Args:
            name: 指标名称
            value: 指标值
            step: 步数
            episode: 回合数
            epoch: 轮次
            metric_type: 指标类型
            tags: 标签
        """
        if self.current_run is None:
            raise ValueError("没有当前运行的实验")
        
        if step is None:
            step = len(self.current_run.metrics)
        
        metric_entry = MetricEntry(
            name=name,
            value=value,
            metric_type=metric_type,
            step=step,
            timestamp=time.time(),
            episode=episode,
            epoch=epoch,
            tags=tags or {}
        )
        
        with self._lock:
            self.current_run.metrics.append(metric_entry)
            self.stats['total_metrics'] += 1
            self._metric_count += 1
        
        # 自动保存
        if self.auto_save and self._metric_count % self.save_frequency == 0:
            self._save_experiment_run(self.current_run)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], **kwargs):
        """
        批量记录指标
        
        Args:
            metrics: 指标字典
            **kwargs: 其他参数
        """
        for name, value in metrics.items():
            self.log_metric(name, value, **kwargs)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        记录超参数
        
        Args:
            hyperparams: 超参数字典
        """
        if self.current_run is None:
            raise ValueError("没有当前运行的实验")
        
        self.current_run.config.hyperparameters.update(hyperparams)
        self._save_experiment_config(self.current_run)
        
        print(f"📝 记录超参数: {list(hyperparams.keys())}")
    
    def log_model_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """
        记录模型检查点
        
        Args:
            checkpoint_path: 检查点路径
            is_best: 是否为最佳模型
        """
        if self.current_run is None:
            raise ValueError("没有当前运行的实验")
        
        with self._lock:
            self.current_run.checkpoints.append(checkpoint_path)
            
            if is_best:
                self.current_run.best_checkpoint = checkpoint_path
        
        print(f"💾 记录检查点: {checkpoint_path}" + (" (最佳)" if is_best else ""))
    
    def log_text(self, text: str, name: str = "log"):
        """
        记录文本日志
        
        Args:
            text: 文本内容
            name: 日志名称
        """
        if self.current_run is None:
            raise ValueError("没有当前运行的实验")
        
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {text}"
        
        with self._lock:
            self.current_run.logs.append(log_entry)
        
        # 同时记录为指标
        self.log_metric(name, text, metric_type=MetricType.TEXT)
    
    def log_system_resource(self):
        """记录系统资源使用"""
        if self.current_run is None:
            return
        
        try:
            import psutil
            
            resource_info = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'timestamp': time.time()
            }
            
            # 尝试获取GPU信息
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    resource_info['gpu_utilization'] = gpus[0].load * 100
                    resource_info['gpu_memory_percent'] = gpus[0].memoryUtil * 100
            except:
                pass
            
            with self._lock:
                if 'resource_history' not in self.current_run.resource_usage:
                    self.current_run.resource_usage['resource_history'] = []
                
                self.current_run.resource_usage['resource_history'].append(resource_info)
            
        except ImportError:
            pass  # psutil不可用
    
    def pause_experiment(self, experiment_id: Optional[str] = None):
        """
        暂停实验
        
        Args:
            experiment_id: 实验ID
        """
        run = self._get_run(experiment_id)
        run.status = ExperimentStatus.PAUSED
        
        self._save_experiment_run(run)
        
        print(f"⏸️ 实验已暂停: {run.config.name}")
    
    def resume_experiment(self, experiment_id: Optional[str] = None):
        """
        恢复实验
        
        Args:
            experiment_id: 实验ID
        """
        run = self._get_run(experiment_id)
        
        if run.status != ExperimentStatus.PAUSED:
            raise ValueError(f"实验状态不是暂停状态: {run.status}")
        
        run.status = ExperimentStatus.RUNNING
        self.current_run = run
        
        self._save_experiment_run(run)
        
        print(f"▶️ 实验已恢复: {run.config.name}")
    
    def complete_experiment(self, 
                           experiment_id: Optional[str] = None,
                           final_results: Optional[Dict[str, Any]] = None):
        """
        完成实验
        
        Args:
            experiment_id: 实验ID
            final_results: 最终结果
        """
        run = self._get_run(experiment_id)
        
        run.status = ExperimentStatus.COMPLETED
        run.end_time = time.time()
        run.duration = run.end_time - run.start_time
        
        if final_results:
            run.final_results = final_results
        
        with self._lock:
            self.stats['completed_experiments'] += 1
            self.stats['total_runtime'] += run.duration
        
        self._save_experiment_run(run)
        
        print(f"✅ 实验完成: {run.config.name} (用时: {run.duration:.2f}s)")
        
        # 重置当前实验
        if self.current_run == run:
            self.current_run = None
    
    def fail_experiment(self, 
                       error_message: str,
                       experiment_id: Optional[str] = None):
        """
        标记实验失败
        
        Args:
            error_message: 错误信息
            experiment_id: 实验ID
        """
        run = self._get_run(experiment_id)
        
        run.status = ExperimentStatus.FAILED
        run.end_time = time.time()
        run.duration = run.end_time - run.start_time
        run.error_message = error_message
        
        with self._lock:
            self.stats['failed_experiments'] += 1
        
        self._save_experiment_run(run)
        
        print(f"❌ 实验失败: {run.config.name} - {error_message}")
        
        # 重置当前实验
        if self.current_run == run:
            self.current_run = None
    
    def cancel_experiment(self, experiment_id: Optional[str] = None):
        """
        取消实验
        
        Args:
            experiment_id: 实验ID
        """
        run = self._get_run(experiment_id)
        
        run.status = ExperimentStatus.CANCELLED
        run.end_time = time.time()
        run.duration = run.end_time - run.start_time
        
        self._save_experiment_run(run)
        
        print(f"🚫 实验已取消: {run.config.name}")
        
        # 重置当前实验
        if self.current_run == run:
            self.current_run = None
    
    def get_experiment(self, experiment_id: str) -> ExperimentRun:
        """
        获取实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验运行对象
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"实验 {experiment_id} 不存在")
        
        return self.experiments[experiment_id]
    
    def list_experiments(self,
                        status: Optional[ExperimentStatus] = None,
                        tags: Optional[List[str]] = None,
                        project: Optional[str] = None,
                        author: Optional[str] = None,
                        limit: Optional[int] = None) -> List[ExperimentRun]:
        """
        列出实验
        
        Args:
            status: 状态过滤
            tags: 标签过滤
            project: 项目过滤
            author: 作者过滤
            limit: 数量限制
            
        Returns:
            实验列表
        """
        experiments = list(self.experiments.values())
        
        # 状态过滤
        if status:
            experiments = [exp for exp in experiments if exp.status == status]
        
        # 标签过滤
        if tags:
            experiments = [exp for exp in experiments 
                          if any(tag in exp.config.tags for tag in tags)]
        
        # 项目过滤
        if project:
            experiments = [exp for exp in experiments if exp.config.project == project]
        
        # 作者过滤
        if author:
            experiments = [exp for exp in experiments if exp.config.author == author]
        
        # 按开始时间排序
        experiments.sort(key=lambda x: x.start_time, reverse=True)
        
        # 数量限制
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def get_metrics(self,
                   experiment_id: str,
                   metric_names: Optional[List[str]] = None,
                   steps: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
        """
        获取指标数据
        
        Args:
            experiment_id: 实验ID
            metric_names: 指标名称列表
            steps: 步数范围
            
        Returns:
            指标DataFrame
        """
        run = self.get_experiment(experiment_id)
        
        # 过滤指标
        metrics = run.metrics
        
        if metric_names:
            metrics = [m for m in metrics if m.name in metric_names]
        
        if steps:
            start_step, end_step = steps
            metrics = [m for m in metrics if start_step <= m.step <= end_step]
        
        # 转换为DataFrame
        data = []
        for metric in metrics:
            data.append({
                'name': metric.name,
                'value': metric.value,
                'step': metric.step,
                'timestamp': metric.timestamp,
                'episode': metric.episode,
                'epoch': metric.epoch,
                'metric_type': metric.metric_type.value
            })
        
        return pd.DataFrame(data)
    
    def compare_experiments(self,
                           experiment_ids: List[str],
                           metric_names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        比较实验
        
        Args:
            experiment_ids: 实验ID列表
            metric_names: 指标名称列表
            
        Returns:
            比较结果字典
        """
        comparison = {}
        
        for exp_id in experiment_ids:
            run = self.get_experiment(exp_id)
            metrics_df = self.get_metrics(exp_id, metric_names)
            
            comparison[f"{run.config.name}_{exp_id[:8]}"] = metrics_df
        
        return comparison
    
    def export_experiment(self,
                         experiment_id: str,
                         export_path: str,
                         include_checkpoints: bool = False):
        """
        导出实验
        
        Args:
            experiment_id: 实验ID
            export_path: 导出路径
            include_checkpoints: 是否包含检查点
        """
        run = self.get_experiment(experiment_id)
        
        # 创建导出目录
        os.makedirs(export_path, exist_ok=True)
        
        # 导出实验配置
        config_path = os.path.join(export_path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(run.config), f, indent=2)
        
        # 导出指标数据
        metrics_df = self.get_metrics(experiment_id)
        metrics_path = os.path.join(export_path, "metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        # 导出实验信息
        experiment_info = {
            'run_id': run.run_id,
            'experiment_id': run.experiment_id,
            'status': run.status.value,
            'start_time': run.start_time,
            'end_time': run.end_time,
            'duration': run.duration,
            'final_results': run.final_results,
            'system_info': run.system_info,
            'resource_usage': run.resource_usage,
            'error_message': run.error_message
        }
        
        info_path = os.path.join(export_path, "experiment_info.json")
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2, default=str)
        
        # 导出日志
        logs_path = os.path.join(export_path, "logs.txt")
        with open(logs_path, 'w') as f:
            f.write('\n'.join(run.logs))
        
        # 导出检查点
        if include_checkpoints and run.checkpoints:
            checkpoints_dir = os.path.join(export_path, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            
            for checkpoint_path in run.checkpoints:
                if os.path.exists(checkpoint_path):
                    checkpoint_name = os.path.basename(checkpoint_path)
                    target_path = os.path.join(checkpoints_dir, checkpoint_name)
                    shutil.copy2(checkpoint_path, target_path)
        
        print(f"✅ 实验已导出: {experiment_id} -> {export_path}")
    
    def create_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """
        创建实验报告
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验报告
        """
        run = self.get_experiment(experiment_id)
        metrics_df = self.get_metrics(experiment_id)
        
        # 基本信息
        report = {
            'experiment_info': {
                'name': run.config.name,
                'description': run.config.description,
                'experiment_id': experiment_id,
                'run_id': run.run_id,
                'status': run.status.value,
                'author': run.config.author,
                'project': run.config.project,
                'tags': run.config.tags
            },
            'timing': {
                'start_time': datetime.fromtimestamp(run.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(run.end_time).isoformat() if run.end_time else None,
                'duration_seconds': run.duration,
                'duration_formatted': self._format_duration(run.duration) if run.duration else None
            },
            'configuration': {
                'hyperparameters': run.config.hyperparameters,
                'model_config': run.config.model_config,
                'training_config': run.config.training_config,
                'random_seed': run.config.random_seed,
                'device': run.config.device
            },
            'metrics_summary': {},
            'checkpoints': {
                'total_checkpoints': len(run.checkpoints),
                'best_checkpoint': run.best_checkpoint,
                'checkpoint_list': run.checkpoints
            },
            'system_info': run.system_info,
            'final_results': run.final_results
        }
        
        # 指标摘要
        if not metrics_df.empty:
            metric_names = metrics_df['name'].unique()
            
            for metric_name in metric_names:
                metric_data = metrics_df[metrics_df['name'] == metric_name]['value']
                
                if metric_data.dtype in ['float64', 'int64']:
                    report['metrics_summary'][metric_name] = {
                        'final_value': float(metric_data.iloc[-1]) if len(metric_data) > 0 else None,
                        'max_value': float(metric_data.max()),
                        'min_value': float(metric_data.min()),
                        'mean_value': float(metric_data.mean()),
                        'total_entries': len(metric_data)
                    }
        
        # 错误信息
        if run.error_message:
            report['error_info'] = {
                'error_message': run.error_message,
                'status': 'failed'
            }
        
        return report
    
    def _get_run(self, experiment_id: Optional[str]) -> ExperimentRun:
        """获取实验运行对象"""
        if experiment_id is None:
            if self.current_run is None:
                raise ValueError("没有当前运行的实验")
            return self.current_run
        else:
            if experiment_id not in self.experiments:
                raise ValueError(f"实验 {experiment_id} 不存在")
            return self.experiments[experiment_id]
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'hostname': platform.node()
        }
        
        try:
            import torch
            system_info['pytorch_version'] = torch.__version__
            system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                system_info['cuda_version'] = torch.version.cuda
                system_info['gpu_count'] = torch.cuda.device_count()
                system_info['gpu_names'] = [torch.cuda.get_device_name(i) 
                                          for i in range(torch.cuda.device_count())]
        except ImportError:
            pass
        
        try:
            import psutil
            system_info['cpu_count'] = psutil.cpu_count()
            system_info['memory_total_gb'] = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            pass
        
        return system_info
    
    def _save_experiment_config(self, run: ExperimentRun):
        """保存实验配置"""
        config_path = os.path.join(self.base_dir, "configs", f"{run.experiment_id}_config.json")
        
        with open(config_path, 'w') as f:
            json.dump(asdict(run.config), f, indent=2)
    
    def _save_experiment_run(self, run: ExperimentRun):
        """保存实验运行数据"""
        run_path = os.path.join(self.base_dir, "runs", f"{run.experiment_id}_run.pkl")
        
        with open(run_path, 'wb') as f:
            pickle.dump(run, f)
    
    def _load_existing_experiments(self):
        """加载现有实验"""
        runs_dir = os.path.join(self.base_dir, "runs")
        
        if not os.path.exists(runs_dir):
            return
        
        for filename in os.listdir(runs_dir):
            if filename.endswith('_run.pkl'):
                run_path = os.path.join(runs_dir, filename)
                
                try:
                    with open(run_path, 'rb') as f:
                        run = pickle.load(f)
                    
                    self.experiments[run.experiment_id] = run
                    
                    # 更新统计
                    if run.status == ExperimentStatus.COMPLETED:
                        self.stats['completed_experiments'] += 1
                        if run.duration:
                            self.stats['total_runtime'] += run.duration
                    elif run.status == ExperimentStatus.FAILED:
                        self.stats['failed_experiments'] += 1
                    
                    self.stats['total_metrics'] += len(run.metrics)
                    
                except Exception as e:
                    print(f"⚠️ 加载实验失败 {filename}: {str(e)}")
        
        self.stats['total_experiments'] = len(self.experiments)
    
    def _format_duration(self, duration: float) -> str:
        """格式化持续时间"""
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats['total_experiments'] > 0:
            stats['success_rate'] = stats['completed_experiments'] / stats['total_experiments']
            stats['failure_rate'] = stats['failed_experiments'] / stats['total_experiments']
            stats['avg_runtime'] = stats['total_runtime'] / max(stats['completed_experiments'], 1)
            stats['avg_metrics_per_experiment'] = stats['total_metrics'] / stats['total_experiments']
        else:
            stats['success_rate'] = 0
            stats['failure_rate'] = 0
            stats['avg_runtime'] = 0
            stats['avg_metrics_per_experiment'] = 0
        
        # 状态分布
        status_counts = {}
        for status in ExperimentStatus:
            count = sum(1 for exp in self.experiments.values() if exp.status == status)
            status_counts[status.value] = count
        
        stats['status_distribution'] = status_counts
        
        return stats
    
    def cleanup_experiments(self,
                           keep_completed: int = 50,
                           keep_failed: int = 10,
                           remove_cancelled: bool = True):
        """
        清理实验
        
        Args:
            keep_completed: 保留的已完成实验数量
            keep_failed: 保留的失败实验数量
            remove_cancelled: 是否移除已取消的实验
        """
        to_remove = []
        
        # 按状态分组
        completed_experiments = [exp for exp in self.experiments.values() 
                               if exp.status == ExperimentStatus.COMPLETED]
        failed_experiments = [exp for exp in self.experiments.values() 
                            if exp.status == ExperimentStatus.FAILED]
        cancelled_experiments = [exp for exp in self.experiments.values() 
                               if exp.status == ExperimentStatus.CANCELLED]
        
        # 排序（最新的在前）
        completed_experiments.sort(key=lambda x: x.start_time, reverse=True)
        failed_experiments.sort(key=lambda x: x.start_time, reverse=True)
        
        # 标记要删除的实验
        if len(completed_experiments) > keep_completed:
            to_remove.extend(completed_experiments[keep_completed:])
        
        if len(failed_experiments) > keep_failed:
            to_remove.extend(failed_experiments[keep_failed:])
        
        if remove_cancelled:
            to_remove.extend(cancelled_experiments)
        
        # 删除实验
        removed_count = 0
        for exp in to_remove:
            try:
                self._remove_experiment(exp.experiment_id)
                removed_count += 1
            except Exception as e:
                print(f"⚠️ 删除实验 {exp.experiment_id} 失败: {str(e)}")
        
        print(f"✅ 清理完成: 删除了 {removed_count} 个实验")
    
    def _remove_experiment(self, experiment_id: str):
        """删除实验"""
        if experiment_id not in self.experiments:
            return
        
        # 删除文件
        run_path = os.path.join(self.base_dir, "runs", f"{experiment_id}_run.pkl")
        config_path = os.path.join(self.base_dir, "configs", f"{experiment_id}_config.json")
        
        for path in [run_path, config_path]:
            if os.path.exists(path):
                os.remove(path)
        
        # 从内存中删除
        del self.experiments[experiment_id]
        
        # 重置当前实验（如果是当前实验）
        if self.current_run and self.current_run.experiment_id == experiment_id:
            self.current_run = None
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"ExperimentTracker({self.tracker_id}): "
                f"实验={len(self.experiments)}, "
                f"成功={self.stats['completed_experiments']}, "
                f"失败={self.stats['failed_experiments']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"ExperimentTracker(tracker_id='{self.tracker_id}', "
                f"base_dir='{self.base_dir}', "
                f"experiments={len(self.experiments)})")
