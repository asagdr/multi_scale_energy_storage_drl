import torch
import pickle
import json
import os
import shutil
import time
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import sys
import glob

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class CheckpointType(Enum):
    """检查点类型枚举"""
    MODEL = "model"                    # 模型检查点
    OPTIMIZER = "optimizer"            # 优化器检查点
    TRAINING_STATE = "training_state"  # 训练状态检查点
    FULL_SYSTEM = "full_system"       # 完整系统检查点
    EXPERIMENT = "experiment"          # 实验检查点
    BEST_MODEL = "best_model"         # 最佳模型检查点
    LATEST = "latest"                 # 最新检查点
    BACKUP = "backup"                 # 备份检查点

@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    checkpoint_id: str
    checkpoint_type: CheckpointType
    creation_time: float
    file_path: str
    file_size: int
    
    # 训练相关信息
    episode: Optional[int] = None
    step: Optional[int] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 模型相关信息
    model_architecture: Optional[str] = None
    model_parameters: Optional[int] = None
    
    # 版本信息
    version: str = "1.0"
    framework_version: Optional[str] = None
    
    # 验证信息
    checksum: Optional[str] = None
    is_valid: bool = True
    
    # 自定义标签
    tags: List[str] = field(default_factory=list)
    description: str = ""

@dataclass
class CheckpointConfig:
    """检查点配置"""
    base_dir: str = "checkpoints"
    save_frequency: int = 100          # 保存频率（回合数）
    max_checkpoints: int = 10          # 最大保存数量
    compress: bool = True              # 是否压缩
    verify_integrity: bool = True      # 是否验证完整性
    auto_cleanup: bool = True          # 是否自动清理
    backup_enabled: bool = True        # 是否启用备份
    async_save: bool = True            # 是否异步保存

class CheckpointManager:
    """
    检查点管理器
    提供全面的模型和训练状态保存/加载功能
    """
    
    def __init__(self, 
                 manager_id: str = "CheckpointManager_001",
                 config: Optional[CheckpointConfig] = None):
        """
        初始化检查点管理器
        
        Args:
            manager_id: 管理器ID
            config: 检查点配置
        """
        self.manager_id = manager_id
        self.config = config or CheckpointConfig()
        
        # === 创建目录结构 ===
        self.base_dir = self.config.base_dir
        self._create_directory_structure()
        
        # === 检查点存储 ===
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.checkpoint_history: List[str] = []  # 按时间排序的检查点ID
        
        # === 锁和线程 ===
        self._lock = threading.Lock()
        self._save_thread_pool = []
        
        # === 统计信息 ===
        self.stats = {
            'total_saves': 0,
            'total_loads': 0,
            'total_size_bytes': 0,
            'save_errors': 0,
            'load_errors': 0,
            'cleanup_count': 0
        }
        
        # === 加载现有检查点 ===
        self._load_existing_checkpoints()
        
        print(f"✅ 检查点管理器初始化完成: {manager_id}")
        print(f"   基础目录: {self.base_dir}")
        print(f"   现有检查点: {len(self.checkpoints)} 个")
    
    def _create_directory_structure(self):
        """创建目录结构"""
        directories = [
            self.base_dir,
            os.path.join(self.base_dir, "models"),
            os.path.join(self.base_dir, "optimizers"),
            os.path.join(self.base_dir, "training_states"),
            os.path.join(self.base_dir, "experiments"),
            os.path.join(self.base_dir, "backups"),
            os.path.join(self.base_dir, "metadata")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_checkpoint(self,
                       obj: Any,
                       checkpoint_type: CheckpointType,
                       checkpoint_id: Optional[str] = None,
                       episode: Optional[int] = None,
                       step: Optional[int] = None,
                       performance_metrics: Optional[Dict[str, float]] = None,
                       tags: Optional[List[str]] = None,
                       description: str = "",
                       **kwargs) -> str:
        """
        保存检查点
        
        Args:
            obj: 要保存的对象
            checkpoint_type: 检查点类型
            checkpoint_id: 检查点ID
            episode: 训练回合
            step: 训练步数
            performance_metrics: 性能指标
            tags: 标签
            description: 描述
            
        Returns:
            检查点ID
        """
        if checkpoint_id is None:
            checkpoint_id = self._generate_checkpoint_id(checkpoint_type, episode, step)
        
        try:
            # 确定保存路径
            file_path = self._get_checkpoint_path(checkpoint_id, checkpoint_type)
            
            # 保存对象
            if self.config.async_save:
                # 异步保存
                thread = threading.Thread(
                    target=self._save_checkpoint_async,
                    args=(obj, file_path, checkpoint_id, checkpoint_type, 
                          episode, step, performance_metrics, tags, description, kwargs)
                )
                thread.start()
                self._save_thread_pool.append(thread)
            else:
                # 同步保存
                self._save_checkpoint_sync(
                    obj, file_path, checkpoint_id, checkpoint_type,
                    episode, step, performance_metrics, tags, description, kwargs
                )
            
            return checkpoint_id
            
        except Exception as e:
            self.stats['save_errors'] += 1
            print(f"❌ 保存检查点失败: {str(e)}")
            raise
    
    def _save_checkpoint_sync(self,
                            obj: Any,
                            file_path: str,
                            checkpoint_id: str,
                            checkpoint_type: CheckpointType,
                            episode: Optional[int],
                            step: Optional[int],
                            performance_metrics: Optional[Dict[str, float]],
                            tags: Optional[List[str]],
                            description: str,
                            kwargs: Dict[str, Any]):
        """同步保存检查点"""
        start_time = time.time()
        
        # 保存对象
        if isinstance(obj, torch.nn.Module) or hasattr(obj, 'state_dict'):
            # PyTorch模型或有state_dict的对象
            if hasattr(obj, 'state_dict'):
                torch.save(obj.state_dict(), file_path)
            else:
                torch.save(obj, file_path)
        elif isinstance(obj, dict):
            # 字典对象
            if checkpoint_type in [CheckpointType.TRAINING_STATE, CheckpointType.FULL_SYSTEM]:
                torch.save(obj, file_path)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(obj, f)
        else:
            # 其他对象
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
        
        # 获取文件信息
        file_size = os.path.getsize(file_path)
        checksum = self._calculate_checksum(file_path) if self.config.verify_integrity else None
        
        # 创建元数据
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            creation_time=time.time(),
            file_path=file_path,
            file_size=file_size,
            episode=episode,
            step=step,
            performance_metrics=performance_metrics or {},
            tags=tags or [],
            description=description,
            checksum=checksum,
            framework_version=torch.__version__
        )
        
        # 如果是模型，添加模型信息
        if isinstance(obj, torch.nn.Module):
            metadata.model_architecture = obj.__class__.__name__
            metadata.model_parameters = sum(p.numel() for p in obj.parameters())
        
        # 保存元数据
        self._save_metadata(metadata)
        
        with self._lock:
            self.checkpoints[checkpoint_id] = metadata
            self.checkpoint_history.append(checkpoint_id)
            
            # 更新统计
            self.stats['total_saves'] += 1
            self.stats['total_size_bytes'] += file_size
        
        # 自动清理
        if self.config.auto_cleanup:
            self._cleanup_old_checkpoints(checkpoint_type)
        
        # 创建备份
        if self.config.backup_enabled:
            self._create_backup(checkpoint_id)
        
        save_time = time.time() - start_time
        print(f"✅ 检查点保存完成: {checkpoint_id} ({file_size/1024/1024:.2f}MB, {save_time:.2f}s)")
    
    def _save_checkpoint_async(self, *args):
        """异步保存检查点"""
        try:
            self._save_checkpoint_sync(*args)
        except Exception as e:
            print(f"❌ 异步保存检查点失败: {str(e)}")
            self.stats['save_errors'] += 1
    
    def load_checkpoint(self,
                       checkpoint_id: str,
                       map_location: Optional[str] = None) -> Any:
        """
        加载检查点
        
        Args:
            checkpoint_id: 检查点ID
            map_location: 设备映射位置
            
        Returns:
            加载的对象
        """
        try:
            if checkpoint_id not in self.checkpoints:
                raise ValueError(f"检查点 {checkpoint_id} 不存在")
            
            metadata = self.checkpoints[checkpoint_id]
            
            # 验证文件完整性
            if self.config.verify_integrity and metadata.checksum:
                if not self._verify_checksum(metadata.file_path, metadata.checksum):
                    raise ValueError(f"检查点 {checkpoint_id} 文件损坏")
            
            # 加载对象
            if metadata.file_path.endswith('.pth') or metadata.file_path.endswith('.pt'):
                # PyTorch格式
                obj = torch.load(metadata.file_path, map_location=map_location)
            else:
                # Pickle格式
                with open(metadata.file_path, 'rb') as f:
                    obj = pickle.load(f)
            
            self.stats['total_loads'] += 1
            print(f"✅ 检查点加载完成: {checkpoint_id}")
            
            return obj
            
        except Exception as e:
            self.stats['load_errors'] += 1
            print(f"❌ 加载检查点失败: {str(e)}")
            raise
    
    def load_latest_checkpoint(self, 
                              checkpoint_type: Optional[CheckpointType] = None) -> Tuple[str, Any]:
        """
        加载最新检查点
        
        Args:
            checkpoint_type: 检查点类型过滤
            
        Returns:
            (检查点ID, 加载的对象)
        """
        # 筛选检查点
        candidates = []
        for checkpoint_id in reversed(self.checkpoint_history):  # 从最新开始
            metadata = self.checkpoints[checkpoint_id]
            if checkpoint_type is None or metadata.checkpoint_type == checkpoint_type:
                candidates.append(checkpoint_id)
        
        if not candidates:
            raise ValueError("没有找到符合条件的检查点")
        
        latest_id = candidates[0]
        obj = self.load_checkpoint(latest_id)
        
        return latest_id, obj
    
    def load_best_checkpoint(self,
                           metric_name: str,
                           maximize: bool = True,
                           checkpoint_type: Optional[CheckpointType] = None) -> Tuple[str, Any]:
        """
        加载最佳检查点
        
        Args:
            metric_name: 指标名称
            maximize: 是否最大化指标
            checkpoint_type: 检查点类型过滤
            
        Returns:
            (检查点ID, 加载的对象)
        """
        # 筛选有指定指标的检查点
        candidates = []
        for checkpoint_id, metadata in self.checkpoints.items():
            if metric_name in metadata.performance_metrics:
                if checkpoint_type is None or metadata.checkpoint_type == checkpoint_type:
                    candidates.append((checkpoint_id, metadata.performance_metrics[metric_name]))
        
        if not candidates:
            raise ValueError(f"没有找到包含指标 {metric_name} 的检查点")
        
        # 找到最佳检查点
        best_id, best_value = max(candidates, key=lambda x: x[1]) if maximize else min(candidates, key=lambda x: x[1])
        obj = self.load_checkpoint(best_id)
        
        print(f"✅ 加载最佳检查点: {best_id} ({metric_name}={best_value})")
        
        return best_id, obj
    
    def delete_checkpoint(self, checkpoint_id: str):
        """删除检查点"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")
        
        metadata = self.checkpoints[checkpoint_id]
        
        try:
            # 删除文件
            if os.path.exists(metadata.file_path):
                os.remove(metadata.file_path)
            
            # 删除元数据文件
            metadata_path = self._get_metadata_path(checkpoint_id)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # 删除备份
            backup_path = self._get_backup_path(checkpoint_id)
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            # 从内存中删除
            with self._lock:
                del self.checkpoints[checkpoint_id]
                if checkpoint_id in self.checkpoint_history:
                    self.checkpoint_history.remove(checkpoint_id)
            
            print(f"✅ 检查点已删除: {checkpoint_id}")
            
        except Exception as e:
            print(f"❌ 删除检查点失败: {str(e)}")
            raise
    
    def list_checkpoints(self,
                        checkpoint_type: Optional[CheckpointType] = None,
                        tags: Optional[List[str]] = None,
                        sort_by: str = "creation_time",
                        reverse: bool = True) -> List[CheckpointMetadata]:
        """
        列出检查点
        
        Args:
            checkpoint_type: 类型过滤
            tags: 标签过滤
            sort_by: 排序字段
            reverse: 是否倒序
            
        Returns:
            检查点元数据列表
        """
        # 筛选检查点
        filtered = []
        for metadata in self.checkpoints.values():
            # 类型过滤
            if checkpoint_type and metadata.checkpoint_type != checkpoint_type:
                continue
            
            # 标签过滤
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            filtered.append(metadata)
        
        # 排序
        if hasattr(CheckpointMetadata, sort_by):
            filtered.sort(key=lambda x: getattr(x, sort_by), reverse=reverse)
        
        return filtered
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Dict[str, Any]:
        """获取检查点详细信息"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")
        
        metadata = self.checkpoints[checkpoint_id]
        
        info = {
            'checkpoint_id': metadata.checkpoint_id,
            'type': metadata.checkpoint_type.value,
            'creation_time': metadata.creation_time,
            'creation_datetime': datetime.fromtimestamp(metadata.creation_time).isoformat(),
            'file_path': metadata.file_path,
            'file_size_mb': metadata.file_size / 1024 / 1024,
            'episode': metadata.episode,
            'step': metadata.step,
            'performance_metrics': metadata.performance_metrics,
            'model_architecture': metadata.model_architecture,
            'model_parameters': metadata.model_parameters,
            'version': metadata.version,
            'framework_version': metadata.framework_version,
            'tags': metadata.tags,
            'description': metadata.description,
            'is_valid': metadata.is_valid,
            'has_backup': os.path.exists(self._get_backup_path(checkpoint_id))
        }
        
        return info
    
    def export_checkpoint(self,
                         checkpoint_id: str,
                         export_path: str,
                         include_metadata: bool = True):
        """导出检查点"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")
        
        metadata = self.checkpoints[checkpoint_id]
        
        # 创建导出目录
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        # 复制检查点文件
        shutil.copy2(metadata.file_path, export_path)
        
        # 导出元数据
        if include_metadata:
            metadata_export_path = export_path.replace('.pth', '_metadata.json').replace('.pkl', '_metadata.json')
            info = self.get_checkpoint_info(checkpoint_id)
            
            with open(metadata_export_path, 'w') as f:
                json.dump(info, f, indent=2)
        
        print(f"✅ 检查点已导出: {checkpoint_id} -> {export_path}")
    
    def import_checkpoint(self,
                         import_path: str,
                         checkpoint_id: Optional[str] = None,
                         checkpoint_type: CheckpointType = CheckpointType.MODEL) -> str:
        """导入检查点"""
        if not os.path.exists(import_path):
            raise ValueError(f"导入文件不存在: {import_path}")
        
        if checkpoint_id is None:
            checkpoint_id = f"imported_{int(time.time()*1000)}"
        
        # 确定目标路径
        target_path = self._get_checkpoint_path(checkpoint_id, checkpoint_type)
        
        # 复制文件
        shutil.copy2(import_path, target_path)
        
        # 创建元数据
        file_size = os.path.getsize(target_path)
        checksum = self._calculate_checksum(target_path) if self.config.verify_integrity else None
        
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            creation_time=time.time(),
            file_path=target_path,
            file_size=file_size,
            checksum=checksum,
            description="Imported checkpoint"
        )
        
        # 保存元数据
        self._save_metadata(metadata)
        
        with self._lock:
            self.checkpoints[checkpoint_id] = metadata
            self.checkpoint_history.append(checkpoint_id)
        
        print(f"✅ 检查点已导入: {checkpoint_id}")
        
        return checkpoint_id
    
    def create_backup(self, checkpoint_id: str) -> str:
        """创建检查点备份"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")
        
        return self._create_backup(checkpoint_id)
    
    def restore_from_backup(self, checkpoint_id: str):
        """从备份恢复检查点"""
        backup_path = self._get_backup_path(checkpoint_id)
        
        if not os.path.exists(backup_path):
            raise ValueError(f"检查点 {checkpoint_id} 的备份不存在")
        
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")
        
        metadata = self.checkpoints[checkpoint_id]
        
        # 从备份恢复
        shutil.copy2(backup_path, metadata.file_path)
        
        print(f"✅ 从备份恢复检查点: {checkpoint_id}")
    
    def cleanup_checkpoints(self,
                           keep_count: Optional[int] = None,
                           checkpoint_type: Optional[CheckpointType] = None,
                           keep_best: bool = True,
                           metric_name: str = "total_reward"):
        """清理旧检查点"""
        if keep_count is None:
            keep_count = self.config.max_checkpoints
        
        # 筛选要清理的检查点
        candidates = []
        for checkpoint_id, metadata in self.checkpoints.items():
            if checkpoint_type is None or metadata.checkpoint_type == checkpoint_type:
                candidates.append((checkpoint_id, metadata))
        
        if len(candidates) <= keep_count:
            return  # 不需要清理
        
        # 按时间排序
        candidates.sort(key=lambda x: x[1].creation_time, reverse=True)
        
        # 保留最佳检查点
        best_checkpoint = None
        if keep_best:
            best_candidates = [(cid, meta) for cid, meta in candidates 
                             if metric_name in meta.performance_metrics]
            if best_candidates:
                best_checkpoint = max(best_candidates, 
                                    key=lambda x: x[1].performance_metrics[metric_name])[0]
        
        # 确定要删除的检查点
        to_delete = []
        keep_count_remaining = keep_count
        
        for checkpoint_id, metadata in candidates:
            if checkpoint_id == best_checkpoint:
                continue  # 保留最佳
            
            if keep_count_remaining > 0:
                keep_count_remaining -= 1
                continue  # 保留最新的
            
            to_delete.append(checkpoint_id)
        
        # 删除检查点
        deleted_count = 0
        for checkpoint_id in to_delete:
            try:
                self.delete_checkpoint(checkpoint_id)
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ 删除检查点 {checkpoint_id} 失败: {str(e)}")
        
        self.stats['cleanup_count'] += deleted_count
        print(f"✅ 清理完成: 删除了 {deleted_count} 个旧检查点")
    
    def _cleanup_old_checkpoints(self, checkpoint_type: CheckpointType):
        """自动清理旧检查点"""
        self.cleanup_checkpoints(checkpoint_type=checkpoint_type)
    
    def _generate_checkpoint_id(self,
                               checkpoint_type: CheckpointType,
                               episode: Optional[int] = None,
                               step: Optional[int] = None) -> str:
        """生成检查点ID"""
        timestamp = int(time.time() * 1000)
        
        id_parts = [checkpoint_type.value, str(timestamp)]
        
        if episode is not None:
            id_parts.append(f"ep{episode}")
        
        if step is not None:
            id_parts.append(f"step{step}")
        
        return "_".join(id_parts)
    
    def _get_checkpoint_path(self, checkpoint_id: str, checkpoint_type: CheckpointType) -> str:
        """获取检查点文件路径"""
        type_dir_map = {
            CheckpointType.MODEL: "models",
            CheckpointType.OPTIMIZER: "optimizers",
            CheckpointType.TRAINING_STATE: "training_states",
            CheckpointType.FULL_SYSTEM: "training_states",
            CheckpointType.EXPERIMENT: "experiments",
            CheckpointType.BEST_MODEL: "models",
            CheckpointType.LATEST: "models",
            CheckpointType.BACKUP: "backups"
        }
        
        subdir = type_dir_map.get(checkpoint_type, "models")
        extension = ".pth" if checkpoint_type in [CheckpointType.MODEL, CheckpointType.OPTIMIZER, 
                                                 CheckpointType.TRAINING_STATE, CheckpointType.FULL_SYSTEM] else ".pkl"
        
        return os.path.join(self.base_dir, subdir, f"{checkpoint_id}{extension}")
    
    def _get_metadata_path(self, checkpoint_id: str) -> str:
        """获取元数据文件路径"""
        return os.path.join(self.base_dir, "metadata", f"{checkpoint_id}_metadata.json")
    
    def _get_backup_path(self, checkpoint_id: str) -> str:
        """获取备份文件路径"""
        if checkpoint_id in self.checkpoints:
            original_path = self.checkpoints[checkpoint_id].file_path
            filename = os.path.basename(original_path)
            return os.path.join(self.base_dir, "backups", f"backup_{filename}")
        else:
            return os.path.join(self.base_dir, "backups", f"backup_{checkpoint_id}.pth")
    
    def _save_metadata(self, metadata: CheckpointMetadata):
        """保存元数据"""
        metadata_path = self._get_metadata_path(metadata.checkpoint_id)
        
        metadata_dict = {
            'checkpoint_id': metadata.checkpoint_id,
            'checkpoint_type': metadata.checkpoint_type.value,
            'creation_time': metadata.creation_time,
            'file_path': metadata.file_path,
            'file_size': metadata.file_size,
            'episode': metadata.episode,
            'step': metadata.step,
            'performance_metrics': metadata.performance_metrics,
            'model_architecture': metadata.model_architecture,
            'model_parameters': metadata.model_parameters,
            'version': metadata.version,
            'framework_version': metadata.framework_version,
            'checksum': metadata.checksum,
            'is_valid': metadata.is_valid,
            'tags': metadata.tags,
            'description': metadata.description
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def _load_existing_checkpoints(self):
        """加载现有检查点"""
        metadata_dir = os.path.join(self.base_dir, "metadata")
        
        if not os.path.exists(metadata_dir):
            return
        
        for filename in os.listdir(metadata_dir):
            if filename.endswith('_metadata.json'):
                metadata_path = os.path.join(metadata_dir, filename)
                
                try:
                    with open(metadata_path, 'r') as f:
                        metadata_dict = json.load(f)
                    
                    # 验证文件是否存在
                    if not os.path.exists(metadata_dict['file_path']):
                        continue
                    
                    # 重建元数据对象
                    metadata = CheckpointMetadata(
                        checkpoint_id=metadata_dict['checkpoint_id'],
                        checkpoint_type=CheckpointType(metadata_dict['checkpoint_type']),
                        creation_time=metadata_dict['creation_time'],
                        file_path=metadata_dict['file_path'],
                        file_size=metadata_dict['file_size'],
                        episode=metadata_dict.get('episode'),
                        step=metadata_dict.get('step'),
                        performance_metrics=metadata_dict.get('performance_metrics', {}),
                        model_architecture=metadata_dict.get('model_architecture'),
                        model_parameters=metadata_dict.get('model_parameters'),
                        version=metadata_dict.get('version', '1.0'),
                        framework_version=metadata_dict.get('framework_version'),
                        checksum=metadata_dict.get('checksum'),
                        is_valid=metadata_dict.get('is_valid', True),
                        tags=metadata_dict.get('tags', []),
                        description=metadata_dict.get('description', '')
                    )
                    
                    self.checkpoints[metadata.checkpoint_id] = metadata
                    self.checkpoint_history.append(metadata.checkpoint_id)
                    
                except Exception as e:
                    print(f"⚠️ 加载元数据失败 {filename}: {str(e)}")
        
        # 按时间排序历史记录
        self.checkpoint_history.sort(key=lambda cid: self.checkpoints[cid].creation_time)
    
    def _create_backup(self, checkpoint_id: str) -> str:
        """创建备份"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"检查点 {checkpoint_id} 不存在")
        
        metadata = self.checkpoints[checkpoint_id]
        backup_path = self._get_backup_path(checkpoint_id)
        
        # 创建备份目录
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        # 复制文件
        shutil.copy2(metadata.file_path, backup_path)
        
        print(f"✅ 创建备份: {checkpoint_id} -> {backup_path}")
        
        return backup_path
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        """验证文件校验和"""
        actual_checksum = self._calculate_checksum(file_path)
        return actual_checksum == expected_checksum
    
    def wait_for_saves(self):
        """等待所有异步保存完成"""
        for thread in self._save_thread_pool:
            if thread.is_alive():
                thread.join()
        
        self._save_thread_pool.clear()
        print("✅ 所有异步保存操作已完成")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_size_mb = self.stats['total_size_bytes'] / 1024 / 1024
        
        stats = self.stats.copy()
        stats.update({
            'total_checkpoints': len(self.checkpoints),
            'total_size_mb': total_size_mb,
            'avg_size_mb': total_size_mb / max(len(self.checkpoints), 1),
            'checkpoints_by_type': {},
            'success_rate_save': (self.stats['total_saves'] - self.stats['save_errors']) / max(self.stats['total_saves'], 1),
            'success_rate_load': (self.stats['total_loads'] - self.stats['load_errors']) / max(self.stats['total_loads'], 1)
        })
        
        # 按类型统计
        for metadata in self.checkpoints.values():
            checkpoint_type = metadata.checkpoint_type.value
            if checkpoint_type not in stats['checkpoints_by_type']:
                stats['checkpoints_by_type'][checkpoint_type] = 0
            stats['checkpoints_by_type'][checkpoint_type] += 1
        
        return stats
    
    def create_checkpoint_report(self) -> Dict[str, Any]:
        """创建检查点报告"""
        report = {
            'manager_info': {
                'manager_id': self.manager_id,
                'base_directory': self.base_dir,
                'config': {
                    'max_checkpoints': self.config.max_checkpoints,
                    'save_frequency': self.config.save_frequency,
                    'auto_cleanup': self.config.auto_cleanup,
                    'backup_enabled': self.config.backup_enabled
                }
            },
            'statistics': self.get_statistics(),
            'recent_checkpoints': [],
            'best_checkpoints': {},
            'storage_analysis': {},
            'recommendations': []
        }
        
        # 最近的检查点
        recent_ids = self.checkpoint_history[-10:] if len(self.checkpoint_history) >= 10 else self.checkpoint_history
        for checkpoint_id in reversed(recent_ids):
            if checkpoint_id in self.checkpoints:
                report['recent_checkpoints'].append(self.get_checkpoint_info(checkpoint_id))
        
        # 最佳检查点
        metrics = set()
        for metadata in self.checkpoints.values():
            metrics.update(metadata.performance_metrics.keys())
        
        for metric in metrics:
            try:
                best_id, _ = self.load_best_checkpoint(metric, maximize=True)
                report['best_checkpoints'][f'best_{metric}'] = self.get_checkpoint_info(best_id)
            except:
                pass
        
        # 存储分析
        type_sizes = {}
        for metadata in self.checkpoints.values():
            checkpoint_type = metadata.checkpoint_type.value
            if checkpoint_type not in type_sizes:
                type_sizes[checkpoint_type] = 0
            type_sizes[checkpoint_type] += metadata.file_size
        
        report['storage_analysis'] = {
            'size_by_type_mb': {k: v/1024/1024 for k, v in type_sizes.items()},
            'largest_checkpoint': max(self.checkpoints.values(), key=lambda x: x.file_size, default=None),
            'oldest_checkpoint': min(self.checkpoints.values(), key=lambda x: x.creation_time, default=None)
        }
        
        # 生成建议
        total_size_gb = self.stats['total_size_bytes'] / 1024 / 1024 / 1024
        if total_size_gb > 10:
            report['recommendations'].append("存储空间使用较大，建议清理旧检查点")
        
        if self.stats['save_errors'] > 0:
            report['recommendations'].append(f"发生了 {self.stats['save_errors']} 次保存错误，请检查存储空间和权限")
        
        if len(self.checkpoints) > self.config.max_checkpoints * 1.5:
            report['recommendations'].append("检查点数量过多，建议启用自动清理")
        
        return report
    
    def export_all_checkpoints(self, export_dir: str):
        """导出所有检查点"""
        os.makedirs(export_dir, exist_ok=True)
        
        exported_count = 0
        for checkpoint_id in self.checkpoints:
            try:
                export_path = os.path.join(export_dir, f"{checkpoint_id}.pth")
                self.export_checkpoint(checkpoint_id, export_path, include_metadata=True)
                exported_count += 1
            except Exception as e:
                print(f"⚠️ 导出检查点 {checkpoint_id} 失败: {str(e)}")
        
        # 导出管理器报告
        report = self.create_checkpoint_report()
        report_path = os.path.join(export_dir, "checkpoint_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ 成功导出 {exported_count}/{len(self.checkpoints)} 个检查点到 {export_dir}")
    
    def __del__(self):
        """析构函数：等待异步操作完成"""
        try:
            self.wait_for_saves()
        except:
            pass
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"CheckpointManager({self.manager_id}): "
                f"检查点={len(self.checkpoints)}, "
                f"总大小={self.stats['total_size_bytes']/1024/1024:.1f}MB")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"CheckpointManager(manager_id='{self.manager_id}', "
                f"base_dir='{self.base_dir}', "
                f"checkpoints={len(self.checkpoints)})")
