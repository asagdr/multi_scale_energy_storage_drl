import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy import signal, interpolate
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class PreprocessingMethod(Enum):
    """预处理方法枚举"""
    NORMALIZATION = "normalization"        # 标准化
    SCALING = "scaling"                    # 缩放
    FILTERING = "filtering"                # 滤波
    RESAMPLING = "resampling"              # 重采样
    IMPUTATION = "imputation"              # 插值补值
    OUTLIER_REMOVAL = "outlier_removal"    # 异常值移除
    DENOISING = "denoising"                # 去噪
    FEATURE_SELECTION = "feature_selection" # 特征选择
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"  # 降维
    AUGMENTATION = "augmentation"          # 数据增强

@dataclass
class PreprocessingConfig:
    """预处理配置"""
    # 标准化参数
    normalization_method: str = "zscore"   # "zscore", "minmax", "robust"
    
    # 滤波参数
    filter_type: str = "lowpass"           # "lowpass", "highpass", "bandpass", "notch"
    filter_order: int = 4                  # 滤波器阶数
    cutoff_frequency: float = 0.1          # 截止频率
    
    # 重采样参数
    target_sampling_rate: float = 1.0      # 目标采样率 (Hz)
    resampling_method: str = "linear"      # "linear", "cubic", "nearest"
    
    # 插值参数
    imputation_strategy: str = "mean"      # "mean", "median", "mode", "knn", "interpolate"
    missing_threshold: float = 0.5         # 缺失值阈值
    
    # 异常值检测参数
    outlier_method: str = "zscore"         # "zscore", "iqr", "isolation_forest"
    outlier_threshold: float = 3.0         # 异常值阈值
    
    # 去噪参数
    denoising_method: str = "savgol"       # "savgol", "median", "gaussian", "wavelet"
    window_length: int = 11                # 窗口长度
    
    # 降维参数
    n_components: int = 10                 # 主成分数量
    variance_threshold: float = 0.95       # 方差保留阈值
    
    # 数据增强参数
    augmentation_factor: float = 2.0       # 增强倍数
    noise_level: float = 0.01              # 噪声水平

@dataclass
class PreprocessedData:
    """预处理后的数据"""
    data_id: str
    original_data_id: str
    methods_applied: List[PreprocessingMethod]
    config: PreprocessingConfig
    
    # 处理后的数据
    processed_data: Dict[str, np.ndarray]
    
    # 处理参数和统计
    scaling_params: Dict[str, Any] = field(default_factory=dict)
    filtering_params: Dict[str, Any] = field(default_factory=dict)
    imputation_stats: Dict[str, Any] = field(default_factory=dict)
    outlier_stats: Dict[str, Any] = field(default_factory=dict)
    
    # 数据质量指标
    data_quality_before: Dict[str, float] = field(default_factory=dict)
    data_quality_after: Dict[str, float] = field(default_factory=dict)
    quality_improvement: Dict[str, float] = field(default_factory=dict)
    
    # 元数据
    processing_time: float = field(default_factory=time.time)
    data_shape_before: Dict[str, Tuple] = field(default_factory=dict)
    data_shape_after: Dict[str, Tuple] = field(default_factory=dict)

class DataPreprocessor:
    """
    数据预处理器
    提供全面的数据预处理功能
    """
    
    def __init__(self, preprocessor_id: str = "DataPreprocessor_001"):
        """
        初始化数据预处理器
        
        Args:
            preprocessor_id: 预处理器ID
        """
        self.preprocessor_id = preprocessor_id
        
        # === 预处理方法映射 ===
        self.preprocessing_methods = {
            PreprocessingMethod.NORMALIZATION: self._apply_normalization,
            PreprocessingMethod.SCALING: self._apply_scaling,
            PreprocessingMethod.FILTERING: self._apply_filtering,
            PreprocessingMethod.RESAMPLING: self._apply_resampling,
            PreprocessingMethod.IMPUTATION: self._apply_imputation,
            PreprocessingMethod.OUTLIER_REMOVAL: self._apply_outlier_removal,
            PreprocessingMethod.DENOISING: self._apply_denoising,
            PreprocessingMethod.FEATURE_SELECTION: self._apply_feature_selection,
            PreprocessingMethod.DIMENSIONALITY_REDUCTION: self._apply_dimensionality_reduction,
            PreprocessingMethod.AUGMENTATION: self._apply_augmentation
        }
        
        # === 预处理统计 ===
        self.preprocessing_stats = {
            'total_preprocessed': 0,
            'methods_usage': {method: 0 for method in PreprocessingMethod},
            'processing_time': 0.0,
            'data_quality_improvements': []
        }
        
        print(f"✅ 数据预处理器初始化完成: {preprocessor_id}")
        print(f"   支持预处理方法: {len(self.preprocessing_methods)} 种")
    
    def preprocess_data(self,
                       data: Dict[str, np.ndarray],
                       methods: List[PreprocessingMethod],
                       config: Optional[PreprocessingConfig] = None,
                       original_data_id: str = "unknown",
                       data_id: Optional[str] = None) -> PreprocessedData:
        """
        预处理数据
        
        Args:
            data: 输入数据字典
            methods: 预处理方法列表
            config: 预处理配置
            original_data_id: 原始数据ID
            data_id: 预处理数据ID
            
        Returns:
            预处理后的数据
        """
        processing_start_time = time.time()
        
        # 使用默认配置或提供的配置
        if config is None:
            config = PreprocessingConfig()
        
        # 生成数据ID
        if data_id is None:
            data_id = f"preprocessed_{int(time.time()*1000)}"
        
        # 评估原始数据质量
        data_quality_before = self._assess_data_quality(data)
        data_shape_before = {name: array.shape for name, array in data.items()}
        
        # 复制数据以避免修改原始数据
        processed_data = {name: array.copy() for name, array in data.items()}
        
        # 初始化预处理结果对象
        result = PreprocessedData(
            data_id=data_id,
            original_data_id=original_data_id,
            methods_applied=methods,
            config=config,
            processed_data=processed_data,
            data_quality_before=data_quality_before,
            data_shape_before=data_shape_before
        )
        
        # 按顺序应用预处理方法
        for method in methods:
            if method in self.preprocessing_methods:
                try:
                    print(f"  应用预处理方法: {method.value}")
                    self.preprocessing_methods[method](result)
                    self.preprocessing_stats['methods_usage'][method] += 1
                except Exception as e:
                    print(f"⚠️ 预处理方法 {method.value} 失败: {str(e)}")
        
        # 评估处理后数据质量
        result.data_quality_after = self._assess_data_quality(result.processed_data)
        result.data_shape_after = {name: array.shape for name, array in result.processed_data.items()}
        
        # 计算质量改善
        result.quality_improvement = {
            name: result.data_quality_after.get(name, 0) - result.data_quality_before.get(name, 0)
            for name in result.data_quality_before.keys()
        }
        
        # 更新统计
        processing_time = time.time() - processing_start_time
        result.processing_time = processing_time
        self._update_preprocessing_stats(len(methods), processing_time, result.quality_improvement)
        
        print(f"✅ 数据预处理完成: {data_id}")
        print(f"   应用方法: {len(methods)}, 处理时间: {processing_time:.2f}s")
        
        return result
    
    def _apply_normalization(self, result: PreprocessedData):
        """应用标准化"""
        config = result.config
        
        for signal_name, signal_data in result.processed_data.items():
            if config.normalization_method == "zscore":
                # Z-score标准化
                mean = np.mean(signal_data)
                std = np.std(signal_data)
                if std > 0:
                    normalized = (signal_data - mean) / std
                    result.processed_data[signal_name] = normalized
                    result.scaling_params[f"{signal_name}_zscore"] = {'mean': mean, 'std': std}
                
            elif config.normalization_method == "minmax":
                # Min-Max标准化
                min_val = np.min(signal_data)
                max_val = np.max(signal_data)
                if max_val > min_val:
                    normalized = (signal_data - min_val) / (max_val - min_val)
                    result.processed_data[signal_name] = normalized
                    result.scaling_params[f"{signal_name}_minmax"] = {'min': min_val, 'max': max_val}
                
            elif config.normalization_method == "robust":
                # 鲁棒标准化
                median = np.median(signal_data)
                mad = np.median(np.abs(signal_data - median))
                if mad > 0:
                    normalized = (signal_data - median) / mad
                    result.processed_data[signal_name] = normalized
                    result.scaling_params[f"{signal_name}_robust"] = {'median': median, 'mad': mad}
    
    def _apply_scaling(self, result: PreprocessedData):
        """应用缩放"""
        # 这里可以实现其他缩放方法
        # 目前与标准化方法类似，可以扩展
        self._apply_normalization(result)
    
    def _apply_filtering(self, result: PreprocessedData):
        """应用滤波"""
        config = result.config
        
        for signal_name, signal_data in result.processed_data.items():
            if len(signal_data) < 2 * config.filter_order:
                continue
                
            try:
                if config.filter_type == "lowpass":
                    # 低通滤波
                    b, a = signal.butter(config.filter_order, config.cutoff_frequency, btype='low')
                    filtered = signal.filtfilt(b, a, signal_data)
                    
                elif config.filter_type == "highpass":
                    # 高通滤波
                    b, a = signal.butter(config.filter_order, config.cutoff_frequency, btype='high')
                    filtered = signal.filtfilt(b, a, signal_data)
                    
                elif config.filter_type == "bandpass":
                    # 带通滤波
                    low_freq = config.cutoff_frequency
                    high_freq = min(config.cutoff_frequency * 10, 0.45)  # 避免超过奈奎斯特频率
                    b, a = signal.butter(config.filter_order, [low_freq, high_freq], btype='band')
                    filtered = signal.filtfilt(b, a, signal_data)
                    
                elif config.filter_type == "notch":
                    # 陷波滤波
                    b, a = signal.iirnotch(config.cutoff_frequency, 30)  # Q=30
                    filtered = signal.filtfilt(b, a, signal_data)
                
                else:
                    continue
                
                result.processed_data[signal_name] = filtered
                result.filtering_params[signal_name] = {
                    'filter_type': config.filter_type,
                    'order': config.filter_order,
                    'cutoff_frequency': config.cutoff_frequency
                }
                
            except Exception as e:
                print(f"⚠️ 滤波失败 {signal_name}: {str(e)}")
    
    def _apply_resampling(self, result: PreprocessedData):
        """应用重采样"""
        config = result.config
        
        for signal_name, signal_data in result.processed_data.items():
            current_length = len(signal_data)
            
            # 计算目标长度（假设原始采样率为1Hz）
            current_sampling_rate = 1.0  # 简化假设
            target_length = int(current_length * config.target_sampling_rate / current_sampling_rate)
            
            if target_length != current_length and target_length > 0:
                # 创建新的时间索引
                old_indices = np.linspace(0, current_length - 1, current_length)
                new_indices = np.linspace(0, current_length - 1, target_length)
                
                # 插值重采样
                if config.resampling_method == "linear":
                    resampled = np.interp(new_indices, old_indices, signal_data)
                elif config.resampling_method == "cubic":
                    f = interpolate.interp1d(old_indices, signal_data, kind='cubic', 
                                           bounds_error=False, fill_value='extrapolate')
                    resampled = f(new_indices)
                elif config.resampling_method == "nearest":
                    f = interpolate.interp1d(old_indices, signal_data, kind='nearest')
                    resampled = f(new_indices)
                else:
                    continue
                
                result.processed_data[signal_name] = resampled
    
    def _apply_imputation(self, result: PreprocessedData):
        """应用插值补值"""
        config = result.config
        
        for signal_name, signal_data in result.processed_data.items():
            # 检测缺失值（NaN和异常大的值）
            nan_mask = np.isnan(signal_data) | np.isinf(signal_data)
            missing_ratio = np.sum(nan_mask) / len(signal_data)
            
            if missing_ratio > config.missing_threshold:
                print(f"⚠️ {signal_name} 缺失值比例过高: {missing_ratio:.2%}")
                continue
            
            if np.sum(nan_mask) > 0:
                if config.imputation_strategy == "mean":
                    # 均值插值
                    fill_value = np.nanmean(signal_data)
                    signal_data[nan_mask] = fill_value
                    
                elif config.imputation_strategy == "median":
                    # 中位数插值
                    fill_value = np.nanmedian(signal_data)
                    signal_data[nan_mask] = fill_value
                    
                elif config.imputation_strategy == "interpolate":
                    # 线性插值
                    valid_indices = np.where(~nan_mask)[0]
                    if len(valid_indices) >= 2:
                        f = interpolate.interp1d(valid_indices, signal_data[valid_indices], 
                                               kind='linear', bounds_error=False, 
                                               fill_value='extrapolate')
                        missing_indices = np.where(nan_mask)[0]
                        signal_data[missing_indices] = f(missing_indices)
                
                result.processed_data[signal_name] = signal_data
                result.imputation_stats[signal_name] = {
                    'missing_count': np.sum(nan_mask),
                    'missing_ratio': missing_ratio,
                    'strategy': config.imputation_strategy
                }
    
    def _apply_outlier_removal(self, result: PreprocessedData):
        """应用异常值移除"""
        config = result.config
        
        for signal_name, signal_data in result.processed_data.items():
            if config.outlier_method == "zscore":
                # Z-score方法
                z_scores = np.abs(stats.zscore(signal_data))
                outlier_mask = z_scores > config.outlier_threshold
                
            elif config.outlier_method == "iqr":
                # IQR方法
                q1 = np.percentile(signal_data, 25)
                q3 = np.percentile(signal_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask = (signal_data < lower_bound) | (signal_data > upper_bound)
                
            else:
                continue
            
            outlier_count = np.sum(outlier_mask)
            if outlier_count > 0:
                # 用中位数替换异常值
                median_value = np.median(signal_data[~outlier_mask])
                signal_data[outlier_mask] = median_value
                
                result.processed_data[signal_name] = signal_data
                result.outlier_stats[signal_name] = {
                    'outlier_count': outlier_count,
                    'outlier_ratio': outlier_count / len(signal_data),
                    'method': config.outlier_method,
                    'replacement_value': median_value
                }
    
    def _apply_denoising(self, result: PreprocessedData):
        """应用去噪"""
        config = result.config
        
        for signal_name, signal_data in result.processed_data.items():
            if config.denoising_method == "savgol":
                # Savitzky-Golay滤波
                if len(signal_data) > config.window_length:
                    try:
                        denoised = signal.savgol_filter(signal_data, config.window_length, 3)
                        result.processed_data[signal_name] = denoised
                    except:
                        pass
                        
            elif config.denoising_method == "median":
                # 中值滤波
                if len(signal_data) > config.window_length:
                    denoised = signal.medfilt(signal_data, kernel_size=config.window_length)
                    result.processed_data[signal_name] = denoised
                    
            elif config.denoising_method == "gaussian":
                # 高斯滤波
                from scipy.ndimage import gaussian_filter1d
                sigma = config.window_length / 6  # 经验值
                denoised = gaussian_filter1d(signal_data, sigma)
                result.processed_data[signal_name] = denoised
    
    def _apply_feature_selection(self, result: PreprocessedData):
        """应用特征选择"""
        # 基于方差的特征选择
        feature_variances = {}
        for signal_name, signal_data in result.processed_data.items():
            variance = np.var(signal_data)
            feature_variances[signal_name] = variance
        
        # 移除低方差特征
        low_variance_threshold = np.percentile(list(feature_variances.values()), 10)
        
        signals_to_remove = []
        for signal_name, variance in feature_variances.items():
            if variance < low_variance_threshold:
                signals_to_remove.append(signal_name)
        
        for signal_name in signals_to_remove:
            if signal_name in result.processed_data:
                del result.processed_data[signal_name]
                print(f"  移除低方差特征: {signal_name}")
    
    def _apply_dimensionality_reduction(self, result: PreprocessedData):
        """应用降维"""
        config = result.config
        
        # 准备数据矩阵
        signal_names = list(result.processed_data.keys())
        if len(signal_names) < 2:
            return
        
        min_length = min(len(data) for data in result.processed_data.values())
        data_matrix = np.array([result.processed_data[name][:min_length] for name in signal_names]).T
        
        # 应用PCA
        try:
            pca = PCA(n_components=min(config.n_components, len(signal_names)))
            transformed_data = pca.fit_transform(data_matrix)
            
            # 检查方差保留比例
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components_needed = np.argmax(cumulative_variance >= config.variance_threshold) + 1
            
            if n_components_needed < len(signal_names):
                # 重新应用PCA
                pca_final = PCA(n_components=n_components_needed)
                final_transformed = pca_final.fit_transform(data_matrix)
                
                # 替换原始数据
                result.processed_data.clear()
                for i in range(final_transformed.shape[1]):
                    result.processed_data[f"pca_component_{i}"] = final_transformed[:, i]
                
                print(f"  降维: {len(signal_names)} -> {n_components_needed} 维")
                
        except Exception as e:
            print(f"⚠️ 降维失败: {str(e)}")
    
    def _apply_augmentation(self, result: PreprocessedData):
        """应用数据增强"""
        config = result.config
        
        augmented_data = {}
        
        for signal_name, signal_data in result.processed_data.items():
            # 原始数据
            augmented_data[signal_name] = signal_data
            
            # 添加噪声
            noise = np.random.normal(0, config.noise_level * np.std(signal_data), len(signal_data))
            augmented_data[f"{signal_name}_noisy"] = signal_data + noise
            
            # 时间拉伸
            if len(signal_data) > 10:
                stretch_factor = np.random.uniform(0.8, 1.2)
                stretched_length = int(len(signal_data) * stretch_factor)
                old_indices = np.linspace(0, len(signal_data) - 1, len(signal_data))
                new_indices = np.linspace(0, len(signal_data) - 1, stretched_length)
                stretched = np.interp(new_indices, old_indices, signal_data)
                augmented_data[f"{signal_name}_stretched"] = stretched
            
            # 幅值缩放
            scale_factor = np.random.uniform(0.9, 1.1)
            augmented_data[f"{signal_name}_scaled"] = signal_data * scale_factor
        
        # 根据增强倍数选择数据
        if config.augmentation_factor > 1.0:
            result.processed_data = augmented_data
        else:
            # 只保留部分增强数据
            n_augmented = int((config.augmentation_factor - 1) * len(result.processed_data))
            augmented_keys = list(augmented_data.keys())[len(result.processed_data):len(result.processed_data) + n_augmented]
            for key in augmented_keys:
                result.processed_data[key] = augmented_data[key]
    
    def _assess_data_quality(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """评估数据质量"""
        quality_scores = {}
        
        for signal_name, signal_data in data.items():
            quality_factors = []
            
            # 1. 完整性（无缺失值）
            completeness = 1.0 - np.sum(np.isnan(signal_data) | np.isinf(signal_data)) / len(signal_data)
            quality_factors.append(completeness)
            
            # 2. 一致性（连续性）
            if len(signal_data) > 1:
                diff_data = np.diff(signal_data)
                consistency = 1.0 - np.sum(np.abs(diff_data) > 5 * np.std(diff_data)) / len(diff_data)
                quality_factors.append(consistency)
            
            # 3. 准确性（基于统计分布）
            try:
                from scipy import stats
                _, p_value = stats.normaltest(signal_data)
                accuracy = min(1.0, p_value * 10)  # 简化的准确性指标
                quality_factors.append(accuracy)
            except:
                quality_factors.append(0.5)
            
            # 4. 稳定性（方差合理性）
            cv = np.std(signal_data) / (np.mean(signal_data) + 1e-10)
            stability = max(0.0, 1.0 - cv)
            quality_factors.append(stability)
            
            # 综合质量分数
            quality_scores[signal_name] = np.mean(quality_factors)
        
        return quality_scores
    
    def _update_preprocessing_stats(self, num_methods: int, processing_time: float, quality_improvement: Dict[str, float]):
        """更新预处理统计"""
        self.preprocessing_stats['total_preprocessed'] += 1
        self.preprocessing_stats['processing_time'] += processing_time
        
        # 记录质量改善
        avg_improvement = np.mean(list(quality_improvement.values())) if quality_improvement else 0
        self.preprocessing_stats['data_quality_improvements'].append(avg_improvement)
    
    def create_preprocessing_pipeline(self, methods: List[PreprocessingMethod], config: Optional[PreprocessingConfig] = None) -> Dict[str, Any]:
        """创建预处理流水线"""
        if config is None:
            config = PreprocessingConfig()
        
        pipeline = {
            'methods': [method.value for method in methods],
            'config': {
                'normalization_method': config.normalization_method,
                'filter_type': config.filter_type,
                'filter_order': config.filter_order,
                'cutoff_frequency': config.cutoff_frequency,
                'target_sampling_rate': config.target_sampling_rate,
                'imputation_strategy': config.imputation_strategy,
                'outlier_method': config.outlier_method,
                'outlier_threshold': config.outlier_threshold,
                'denoising_method': config.denoising_method
            },
            'created_time': time.time()
        }
        
        return pipeline
    
    def apply_pipeline(self, data: Dict[str, np.ndarray], pipeline: Dict[str, Any], data_id: Optional[str] = None) -> PreprocessedData:
        """应用预处理流水线"""
        methods = [PreprocessingMethod(method) for method in pipeline['methods']]
        
        # 重建配置
        config = PreprocessingConfig()
        config_dict = pipeline['config']
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return self.preprocess_data(data, methods, config, data_id=data_id)
    
    def export_preprocessed_data(self, preprocessed_data: PreprocessedData, file_path: str, format: str = 'json'):
        """导出预处理后的数据"""
        try:
            if format.lower() == 'json':
                export_data = {
                    'data_id': preprocessed_data.data_id,
                    'original_data_id': preprocessed_data.original_data_id,
                    'methods_applied': [method.value for method in preprocessed_data.methods_applied],
                    'processed_data': {name: data.tolist() for name, data in preprocessed_data.processed_data.items()},
                    'scaling_params': preprocessed_data.scaling_params,
                    'filtering_params': preprocessed_data.filtering_params,
                    'imputation_stats': preprocessed_data.imputation_stats,
                    'outlier_stats': preprocessed_data.outlier_stats,
                    'data_quality_before': preprocessed_data.data_quality_before,
                    'data_quality_after': preprocessed_data.data_quality_after,
                    'quality_improvement': preprocessed_data.quality_improvement,
                    'processing_time': preprocessed_data.processing_time,
                    'data_shape_before': {name: list(shape) for name, shape in preprocessed_data.data_shape_before.items()},
                    'data_shape_after': {name: list(shape) for name, shape in preprocessed_data.data_shape_after.items()}
                }
                
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format.lower() == 'csv':
                # 创建综合DataFrame
                data_dict = {}
                for name, array in preprocessed_data.processed_data.items():
                    if len(array.shape) == 1:
                        data_dict[name] = array
                    else:
                        # 多维数组展平
                        for i in range(array.shape[1]):
                            data_dict[f"{name}_dim_{i}"] = array[:, i]
                
                df = pd.DataFrame(data_dict)
                df.to_csv(file_path, index=False)
            
            print(f"✅ 预处理数据已导出: {file_path}")
            
        except Exception as e:
            print(f"❌ 预处理数据导出失败: {str(e)}")
    
    def compare_preprocessing_effects(self, original_data: Dict[str, np.ndarray], 
                                    preprocessed_data: PreprocessedData) -> Dict[str, Any]:
        """比较预处理效果"""
        comparison = {
            'data_quality_improvement': {},
            'signal_statistics_comparison': {},
            'processing_summary': {},
            'recommendations': []
        }
        
        # 数据质量改善
        comparison['data_quality_improvement'] = preprocessed_data.quality_improvement
        
        # 信号统计对比
        for signal_name in original_data.keys():
            if signal_name in preprocessed_data.processed_data:
                original = original_data[signal_name]
                processed = preprocessed_data.processed_data[signal_name]
                
                comparison['signal_statistics_comparison'][signal_name] = {
                    'original': {
                        'mean': float(np.mean(original)),
                        'std': float(np.std(original)),
                        'min': float(np.min(original)),
                        'max': float(np.max(original)),
                        'length': len(original)
                    },
                    'processed': {
                        'mean': float(np.mean(processed)),
                        'std': float(np.std(processed)),
                        'min': float(np.min(processed)),
                        'max': float(np.max(processed)),
                        'length': len(processed)
                    }
                }
        
        # 处理摘要
        comparison['processing_summary'] = {
            'methods_applied': [method.value for method in preprocessed_data.methods_applied],
            'processing_time': preprocessed_data.processing_time,
            'data_shape_changes': {
                'before': preprocessed_data.data_shape_before,
                'after': preprocessed_data.data_shape_after
            },
            'overall_quality_improvement': np.mean(list(preprocessed_data.quality_improvement.values())) if preprocessed_data.quality_improvement else 0
        }
        
        # 生成建议
        avg_improvement = comparison['processing_summary']['overall_quality_improvement']
        if avg_improvement > 0.1:
            comparison['recommendations'].append("预处理效果良好，建议保留当前配置")
        elif avg_improvement > 0.05:
            comparison['recommendations'].append("预处理有一定效果，可考虑调整参数")
        else:
            comparison['recommendations'].append("预处理效果有限，建议尝试其他方法")
        
        if preprocessed_data.outlier_stats:
            total_outliers = sum(stats['outlier_count'] for stats in preprocessed_data.outlier_stats.values())
            if total_outliers > 0:
                comparison['recommendations'].append(f"检测到 {total_outliers} 个异常值已被处理")
        
        if preprocessed_data.imputation_stats:
            total_missing = sum(stats['missing_count'] for stats in preprocessed_data.imputation_stats.values())
            if total_missing > 0:
                comparison['recommendations'].append(f"补充了 {total_missing} 个缺失值")
        
        return comparison
    
    def get_preprocessing_statistics(self) -> Dict[str, Any]:
        """获取预处理统计信息"""
        stats = self.preprocessing_stats.copy()
        
        if stats['total_preprocessed'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_preprocessed']
            
            if stats['data_quality_improvements']:
                stats['avg_quality_improvement'] = np.mean(stats['data_quality_improvements'])
                stats['max_quality_improvement'] = np.max(stats['data_quality_improvements'])
                stats['min_quality_improvement'] = np.min(stats['data_quality_improvements'])
        else:
            stats['avg_processing_time'] = 0
            stats['avg_quality_improvement'] = 0
            stats['max_quality_improvement'] = 0
            stats['min_quality_improvement'] = 0
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"DataPreprocessor({self.preprocessor_id}): "
                f"处理次数={self.preprocessing_stats['total_preprocessed']}, "
                f"平均质量改善={np.mean(self.preprocessing_stats['data_quality_improvements']) if self.preprocessing_stats['data_quality_improvements'] else 0:.3f}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"DataPreprocessor(preprocessor_id='{self.preprocessor_id}', "
                f"methods={len(self.preprocessing_methods)}, "
                f"total_preprocessed={self.preprocessing_stats['total_preprocessed']})")
