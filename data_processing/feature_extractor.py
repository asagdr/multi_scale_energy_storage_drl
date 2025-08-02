import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class FeatureType(Enum):
    """特征类型枚举"""
    TEMPORAL = "temporal"              # 时域特征
    FREQUENCY = "frequency"            # 频域特征
    STATISTICAL = "statistical"       # 统计特征
    PHYSICAL = "physical"              # 物理特征
    PATTERN = "pattern"                # 模式特征
    CORRELATION = "correlation"        # 相关性特征
    TREND = "trend"                    # 趋势特征
    ANOMALY = "anomaly"               # 异常特征

@dataclass
class FeatureConfig:
    """特征提取配置"""
    window_size: int = 60              # 窗口大小（分钟）
    overlap_ratio: float = 0.5         # 重叠比例
    sampling_rate: float = 1.0         # 采样率（Hz）
    
    # 频域分析参数
    fft_size: int = 1024               # FFT大小
    frequency_bands: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.01),      # 超低频
        (0.01, 0.1),      # 低频
        (0.1, 1.0),       # 中频
        (1.0, 10.0),      # 高频
    ])
    
    # 统计分析参数
    percentiles: List[float] = field(default_factory=lambda: [5, 25, 50, 75, 95])
    
    # 模式识别参数
    pattern_length: int = 24           # 模式长度（小时）
    similarity_threshold: float = 0.8  # 相似度阈值
    
    # 异常检测参数
    anomaly_threshold: float = 3.0     # 异常阈值（标准差倍数）
    
@dataclass
class ExtractedFeatures:
    """提取的特征"""
    feature_id: str
    source_data_id: str
    feature_types: List[FeatureType]
    config: FeatureConfig
    
    # 特征数据
    temporal_features: Dict[str, np.ndarray] = field(default_factory=dict)
    frequency_features: Dict[str, np.ndarray] = field(default_factory=dict)
    statistical_features: Dict[str, np.ndarray] = field(default_factory=dict)
    physical_features: Dict[str, np.ndarray] = field(default_factory=dict)
    pattern_features: Dict[str, np.ndarray] = field(default_factory=dict)
    correlation_features: Dict[str, np.ndarray] = field(default_factory=dict)
    trend_features: Dict[str, np.ndarray] = field(default_factory=dict)
    anomaly_features: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 特征元数据
    feature_importance: Dict[str, float] = field(default_factory=dict)
    extraction_time: float = field(default_factory=time.time)
    quality_score: float = 0.0

class FeatureExtractor:
    """
    特征提取器
    从时间序列数据中提取多维度特征
    """
    
    def __init__(self, extractor_id: str = "FeatureExtractor_001"):
        """
        初始化特征提取器
        
        Args:
            extractor_id: 提取器ID
        """
        self.extractor_id = extractor_id
        
        # === 特征提取方法映射 ===
        self.extraction_methods = {
            FeatureType.TEMPORAL: self._extract_temporal_features,
            FeatureType.FREQUENCY: self._extract_frequency_features,
            FeatureType.STATISTICAL: self._extract_statistical_features,
            FeatureType.PHYSICAL: self._extract_physical_features,
            FeatureType.PATTERN: self._extract_pattern_features,
            FeatureType.CORRELATION: self._extract_correlation_features,
            FeatureType.TREND: self._extract_trend_features,
            FeatureType.ANOMALY: self._extract_anomaly_features
        }
        
        # === 特征重要性权重 ===
        self.feature_weights = {
            'mean': 0.8, 'std': 0.9, 'max': 0.7, 'min': 0.7,
            'skewness': 0.6, 'kurtosis': 0.5, 'energy': 0.8,
            'peak_frequency': 0.7, 'spectral_centroid': 0.6,
            'ramp_rate': 0.9, 'peak_count': 0.7, 'valley_count': 0.7,
            'trend_slope': 0.8, 'seasonality': 0.6, 'autocorr': 0.7
        }
        
        # === 提取统计 ===
        self.extraction_stats = {
            'total_extractions': 0,
            'extractions_by_type': {ftype: 0 for ftype in FeatureType},
            'total_features': 0,
            'extraction_time': 0.0
        }
        
        print(f"✅ 特征提取器初始化完成: {extractor_id}")
        print(f"   支持特征类型: {len(self.extraction_methods)} 种")
    
    def extract_features(self,
                        data: Dict[str, np.ndarray],
                        feature_types: List[FeatureType],
                        config: Optional[FeatureConfig] = None,
                        source_data_id: str = "unknown",
                        feature_id: Optional[str] = None) -> ExtractedFeatures:
        """
        提取特征
        
        Args:
            data: 输入数据字典
            feature_types: 要提取的特征类型列表
            config: 特征提取配置
            source_data_id: 源数据ID
            feature_id: 特征ID
            
        Returns:
            提取的特征
        """
        extraction_start_time = time.time()
        
        # 使用默认配置或提供的配置
        if config is None:
            config = FeatureConfig()
        
        # 生成特征ID
        if feature_id is None:
            feature_id = f"features_{int(time.time()*1000)}"
        
        # 初始化特征对象
        features = ExtractedFeatures(
            feature_id=feature_id,
            source_data_id=source_data_id,
            feature_types=feature_types,
            config=config
        )
        
        # 逐一提取各类型特征
        for feature_type in feature_types:
            if feature_type in self.extraction_methods:
                try:
                    method = self.extraction_methods[feature_type]
                    extracted = method(data, config)
                    
                    # 根据特征类型存储
                    if feature_type == FeatureType.TEMPORAL:
                        features.temporal_features.update(extracted)
                    elif feature_type == FeatureType.FREQUENCY:
                        features.frequency_features.update(extracted)
                    elif feature_type == FeatureType.STATISTICAL:
                        features.statistical_features.update(extracted)
                    elif feature_type == FeatureType.PHYSICAL:
                        features.physical_features.update(extracted)
                    elif feature_type == FeatureType.PATTERN:
                        features.pattern_features.update(extracted)
                    elif feature_type == FeatureType.CORRELATION:
                        features.correlation_features.update(extracted)
                    elif feature_type == FeatureType.TREND:
                        features.trend_features.update(extracted)
                    elif feature_type == FeatureType.ANOMALY:
                        features.anomaly_features.update(extracted)
                    
                    self.extraction_stats['extractions_by_type'][feature_type] += 1
                    
                except Exception as e:
                    print(f"⚠️ 提取 {feature_type.value} 特征失败: {str(e)}")
        
        # 计算特征重要性
        features.feature_importance = self._calculate_feature_importance(features)
        
        # 评估特征质量
        features.quality_score = self._assess_feature_quality(features)
        
        # 更新统计
        extraction_time = time.time() - extraction_start_time
        self._update_extraction_stats(len(feature_types), extraction_time)
        
        print(f"✅ 特征提取完成: {feature_id}")
        print(f"   特征类型: {len(feature_types)}, 质量分数: {features.quality_score:.3f}")
        
        return features
    
    def _extract_temporal_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取时域特征"""
        temporal_features = {}
        
        for signal_name, signal_data in data.items():
            prefix = f"{signal_name}_temporal"
            
            # 基础时域统计
            temporal_features[f"{prefix}_mean"] = np.array([np.mean(signal_data)])
            temporal_features[f"{prefix}_std"] = np.array([np.std(signal_data)])
            temporal_features[f"{prefix}_var"] = np.array([np.var(signal_data)])
            temporal_features[f"{prefix}_max"] = np.array([np.max(signal_data)])
            temporal_features[f"{prefix}_min"] = np.array([np.min(signal_data)])
            temporal_features[f"{prefix}_range"] = np.array([np.max(signal_data) - np.min(signal_data)])
            
            # 高阶统计量
            temporal_features[f"{prefix}_skewness"] = np.array([stats.skew(signal_data)])
            temporal_features[f"{prefix}_kurtosis"] = np.array([stats.kurtosis(signal_data)])
            
            # 能量相关特征
            temporal_features[f"{prefix}_energy"] = np.array([np.sum(signal_data**2)])
            temporal_features[f"{prefix}_rms"] = np.array([np.sqrt(np.mean(signal_data**2))])
            temporal_features[f"{prefix}_crest_factor"] = np.array([np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2))])
            
            # 变化率特征
            if len(signal_data) > 1:
                diff_signal = np.diff(signal_data)
                temporal_features[f"{prefix}_max_diff"] = np.array([np.max(np.abs(diff_signal))])
                temporal_features[f"{prefix}_mean_diff"] = np.array([np.mean(np.abs(diff_signal))])
                temporal_features[f"{prefix}_std_diff"] = np.array([np.std(diff_signal)])
                
                # 爬坡率特征
                positive_ramps = diff_signal[diff_signal > 0]
                negative_ramps = diff_signal[diff_signal < 0]
                temporal_features[f"{prefix}_max_pos_ramp"] = np.array([np.max(positive_ramps) if len(positive_ramps) > 0 else 0])
                temporal_features[f"{prefix}_max_neg_ramp"] = np.array([np.min(negative_ramps) if len(negative_ramps) > 0 else 0])
            
            # 零交叉率
            if len(signal_data) > 1:
                zero_crossings = np.sum(np.diff(np.sign(signal_data - np.mean(signal_data))) != 0)
                temporal_features[f"{prefix}_zero_crossing_rate"] = np.array([zero_crossings / len(signal_data)])
            
            # 窗口特征
            if len(signal_data) >= config.window_size:
                windows = self._create_windows(signal_data, config.window_size, config.overlap_ratio)
                window_means = np.array([np.mean(window) for window in windows])
                window_stds = np.array([np.std(window) for window in windows])
                
                temporal_features[f"{prefix}_window_mean_std"] = np.array([np.std(window_means)])
                temporal_features[f"{prefix}_window_std_mean"] = np.array([np.mean(window_stds)])
        
        return temporal_features
    
    def _extract_frequency_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取频域特征"""
        frequency_features = {}
        
        for signal_name, signal_data in data.items():
            prefix = f"{signal_name}_frequency"
            
            # FFT分析
            if len(signal_data) >= config.fft_size:
                # 计算功率谱密度
                freqs, psd = signal.periodogram(signal_data, fs=config.sampling_rate, nfft=config.fft_size)
                
                # 总功率
                total_power = np.sum(psd)
                frequency_features[f"{prefix}_total_power"] = np.array([total_power])
                
                # 频带功率
                for i, (low_freq, high_freq) in enumerate(config.frequency_bands):
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.sum(psd[band_mask])
                    frequency_features[f"{prefix}_band_{i}_power"] = np.array([band_power])
                    frequency_features[f"{prefix}_band_{i}_power_ratio"] = np.array([band_power / total_power if total_power > 0 else 0])
                
                # 峰值频率
                peak_freq_idx = np.argmax(psd[1:]) + 1  # 排除DC分量
                peak_frequency = freqs[peak_freq_idx]
                frequency_features[f"{prefix}_peak_frequency"] = np.array([peak_frequency])
                frequency_features[f"{prefix}_peak_power"] = np.array([psd[peak_freq_idx]])
                
                # 频谱质心
                spectral_centroid = np.sum(freqs * psd) / total_power if total_power > 0 else 0
                frequency_features[f"{prefix}_spectral_centroid"] = np.array([spectral_centroid])
                
                # 频谱带宽
                spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power) if total_power > 0 else 0
                frequency_features[f"{prefix}_spectral_bandwidth"] = np.array([spectral_bandwidth])
                
                # 频谱滚降
                cumulative_power = np.cumsum(psd)
                rolloff_threshold = 0.85 * total_power
                rolloff_idx = np.where(cumulative_power >= rolloff_threshold)[0]
                spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
                frequency_features[f"{prefix}_spectral_rolloff"] = np.array([spectral_rolloff])
                
                # 频谱平坦度
                geometric_mean = np.exp(np.mean(np.log(psd[1:] + 1e-10)))
                arithmetic_mean = np.mean(psd[1:])
                spectral_flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
                frequency_features[f"{prefix}_spectral_flatness"] = np.array([spectral_flatness])
            
            # 自相关分析
            if len(signal_data) > 10:
                autocorr = np.correlate(signal_data, signal_data, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # 归一化
                
                # 第一个显著的自相关峰（周期性检测）
                significant_corr = np.where(autocorr[1:] > 0.5)[0]
                if len(significant_corr) > 0:
                    primary_period = significant_corr[0] + 1
                    frequency_features[f"{prefix}_primary_period"] = np.array([primary_period])
                    frequency_features[f"{prefix}_primary_period_strength"] = np.array([autocorr[primary_period]])
                else:
                    frequency_features[f"{prefix}_primary_period"] = np.array([0])
                    frequency_features[f"{prefix}_primary_period_strength"] = np.array([0])
        
        return frequency_features
    
    def _extract_statistical_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取统计特征"""
        statistical_features = {}
        
        for signal_name, signal_data in data.items():
            prefix = f"{signal_name}_statistical"
            
            # 百分位数
            percentiles = np.percentile(signal_data, config.percentiles)
            for i, p in enumerate(config.percentiles):
                statistical_features[f"{prefix}_percentile_{p}"] = np.array([percentiles[i]])
            
            # 四分位数间距
            iqr = np.percentile(signal_data, 75) - np.percentile(signal_data, 25)
            statistical_features[f"{prefix}_iqr"] = np.array([iqr])
            
            # 变异系数
            cv = np.std(signal_data) / np.mean(signal_data) if np.mean(signal_data) != 0 else 0
            statistical_features[f"{prefix}_coefficient_of_variation"] = np.array([cv])
            
            # 中位数绝对偏差
            mad = np.median(np.abs(signal_data - np.median(signal_data)))
            statistical_features[f"{prefix}_mad"] = np.array([mad])
            
            # 熵（基于直方图）
            hist, _ = np.histogram(signal_data, bins=50, density=True)
            hist = hist[hist > 0]  # 去除零值
            entropy = -np.sum(hist * np.log2(hist))
            statistical_features[f"{prefix}_entropy"] = np.array([entropy])
            
            # 正态性检验（Shapiro-Wilk）
            if len(signal_data) >= 3 and len(signal_data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(signal_data)
                statistical_features[f"{prefix}_shapiro_stat"] = np.array([shapiro_stat])
                statistical_features[f"{prefix}_shapiro_p"] = np.array([shapiro_p])
            
            # 分布拟合（假设正态分布）
            mu, sigma = stats.norm.fit(signal_data)
            statistical_features[f"{prefix}_norm_mu"] = np.array([mu])
            statistical_features[f"{prefix}_norm_sigma"] = np.array([sigma])
            
            # KS检验（与正态分布比较）
            ks_stat, ks_p = stats.kstest(signal_data, lambda x: stats.norm.cdf(x, mu, sigma))
            statistical_features[f"{prefix}_ks_stat"] = np.array([ks_stat])
            statistical_features[f"{prefix}_ks_p"] = np.array([ks_p])
        
        return statistical_features
    
    def _extract_physical_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取物理特征"""
        physical_features = {}
        
        # 电池相关的物理特征
        if 'soc' in data:
            soc = data['soc']
            prefix = "soc_physical"
            
            # SOC利用率
            soc_utilization = (np.max(soc) - np.min(soc)) / 100.0
            physical_features[f"{prefix}_utilization"] = np.array([soc_utilization])
            
            # SOC均衡度（标准差）
            physical_features[f"{prefix}_balance"] = np.array([1.0 - np.std(soc) / 10.0])  # 归一化
            
            # 充放电循环识别
            if len(soc) > 1:
                soc_diff = np.diff(soc)
                charge_periods = np.sum(soc_diff > 0.1)  # 充电
                discharge_periods = np.sum(soc_diff < -0.1)  # 放电
                physical_features[f"{prefix}_charge_cycles"] = np.array([charge_periods])
                physical_features[f"{prefix}_discharge_cycles"] = np.array([discharge_periods])
        
        # 温度相关的物理特征
        if 'temperature' in data:
            temp = data['temperature']
            prefix = "temp_physical"
            
            # 温度梯度
            if len(temp) > 1:
                temp_gradient = np.mean(np.abs(np.diff(temp)))
                physical_features[f"{prefix}_gradient"] = np.array([temp_gradient])
            
            # 热应力指数
            thermal_stress = np.sum(np.abs(temp - 25)) / len(temp)  # 偏离最佳温度
            physical_features[f"{prefix}_thermal_stress"] = np.array([thermal_stress])
            
            # 极端温度事件
            extreme_high = np.sum(temp > 40)
            extreme_low = np.sum(temp < 0)
            physical_features[f"{prefix}_extreme_high_events"] = np.array([extreme_high])
            physical_features[f"{prefix}_extreme_low_events"] = np.array([extreme_low])
        
        # 功率相关的物理特征
        if 'power' in data:
            power = data['power']
            prefix = "power_physical"
            
            # 功率密度分布
            positive_power = power[power > 0]
            negative_power = power[power < 0]
            
            if len(positive_power) > 0:
                physical_features[f"{prefix}_avg_charge_power"] = np.array([np.mean(positive_power)])
                physical_features[f"{prefix}_max_charge_power"] = np.array([np.max(positive_power)])
            
            if len(negative_power) > 0:
                physical_features[f"{prefix}_avg_discharge_power"] = np.array([np.mean(np.abs(negative_power))])
                physical_features[f"{prefix}_max_discharge_power"] = np.array([np.max(np.abs(negative_power))])
            
            # 功率变化率（dP/dt）
            if len(power) > 1:
                power_rate = np.diff(power)
                physical_features[f"{prefix}_max_ramp_up"] = np.array([np.max(power_rate)])
                physical_features[f"{prefix}_max_ramp_down"] = np.array([np.min(power_rate)])
                physical_features[f"{prefix}_avg_ramp_rate"] = np.array([np.mean(np.abs(power_rate))])
        
        # 效率相关特征
        if 'power' in data and 'energy' in data:
            power = data['power']
            energy = data['energy']
            
            # 转换效率
            if len(power) > 1 and len(energy) > 1:
                energy_change = np.diff(energy)
                power_avg = (power[1:] + power[:-1]) / 2
                
                # 只考虑有功率的时段
                active_mask = np.abs(power_avg) > 0.01
                if np.sum(active_mask) > 0:
                    efficiency = np.abs(energy_change[active_mask]) / np.abs(power_avg[active_mask])
                    efficiency = np.clip(efficiency, 0, 1)  # 限制在合理范围
                    physical_features["power_physical_avg_efficiency"] = np.array([np.mean(efficiency)])
                    physical_features["power_physical_min_efficiency"] = np.array([np.min(efficiency)])
        
        return physical_features
    
    def _extract_pattern_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取模式特征"""
        pattern_features = {}
        
        for signal_name, signal_data in data.items():
            prefix = f"{signal_name}_pattern"
            
            # 峰值检测
            peaks, peak_properties = signal.find_peaks(signal_data, 
                                                     height=np.mean(signal_data),
                                                     distance=10)
            valleys, valley_properties = signal.find_peaks(-signal_data,
                                                          height=-np.mean(signal_data),
                                                          distance=10)
            
            pattern_features[f"{prefix}_peak_count"] = np.array([len(peaks)])
            pattern_features[f"{prefix}_valley_count"] = np.array([len(valleys)])
            
            if len(peaks) > 0:
                pattern_features[f"{prefix}_avg_peak_height"] = np.array([np.mean(signal_data[peaks])])
                pattern_features[f"{prefix}_max_peak_height"] = np.array([np.max(signal_data[peaks])])
                
                # 峰值间隔
                if len(peaks) > 1:
                    peak_intervals = np.diff(peaks)
                    pattern_features[f"{prefix}_avg_peak_interval"] = np.array([np.mean(peak_intervals)])
                    pattern_features[f"{prefix}_std_peak_interval"] = np.array([np.std(peak_intervals)])
            
            # 周期性检测（基于自相关）
            if len(signal_data) >= config.pattern_length:
                # 每日模式检测（假设每小时一个数据点）
                if len(signal_data) >= 24:
                    daily_patterns = signal_data[:len(signal_data)//24*24].reshape(-1, 24)
                    if daily_patterns.shape[0] > 1:
                        # 计算日间相似性
                        daily_corr = np.corrcoef(daily_patterns)
                        avg_daily_similarity = np.mean(daily_corr[np.triu_indices_from(daily_corr, k=1)])
                        pattern_features[f"{prefix}_daily_similarity"] = np.array([avg_daily_similarity])
                        
                        # 日模式的标准偏差
                        daily_std = np.std(daily_patterns, axis=0)
                        pattern_features[f"{prefix}_daily_variability"] = np.array([np.mean(daily_std)])
            
            # 趋势模式检测
            if len(signal_data) > 10:
                # 分段线性拟合
                segment_length = min(len(signal_data) // 4, 100)
                if segment_length > 2:
                    segments = [signal_data[i:i+segment_length] for i in range(0, len(signal_data), segment_length)]
                    slopes = []
                    for segment in segments:
                        if len(segment) > 1:
                            x = np.arange(len(segment))
                            slope, _ = np.polyfit(x, segment, 1)
                            slopes.append(slope)
                    
                    if slopes:
                        pattern_features[f"{prefix}_trend_consistency"] = np.array([np.std(slopes)])
                        pattern_features[f"{prefix}_avg_trend_slope"] = np.array([np.mean(slopes)])
            
            # 重复模式检测
            if len(signal_data) >= 48:  # 至少2天的数据
                pattern_scores = []
                for start in range(0, len(signal_data) - 24, 24):
                    if start + 48 <= len(signal_data):
                        pattern1 = signal_data[start:start+24]
                        pattern2 = signal_data[start+24:start+48]
                        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                        if not np.isnan(correlation):
                            pattern_scores.append(correlation)
                
                if pattern_scores:
                    pattern_features[f"{prefix}_pattern_repeatability"] = np.array([np.mean(pattern_scores)])
        
        return pattern_features
    
    def _extract_correlation_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取相关性特征"""
        correlation_features = {}
        
        signal_names = list(data.keys())
        
        # 成对相关性分析
        for i, signal1_name in enumerate(signal_names):
            for j, signal2_name in enumerate(signal_names[i+1:], i+1):
                signal1 = data[signal1_name]
                signal2 = data[signal2_name]
                
                # 确保信号长度一致
                min_length = min(len(signal1), len(signal2))
                signal1_trim = signal1[:min_length]
                signal2_trim = signal2[:min_length]
                
                prefix = f"{signal1_name}_{signal2_name}_correlation"
                
                # 皮尔逊相关系数
                if min_length > 2:
                    pearson_corr = np.corrcoef(signal1_trim, signal2_trim)[0, 1]
                    if not np.isnan(pearson_corr):
                        correlation_features[f"{prefix}_pearson"] = np.array([pearson_corr])
                    
                    # 斯皮尔曼相关系数
                    spearman_corr, _ = stats.spearmanr(signal1_trim, signal2_trim)
                    if not np.isnan(spearman_corr):
                        correlation_features[f"{prefix}_spearman"] = np.array([spearman_corr])
                    
                    # 互相关分析
                    cross_corr = np.correlate(signal1_trim, signal2_trim, mode='full')
                    cross_corr = cross_corr / (np.linalg.norm(signal1_trim) * np.linalg.norm(signal2_trim))
                    max_cross_corr = np.max(cross_corr)
                    correlation_features[f"{prefix}_max_cross_corr"] = np.array([max_cross_corr])
                    
                    # 延迟相关分析
                    max_lag_idx = np.argmax(cross_corr)
                    lag = max_lag_idx - len(signal1_trim) + 1
                    correlation_features[f"{prefix}_optimal_lag"] = np.array([lag])
                    
                    # 互信息（简化版本）
                    mutual_info = self._calculate_mutual_information(signal1_trim, signal2_trim)
                    correlation_features[f"{prefix}_mutual_info"] = np.array([mutual_info])
        
        # 多变量相关性分析
        if len(signal_names) >= 3:
            # 构建相关矩阵
            min_length = min(len(data[name]) for name in signal_names)
            data_matrix = np.array([data[name][:min_length] for name in signal_names]).T
            
            if min_length > len(signal_names):
                # 相关矩阵的特征值
                corr_matrix = np.corrcoef(data_matrix.T)
                eigenvalues = np.linalg.eigvals(corr_matrix)
                eigenvalues = eigenvalues[~np.isnan(eigenvalues)]
                
                if len(eigenvalues) > 0:
                    correlation_features["multivariate_max_eigenvalue"] = np.array([np.max(eigenvalues)])
                    correlation_features["multivariate_min_eigenvalue"] = np.array([np.min(eigenvalues)])
                    correlation_features["multivariate_eigenvalue_ratio"] = np.array([np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)])
                
                # 条件数
                cond_number = np.linalg.cond(corr_matrix)
                if not np.isnan(cond_number) and np.isfinite(cond_number):
                    correlation_features["multivariate_condition_number"] = np.array([cond_number])
        
        return correlation_features
    
    def _extract_trend_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取趋势特征"""
        trend_features = {}
        
        for signal_name, signal_data in data.items():
            prefix = f"{signal_name}_trend"
            
            if len(signal_data) > 2:
                # 线性趋势
                x = np.arange(len(signal_data))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, signal_data)
                
                trend_features[f"{prefix}_linear_slope"] = np.array([slope])
                trend_features[f"{prefix}_linear_intercept"] = np.array([intercept])
                trend_features[f"{prefix}_linear_r_squared"] = np.array([r_value**2])
                trend_features[f"{prefix}_linear_p_value"] = np.array([p_value])
                trend_features[f"{prefix}_linear_std_err"] = np.array([std_err])
                
                # 多项式趋势拟合
                if len(signal_data) > 4:
                    # 二次趋势
                    poly2_coeffs = np.polyfit(x, signal_data, 2)
                    poly2_fitted = np.polyval(poly2_coeffs, x)
                    poly2_r_squared = 1 - np.sum((signal_data - poly2_fitted)**2) / np.sum((signal_data - np.mean(signal_data))**2)
                    
                    trend_features[f"{prefix}_poly2_coeff_2"] = np.array([poly2_coeffs[0]])
                    trend_features[f"{prefix}_poly2_coeff_1"] = np.array([poly2_coeffs[1]])
                    trend_features[f"{prefix}_poly2_coeff_0"] = np.array([poly2_coeffs[2]])
                    trend_features[f"{prefix}_poly2_r_squared"] = np.array([poly2_r_squared])
                
                # 局部趋势分析
                window_size = min(len(signal_data) // 4, 100)
                if window_size > 2:
                    local_slopes = []
                    for i in range(0, len(signal_data) - window_size + 1, window_size // 2):
                        window_data = signal_data[i:i + window_size]
                        window_x = np.arange(len(window_data))
                        if len(window_data) > 2:
                            local_slope, _ = np.polyfit(window_x, window_data, 1)
                            local_slopes.append(local_slope)
                    
                    if local_slopes:
                        trend_features[f"{prefix}_local_slope_mean"] = np.array([np.mean(local_slopes)])
                        trend_features[f"{prefix}_local_slope_std"] = np.array([np.std(local_slopes)])
                        trend_features[f"{prefix}_local_slope_max"] = np.array([np.max(local_slopes)])
                        trend_features[f"{prefix}_local_slope_min"] = np.array([np.min(local_slopes)])
                
                # 趋势变化点检测
                change_points = self._detect_change_points(signal_data)
                trend_features[f"{prefix}_change_points"] = np.array([len(change_points)])
                
                if len(change_points) > 0:
                    # 趋势段分析
                    segments = []
                    prev_point = 0
                    for point in change_points + [len(signal_data)]:
                        if point > prev_point + 2:
                            segment = signal_data[prev_point:point]
                            segment_x = np.arange(len(segment))
                            if len(segment) > 2:
                                segment_slope, _ = np.polyfit(segment_x, segment, 1)
                                segments.append(segment_slope)
                        prev_point = point
                    
                    if segments:
                        trend_features[f"{prefix}_segment_slope_var"] = np.array([np.var(segments)])
                        trend_features[f"{prefix}_avg_segment_slope"] = np.array([np.mean(segments)])
                
                # 季节性趋势
                if len(signal_data) >= 24:  # 至少24小时数据
                    seasonal_trend = self._extract_seasonal_trend(signal_data)
                    trend_features[f"{prefix}_seasonal_strength"] = np.array([seasonal_trend])
        
        return trend_features
    
    def _extract_anomaly_features(self, data: Dict[str, np.ndarray], config: FeatureConfig) -> Dict[str, np.ndarray]:
        """提取异常特征"""
        anomaly_features = {}
        
        for signal_name, signal_data in data.items():
            prefix = f"{signal_name}_anomaly"
            
            # 基于Z-score的异常检测
            z_scores = np.abs(stats.zscore(signal_data))
            anomaly_mask_z = z_scores > config.anomaly_threshold
            
            anomaly_features[f"{prefix}_z_score_anomalies"] = np.array([np.sum(anomaly_mask_z)])
            anomaly_features[f"{prefix}_z_score_anomaly_ratio"] = np.array([np.mean(anomaly_mask_z)])
            anomaly_features[f"{prefix}_max_z_score"] = np.array([np.max(z_scores)])
            
            # 基于IQR的异常检测
            q1 = np.percentile(signal_data, 25)
            q3 = np.percentile(signal_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            anomaly_mask_iqr = (signal_data < lower_bound) | (signal_data > upper_bound)
            anomaly_features[f"{prefix}_iqr_anomalies"] = np.array([np.sum(anomaly_mask_iqr)])
            anomaly_features[f"{prefix}_iqr_anomaly_ratio"] = np.array([np.mean(anomaly_mask_iqr)])
            
            # 基于移动平均的异常检测
            if len(signal_data) >= 10:
                window_size = min(10, len(signal_data) // 3)
                moving_avg = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
                moving_std = np.array([np.std(signal_data[max(0, i-window_size//2):min(len(signal_data), i+window_size//2+1)]) 
                                     for i in range(len(signal_data))])
                
                deviation = np.abs(signal_data - moving_avg)
                anomaly_mask_ma = deviation > (config.anomaly_threshold * moving_std)
                
                anomaly_features[f"{prefix}_ma_anomalies"] = np.array([np.sum(anomaly_mask_ma)])
                anomaly_features[f"{prefix}_ma_anomaly_ratio"] = np.array([np.mean(anomaly_mask_ma)])
                anomaly_features[f"{prefix}_max_ma_deviation"] = np.array([np.max(deviation)])
            
            # 连续异常检测
            anomaly_groups = self._find_anomaly_groups(anomaly_mask_z)
            anomaly_features[f"{prefix}_anomaly_groups"] = np.array([len(anomaly_groups)])
            
            if anomaly_groups:
                group_lengths = [group[1] - group[0] + 1 for group in anomaly_groups]
                anomaly_features[f"{prefix}_max_anomaly_group_length"] = np.array([max(group_lengths)])
                anomaly_features[f"{prefix}_avg_anomaly_group_length"] = np.array([np.mean(group_lengths)])
            
            # 基于梯度的异常检测
            if len(signal_data) > 1:
                gradient = np.abs(np.diff(signal_data))
                gradient_threshold = np.percentile(gradient, 95)
                sudden_changes = np.sum(gradient > gradient_threshold)
                
                anomaly_features[f"{prefix}_sudden_changes"] = np.array([sudden_changes])
                anomaly_features[f"{prefix}_max_gradient"] = np.array([np.max(gradient)])
        
        return anomaly_features
    
    def _calculate_mutual_information(self, x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
        """计算互信息（简化版本）"""
        try:
            # 创建联合直方图
            hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
            
            # 计算边际分布
            hist_x = np.sum(hist_2d, axis=1)
            hist_y = np.sum(hist_2d, axis=0)
            
            # 归一化为概率
            p_xy = hist_2d / np.sum(hist_2d)
            p_x = hist_x / np.sum(hist_x)
            p_y = hist_y / np.sum(hist_y)
            
            # 计算互信息
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return mi
        except:
            return 0.0
    
    def _create_windows(self, signal: np.ndarray, window_size: int, overlap_ratio: float) -> List[np.ndarray]:
        """创建滑动窗口"""
        step_size = int(window_size * (1 - overlap_ratio))
        windows = []
        
        for i in range(0, len(signal) - window_size + 1, step_size):
            windows.append(signal[i:i + window_size])
        
        return windows
    
    def _detect_change_points(self, signal: np.ndarray, min_segment_length: int = 10) -> List[int]:
        """检测趋势变化点"""
        change_points = []
        
        if len(signal) < min_segment_length * 2:
            return change_points
        
        # 简化的变化点检测：基于斜率变化
        window_size = min_segment_length
        slopes = []
        
        for i in range(window_size, len(signal) - window_size):
            # 计算左侧窗口斜率
            left_window = signal[i - window_size:i]
            left_x = np.arange(len(left_window))
            left_slope, _ = np.polyfit(left_x, left_window, 1)
            
            # 计算右侧窗口斜率
            right_window = signal[i:i + window_size]
            right_x = np.arange(len(right_window))
            right_slope, _ = np.polyfit(right_x, right_window, 1)
            
            # 斜率差异
            slope_diff = abs(right_slope - left_slope)
            slopes.append((i, slope_diff))
        
        if slopes:
            # 找到斜率变化最大的点
            slopes.sort(key=lambda x: x[1], reverse=True)
            threshold = np.percentile([s[1] for s in slopes], 90)
            
            for i, slope_diff in slopes:
                if slope_diff > threshold:
                    # 检查是否与已有变化点太近
                    if not any(abs(i - cp) < min_segment_length for cp in change_points):
                        change_points.append(i)
        
        return sorted(change_points)
    
    def _extract_seasonal_trend(self, signal: np.ndarray) -> float:
        """提取季节性趋势强度"""
        if len(signal) < 24:
            return 0.0
        
        # 假设24小时为一个周期
        period = 24
        num_periods = len(signal) // period
        
        if num_periods < 2:
            return 0.0
        
        # 重塑为周期矩阵
        seasonal_data = signal[:num_periods * period].reshape(num_periods, period)
        
        # 计算每个时间点的方差
        temporal_variance = np.var(seasonal_data, axis=0)
        
        # 计算总方差
        total_variance = np.var(signal[:num_periods * period])
        
        # 季节性强度 = 1 - (平均时间方差 / 总方差)
        seasonal_strength = 1.0 - np.mean(temporal_variance) / (total_variance + 1e-10)
        
        return max(0.0, seasonal_strength)
    
    def _find_anomaly_groups(self, anomaly_mask: np.ndarray) -> List[Tuple[int, int]]:
        """查找连续异常组"""
        groups = []
        in_group = False
        start_idx = 0
        
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly and not in_group:
                start_idx = i
                in_group = True
            elif not is_anomaly and in_group:
                groups.append((start_idx, i - 1))
                in_group = False
        
        # 处理最后一个组
        if in_group:
            groups.append((start_idx, len(anomaly_mask) - 1))
        
        return groups
    
    def _calculate_feature_importance(self, features: ExtractedFeatures) -> Dict[str, float]:
        """计算特征重要性"""
        importance = {}
        
        # 合并所有特征
        all_features = {}
        all_features.update(features.temporal_features)
        all_features.update(features.frequency_features)
        all_features.update(features.statistical_features)
        all_features.update(features.physical_features)
        all_features.update(features.pattern_features)
        all_features.update(features.correlation_features)
        all_features.update(features.trend_features)
        all_features.update(features.anomaly_features)
        
        for feature_name, feature_value in all_features.items():
            # 基于特征名称和预定义权重计算重要性
            base_importance = 0.5
            
            for key_word, weight in self.feature_weights.items():
                if key_word in feature_name.lower():
                    base_importance = max(base_importance, weight)
            
            # 基于特征值的变异性调整重要性
            if len(feature_value) > 0 and not np.isnan(feature_value[0]):
                value_factor = min(1.0, abs(feature_value[0]) / (abs(feature_value[0]) + 1.0))
                importance[feature_name] = base_importance * (0.5 + 0.5 * value_factor)
            else:
                importance[feature_name] = base_importance * 0.1
        
        return importance
    
    def _assess_feature_quality(self, features: ExtractedFeatures) -> float:
        """评估特征质量"""
        quality_factors = []
        
        # 合并所有特征
        all_features = {}
        all_features.update(features.temporal_features)
        all_features.update(features.frequency_features)
        all_features.update(features.statistical_features)
        all_features.update(features.physical_features)
        all_features.update(features.pattern_features)
        all_features.update(features.correlation_features)
        all_features.update(features.trend_features)
        all_features.update(features.anomaly_features)
        
        if len(all_features) == 0:
            return 0.0
        
        # 1. 特征完整性
        valid_features = sum(1 for v in all_features.values() 
                           if len(v) > 0 and not np.isnan(v[0]) and np.isfinite(v[0]))
        completeness = valid_features / len(all_features)
        quality_factors.append(completeness)
        
        # 2. 特征多样性
        feature_types_present = len([ft for ft in features.feature_types 
                                   if any(ft.value in fname for fname in all_features.keys())])
        diversity = feature_types_present / len(FeatureType)
        quality_factors.append(diversity)
        
        # 3. 数值稳定性
        finite_values = sum(1 for v in all_features.values() 
                          if len(v) > 0 and np.isfinite(v[0]))
        stability = finite_values / len(all_features)
        quality_factors.append(stability)
        
        # 4. 信息内容（基于方差）
        if valid_features > 1:
            valid_values = [v[0] for v in all_features.values() 
                          if len(v) > 0 and np.isfinite(v[0])]
            if len(valid_values) > 1:
                information_content = min(1.0, np.std(valid_values) / (np.mean(np.abs(valid_values)) + 1e-10))
                quality_factors.append(information_content)
        
        return np.mean(quality_factors)
    
    def _update_extraction_stats(self, num_feature_types: int, extraction_time: float):
        """更新提取统计"""
        self.extraction_stats['total_extractions'] += 1
        self.extraction_stats['total_features'] += num_feature_types
        self.extraction_stats['extraction_time'] += extraction_time
    
    def analyze_feature_importance(self, features: ExtractedFeatures) -> Dict[str, Any]:
        """分析特征重要性"""
        # 合并所有特征
        all_features = {}
        all_features.update(features.temporal_features)
        all_features.update(features.frequency_features)
        all_features.update(features.statistical_features)
        all_features.update(features.physical_features)
        all_features.update(features.pattern_features)
        all_features.update(features.correlation_features)
        all_features.update(features.trend_features)
        all_features.update(features.anomaly_features)
        
        importance_analysis = {
            'top_features': {},
            'feature_type_importance': {},
            'redundancy_analysis': {},
            'selection_recommendations': []
        }
        
        # 特征重要性排序
        sorted_importance = sorted(features.feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)
        importance_analysis['top_features'] = dict(sorted_importance[:20])  # 前20个重要特征
        
        # 按特征类型分组重要性
        for feature_type in FeatureType:
            type_features = [name for name in all_features.keys() if feature_type.value in name]
            if type_features:
                type_importance = np.mean([features.feature_importance.get(name, 0) for name in type_features])
                importance_analysis['feature_type_importance'][feature_type.value] = type_importance
        
        # 冗余性分析
        if len(all_features) > 1:
            feature_values = np.array([v[0] if len(v) > 0 else 0 for v in all_features.values()])
            feature_names = list(all_features.keys())
            
            # 计算特征间相关性
            corr_matrix = np.corrcoef(feature_values.reshape(1, -1) if len(feature_values.shape) == 1 
                                    else feature_values)
            
            # 找到高相关的特征对
            high_corr_pairs = []
            if corr_matrix.shape[0] > 1:
                for i in range(len(feature_names)):
                    for j in range(i+1, len(feature_names)):
                        if abs(corr_matrix[i, j]) > 0.9:
                            high_corr_pairs.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
            
            importance_analysis['redundancy_analysis']['high_correlation_pairs'] = high_corr_pairs
        
        # 特征选择建议
        if sorted_importance:
            # 基于重要性的建议
            high_importance_features = [name for name, imp in sorted_importance if imp > 0.7]
            medium_importance_features = [name for name, imp in sorted_importance if 0.4 < imp <= 0.7]
            low_importance_features = [name for name, imp in sorted_importance if imp <= 0.4]
            
            importance_analysis['selection_recommendations'] = {
                'must_include': high_importance_features[:10],
                'consider_include': medium_importance_features[:15],
                'optional': low_importance_features[:5],
                'total_recommended': len(high_importance_features[:10]) + len(medium_importance_features[:15])
            }
        
        return importance_analysis
    
    def export_features(self, features: ExtractedFeatures, file_path: str, format: str = 'json'):
        """导出特征"""
        try:
            if format.lower() == 'json':
                # 合并所有特征
                all_features = {}
                all_features.update({f"temporal_{k}": v.tolist() for k, v in features.temporal_features.items()})
                all_features.update({f"frequency_{k}": v.tolist() for k, v in features.frequency_features.items()})
                all_features.update({f"statistical_{k}": v.tolist() for k, v in features.statistical_features.items()})
                all_features.update({f"physical_{k}": v.tolist() for k, v in features.physical_features.items()})
                all_features.update({f"pattern_{k}": v.tolist() for k, v in features.pattern_features.items()})
                all_features.update({f"correlation_{k}": v.tolist() for k, v in features.correlation_features.items()})
                all_features.update({f"trend_{k}": v.tolist() for k, v in features.trend_features.items()})
                all_features.update({f"anomaly_{k}": v.tolist() for k, v in features.anomaly_features.items()})
                
                export_data = {
                    'feature_id': features.feature_id,
                    'source_data_id': features.source_data_id,
                    'feature_types': [ft.value for ft in features.feature_types],
                    'features': all_features,
                    'feature_importance': features.feature_importance,
                    'quality_score': features.quality_score,
                    'extraction_time': features.extraction_time,
                    'config': {
                        'window_size': features.config.window_size,
                        'overlap_ratio': features.config.overlap_ratio,
                        'sampling_rate': features.config.sampling_rate
                    }
                }
                
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format.lower() == 'csv':
                # 创建特征DataFrame
                all_features = {}
                all_features.update(features.temporal_features)
                all_features.update(features.frequency_features)
                all_features.update(features.statistical_features)
                all_features.update(features.physical_features)
                all_features.update(features.pattern_features)
                all_features.update(features.correlation_features)
                all_features.update(features.trend_features)
                all_features.update(features.anomaly_features)
                
                # 转换为单行DataFrame
                feature_dict = {}
                for name, value in all_features.items():
                    if len(value) > 0:
                        feature_dict[name] = value[0]
                    else:
                        feature_dict[name] = np.nan
                
                df = pd.DataFrame([feature_dict])
                df.to_csv(file_path, index=False)
            
            print(f"✅ 特征已导出: {file_path}")
            
        except Exception as e:
            print(f"❌ 特征导出失败: {str(e)}")
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """获取提取统计信息"""
        stats = self.extraction_stats.copy()
        
        if stats['total_extractions'] > 0:
            stats['avg_features_per_extraction'] = stats['total_features'] / stats['total_extractions']
            stats['avg_extraction_time'] = stats['extraction_time'] / stats['total_extractions']
        else:
            stats['avg_features_per_extraction'] = 0
            stats['avg_extraction_time'] = 0
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"FeatureExtractor({self.extractor_id}): "
                f"提取次数={self.extraction_stats['total_extractions']}, "
                f"总特征={self.extraction_stats['total_features']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"FeatureExtractor(extractor_id='{self.extractor_id}', "
                f"feature_types={len(self.extraction_methods)}, "
                f"total_extractions={self.extraction_stats['total_extractions']})")
