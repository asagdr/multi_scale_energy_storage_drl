import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config.training_config import UpperLayerConfig

@dataclass
class ParetoSolution:
    """帕累托解数据结构"""
    objectives: np.ndarray                  # 目标值向量
    weights: np.ndarray                     # 权重向量
    action: np.ndarray                      # 对应动作
    state: Optional[np.ndarray] = None      # 对应状态
    dominance_count: int = 0                # 被支配次数
    dominated_solutions: List[int] = field(default_factory=list)  # 支配的解的索引
    rank: int = 0                          # 帕累托等级
    crowding_distance: float = 0.0         # 拥挤距离
    
class ParetoFront:
    """帕累托前沿管理"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.solutions: List[ParetoSolution] = []
        self.fronts: List[List[int]] = []  # 各等级的解索引
        
    def add_solution(self, solution: ParetoSolution) -> bool:
        """添加解到帕累托集合"""
        # 检查是否被现有解支配
        is_dominated = False
        dominated_indices = []
        
        for i, existing_solution in enumerate(self.solutions):
            dominance = self._check_dominance(solution.objectives, existing_solution.objectives)
            
            if dominance == -1:  # 新解被支配
                is_dominated = True
                break
            elif dominance == 1:  # 新解支配现有解
                dominated_indices.append(i)
        
        if is_dominated:
            return False
        
        # 移除被新解支配的解
        for idx in sorted(dominated_indices, reverse=True):
            self.solutions.pop(idx)
        
        # 添加新解
        self.solutions.append(solution)
        
        # 如果超过最大大小，移除拥挤度最高的解
        if len(self.solutions) > self.max_size:
            self._maintain_diversity()
        
        return True
    
    def _check_dominance(self, obj1: np.ndarray, obj2: np.ndarray) -> int:
        """
        检查支配关系
        
        Returns:
            1: obj1支配obj2
            -1: obj2支配obj1  
            0: 无支配关系
        """
        better = np.sum(obj1 >= obj2)
        worse = np.sum(obj1 <= obj2)
        
        if better == len(obj1) and np.any(obj1 > obj2):
            return 1  # obj1支配obj2
        elif worse == len(obj1) and np.any(obj1 < obj2):
            return -1  # obj2支配obj1
        else:
            return 0  # 无支配关系
    
    def _maintain_diversity(self):
        """维护解集多样性"""
        if len(self.solutions) <= self.max_size:
            return
        
        # 计算拥挤距离
        self._calculate_crowding_distances()
        
        # 按拥挤距离排序，移除距离最小的解
        self.solutions.sort(key=lambda x: x.crowding_distance, reverse=True)
        self.solutions = self.solutions[:self.max_size]
    
    def _calculate_crowding_distances(self):
        """计算拥挤距离"""
        n_solutions = len(self.solutions)
        if n_solutions <= 2:
            for solution in self.solutions:
                solution.crowding_distance = float('inf')
            return
        
        # 重置拥挤距离
        for solution in self.solutions:
            solution.crowding_distance = 0.0
        
        # 为每个目标计算拥挤距离
        objectives_matrix = np.array([sol.objectives for sol in self.solutions])
        n_objectives = objectives_matrix.shape[1]
        
        for obj_idx in range(n_objectives):
            # 按第obj_idx个目标排序
            sorted_indices = np.argsort(objectives_matrix[:, obj_idx])
            
            # 边界解设为无穷大
            self.solutions[sorted_indices[0]].crowding_distance = float('inf')
            self.solutions[sorted_indices[-1]].crowding_distance = float('inf')
            
            # 计算中间解的拥挤距离
            obj_range = objectives_matrix[sorted_indices[-1], obj_idx] - objectives_matrix[sorted_indices[0], obj_idx]
            
            if obj_range > 0:
                for i in range(1, n_solutions - 1):
                    distance = (objectives_matrix[sorted_indices[i+1], obj_idx] - 
                               objectives_matrix[sorted_indices[i-1], obj_idx]) / obj_range
                    self.solutions[sorted_indices[i]].crowding_distance += distance
    
    def get_best_solutions(self, n: int = 10) -> List[ParetoSolution]:
        """获取最佳解"""
        if len(self.solutions) <= n:
            return self.solutions.copy()
        
        # 计算拥挤距离并排序
        self._calculate_crowding_distances()
        sorted_solutions = sorted(self.solutions, key=lambda x: x.crowding_distance, reverse=True)
        
        return sorted_solutions[:n]

class WeightVectorGenerator:
    """权重向量生成器"""
    
    def __init__(self, n_objectives: int = 4):
        self.n_objectives = n_objectives
    
    def generate_uniform_weights(self, n_vectors: int) -> np.ndarray:
        """生成均匀分布的权重向量"""
        if self.n_objectives == 2:
            weights = np.zeros((n_vectors, 2))
            for i in range(n_vectors):
                w1 = i / (n_vectors - 1)
                weights[i] = [w1, 1 - w1]
            return weights
        
        elif self.n_objectives == 3:
            weights = []
            n_per_dim = int(np.ceil(n_vectors ** (1/3)))
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    for k in range(n_per_dim):
                        w = np.array([i, j, k], dtype=float)
                        w = w / np.sum(w) if np.sum(w) > 0 else np.array([1/3, 1/3, 1/3])
                        weights.append(w)
                        if len(weights) >= n_vectors:
                            return np.array(weights[:n_vectors])
            return np.array(weights)
        
        else:
            # 对于4维或更高维，使用随机生成+归一化
            weights = np.random.dirichlet(np.ones(self.n_objectives), n_vectors)
            return weights
    
    def generate_corner_weights(self) -> np.ndarray:
        """生成角点权重向量"""
        weights = np.eye(self.n_objectives)
        return weights
    
    def generate_adaptive_weights(self, 
                                current_performance: np.ndarray,
                                n_vectors: int = 20) -> np.ndarray:
        """根据当前性能生成自适应权重"""
        # 基于性能差距生成权重
        performance_gaps = 1.0 - current_performance  # 性能差距
        
        # 归一化差距
        if np.sum(performance_gaps) > 0:
            base_weights = performance_gaps / np.sum(performance_gaps)
        else:
            base_weights = np.ones(self.n_objectives) / self.n_objectives
        
        # 生成围绕基础权重的变体
        weights = []
        for i in range(n_vectors):
            # 添加噪声
            noise = np.random.normal(0, 0.1, self.n_objectives)
            variant = base_weights + noise
            variant = np.maximum(variant, 0.01)  # 最小权重0.01
            variant = variant / np.sum(variant)  # 归一化
            weights.append(variant)
        
        return np.array(weights)

class ParetoOptimizer(nn.Module):
    """
    帕累托优化器
    实现多目标优化的帕累托前沿搜索和权重向量自适应调整
    """
    
    def __init__(self,
                 config: UpperLayerConfig,
                 n_objectives: int = 4,
                 optimizer_id: str = "ParetoOptimizer_001"):
        """
        初始化帕累托优化器
        
        Args:
            config: 上层配置
            n_objectives: 目标数量
            optimizer_id: 优化器ID
        """
        super(ParetoOptimizer, self).__init__()
        
        self.config = config
        self.n_objectives = n_objectives
        self.optimizer_id = optimizer_id
        
        # === 帕累托前沿管理 ===
        self.pareto_front = ParetoFront(max_size=1000)
        self.weight_generator = WeightVectorGenerator(n_objectives)
        
        # === 权重向量集合 ===
        self.weight_vectors = self.weight_generator.generate_uniform_weights(50)
        self.current_weight_index = 0
        
        # === 优化统计 ===
        self.optimization_history: List[Dict] = []
        self.objective_ranges = np.ones((n_objectives, 2))  # [min, max] for each objective
        self.objective_ranges[:, 0] = 0.0  # min = 0
        
        # === 自适应参数 ===
        self.adaptation_frequency = 100  # 权重适应频率
        self.diversity_threshold = 0.1   # 多样性阈值
        
        print(f"✅ 帕累托优化器初始化完成: {optimizer_id}")
        print(f"   目标数量: {n_objectives}, 权重向量数: {len(self.weight_vectors)}")
    
    def add_solution(self, 
                    objectives: np.ndarray,
                    weights: np.ndarray,
                    action: np.ndarray,
                    state: Optional[np.ndarray] = None) -> bool:
        """
        添加解到帕累托前沿
        
        Args:
            objectives: 目标值向量
            weights: 权重向量
            action: 动作向量
            state: 状态向量
            
        Returns:
            是否成功添加
        """
        solution = ParetoSolution(
            objectives=objectives.copy(),
            weights=weights.copy(),
            action=action.copy(),
            state=state.copy() if state is not None else None
        )
        
        success = self.pareto_front.add_solution(solution)
        
        # 更新目标范围
        self._update_objective_ranges(objectives)
        
        # 记录历史
        self._record_optimization_step(solution, success)
        
        return success
    
    def _update_objective_ranges(self, objectives: np.ndarray):
        """更新目标值范围"""
        for i, obj_value in enumerate(objectives):
            self.objective_ranges[i, 0] = min(self.objective_ranges[i, 0], obj_value)
            self.objective_ranges[i, 1] = max(self.objective_ranges[i, 1], obj_value)
    
    def _record_optimization_step(self, solution: ParetoSolution, success: bool):
        """记录优化步骤"""
        record = {
            'step': len(self.optimization_history),
            'objectives': solution.objectives.tolist(),
            'weights': solution.weights.tolist(),
            'added_to_front': success,
            'front_size': len(self.pareto_front.solutions),
            'hypervolume': self.calculate_hypervolume() if len(self.pareto_front.solutions) > 0 else 0.0
        }
        
        self.optimization_history.append(record)
        
        # 维护历史长度
        if len(self.optimization_history) > 10000:
            self.optimization_history.pop(0)
    
    def get_next_weight_vector(self) -> np.ndarray:
        """获取下一个权重向量"""
        if len(self.optimization_history) % self.adaptation_frequency == 0:
            # 定期更新权重向量集合
            self._adapt_weight_vectors()
        
        # 循环选择权重向量
        weight = self.weight_vectors[self.current_weight_index]
        self.current_weight_index = (self.current_weight_index + 1) % len(self.weight_vectors)
        
        return weight.copy()
    
    def _adapt_weight_vectors(self):
        """自适应调整权重向量"""
        if len(self.pareto_front.solutions) < 5:
            return
        
        # 分析当前帕累托前沿的分布
        objectives_matrix = np.array([sol.objectives for sol in self.pareto_front.solutions])
        
        # 检查目标空间的覆盖情况
        coverage_gaps = self._analyze_coverage_gaps(objectives_matrix)
        
        if np.max(coverage_gaps) > self.diversity_threshold:
            # 生成新的权重向量以改善覆盖
            new_weights = self._generate_gap_filling_weights(coverage_gaps)
            
            # 替换部分现有权重向量
            n_replace = min(len(new_weights), len(self.weight_vectors) // 4)
            self.weight_vectors[-n_replace:] = new_weights[:n_replace]
            
            print(f"🔄 权重向量已自适应调整，替换了 {n_replace} 个向量")
    
    def _analyze_coverage_gaps(self, objectives_matrix: np.ndarray) -> np.ndarray:
        """分析目标空间覆盖差距"""
        if len(objectives_matrix) < 2:
            return np.zeros(self.n_objectives)
        
        # 归一化目标值
        normalized_objectives = np.zeros_like(objectives_matrix)
        for i in range(self.n_objectives):
            obj_range = self.objective_ranges[i, 1] - self.objective_ranges[i, 0]
            if obj_range > 0:
                normalized_objectives[:, i] = (objectives_matrix[:, i] - self.objective_ranges[i, 0]) / obj_range
            else:
                normalized_objectives[:, i] = objectives_matrix[:, i]
        
        # 使用聚类分析覆盖情况
        n_clusters = min(10, len(objectives_matrix))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_objectives)
            
            # 计算每个目标维度的覆盖方差
            coverage_gaps = np.zeros(self.n_objectives)
            for i in range(self.n_objectives):
                cluster_means = []
                for cluster_id in range(n_clusters):
                    cluster_mask = cluster_labels == cluster_id
                    if np.any(cluster_mask):
                        cluster_mean = np.mean(normalized_objectives[cluster_mask, i])
                        cluster_means.append(cluster_mean)
                
                if len(cluster_means) > 1:
                    coverage_gaps[i] = 1.0 - np.std(cluster_means)  # 标准差越小，gap越大
                
            return coverage_gaps
        else:
            return np.zeros(self.n_objectives)
    
    def _generate_gap_filling_weights(self, coverage_gaps: np.ndarray) -> np.ndarray:
        """生成填补覆盖差距的权重向量"""
        # 基于覆盖差距生成权重
        gap_weights = coverage_gaps / np.sum(coverage_gaps) if np.sum(coverage_gaps) > 0 else np.ones(self.n_objectives) / self.n_objectives
        
        # 生成围绕差距权重的变体
        n_new_weights = 10
        new_weights = []
        
        for i in range(n_new_weights):
            # 添加随机扰动
            noise = np.random.normal(0, 0.15, self.n_objectives)
            weight = gap_weights + noise
            weight = np.maximum(weight, 0.01)  # 最小权重
            weight = weight / np.sum(weight)   # 归一化
            new_weights.append(weight)
        
        return np.array(new_weights)
    
    def calculate_hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """计算超体积指标"""
        if len(self.pareto_front.solutions) == 0:
            return 0.0
        
        if reference_point is None:
            # 使用目标空间的最差点作为参考点
            reference_point = self.objective_ranges[:, 0] - 0.1
        
        objectives_matrix = np.array([sol.objectives for sol in self.pareto_front.solutions])
        
        # 简化的超体积计算（适用于低维目标空间）
        if self.n_objectives <= 4:
            return self._calculate_hypervolume_simple(objectives_matrix, reference_point)
        else:
            # 对于高维，使用近似方法
            return self._calculate_hypervolume_approximate(objectives_matrix, reference_point)
    
    def _calculate_hypervolume_simple(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """简单超体积计算"""
        # 对于2D情况的精确计算
        if self.n_objectives == 2:
            # 按第一个目标排序
            sorted_indices = np.argsort(objectives[:, 0])
            sorted_objectives = objectives[sorted_indices]
            
            hypervolume = 0.0
            prev_x = ref_point[0]
            
            for i, point in enumerate(sorted_objectives):
                if i == 0 or point[1] > sorted_objectives[i-1, 1]:
                    width = point[0] - prev_x
                    height = point[1] - ref_point[1]
                    hypervolume += width * height
                    prev_x = point[0]
            
            return max(0.0, hypervolume)
        
        else:
            # 对于3D和4D，使用蒙特卡洛近似
            return self._calculate_hypervolume_approximate(objectives, ref_point)
    
    def _calculate_hypervolume_approximate(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """近似超体积计算"""
        n_samples = 10000
        
        # 定义采样边界
        max_point = np.max(objectives, axis=0)
        
        # 随机采样
        samples = np.random.uniform(
            low=ref_point,
            high=max_point,
            size=(n_samples, self.n_objectives)
        )
        
        # 检查每个样本是否被至少一个解支配
        dominated_count = 0
        for sample in samples:
            for objective in objectives:
                if np.all(objective >= sample):
                    dominated_count += 1
                    break
        
        # 计算体积
        total_volume = np.prod(max_point - ref_point)
        hypervolume = (dominated_count / n_samples) * total_volume
        
        return hypervolume
    
    def get_diverse_solutions(self, n_solutions: int = 10) -> List[ParetoSolution]:
        """获取多样化的解集"""
        if len(self.pareto_front.solutions) <= n_solutions:
            return self.pareto_front.solutions.copy()
        
        # 使用k-means聚类选择多样化解
        objectives_matrix = np.array([sol.objectives for sol in self.pareto_front.solutions])
        
        # 归一化
        normalized_objectives = np.zeros_like(objectives_matrix)
        for i in range(self.n_objectives):
            obj_range = self.objective_ranges[i, 1] - self.objective_ranges[i, 0]
            if obj_range > 0:
                normalized_objectives[:, i] = (objectives_matrix[:, i] - self.objective_ranges[i, 0]) / obj_range
            else:
                normalized_objectives[:, i] = objectives_matrix[:, i]
        
        # 聚类
        kmeans = KMeans(n_clusters=n_solutions, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_objectives)
        
        # 从每个聚类中选择最好的解
        diverse_solutions = []
        for cluster_id in range(n_solutions):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # 选择该聚类中拥挤距离最大的解
                best_idx = cluster_indices[0]
                best_distance = self.pareto_front.solutions[best_idx].crowding_distance
                
                for idx in cluster_indices:
                    if self.pareto_front.solutions[idx].crowding_distance > best_distance:
                        best_idx = idx
                        best_distance = self.pareto_front.solutions[idx].crowding_distance
                
                diverse_solutions.append(self.pareto_front.solutions[best_idx])
        
        return diverse_solutions
    
    def recommend_weight_vector(self, 
                              current_performance: np.ndarray,
                              priority_objectives: Optional[List[int]] = None) -> np.ndarray:
        """
        推荐权重向量
        
        Args:
            current_performance: 当前各目标性能 [0,1]
            priority_objectives: 优先目标索引列表
            
        Returns:
            推荐的权重向量
        """
        # 基于性能差距的基础权重
        performance_gaps = 1.0 - current_performance
        base_weights = performance_gaps / np.sum(performance_gaps) if np.sum(performance_gaps) > 0 else np.ones(self.n_objectives) / self.n_objectives
        
        # 如果有优先目标，增加其权重
        if priority_objectives:
            priority_bonus = 0.2 / len(priority_objectives)
            for obj_idx in priority_objectives:
                if 0 <= obj_idx < self.n_objectives:
                    base_weights[obj_idx] += priority_bonus
        
        # 归一化
        base_weights = base_weights / np.sum(base_weights)
        
        # 从帕累托前沿中寻找最相似的权重向量
        if len(self.pareto_front.solutions) > 0:
            solution_weights = np.array([sol.weights for sol in self.pareto_front.solutions])
            
            # 计算权重相似度
            similarities = []
            for sol_weight in solution_weights:
                similarity = 1.0 - np.linalg.norm(base_weights - sol_weight)
                similarities.append(similarity)
            
            # 选择最相似的权重
            best_idx = np.argmax(similarities)
            if similarities[best_idx] > 0.7:  # 相似度阈值
                return solution_weights[best_idx].copy()
        
        return base_weights
    
    def analyze_pareto_front(self) -> Dict[str, Any]:
        """分析帕累托前沿"""
        if len(self.pareto_front.solutions) == 0:
            return {'error': 'Empty Pareto front'}
        
        objectives_matrix = np.array([sol.objectives for sol in self.pareto_front.solutions])
        weights_matrix = np.array([sol.weights for sol in self.pareto_front.solutions])
        
        analysis = {
            'front_size': len(self.pareto_front.solutions),
            'hypervolume': self.calculate_hypervolume(),
            
            'objective_statistics': {
                f'objective_{i}': {
                    'mean': float(np.mean(objectives_matrix[:, i])),
                    'std': float(np.std(objectives_matrix[:, i])),
                    'min': float(np.min(objectives_matrix[:, i])),
                    'max': float(np.max(objectives_matrix[:, i]))
                } for i in range(self.n_objectives)
            },
            
            'weight_statistics': {
                f'weight_{i}': {
                    'mean': float(np.mean(weights_matrix[:, i])),
                    'std': float(np.std(weights_matrix[:, i]))
                } for i in range(self.n_objectives)
            },
            
            'diversity_metrics': {
                'objective_spread': float(np.mean(np.std(objectives_matrix, axis=0))),
                'weight_spread': float(np.mean(np.std(weights_matrix, axis=0))),
                'coverage_score': self._calculate_coverage_score(objectives_matrix)
            }
        }
        
        return analysis
    
    def _calculate_coverage_score(self, objectives_matrix: np.ndarray) -> float:
        """计算覆盖评分"""
        if len(objectives_matrix) < 2:
            return 0.0
        
        # 计算解之间的平均距离
        distances = []
        for i in range(len(objectives_matrix)):
            for j in range(i + 1, len(objectives_matrix)):
                distance = np.linalg.norm(objectives_matrix[i] - objectives_matrix[j])
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            max_possible_distance = np.linalg.norm(self.objective_ranges[:, 1] - self.objective_ranges[:, 0])
            coverage_score = min(1.0, avg_distance / max_possible_distance) if max_possible_distance > 0 else 0.0
        else:
            coverage_score = 0.0
        
        return coverage_score
    
    def visualize_pareto_front(self, save_path: Optional[str] = None):
        """可视化帕累托前沿"""
        if len(self.pareto_front.solutions) == 0:
            print("⚠️ 帕累托前沿为空，无法可视化")
            return
        
        objectives_matrix = np.array([sol.objectives for sol in self.pareto_front.solutions])
        
        if self.n_objectives == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(objectives_matrix[:, 0], objectives_matrix[:, 1], alpha=0.7)
            plt.xlabel('Objective 1 (SOC Balance)')
            plt.ylabel('Objective 2 (Temperature Balance)')
            plt.title('Pareto Front - 2D')
            plt.grid(True, alpha=0.3)
            
        elif self.n_objectives == 3:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(objectives_matrix[:, 0], objectives_matrix[:, 1], objectives_matrix[:, 2], alpha=0.7)
            ax.set_xlabel('Objective 1 (SOC Balance)')
            ax.set_ylabel('Objective 2 (Temperature Balance)')
            ax.set_zlabel('Objective 3 (Lifetime Cost)')
            ax.set_title('Pareto Front - 3D')
            
        elif self.n_objectives >= 4:
            # 对于4维或更高维，使用平行坐标图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            objective_names = ['SOC Balance', 'Temp Balance', 'Lifetime Cost', 'Constraint Satisfaction']
            
            for i in range(min(4, self.n_objectives)):
                for j in range(i + 1, min(4, self.n_objectives)):
                    if i < 4 and j < 4:
                        axes[i].scatter(objectives_matrix[:, i], objectives_matrix[:, j], alpha=0.7)
                        axes[i].set_xlabel(objective_names[i])
                        axes[i].set_ylabel(objective_names[j])
                        axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.suptitle('Pareto Front - Multi-objective Projections', y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 帕累托前沿图已保存: {save_path}")
        
        plt.show()
    
    def get_optimizer_statistics(self) -> Dict[str, Any]:
        """获取优化器统计信息"""
        stats = {
            'optimizer_id': self.optimizer_id,
            'optimization_steps': len(self.optimization_history),
            'pareto_front_analysis': self.analyze_pareto_front(),
            'weight_vectors_count': len(self.weight_vectors),
            'current_weight_index': self.current_weight_index,
            
            'performance_trends': {
                'hypervolume_trend': [record['hypervolume'] for record in self.optimization_history[-100:]],
                'front_size_trend': [record['front_size'] for record in self.optimization_history[-100:]],
                'success_rate': np.mean([record['added_to_front'] for record in self.optimization_history[-100:]]) if len(self.optimization_history) >= 100 else 0.0
            }
        }
        
        return stats
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"ParetoOptimizer({self.optimizer_id}): "
                f"objectives={self.n_objectives}, front_size={len(self.pareto_front.solutions)}, "
                f"hypervolume={self.calculate_hypervolume():.4f}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"ParetoOptimizer(optimizer_id='{self.optimizer_id}', "
                f"n_objectives={self.n_objectives}, "
                f"front_size={len(self.pareto_front.solutions)})")
