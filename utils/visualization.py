import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class PlotType(Enum):
    """图表类型枚举"""
    LINE = "line"                      # 线图
    SCATTER = "scatter"                # 散点图
    BAR = "bar"                        # 柱状图
    HISTOGRAM = "histogram"            # 直方图
    HEATMAP = "heatmap"               # 热力图
    BOX = "box"                       # 箱线图
    VIOLIN = "violin"                 # 小提琴图
    SURFACE = "surface"               # 三维表面图
    CONTOUR = "contour"               # 等高线图
    RADAR = "radar"                   # 雷达图
    SANKEY = "sankey"                 # 桑基图
    TREEMAP = "treemap"               # 树状图
    PARALLEL = "parallel"             # 平行坐标图
    GAUGE = "gauge"                   # 仪表盘
    CANDLESTICK = "candlestick"       # K线图

@dataclass
class PlotConfig:
    """图表配置"""
    plot_type: PlotType
    title: str = ""
    width: int = 800
    height: int = 600
    
    # 样式配置
    theme: str = "plotly_white"        # plotly_white, plotly_dark, seaborn
    color_palette: str = "viridis"     # viridis, plasma, inferno, Set1
    font_size: int = 12
    
    # 坐标轴配置
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    
    # 图例配置
    show_legend: bool = True
    legend_position: str = "top"       # top, bottom, left, right
    
    # 交互配置
    interactive: bool = True
    show_toolbar: bool = True
    
    # 动画配置
    animation_frame: Optional[str] = None
    animation_duration: int = 500
    
    # 导出配置
    save_path: Optional[str] = None
    save_format: str = "png"           # png, jpg, pdf, svg, html

@dataclass
class VisualizationResult:
    """可视化结果"""
    plot_id: str
    plot_type: PlotType
    config: PlotConfig
    figure: Any  # matplotlib.Figure 或 plotly.Figure
    creation_time: float = field(default_factory=time.time)
    file_path: Optional[str] = None

class Visualizer:
    """
    可视化工具
    提供丰富的数据可视化功能
    """
    
    def __init__(self, visualizer_id: str = "Visualizer_001"):
        """
        初始化可视化工具
        
        Args:
            visualizer_id: 可视化器ID
        """
        self.visualizer_id = visualizer_id
        
        # === 设置样式 ===
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # === 图表创建方法映射 ===
        self.plot_creators = {
            PlotType.LINE: self._create_line_plot,
            PlotType.SCATTER: self._create_scatter_plot,
            PlotType.BAR: self._create_bar_plot,
            PlotType.HISTOGRAM: self._create_histogram,
            PlotType.HEATMAP: self._create_heatmap,
            PlotType.BOX: self._create_box_plot,
            PlotType.VIOLIN: self._create_violin_plot,
            PlotType.SURFACE: self._create_surface_plot,
            PlotType.CONTOUR: self._create_contour_plot,
            PlotType.RADAR: self._create_radar_chart,
            PlotType.SANKEY: self._create_sankey_diagram,
            PlotType.TREEMAP: self._create_treemap,
            PlotType.PARALLEL: self._create_parallel_coordinates,
            PlotType.GAUGE: self._create_gauge_chart,
            PlotType.CANDLESTICK: self._create_candlestick_chart
        }
        
        # === 可视化统计 ===
        self.visualization_stats = {
            'total_plots': 0,
            'plots_by_type': {plot_type: 0 for plot_type in PlotType},
            'creation_time': 0.0,
            'saved_plots': 0
        }
        
        # === 创建的图表存储 ===
        self.created_plots: Dict[str, VisualizationResult] = {}
        
        print(f"✅ 可视化工具初始化完成: {visualizer_id}")
        print(f"   支持图表类型: {len(self.plot_creators)} 种")
    
    def create_plot(self,
                   data: Union[Dict[str, np.ndarray], pd.DataFrame],
                   config: PlotConfig,
                   plot_id: Optional[str] = None) -> VisualizationResult:
        """
        创建图表
        
        Args:
            data: 数据
            config: 图表配置
            plot_id: 图表ID
            
        Returns:
            可视化结果
        """
        creation_start_time = time.time()
        
        if plot_id is None:
            plot_id = f"{config.plot_type.value}_{int(time.time()*1000)}"
        
        # 创建图表
        if config.plot_type in self.plot_creators:
            try:
                creator = self.plot_creators[config.plot_type]
                figure = creator(data, config)
                
                # 创建结果对象
                result = VisualizationResult(
                    plot_id=plot_id,
                    plot_type=config.plot_type,
                    config=config,
                    figure=figure
                )
                
                # 保存图表
                if config.save_path:
                    file_path = self._save_plot(figure, config)
                    result.file_path = file_path
                    self.visualization_stats['saved_plots'] += 1
                
                # 存储结果
                self.created_plots[plot_id] = result
                
                # 更新统计
                creation_time = time.time() - creation_start_time
                self._update_visualization_stats(config.plot_type, creation_time)
                
                print(f"✅ 图表创建完成: {plot_id} ({config.plot_type.value})")
                
                return result
                
            except Exception as e:
                print(f"❌ 图表创建失败: {str(e)}")
                raise
        else:
            raise ValueError(f"不支持的图表类型: {config.plot_type}")
    
    def _create_line_plot(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建线图"""
        if config.interactive:
            return self._create_plotly_line(data, config)
        else:
            return self._create_matplotlib_line(data, config)
    
    def _create_plotly_line(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建Plotly线图"""
        fig = go.Figure()
        
        if isinstance(data, dict):
            # 假设第一个数组是x轴，其余是y轴数据
            x_key = list(data.keys())[0]
            x_values = data[x_key]
            
            for key, values in data.items():
                if key != x_key:
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=values,
                        mode='lines+markers',
                        name=key,
                        line=dict(width=2)
                    ))
        
        elif isinstance(data, pd.DataFrame):
            for column in data.columns[1:]:  # 假设第一列是x轴
                fig.add_trace(go.Scatter(
                    x=data.iloc[:, 0],
                    y=data[column],
                    mode='lines+markers',
                    name=column,
                    line=dict(width=2)
                ))
        
        # 更新布局
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=config.show_legend,
            font=dict(size=config.font_size)
        )
        
        if config.x_range:
            fig.update_xaxes(range=config.x_range)
        if config.y_range:
            fig.update_yaxes(range=config.y_range)
        
        return fig
    
    def _create_matplotlib_line(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建Matplotlib线图"""
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
        
        if isinstance(data, dict):
            x_key = list(data.keys())[0]
            x_values = data[x_key]
            
            for key, values in data.items():
                if key != x_key:
                    ax.plot(x_values, values, label=key, linewidth=2, marker='o', markersize=4)
        
        elif isinstance(data, pd.DataFrame):
            for column in data.columns[1:]:
                ax.plot(data.iloc[:, 0], data[column], label=column, linewidth=2, marker='o', markersize=4)
        
        ax.set_title(config.title, fontsize=config.font_size + 2)
        ax.set_xlabel(config.x_label, fontsize=config.font_size)
        ax.set_ylabel(config.y_label, fontsize=config.font_size)
        
        if config.x_range:
            ax.set_xlim(config.x_range)
        if config.y_range:
            ax.set_ylim(config.y_range)
        
        if config.show_legend:
            ax.legend()
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def _create_scatter_plot(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建散点图"""
        if config.interactive:
            fig = go.Figure()
            
            if isinstance(data, dict):
                x_key, y_key = list(data.keys())[:2]
                fig.add_trace(go.Scatter(
                    x=data[x_key],
                    y=data[y_key],
                    mode='markers',
                    marker=dict(
                        size=8,
                        opacity=0.7,
                        colorscale=config.color_palette
                    ),
                    name='Data'
                ))
            
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label,
                yaxis_title=config.y_label,
                width=config.width,
                height=config.height,
                template=config.theme
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            if isinstance(data, dict):
                x_key, y_key = list(data.keys())[:2]
                ax.scatter(data[x_key], data[y_key], alpha=0.7, s=50)
            
            ax.set_title(config.title, fontsize=config.font_size + 2)
            ax.set_xlabel(config.x_label, fontsize=config.font_size)
            ax.set_ylabel(config.y_label, fontsize=config.font_size)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return fig
    
    def _create_bar_plot(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建柱状图"""
        if config.interactive:
            fig = go.Figure()
            
            if isinstance(data, dict):
                categories = list(data.keys())
                values = [np.mean(data[key]) if isinstance(data[key], np.ndarray) else data[key] 
                         for key in categories]
                
                fig.add_trace(go.Bar(
                    x=categories,
                    y=values,
                    marker_color=px.colors.qualitative.Set1
                ))
            
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label,
                yaxis_title=config.y_label,
                width=config.width,
                height=config.height,
                template=config.theme
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            if isinstance(data, dict):
                categories = list(data.keys())
                values = [np.mean(data[key]) if isinstance(data[key], np.ndarray) else data[key] 
                         for key in categories]
                
                ax.bar(categories, values)
            
            ax.set_title(config.title, fontsize=config.font_size + 2)
            ax.set_xlabel(config.x_label, fontsize=config.font_size)
            ax.set_ylabel(config.y_label, fontsize=config.font_size)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
    
    def _create_histogram(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建直方图"""
        if config.interactive:
            fig = go.Figure()
            
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, np.ndarray):
                        fig.add_trace(go.Histogram(
                            x=values,
                            name=key,
                            opacity=0.7,
                            nbinsx=30
                        ))
            
            fig.update_layout(
                title=config.title,
                xaxis_title=config.x_label,
                yaxis_title=config.y_label or "Frequency",
                width=config.width,
                height=config.height,
                template=config.theme,
                barmode='overlay'
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, np.ndarray):
                        ax.hist(values, bins=30, alpha=0.7, label=key)
            
            ax.set_title(config.title, fontsize=config.font_size + 2)
            ax.set_xlabel(config.x_label, fontsize=config.font_size)
            ax.set_ylabel(config.y_label or "Frequency", fontsize=config.font_size)
            
            if config.show_legend:
                ax.legend()
            
            plt.tight_layout()
            return fig
    
    def _create_heatmap(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建热力图"""
        if isinstance(data, dict):
            # 转换为矩阵形式
            if 'matrix' in data:
                matrix_data = data['matrix']
            else:
                # 尝试从多个数组构建矩阵
                arrays = [v for v in data.values() if isinstance(v, np.ndarray)]
                if arrays:
                    matrix_data = np.array(arrays)
                else:
                    raise ValueError("热力图需要矩阵数据")
        elif isinstance(data, pd.DataFrame):
            matrix_data = data.values
        else:
            matrix_data = data
        
        if config.interactive:
            fig = go.Figure(data=go.Heatmap(
                z=matrix_data,
                colorscale=config.color_palette,
                showscale=True
            ))
            
            fig.update_layout(
                title=config.title,
                width=config.width,
                height=config.height,
                template=config.theme
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            im = ax.imshow(matrix_data, cmap=config.color_palette, aspect='auto')
            ax.set_title(config.title, fontsize=config.font_size + 2)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            return fig
    
    def _create_box_plot(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建箱线图"""
        if config.interactive:
            fig = go.Figure()
            
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, np.ndarray):
                        fig.add_trace(go.Box(
                            y=values,
                            name=key,
                            boxpoints='outliers'
                        ))
            
            fig.update_layout(
                title=config.title,
                yaxis_title=config.y_label,
                width=config.width,
                height=config.height,
                template=config.theme
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            if isinstance(data, dict):
                values_list = []
                labels = []
                for key, values in data.items():
                    if isinstance(values, np.ndarray):
                        values_list.append(values)
                        labels.append(key)
                
                ax.boxplot(values_list, labels=labels)
            
            ax.set_title(config.title, fontsize=config.font_size + 2)
            ax.set_ylabel(config.y_label, fontsize=config.font_size)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
    
    def _create_violin_plot(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建小提琴图"""
        if config.interactive:
            fig = go.Figure()
            
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, np.ndarray):
                        fig.add_trace(go.Violin(
                            y=values,
                            name=key,
                            box_visible=True,
                            meanline_visible=True
                        ))
            
            fig.update_layout(
                title=config.title,
                yaxis_title=config.y_label,
                width=config.width,
                height=config.height,
                template=config.theme
            )
            
            return fig
        else:
            fig, ax = plt.subplots(figsize=(config.width/100, config.height/100))
            
            if isinstance(data, dict):
                values_list = []
                labels = []
                for key, values in data.items():
                    if isinstance(values, np.ndarray):
                        values_list.append(values)
                        labels.append(key)
                
                ax.violinplot(values_list)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels)
            
            ax.set_title(config.title, fontsize=config.font_size + 2)
            ax.set_ylabel(config.y_label, fontsize=config.font_size)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
    
    def _create_surface_plot(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建三维表面图"""
        if 'z_matrix' not in data:
            raise ValueError("三维表面图需要z_matrix数据")
        
        z_matrix = data['z_matrix']
        x_values = data.get('x', np.arange(z_matrix.shape[1]))
        y_values = data.get('y', np.arange(z_matrix.shape[0]))
        
        fig = go.Figure(data=[go.Surface(
            z=z_matrix,
            x=x_values,
            y=y_values,
            colorscale=config.color_palette
        )])
        
        fig.update_layout(
            title=config.title,
            scene=dict(
                xaxis_title=config.x_label,
                yaxis_title=config.y_label,
                zaxis_title=config.z_label
            ),
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def _create_contour_plot(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建等高线图"""
        if 'z_matrix' not in data:
            raise ValueError("等高线图需要z_matrix数据")
        
        z_matrix = data['z_matrix']
        x_values = data.get('x', np.arange(z_matrix.shape[1]))
        y_values = data.get('y', np.arange(z_matrix.shape[0]))
        
        fig = go.Figure(data=go.Contour(
            z=z_matrix,
            x=x_values,
            y=y_values,
            colorscale=config.color_palette,
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def _create_radar_chart(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建雷达图"""
        if isinstance(data, dict):
            categories = list(data.keys())
            values = [data[key] if isinstance(data[key], (int, float)) else np.mean(data[key]) 
                     for key in categories]
        else:
            categories = data.columns.tolist()
            values = data.iloc[0].values.tolist() if len(data) > 0 else []
        
        # 闭合雷达图
        categories += categories[:1]
        values += values[:1]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Values'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values[:-1]) * 1.1] if values else [0, 1]
                )
            ),
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme,
            showlegend=config.show_legend
        )
        
        return fig
    
    def _create_sankey_diagram(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建桑基图"""
        if not all(key in data for key in ['source', 'target', 'value']):
            raise ValueError("桑基图需要source, target, value数据")
        
        # 创建节点标签
        nodes = list(set(data['source']) | set(data['target']))
        node_dict = {node: i for i, node in enumerate(nodes)}
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color="blue"
            ),
            link=dict(
                source=[node_dict[src] for src in data['source']],
                target=[node_dict[tgt] for tgt in data['target']],
                value=data['value']
            )
        )])
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def _create_treemap(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建树状图"""
        if isinstance(data, dict) and 'labels' in data and 'values' in data:
            fig = go.Figure(go.Treemap(
                labels=data['labels'],
                values=data['values'],
                parents=data.get('parents', [""] * len(data['labels']))
            ))
        else:
            raise ValueError("树状图需要labels和values数据")
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def _create_parallel_coordinates(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建平行坐标图"""
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
        
        # 标准化数据
        df_normalized = (df - df.min()) / (df.max() - df.min())
        
        dimensions = []
        for col in df.columns:
            dimensions.append(dict(
                label=col,
                values=df_normalized[col]
            ))
        
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=df_normalized.iloc[:, 0], colorscale=config.color_palette),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def _create_gauge_chart(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建仪表盘图"""
        if isinstance(data, dict) and 'value' in data:
            value = data['value']
            max_value = data.get('max_value', 100)
            min_value = data.get('min_value', 0)
        else:
            raise ValueError("仪表盘图需要value数据")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': config.title},
            delta={'reference': data.get('reference', value * 0.9)},
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_value, max_value * 0.6], 'color': "lightgray"},
                    {'range': [max_value * 0.6, max_value * 0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(
            width=config.width,
            height=config.height,
            template=config.theme
        )
        
        return fig
    
    def _create_candlestick_chart(self, data: Union[Dict, pd.DataFrame], config: PlotConfig):
        """创建K线图"""
        required_fields = ['open', 'high', 'low', 'close']
        if not all(field in data for field in required_fields):
            raise ValueError("K线图需要open, high, low, close数据")
        
        x_values = data.get('x', np.arange(len(data['open'])))
        
        fig = go.Figure(data=go.Candlestick(
            x=x_values,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        ))
        
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label,
            yaxis_title=config.y_label,
            width=config.width,
            height=config.height,
            template=config.theme,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_dashboard(self,
                        plots: List[VisualizationResult],
                        dashboard_title: str = "Dashboard",
                        cols: int = 2) -> go.Figure:
        """创建仪表板"""
        rows = (len(plots) + cols - 1) // cols
        
        # 创建子图
        subplot_titles = [plot.config.title for plot in plots]
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
        )
        
        for i, plot in enumerate(plots):
            row = i // cols + 1
            col = i % cols + 1
            
            # 添加轨迹到子图（简化处理）
            if hasattr(plot.figure, 'data'):
                for trace in plot.figure.data:
                    fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            title=dashboard_title,
            height=400 * rows,
            showlegend=False
        )
        
        return fig
    
    def create_training_dashboard(self, training_data: Dict[str, np.ndarray]) -> go.Figure:
        """创建训练仪表板"""
        # 创建2x2子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Training Rewards',
                'Loss Curves',
                'Performance Metrics',
                'System Status'
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 训练奖励
        if 'rewards' in training_data:
            fig.add_trace(
                go.Scatter(
                    y=training_data['rewards'],
                    mode='lines',
                    name='Rewards',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # 损失曲线
        if 'actor_loss' in training_data:
            fig.add_trace(
                go.Scatter(
                    y=training_data['actor_loss'],
                    mode='lines',
                    name='Actor Loss',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
        
        if 'critic_loss' in training_data:
            fig.add_trace(
                go.Scatter(
                    y=training_data['critic_loss'],
                    mode='lines',
                    name='Critic Loss',
                    line=dict(color='green')
                ),
                row=1, col=2
            )
        
        # 性能指标
        if 'tracking_accuracy' in training_data:
            fig.add_trace(
                go.Scatter(
                    y=training_data['tracking_accuracy'],
                    mode='lines',
                    name='Tracking Accuracy',
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
        
        # 系统状态
        if 'temperatures' in training_data:
            fig.add_trace(
                go.Scatter(
                    y=training_data['temperatures'],
                    mode='lines',
                    name='Temperature',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title="Training Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_performance_comparison(self, 
                                   methods_data: Dict[str, Dict[str, np.ndarray]],
                                   metrics: List[str]) -> go.Figure:
        """创建性能对比图"""
        fig = go.Figure()
        
        for method_name, method_data in methods_data.items():
            metric_values = []
            for metric in metrics:
                if metric in method_data:
                    value = np.mean(method_data[metric]) if isinstance(method_data[metric], np.ndarray) else method_data[metric]
                    metric_values.append(value)
                else:
                    metric_values.append(0)
            
            fig.add_trace(go.Scatterpolar(
                r=metric_values,
                theta=metrics,
                fill='toself',
                name=method_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Performance Comparison",
            showlegend=True
        )
        
        return fig
    
    def _save_plot(self, figure: Any, config: PlotConfig) -> str:
        """保存图表"""
        file_path = config.save_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 根据图表类型和格式保存
        if hasattr(figure, 'write_html'):  # Plotly图表
            if config.save_format.lower() == 'html':
                figure.write_html(file_path)
            elif config.save_format.lower() in ['png', 'jpg', 'jpeg']:
                figure.write_image(file_path, format=config.save_format.lower())
            elif config.save_format.lower() == 'pdf':
                figure.write_image(file_path, format='pdf')
            elif config.save_format.lower() == 'svg':
                figure.write_image(file_path, format='svg')
        else:  # Matplotlib图表
            if config.save_format.lower() in ['png', 'jpg', 'jpeg', 'pdf', 'svg']:
                figure.savefig(file_path, format=config.save_format.lower(), 
                              dpi=300, bbox_inches='tight')
        
        return file_path
    
    def _update_visualization_stats(self, plot_type: PlotType, creation_time: float):
        """更新可视化统计"""
        self.visualization_stats['total_plots'] += 1
        self.visualization_stats['plots_by_type'][plot_type] += 1
        self.visualization_stats['creation_time'] += creation_time
    
    def export_all_plots(self, output_dir: str, format: str = 'png'):
        """导出所有图表"""
        os.makedirs(output_dir, exist_ok=True)
        
        exported_count = 0
        for plot_id, result in self.created_plots.items():
            try:
                file_path = os.path.join(output_dir, f"{plot_id}.{format}")
                
                # 临时修改配置进行保存
                temp_config = result.config
                temp_config.save_path = file_path
                temp_config.save_format = format
                
                saved_path = self._save_plot(result.figure, temp_config)
                result.file_path = saved_path
                exported_count += 1
                
            except Exception as e:
                print(f"⚠️ 导出图表 {plot_id} 失败: {str(e)}")
        
        print(f"✅ 成功导出 {exported_count}/{len(self.created_plots)} 个图表到 {output_dir}")
    
    def get_visualization_statistics(self) -> Dict[str, Any]:
        """获取可视化统计信息"""
        stats = self.visualization_stats.copy()
        
        if stats['total_plots'] > 0:
            stats['avg_creation_time'] = stats['creation_time'] / stats['total_plots']
        else:
            stats['avg_creation_time'] = 0
        
        stats['plots_created'] = len(self.created_plots)
        
        return stats
    
    def show_plot(self, plot_id: str):
        """显示图表"""
        if plot_id in self.created_plots:
            result = self.created_plots[plot_id]
            
            if hasattr(result.figure, 'show'):  # Plotly图表
                result.figure.show()
            else:  # Matplotlib图表
                plt.show()
        else:
            print(f"❌ 图表 {plot_id} 不存在")
    
    def clear_plots(self):
        """清除所有图表"""
        self.created_plots.clear()
        print("✅ 所有图表已清除")
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"Visualizer({self.visualizer_id}): "
                f"创建图表={self.visualization_stats['total_plots']}, "
                f"保存图表={self.visualization_stats['saved_plots']}")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"Visualizer(visualizer_id='{self.visualizer_id}', "
                f"plot_types={len(self.plot_creators)}, "
                f"total_plots={self.visualization_stats['total_plots']})")
