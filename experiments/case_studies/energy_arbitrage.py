import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from ..basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType
from utils.logger import Logger
from utils.metrics import MetricsCalculator
from utils.visualization import Visualizer, PlotConfig, PlotType

class ArbitragePricingModel(Enum):
    """套利定价模式"""
    TIME_OF_USE = "time_of_use"              # 分时电价
    REAL_TIME_PRICING = "real_time_pricing"  # 实时电价
    DAY_AHEAD_MARKET = "day_ahead_market"    # 日前市场
    RENEWABLE_INTEGRATION = "renewable_integration"  # 可再生能源整合

@dataclass
class EnergyArbitrageConfig:
    """能量套利配置"""
    pricing_model: ArbitragePricingModel
    
    # 储能系统参数
    battery_capacity_kwh: float = 2000.0    # 电池容量
    max_power_kw: float = 1000.0            # 最大功率
    round_trip_efficiency: float = 0.85     # 往返效率
    self_discharge_rate: float = 0.001      # 自放电率 (每小时)
    
    # 市场参数
    trading_period_hours: int = 24 * 30     # 交易周期（30天）
    time_resolution_minutes: int = 15       # 时间分辨率
    price_volatility: float = 0.3           # 价格波动性
    
    # 价格范围
    min_price: float = 0.2                  # 最低电价 (元/kWh)
    max_price: float = 1.5                  # 最高电价 (元/kWh)
    avg_price: float = 0.6                  # 平均电价 (元/kWh)
    
    # 交易约束
    min_soc: float = 0.1                    # 最小SOC
    max_soc: float = 0.9                    # 最大SOC
    min_arbitrage_margin: float = 0.1       # 最小套利价差 (元/kWh)
    
    # 成本参数
    trading_fee_rate: float = 0.02          # 交易手续费率
    battery_degradation_cost: float = 0.05  # 电池损耗成本 (元/kWh)

@dataclass
class EnergyArbitrageResults:
    """能量套利结果"""
    experiment_id: str
    config: EnergyArbitrageConfig
    
    # 套利性能
    total_energy_traded_mwh: float = 0.0    # 总交易电量
    arbitrage_opportunities: int = 0         # 套利机会次数
    successful_arbitrages: int = 0           # 成功套利次数
    success_rate: float = 0.0               # 成功率
    
    # 经济效益
    gross_revenue: float = 0.0              # 总收入
    energy_costs: float = 0.0               # 能量成本
    trading_fees: float = 0.0               # 交易费用
    degradation_costs: float = 0.0          # 损耗成本
    net_profit: float = 0.0                 # 净利润
    profit_margin: float = 0.0              # 利润率
    
    # 市场表现
    avg_buy_price: float = 0.0              # 平均买入价格
    avg_sell_price: float = 0.0             # 平均卖出价格
    avg_arbitrage_margin: float = 0.0       # 平均套利价差
    market_timing_accuracy: float = 0.0     # 市场时机把握准确度
    
    # 系统运行
    avg_soc: float = 0.0                    # 平均SOC
    capacity_utilization: float = 0.0       # 容量利用率
    cycle_count: int = 0                    # 循环次数
    energy_efficiency: float = 0.0          # 能量效率
    
    # 时间序列数据
    electricity_prices: np.ndarray = field(default_factory=lambda: np.array([]))
    battery_power: np.ndarray = field(default_factory=lambda: np.array([]))
    battery_soc: np.ndarray = field(default_factory=lambda: np.array([]))
    trading_decisions: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulative_profit: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))

class EnergyArbitrageExperiment:
    """
    能量套利案例研究
    评估储能系统在电能套利应用中的收益性能
    """
    
    def __init__(self, config: EnergyArbitrageConfig, experiment_id: Optional[str] = None):
        """
        初始化能量套利实验
        
        Args:
            config: 能量套利配置
            experiment_id: 实验ID
        """
        self.config = config
        self.experiment_id = experiment_id or f"energy_arbitrage_{int(time.time()*1000)}"
        
        # 初始化组件
        self.logger = Logger(f"EnergyArbitrage_{self.experiment_id}")
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # 生成电价数据
        self._generate_price_data()
        
        # 创建实验目录
        self.experiment_dir = f"experiments/case_studies/energy_arbitrage/{self.experiment_id}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f"✅ 能量套利实验初始化完成: {config.pricing_model.value}")
        print(f"   实验ID: {self.experiment_id}")
        print(f"   交易周期: {config.trading_period_hours} 小时")
    
    def run_case_study(self) -> EnergyArbitrageResults:
        """
        运行能量套利案例研究
        
        Returns:
            能量套利结果
        """
        study_start_time = time.time()
        
        self.logger.info(f"🚀 开始能量套利案例研究: {self.config.pricing_model.value}")
        
        try:
            # 阶段1: 市场分析
            self.logger.info("📊 阶段1: 电价市场分析")
            market_analysis = self._analyze_price_market()
            
            # 阶段2: 套利策略训练
            self.logger.info("🎯 阶段2: 套利策略训练")
            arbitrage_strategy = self._train_arbitrage_strategy()
            
            # 阶段3: 套利交易仿真
            self.logger.info("💱 阶段3: 套利交易仿真")
            trading_results = self._simulate_arbitrage_trading(arbitrage_strategy)
            
            # 阶段4: 收益性分析
            self.logger.info("💰 阶段4: 收益性分析")
            profitability_analysis = self._analyze_profitability(trading_results)
            
            # 阶段5: 风险评估
            self.logger.info("⚠️ 阶段5: 风险评估")
            risk_assessment = self._assess_arbitrage_risks(trading_results)
            
            # 阶段6: 结果整合
            self.logger.info("📊 阶段6: 结果整合")
            final_results = self._integrate_arbitrage_results(
                market_analysis, trading_results,
                profitability_analysis, risk_assessment
            )
            
            # 生成报告
            self._generate_arbitrage_report(final_results)
            
            study_time = time.time() - study_start_time
            self.logger.info(f"✅ 能量套利案例研究完成，用时: {study_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"❌ 能量套利案例研究失败: {str(e)}")
            raise
    
    def _generate_price_data(self):
        """生成电价数据"""
        # 计算数据点数量
        hours = self.config.trading_period_hours
        points_per_hour = 60 // self.config.time_resolution_minutes
        num_points = hours * points_per_hour
        
        # 生成时间序列
        self.timestamps = np.arange(0, hours, 1/points_per_hour)
        
        # 根据定价模式生成电价
        if self.config.pricing_model == ArbitragePricingModel.TIME_OF_USE:
            self.electricity_prices = self._generate_tou_prices()
        elif self.config.pricing_model == ArbitragePricingModel.REAL_TIME_PRICING:
            self.electricity_prices = self._generate_rtp_prices()
        elif self.config.pricing_model == ArbitragePricingModel.DAY_AHEAD_MARKET:
            self.electricity_prices = self._generate_dam_prices()
        elif self.config.pricing_model == ArbitragePricingModel.RENEWABLE_INTEGRATION:
            self.electricity_prices = self._generate_renewable_prices()
        
        # 确保价格在合理范围内
        self.electricity_prices = np.clip(
            self.electricity_prices, 
            self.config.min_price, 
            self.config.max_price
        )
        
        self.logger.info(f"生成电价数据: {len(self.electricity_prices)} 个数据点")
    
    def _generate_tou_prices(self) -> np.ndarray:
        """生成分时电价"""
        num_points = len(self.timestamps)
        prices = np.zeros(num_points)
        
        for i, hour in enumerate(self.timestamps):
            hour_of_day = hour % 24
            
            # 分时电价模式
            if 8 <= hour_of_day < 12 or 18 <= hour_of_day < 22:
                # 峰时
                base_price = self.config.avg_price * 1.5
            elif 23 <= hour_of_day or hour_of_day < 7:
                # 谷时
                base_price = self.config.avg_price * 0.5
            else:
                # 平时
                base_price = self.config.avg_price
            
            # 添加随机波动
            daily_variation = 0.1 * np.sin(2 * np.pi * hour / 24)
            random_variation = np.random.normal(0, 0.05)
            
            prices[i] = base_price + daily_variation + random_variation
        
        return prices
    
    def _generate_rtp_prices(self) -> np.ndarray:
        """生成实时电价"""
        num_points = len(self.timestamps)
        
        # 基础价格趋势
        base_trend = self.config.avg_price + 0.2 * np.sin(2 * np.pi * self.timestamps / 24)
        
        # 高频波动
        high_freq = 0.1 * np.random.normal(0, self.config.price_volatility, num_points)
        
        # 价格冲击事件
        shock_events = np.random.poisson(0.1, num_points)  # 低概率事件
        price_shocks = shock_events * np.random.uniform(-0.3, 0.5, num_points)
        
        # 自相关过程
        autocorr_factor = 0.8
        correlated_noise = np.zeros(num_points)
        correlated_noise[0] = high_freq[0]
        
        for i in range(1, num_points):
            correlated_noise[i] = (autocorr_factor * correlated_noise[i-1] + 
                                 np.sqrt(1 - autocorr_factor**2) * high_freq[i])
        
        prices = base_trend + correlated_noise + price_shocks
        
        return prices
    
    def _generate_dam_prices(self) -> np.ndarray:
        """生成日前市场电价"""
        num_points = len(self.timestamps)
        prices = np.zeros(num_points)
        
        points_per_day = 96  # 24小时 * 4点/小时
        
        for day in range(0, num_points, points_per_day):
            # 每日价格模式
            day_hours = np.arange(24)
            
            # 负荷曲线影响的基础价格
            daily_pattern = (0.4 + 0.3 * np.sin(2 * np.pi * (day_hours - 6) / 24) + 
                           0.2 * np.sin(4 * np.pi * (day_hours - 6) / 24))
            
            # 扩展到15分钟分辨率
            daily_prices = np.repeat(daily_pattern, 4) * self.config.avg_price
            
            # 添加日间变异性
            daily_variation = np.random.normal(1, 0.1)
            market_stress = np.random.uniform(0.8, 1.3)
            
            end_idx = min(day + points_per_day, num_points)
            actual_points = end_idx - day
            
            prices[day:end_idx] = daily_prices[:actual_points] * daily_variation * market_stress
        
        return prices
    
    def _generate_renewable_prices(self) -> np.ndarray:
        """生成含可再生能源的电价"""
        num_points = len(self.timestamps)
        
        # 基础电价
        base_prices = self._generate_tou_prices()
        
        # 可再生能源出力模式（简化的太阳能+风能）
        solar_pattern = np.maximum(0, np.sin(2 * np.pi * (self.timestamps % 24 - 6) / 12))
        wind_pattern = 0.3 + 0.4 * np.sin(2 * np.pi * self.timestamps / 48) + 0.3 * np.random.random(num_points)
        
        renewable_output = 0.7 * solar_pattern + 0.3 * wind_pattern
        
        # 可再生能源出力对价格的影响（出力高时价格低）
        price_impact = 1 - 0.5 * renewable_output
        
        # 可再生能源的间歇性影响
        intermittency = 0.1 * np.random.normal(0, 1, num_points)
        
        prices = base_prices * price_impact + intermittency
        
        return prices
    
    def _analyze_price_market(self) -> Dict[str, Any]:
        """分析电价市场"""
        prices = self.electricity_prices
        
        # 基本统计
        analysis = {
            'price_statistics': {
                'mean_price': np.mean(prices),
                'price_std': np.std(prices),
                'min_price': np.min(prices),
                'max_price': np.max(prices),
                'price_range': np.max(prices) - np.min(prices),
                'volatility': np.std(prices) / np.mean(prices)
            },
            'arbitrage_opportunities': self._identify_arbitrage_opportunities(),
            'price_patterns': self._analyze_price_patterns(),
            'market_characteristics': self._characterize_market()
        }
        
        self.logger.info(f"市场分析完成 - 平均价格: {analysis['price_statistics']['mean_price']:.3f} 元/kWh")
        
        return analysis
    
    def _identify_arbitrage_opportunities(self) -> Dict[str, Any]:
        """识别套利机会"""
        prices = self.electricity_prices
        min_margin = self.config.min_arbitrage_margin
        
        opportunities = []
        
        # 滑动窗口寻找套利机会
        window_size = 96  # 24小时窗口
        
        for i in range(len(prices) - window_size):
            window_prices = prices[i:i + window_size]
            
            min_price = np.min(window_prices)
            max_price = np.max(window_prices)
            
            arbitrage_margin = max_price - min_price
            
            if arbitrage_margin >= min_margin:
                min_idx = np.argmin(window_prices) + i
                max_idx = np.argmax(window_prices) + i
                
                opportunities.append({
                    'buy_time': min_idx,
                    'sell_time': max_idx,
                    'buy_price': min_price,
                    'sell_price': max_price,
                    'margin': arbitrage_margin,
                    'time_span': abs(max_idx - min_idx) * self.config.time_resolution_minutes / 60
                })
        
        # 统计套利机会
        if opportunities:
            margins = [op['margin'] for op in opportunities]
            time_spans = [op['time_span'] for op in opportunities]
            
            opportunity_stats = {
                'total_opportunities': len(opportunities),
                'avg_margin': np.mean(margins),
                'max_margin': np.max(margins),
                'avg_time_span_hours': np.mean(time_spans),
                'opportunities_per_day': len(opportunities) / (self.config.trading_period_hours / 24)
            }
        else:
            opportunity_stats = {
                'total_opportunities': 0,
                'avg_margin': 0,
                'max_margin': 0,
                'avg_time_span_hours': 0,
                'opportunities_per_day': 0
            }
        
        return opportunity_stats
    
    def _analyze_price_patterns(self) -> Dict[str, Any]:
        """分析价格模式"""
        prices = self.electricity_prices
        
        # 日内模式
        points_per_day = 96
        if len(prices) >= points_per_day:
            daily_prices = prices[:len(prices)//points_per_day*points_per_day].reshape(-1, points_per_day)
            daily_pattern = np.mean(daily_prices, axis=0)
            daily_std = np.std(daily_prices, axis=0)
        else:
            daily_pattern = prices
            daily_std = np.zeros_like(prices)
        
        # 价格趋势
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        
        # 周期性分析
        from scipy.fft import fft, fftfreq
        fft_prices = fft(prices - np.mean(prices))
        frequencies = fftfreq(len(prices), d=self.config.time_resolution_minutes/60)
        
        # 找到主要周期
        power_spectrum = np.abs(fft_prices)**2
        dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
        dominant_period = 1 / abs(frequencies[dominant_freq_idx]) if frequencies[dominant_freq_idx] != 0 else 0
        
        patterns = {
            'daily_pattern': daily_pattern.tolist(),
            'daily_volatility': daily_std.tolist(),
            'price_trend': price_trend,
            'dominant_period_hours': dominant_period,
            'peak_hours': np.argmax(daily_pattern) * self.config.time_resolution_minutes / 60,
            'valley_hours': np.argmin(daily_pattern) * self.config.time_resolution_minutes / 60
        }
        
        return patterns
    
    def _characterize_market(self) -> Dict[str, Any]:
        """市场特征分析"""
        prices = self.electricity_prices
        
        # 价格分布特征
        price_percentiles = np.percentile(prices, [10, 25, 50, 75, 90])
        
        # 价格跳跃检测
        price_changes = np.abs(np.diff(prices))
        jump_threshold = np.percentile(price_changes, 95)
        jumps = np.sum(price_changes > jump_threshold)
        
        # 市场效率指标
        returns = np.diff(np.log(prices + 1e-6))
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        
        characteristics = {
            'price_percentiles': {
                'p10': price_percentiles[0],
                'p25': price_percentiles[1], 
                'p50': price_percentiles[2],
                'p75': price_percentiles[3],
                'p90': price_percentiles[4]
            },
            'market_dynamics': {
                'price_jumps': jumps,
                'jump_frequency': jumps / len(prices),
                'price_autocorrelation': autocorr,
                'market_efficiency': 1 - abs(autocorr)  # 低自相关表示高效率
            }
        }
        
        return characteristics
    
    def _train_arbitrage_strategy(self) -> Dict[str, Any]:
        """训练套利策略"""
        # 创建DRL训练配置
        experiment_settings = ExperimentSettings(
            experiment_name=f"energy_arbitrage_training_{self.config.pricing_model.value}",
            experiment_type=ExperimentType.HIERARCHICAL,
            description="能量套利策略训练",
            total_episodes=400,
            evaluation_frequency=80,
            save_frequency=160,
            use_pretraining=True,
            enable_hierarchical=True,
            enable_visualization=False,
            device="cpu",
            random_seed=42
        )
        
        # 运行训练
        training_experiment = BasicExperiment(
            settings=experiment_settings,
            experiment_id=f"{self.experiment_id}_training"
        )
        
        training_results = training_experiment.run_experiment()
        
        arbitrage_strategy = {
            'type': 'drl_arbitrage',
            'model_path': training_results.best_checkpoint_path,
            'performance': training_results.best_performance,
            'training_time': training_results.training_time
        }
        
        self.logger.info("套利策略训练完成")
        
        return arbitrage_strategy
    
    def _simulate_arbitrage_trading(self, arbitrage_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """仿真套利交易"""
        prices = self.electricity_prices
        num_points = len(prices)
        
        # 初始化状态
        battery_soc = np.zeros(num_points)
        battery_power = np.zeros(num_points)
        trading_decisions = np.zeros(num_points)  # 1=买入, -1=卖出, 0=持有
        cumulative_profit = np.zeros(num_points)
        
        # 初始状态
        current_soc = 0.5  # 50%初始SOC
        total_profit = 0.0
        
        # 简化的套利策略
        price_ma_short = 4   # 1小时移动平均
        price_ma_long = 24   # 6小时移动平均
        
        for i in range(max(price_ma_long, 1), num_points):
            current_price = prices[i]
            
            # 计算移动平均
            short_ma = np.mean(prices[i-price_ma_short:i])
            long_ma = np.mean(prices[i-price_ma_long:i])
            
            # 价格分位数策略
            recent_prices = prices[max(0, i-96):i]  # 最近24小时
            price_percentile = (np.sum(recent_prices <= current_price) / len(recent_prices) * 100)
            
            # 决策逻辑
            decision = 0  # 默认持有
            power = 0
            
            # 买入条件：价格低 + SOC有空间
            if (price_percentile < 20 and current_price < short_ma and 
                current_soc < self.config.max_soc - 0.05):
                
                # 买入（充电）
                max_charge_power = min(
                    self.config.max_power_kw,
                    (self.config.max_soc - current_soc) * self.config.battery_capacity_kwh * 4  # 15分钟充电量
                )
                
                power = max_charge_power * 0.8  # 保守充电
                decision = 1
                
            # 卖出条件：价格高 + SOC有余量
            elif (price_percentile > 80 and current_price > short_ma and 
                  current_soc > self.config.min_soc + 0.05):
                
                # 卖出（放电）
                max_discharge_power = min(
                    self.config.max_power_kw,
                    (current_soc - self.config.min_soc) * self.config.battery_capacity_kwh * 4
                )
                
                power = -max_discharge_power * 0.8  # 保守放电
                decision = -1
            
            # 执行交易
            battery_power[i] = power
            trading_decisions[i] = decision
            
            # 更新SOC
            time_resolution_hours = self.config.time_resolution_minutes / 60
            
            if power > 0:  # 充电
                energy_change = power * time_resolution_hours * self.config.round_trip_efficiency**0.5
                cost = power * time_resolution_hours * current_price
                total_profit -= cost
            elif power < 0:  # 放电
                energy_change = power * time_resolution_hours / self.config.round_trip_efficiency**0.5
                revenue = abs(power) * time_resolution_hours * current_price
                total_profit += revenue
            else:
                energy_change = 0
            
            # 自放电
            self_discharge = current_soc * self.config.self_discharge_rate * time_resolution_hours
            energy_change -= self_discharge * self.config.battery_capacity_kwh
            
            current_soc += energy_change / self.config.battery_capacity_kwh
            current_soc = np.clip(current_soc, self.config.min_soc, self.config.max_soc)
            
            battery_soc[i] = current_soc
            cumulative_profit[i] = total_profit
        
        trading_results = {
            'electricity_prices': prices,
            'battery_power': battery_power,
            'battery_soc': battery_soc,
            'trading_decisions': trading_decisions,
            'cumulative_profit': cumulative_profit,
            'timestamps': self.timestamps,
            'final_profit': total_profit
        }
        
        self.logger.info(f"套利交易仿真完成 - 总利润: {total_profit:.2f} 元")
        
        return trading_results
    
    def _analyze_profitability(self, trading_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析收益性"""
        prices = trading_results['electricity_prices']
        battery_power = trading_results['battery_power']
        trading_decisions = trading_results['trading_decisions']
        
        # 交易统计
        buy_trades = np.sum(trading_decisions == 1)
        sell_trades = np.sum(trading_decisions == -1)
        
        # 收入和成本计算
        time_resolution_hours = self.config.time_resolution_minutes / 60
        
        # 买入成本
        buy_mask = battery_power > 0
        energy_bought = np.sum(battery_power[buy_mask] * time_resolution_hours)
        avg_buy_price = np.average(prices[buy_mask], weights=battery_power[buy_mask]) if np.sum(buy_mask) > 0 else 0
        energy_costs = np.sum(battery_power[buy_mask] * time_resolution_hours * prices[buy_mask])
        
        # 卖出收入
        sell_mask = battery_power < 0
        energy_sold = np.sum(np.abs(battery_power[sell_mask]) * time_resolution_hours)
        avg_sell_price = np.average(prices[sell_mask], weights=np.abs(battery_power[sell_mask])) if np.sum(sell_mask) > 0 else 0
        gross_revenue = np.sum(np.abs(battery_power[sell_mask]) * time_resolution_hours * prices[sell_mask])
        
        # 交易费用
        total_traded_energy = energy_bought + energy_sold
        trading_fees = total_traded_energy * self.config.trading_fee_rate
        
        # 损耗成本
        degradation_costs = total_traded_energy * self.config.battery_degradation_cost
        
        # 净利润
        net_profit = gross_revenue - energy_costs - trading_fees - degradation_costs
        
        # 利润率
        profit_margin = net_profit / gross_revenue if gross_revenue > 0 else 0
        
        # 套利效果
        arbitrage_margin = avg_sell_price - avg_buy_price if avg_buy_price > 0 and avg_sell_price > 0 else 0
        
        profitability = {
            'trading_statistics': {
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'total_trades': buy_trades + sell_trades,
                'energy_bought_mwh': energy_bought / 1000,
                'energy_sold_mwh': energy_sold / 1000,
                'total_energy_traded_mwh': total_traded_energy / 1000
            },
            'financial_performance': {
                'gross_revenue': gross_revenue,
                'energy_costs': energy_costs,
                'trading_fees': trading_fees,
                'degradation_costs': degradation_costs,
                'net_profit': net_profit,
                'profit_margin': profit_margin
            },
            'trading_performance': {
                'avg_buy_price': avg_buy_price,
                'avg_sell_price': avg_sell_price,
                'arbitrage_margin': arbitrage_margin,
                'arbitrage_margin_percent': arbitrage_margin / avg_buy_price * 100 if avg_buy_price > 0 else 0
            }
        }
        
        return profitability
    
    def _assess_arbitrage_risks(self, trading_results: Dict[str, Any]) -> Dict[str, Any]:
        """评估套利风险"""
        cumulative_profit = trading_results['cumulative_profit']
        battery_soc = trading_results['battery_soc']
        
        # 收益波动性
        daily_profits = []
        points_per_day = 96
        
        for day in range(0, len(cumulative_profit), points_per_day):
            if day + points_per_day <= len(cumulative_profit):
                daily_profit = cumulative_profit[day + points_per_day - 1] - (cumulative_profit[day] if day > 0 else 0)
                daily_profits.append(daily_profit)
        
        if daily_profits:
            profit_volatility = np.std(daily_profits)
            max_drawdown = self._calculate_max_drawdown(cumulative_profit)
        else:
            profit_volatility = 0
            max_drawdown = 0
        
        # SOC风险
        soc_risk = {
            'min_soc_reached': np.min(battery_soc),
            'max_soc_reached': np.max(battery_soc),
            'soc_constraint_violations': np.sum((battery_soc < self.config.min_soc) | (battery_soc > self.config.max_soc)),
            'avg_soc_deviation': np.std(battery_soc)
        }
        
        # 市场风险
        prices = trading_results['electricity_prices']
        price_volatility = np.std(prices) / np.mean(prices)
        
        risk_assessment = {
            'profit_risk': {
                'profit_volatility': profit_volatility,
                'max_drawdown': max_drawdown,
                'negative_profit_days': np.sum(np.array(daily_profits) < 0) if daily_profits else 0,
                'sharpe_ratio': np.mean(daily_profits) / (profit_volatility + 1e-6) if daily_profits else 0
            },
            'operational_risk': soc_risk,
            'market_risk': {
                'price_volatility': price_volatility,
                'market_efficiency_risk': 'medium'  # 简化评估
            }
        }
        
        return risk_assessment
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """计算最大回撤"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-6)
        return np.max(drawdown)
    
    def _integrate_arbitrage_results(self, market_analysis: Dict[str, Any],
                                   trading_results: Dict[str, Any],
                                   profitability_analysis: Dict[str, Any],
                                   risk_assessment: Dict[str, Any]) -> EnergyArbitrageResults:
        """整合套利结果"""
        results = EnergyArbitrageResults(
            experiment_id=self.experiment_id,
            config=self.config
        )
        
        # 套利性能
        results.total_energy_traded_mwh = profitability_analysis['trading_statistics']['total_energy_traded_mwh']
        results.arbitrage_opportunities = market_analysis['arbitrage_opportunities']['total_opportunities']
        results.successful_arbitrages = profitability_analysis['trading_statistics']['total_trades']
        results.success_rate = (results.successful_arbitrages / max(results.arbitrage_opportunities, 1))
        
        # 经济效益
        results.gross_revenue = profitability_analysis['financial_performance']['gross_revenue']
        results.energy_costs = profitability_analysis['financial_performance']['energy_costs']
        results.trading_fees = profitability_analysis['financial_performance']['trading_fees']
        results.degradation_costs = profitability_analysis['financial_performance']['degradation_costs']
        results.net_profit = profitability_analysis['financial_performance']['net_profit']
        results.profit_margin = profitability_analysis['financial_performance']['profit_margin']
        
        # 市场表现
        results.avg_buy_price = profitability_analysis['trading_performance']['avg_buy_price']
        results.avg_sell_price = profitability_analysis['trading_performance']['avg_sell_price']
        results.avg_arbitrage_margin = profitability_analysis['trading_performance']['arbitrage_margin']
        
        # 系统运行
        battery_soc = trading_results['battery_soc']
        results.avg_soc = np.mean(battery_soc)
        results.capacity_utilization = (np.max(battery_soc) - np.min(battery_soc)) / (self.config.max_soc - self.config.min_soc)
        
        # 估算循环次数
        soc_changes = np.abs(np.diff(battery_soc))
        results.cycle_count = int(np.sum(soc_changes) / 2)
        
        # 能量效率
        energy_in = np.sum(trading_results['battery_power'][trading_results['battery_power'] > 0])
        energy_out = np.sum(np.abs(trading_results['battery_power'][trading_results['battery_power'] < 0]))
        results.energy_efficiency = energy_out / (energy_in + 1e-6) if energy_in > 0 else 0
        
        # 时间序列数据
        results.electricity_prices = trading_results['electricity_prices']
        results.battery_power = trading_results['battery_power']
        results.battery_soc = trading_results['battery_soc']
        results.trading_decisions = trading_results['trading_decisions']
        results.cumulative_profit = trading_results['cumulative_profit']
        results.timestamps = trading_results['timestamps']
        
        return results
    
    def _generate_arbitrage_report(self, results: EnergyArbitrageResults):
        """生成套利报告"""
        report = {
            'case_study_info': {
                'experiment_id': results.experiment_id,
                'pricing_model': results.config.pricing_model.value,
                'trading_period_hours': results.config.trading_period_hours,
                'battery_capacity_kwh': results.config.battery_capacity_kwh,
                'max_power_kw': results.config.max_power_kw
            },
            'arbitrage_performance': {
                'total_energy_traded_mwh': results.total_energy_traded_mwh,
                'arbitrage_opportunities': results.arbitrage_opportunities,
                'successful_arbitrages': results.successful_arbitrages,
                'success_rate_percent': results.success_rate * 100,
                'avg_arbitrage_margin': results.avg_arbitrage_margin,
                'market_timing_accuracy': results.market_timing_accuracy
            },
            'financial_performance': {
                'revenue': {
                    'gross_revenue': results.gross_revenue,
                    'net_profit': results.net_profit,
                    'profit_margin_percent': results.profit_margin * 100
                },
                'costs': {
                    'energy_costs': results.energy_costs,
                    'trading_fees': results.trading_fees,
                    'degradation_costs': results.degradation_costs
                },
                'trading_metrics': {
                    'avg_buy_price': results.avg_buy_price,
                    'avg_sell_price': results.avg_sell_price,
                    'price_spread': results.avg_sell_price - results.avg_buy_price
                }
            },
            'system_performance': {
                'avg_soc_percent': results.avg_soc * 100,
                'capacity_utilization_percent': results.capacity_utilization * 100,
                'cycle_count': results.cycle_count,
                'energy_efficiency_percent': results.energy_efficiency * 100
            },
            'key_findings': [],
            'recommendations': []
        }
        
        # 关键发现
        if results.net_profit > 0:
            report['key_findings'].append(f"实现盈利：净利润 {results.net_profit:.0f} 元")
        
        if results.success_rate > 0.6:
            report['key_findings'].append(f"较高成功率：套利成功率 {results.success_rate:.1%}")
        
        if results.avg_arbitrage_margin > 0.1:
            report['key_findings'].append(f"良好价差：平均套利价差 {results.avg_arbitrage_margin:.3f} 元/kWh")
        
        # 建议
        if results.net_profit <= 0:
            report['recommendations'].append("当前市场条件下盈利困难，建议调整策略或等待更好的市场时机")
        
        if results.capacity_utilization < 0.5:
            report['recommendations'].append("容量利用率较低，建议优化交易策略以提高资产利用效率")
        
        if results.energy_efficiency < 0.8:
            report['recommendations'].append("能量效率偏低，建议关注电池性能和控制策略优化")
        
        # 保存报告
        report_path = os.path.join(self.experiment_dir, "energy_arbitrage_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 生成可视化
        self._create_arbitrage_visualizations(results)
        
        self.logger.info(f"能量套利报告已保存: {report_path}")
        
        return report
    
    def _create_arbitrage_visualizations(self, results: EnergyArbitrageResults):
        """创建套利可视化"""
        # 选择前7天数据进行展示
        points_per_week = 7 * 96
        show_points = min(points_per_week, len(results.timestamps))
        
        # 1. 电价和交易决策图
        price_trading_config = PlotConfig(
            plot_type=PlotType.LINE,
            title="电价波动与交易决策",
            x_label="时间 (小时)",
            y_label="电价 (元/kWh) / 交易信号",
            width=1200,
            height=600,
            save_path=os.path.join(self.experiment_dir, "price_trading.png")
        )
        
        price_trading_data = {
            'time': results.timestamps[:show_points],
            'electricity_price': results.electricity_prices[:show_points],
            'trading_decisions': results.trading_decisions[:show_points] * 0.2 + 0.6  # 缩放以便显示
        }
        
        self.visualizer.create_plot(price_trading_data, price_trading_config)
        
        # 2. 累积利润图
        profit_config = PlotConfig(
            plot_type=PlotType.LINE,
            title="累积利润变化",
            x_label="时间 (小时)",
            y_label="累积利润 (元)",
            width=1200,
            height=500,
            save_path=os.path.join(self.experiment_dir, "cumulative_profit.png")
        )
        
        profit_data = {
            'time': results.timestamps[:show_points],
            'cumulative_profit': results.cumulative_profit[:show_points]
        }
        
        self.visualizer.create_plot(profit_data, profit_config)
        
        # 3. 电池SOC变化图
        soc_config = PlotConfig(
            plot_type=PlotType.LINE,
            title="电池SOC变化",
            x_label="时间 (小时)",
            y_label="SOC (%) / 功率 (kW)",
            width=1200,
            height=500,
            save_path=os.path.join(self.experiment_dir, "battery_operation.png")
        )
        
        soc_data = {
            'time': results.timestamps[:show_points],
            'soc': results.battery_soc[:show_points] * 100,
            'power': results.battery_power[:show_points] / 10  # 缩放以便显示
        }
        
        self.visualizer.create_plot(soc_data, soc_config)
        
        # 4. 收益结构图
        revenue_config = PlotConfig(
            plot_type=PlotType.BAR,
            title="收益成本结构分析",
            x_label="项目",
            y_label="金额 (元)",
            width=800,
            height=600,
            save_path=os.path.join(self.experiment_dir, "revenue_structure.png")
        )
        
        revenue_data = {
            'gross_revenue': results.gross_revenue,
            'energy_costs': -results.energy_costs,  # 负值表示成本
            'trading_fees': -results.trading_fees,
            'degradation_costs': -results.degradation_costs,
            'net_profit': results.net_profit
        }
        
        self.visualizer.create_plot(revenue_data, revenue_config)
        
        self.logger.info("能量套利可视化图表生成完成")
