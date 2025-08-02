#!/usr/bin/env python3
"""
多尺度储能系统深度强化学习主程序
Multi-scale Energy Storage DRL System Main Entry Point

作者: asagdr
创建时间: 2025-08-01
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from config.training_config import TrainingConfig
from config.model_config import ModelConfig
from config.environment_config import EnvironmentConfig
from experiments.basic_experiments import BasicExperiment, ExperimentSettings, ExperimentType
from experiments.case_studies.peak_shaving import PeakShavingExperiment, PeakShavingConfig, PeakShavingScenario
from experiments.case_studies.frequency_regulation import FrequencyRegulationExperiment, FrequencyRegulationConfig, FrequencyRegulationService
from experiments.case_studies.energy_arbitrage import EnergyArbitrageExperiment, EnergyArbitrageConfig, ArbitragePricingModel
from experiments.ablation_studies import AblationStudy, AblationConfig, AblationComponent
from experiments.sensitivity_analysis import SensitivityAnalysis, SensitivityConfig, ParameterRange, ParameterType
from utils.logger import Logger
from data_processing.scenario_generator import ScenarioType

# 版本信息
__version__ = "0.1.0"
__author__ = "asagdr"
__email__ = "asagdr@example.com"

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志系统"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 设置根日志器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    # 创建项目日志器
    logger = Logger("MainProgram")
    return logger

def print_banner():
    """打印项目横幅"""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     多尺度储能系统深度强化学习平台                           ║
║                Multi-scale Energy Storage DRL System                        ║
║                                                                              ║
║  版本: {__version__:<10}  作者: {__author__:<20}  日期: 2025-08-01          ║
║                                                                              ║
║  功能特性:                                                                   ║
║  • 分层深度强化学习架构                                                     ║
║  • 多尺度电池建模与仿真                                                     ║
║  • 智能储能控制策略                                                         ║
║  • 全面实验评估框架                                                         ║
║  • 实际应用案例研究                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def create_basic_experiment_parser(subparsers):
    """创建基础实验参数解析器"""
    parser = subparsers.add_parser("experiment", help="运行基础实验")
    parser.add_argument("--name", type=str, default="basic_experiment", help="实验名称")
    parser.add_argument("--type", type=str, choices=["hierarchical", "single_objective", "multi_objective"], 
                       default="hierarchical", help="实验类型")
    parser.add_argument("--episodes", type=int, default=1000, help="训练回合数")
    parser.add_argument("--scenario", type=str, choices=["daily_cycle", "peak_shaving", "frequency_regulation"], 
                       default="daily_cycle", help="场景类型")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", type=str, default="./results", help="输出目录")
    return parser

def create_case_study_parser(subparsers):
    """创建案例研究参数解析器"""
    parser = subparsers.add_parser("case-study", help="运行案例研究")
    parser.add_argument("--type", type=str, choices=["peak_shaving", "frequency_regulation", "energy_arbitrage"], 
                       required=True, help="案例研究类型")
    parser.add_argument("--scenario", type=str, help="场景参数")
    parser.add_argument("--battery-capacity", type=float, default=1000.0, help="电池容量 (kWh)")
    parser.add_argument("--max-power", type=float, default=500.0, help="最大功率 (kW)")
    parser.add_argument("--duration", type=int, default=24, help="仿真时长 (小时)")
    parser.add_argument("--output-dir", type=str, default="./case_studies", help="输出目录")
    return parser

def create_ablation_parser(subparsers):
    """创建消融实验参数解析器"""
    parser = subparsers.add_parser("ablation", help="运行消融实验")
    parser.add_argument("--components", type=str, nargs="+", 
                       choices=["hierarchical_structure", "pretraining", "multi_objective", "communication"],
                       default=["hierarchical_structure", "pretraining"], help="要消融的组件")
    parser.add_argument("--repetitions", type=int, default=3, help="重复次数")
    parser.add_argument("--episodes", type=int, default=500, help="训练回合数")
    parser.add_argument("--output-dir", type=str, default="./ablation_results", help="输出目录")
    return parser

def create_sensitivity_parser(subparsers):
    """创建敏感性分析参数解析器"""
    parser = subparsers.add_parser("sensitivity", help="运行敏感性分析")
    parser.add_argument("--parameters", type=str, nargs="+",
                       choices=["learning_rate", "batch_size", "hidden_size", "discount_factor"],
                       default=["learning_rate", "batch_size"], help="要分析的参数")
    parser.add_argument("--analysis-type", type=str, choices=["one_at_a_time", "factorial", "sobol"],
                       default="one_at_a_time", help="分析类型")
    parser.add_argument("--samples", type=int, default=5, help="每个参数的采样点数")
    parser.add_argument("--output-dir", type=str, default="./sensitivity_results", help="输出目录")
    return parser

def run_basic_experiment(args, logger: Logger) -> Dict[str, Any]:
    """运行基础实验"""
    logger.info(f"🚀 开始运行基础实验: {args.name}")
    
    # 创建实验设置
    experiment_type_map = {
        "hierarchical": ExperimentType.HIERARCHICAL,
        "single_objective": ExperimentType.SINGLE_OBJECTIVE,
        "multi_objective": ExperimentType.MULTI_OBJECTIVE
    }
    
    scenario_type_map = {
        "daily_cycle": ScenarioType.DAILY_CYCLE,
        "peak_shaving": ScenarioType.PEAK_SHAVING,
        "frequency_regulation": ScenarioType.FREQUENCY_REGULATION
    }
    
    settings = ExperimentSettings(
        experiment_name=args.name,
        experiment_type=experiment_type_map[args.type],
        description=f"基础实验 - {args.type}",
        total_episodes=args.episodes,
        scenario_types=[scenario_type_map[args.scenario]],
        device=args.device,
        random_seed=args.seed,
        enable_visualization=True
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    experiment = BasicExperiment(settings=settings)
    results = experiment.run_experiment()
    
    # 保存结果
    results_file = output_dir / f"{args.name}_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "experiment_id": results.experiment_id,
            "settings": {
                "experiment_name": settings.experiment_name,
                "experiment_type": settings.experiment_type.value,
                "total_episodes": settings.total_episodes,
                "device": settings.device,
                "random_seed": settings.random_seed
            },
            "performance": {
                "total_time": results.total_time,
                "training_time": results.training_time,
                "final_performance": results.final_performance,
                "best_performance": results.best_performance
            }
        }, f, indent=2, default=str)
    
    logger.info(f"✅ 基础实验完成，结果保存到: {results_file}")
    return {"status": "success", "results_file": str(results_file)}

def run_case_study(args, logger: Logger) -> Dict[str, Any]:
    """运行案例研究"""
    logger.info(f"🏗️ 开始运行案例研究: {args.type}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.type == "peak_shaving":
        # 削峰填谷案例
        config = PeakShavingConfig(
            scenario_type=PeakShavingScenario.COMMERCIAL_BUILDING,
            battery_capacity_kwh=args.battery_capacity,
            max_power_kw=args.max_power,
            load_profile_days=args.duration // 24
        )
        
        experiment = PeakShavingExperiment(config=config)
        results = experiment.run_case_study()
        
        # 保存结果
        results_file = output_dir / f"peak_shaving_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "experiment_id": results.experiment_id,
                "case_type": "peak_shaving",
                "configuration": {
                    "battery_capacity_kwh": config.battery_capacity_kwh,
                    "max_power_kw": config.max_power_kw,
                    "scenario_type": config.scenario_type.value
                },
                "performance": {
                    "peak_reduction_ratio": results.peak_reduction_ratio,
                    "total_cost_savings": results.total_cost_savings,
                    "payback_period_years": results.payback_period_years
                }
            }, f, indent=2, default=str)
    
    elif args.type == "frequency_regulation":
        # 频率调节案例
        config = FrequencyRegulationConfig(
            service_type=FrequencyRegulationService.PRIMARY_RESERVE,
            battery_capacity_kwh=args.battery_capacity,
            max_power_kw=args.max_power,
            service_duration_hours=args.duration
        )
        
        experiment = FrequencyRegulationExperiment(config=config)
        results = experiment.run_case_study()
        
        # 保存结果
        results_file = output_dir / f"frequency_regulation_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "experiment_id": results.experiment_id,
                "case_type": "frequency_regulation",
                "configuration": {
                    "battery_capacity_kwh": config.battery_capacity_kwh,
                    "max_power_kw": config.max_power_kw,
                    "service_type": config.service_type.value
                },
                "performance": {
                    "regulation_accuracy": results.regulation_accuracy,
                    "total_revenue": results.total_revenue,
                    "net_profit": results.net_profit
                }
            }, f, indent=2, default=str)
    
    elif args.type == "energy_arbitrage":
        # 能量套利案例
        config = EnergyArbitrageConfig(
            pricing_model=ArbitragePricingModel.TIME_OF_USE,
            battery_capacity_kwh=args.battery_capacity,
            max_power_kw=args.max_power,
            trading_period_hours=args.duration
        )
        
        experiment = EnergyArbitrageExperiment(config=config)
        results = experiment.run_case_study()
        
        # 保存结果
        results_file = output_dir / f"energy_arbitrage_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "experiment_id": results.experiment_id,
                "case_type": "energy_arbitrage",
                "configuration": {
                    "battery_capacity_kwh": config.battery_capacity_kwh,
                    "max_power_kw": config.max_power_kw,
                    "pricing_model": config.pricing_model.value
                },
                "performance": {
                    "net_profit": results.net_profit,
                    "success_rate": results.success_rate,
                    "total_energy_traded_mwh": results.total_energy_traded_mwh
                }
            }, f, indent=2, default=str)
    
    logger.info(f"✅ 案例研究完成，结果保存到: {results_file}")
    return {"status": "success", "results_file": str(results_file)}

def run_ablation_study(args, logger: Logger) -> Dict[str, Any]:
    """运行消融实验"""
    logger.info(f"🔬 开始运行消融实验")
    
    # 创建基线配置
    baseline_config = ExperimentSettings(
        experiment_name="ablation_baseline",
        experiment_type=ExperimentType.HIERARCHICAL,
        total_episodes=args.episodes,
        scenario_types=[ScenarioType.DAILY_CYCLE],
        use_pretraining=True,
        enable_hierarchical=True,
        enable_visualization=False,
        device="cpu",
        random_seed=42
    )
    
    # 映射组件名称
    component_map = {
        "hierarchical_structure": AblationComponent.HIERARCHICAL_STRUCTURE,
        "pretraining": AblationComponent.PRETRAINING,
        "multi_objective": AblationComponent.MULTI_OBJECTIVE,
        "communication": AblationComponent.COMMUNICATION
    }
    
    components_to_ablate = [component_map[name] for name in args.components]
    
    # 创建消融配置
    ablation_config = AblationConfig(
        study_name="main_ablation_study",
        description="主程序消融实验",
        components_to_ablate=components_to_ablate,
        baseline_config=baseline_config,
        num_repetitions=args.repetitions,
        primary_metrics=['episode_reward', 'tracking_accuracy', 'energy_efficiency']
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行消融实验
    ablation_study = AblationStudy(ablation_config)
    results = ablation_study.run_study()
    
    # 保存结果
    results_file = output_dir / "ablation_study_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "study_info": {
                "study_name": ablation_config.study_name,
                "components_ablated": [c.value for c in components_to_ablate],
                "repetitions": args.repetitions,
                "episodes": args.episodes
            },
            "component_importance": ablation_study.get_component_importance(),
            "summary": "消融实验已完成，详细结果请查看具体的结果文件"
        }, f, indent=2, default=str)
    
    logger.info(f"✅ 消融实验完成，结果保存到: {results_file}")
    return {"status": "success", "results_file": str(results_file)}

def run_sensitivity_analysis(args, logger: Logger) -> Dict[str, Any]:
    """运行敏感性分析"""
    logger.info(f"📊 开始运行敏感性分析")
    
    # 创建基线配置
    baseline_config = ExperimentSettings(
        experiment_name="sensitivity_baseline",
        experiment_type=ExperimentType.HIERARCHICAL,
        total_episodes=200,
        scenario_types=[ScenarioType.DAILY_CYCLE],
        use_pretraining=False,
        enable_hierarchical=True,
        enable_visualization=False,
        device="cpu",
        random_seed=42
    )
    
    # 参数映射
    param_map = {
        "learning_rate": ParameterType.LEARNING_RATE,
        "batch_size": ParameterType.BATCH_SIZE,
        "hidden_size": ParameterType.NETWORK_HIDDEN_SIZE,
        "discount_factor": ParameterType.DISCOUNT_FACTOR
    }
    
    # 创建参数范围
    parameter_ranges = []
    for param_name in args.parameters:
        param_type = param_map[param_name]
        
        if param_name == "learning_rate":
            param_range = ParameterRange(
                param_type=param_type,
                min_value=0.0001,
                max_value=0.01,
                num_samples=args.samples,
                scale="log"
            )
        elif param_name == "batch_size":
            param_range = ParameterRange(
                param_type=param_type,
                min_value=16,
                max_value=128,
                num_samples=args.samples,
                scale="linear"
            )
        elif param_name == "hidden_size":
            param_range = ParameterRange(
                param_type=param_type,
                min_value=64,
                max_value=512,
                num_samples=args.samples,
                scale="linear"
            )
        elif param_name == "discount_factor":
            param_range = ParameterRange(
                param_type=param_type,
                min_value=0.9,
                max_value=0.999,
                num_samples=args.samples,
                scale="linear"
            )
        
        parameter_ranges.append(param_range)
    
    # 创建敏感性分析配置
    sensitivity_config = SensitivityConfig(
        study_name="main_sensitivity_analysis",
        description="主程序敏感性分析",
        parameters_to_analyze=parameter_ranges,
        baseline_config=baseline_config,
        analysis_type=args.analysis_type,
        num_repetitions=2,
        primary_metrics=['episode_reward', 'tracking_accuracy']
    )
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行敏感性分析
    sensitivity_analysis = SensitivityAnalysis(sensitivity_config)
    results = sensitivity_analysis.run_analysis()
    
    # 保存结果
    results_file = output_dir / "sensitivity_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "study_info": {
                "study_name": sensitivity_config.study_name,
                "parameters_analyzed": args.parameters,
                "analysis_type": args.analysis_type,
                "samples_per_parameter": args.samples
            },
            "parameter_rankings": results.get("parameter_rankings", {}),
            "summary": "敏感性分析已完成，详细结果请查看具体的结果文件"
        }, f, indent=2, default=str)
    
    logger.info(f"✅ 敏感性分析完成，结果保存到: {results_file}")
    return {"status": "success", "results_file": str(results_file)}

def main():
    """主函数"""
    # 打印横幅
    print_banner()
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="多尺度储能系统深度强化学习平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s experiment --name my_experiment --episodes 1000
  %(prog)s case-study --type peak_shaving --battery-capacity 1000
  %(prog)s ablation --components hierarchical_structure pretraining
  %(prog)s sensitivity --parameters learning_rate batch_size
        """
    )
    
    # 通用参数
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    parser.add_argument("--log-file", type=str, help="日志文件路径")
    parser.add_argument("--config", type=str, help="配置文件路径")
    
    # 创建子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 子命令解析器
    create_basic_experiment_parser(subparsers)
    create_case_study_parser(subparsers)
    create_ablation_parser(subparsers)
    create_sensitivity_parser(subparsers)
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        parser.print_help()
        return 1
    
    # 设置日志
    logger = setup_logging(args.log_level, args.log_file)
    
    try:
        logger.info(f"🎯 执行命令: {args.command}")
        
        # 根据命令执行相应功能
        if args.command == "experiment":
            result = run_basic_experiment(args, logger)
        elif args.command == "case-study":
            result = run_case_study(args, logger)
        elif args.command == "ablation":
            result = run_ablation_study(args, logger)
        elif args.command == "sensitivity":
            result = run_sensitivity_analysis(args, logger)
        else:
            logger.error(f"❌ 未知命令: {args.command}")
            return 1
        
        logger.info(f"🎉 程序执行完成: {result['status']}")
        logger.info(f"📁 结果文件: {result['results_file']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断程序执行")
        return 130
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {str(e)}")
        logger.debug("详细错误信息:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
