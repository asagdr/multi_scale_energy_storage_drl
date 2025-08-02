import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from enum import Enum

class BatteryType(Enum):
    """电池类型枚举"""
    LFP = "LiFePO4"  # 磷酸铁锂
    NCM = "LiNiCoMnO2"  # 三元锂
    LTO = "Li4Ti5O12"  # 钛酸锂

@dataclass
class BatteryParams:
    """
    电池参数配置类
    包含完整的电池物理和化学特性参数
    """
    
    # === 基本电池信息 ===
    battery_type: BatteryType = BatteryType.LFP
    manufacturer: str = "CATL"
    model: str = "280Ah-LFP"
    
    # === 电池单体规格 ===
    CELL_CAPACITY: float = 280.0        # Ah, 标称容量
    NOMINAL_VOLTAGE: float = 3.2        # V, 标称电压
    MAX_VOLTAGE: float = 3.65           # V, 充电截止电压
    MIN_VOLTAGE: float = 2.5            # V, 放电截止电压
    CELL_ENERGY: float = 0.896          # kWh, 单体能量 (280Ah × 3.2V)
    
    # === 电池组配置 ===
    SERIES_NUM: int = 100               # 串联单体数量
    PARALLEL_NUM: int = 10              # 并联单体数量
    
    # === SOC操作范围 ===
    MAX_SOC: float = 95.0               # %, 最大允许SOC
    MIN_SOC: float = 5.0                # %, 最小允许SOC  
    NOMINAL_SOC: float = 50.0           # %, 标称SOC
    OPTIMAL_SOC_RANGE: Tuple[float, float] = (20.0, 80.0)  # 最佳SOC范围
    
    # === 温度参数 ===
    MAX_TEMP: float = 60.0              # ℃, 最大工作温度
    MIN_TEMP: float = -20.0             # ℃, 最小工作温度
    NOMINAL_TEMP: float = 25.0          # ℃, 标称温度
    OPTIMAL_TEMP_RANGE: Tuple[float, float] = (15.0, 35.0)  # 最佳温度范围
    
    # === 充放电性能参数 ===
    MAX_CHARGE_C_RATE: float = 1.0      # 最大充电C率
    MAX_DISCHARGE_C_RATE: float = 3.0   # 最大放电C率
    CONTINUOUS_C_RATE: float = 1.0      # 连续充放电C率
    
    # === 内阻和损耗参数 ===
    INTERNAL_RESISTANCE: float = 0.001  # Ω, 内阻 (25℃, 50%SOC)
    CHARGE_EFFICIENCY: float = 0.98     # 充电效率
    DISCHARGE_EFFICIENCY: float = 0.95  # 放电效率
    
    # === 循环寿命参数 ===
    CYCLE_LIFE: int = 6000              # 循环次数 (80%DOD)
    CALENDAR_LIFE: int = 15             # 日历寿命 (年)
    EOL_CAPACITY: float = 80.0          # %, 寿命终止容量
    
    # === 劣化模型参数 ===
    BATTERY_PRICE: float = 0.486        # 元/Wh, 电池价格
    LIFECYCLE_CAPACITY_LOSS: float = 20.0  # %, 全生命周期容量损失 (100%-80%)
    
    # 劣化公式核心参数
    E_A: float = -31700                 # J, 活化能
    Z: float = 0.552                    # 电流密度指数
    BETA: float = 370.3                 # 容量衰减系数
    R: float = 8.314                    # J/(mol·K), 气体常数
    
    # 温度相关系数
    TEMP_COEFFICIENT: float = 1.421     # 温度系数 (℃/C²)
    
    # 倍率系数多项式参数 [常数项, 一次项, 二次项]
    B_COEFF: Tuple[float, float, float] = (33840.0, -6301.1, 448.96)
    
    # === BMS集群配置参数 ===
    NUM_BMS: int = 10                      # BMS数量
    CELLS_PER_BMS: int = 100               # 每个BMS管理的单体数量
    
    # BMS间协调参数
    INTER_BMS_SOC_THRESHOLD: float = 5.0   # BMS间SOC差异阈值 (%)
    INTER_BMS_TEMP_THRESHOLD: float = 10.0 # BMS间温度差异阈值 (℃)
    
    # BMS内均衡参数
    INTRA_BMS_SOC_THRESHOLD: float = 2.0   # BMS内SOC差异阈值 (%)
    INTRA_BMS_TEMP_THRESHOLD: float = 5.0  # BMS内温度差异阈值 (℃)
    
    # 功率分配参数
    POWER_ALLOCATION_STRATEGY: str = "multi_objective"  # 功率分配策略
    MAX_POWER_BIAS: float = 0.3            # 最大功率偏置比例
    MIN_BMS_POWER_RATIO: float = 0.1       # BMS最小功率分配比例
    
    # === 安全参数 ===
    SAFETY_MARGINS: Dict[str, float] = field(default_factory=lambda: {
        'voltage_high': 0.05,    # V, 电压上限安全裕度
        'voltage_low': 0.05,     # V, 电压下限安全裕度
        'temperature_high': 5.0, # ℃, 温度上限安全裕度
        'temperature_low': 5.0,  # ℃, 温度下限安全裕度
        'current_factor': 0.9    # 电流安全系数
    })
    
    # === 均衡参数 ===
    BALANCE_THRESHOLD: float = 0.05     # V, 电压均衡阈值
    BALANCE_CURRENT: float = 0.1        # A, 均衡电流
    
    # === 仿真参数 ===
    DEFAULT_TIME_STEP: float = 1.0      # s, 默认仿真时间步长
    NOISE_LEVELS: Dict[str, float] = field(default_factory=lambda: {
        'voltage_noise': 0.001,  # V, 电压测量噪声
        'current_noise': 0.1,    # A, 电流测量噪声
        'temp_noise': 0.5        # ℃, 温度测量噪声
    })
    
    def __post_init__(self):
        """初始化后计算衍生参数"""
        self._calculate_pack_parameters()
        self._calculate_power_limits()
        self._generate_soc_ocv_table()
        self._calculate_bms_cluster_parameters()  # 新增：计算BMS集群参数
        self._validate_parameters()
    
    def _calculate_pack_parameters(self):
        """计算电池组参数"""
        # 电池组总容量和电压
        self.pack_capacity = self.CELL_CAPACITY * self.PARALLEL_NUM  # Ah
        self.pack_voltage = self.NOMINAL_VOLTAGE * self.SERIES_NUM   # V
        self.pack_energy = self.pack_capacity * self.pack_voltage / 1000  # kWh
        
        # 总单体数量
        self.total_cells = self.SERIES_NUM * self.PARALLEL_NUM
        
        # 电池组内阻
        self.pack_resistance = (self.INTERNAL_RESISTANCE * self.SERIES_NUM / 
                              self.PARALLEL_NUM)  # Ω
    
    def _calculate_power_limits(self):
        """计算功率限制"""
        # 最大功率 (基于C率限制)
        self.max_charge_power = (self.pack_capacity * self.MAX_CHARGE_C_RATE * 
                               self.pack_voltage)  # W
        self.max_discharge_power = (self.pack_capacity * self.MAX_DISCHARGE_C_RATE * 
                                  self.pack_voltage)  # W
        
        # 连续功率
        self.continuous_power = (self.pack_capacity * self.CONTINUOUS_C_RATE * 
                               self.pack_voltage)  # W
        
        # 最大电流
        self.max_charge_current = self.pack_capacity * self.MAX_CHARGE_C_RATE  # A
        self.max_discharge_current = self.pack_capacity * self.MAX_DISCHARGE_C_RATE  # A
    
    def _calculate_bms_cluster_parameters(self):
        """计算BMS集群参数"""
        # 验证单体总数一致性
        total_cells_check = self.NUM_BMS * self.CELLS_PER_BMS
        if total_cells_check != self.total_cells:
            print(f"⚠️ BMS配置不一致: {self.NUM_BMS}x{self.CELLS_PER_BMS}={total_cells_check} != {self.total_cells}")
            # 自动调整
            self.CELLS_PER_BMS = self.total_cells // self.NUM_BMS
            print(f"🔧 自动调整: 每BMS单体数={self.CELLS_PER_BMS}")
        
        # 单个BMS参数
        # 假设每个BMS是串联的一组，并联分布在多个BMS中
        parallel_per_bms = self.PARALLEL_NUM // self.NUM_BMS if self.NUM_BMS <= self.PARALLEL_NUM else 1
        series_per_bms = self.SERIES_NUM  # 每个BMS都是完整的串联链
        
        self.bms_capacity = self.CELL_CAPACITY * parallel_per_bms  # 单BMS容量
        self.bms_voltage = self.NOMINAL_VOLTAGE * series_per_bms  # 单BMS电压
        self.bms_energy = self.bms_capacity * self.bms_voltage / 1000  # 单BMS能量 (kWh)
        
        # 单BMS功率限制
        self.bms_max_charge_power = self.max_charge_power / self.NUM_BMS
        self.bms_max_discharge_power = self.max_discharge_power / self.NUM_BMS
        
        print(f"📊 BMS集群参数计算完成:")
        print(f"   BMS数量: {self.NUM_BMS}")
        print(f"   每BMS单体数: {self.CELLS_PER_BMS}")
        print(f"   单BMS容量: {self.bms_capacity:.1f}Ah")
        print(f"   单BMS功率: 充电{self.bms_max_charge_power/1000:.1f}kW, 放电{self.bms_max_discharge_power/1000:.1f}kW")
    
    def _generate_soc_ocv_table(self):
        """生成SOC-OCV对应表"""
        # 磷酸铁锂电池典型的SOC-OCV关系点
        self.soc_ocv_points = {
            0.0: 2.80,   # 0% SOC
            5.0: 3.15,   # 5% SOC
            10.0: 3.20,  # 10% SOC (平台开始)
            20.0: 3.23,  # 20% SOC
            30.0: 3.25,  # 30% SOC
            40.0: 3.26,  # 40% SOC
            50.0: 3.275, # 50% SOC (平台中点)
            60.0: 3.29,  # 60% SOC
            70.0: 3.31,  # 70% SOC
            80.0: 3.33,  # 80% SOC
            90.0: 3.35,  # 90% SOC (平台结束)
            95.0: 3.40,  # 95% SOC
            100.0: 3.65  # 100% SOC
        }
        
        # 生成插值用的数组
        self.soc_array = np.array(list(self.soc_ocv_points.keys()))
        self.ocv_array = np.array(list(self.soc_ocv_points.values()))
    
    def _validate_parameters(self):
        """验证参数合理性"""
        validations = [
            (self.CELL_CAPACITY > 0, "电池容量必须大于0"),
            (self.MAX_VOLTAGE > self.MIN_VOLTAGE, "最大电压必须大于最小电压"),
            (self.MAX_SOC > self.MIN_SOC, "最大SOC必须大于最小SOC"),
            (self.MAX_TEMP > self.MIN_TEMP, "最大温度必须大于最小温度"),
            (self.SERIES_NUM > 0, "串联数量必须大于0"),
            (self.PARALLEL_NUM > 0, "并联数量必须大于0"),
            (0 < self.CHARGE_EFFICIENCY <= 1, "充电效率必须在0-1之间"),
            (0 < self.DISCHARGE_EFFICIENCY <= 1, "放电效率必须在0-1之间"),
            (self.NUM_BMS > 0, "BMS数量必须大于0"),
            (self.CELLS_PER_BMS > 0, "每BMS单体数必须大于0")
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"参数验证失败: {message}")
    
    def get_soc_from_ocv(self, ocv: float) -> float:
        """
        从开路电压获取SOC
        
        Args:
            ocv: 开路电压 (V)
            
        Returns:
            soc: 荷电状态 (%)
        """
        # 线性插值
        soc = np.interp(ocv, self.ocv_array, self.soc_array)
        return np.clip(soc, 0.0, 100.0)
    
    def get_ocv_from_soc(self, soc: float) -> float:
        """
        从SOC获取开路电压
        
        Args:
            soc: 荷电状态 (%)
            
        Returns:
            ocv: 开路电压 (V)
        """
        # 线性插值
        ocv = np.interp(soc, self.soc_array, self.ocv_array)
        return np.clip(ocv, self.MIN_VOLTAGE, self.MAX_VOLTAGE)
    
    def get_c_rate_limits(self, soc: float, temperature: float) -> Tuple[float, float]:
        """
        获取基于SOC和温度的C率限制
        
        Args:
            soc: 当前SOC (%)
            temperature: 当前温度 (℃)
            
        Returns:
            (max_charge_c, max_discharge_c): 最大充放电C率
        """
        # SOC限制因子
        if soc >= 90:
            soc_charge_factor = max(0, (95 - soc) / 5)
        elif soc <= 10:
            soc_discharge_factor = max(0, (soc - 5) / 5)
        else:
            soc_charge_factor = 1.0
            soc_discharge_factor = 1.0
        
        # 温度限制因子
        if temperature < 0:
            temp_factor = max(0.1, (temperature + 20) / 20)
        elif temperature > 45:
            temp_factor = max(0.1, (60 - temperature) / 15)
        else:
            temp_factor = 1.0
        
        # 应用限制因子
        max_charge_c = self.MAX_CHARGE_C_RATE * soc_charge_factor * temp_factor
        max_discharge_c = self.MAX_DISCHARGE_C_RATE * soc_discharge_factor * temp_factor
        
        return max_charge_c, max_discharge_c
    
    def get_bms_config(self) -> Dict:
        """获取BMS配置信息"""
        return {
            'num_bms': self.NUM_BMS,
            'cells_per_bms': self.CELLS_PER_BMS,
            'bms_capacity': getattr(self, 'bms_capacity', 0),
            'bms_voltage': getattr(self, 'bms_voltage', 0),
            'bms_energy': getattr(self, 'bms_energy', 0),
            'bms_max_charge_power': getattr(self, 'bms_max_charge_power', 0),
            'bms_max_discharge_power': getattr(self, 'bms_max_discharge_power', 0),
            'inter_bms_soc_threshold': self.INTER_BMS_SOC_THRESHOLD,
            'inter_bms_temp_threshold': self.INTER_BMS_TEMP_THRESHOLD,
            'intra_bms_soc_threshold': self.INTRA_BMS_SOC_THRESHOLD,
            'intra_bms_temp_threshold': self.INTRA_BMS_TEMP_THRESHOLD,
            'power_allocation_strategy': self.POWER_ALLOCATION_STRATEGY,
            'max_power_bias': self.MAX_POWER_BIAS
        }
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'battery_type': self.battery_type.value,
            'cell_capacity': self.CELL_CAPACITY,
            'nominal_voltage': self.NOMINAL_VOLTAGE,
            'pack_capacity': self.pack_capacity,
            'pack_voltage': self.pack_voltage,
            'pack_energy': self.pack_energy,
            'max_power': {
                'charge': self.max_charge_power,
                'discharge': self.max_discharge_power
            },
            'configuration': {
                'series': self.SERIES_NUM,
                'parallel': self.PARALLEL_NUM,
                'total_cells': self.total_cells
            },
            'bms_cluster': self.get_bms_config()
        }
    
    @classmethod
    def create_custom_config(cls, capacity: float, series: int, parallel: int) -> 'BatteryParams':
        """
        创建自定义配置
        
        Args:
            capacity: 单体容量 (Ah)
            series: 串联数量
            parallel: 并联数量
            
        Returns:
            自定义配置实例
        """
        return cls(
            CELL_CAPACITY=capacity,
            SERIES_NUM=series,
            PARALLEL_NUM=parallel
        )

# 预定义配置
class PresetConfigs:
    """预定义的电池配置"""
    
    @staticmethod
    def small_ess() -> BatteryParams:
        """小型储能系统 (100kWh) - 5个BMS"""
        config = BatteryParams.create_custom_config(
            capacity=100.0, series=100, parallel=3
        )
        config.NUM_BMS = 5
        config.CELLS_PER_BMS = 60  # 300总单体/5个BMS
        return config
    
    @staticmethod
    def medium_ess() -> BatteryParams:
        """中型储能系统 (1MWh) - 10个BMS"""
        config = BatteryParams.create_custom_config(
            capacity=280.0, series=100, parallel=10
        )
        config.NUM_BMS = 10
        config.CELLS_PER_BMS = 100
        return config
    
    @staticmethod
    def large_ess() -> BatteryParams:
        """大型储能系统 (10MWh) - 20个BMS"""
        config = BatteryParams.create_custom_config(
            capacity=280.0, series=100, parallel=100
        )
        config.NUM_BMS = 20
        config.CELLS_PER_BMS = 500
        return config
