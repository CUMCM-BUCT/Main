"""V4 中已固定、且后续求解器必须遵守的建模约定。"""

from dataclasses import dataclass


MODEL_VERSION = "A-V4-2026-09-11"


@dataclass(frozen=True)
class UnitConvention:
    length: str = "m"
    time: str = "s"
    temperature_field: str = "degC"
    arrhenius_temperature: str = "K"
    dry_basis_moisture: str = "kg_water/kg_dry_solid"
    diffusivity: str = "m^2/s"
    heat_transfer_coefficient: str = "W/(m^2*K)"
    mass_transfer_coefficient: str = "m/s"


UNITS = UnitConvention()

ASSUMPTIONS: dict[str, str] = {
    "H1": (
        "药材均质、各向同性、轴对称，径向传热传质占主导；忽略轴向非均匀和端面交换，"
        "圆柱长度固定为0.25 m。本模型的全域终点仅指该一维径向模型的全域终点。"
    ),
    "H2": (
        "干物质无损。Q1-Q3固体骨架固定且干固体密度rho_d在空间上均匀；Q4初始rho_d均匀，"
        "材料作径向均匀比例收缩、轴向长度不变，固体骨架速度v_s=xi*dR/dt。"
    ),
    "H3": (
        "含水率C按kg水/kg干物质定义，q_C=-D*grad(C)，物理水质量通量j_w=rho_d*q_C。"
        "rho_d由干固体连续性决定；Q4中J=(R/R0)^2且rho_d*J=rho_d0。因rho_d仅随时间变化，"
        "统一密度尺度在干基方程和归一化水量中约去，材料坐标下不再重复加入收缩对流或浓缩项。"
    ),
    "H4": (
        "题给rho_g*c_p仅解释为有效体积热储存系数，不默认rho_g=rho_d*(1+C)。"
        "主温度模型省略相变潜热、迁移水分携焓和形变做功，不能称为完整混合物焓模型。"
    ),
    "H5": (
        "附件1空气水分数值H_inf作为等效干基平衡含水率C_e的数值输入，并与题给h_m配套；"
        "这不表示空气湿度与药材干基含水率的物理定义天然相同。"
    ),
    "H6": (
        "环境实测区间内线性插值，4 h后采用50 degC与C_e=0.05；半径实测区间内线性插值，"
        "72 h后冻结在1.198 cm。域外延拓属于显式假设，后续须单独做敏感性分析。"
    ),
}


INITIAL_TEMPERATURE_C = 28.0
INITIAL_MOISTURE_DRY_BASIS = 2.55
INITIAL_RADIUS_M = 0.02
CYLINDER_LENGTH_M = 0.25
HEAT_TRANSFER_COEFFICIENT = 25.0
MASS_TRANSFER_COEFFICIENT = 8.0e-7
TARGET_MAX_MOISTURE = 0.15


def to_kelvin(temperature_c: float) -> float:
    """Only Arrhenius factors use Kelvin."""

    return temperature_c + 273.15
