import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# 获取当前文件目录并构建USD模型路径
# Get current file directory and construct USD model path
current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/PF_TRON1A/PF_TRON1A.usd")

# 定义双足机器人的关节配置
# Define the articulated robot configuration for the biped
POINTFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # 刚体物理属性配置
            # Rigid body physics properties configuration
            rigid_body_enabled=True,        # 启用刚体物理 / Enable rigid body physics
            disable_gravity=False,          # 不禁用重力 / Don't disable gravity
            linear_damping=0.0,             # 线性阻尼系数 / Linear damping coefficient
            angular_damping=0.0,            # 角度阻尼系数 / Angular damping coefficient
            max_linear_velocity=1000.0,     # 最大线性速度限制 / Maximum linear velocity limit
            max_angular_velocity=1000.0,    # 最大角速度限制 / Maximum angular velocity limit
            max_depenetration_velocity=1.0, # 最大去穿透速度 / Maximum depenetration velocity
        ),
        # 关节根部属性配置
        # Articulation root properties configuration
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,       # 启用自碰撞检测 / Enable self-collision detection
            solver_position_iteration_count=4,  # 位置求解器迭代次数 / Position solver iteration count
            solver_velocity_iteration_count=4,  # 速度求解器迭代次数 / Velocity solver iteration count
        ),
        activate_contact_sensors=True,   # 激活接触传感器 / Activate contact sensors
    ),
    # 机器人初始状态配置（弧度）
    # Robot initial state configuration (in radians)
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),   # 初始位置 (x, y, z) / Initial position (x, y, z)
        joint_pos={
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
            "foot_L_Joint": 0.0,
            "foot_R_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # 执行器配置 - 定义如何控制机器人关节
    # Actuator configuration - defines how to control robot joints
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit=300,      # 最大输出力矩 (N⋅m) / Maximum output torque (N⋅m)
            velocity_limit=100.0,  # 最大角速度 (rad/s) / Maximum angular velocity (rad/s)

            # 各关节刚度参数 - 控制位置跟踪精度
            # Joint stiffness parameters - controls position tracking accuracy
            stiffness={
                "abad_L_Joint": 40.0,
                "abad_R_Joint": 40.0,
                "hip_L_Joint": 40.0,
                "hip_R_Joint": 40.0,
                "knee_L_Joint": 40.0,
                "knee_R_Joint": 40.0,
            },

            # 各关节阻尼参数 - 控制运动平滑性
            # Joint damping parameters - controls motion smoothness
            damping={
                "abad_L_Joint": 2.5,
                "abad_R_Joint": 2.5,
                "hip_L_Joint": 2.5,
                "hip_R_Joint": 2.5,
                "knee_L_Joint": 2.5,
                "knee_R_Joint": 2.5,
            },
        ),
    },
)
