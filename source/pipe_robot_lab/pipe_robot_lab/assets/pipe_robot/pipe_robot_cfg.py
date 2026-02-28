import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# USD_PATH = os.path.join(CURRENT_DIR, "../source/model/pipe_robot/USD/pipe_robot_rename/pipe_robot_rename.usd")
USD_PATH = os.path.join(CURRENT_DIR, "./usd/pipe_robot_rename.usd")

PIPE_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            # 适度阻尼可减少高频抖动，降低“压入网格后瞬间穿透”的概率
            linear_damping=0.2,
            angular_damping=0.4,
            # 限制刚体极限速度，避免单步位移过大导致漏检碰撞
            max_linear_velocity=5.0,
            max_angular_velocity=8.0,
            # 控制解穿透速度，减少接触时的不稳定弹飞/抖动
            max_depenetration_velocity=3.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # 提高求解迭代次数，增强复杂接触（轮子-管道）稳定性
            solver_position_iteration_count=24,
            solver_velocity_iteration_count=16,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.1, 0.7),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            r".*main_steer_.*": 0.0,  # 所有主动轮舵向
            r".*main_wheel_.*": 0.0,  # 所有主动轮轮速
            r".*up_arm_.*": 0.0,      # 所有上臂关节(机身与大臂连接点)
            r".*mid_arm_.*": 1.2,     # 所有中臂关节(大臂与小臂连接点)
            r".*tail_arm_.*": 0.0,    # 所有尾臂关节(小臂与末端辅助轮连接点)
            r".*assist_steer_.*": 0.0, # 所有辅助轮舵向
            r".*assist_wheel_.*": 0.0, # 所有辅助轮轮速
            r".*bend_.*": 0.0,         # 所有弯折变形关节
        },
        joint_vel={
            r".*": 0.0,
        }
    ),
    actuators={
        "steer": ImplicitActuatorCfg(
            joint_names_expr=[".*main_steer_.*", ".*assist_steer_.*"],
            effort_limit_sim={
                ".*main_steer_.*":    100.0,
                ".*assist_steer_.*":  100.0,
            },
            velocity_limit_sim=     5.0,
            # 转向关节降刚度并增阻尼，减少刚性挤压引起的穿透
            stiffness=              1500.0,
            damping=                150.0,
            # effort_limit_sim=.0,
        ),
        "wheel": ImplicitActuatorCfg( # 轮子: 适合速度控制 (Stiffness=0)
            joint_names_expr=[".*main_wheel_.*",".*assist_wheel_.*"],
            effort_limit_sim={
                ".*main_wheel_.*":    100.0,
                ".*assist_wheel_.*":  100.0,
            },
            velocity_limit_sim=     20.0,
            stiffness=              0.0,
            damping=                10.0,
            # effort_limit_sim=50.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*up_arm_.*",".*mid_arm_.*",".*tail_arm_.*"],
            effort_limit_sim={
                ".*up_arm_.*":    2000.0,
                ".*mid_arm_.*":   200.0,
                ".*tail_arm_.*":  4000.0,
            },
            velocity_limit_sim= {
                ".*up_arm_.*":    6.0,
                ".*mid_arm_.*":   6.0,
                ".*tail_arm_.*":  10.0,
            },
            stiffness=          5000.0,
            damping=            200.0,
        ),
        "bend": ImplicitActuatorCfg(
            joint_names_expr=[".*bend_.*"],
            effort_limit_sim=   500.0,
            velocity_limit_sim= 10.0,
            stiffness=          1000.0,
            damping=            100.0,
        )
    },

)