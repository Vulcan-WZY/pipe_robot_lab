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
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.1, 0.4),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            r".*main_steer_.*": 0.0,  # 所有主动轮舵向
            r".*main_wheel_.*": 0.0,  # 所有主动轮轮速
            r".*up_arm_.*": 0.0,      # 所有上臂关节(机身与大臂连接点)
            r".*mid_arm_.*": 0.8,     # 所有中臂关节(大臂与小臂连接点)
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
            stiffness=              5000.0, # 提高刚度确保位置准确
            damping=                100.0,
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
                ".*mid_arm_.*":   2000.0,
                ".*tail_arm_.*":  5000.0,
            },
            velocity_limit_sim= {
                ".*up_arm_.*":    10.0,
                ".*mid_arm_.*":   10.0,
                ".*tail_arm_.*":  20.0,
            },
            stiffness=          5000.0,
            damping=            100.0,
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