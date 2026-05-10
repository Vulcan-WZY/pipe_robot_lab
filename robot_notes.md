# Pipe Robot Lab - 项目背景与改进计划

> 本文档供新的开发环境/智能体快速理解项目全貌与当前工作节点。
> 最后更新：2026-05-09

---

## 一、项目背景

### 1.1 研究目标

本项目旨在开发一台具有多模态感知能力的 **外管道检测机器人** 的 **端到端强化学习控制系统**。期望机器人能够在管道外表面自主运动，并借助深度相机和IMU等传感器感知环境（管径突变、法兰、吊架、弯管等障碍），自主规划夹持与越障变形动作。

### 1.2 机器人结构

机器人采用 **前后夹持臂 + 中部变形模块** 的对称结构：

- **单侧夹持臂**：包含 3 组舵轮（1个主动轮 + 2个辅助轮），每组舵轮由轮电机和舵向电机联合控制
- **整机共计 6 组舵轮**（前3 + 后3），是机器人在管道表面运动的驱动来源
- **小臂 (mid_arm)**：调节小臂张角可适配不同直径管道，存在四连杆联动机构（mid_arm → tail_arm）
- **大臂 (up_arm)**：调节大臂张角可实现对管道表面的夹紧抱持
- **中部变形模块 (bend)**：两段弯折推杆，用于尺蠖式交替夹持越障

**关节/动作空间汇总（22维）**：

| 动作组 | 维度 | 说明 |
|--------|------|------|
| 6组舵轮 (SteerWheel) | 6×2=12 | 每组输出 [Vx, Vy]，内部自动完成swerve逆解 |
| 2组联动臂 (LinkedArm) | 2×2=4 | 控制mid_arm，自动联动tail_arm（四连杆） |
| 2组大臂 (up_arm) | 2×2=4 | 位置控制 |
| 2个弯折推杆 (bend) | 2×1=2 | 位置控制 |

### 1.3 传感器配置

| 传感器 | 位置 | 参数 | 用途 |
|--------|------|------|------|
| TiledCamera (back_cam) | 后侧夹持臂 BM_02_link_cam | 120×160, 深度图, 10Hz | 后方环境感知 |
| TiledCamera (front_cam) | 前侧夹持臂 FM_25_link_cam | 120×160, 深度图, 10Hz | 前方环境感知 |
| ContactSensor ×6 | 6个车轮 link | 50Hz | 轮子接触状态检测 |
| IMU（暂未启用） | 前后夹持臂 | 200Hz | 姿态/加速度 |

**注意**：实机上接触力传感器难以部署，因此管径等信息最终必须通过深度相机获取。

### 1.4 运动机理

- **管道适配**：通过调节小臂 (mid_arm) 张角适配不同直径管道
- **夹紧抱持**：通过大臂 (up_arm) 压紧管道表面
- **表面运动**：借助 6 组舵轮在管道外表面做全向运动
- **越障变形**：单侧夹持 → 中部变形模块变形 → 另一侧恢复抱持（尺蠖式交替）

---

## 二、技术架构

### 2.1 训练框架

- **仿真环境**：NVIDIA IsaacLab (Isaac Sim)
- **RL框架**：SKRL (支持 PPO)
- **环境类型**：Manager-Based RL Environment
- **算法**：PPO（Proximal Policy Optimization）
- **网络**：自定义多模态 CNN+MLP 融合网络（Actor-Critic，非对称AAC）

### 2.2 网络模型结构

位于 `source/.../agents/models/custom_skrl_model.py`：

```
前后深度图 [N, 2, H, W]
    │
    ▼
ResNet-like CNN (4层Conv + 4个ResBlock, GroupNorm, ELU)
    │
    ▼
Vision Projection (Linear + LayerNorm + ELU) → 256维
    │
    ├─── Concatenate ──────────────────────────┐
    │                                          │
    ▼                                          ▼
                                        本体观测 (policy_dim)
                                              │
                                        3层MLP (512→512→256, LayerNorm + ELU)
                                              │ 256维
                                              ▼
                    Fusion MLP (512→256, LayerNorm + ELU) ← 512维拼接
                              │
                              ▼
                    Actor: Linear → 22维动作 + log_std (GaussianMixin)
                    Critic: Linear → 1维价值 (DeterministicMixin)
```

特点：正交初始化，Actor最终层gain=0.01（初期低幅值输出），Critic采用非对称AAC（额外接收线速度特权信息）。

### 2.3 观测空间

**PolicyCfg（数值观测，拼接为向量）**：
- 舵角归一化位置、主轮/辅助轮速度
- 变形推杆位置、小臂位置
- 大臂力矩
- 前后基座四元数姿态、重力投影
- 6个接触状态（±1）
- 历史5步动作

**CameraCfg（视觉观测，字典形式）**：
- 前后深度图 [N, 1, H, W]，归一化到 [0, 1]

**CriticCfg（非对称特权观测）**：
- 继承 PolicyCfg + 前后基座线速度

### 2.4 奖励函数

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| survival_bonus | +200 | 存活至超时 |
| alive_penalty | +0.03 | 每步存活奖励 |
| fall_off_penalty | -200 | 掉管重罚 |
| stagnation_penalty | -200 | 卡死重罚（当前已关闭） |
| front/back_dia_matched | +2.0 | 六次多项式管径-小臂角度匹配，高斯核 sigma=0.1 |
| wheel_contact_reward | +2.0 | 接触轮数奖励（每轮+1，最高+6） |
| action_rate_l2 | -0.01 | 动作平滑正则 |
| joint_acc_l2 | -1e-7 | 抑制震荡 |
| progress_reward | 0（关闭） | 轴向里程（后续课程启用） |

### 2.5 终止条件

- 超时（episode_length_s=12秒，600步）
- 6轮全脱离接触（min_steps=100保护）
- 到达终点（已关闭）
- 卡死检测（已关闭）

### 2.6 课程学习系统

通过 `sh/start_curriculum.sh` → `sh/auto_train_loop.py` 实现外层循环：

- 每轮从 `auto_train.yaml` 配置的 `env_pipe_id_range` 区间内随机选取一根管道
- 训练固定 `max_iterations_per_round` 次PPO迭代后保存checkpoint退出
- 下一轮自动恢复最新权重，换新管道继续训练
- 附带自动清理历史PT文件的硬盘保护机制

### 2.7 管道几何工具

`funcs/pipe_geom_utils.py` 提供：
- `is_point_on_pipe`：世界坐标点投影到管路各段，返回局部坐标、段号+轴向进度、径向距离比
- `get_cached_pipe_info`：单帧缓存机制，同一物理步内多次调用只计算一次
- `get_target_relative_pose`：计算机器人在管面目标坐标系下的姿态偏差

管道资产以USD + JSON描述文件对的形式存储，JSON中每段管元包含 `transform(4×4)` 和 `info([编号, 类型, 直径D, 长度L, 前端偏转, 后端偏转])`。

---

## 三、当前训练阶段与状态

### 3.1 课程学习第一阶段

**目标**：机器人被放置在随机直径的直管道上方后，学会自主调节小臂张角 (mid_arm) 以适配当前管道直径，保持稳定夹持。

**状态**：尚未实现稳定收敛。初步分析原因包括奖励信号设计问题、动作空间过大、训练效率配置不佳等。

### 3.2 当前训练配置（auto_train.yaml）

| 参数 | 值 |
|------|------|
| framework | skrl |
| task | Template-Pipe-Robot-Lab-v1 |
| num_envs | 1 |
| max_iterations_per_round | 100 |
| total_rounds | 100 |
| env_pipe_id_range | [91, 170] |
| headless | false |
| episode_length_s | 12.0 |
| 深度相机分辨率 | 120×160 |

---

## 四、改进计划

### 4.1 近期优化（解决当前无法收敛的问题）

#### 4.1.1 动作空间约束

**目标**：保持22维动作空间不变（便于后期复用），但通过奖励惩罚隐式约束非核心维度。

- 对 **6组舵轮** 的动作输出施加较大的L2惩罚（如 weight=-5.0），迫使策略在这些维度输出接近0
- 对 **弯折推杆 (bend)** 偏离初始值的动作施加惩罚
- 考虑将单侧左右大臂 (up_arm_01/02) 合并为同一指令（即强制两者数值相等），减少有效自由度
- **后续课程阶段**：逐步降低这些惩罚权重，让策略逐渐"解锁"这些自由度

#### 4.1.2 奖励函数调参

| 调整项 | 当前值 | 目标值 | 理由 |
|--------|--------|--------|------|
| `dia_matched_reward` 的 sigma | 0.1 | **0.3 ~ 0.5** | 当前高斯核太窄（偏离0.2rad奖励衰减至0.018），策略初期无法感受到梯度方向 |
| `wheel_contact_reward` 权重 | 2.0 | **0.5 ~ 1.0** | 接触奖励量级远大于角度匹配，策略易陷入"粗暴夹死"局部最优 |
| `survival_bonus` 权重 | 200 | **50 ~ 100** | 存在稳态增益 (alive_penalty)，终局大额奖励易导致"什么都不做"策略 |
| `dia_matched_reward` 启用条件 | 无条件 | **单侧接触数 ≥ 2 时才启用** | 避免在空中时给出无意义的角度匹配信号 |

#### 4.1.3 训练效率提升

| 参数 | 当前值 | 目标值 | 理由 |
|------|--------|--------|------|
| `num_envs` | 1 | **16 ~ 32** | PPO依赖大规模并行采样，1个环境极低效 |
| `episode_length_s` | 12秒 | **4 ~ 6秒** | 夹持任务不需要12秒，缩短后加快训练周转 |
| `max_iterations_per_round` | 100 | **300 ~ 500** | 给策略足够时间在单管上探索稳定解 |
| 深度相机分辨率 | 120×160 | **60×80** | 管径是全局几何特征不需高分辨率，降低显存占用 |

### 4.2 中期改进方向

#### 4.2.1 辅助损失 (Auxiliary Loss) 加速视觉特征学习

在CNN输出后增加辅助预测头，直接监督预测管道直径：

```
CNN → vision_proj → 256维
                      │
                      ├── 正常流程：融合 → Actor/Critic
                      │
                      └── 辅助头：Linear(256, 1) → 预测管道直径（MSE Loss）
```

- 监督信号来自JSON中的管径真值（训练时可获取），推理时不使用
- 作用：加速CNN从深度图中提取管径几何信息，避免依赖PPO稀疏回报梯度的缓慢反传
- 这是视觉RL中的经典技巧（Dreamer、SAC-AE、DRQ等工作均采用类似方法）

#### 4.2.2 Teacher-Student 架构

这是 Legged Locomotion 领域（ANYmal、MIT Cheetah等）最成功的 sim-to-real 方案，与本项目高度契合：

- **Teacher**：用特权信息（管道直径真值、接触力真值、管元编号等仿真可获取但实机不可获取的信息）训练纯MLP策略，无需视觉分支，收敛极快
- **Student**：以Teacher的动作输出为监督信号，训练带CNN的策略网络（行为克隆 + RL微调）

优势：
- Teacher训练无需CNN，num_envs可开到数千，几分钟内收敛
- Student从Teacher的稳定行为中学习，解决了"CNN长期收不到有效梯度"的难题
- 实机不可用的传感器（接触力）信息在Teacher阶段充分利用，Student阶段仅依赖实机可获取的传感器（深度相机、IMU）

#### 4.2.3 环境内动态换管

避免当前"杀进程→重启IsaacLab→重建场景"的高开销换管机制：

- 在 `EventCfg` 中注册 `mode="reset"` 事件，episode reset时动态更新管道参数
- 场景初始化时一次性加载多根不同直径管道，每个env分配不同管道
- 实现在一个训练步内同时学习多种管径适配

---

## 五、工程目录结构速查

```
pipe_robot_lab/
├── source/pipe_robot_lab/pipe_robot_lab/
│   ├── assets/
│   │   ├── pipe_robot/            # 机器人URDF/USD模型、mesh、配置
│   │   │   ├── pipe_robot_cfg.py  # 机器人Articulation配置
│   │   │   ├── usd/               # 机器人USD文件
│   │   │   └── meshes/            # STL网格文件
│   │   └── pipe_env/              # 管道环境资产
│   │       ├── usd/               # 管道USD文件 (000001~000200)
│   │       ├── json/              # 管道JSON描述文件
│   │       └── pipe_generator.py  # 管道生成器
│   ├── tasks/manager_based/pipe_robot_lab/
│   │   ├── pipe_robot_lab_env_cfg.py  # 环境顶层配置（场景、传感器、管道加载）
│   │   ├── mdp/
│   │   │   ├── actions.py         # 动作空间定义（舵轮、联动臂、大臂、弯折）
│   │   │   ├── observations.py    # 观测空间定义（数值+视觉+Critic特权）
│   │   │   ├── rewards.py         # 奖励函数（管径匹配、接触、存活等）
│   │   │   ├── terminations.py    # 终止条件（超时、脱离、卡死、到达终点）
│   │   │   └── events.py          # 事件配置（reset时的位姿/关节重置）
│   │   ├── agents/
│   │   │   ├── models/
│   │   │   │   └── custom_skrl_model.py  # 多模态CNN+MLP网络
│   │   │   ├── skrl_vision_ppo_cfg.yaml  # SKRL PPO超参数配置
│   │   │   └── rsl_rl_ppo_cfg.py         # RSL_RL配置（备选）
│   │   └── network/               # 网络模块（预留）
│   └── funcs/
│       ├── pipe_geom_utils.py     # 管道几何计算（点-管投影、姿态偏差、缓存）
│       └── pose_utils.py          # 四元数/旋转矩阵工具函数
├── scripts/
│   ├── skrl/train.py              # SKRL训练入口（含VisionIsaacLabWrapper）
│   ├── skrl/play.py               # 推理/评估脚本
│   └── rsl_rl/                    # RSL_RL训练/推理脚本（备选）
├── sh/
│   ├── start_curriculum.sh        # 课程学习启动脚本
│   ├── auto_train_loop.py         # 自动训练循环（换管+断点续训）
│   └── config/auto_train.yaml     # 训练配置（环境数、迭代、管道ID区间等）
└── logs/                          # Tensorboard日志与checkpoint存储
```

---

## 六、关键文件快速索引

| 需求 | 文件路径 |
|------|---------|
| 修改奖励函数 | `source/.../mdp/rewards.py` |
| 修改观测空间 | `source/.../mdp/observations.py` |
| 修改动作空间 | `source/.../mdp/actions.py` |
| 修改终止条件 | `source/.../mdp/terminations.py` |
| 修改网络模型 | `source/.../agents/models/custom_skrl_model.py` |
| 修改PPO超参数 | `source/.../agents/skrl_vision_ppo_cfg.yaml` |
| 修改环境/场景配置 | `source/.../pipe_robot_lab_env_cfg.py` |
| 修改训练循环配置 | `sh/config/auto_train.yaml` |
| 修改训练入口 | `scripts/skrl/train.py` |
| 管道几何工具 | `source/.../funcs/pipe_geom_utils.py` |