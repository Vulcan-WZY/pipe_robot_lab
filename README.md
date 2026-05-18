# pipe_robot强化学习训练环境

## 模型文件备注

导入IsaacSim的环节遇到了太多问题，最终还是确定无法实现闭链结构的机器人，准备改用开链机器人了，中间涉及到了多个URDF文件，特此说明：

- `pipe_robot.urdf`: 原生SolidWorks导出的，包含所有的完整link与joint，同时采用的是原生的基于pickage命名的STL文件地址路径
- `pipe_robot_full.urdf`：基于原生的urdf文件，仅仅将其中的相对路径改为了绝对路径
- `pipe_robot_mini.urdf`: 最终导入IsaacLab中使用的，去除了无法形成闭链的几个连接点以及link，整体为最简形式
- `pipe_robot_rename.urdf`: 基于mini版本进一步迭代而来,修改了其中的关节名称,  更合理的命名使得便于在IsaacLab中统一配置以及统一控制

# 常用调试指令

- 启动Demo程序并打开相机：`python ./06_pipe_ctrl.py --enable_cameras`
- 开启tensorboard查看训练过程: `tensorboard --logdir logs/skrl/pipe_robot_vision_ppo --port 6006`
- 开启训练：`python ./scripts/skrl/train.py --task Template-Pipe-Robot-Lab-v1 --headless`
- 测试训练后的模型运行推理：`python ./scripts/skrl/play.py --task Template-Pipe-Robot-Lab-v1 --checkpoint ./logs/skrl/pipe_robot_vision_ppo/2026-03-24_10-15-14_ppo_torch/checkpoints/best_agent.pt  --video --video_length 400 --headless --enable_cameras`

# 仓库部署流程

## 必要包版本

| name | version |
| ---- | ------- |
| SKRL | 2.0.0   |
|      |         |

## 安装IsaacLab环境

1. 创建conda环境: conda create -n isaaclab python=3.11 -y
2. 激活刚刚创建的isaaclab环境: conda activate isaaclab
3. 更新环境中的pip: pip install --upgrade pip
4. 安装对应版本的pytorch: `pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128`
5. 通过pip安装IsaacSim: `pip install --upgrade "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com`
6. clone仓库: `git clone https://github.com/isaac-sim/IsaacLab.git`
7. 在conda环境+IsaacLab仓库目录下, 自动安装IsaacLab全部扩展: ./isaaclab.sh --install
8. 激活环境验证安装: python scripts/tutorials/02_scene/create_scene.py

## 安装功能包

```bash
# use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python -m pip install -e source/pipe_robot_lab
```

# 使用记录

## 最常用调试脚本

在工作空间根目录运行 `python ./scripts/persional/06_pipe_ctrl.py`就可以打开最最常用的UI渲染的仿真环境, 这个环境中, 可以使用键盘按键以及游戏手柄来控制机器人运动与变形. 同时这个文件中预留了很多可以直接按照格式打印Observations以及reward等等常用数值的接口, 很适合调试.

- 截止260506, 已经引入了基于输入的ID号闭区间随机片选管路USD资产的能力,  跑这个python测试文件时环境会自动根据 `auto_train.yaml`文件中的 `env_pipe_id_range`数值随机选择已有的管道资产. 相应的极限情况, 如果设置为 `[1,1]`, 就相当于定死了只拿1号管路作为测试用例了.

## 网格生成器与删除器

### 生成器

使用网络生成器可以自动在 `./source/pipe_robot_lab/pipe_robot_lab/assets/pipe_env/`路径中生成一系列管路的STL文件, 同时也会在json文件夹中生成相应的管道编号所对应的描述文件.

可以修改 `pipe_env`中的配置文件 `pipe_env.yaml`来详细调节生成器工作时的各种参数. 每个参数文件中都有详细的描述, 不再做过多解释.

配置好yaml文件中的参数后, 在项目的根目录运行 `./sh/generate_pipe.sh`就可以自动开始生成管道了. 生成完成后,  可以动手检查一下是否有问题. 如果没问题就可以进一步运行 `./sh/stl2usd.sh`来批量将新生成的STL文件转换为USD资产文件, 在转换过程中会借助IsaacLab为管路打上随机化的摩擦因数.

### 删除器

`./sh/clear_pipe.sh 30 40`可以将对应闭区间编号范围内的管道的STL文件, JSON文件, 以及USD文件全部删除

# 开发记录

## 2026.05

### 0504~0505

#### 修改机器人reset时的加载位置

这里是在开始构建课程学习框架初期, 由于需要借助外部脚本重置IsaacLab整个训练流程, 因此计划优化一下之前一贯以来的"机器人从空中丢到管道上"的方式, 改为根据管道yaml描述文件的第一节信息, 将机器人**尽可能准确地放置在最开始的管道表面**上.

在现有方案基础上,  在开头随机选择文件的部分即时读取第一节管道的直径,  然后在后续配置管道环境时显示更新机器人的加载位置. 注意经过测量, 这里的由机械结构决定的偏置值为**0.031m**.

#### 解决并行环境共用同一个管道问题

暂时搁置了, 本来想实现在在线训练过程中没reset一次, 就更新一次管道网路的, 但是发现这样是较难实现且对于IsaacLab并行环境来说是比较不稳定的.  决定只靠多轮启动器脚本来实现.

#### 引入外部多轮启动器脚本

这个工作其实之前在最初完善好管道网格随机生成器时就已经想做了, 由于IsaacLab的复杂机制, 使得很难稳定实现在线训练过程中每次reset都换一次环境. 为此计划编写一个私有外部训练脚本, 训练一定轮数后就自动重启IsaacLab更换环境.

也就是做一个**自动多轮启动器**, 利用Bash脚本和算法"断点续训"的特性来解决问题.现在这个多轮启动器本质上是通过python实现的, 只是通过sh来启动. 对应的启动文件就是 `start_curriculum.sh`脚本, 这个脚本事实上对应启动的是 `auto_train_loop.py`文件, 该文件自动读取了存储在相对路径config下的 `auto_train.yaml`配置文件. 该文件中配置了训练需要用到的多种参数与选项.

#### 引入管道直径与夹持角度匹配评价奖励函数

目前计划设计的管线是这样, 由外部调用输入后侧主机架节点(BM_01)或前侧节点(FM_24), 然后函数内部会先判断这一侧夹持臂是否与管道表面有接触(这里我认为只要三个驱动轮有一个有接触力就认为是接触), 如果有再进一步借助 `pipe_geom_utils` 工具包中的匹配函数, 解算出当前单侧夹持臂所匹配到的管元, 正如这段函数注释中写的:

```python
target_angle(dia) = -3.1762   38.3970 -184.8731  453.9609 -593.4035  408.7483  -85.6488
```

采用一个六次多项式来讲管道直径映射为最优的小臂角度(mid_arm), 这个六次多项式的输入为管道直径 mm/100, 如直径0.36m映射为3.6, 输出的就是理论最优夹持角度. 需要注意, 这里输出的角度是纯粹的几何意义上的. 因此还需要将程序中的角度做一个偏置. 这个偏置我可以告诉你是57.63°, 也即仿真读到的mid_arm角度为0时, 实际上几何角度已经是这个值了. 此外, 模型中mid_arm的正方向与几何正方向一致, 所以只需要做偏置即可.

至此就可以得到误差err了, 关于奖励函数的取值设计, 我希望函数本身有这样的特征: 当err的绝对值在0附近时, 奖励函数的整体数值较大, 然后当err越来越大时 , 奖励函数的值迅速衰减逐步到0, 感觉可以利用对数函数的性质

$$
W(err)=e^{-\frac{err^2}{\sigma^2}}
$$

如上:

- 当err为0时, 取得最大理想奖励值1
- 当err变大时, 快速向0衰减
- 预设了一个参数$\sigma$, 默认0.1大约5.7°, 如果想让机器人更严苛的接近标准值可以调小

### 0506

智元入职第一天

#### 改进了网格删除器与自启动脚本

现在的网格删除器支持输入一个闭区间范围, 表示管元的编号, 可以定向删除指定区间内的所有管道, 使用方式如: `bash sh/clear_pipe.sh 30 40`

同时在自启动脚本配置文件 `auto_train.yaml`中也新增了一个 `env_pipe_id_range`条目, 可以配置自启动小轮随机选用的管道网路范围. 同样是给一个闭区间

### 0510

#### 提升收敛性做的改动

**目标**: 解决课程学习第一阶段（夹持管径适配）无法收敛的问题。

**rewards.py 改动**:

| 参数                                | 原值  | 新值          | 理由                                                 |
| ----------------------------------- | ----- | ------------- | ---------------------------------------------------- |
| `dia_matched_reward` sigma        | 0.1   | 0.4           | 高斯核过窄，初始偏差~0.5rad时奖励≈0，策略无梯度方向 |
| `dia_matched_reward` 新增接触掩码 | 无    | min_contact=2 | 空中时不应给出匹配奖励信号                           |
| `wheel_contact_reward` weight     | 2.0   | 0.5           | 避免接触奖励远大于匹配奖励导致粗暴夹死局部最优       |
| `wheel_contact_reward` min_steps  | 200   | 100           | 适配缩短后的episode                                  |
| `survival_bonus` weight           | 200.0 | 50.0          | 有稳态 alive 奖励即可，终局大额易诱导什么都不做      |
| 新增 `steer_wheel_lock`           | -     | weight=-5.0   | 惩罚舵轮12维动作输出，课程阶段锁轮                   |
| 新增 `bend_lock`                  | -     | weight=-3.0   | 惩罚 bend 关节偏离零位，冻结弯折自由度               |

**pipe_robot_lab_env_cfg.py 改动**:

| 参数                   | 原值     | 新值   | 理由                                      |
| ---------------------- | -------- | ------ | ----------------------------------------- |
| 深度相机 height×width | 120×160 | 60×80 | 管径是低频几何特征，降分辨率节省显存      |
| episode_length_s       | 12.0     | 5.0    | 夹持任务不需要12秒，缩短后加快episode周转 |
| num_envs 默认值        | 1        | 16     | PPO依赖大规模并行采样                     |

**terminations.py 改动**:

| 参数                        | 原值 | 新值 | 理由                          |
| --------------------------- | ---- | ---- | ----------------------------- |
| lost_all_contacts min_steps | 100  | 50   | episode缩短后保护期按比例下调 |

**auto_train.yaml 改动**:

| 参数                     | 原值  | 新值 | 理由                     |
| ------------------------ | ----- | ---- | ------------------------ |
| num_envs                 | 1     | 16   | PPO并行采样效率          |
| max_iterations_per_round | 100   | 400  | 单管上给策略充分探索时间 |
| headless                 | false | true | 无头模式提升训练吞吐     |

### 0512

#### 修复多轮训练 timestep 编号冲突

**问题**: SKRL 新版本的 Trainer 循环硬编码 `range(self.cfg.timesteps)`，始终从 0 开始计数，导致每个小轮训练都从 timestep=0 开始，TensorBoard 数据相互覆盖。

**解决方案**:

1. `trainer["timesteps"]` 只设为本轮的增量步数，不再累加历史步数
2. 通过 monkey-patch Agent 的 `write_tracking_data` 和 `write_checkpoint` 方法，注入全局 timestep 偏移量

效果: 每轮进度条只跑本轮增量步数，TensorBoard x 轴全局连续，checkpoint 文件名包含全局编号。

#### 持久化训练会话 (tmux)

**需求**: 通过 Tailscale 远程启动训练后，断联不影响训练进程，重连后可查看进度或停止训练。

**方案**: 将 `start_curriculum.sh` 改造为基于 tmux 的会话管理脚本，同时自动启动 TensorBoard 后台会话。

**使用方法**:

| 命令                                | 作用                          |
| ----------------------------------- | ----------------------------- |
| `./sh/start_curriculum.sh`        | 启动后台训练 + TensorBoard    |
| `./sh/start_curriculum.sh attach` | 连接到训练终端查看实时进度条  |
| `./sh/start_curriculum.sh tb`     | 连接到 TensorBoard 终端       |
| `./sh/start_curriculum.sh status` | 查看训练和TensorBoard运行状态 |
| `./sh/start_curriculum.sh stop`   | 终止训练和TensorBoard         |

**关键操作**: 在 attach 进入 tmux 会话后，按 `Ctrl+B` 再按 `D` 可安全脱离会话（训练继续运行）。

### 0513

#### 重构多轮训练的全局 timestep 管理

**问题**: 之前的方案依赖从 checkpoint 文件名中正则提取 step 编号作为全局偏移量。但 `cleanup_old_checkpoints` 按时间戳删除旧文件后，真正具有最大编号的 checkpoint 可能被误删，导致下一轮提取到的偏移量骤降，进而 checkpoint 文件名重复覆盖，TensorBoard 数据出现从原点到末端的直线和 step 重叠。

**解决方案**: 将全局 timestep 的管理职责**从 `train.py` 上移至 `auto_train_loop.py`**。

核心改动:

1. **`auto_train_loop.py` 重写**:

   - 启动时一次性扫描 log 目录，按文件名编号（而非时间戳）找到最大 step 作为全局起点
   - 每轮结束后，外层精确计算 `global_timestep += steps_per_round`
   - 通过新增的 `--timestep_offset` 命令行参数将偏移量传给 `train.py`
   - checkpoint 清理改为**按文件名编号降序排列**保留最新 N 个，不再依赖时间戳
   - 每轮结束后验证预期的 checkpoint 是否存在，不存在时 fallback 检测并自动修正 global_timestep
2. **`train.py` 新增 `--timestep_offset` 参数**:

   - 优先使用外部传入的偏移量，fallback 才从 checkpoint 文件名提取
   - 确保 monkey-patch 的偏移量始终由外层权威计算，不受 checkpoint 清理影响

#### 新增 Play 推理演示脚本

新增文件:

- `sh/config/play.yaml` — 推理配置 (checkpoint路径、渲染模式、视频录制参数)
- `sh/play_runner.py` — 读取 yaml、搜索 checkpoint、组装命令行参数
- `sh/play.sh` — Shell 启动入口

0516

#### 三层 NaN/Inf 防御体系（阻断仿真异常→训练的传染链）

**问题**: 训练过程中偶发 USD 底层 `Orthonormalize did not converge` 警告（仿真矩阵退化），一旦出现就会导致后续所有训练轮次失效。根因分析确认传染链条为：

```
仿真矩阵退化 → 观测值出现 NaN/Inf → PPO 用垃圾数据更新 → 网络权重被污染
→ checkpoint 保存污染权重 → 下一轮加载污染权重 → 恶性循环
```

**核心漏洞**: 全链路（观测→前向→输出→checkpoint）无任何 NaN/Inf 检测与阻断机制。

**解决方案**: 三层纵深防御，在污染链条的每个关键节点设置阻断点。

**L1 — 观测层阻断** (`scripts/skrl/train.py` — `VisionIsaacLabWrapper`):

- 在 `step()` 和 `reset()` 返回观测前，遍历所有观测子张量检测 NaN/Inf
- 检测到异常时用 `torch.nan_to_num` 替换为安全值（0），并记录 warning 日志
- 日志有上限控制（最多 20 条），防止刷屏
- 覆盖本体观测 (policy)、视觉观测 (camera)、Critic 特权观测全部子空间

**L2 — 网络层阻断** (`custom_skrl_model.py` — `CustomActorCritic.compute()`):

- 本体观测 (prop_input) 进入 MLP 前检测 NaN/Inf 并替换
- Actor 输出动作后检测 NaN/Inf，检测到时归零并记录 warning（上限 10 条）
- Critic 输出价值后同样检测，防止异常价值估计污染 GAE 计算

**L3 — Checkpoint 层阻断** (`sh/auto_train_loop.py`):

- 新增 `validate_checkpoint_weights()` 函数：加载 .pt 文件递归扫描所有权重张量
- 每轮训练结束后，验证最新 checkpoint 是否含 NaN/Inf
- 检测到污染 checkpoint 时：自动删除损坏文件，回退到上一个已验证安全的 checkpoint
- 维护安全 checkpoint 栈（最多 5 层），支持多级回退
- 初始 checkpoint 也经过验证，启动即发现残留污染
- 所有安全 checkpoint 都损坏时，fatal error 退出避免静默扩散

**关键设计考量**:

- L1 + L2 解决**本轮训练被污染**的问题（实时阻断）
- L3 解决**即使本轮污染了，也不影响下一轮**的问题（事后隔离）
- 三层互相独立，任一层失效不影响其他层的保护能力
- 日志上限控制避免正常训练时日志爆炸
- checkpoint 验证使用 CPU 加载，不影响 GPU 显存

#### 训练脚本前台/后台模式切换

**需求**: debug 模式下需要直接在终端观察训练启动输出，而非通过 tmux attach 间接查看。

**方案**: 在 `sh/config/auto_train.yaml` → `debug` 段新增 `background` 开关，`start_curriculum.sh` 启动时读取该值决定运行模式。

**使用方式**:

| `background` 值 | 行为                          | 适用场景     |
| :---------------: | ----------------------------- | ------------ |
|  `true` (默认)  | tmux 后台运行，断联不中断     | 长期正式训练 |
|     `false`     | 终端前台直接运行，Ctrl+C 中断 | 调试验证改动 |

切换到前台模式时，`start_curriculum.sh attach` 和 `stop` 命令会自动适配（attach 提示无需连接，stop 只停 TensorBoard）。

**涉及文件**:

- `sh/config/auto_train.yaml` — 新增 `debug.background` 字段
- `sh/start_curriculum.sh` — 读取 yaml 配置，分支执行前台/后台逻辑

#### NaN 溯源调试工具 (nan_trace)

**需求**: 三层 NaN 防御能阻断污染传播，但无法回答"NaN 最初是从哪个传感器/关节/操作产生的"。需要事后可分析的完整证据。

**方案**: 在 `debug` 段新增两个开关：

| 配置项             |  默认值  | 作用                                                                                  |
| ------------------ | :-------: | ------------------------------------------------------------------------------------- |
| `nan_trace`      | `true` | 首次检测到 NaN 时保存完整张量快照到 `<log_dir>/nan_traces/` 目录                    |
| `anomaly_detect` | `false` | 启用 PyTorch `autograd.set_detect_anomaly(True)`，定位反向传播中产生 NaN 的具体操作 |

**nan_trace 快照包含**:

- **观测快照** (`obs_nan_<key>_<timestamp>.pt`): 原始张量 (清洗前)、NaN/Inf 数量、前 50 个 NaN 维度索引
- **动作快照** (`act_nan_<timestamp>.pt`): 原始动作张量 (归零前)、NaN 维度索引、当前 log_std 值

**增强日志**: 检测到 NaN 时日志会同时打印前 20 个具体的 NaN 维度索引，例如:

```
[OBS-NaN] NaN detected in observation key 'policy',
  NaN dims: [[0, 15], [0, 38], [0, 72], ...], replaced with 0.
```

根据索引即可对照 `observations.py` 中的拼接顺序定位到具体是哪个关节位置/四元数分量/接触传感器。

**事后分析流程**:

1. 检查日志中的 NaN 维度索引，对照观测拼接顺序确定问题传感器/关节
2. 加载 `.pt` 快照文件 (`torch.load("obs_nan_policy_*.pt")`)，查看完整张量值
3. 如 NaN 出现在动作中，说明网络前向本身出问题，可开启 `anomaly_detect` 重跑定位

**涉及文件**:

- `sh/config/auto_train.yaml` — 新增 `debug.nan_trace`、`debug.anomaly_detect`
- `scripts/skrl/train.py` — 新增 CLI 参数、快照保存逻辑、逐维度 NaN 索引日志
- `sh/auto_train_loop.py` — 传递新参数
- `custom_skrl_model.py` — 动作 NaN 快照保存

### 0518

#### 第一阶段夹持课程的收敛性改造

本轮改动的目标不是直接追求多管径泛化，而是先把**固定单管道场景下的夹持课程训练稳定下来**，包括：

- 避免继续出现历史上那种 checkpoint / actor 输出 NaN 污染
- 让策略先学会 `up_arm` 收紧 + `mid_arm` 管径适配 + 稳定接触
- 在不改 actor 输出维度的前提下，通过动作课程冻结无关自由度

当前阶段的具体策略如下：

1. **补齐观测闭环**

- 在 policy observation 中新增 `up_arm` 归一化位置
- 暂时保留原有 `up_arm` 力矩观测
- 新增 6 个轮子的连续接触力观测，用于训练阶段提供更平滑的接近信号
- 接触的二值状态 (`1 / -1`) 仍然保留，便于后续保持与真机部署的抽象一致性

2. **引入更稠密的弱引导 reward**

- 新增 `up_arm_pre_grasp_reward`
  - 只在尚未形成有效接触前生效
  - `up_arm` 越往“夹紧”方向运动，给越弱但连续的正向引导
  - 一旦形成接触自动关闭，避免无脑把大臂夹到极限
- 将 `dia_matched_reward` 从硬门控改为**软门控**
  - 接触后给完整奖励
  - 接触前仍保留较小比例的几何匹配引导

3. **动作课程冻结**

- 当前阶段保持 actor 的 22 维输出结构不变，保证 checkpoint 结构以后可以直接续训
- 但在动作执行层：
  - 6 组 `steer_wheel` 全部冻结
  - `bend_01 / bend_02` 全部冻结
- 这样当前阶段只让策略真正学习：
  - `mid_arm`
  - `up_arm`

4. **深度输入数值修正**

- 修正了深度图进入网络前的重复归一化问题
- 之前 observation 已经做过 `/10.0` 归一化，模型侧又做了一次 `/10.0`
- 现在模型侧只做 NaN/Inf 清洗和 `[0, 1]` clamp，不再重复缩放

5. **当前阶段的训练验收标准**

- fresh start 与 resume 都不应再出现新的 NaN 污染
- 单根管道上训练应逐步收敛，而不是长期剧烈发散
- `Episode / Total timesteps` 应整体向超时上限靠近
- `dia_match` 与接触相关项应从“几乎不激活”变为“稳定可见”

---

## Future Roadmap

这一节专门记录**中期异构环境改造规划**，避免后续上下文丢失时遗忘当前讨论结果。

### 目标

让每个并行 env 都能够加载**不同的管道 USD + 对应 JSON 描述文件**，从而在同一轮 PPO 采样中同时见到多种管径/多种管网，提高真正的多管径适配能力。

### 当前技术障碍

当前工程虽然已经能在每轮训练启动时随机选择一根管道，但本质上仍然是：

- 本轮所有 env 共用同一个 `usd`
- 本轮所有 reward / observation / termination 共用同一份 `pipe_info / pipe_transform_inv`

也就是说，当前几何匹配工具链默认是：

- `points`: `[num_envs, 3]`
- `pipe metadata`: `[num_segments, ...]`

而不是未来需要的：

- `points`: `[num_envs, 3]`
- `pipe metadata`: `[num_envs, num_segments, ...]`

### 中期改造步骤

1. **场景层异构化**

- 将 `InteractiveSceneCfg.replicate_physics` 切换为 `False`
- 允许每个 env 持有独立资产实例，而不是基于 `env_0` 的物理复制
- 为每个 env 分别指定管道 `usd_path`

2. **管道元数据改造**

- 将当前单份 `pipe_info / pipe_transform / pipe_transform_inv` 改造为按 env 存储
- 支持每个 env 对应不同长度、不同段数的管网描述
- 需要设计 padding / mask 或 ragged 访问方案

3. **几何查询函数重构**

- 重写 `pipe_geom_utils.py` 中的几何匹配入口
- 让 `get_cached_pipe_info()`、`get_target_relative_pose()`、`is_point_on_pipe()` 支持按 env 取各自 metadata
- 当前单帧缓存逻辑也要升级为按 env/step 管理

4. **奖励与终止项升级**

- `dia_match`
- `reach_goal`
- `progress_reward`
- `pose_error`

以上所有依赖 `pipe_info` 的项都要切换到 per-env metadata 查询

5. **训练策略**

- 异构 env 改造建议单独开分支进行
- 先在当前单管课程阶段稳定收敛
- 再迁移到多 env 随机管道泛化训练

### 保留的原则

- 真机上难以可靠拿到连续接触力，因此最终部署时不应强依赖接触力作为唯一核心信号
- 但训练阶段允许使用更平滑的几何/接触辅助信号，帮助学习形成稳定夹持
- 多管径泛化必须建立在“单管先能稳定收敛”的基础之上

使用方式:

| 命令                                | 作用                                             |
| ----------------------------------- | ------------------------------------------------ |
| `./sh/play.sh`                    | 使用 play.yaml 默认配置运行推理 (GUI + 实时步频) |
| `./sh/play.sh --headless --video` | 无头模式录制视频                                 |
| `./sh/play.sh --num_envs 4`       | 额外 CLI 参数直接透传给 play.py                  |

配置文件 `sh/config/play.yaml` 中可设置 checkpoint 名称 (支持 `best_agent.pt` / 绝对路径 / `agent_xxxxx.pt`)、渲染模式、视频保存路径等。

#### [严重BUG修复] CNN 视觉分支从未接收到真实深度图

**问题**: `custom_skrl_model.py` 的 `compute` 方法中第一行为 `obs_dict = inputs["states"]`。但在 SKRL 新版本的 `PPO.act()` 实现中，环境观测数据被放在 `inputs["observations"]` 键中，而 `inputs["states"]` 对应的是 `env.state()` 的返回值（在我们的环境中始终为 `None`）。

**后果**: 自项目开始以来的所有训练中，CNN 视觉分支的输入**始终为全零张量**（走 fallback 路径），导致：

- 深度图信息从未参与策略决策
- CNN 权重从未被有效梯度更新
- 策略实质上退化为一个输入全零的 MLP，无法利用视觉信息

**修复**: 将 `inputs["states"]` 改为 `inputs.get("observations", inputs.get("states", None))`，优先读取 `"observations"` 键（正常训练/推理路径），fallback 到 `"states"` 键（兼容 `init_state_dict` 阶段的探测调用）。

**影响**: 此修复前产生的所有 checkpoint 权重均无效，需从头重新训练。

#### 网络健康监控系统 (Network Diagnostics)

为防止多模态网络中某个子模块"静默失效"（如 CNN 输入全零、梯度为零等），新增可开关的诊断系统。

**开关位置**: `sh/config/auto_train.yaml` → `debug` 段

```yaml
debug:
    enabled: true             # 总开关
    log_interval: 50          # 每 50 次 PPO 更新记录一次
    log_images: true          # 将深度图写入 TensorBoard Images tab
    log_grad_norm: true       # 记录 CNN/MLP/Fusion 各模块梯度范数
    log_activation_stats: true # 记录激活值均值与标准差
    assert_nonzero_input: true # 断言输入非全零
```

**TensorBoard 中新增的指标** (Diagnostics/ 前缀):

| 指标名                         | 含义                          | 异常判断                      |
| ------------------------------ | ----------------------------- | ----------------------------- |
| `grad_norm_cnn`              | CNN+vision_proj 梯度 L2 范数  | 持续为 0 → 视觉分支未训练    |
| `grad_norm_prop_mlp`         | 本体 MLP 梯度范数             | 远大于 CNN → 视觉信号无效    |
| `grad_norm_fusion`           | 融合层+输出层梯度范数         | 为 0 → 整体反向传播中断      |
| `activation_mean/std_vision` | CNN 输出特征统计              | mean恒为0 → 输入或前向有问题 |
| `activation_mean/std_prop`   | 本体 MLP 输出特征统计         | 同上                          |
| `input_mean_depth`           | 深度图输入均值                | 恒为 0 → 相机数据未送入      |
| `input_nonzero_ratio`        | 深度图非零像素占比            | 恒为 0 → 同上                |
| `depth_input_ch0_front`      | 前置深度图可视化 (Images tab) | 全黑 → 相机失效              |
| `depth_input_ch1_back`       | 后置深度图可视化 (Images tab) | 全黑 → 同上                  |

**性能影响**: 诊断模块仅在 `debug.enabled=true` 时激活，每次前向传播额外开销为 4 次 `.detach()` 调用（约 0.01ms），TensorBoard 写入仅在 log_interval 间隔触发。关闭 debug 后零开销。

#### 相机 clipping_range 调优

将前后深度相机的 `clipping_range` 从 `(0.1, 10.0)` 改为 `(0.01, 3.0)`。原因：机器人在管道外表面夹持，相机到管壁的有效距离约 0.05~1.5m，10m 量程导致归一化后有效信息仅占值域 5%~15%，CNN 需要额外学会放大这个极小的差异。缩小量程后深度图信息密度提升约 7 倍。

#### 深度图 TensorBoard 可视化修复

诊断模块写入 Images tab 时显示全黑，原因是归一化后深度值集中在 0.05~0.15（对应灰度极暗）。修复为写入前对每帧做独立的 **min-max 拉伸到 [0,1]**，确保管道轮廓清晰可见。

#### PPO 超参数调优（正式训练前）

| 参数                     | 调整前 | 调整后 | 理由                                            |
| ------------------------ | ------ | ------ | ----------------------------------------------- |
| `learning_epochs`      | 5      | 8      | CNN 特征需要更多 epoch 从稀疏梯度中提取         |
| `mini_batches`         | 4      | 8      | 更小 batch size 让 CNN 获得更频繁的梯度更新     |
| `kl_threshold`         | 0.008  | 0.012  | CNN+MLP 的 KL 波动天然更大，过紧会频繁压低 lr   |
| `value_clip`           | 0.3    | 0.2    | 收紧 Critic 更新幅度，增强稳定性                |
| `time_limit_bootstrap` | False  | True   | 超时截断时 bootstrap 可减少对长生存策略的负偏估 |

这些改动零额外时间开销，但提升了样本利用效率。
