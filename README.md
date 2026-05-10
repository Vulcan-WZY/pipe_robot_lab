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

1

1

1

1

1

1
