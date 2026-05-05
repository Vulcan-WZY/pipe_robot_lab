<!--
 * @Date: 2026-05-03 19:31
 * @Author: Vulcan
 * @LastEditTime: 2026-05-03 20:11
 * @Description: 
-->

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

1

1

1

1

1

1

1

1

111
