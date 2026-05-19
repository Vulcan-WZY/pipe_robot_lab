# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--save_interval", type=int, default=None, help="Steps interval to save model checkpoints.")
parser.add_argument("--experiment_dir", type=str, default=None, help="Override the experiment directory for logging.")
parser.add_argument("--run_name", type=str, default=None, help="Override the run name (subfolder) to prevent timestamp folder creation.")
parser.add_argument("--timestep_offset", type=int, default=None, help="Global timestep offset for TensorBoard and checkpoint naming (set by auto_train_loop).")
parser.add_argument("--debug", action="store_true", default=False, help="Enable network diagnostics (grad norm, activation stats, input images to TensorBoard).")
parser.add_argument("--debug_log_interval", type=int, default=50, help="How often (in PPO updates) to log diagnostic data.")
parser.add_argument("--nan_trace", action="store_true", default=False, help="On first NaN detection, save full tensor snapshot to .pt file and print NaN dimension indices.")
parser.add_argument("--anomaly_detect", action="store_true", default=False, help="Enable torch.autograd anomaly detection to pinpoint NaN-producing ops (slower).")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# enable camera rendering for vision-based tasks (required by tiled cameras)
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import random
import time
import torch
from datetime import datetime
from packaging import version
from collections import OrderedDict

import skrl

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
    from skrl.envs.wrappers.torch.base import Wrapper
    from skrl.utils.spaces.torch import flatten_tensorized_space, tensorize_space, unflatten_tensorized_space
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import pipe_robot_lab.tasks  # noqa: F401


class VisionIsaacLabWrapper(Wrapper):
    """SKRL wrapper that keeps multimodal observations (camera/policy/critic).

    Includes NaN/Inf detection on observations to prevent simulation instability
    from corrupting the training pipeline.
    """

    def __init__(self, env):
        super().__init__(env)
        self._reset_once = True
        self._observations = None
        self._info = {}
        self._nan_warn_count = 0
        self._inf_warn_count = 0
        self._max_warnings = 20  # 避免日志刷屏

        try:
            base_space = self._unwrapped.single_observation_space
        except Exception:
            base_space = self._unwrapped.observation_space

        if hasattr(base_space, "spaces"):
            base_spaces = base_space.spaces
        elif isinstance(base_space, dict):
            base_spaces = base_space
        else:
            base_spaces = {"policy": base_space}

        self._obs_keys = [k for k in ["camera", "policy", "critic"] if k in base_spaces]
        if not self._obs_keys and "policy" in base_spaces:
            self._obs_keys = ["policy"]

        self._observation_space = gym.spaces.Dict(OrderedDict((k, base_spaces[k]) for k in self._obs_keys))

    @property
    def state_space(self):
        return None

    def state(self):
        return None

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        try:
            return self._unwrapped.single_action_space
        except Exception:
            return self._unwrapped.action_space

    def _extract_observations(self, observations):
        if isinstance(observations, dict):
            return {k: observations[k] for k in self._obs_keys if k in observations}
        return observations

    @staticmethod
    def _sanitize_tensor(t, label=""):
        """将张量中的 NaN/Inf 替换为 0，返回是否进行了替换以及安全后的张量。"""
        has_nan = torch.isnan(t).any()
        has_inf = torch.isinf(t).any()
        if has_nan or has_inf:
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t, has_nan.item() if isinstance(has_nan, torch.Tensor) else has_nan, has_inf.item() if isinstance(has_inf, torch.Tensor) else has_inf

    def _sanitize_obs_dict(self, obs_dict):
        """遍历观测字典，检测并清理所有子张量中的 NaN/Inf。
        当 nan_trace 启用时，首次检测到 NaN 会保存完整快照到磁盘。
        """
        if not isinstance(obs_dict, dict):
            return obs_dict

        for key, val in obs_dict.items():
            if torch.is_tensor(val):
                has_nan = torch.isnan(val).any()
                has_inf = torch.isinf(val).any()
                if has_nan or has_inf:
                    # 保存快照 (在清理前)
                    self._maybe_save_nan_trace(key, val, has_nan, has_inf)
                    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                    obs_dict[key] = val
                if has_nan:
                    if self._nan_warn_count < self._max_warnings:
                        nan_idxs = torch.isnan(val).nonzero(as_tuple=False)
                        # 打印前 20 个 NaN 维度索引，便于定位具体传感器/关节
                        idx_str = ", ".join(str(idx.tolist()) for idx in nan_idxs[:20])
                        if nan_idxs.shape[0] > 20:
                            idx_str += f", ... ({nan_idxs.shape[0]} total)"
                        logger.warning(f"[OBS-NaN] NaN detected in observation key '{key}', "
                                       f"NaN dims: [{idx_str}], replaced with 0. "
                                       f"(warning {self._nan_warn_count + 1}/{self._max_warnings})")
                        self._nan_warn_count += 1
                if has_inf:
                    if self._inf_warn_count < self._max_warnings:
                        logger.warning(f"[OBS-Inf] Inf detected in observation key '{key}', replaced with 0. "
                                       f"(warning {self._inf_warn_count + 1}/{self._max_warnings})")
                        self._inf_warn_count += 1
            elif isinstance(val, dict):
                self._sanitize_obs_dict(val)
        return obs_dict

    def _maybe_save_nan_trace(self, key, tensor, has_nan, has_inf):
        """首次检测到 NaN/Inf 时保存完整张量快照到磁盘。"""
        if not hasattr(self, "_nan_trace_saved"):
            self._nan_trace_saved = set()
        if key in self._nan_trace_saved:
            return
        self._nan_trace_saved.add(key)

        trace_dir = os.environ.get("PIPE_ROBOT_NAN_TRACE_DIR", "")
        if not trace_dir:
            return

        import time as _time
        ts = _time.strftime("%Y%m%d_%H%M%S")
        fname = f"obs_nan_{key}_{ts}.pt"
        fpath = os.path.join(trace_dir, fname)
        pipe_txt_path = os.path.join(trace_dir, f"{ts}.txt")
        try:
            torch.save({
                "key": key,
                "tensor": tensor.detach().cpu(),
                "shape": tuple(tensor.shape),
                "has_nan": bool(has_nan),
                "has_inf": bool(has_inf),
                "nan_count": int(torch.isnan(tensor).sum().item()),
                "inf_count": int(torch.isinf(tensor).sum().item()),
                "nan_dims": torch.isnan(tensor).nonzero(as_tuple=False)[:50].cpu().tolist(),
            }, fpath)
            selected_pipe = os.environ.get("PIPE_ROBOT_SELECTED_PIPE_USD", "")
            with open(pipe_txt_path, "w", encoding="utf-8") as f:
                f.write(selected_pipe)
            logger.info(f"[NAN-TRACE] Saved observation snapshot to {fpath} "
                        f"(NaN count: {torch.isnan(tensor).sum().item()})")
        except Exception:
            pass

    def step(self, actions):
        actions = unflatten_tensorized_space(self.action_space, actions)
        observations, reward, terminated, truncated, self._info = self._env.step(actions)
        obs_selected = self._extract_observations(observations)
        obs_selected = self._sanitize_obs_dict(obs_selected)
        self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, obs_selected))
        return self._observations, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), self._info

    def reset(self):
        if self._reset_once:
            observations, self._info = self._env.reset()
            obs_selected = self._extract_observations(observations)
            obs_selected = self._sanitize_obs_dict(obs_selected)
            self._observations = flatten_tensorized_space(tensorize_space(self.observation_space, obs_selected))
            self._reset_once = False
        return self._observations, self._info

    def render(self, *args, **kwargs):
        return None

    def close(self):
        self._env.close()

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # Set debug environment variables for model diagnostics
    if args_cli.debug:
        os.environ["PIPE_ROBOT_DEBUG"] = "1"
        os.environ["PIPE_ROBOT_DEBUG_INTERVAL"] = str(args_cli.debug_log_interval)
    if args_cli.nan_trace:
        os.environ["PIPE_ROBOT_NAN_TRACE"] = "1"
    if args_cli.anomaly_detect:
        torch.autograd.set_detect_anomaly(True)
        print("[INFO] 🔍 PyTorch autograd anomaly detection enabled (expect ~15% slowdown).")
    
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        
    # Get checkpoint path early to extract accumulated timestep
    resume_path_early = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    start_timestep = 0
    if args_cli.timestep_offset is not None:
        start_timestep = args_cli.timestep_offset
        print(f"[INFO] ⏱️ Using externally provided timestep offset: {start_timestep}")
    elif resume_path_early:
        import re
        match = re.search(r"(\d+)\.pt", resume_path_early)
        if match:
            start_timestep = int(match.group(1))
            print(f"[INFO] ⏱️ Discovered resume timestep from checkpoint: {start_timestep}")

    # max iterations for training
    if args_cli.max_iterations:
        additional_timesteps = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        agent_cfg["trainer"]["timesteps"] = additional_timesteps
        print(f"[INFO] 🎯 Training target: {additional_timesteps} new timesteps this round (global offset: {start_timestep}).")
        
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # modify the checkpointing interval if requested
    if args_cli.save_interval is not None:
        # SKRL 规定 checkpoint_interval 是以底层物理步数 (timesteps) 为单位，而不是 PPO iteration。
        # 这里自动用传进来的 save_interval 乘以 rollouts 步长来校正，使其行为表现和我们外层的直觉完全一致。
        rollouts = agent_cfg["agent"].get("rollouts", 1)
        real_save_interval = args_cli.save_interval * rollouts
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = real_save_interval
        print(f"[INFO] Override model save interval to {args_cli.save_interval} iterations (which is {real_save_interval} timesteps)")

    # specify directory for logging experiments
    if args_cli.experiment_dir:
        log_root_path = os.path.abspath(args_cli.experiment_dir)
    else:
        log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    if args_cli.run_name is not None:
        log_dir = args_cli.run_name
    else:
        # specify directory for logging runs using timestamp style: ppo-YY-MM-HH
        default_run_name = f"{algorithm}-{datetime.now().strftime('%y-%m-%H')}"
        custom_run_name = str(agent_cfg["agent"]["experiment"].get("experiment_name", "")).strip()
        log_dir = custom_run_name if custom_run_name else default_run_name
        
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # 为 NaN trace 快照设置保存目录
    if args_cli.nan_trace:
        trace_dir = os.path.join(log_dir, "nan_traces")
        os.makedirs(trace_dir, exist_ok=True)
        os.environ["PIPE_ROBOT_NAN_TRACE_DIR"] = trace_dir
        print(f"[INFO] 🔍 NaN trace snapshots will be saved to: {trace_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # 如果使用自定义多模态模型，需要保留 camera/policy/critic 三组观测
    using_custom_models = not agent_cfg.get("models", {})
    if using_custom_models and args_cli.ml_framework.startswith("torch"):
        env = VisionIsaacLabWrapper(env)
        print(f"[INFO] Environment wrapper: VisionIsaacLabWrapper (obs keys: {env._obs_keys})")
    else:
        # 默认包装器会将观测压缩到 policy 组
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    
    # === [Custom Model Injection] ===
    # 判断是否使用了非空 models 配置 (也就是原版的默认自动构建)
    # 如果为空 (models: {})，则我们手动实例化多模态神经网络并注入 agent_cfg 中
    if not agent_cfg.get("models", {}):
        print("[INFO] Injecting custom multimodal CNN+MLP model...")
        from pipe_robot_lab.tasks.manager_based.pipe_robot_lab.agents.models.custom_skrl_model import CustomActor, CustomCritic

        # Runner 需要 models 配置字典 (而不是模型实例)，因此为自定义类注册组件解析
        _original_component = Runner._component

        def _custom_component(self, name: str):
            lowered = name.lower()
            if lowered == "customactor":
                return CustomActor
            if lowered == "customcritic":
                return CustomCritic
            return _original_component(self, name)

        Runner._component = _custom_component

        # 使用 Runner 支持的配置形式，让其自动实例化 policy / value
        agent_cfg["models"] = {
            "separate": True,
            "policy": {"class": "CustomActor"},
            "value": {"class": "CustomCritic"},
        }
    # ==================================

    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    if start_timestep > 0 or args_cli.debug:
        agent = runner.agent
        _original_write_tracking = agent.write_tracking_data.__func__
        _original_write_checkpoint = agent.write_checkpoint.__func__
        _original_update = agent.update.__func__
        _offset = start_timestep

        import types
        
        if args_cli.debug:
            from pipe_robot_lab.tasks.manager_based.pipe_robot_lab.agents.models.custom_skrl_model import get_diagnostics
            _diag = get_diagnostics()
            diag_log_dir = os.path.join(log_dir, "continuous_run")
            _diag.set_log_dir(diag_log_dir)
            print(f"[INFO] 🔬 Diagnostics writer initialized at: {diag_log_dir}")

        def _patched_write_tracking(self, *, timestep, timesteps):
            global_ts = timestep + _offset
            _original_write_tracking(self, timestep=global_ts, timesteps=timesteps + _offset)
            if args_cli.debug and _diag.should_log():
                _diag.log_to_writer(runner.agent.policy, global_ts)
                try:
                    value_model = getattr(runner.agent, "value", None)
                    if value_model is not None:
                        _diag.log_to_writer(value_model, global_ts)
                except Exception:
                    pass
                try:
                    value_preprocessor = getattr(runner.agent, "_value_preprocessor", None)
                    writer = getattr(_diag, "_writer", None)
                    if writer is not None and value_preprocessor is not None:
                        running_mean = getattr(value_preprocessor, "running_mean", None)
                        running_variance = getattr(value_preprocessor, "running_variance", None)
                        if torch.is_tensor(running_mean):
                            writer.add_scalar("Diagnostics/value_preproc_mean", running_mean.mean().item(), global_ts)
                        if torch.is_tensor(running_variance):
                            writer.add_scalar("Diagnostics/value_preproc_var", running_variance.mean().item(), global_ts)
                        writer.flush()
                except Exception:
                    pass

        def _log_tensor_stats(writer, tag_prefix, tensor, global_ts):
            if writer is None or tensor is None or not torch.is_tensor(tensor):
                return
            finite = torch.isfinite(tensor)
            finite_vals = tensor[finite]
            if finite_vals.numel() > 0:
                writer.add_scalar(f"{tag_prefix}_min", finite_vals.min().item(), global_ts)
                writer.add_scalar(f"{tag_prefix}_max", finite_vals.max().item(), global_ts)
                writer.add_scalar(f"{tag_prefix}_mean", finite_vals.float().mean().item(), global_ts)
                writer.add_scalar(f"{tag_prefix}_std", finite_vals.float().std(unbiased=False).item(), global_ts)
            writer.add_scalar(f"{tag_prefix}_finite_ratio", finite.float().mean().item(), global_ts)

        def _patched_update(self, *, timestep, timesteps):
            global_ts = timestep + _offset
            writer = getattr(_diag, "_writer", None) if args_cli.debug else None
            try:
                if writer is not None:
                    rewards = self.memory.get_tensor_by_name("rewards")
                    values = self.memory.get_tensor_by_name("values")
                    _log_tensor_stats(writer, "Diagnostics/reward", rewards, global_ts)
                    _log_tensor_stats(writer, "Diagnostics/value_target_raw", values, global_ts)
            except Exception:
                pass

            result = _original_update(self, timestep=timestep, timesteps=timesteps)

            try:
                if writer is not None:
                    values = self.memory.get_tensor_by_name("values")
                    returns = self.memory.get_tensor_by_name("returns")
                    advantages = self.memory.get_tensor_by_name("advantages")
                    _log_tensor_stats(writer, "Diagnostics/value_target_proc", values, global_ts)
                    _log_tensor_stats(writer, "Diagnostics/returns", returns, global_ts)
                    _log_tensor_stats(writer, "Diagnostics/advantages", advantages, global_ts)
                    writer.flush()
            except Exception:
                pass

            return result

        def _patched_write_checkpoint(self, *, timestep, timesteps):
            _original_write_checkpoint(self, timestep=timestep + _offset, timesteps=timesteps + _offset)

        agent.update = types.MethodType(_patched_update, agent)
        agent.write_tracking_data = types.MethodType(_patched_write_tracking, agent)
        agent.write_checkpoint = types.MethodType(_patched_write_checkpoint, agent)
        print(f"[INFO] ⏱️ Monkey-patched Agent with timestep offset = {_offset} for TensorBoard & checkpoint naming.")
        if args_cli.debug:
            print(f"[INFO] 🔬 Network diagnostics enabled (interval={args_cli.debug_log_interval}).")

    # run training
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
