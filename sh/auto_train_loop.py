# ===========
# Date: 2026-05-05 14:23
# Author: Vulcan
# LastEditTime: 2026-05-13 23:53
# Description: 
# ==========
import os
import sys
import re
import yaml
import glob
import subprocess
import time


def detect_global_timestep(root_dir):
    """
    扫描 log_root_dir 下所有 checkpoints 中的 .pt 文件,
    从文件名中提取数字编号 (如 agent_12800.pt -> 12800),
    返回最大编号作为全局 timestep 起点. 如果没有任何 checkpoint 则返回 0.
    同时返回对应的文件路径用于续训加载.
    """
    checkpoints_path = os.path.join(root_dir, "**", "checkpoints", "*.pt")
    files = glob.glob(checkpoints_path, recursive=True)
    if not files:
        return 0, None

    best_step = -1
    best_file = None
    for f in files:
        basename = os.path.basename(f)
        if "best_agent" in basename:
            continue
        match = re.search(r"(\d+)\.pt$", basename)
        if match:
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_file = f

    if best_step < 0:
        return 0, None
    return best_step, best_file


def cleanup_checkpoints_by_name(root_dir, keep_qty):
    """
    按文件名中的数字编号降序排列, 仅保留编号最大的 keep_qty 个 .pt 文件.
    best_agent.pt 始终保留.
    """
    checkpoints_path = os.path.join(root_dir, "**", "checkpoints", "*.pt")
    files = glob.glob(checkpoints_path, recursive=True)
    if not files:
        return

    numbered_files = []
    for f in files:
        basename = os.path.basename(f)
        if "best_agent" in basename:
            continue
        match = re.search(r"(\d+)\.pt$", basename)
        if match:
            numbered_files.append((int(match.group(1)), f))

    numbered_files.sort(key=lambda x: x[0], reverse=True)

    if len(numbered_files) > keep_qty:
        to_delete = numbered_files[keep_qty:]
        for _, pt_path in to_delete:
            os.remove(pt_path)
        print(f"[INFO] 🧹 PT Cleanup: Deleted {len(to_delete)} old pt files. Kept top {keep_qty} by step number.")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "auto_train.yaml")

    workspace_root = os.path.join(script_dir, "..")
    os.chdir(workspace_root)

    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    isl_cfg = cfg.get("isaaclab", {})
    ckpt_cfg = cfg.get("checkpointing", {})
    debug_cfg = cfg.get("debug", {})

    framework = train_cfg.get("framework", "skrl")
    script_path = os.path.join("scripts", framework, "train.py")

    log_root_dir = ckpt_cfg.get("log_root_dir", "logs/skrl/pipe_robot_vision_ppo")
    max_iterations = train_cfg.get("max_iterations_per_round", 150)
    total_rounds = train_cfg.get("total_rounds", 500)
    saves_per_round = ckpt_cfg.get("saves_per_round", 2)
    keep_max_pt = ckpt_cfg.get("keep_max_pt_files", 5)
    rollouts = 32

    calc_save_interval = max(1, max_iterations // saves_per_round) if saves_per_round > 0 else max_iterations
    steps_per_round = max_iterations * rollouts

    if not os.path.exists(log_root_dir):
        os.makedirs(log_root_dir, exist_ok=True)
        print(f"[INFO] 📁 Created log directory: {log_root_dir}")

    global_timestep, resume_ckpt = detect_global_timestep(log_root_dir)
    if resume_ckpt:
        print(f"[INFO] 💎 Detected global timestep = {global_timestep} from checkpoint:\n       {resume_ckpt}")
    else:
        print(f"[INFO] 🌱 No checkpoint found. Starting fresh from timestep 0.")

    for round_idx in range(1, total_rounds + 1):
        print(f"\n{'='*60}")
        print(f"[INFO] 🚀 Round {round_idx} / {total_rounds} | Global offset: {global_timestep}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, script_path,
            "--task", train_cfg.get("task", "Template-Pipe-Robot-Lab-v1"),
            "--num_envs", str(train_cfg.get("num_envs", 32)),
            "--max_iterations", str(max_iterations),
            "--save_interval", str(calc_save_interval),
            "--experiment_dir", log_root_dir,
            "--run_name", "continuous_run",
            "--timestep_offset", str(global_timestep),
        ]

        if isl_cfg.get("headless", True):
            cmd.append("--headless")
        if isl_cfg.get("video", False):
            cmd.append("--video")
            if "video_interval" in isl_cfg:
                cmd.extend(["--video_interval", str(isl_cfg["video_interval"])])

        if debug_cfg.get("enabled", False):
            cmd.append("--debug")
            cmd.extend(["--debug_log_interval", str(debug_cfg.get("log_interval", 50))])

        if resume_ckpt and ckpt_cfg.get("resume_from_latest", True):
            cmd.extend(["--checkpoint", resume_ckpt])

        print(f"\n[INFO] Execute Command:\n       > {' '.join(cmd)}\n")

        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
        except KeyboardInterrupt:
            print("\n[INFO] Auto-Train Loop interrupted by user. Terminating process...")
            proc.terminate()
            sys.exit(0)

        if proc.returncode != 0:
            print(f"[ERROR] ❌ Training script crashed with code {proc.returncode}. Stopping auto-loop.")
            sys.exit(proc.returncode)

        print(f"[INFO] ✓ Round {round_idx} completed successfully.")

        global_timestep += steps_per_round

        expected_ckpt_step = global_timestep
        expected_ckpt_name = f"agent_{expected_ckpt_step}.pt"
        checkpoints_dir = os.path.join(log_root_dir, "continuous_run", "checkpoints")
        expected_ckpt_path = os.path.join(checkpoints_dir, expected_ckpt_name)

        if os.path.exists(expected_ckpt_path):
            resume_ckpt = expected_ckpt_path
        else:
            _, resume_ckpt = detect_global_timestep(log_root_dir)
            if resume_ckpt:
                actual_step = int(re.search(r"(\d+)\.pt$", resume_ckpt).group(1))
                if actual_step != expected_ckpt_step:
                    print(f"[WARNING] Expected checkpoint step {expected_ckpt_step} but found {actual_step}. Using found checkpoint.")
                    global_timestep = actual_step

        cleanup_checkpoints_by_name(log_root_dir, keep_max_pt)

        print("[INFO] Waiting 5 seconds for IsaacLab and GPU VRAM resources to be fully released...")
        time.sleep(5)


if __name__ == "__main__":
    main()