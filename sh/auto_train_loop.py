# ===========
# Date: 2026-05-05 14:23
# Author: Vulcan
# LastEditTime: 2026-05-05 16:17
# Description: 
# ==========
import os
import sys
import yaml
import glob
import subprocess
import time

def get_latest_checkpoint(root_dir):
    """
    递归寻找 logging 目录下所有 checkpoints 文件夹中，修改时间最新的 `.pt` 模型权重文件
    """
    checkpoints_path = os.path.join(root_dir, "**", "checkpoints", "*.pt")
    # 设置 recursive=True 支持子目录递归检索
    files = glob.glob(checkpoints_path, recursive=True)
    if not files:
        return None
    # 按照系统记录的最近修改时间降序排列
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]

def cleanup_old_checkpoints(root_dir, keep_qty):
    """
    现在训练被强制合并到了单个连续的文件夹中，此函数全局扫描 checkpoins 里的 `.pt` 文件，
    按时间降序排列，仅保留最新的 keep_qty 个文件（best_agent.pt 会被始终保护避免误删）。
    """
    checkpoints_path = os.path.join(root_dir, "**", "checkpoints", "*.pt")
    files = glob.glob(checkpoints_path, recursive=True)
    if not files:
        return
        
    # 过滤掉名为 best_agent 的权重, 以免辛苦训练突破的最佳模型被丢掉
    files = [f for f in files if "best_agent" not in os.path.basename(f)]
    # 按修改时间从新到旧排
    files.sort(key=os.path.getmtime, reverse=True)
    
    if len(files) > keep_qty:
        to_delete = files[keep_qty:]
        for pt in to_delete:
            os.remove(pt)
        print(f"[INFO] 🧹 PT Cleanup: Deleted {len(to_delete)} old pt files. Kept newest {keep_qty}.")

def main():
    # 获取 yaml 和当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "auto_train.yaml")
    
    # 统一将由于脚本引发的终端工作目录切换回工程根目录(即 scripts、sh 等文件夹的同级目录)
    workspace_root = os.path.join(script_dir, "..")
    os.chdir(workspace_root)
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    train_cfg = cfg.get("training", {})     # 训练相关配置
    isl_cfg = cfg.get("isaaclab", {})       # Isaac Lab 相关配置
    ckpt_cfg = cfg.get("checkpointing", {}) # Checkpoint 相关配置
    
    framework = train_cfg.get("framework", "skrl")
    script_path = os.path.join("scripts", framework, "train.py") # 指向根目录下的真实训练脚本
    
    total_rounds = train_cfg.get("total_rounds", 500)
    for round_idx in range(1, total_rounds + 1):
        print(f"\n{'='*60}")
        print(f"[INFO] 🚀 Curriculum Training Loop - Round {round_idx} / {total_rounds}")
        print(f"{'='*60}")
        
        log_root_dir = ckpt_cfg.get("log_root_dir", "logs/skrl/pipe_robot_vision_ppo")
        
        # 动态计算每次保存的间隔步数
        max_iterations = train_cfg.get("max_iterations_per_round", 150)
        saves_per_round = ckpt_cfg.get("saves_per_round", 2)
        calc_save_interval = max(1, max_iterations // saves_per_round) if saves_per_round > 0 else max_iterations
        
        # 构建启动命令
        cmd = [
            sys.executable, script_path,
            "--task", train_cfg.get("task", "Template-Pipe-Robot-Lab-v1"),
            "--num_envs", str(train_cfg.get("num_envs", 32)),
            "--max_iterations", str(max_iterations),
            "--save_interval", str(calc_save_interval),
            "--experiment_dir", log_root_dir,  # 传递日志路径
            "--run_name", "continuous_run"     # 强行锁死子文件夹名称为连续运行模式，不再按时间戳割裂
        ]
        
        # 判断并添加 IsaacLab 以及模拟器的解析附加参数
        if isl_cfg.get("headless", True):
            cmd.append("--headless")
        if isl_cfg.get("video", False):
            cmd.append("--video")
            if "video_interval" in isl_cfg:
                cmd.extend(["--video_interval", str(isl_cfg["video_interval"])])
                
        # 核心：寻找上一次退出时保存的断点续训权重
        
        # 检验并自动创建设定的 Tensorboard / Checkpoint 保存路径
        if not os.path.exists(log_root_dir):
            os.makedirs(log_root_dir, exist_ok=True)
            print(f"[INFO] 📁 Created newly specified log directory: {log_root_dir}")

        if ckpt_cfg.get("resume_from_latest", True):
            latest_ckpt = get_latest_checkpoint(log_root_dir)
            if latest_ckpt:
                print(f"[INFO] 💎 Discovered latest checkpoint, resuming from:\n       {latest_ckpt}")
                cmd.extend(["--checkpoint", latest_ckpt])
            else:
                if round_idx > 1:
                    print("[WARNING] Could not find checkpoint for continuation! Starting from scratch!")
                else:
                    print("[INFO] 🌱 First round: No checkpoint found, starting fresh.")
                    
        print(f"\n[INFO] Execute Command:\n       > {' '.join(cmd)}\n")
        
        # 开启子进程
        proc = subprocess.Popen(cmd)
        try:
            proc.wait()
        except KeyboardInterrupt:
            # 捕获用户的 Ctrl+C 打断并友好终止整个外层进程
            print("\n[INFO] Auto-Train Loop interrupted by user. Terminating process...")
            proc.terminate()
            sys.exit(0)
        
        # 分析子进程 (train.py) 的退出状态码
        if proc.returncode != 0:
            print(f"[ERROR] ❌ Training script crashed with code {proc.returncode}. Stopping auto-loop.")
            sys.exit(proc.returncode)
            
        print(f"[INFO] ✓ Round {round_idx} completed successfully.")
        
        # 触发本地硬盘保护：全局清理超过数量上限的历史PT权重
        keep_max_pt = ckpt_cfg.get("keep_max_pt_files", 5)
        cleanup_old_checkpoints(log_root_dir, keep_max_pt)
        
        print("[INFO] Waiting 5 seconds for IsaacLab and GPU VRAM resources to be fully released...")
        time.sleep(5) 
        
if __name__ == "__main__":
    main()