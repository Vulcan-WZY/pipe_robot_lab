# ===========
# Date: 2026-05-08 09:19
# Author: Vulcan
# LastEditTime: 2026-05-12 21:57
# Description: 
# ==========
import os
import yaml
import subprocess

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "auto_train.yaml")
    
    workspace_root = os.path.join(script_dir, "..")
    os.chdir(workspace_root)
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found at {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    ckpt_cfg = cfg.get("checkpointing", {})
    log_root_dir = ckpt_cfg.get("log_root_dir", "logs/skrl/pipe_robot_vision_ppo")
    
    if not os.path.exists(log_root_dir):
        print(f"[WARNING] Log directory {log_root_dir} does not exist yet. Tensorboard might be empty.")
        
    print(f"\n[INFO] 📊 Launching TensorBoard pointing to: {log_root_dir}")
    print("[INFO] Any separated running logs within this folder will be plotted simultaneously!")
    print("[INFO] Press Ctrl+C to stop TensorBoard.\n")
    
    tailscale_ip = subprocess.run(
        ["tailscale", "ip", "-4"], capture_output=True, text=True
    ).stdout.strip()
    if tailscale_ip:
        print(f"\033[1;31m[INFO] 🌐 Use This !!!!! Access from remote: http://{tailscale_ip}:6006\033[0m")
    
    cmd = ["tensorboard", "--logdir", log_root_dir, "--bind_all"]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[INFO] TensorBoard stopped.")

if __name__ == "__main__":
    main()