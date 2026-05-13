import os
import re
import sys
import yaml
import glob


def find_checkpoint(log_root_dir, checkpoint_name):
    """
    在 log_root_dir 中搜索 checkpoint 文件.
    如果 checkpoint_name 是绝对路径且存在, 直接返回.
    否则在 log_root_dir 下递归搜索匹配的文件名.
    对于 "best_agent.pt", 直接搜索该文件.
    对于其他名称, 尝试精确匹配或按编号找最大的.
    """
    if os.path.isabs(checkpoint_name) and os.path.exists(checkpoint_name):
        return checkpoint_name

    search_pattern = os.path.join(log_root_dir, "**", "checkpoints", checkpoint_name)
    files = glob.glob(search_pattern, recursive=True)
    if files:
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]

    if checkpoint_name == "best_agent.pt":
        search_pattern = os.path.join(log_root_dir, "**", "checkpoints", "best_agent.pt")
        files = glob.glob(search_pattern, recursive=True)
        if files:
            return files[0]

    all_pt = glob.glob(os.path.join(log_root_dir, "**", "checkpoints", "*.pt"), recursive=True)
    if not all_pt:
        return None

    numbered = []
    for f in all_pt:
        basename = os.path.basename(f)
        if "best_agent" in basename:
            continue
        match = re.search(r"(\d+)\.pt$", basename)
        if match:
            numbered.append((int(match.group(1)), f))

    if numbered:
        numbered.sort(key=lambda x: x[0], reverse=True)
        return numbered[0][1]

    return None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "play.yaml")
    workspace_root = os.path.join(script_dir, "..")
    os.chdir(workspace_root)

    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    play_cfg = cfg.get("play", {})
    render_cfg = cfg.get("rendering", {})
    video_cfg = cfg.get("video", {})

    task = play_cfg.get("task", "Template-Pipe-Robot-Lab-v1")
    num_envs = play_cfg.get("num_envs", 1)
    log_root_dir = play_cfg.get("log_root_dir", "logs/skrl/ppo_auto_04")
    checkpoint_name = play_cfg.get("checkpoint", "best_agent.pt")

    headless = render_cfg.get("headless", False)
    real_time = render_cfg.get("real_time", True)

    video_enabled = video_cfg.get("enabled", False)
    video_length = video_cfg.get("length", 500)
    video_save_dir = video_cfg.get("save_dir", "videos/play_results")

    cli_overrides = sys.argv[1:]
    if "--headless" in cli_overrides:
        headless = True
        cli_overrides.remove("--headless")
    if "--video" in cli_overrides:
        video_enabled = True
        cli_overrides.remove("--video")

    checkpoint_path = find_checkpoint(log_root_dir, checkpoint_name)
    if not checkpoint_path:
        print(f"[ERROR] Could not find checkpoint '{checkpoint_name}' in {log_root_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"[INFO] 🎮 Play Configuration")
    print(f"{'='*60}")
    print(f"  Task:        {task}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Num Envs:    {num_envs}")
    print(f"  Headless:    {headless}")
    print(f"  Real-time:   {real_time}")
    print(f"  Video:       {video_enabled}")
    if video_enabled:
        print(f"  Video Len:   {video_length} steps")
        print(f"  Video Dir:   {os.path.abspath(video_save_dir)}")
    print(f"{'='*60}\n")

    play_script = os.path.join("scripts", "skrl", "play.py")

    cmd = [
        sys.executable, play_script,
        "--task", task,
        "--num_envs", str(num_envs),
        "--checkpoint", checkpoint_path,
    ]

    if headless:
        cmd.append("--headless")

    if real_time:
        cmd.append("--real-time")

    if video_enabled:
        cmd.append("--video")
        cmd.extend(["--video_length", str(video_length)])

    cmd.extend(cli_overrides)

    print(f"[INFO] Execute Command:\n       > {' '.join(cmd)}\n")

    os.environ["PLAY_VIDEO_SAVE_DIR"] = os.path.abspath(video_save_dir)

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()