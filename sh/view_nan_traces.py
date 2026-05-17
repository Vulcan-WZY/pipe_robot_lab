#!/usr/bin/env python3
# ===========
# Date: 2026-05-18
# Author: Codex
# Description: 交互式查看 nan_traces 快照文件。自动读取 auto_train.yaml 中的 log_root_dir，
#              若存在 nan_traces 则支持通过上下方向键选择文件，回车查看，q 退出。
# ==========

from __future__ import annotations

import os
import sys
import textwrap
from typing import Any

import torch
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "auto_train.yaml")


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def read_config() -> dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        print(f"[ERROR] Config file not found: {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_trace_dir(cfg: dict[str, Any]) -> str:
    log_root_dir = cfg.get("checkpointing", {}).get("log_root_dir", "logs/skrl/pipe_robot_vision_ppo")
    abs_log_root_dir = os.path.join(WORKSPACE_ROOT, log_root_dir)
    return os.path.join(abs_log_root_dir, "continuous_run", "nan_traces")


def list_trace_files(trace_dir: str) -> list[str]:
    if not os.path.isdir(trace_dir):
        return []
    files = []
    for name in os.listdir(trace_dir):
        path = os.path.join(trace_dir, name)
        if os.path.isfile(path) and name.endswith(".pt"):
            files.append(path)
    files.sort()
    return files


def load_trace_file(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return {"raw_object": obj}
    return obj


def summarize_tensor(tensor: torch.Tensor) -> list[str]:
    lines: list[str] = []
    lines.append(f"shape: {tuple(tensor.shape)}")
    lines.append(f"dtype: {tensor.dtype}")

    finite_mask = torch.isfinite(tensor) if tensor.is_floating_point() or tensor.is_complex() else None
    if finite_mask is not None:
        finite_count = int(finite_mask.sum().item())
        total_count = tensor.numel()
        lines.append(f"finite_count: {finite_count}/{total_count}")
        if finite_count > 0:
            finite_vals = tensor[finite_mask]
            lines.append(f"finite_min: {finite_vals.min().item():.6g}")
            lines.append(f"finite_max: {finite_vals.max().item():.6g}")
            lines.append(f"finite_mean: {finite_vals.float().mean().item():.6g}")
        else:
            lines.append("finite_min: N/A")
            lines.append("finite_max: N/A")
            lines.append("finite_mean: N/A")

        if tensor.dim() == 2:
            nan_by_col = torch.isnan(tensor).sum(dim=0) if tensor.is_floating_point() else None
            if nan_by_col is not None:
                bad_cols = (nan_by_col > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
                lines.append(f"nan_columns: {bad_cols[:80]}")
    else:
        lines.append(f"min: {tensor.min().item():.6g}")
        lines.append(f"max: {tensor.max().item():.6g}")

    return lines


def format_value(key: str, value: Any) -> list[str]:
    if torch.is_tensor(value):
        lines = [f"{key}:"]
        lines.extend([f"  {line}" for line in summarize_tensor(value)])
        return lines

    if isinstance(value, (list, tuple)):
        text = repr(value)
    elif isinstance(value, dict):
        text = repr(value)
    else:
        text = str(value)

    wrapped = textwrap.wrap(text, width=110) or [""]
    return [f"{key}: {wrapped[0]}"] + [f"  {line}" for line in wrapped[1:]]


def print_trace_details(path: str) -> None:
    clear_screen()
    print(f"[TRACE] {os.path.basename(path)}")
    print(f"[PATH]  {path}")
    print("-" * 120)

    try:
        obj = load_trace_file(path)
    except Exception as exc:
        print(f"[ERROR] Failed to load file: {exc}")
        print("\nPress Enter to return...")
        input()
        return

    preferred_order = [
        "key",
        "shape",
        "action_shape",
        "has_nan",
        "has_inf",
        "nan_count",
        "inf_count",
        "nan_dims",
        "log_std",
        "tensor",
        "actions",
    ]

    printed = set()
    for key in preferred_order:
        if key in obj:
            for line in format_value(key, obj[key]):
                print(line)
            print()
            printed.add(key)

    for key in sorted(obj.keys()):
        if key in printed:
            continue
        for line in format_value(key, obj[key]):
            print(line)
        print()

    print("Press Enter to return to file list...")
    input()


def get_key() -> str:
    if not sys.stdin.isatty():
        raw = input("Select index / q: ").strip().lower()
        if raw == "q":
            return "quit"
        if raw == "":
            return "enter"
        if raw.isdigit():
            return f"index:{raw}"
        return ""

    if os.name == "nt":
        import msvcrt

        first = msvcrt.getch()
        if first in (b"\x00", b"\xe0"):
            second = msvcrt.getch()
            mapping = {b"H": "up", b"P": "down"}
            return mapping.get(second, "")
        if first in (b"\r", b"\n"):
            return "enter"
        if first in (b"q", b"Q"):
            return "quit"
        return ""

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch1 = sys.stdin.read(1)
        if ch1 == "\x1b":
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == "[":
                if ch3 == "A":
                    return "up"
                if ch3 == "B":
                    return "down"
            return ""
        if ch1 in ("\r", "\n"):
            return "enter"
        if ch1.lower() == "q":
            return "quit"
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def draw_menu(files: list[str], selected_index: int, trace_dir: str) -> None:
    clear_screen()
    print("[INFO] NaN Trace Viewer")
    print(f"[INFO] Trace directory: {trace_dir}")
    print("[INFO] Controls: Up/Down move, Enter open, q quit")
    print("-" * 120)

    for idx, path in enumerate(files):
        marker = ">" if idx == selected_index else " "
        print(f"{marker} [{idx + 1:02d}/{len(files):02d}] {os.path.basename(path)}")


def main() -> None:
    cfg = read_config()
    trace_dir = resolve_trace_dir(cfg)
    files = list_trace_files(trace_dir)

    if not files:
        print(f"[INFO] No nan_traces found under: {trace_dir}")
        sys.exit(0)

    selected_index = 0

    while True:
        draw_menu(files, selected_index, trace_dir)
        key = get_key()

        if key == "up":
            selected_index = (selected_index - 1) % len(files)
        elif key == "down":
            selected_index = (selected_index + 1) % len(files)
        elif key == "enter":
            print_trace_details(files[selected_index])
        elif key.startswith("index:"):
            try:
                selected = int(key.split(":", 1)[1]) - 1
            except ValueError:
                selected = -1
            if 0 <= selected < len(files):
                selected_index = selected
                print_trace_details(files[selected_index])
        elif key == "quit":
            clear_screen()
            print("[INFO] Exit NaN Trace Viewer.")
            break


if __name__ == "__main__":
    main()
