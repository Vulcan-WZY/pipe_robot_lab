import argparse
import sys
# Make sure Isaac Sim paths are loaded or run with isaac python
try:
    from pxr import Usd, UsdPhysics
except ImportError:
    print("Run with Isaac Sim python")

stage = Usd.Stage.Open("source/pipe_robot_lab/pipe_robot_lab/assets/pipe_robot/usd/pipe_robot_rename.usd")
if not stage:
    print("Failed")
else:
    for link in ["BM_08_link", "FM_31_link", "BL_22_link", "BR_23_link", "FL_44_link", "FL_45_link"]:
        prim = stage.GetPrimAtPath(f"/pipe_robot/{link}")
        if prim:
            for child in prim.GetChildren():
                print(f"Under {link}: {child.GetPath()} ({child.GetTypeName()})")
