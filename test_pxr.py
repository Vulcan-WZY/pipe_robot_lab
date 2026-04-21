import argparse
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app
from pxr import Usd
print("Success!")
