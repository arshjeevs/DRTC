"""Replay robot: runs the whole robot-side control loop without hardware (--robot.type=replay)."""

from .config_replay import ReplayRobotConfig
from .replay import ReplayRobot

__all__ = ["ReplayRobot", "ReplayRobotConfig"]
