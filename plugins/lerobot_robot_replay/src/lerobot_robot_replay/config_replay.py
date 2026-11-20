from dataclasses import dataclass, field

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("replay")
@dataclass
class ReplayRobotConfig(RobotConfig):
    """Replays one episode of a LeRobot dataset as if it were the robot.

    Observations follow the wall clock through the episode (the dataset's fps): the recorded camera frames, and a
    joint state that either follows the commanded actions with a first-order lag ("follow") or is the recorded state
    ("dataset"). Commands are accepted and logged. Cameras keep their dataset names (observation.images.<key>).
    Images do not react to commands (open loop), so this tests timing, scheduling, faults and smoothness, not task
    success.
    """

    repo_id: str = "orangemoon599/so101_cubebox_v1"
    root: str | None = None
    revision: str | None = None
    episode: int = 0
    state: str = "follow"  # follow | dataset
    follow_alpha: float = 0.35  # fraction of the way to the commanded target per observation (~30 Hz)
    loop: bool = False  # at the end: loop the episode, or hold its last frame
    preload: bool = True  # decode the episode once before the run (30 Hz on a slow or shared CPU), not every tick
    log_path: str | None = None  # where to write the received commands (JSON lines) on disconnect
    cameras: dict = field(default_factory=dict)  # unused: cameras come from the dataset
