"""Find the follower USB serial port and save it in config/robot.yaml.

Plug the adapter in, press Enter, unplug it, press Enter. Motors are not powered
and the arm does not move.
"""

from so101_utils.config import CONFIG
from so101_utils.ports import find_port


if __name__ == "__main__":
    find_port(CONFIG / "robot.yaml")
