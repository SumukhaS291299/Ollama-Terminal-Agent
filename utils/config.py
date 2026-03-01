import configparser
from pathlib import Path

from .agentLogger import setup_logger


class Readconfig:
    def __init__(self, path=Path.cwd().parent / "config.ini"):
        self.logger = setup_logger("config")
        self.path = path
        self.config = configparser.ConfigParser()
        self.logger.info(f"Loaded logs from config file: {self.path}")

    def read(self):
        readPath = self.config.read(self.path)
        self.logger.info(f"Successfully read config file from path: {readPath}")
        return self.config


if __name__ == "__main__":
    Readconfig().read()
