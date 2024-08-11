import os
import sys
import yaml

sys.path.append("./src/")


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
