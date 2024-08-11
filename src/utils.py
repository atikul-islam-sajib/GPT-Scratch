import os
import sys
import yaml

sys.path.append("./src/")


def dump(value=None, filename=None):
    if (value is not None) and (filename is not None):
        yaml.safe_dump(value=value, filename=filename)

    else:
        raise ValueError("value and filename must be provided".capitalize())


def load(filename=None):
    if filename is not None:
        yaml.safe_load(filename=filename)

    else:
        raise ValueError("filename must be provided".capitalize())


def config():
    with open("./config.yml", "r") as file:
        return yaml.safe_load(file)
