import os

import yaml


def load_config():
    """
    Loads configuration from YAML file.
    :return: Configuration
    """
    cfg_path = os.path.abspath(os.path.join(__file__, "../../conf/config.yaml"))

    try:
        with open(cfg_path, "r") as stream:
            config = yaml.load(stream, Loader=yaml.SafeLoader)
            return config

    except Exception as ex:
        # Many possibilities to raise exceptions
        msg = str(ex)
        print(msg)


"""Configuration settings"""
cfg = load_config()

env = cfg["env"]
postgres = cfg["postgres"]

# Import environment setting
cfg = cfg[env]
cfg["env"] = env
cfg["postgres"] = postgres
