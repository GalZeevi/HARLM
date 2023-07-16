import json

DEFAULT_PATH = 'config.json'


class ConfigManager:
    config = None

    @staticmethod
    def _init_config(path=DEFAULT_PATH):
        with open(path) as configJson:
            ConfigManager.config = json.load(configJson)

    @staticmethod
    def get_config(path):
        if ConfigManager.config is None:
            ConfigManager._init_config()

        steps = path.split('.')
        res = ConfigManager.config
        for step in steps:
            res = res[step]
        return res

    @staticmethod
    def reload_config(path=DEFAULT_PATH):
        with open(path) as configJson:
            ConfigManager.config = json.load(configJson)
