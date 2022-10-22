import json


class ConfigManager:
    with open('config.json') as configJson:
        config = json.load(configJson)

    @staticmethod
    def get_config(path):
        steps = path.split('.')
        res = ConfigManager.config
        for step in steps:
            res = res[step]
        return res
