import json


def read_config(conf_file):
    with open(conf_file) as config_f:
        config = json.load(config_f)
        if 'neo4j' in config:
            config['neo4j']['auth'] = tuple(config['neo4j']['auth'])
        return config
