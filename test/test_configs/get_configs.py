import os


def get_config(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)