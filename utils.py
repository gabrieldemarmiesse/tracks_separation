import os


def mk(path_to_directory):
    if not os.path.isdir(path_to_directory):
        os.mkdir(path_to_directory)
