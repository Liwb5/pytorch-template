# coding:utf-8
import os

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
