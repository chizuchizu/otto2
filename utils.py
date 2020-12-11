import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from time import time
from datetime import datetime
import gc
import argparse
import inspect
import csv
import re
import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
import git
import yaml
import shutil
import json

# from itertools import chain
warnings.filterwarnings("ignore")


@contextmanager
def timer(name):
    t0 = time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time() - t0:.0f} s')


def get_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--force', '-f', action='store_true', help='Overwrite existing files')
    return parser.parse_args()


def get_features(namespace):
    for k, v in ({k: v for k, v in namespace.items()}).items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


# @hydra.main(config_path="config.yaml")
def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.data_path.exists() and not overwrite:
            print(f.name, 'was skipped')
        else:
            f.run().save()


class Feature(metaclass=ABCMeta):
    prefix = ''
    suffix = ''
    dir = 'kaggle_env'

    def __init__(self):
        if self.__class__.__name__.isupper():
            self.name = self.__class__.__name__.lower()
        else:
            self.name = re.sub("([A-Z])", lambda x: "_" + x.group(1).lower(), self.__class__.__name__).lstrip('_')

        # ユーザーに登録してもらう
        # self.data = pd.read_pickle("../features_data/data.pkl")
        self.data = pd.DataFrame()
        self.data_path = Path(self.dir) / f"{self.name}.pkl"

    def run(self):
        with timer(self.name):
            self.create_features()
            prefix = self.prefix + '_' if self.prefix else ''
            suffix = '_' + self.suffix if self.suffix else ''

            self.data.columns = prefix + self.data.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        self.data.to_pickle(str(self.data_path))

    def load(self):
        self.data = pd.read_pickle(str(self.data_path))


def create_memo(col_name, desc):
    file_path = Feature.dir + "/_features_memo.csv"

    # hydraのログパスにカレントディレクトリが移動してしまうので初期化
    # 影響がないことは確認済み
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isfile(file_path):
        with open(file_path, "w"): pass

    with open(file_path, "r+") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        col = [line for line in lines if line.split(",")[0] == col_name]
        if len(col) != 0: return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])


"""
training用
"""


def add_experiment_name(rand):
    with open(".hydra/config.yaml", "r+") as f:
        data = yaml.load(f)

        data["experiment_name"] = str(rand)

        # f.write(yaml.dump(data))
    with open(".hydra/config.yaml", "w") as f:
        yaml.dump(data, f)


def git_commits(func):
    def wrapper(*args, **kwargs):
        rand = args[0]
        repo = git.Repo("/home/yuma/PycharmProjects/cassava")
        repo.git.diff("HEAD")
        repo.git.add(".")
        repo.index.commit(f"{rand}(before running)")
        func(*args, **kwargs)

        repo.index.commit(f"{rand}(after running)")
        repo.git.push('origin', 'master')

    return wrapper


def kaggle_wrapper(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        rand = args[0]
        cwd = args[1]
        cfg = args[2]
        add_experiment_name(rand=rand)
        add_datasets(rand)
        add_notebooks(rand, cwd, cfg)

    return wrapper


def add_datasets(rand):
    """upload to kaggle datasets
    hydraパス内で実行して
    notebooksの前に実行して
    """
    metadata = {
        "title": f"{rand}",
        "id": f"chizuchizu/{rand}",
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }

    data_json = eval(json.dumps(metadata))
    with open("dataset-metadata.json", "w") as f:
        json.dump(data_json, f)

    shutil.copy(".hydra/config.yaml", "config.yaml")
    os.system("kaggle datasets create -p .")


def add_notebooks(rand, cwd, cfg):
    """
    hydraパス内で実行して
    :return: None
    """
    meta = {
        "id": f"chizuchizu/{rand} inference",
        "title": f"{rand} inference",
        "language": "python",
        "kernel_type": "script",
        "code_file": str(cwd / "inference.py"),
        "is_private": "true",
        "enable_gpu": cfg.kaggle.enable_gpu,
        "dataset_sources": [
                               f"chizuchizu/{rand}",
                           ] + cfg.kaggle.data_sources,
        "competition_sources": cfg.kaggle.competitions,
    }
    data_json = eval(json.dumps(meta))
    with open("kernel-metadata.json", "w") as f:
        json.dump(data_json, f)
    os.system("kaggle kernels push -p .")
