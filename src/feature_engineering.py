from utils import Feature, generate_features, create_memo
from src.preprocess import base_data

import pandas as pd
import hydra
import numpy as np
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

Feature.dir = "features"
data = base_data()


class Base_data(Feature):
    def create_features(self):
        self.data = data.drop(columns=["id"])
        create_memo("base_data", "初期")


class Pca(Feature):
    def create_features(self):
        n = 20
        pca = PCA(n_components=n)
        pca.fit(
            data.drop(
                columns=["train", "target", "id"]
            )
        )
        n_name = [f"pca_{i}" for i in range(n)]
        df_pca = pd.DataFrame(
            pca.transform(data.drop(
                columns=["train", "target", "id"]
            )),
            columns=n_name
        )
        self.data = df_pca.copy()
        create_memo("pca", "pcaかけただけ")


@hydra.main(config_name="../config/config.yaml")
def run(cfg):
    generate_features(globals(), cfg.base.overwrite)


# デバッグ用
if __name__ == "__main__":
    run()
