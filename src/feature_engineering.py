from utils import Feature, generate_features, create_memo
from src.preprocess import base_data

import pandas as pd
import hydra
from sklearn.decomposition import PCA

# 生成された特徴量を保存するパス
Feature.dir = "features"
# trainとtestを結合して基本的な前処理を行ったデータを呼ぶ
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
        # カラム名
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
    # overwriteがfalseなら上書きはされない
    # globals()からこのファイルの中にある特徴量クラスが選別されてそれぞれ実行される
    generate_features(globals(), cfg.base.overwrite)


# デバッグ用
if __name__ == "__main__":
    run()
