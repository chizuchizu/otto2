from utils import Feature, generate_features, create_memo
from src.preprocess import base_data

import pandas as pd
import hydra
import numpy as np
from sklearn.model_selection import KFold

Feature.dir = "../features_data"
data = base_data()

groupby_cols = ["shipment_mode", "weekday", "drop_off_point", "shipping_company", "hour", "destination_country",
                "shipment_id"]
calc_cols = ["freight_cost", "gross_weight", "shipment_charges", "cost"]


class Base_data(Feature):
    def create_features(self):
        self.data = data.drop(columns=["processing_days", "cut_off_time"])
        create_memo("base_data", "初期")




@hydra.main(config_name="../config/features.yaml")
def run(cfg):
    generate_features(globals(), cfg.overwrite)


# デバッグ用
if __name__ == "__main__":
    run()
