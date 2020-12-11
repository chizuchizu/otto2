from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import shap
import hydra
import gc

from pathlib import Path

import mlflow
import mlflow.lightgbm

from utils import git_commits

rand = np.random.randint(0, 1000000)


def save_log(score_dict):
    mlflow.log_metrics(score_dict)
    mlflow.log_artifact(".hydra/config.yaml")
    mlflow.log_artifact(".hydra/hydra.yaml")
    mlflow.log_artifact(".hydra/overrides.yaml")
    mlflow.log_artifact("train_hydra.log")
    mlflow.log_artifact("features.csv")


def run(cfg):
    cwd = Path(hydra.utils.get_original_cwd())

    if cfg.base.optuna:
        import optuna.integration.lightgbm as lgb
    else:
        import lightgbm as lgb

    data = [pd.read_pickle(cwd / f"../features/{f}.pkl") for f in cfg.features]
    data = pd.concat(data, axis=1)

    train = data[data["train"]].drop(columns="train")
    test = data[~data["train"]].drop(columns=["train", "target"])
    target = train["target"]
    train = train.drop(columns="target")

    del data
    gc.collect()
    kfold = KFold(n_splits=cfg.base.n_folds, shuffle=True, random_state=cfg.base.seed)

    pred = np.zeros(test.shape[0])
    score = 0

    experiment_name = f"{'optuna_' if cfg.base.optuna else ''}{rand}"
    print("file:///" + hydra.utils.get_original_cwd() + "mlruns")
    mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")

    use_cols = pd.Series(train.columns)
    use_cols.to_csv("features.csv", index=False, header=False)

    mlflow.lightgbm.autolog()
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train, target)):
        x_train, x_valid = train.loc[train_idx], train.loc[valid_idx]
        y_train, y_valid = target[train_idx], target[valid_idx]

        d_train = lgb.Dataset(x_train, label=y_train)
        d_valid = lgb.Dataset(x_valid, label=y_valid)
        del x_train
        del x_valid
        del y_train
        del y_valid
        gc.collect()
        mlflow.set_experiment(f"fold_{fold + 1}")

        with mlflow.start_run(run_name=f"{experiment_name}"):
            estimator = lgb.train(
                params=dict(cfg.parameters),
                train_set=d_train,
                num_boost_round=cfg.base.num_boost_round,
                valid_sets=[d_train, d_valid],
                verbose_eval=500,
                early_stopping_rounds=100
            )

            y_pred = estimator.predict(test)
            pred += y_pred / cfg.base.n_folds

            print(fold + 1, "done")

            score_ = estimator.best_score["valid_1"][cfg.base.metric]
            score += score_ / cfg.base.n_folds

            save_log(
                {
                    "score": score
                }
            )


@git_commits(rand)
@hydra.main(config_name="../config/training.yaml")
def main(cfg):
    run(cfg)
    return None
