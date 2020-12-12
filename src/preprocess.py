import pandas as pd


def preprocess(train, test):
    train["train"] = True
    test["train"] = False
    test["target"] = None

    data = pd.concat([
        train, test
    ]).reset_index(drop=True)
    return data


def base_data():
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    data = preprocess(train, test)
    return data
