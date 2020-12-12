import pandas as pd


def preprocess(train, test):

    dict_a = {"Class_1": 0,
              "Class_2": 1,
              "Class_3": 2,
              "Class_4": 3,
              "Class_5": 4,
              "Class_6": 5,
              "Class_7": 6,
              "Class_8": 7,
              "Class_9": 8,
              }

    train["target"] = train["target"].map(dict_a).astype(int)

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
