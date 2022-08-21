# coding: utf-8

import os

from svm import SVM
from lr import LR
from dt import DecisionTree
from data_loader import DataLoader
from data_clean import DataCleaner
from preprocessing import FeatureExtractor


label_csv = "./data/data_label.csv"
train_csv = "./data/data_100k.log_structured.csv"

input_dir = "./data"
data_clean_output_dir = "./data/clean_result"


def data_clean():
    data_clean_settings = {
        'HDFS': {
            'log_file': "data_100k.log",
            'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
            'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
            'group_num': 15
        }
    }

    for dataset, setting in data_clean_settings.items():
        print("\n=== Evaluation on %s ===" % dataset)
        in_dir = os.path.join(input_dir, os.path.dirname(setting["log_file"]))
        log_file = os.path.basename(setting["log_file"])

        parser = DataCleaner(
            log_format=setting["log_format"],
            input_path=input_dir,
            output_path=data_clean_output_dir,
            rex=setting["regex"],
            group_num=setting["group_num"]
        )
        parser.analyze(log_file)


def main():
    (x_train, y_train), (x_test, y_test) = DataLoader(
        train_csv=train_csv, label_csv=label_csv, split_ratio=0.7
    ).run()

    feature_extractor = FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    model = SVM()
    # model = LR()
    # model = DecisionTree()
    model.train(x_train, y_train)

    print("Train Validation: ")
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print("Test validation: ")
    precision, recall, f1 = model.evaluate(x_test, y_test)


if __name__ == "__main__":
    # data_clean()
    main()
