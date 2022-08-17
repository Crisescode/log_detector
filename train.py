# coding: utf-8


from .svm import SVM
from .data_loader import DataLoader
from .preprocessing import FeatureExtractor


train_csv = "./data/"
label_csv = "./data/"


def main():
    (x_train, y_train), (x_test, y_test) = DataLoader(
        train_csv=train_csv, label_csv=label_csv, split_ratio=0.7
    ).run()

    feature_extractor = FeatureExtractor()

    model = SVM()
    model.train(x_train, y_train)

    print("Train Validation: ")
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print("Test validation: ")
    precision, recall, f1 = model.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
