# coding: utf-8


from sklearn import svm
from .utils import metrics


class SVM(object):
    def __init__(self, penalty="l1", tol=0.1, C=1, dual=False, class_weight=None, max_iter=100):
        self.classifier = svm.LinearSVC(
            penalty=penalty,
            tol=tol,
            C=C,
            dual=dual,
            class_weight=class_weight,
            max_iter=max_iter
        )

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_true, y_pred)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
