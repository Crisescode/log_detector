# coding: utf-8


from sklearn import tree
from utils import metrics


class DecisionTree(object):

    def __init__(self, criterion='gini', max_depth=None, max_features=None, class_weight=None):
        self.classifier = tree.DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features,
            class_weight=class_weight
        )

    def train(self, X, y):
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
