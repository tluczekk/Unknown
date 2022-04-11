import abc
from sklearn.metrics import accuracy_score, recall_score, f1_score


class ModelInterface(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'predict') and
                callable(subclass.predict))

    @classmethod
    def evaluate(actual_y, predictions):
        acc = accuracy_score(actual_y, predictions)
        f1 = f1_score(actual_y, predictions)
        recall = recall_score(actual_y, predictions)
        return [acc, f1, recall]
