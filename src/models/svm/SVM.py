from sklearn import svm
from sklearn.model_selection import cross_validate
import numpy as np
from models.model import ModelInterface


class SVMClassifier(ModelInterface):

    def __init__(self, kernel, C, gamma):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self, training_set):
        print('started training SVM model...')
        cv_values = cross_validate(self.model, training_set.X, training_set.y, return_estimator=True)
        estimator_to_return = np.argmax(cv_values['test_score'])
        self.model = cv_values['estimator'][estimator_to_return]
        print('done training SVM model!')

    def predict(self, test_set):
        return self.model.predict(test_set)

    def evaluate(self, test_set):
        predict = self.model.predict(test_set.X)
        acc, f1, recall = super().evaluate(test_set.y, predict)
        print("________SVM(kernel=" + str(self.kernel)+ ", C=" + str(self.C) + ", gamma=" + str(self.gamma) + ")_______")
        print(" acc=" + str(acc))
        print(" f1=" + str(f1))
        print(" recall=" + str(recall))
        print("____________________________")

