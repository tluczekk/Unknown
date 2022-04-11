from sklearn import svm
from sklearn.model_selection import cross_validate
from models.model import ModelInterface


class SVMClassifier(ModelInterface):

    def __init__(self, kernel, C, gamma):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma)

    def train(self, training_set):
        print('started training SVM model...')
        cross_validate(self.model, training_set.X, training_set.y)
        print('done training SVM model!')

    def predict(self, test_set):
        return self.model.predict(test_set.X)

    def evaluate(self, test_set):
        predict = self.model.predict(test_set.X)
        acc, f1, recall = super().evaluate(test_set.y, predict)
        print("________SVM(C=" + str(self.C) + ", gamma=" + str(self.gamma) + ")_______")
        print(" acc=" + str(acc))
        print(" acc=" + str(acc))
        print(" acc=" + str(acc))
        print("____________________________")

