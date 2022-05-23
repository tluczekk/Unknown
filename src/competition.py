from models.mlp.MLP import MLP
import numpy as np

with open('../data/mnist-test/mnist_test.csv') as ts:
    mnist_csv = np.loadtxt(ts, delimiter=',', dtype=int)

with open('../data/mnist-csv-format/mnist_train.csv') as tr:
    training = np.loadtxt(tr, delimiter=',', dtype=int)
    training_samples = training[:, 1:]
    training_labels = training[:, 0]

"""
SVM model
"""

"""
MLP model
"""
mlp = MLP()
estimator = mlp.get_estimator(training_samples, training_labels)
with open('results/mlp.txt', 'w') as f:
    i = 0
    for digit in mnist_csv:
        f.write(str(estimator.predict([digit])[0]) + '\n')
        print(f"Predicted {i}")
        i += 1

