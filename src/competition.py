from models.mlp.MLP import MLP
import numpy as np
from util.signatures_util import *
import os
from fastdtw import fastdtw

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
    for digit in mnist_csv:
        f.write(str(estimator.predict([digit])[0]) + '\n')

"""
CNN model
"""

"""
KWS
"""

"""
Signatures
"""
USERS = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/TestSignatures/users.txt'))
genuine_signatures = get_genuine_test(USERS)
verification_signatures = get_verification_test(USERS)
verification_scores = []
for i in range(70):
    for verification_signature in verification_signatures[i]:
        mean_distance = np.mean([fastdtw(verification_signature, genuine_signature, dist=euclidean)[0] for genuine_signature in genuine_signatures[i]])
        print(mean_distance)
        verification_scores.append(mean_distance)

with open('results/sign-ver.csv', 'w') as f:
    i = 0
    user_maxs = get_max_for_users(genuine_signatures)
    for j in range(70):
        line = str(j+31).zfill(3)
        for k in range(45):
            line += ', ' + str(k + 1).zfill(2) + ', ' + str(verification_scores[i])
            i += 1
        f.write(line + '\n')


