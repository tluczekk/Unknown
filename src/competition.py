from models.mlp.MLP import MLP
import numpy as np
from util.signatures_util import get_genuine_test, get_verification_test, get_max_for_users
import os
from fastdtw import fastdtw
from util.handwriting_set_loader import *
from util.handwriting_cleantext import *
from util.handwriting_normalization import *
from util.handwriting_features import *
from util.handwriting_ground_truth import *
from scipy.spatial.distance import euclidean


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

KEYWORD_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/TestKWS/task/keywords.txt'))
TRAIN_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/train.txt'))
TEST_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/TestKWS/task/test.txt'))
TEST_IMAGES_LOCATION = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/TestKWS/images/'))
SVG_LOCATION = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/TestKWS/ground-truth/locations/'))

train = get_training_set()
test = get_test_set(path=TEST_FILE, images_location=TEST_IMAGES_LOCATION)

word_images, word_texts = get_ground_truth(train)
test_images, test_ids = get_word_images(test, svg_location=SVG_LOCATION)

word_texts = [clean_label(w_text) for w_text in word_texts]

word_images_features = [norm_features(get_features(img)) for img in word_images]
test_images_features= [norm_features(get_features(img)) for img in test_images]

keywords = []
with open(KEYWORD_FILE) as keyword_file:
    for line in keyword_file:
        keywords.append(clean_label_test(line))

with open('results/kws.csv', 'w') as f:
    print("In a writing loop")
    # Main loop
    id_keyword = 0
    for keyword in keywords:
        id = word_texts.index(keyword)
        id_testword = 0
        line = keyword
        img_for_id = word_images_features[id]
        for img in test_images_features:
            distance = fastdtw.fastdtw(img_for_id, img, dist=euclidean)[0]
            line += ', ' + str(test_ids[id_testword]) + ', ' + str(distance)
            print(f"Done {id_keyword}-{id_testword}")
            id_testword += 1
        id_keyword += 1
        f.write(line+'\n')

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


