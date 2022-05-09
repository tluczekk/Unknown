from util.handwriting_features import get_features
from util.handwriting_ground_truth import get_ground_truth
from util.handwriting_normalization import norm_features, norm_im
from util.handwriting_set_loader import get_training_set, get_test_set
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
import os

KEYWORD_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/keywords.txt'))

# Getting training set
train = get_training_set()
validate = get_test_set()
# Getting images and texts associated with them
word_images, word_texts = get_ground_truth(train)
word_images_valid, word_texts_valid = get_ground_truth(validate)
# Normalizing images
word_images_norm = [norm_im(img) for img in word_images]
word_images_norm_valid = [norm_im(img) for img in word_images_valid]
# Normalizing features
word_images_features = [get_features(img) for img in word_images_norm]
word_images_features_valid = [get_features(img) for img in word_images_norm_valid]

keywords = []
with open(KEYWORD_FILE) as keyword_file:
    for line in keyword_file:
        keywords.append(line)

for keyword in keywords:
    id = word_texts.index(keyword)
    keyword_distances = [fastdtw(word_images_features[id], img, dist=euclidean)[0] for img in word_images_features_valid]
    print(keyword_distances)
    print(f"For keyword {keyword} the shortest distance was to image with word {word_texts_valid[np.argmin(keyword_distances)]}")
