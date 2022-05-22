from util.handwriting_features import get_features
from util.handwriting_ground_truth import get_ground_truth
from util.handwriting_normalization import norm_features, norm_im
from util.handwriting_set_loader import get_training_set, get_test_set
from pyts.metrics import dtw
import numpy as np
import os

KEYWORD_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/keywords.txt'))
OUTPUT_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../kws.csv'))

# Getting training set
train = get_training_set()
validate = get_test_set()
# Getting images and texts associated with them
word_images, word_texts = get_ground_truth(train)
word_images_valid, word_texts_valid = get_ground_truth(validate)
# Normalizing images
# word_images_norm = [norm_im(img) for img in word_images]
# word_images_norm_valid = [norm_im(img) for img in word_images_valid]
# Normalizing features
word_images_features = [get_features(img) for img in word_images]
word_images_features_valid = [get_features(img) for img in word_images_valid]

keywords = []
with open(KEYWORD_FILE) as keyword_file:
    for line in keyword_file:
        keywords.append(line)

with open(OUTPUT_FILE, 'w') as output_file:
    for keyword in keywords:
        id = word_texts.index(keyword)
        id_testword = 0
        line = keyword
        img_for_id = np.array(word_images[id]).flatten()
        keyword_distances = []
        for img in word_images_valid:
            distance = dtw(img_for_id, np.array(img).flatten(), method='sakoechiba', dist='square')
            test_word = word_texts[id_testword]
            line += ',' + test_word + ',' + str(distance)
            id_testword += 1
        output_file.write(line+';\n')
        # print(keyword_distances)
        # print(f"For keyword {keyword} the shortest distance was to image with word {word_texts_valid[np.argmin(keyword_distances)]}")
    output_file.close()