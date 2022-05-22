from sympy import true
from util.handwriting_features import get_features
from util.handwriting_ground_truth import get_ground_truth
from util.handwriting_normalization import norm_features, norm_im
from util.handwriting_set_loader import get_training_set, get_test_set
from util.handwriting_cleantext import clean_label, precision_recall_curve
from pyts.metrics import dtw
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt

KEYWORD_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/keywords.txt'))
OUTPUT_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../kws.csv'))

# Getting training set
train = get_training_set()
validate = get_test_set()
# Getting images and texts associated with them
word_images, word_texts = get_ground_truth(train)
word_images_valid, word_texts_valid = get_ground_truth(validate)
# Cleaning the labels
word_texts = [clean_label(w_text) for w_text in word_texts]
word_texts_valid = [clean_label(w_text_v) for w_text_v in word_texts_valid]
# Normalizing images
# word_images_norm = [norm_im(img) for img in word_images]
# word_images_norm_valid = [norm_im(img) for img in word_images_valid]
# Normalizing features
word_images_features = [norm_features(get_features(img)) for img in word_images]
word_images_features_valid = [norm_features(get_features(img)) for img in word_images_valid]

# Getting keywords to evaluate
keywords = []
with open(KEYWORD_FILE) as keyword_file:
    for line in keyword_file:
        keywords.append(clean_label(line))

# PRC arrays
word_true = []
pred_dist = []

with open(OUTPUT_FILE, 'w') as output_file:
    print("In a writing loop")
    # Main loop
    id_keyword = 0
    for keyword in keywords:
        id = word_texts.index(keyword)
        id_testword = 0
        line = keyword
        img_for_id = word_images_features[id]
        for img in word_images_features_valid:
            distance = fastdtw(img_for_id, img, dist=euclidean)[0]
            test_word = word_texts[id_testword]
            # PRC labels and distances
            if test_word == keyword:
                word_true.append(1)
            else:
                word_true.append(0)
            pred_dist.append(distance)
            # Adding word and its distance to the keyword to output
            line += ',' + test_word + ',' + str(distance)
            print(f"Done {id_keyword}-{id_testword}")
            id_testword += 1
        id_keyword += 1
        output_file.write(line+';\n')
        # print(keyword_distances)
        # print(f"For keyword {keyword} the shortest distance was to image with word {word_texts_valid[np.argmin(keyword_distances)]}")
    output_file.close()

# Computing Precision-Recall curve (as in Signatures task)

thresholds = np.arange(start=20, stop=100, step=5)

precisions, recalls = precision_recall_curve(y_true=word_true, 
                                             pred_scores=pred_dist,
                                             thresholds=thresholds)

plt.plot(recalls, precisions, linewidth=4, color="red")
plt.xlabel("Recall", fontsize=12, fontweight='bold')
plt.ylabel("Precision", fontsize=12, fontweight='bold')
plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
plt.savefig("Keywords_PRC.png")