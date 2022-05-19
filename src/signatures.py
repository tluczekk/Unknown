import os
from cv2 import threshold
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from util.signatures_util import *

USERS = os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/SignatureVerification/users.txt'))
GROUND_TRUTH =os.path.realpath(os.path.join(os.path.dirname(__file__), '../data/SignatureVerification/gt.txt'))

# Getting normalized features for genuine and verification signatures
genuine_signatures = get_genuine(USERS)
verification_signatures = get_verification(USERS)

# Computing mean distances of every verification signature to corresponding 5 genuine signatures
verification_scores = []
for i in range(30):
    for verification_signature in verification_signatures[i]:
        mean_distance = np.mean([fastdtw(verification_signature, genuine_signature, dist=euclidean)[0] for genuine_signature in genuine_signatures[i]])
        print(mean_distance)
        verification_scores.append(mean_distance)

# Evaluation
# source: https://blog.paperspace.com/mean-average-precision/
# Building precision recall curve and choosing threshold based on f1 score
ground_truth = get_ground_truth(GROUND_TRUTH)
thresholds = np.arange(start=60, stop=300, step=10)

precisions, recalls = precision_recall_curve(y_true=ground_truth, 
                                             pred_scores=verification_scores,
                                             thresholds=thresholds)

plt.plot(recalls, precisions, linewidth=4, color="red")
plt.xlabel("Recall", fontsize=12, fontweight='bold')
plt.ylabel("Precision", fontsize=12, fontweight='bold')
plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
plt.savefig("Signatures_PRC.png")

f1 = 2 * ((np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls)))

print(thresholds)
print(f1)
print(f"The best threshold is {thresholds[np.argmax(f1)]}")

## Results have show that fixed threshold of 120 produces the best results
threshold = 120

# Writing results to the file and simultaneously calculating "dynamic" thresholds, that is different thresholds 
# for each signature. This way accuracy improved from 74% to 91%. Dynamic threshold is essentialy the maximum distance
# between pairs of genuine signatures of a given user. These are the classes that are then saved in file
with open('signatures_results.txt', 'w') as f:
    i = 0
    pred = ["f" if score >= threshold else "g" for score in verification_scores]
    user_maxs = get_max_for_users(genuine_signatures)
    pred_var = []
    print(f"Accuracy on fixed threshold: {metrics.accuracy_score(ground_truth, pred)}")
    for j in range(30):
        for k in range(45):
            # classification based on dynamic threshold
            if verification_scores[i] >= user_maxs[j]:
                pred_var.append("f")
            else:
                pred_var.append("g")
            f.write(str(j+1).zfill(3) + '-' + str(k+1).zfill(2) + ' ' + pred_var[i] + '\n')
            i += 1
    print(f"Accuracy on dynamic threshold: {metrics.accuracy_score(ground_truth, pred_var)}")