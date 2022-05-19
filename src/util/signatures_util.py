import os
import numpy as np
from sklearn import metrics
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Getting features for genuine signatures
def get_genuine(path):
    all_users_genuine = []
    with open(path) as users:
        for line in users.read().split('\n'):
            if line != '':
                user_features = [get_features(os.path.realpath(os.path.dirname(__file__) + '../../../data/SignatureVerification/enrollment/' +
                                    str(line) + '-g-' + str(i).zfill(2) + '.txt')) for i in range(1,6)]
                all_users_genuine.append(user_features)

    all_users_genuine = np.array(all_users_genuine, dtype=object)

    return all_users_genuine

# Getting features for verification signatures
def get_verification(path):
    all_users_verification = []
    with open(path) as users:
        for line in users.read().split('\n'):
            if line != '':
                user_features = [get_features(os.path.realpath(os.path.dirname(__file__) + '../../../data/SignatureVerification/verification/' + 
                                    str(line) + '-' + str(i).zfill(2) + '.txt')) for i in range(1,46)]
                all_users_verification.append(user_features)
    
    all_users_verification = np.array(all_users_verification, dtype=object)

    return all_users_verification

def get_features(file):
    # input:
    # t x y pressure penup azimuth inclination
    # features
    # x y vx vy pressure

    data = []
    features = []
    with open(file) as f:
        for line in f:
            line_data = list(map(float, line.split()))
            if len(data) == 0:
                features.append([line_data[1], line_data[2], 0.0, 0.0, line_data[3]])
            else:
                dt = line_data[0] - data[-1][0]
                vx = (line_data[1] - data[-1][1]) / dt
                vy = (line_data[2] - data[-1][2]) / dt
                features.append([line_data[1], line_data[2], vx, vy, line_data[3]])
            data.append(line_data)
    
    # Feature normalization
    # source: https://stackoverflow.com/a/29661707
    features = np.array(features)
    features_norm = (features - features.min(0)) / features.ptp(0)

    return features_norm

# Getting ground truth values
def get_ground_truth(path):
    gts = []
    with open(path) as gt:
        for line in gt.read().split('\n'):
            if line != '':
                gts.append(line.split(' ')[-1])

    return gts

# Getting precision-recall curve
# source: https://blog.paperspace.com/mean-average-precision/
def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["f" if score >= threshold else "g" for score in pred_scores]

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="g")
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="g")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def get_max_for_users(user_features):
    maxs = []
    for user in user_features:
        user_intra_distances = [fastdtw(user[i], user[j], dist=euclidean)[0] for i in range(len(user)) for j in range(i, len(user))]
        user_max = np.max(user_intra_distances)
        maxs.append(user_max)

    return maxs
