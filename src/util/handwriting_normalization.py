import cv2
import numpy as np

def _min_max(vector):
    v_min, v_max = np.min(vector), np.max(vector)
    distance = v_max - v_min
    return (vector - v_min) / distance

def norm_im(image, w=100, h=100):
    return cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

def norm_features(features, method='minmax'):
    if method == 'minmax':
        return np.array(list(map(_min_max, features.T))).T