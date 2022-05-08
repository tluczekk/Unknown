import fastdtw
from handwriting_features import get_features

def get_distance(im1, im2):
    distance, path = fastdtw(get_features(im1), get_features(im2))
    return distance