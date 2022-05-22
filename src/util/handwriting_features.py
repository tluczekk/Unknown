import fastdtw
import numpy as np

def get_features(image):
    """
    Feature vector consists of:
        - upper contour, that is first black pixel
        - lower contour, that is last black pixel
        - number of black-white transitions
        - percent of black pixels in the window
        - percent of black pixels inbetween the contours
        ? gradient of upper contour from window to window
        ? gradient of lower contour from window to window
    """
    image_array = np.array(image)
    features = []

    for window in range(image_array.shape[1]):
        # extracting current window
        current_window = image_array[:, window]
        # getting indices of ones
        ones = [i for i, val in enumerate(current_window) if val]
        # Trivial cases of fully white window or only one black pixel
        # Putting placeholders for gradients
        if len(ones) == 0:
            features.append([0.0, 0.0, 0, 0.0, 0.0])
        elif len(ones) == 1:
            features.append([ones[0]/100, ones[0]/100, 2, 1 / len(current_window), 1.0])
        else:
            lc = ones[0] / 100
            uc = ones[-1] / 100
            # Calculating transitions with shifted slices
            # https://stackoverflow.com/a/47750613
            transitions = np.count_nonzero(current_window[:-1] < current_window[1:])
            percent_black = len(ones) / len(current_window)
            percent_black_cont = len(ones) / (uc - lc + 1)
            features.append([lc, uc, transitions, percent_black, percent_black_cont])
    
    features = np.array(features)

    #Gradient pass
    #for window in range(1, image_array.shape[1]):
    #    features[window, 5] = features[window, 0] - features[window - 1, 0]
    #    features[window, 6] = features[window, 1] - features[window - 1, 1]

    return features