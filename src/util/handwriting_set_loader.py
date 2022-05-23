import os
import cv2
from scipy.stats import zscore
import numpy as np

TRAIN_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/train.txt'))
TEST_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/valid.txt'))
IMAGES_LOCATION = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/images/'))

def get_training_set(path=TRAIN_FILE, images_location=IMAGES_LOCATION):
    data = []
    with open(path) as train_file:
        for line in train_file:
            data.append(images_location + "\\" + line.rstrip() + ".jpg")
    return data


def get_test_set(path=TEST_FILE, images_location=IMAGES_LOCATION):
    data = []
    with open(path) as test_file:
        for line in test_file:
            data.append(images_location + "\\" + line.rstrip() + ".jpg")
    return data


def preprocess(data):
    words = []

    for line in data:
        im = cv2.imread(line)
        inputImageCopy = im.copy()

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

        # Applying dilation on the threshold image
        dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

        # Finding contours
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)

        for _, c in enumerate(contours):

            # Get the bounding rectangle of the current contour:
            (x, y, w, h) = cv2.boundingRect(c)

            # filter out bounding boxes, ensuring they are neither too small
            # nor too large
            if (w > 10 and w <= 700) and (h > 10 and h <= 700):
                tb = int(h/2)
                rl = int(w/2)

                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

                (tH, tW) = thresh.shape
                dX = int(max(0, 100 - tW) / 2.0)
                dY = int(max(0, 100 - tH) / 2.0)
                # pad the image and force 32x32 dimensions
                padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
                padded = cv2.resize(padded, (100, 100)).astype("float32")
                #padded.astype("float32") / 255.0  # convert to pixel values (1 or 0)
                normalized = zscore(np.array(padded), axis=None, ddof=1)
                words.append(normalized)

                #uncomment to paint the rectangle around selection for debuging!
                #cv2.rectangle(inputImageCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #cv2.imshow("Image", inputImageCopy)
        #cv2.waitKey(0)

        return words
