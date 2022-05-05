import os
import cv2
import numpy as np

TRAIN_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/train.txt'))
TEST_FILE = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/test.txt'))
IMAGES_LOCATION = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/images/'))

def get_training_set():
    data = []
    with open(TRAIN_FILE) as train_file:
        for line in train_file:
            data.append(IMAGES_LOCATION + "\\" + line.rstrip() + ".jpg")
    return data


def get_test_set():
    data = []
    with open(TEST_FILE) as test_file:
        for line in test_file:
            data.append(IMAGES_LOCATION + "\\" + line.rstrip() + ".jpg")
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
                padded = cv2.copyMakeBorder(thresh, top=tb, bottom=tb, left=rl, right=rl, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
                padded = padded.astype("float32") / 255.0 #convert to pixel values (1 or 0)
                words.append(padded)

                #uncomment to paint the rectangle around selection for debuging!
                #cv2.rectangle(inputImageCopy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #cv2.imshow("Image", inputImageCopy)
        #cv2.waitKey(0)

        return words
