import numpy as np
import os
import cv2
from scipy.stats import zscore

SVG_LOCATION = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/locations/'))
TRANS_LOCATION = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data/transcription.txt'))

def _get_bounds(image_id, svg_location=SVG_LOCATION):
    """
    Read rectangular bounds from SVG file. Probably some SVG library that would
        have handled this better.
    """
    im_path = svg_location + '/' + str(image_id) + ".svg"
    bounds = []
    with open(im_path) as f:
        lines = f.readlines()
        for l in lines:
            if 'path fill' in l:
                id = l.split('id="')[1].split('"')[0]
                f = l.split('"none" d="')[1].split(' Z" stroke-width')[0].split(' L ')
                f = [ s.replace('M ', '') for s in f]
                f = [ [float(d) for d in s.split(' ')] for s in f]
                f = np.array(f).T
                min_x, max_x = int(np.min(f[0])), int(np.max(f[0]))
                min_y, max_y = int(np.min(f[1])), int(np.max(f[1]))
                x, y, w, h = min_x, min_y, max_x-min_x, abs(min_y-max_y)
                bounds.append({'x': x, 'y': y, 'w': w, 'h': h, 'id': id})
    return bounds

def _get_boundary_word(transcription, id):
    for l in transcription:
        if l.split(' ')[0] == id:
            return l.split(' ')[1]
    return '' # Return empty string upon fail. Error handling could be better...

def get_ground_truth(data, trans_location=TRANS_LOCATION, svg_location=SVG_LOCATION):
    """
    Same method as implemented in handwriting_set_loader.
        ay consider to make this methods more general...
    """

    transcription = open(trans_location, 'r').readlines()

    word_images = []
    word_texts = []

    for line in data:
        print(line)
        im = cv2.imread(line)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        id = int(line.split('\\')[-1].split('.')[0])

        for _, c in enumerate(_get_bounds(id, svg_location=svg_location)):

            # Extract relevant word from transcript
            word = _get_boundary_word(transcription, c['id'])
            word_texts.append(word)

            # Get the bounding rectangle
            (x, y, w, h) = c['x'], c['y'], c['w'], c['h']

            # filter out bounding boxe
            if (w > 10 and w <= 700) and (h > 10 and h <= 700):
                tb = int(h/2)
                rl = int(w/2)
                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (tH, tW) = thresh.shape
                dX = int(max(0, 100 - tW) / 2.0)
                dY = int(max(0, 100 - tH) / 2.0)
                padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
                padded = cv2.resize(padded, (100, 100), interpolation=cv2.INTER_NEAREST).astype("float32")
                normalized = padded.astype("float32") / 255.0  # convert to pixel values (1 or 0)
                #normalized = zscore(np.array(padded), axis=None, ddof=1)
                word_images.append(normalized)

    return word_images, word_texts

def get_word_images(data, svg_location=SVG_LOCATION):

    word_images = []
    ids = []

    for line in data:
        print(line)
        im = cv2.imread(line)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        id = int(line.split('\\')[-1].split('.')[0])

        for _, c in enumerate(_get_bounds(id, svg_location=svg_location)):

            # Get the bounding rectangle
            (x, y, w, h) = c['x'], c['y'], c['w'], c['h']
            ids.append(c['id'])

            # filter out bounding boxe
            if (w > 10 and w <= 700) and (h > 10 and h <= 700):
                tb = int(h/2)
                rl = int(w/2)
                roi = gray[y:y + h, x:x + w]
                thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                (tH, tW) = thresh.shape
                dX = int(max(0, 100 - tW) / 2.0)
                dY = int(max(0, 100 - tH) / 2.0)
                padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
                                            left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
                                            value=(0, 0, 0))
                padded = cv2.resize(padded, (100, 100), interpolation=cv2.INTER_NEAREST).astype("float32")
                normalized = padded.astype("float32") / 255.0  # convert to pixel values (1 or 0)
                #normalized = zscore(np.array(padded), axis=None, ddof=1)
                word_images.append(normalized)

    return word_images, ids