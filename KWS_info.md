# Keyword spotting

This task's goal was to implement keyword spotting on letters of George Washington, first president and one of the Founding Fathers of the United States. Data was divided into training and validation set, with each containing an occurence of certain keyword, stored in *keyword.txt*. Spotting keywords should be done using Dynamic Time Warping - we used *fastdtw* implementation.

## Extracting 

First task was to actually get the keywords out of images. This has been done in **handwriting_set_loader** and **handwriting_ground_truth** scripts. Words are extracted by masks provided in SVG, and then rescaled to 100x100 images and binarized.

## Features

Features for sliding window through image are implemented in **handwriting_features** script. We decided to use following features:

- upper contour, that is first black pixel
- lower contour, that is last black pixel
- number of black-white transitions
- percent of black pixels in the window
- percent of black pixels inbetween the contours

Features are then normalized in a way coded in **handwriting_normalization** script, that is min-max normalization.

## Cleaning labels

Labels are cleaned in a way provided in **handwriting_cleantext** script. First, any trailing tabs, whitespaces and newlines are stripped, then the resulting string is split on dashes, creating an array of strings - only standalone characters are considered, and then are once again joined by dashes.

## Calculating distances

For each keyword distances are calculated to each word in validation set (with distances and true labels being saved for the sake of PR curve). Then the label and corresponding distance to said keyword are saved in **ksw.csv** file. 

## Precision recall curve

It is produced by saving true labels and distances, and computing resulting labels by applying several thresholds.