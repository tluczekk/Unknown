# Signature Verification

This task concerned classifying 45 signatures for each of 30 users, having an access to 5 signatures for each, that are for sure true.

## Features

Having an access to neatly packed data which consisted of:

- time
- x coordinate
- y coordinate
- pressure applied
- penup 
- azimuth
- inclination

I've calculated features for each signature as follows:

- x coordinate
- y coordinate
- x coordinate speed
- y coordinate speed
- pressure applied

Speeds are calculated based on the distance travelled in time calculated as delta between t and t-1 (first measure is set to 0). Features are then normalized for each signature. 

## Distances

Distances are calculated using **fastdtw** implementation of Dynamic Time Warping. For each signature in validation set, distance to each genuine signature is calculated, and then the mean of them is taken into account. As soon as all distances are calculated, precision-recall curve is drawn. Based on F1 metric, general threshold of 120 produced the best results, giving the accuracy of ca. 74%. 

However, in order to improve the results I tried to look at this problem as "pseudo-clustering" sense. In essence, "dynamic thresholds", as I called them, are the maximum distance between signatures in "cluster" of genuine ones. That is, if mean of distances to genuine signatures is smaller than maximum distance between genuine signatures themselves, then the said signature might be labelled as genuine as well. This way accuracy has been improved to 91%. Looking at diff of *signatures_results* and *gt* I noticed, that in majority of cases where this model classifies signature wrongly, it is a genuine signature, so different from provided ones, that it is deemed false by our model.