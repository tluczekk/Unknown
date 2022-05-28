# Project Recognition group Report, Group "Unknown"

Following is the group report for the group "Unknown", performed as a part of "Patterns Recognition" course at University of Fribourg in spring semester of 2022.

## Task descriptions

### MNIST models

First tasks were done on an MNIST dataset, which contained a set of handwritten digits with corresponding labels. Our task was to perform classification and tune the hyperparameters of 3 type of models: support vector machine, multi-layer perceptron and convolutional neural network. An additional task also consisted of running MLP and CNN on a permutated set.

MNIST tasks have been done relatively smoothly. We have divided the work to be done by one person at a time, with greatest difficulty being getting together in the beginning, as we had only our email addresses in the beginning, not knowing each other beforehand. With one week deadline extension we managed to build and tune our models, only having to cover for one member to do permutated part with a little delay.

### Keyword spotting task

The task was to extract words from letters of George Washington, having provided SVG masks. Next goal would be to extract some features from resized images to compute distances between keywords and words in test images to find those keywords there. One another important aspect was to clean labels from additional characters.

There were several challenges while doing this task. First, task description seemed a little bit unclear at times, nevertheless we set to create issues as separate tasks, and take one by one. This way all utility functions were created, without the final evaluation, which regrettably, had only been finished upon completing next task, having exceeded the deadline by quite some margin. Communication was again an Achilles heel of our group, with some members stopped being involved in project altogether after utility functions phase, due to other duties.

### Signature Verification Task

Last task concerned classifying signatures as either genuine or false ones, based on features computed on top of the data provided. Calculating distances wasn't really that challenging, selecting appropriate threshold although was very much a challenge. First, a variety of threshold was tested to produce Precision-Recall curve, and based on F1 score, a best performing threshold was selected, only to get 74% accuracy, a very long way from a satisfying result. Later however, an another approach was taken - namely, to consider one user at a time, and classify the signature on the basis of whether or not an average distance to each of genuine signatures is lower than the biggest distance between two genuine signatures. This way accuracy was taken to 91%.

This task was done by one member in two days, with the rest of group left with task of either doing the graph task or finishing KWS task, with neither happening. Again, communication was the culprit.

### Competition

In order to produce results for the competition, all previous models were scored with the test data, sometimes with need of unifying or correcting existing models to conform to the format of the data delivered, and the format of the output expected. Unfortunately, only one member took part in that.

## Lessons learned 

There were quite a few of lessons learned during these projects. While having been exposed to machine learning algorithms before, in either academic or professional environment, a look "under the hood" was a very interesting experience to say the least. To see how are they working step by step really enhanced an understanding of the methods in general. Having learned the power of list comprehension in Python has been an enlightening experience, as well as computing all steps of CNN by hand - a thing that every data scientist indeed should do at least once in a lifetime. An emphasis on using version control was also a highlight of these group projects.

On other note, one can't overlook an importance of knowing each member of a team for it to work properly. Being from separate cities, seeing each other just once and then having to work together was an extremely challenging task, that ultimately failed. Whatsapp is also not a sufficient means of communication, with notifications often marked as "to read later" end up being never read again.