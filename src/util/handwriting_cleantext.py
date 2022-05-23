from sklearn import metrics
def clean_label(label):
    # stripping trailing tabs, newlines and whitespaces, then splitting it on dashes
    letters = label.rstrip().split('-')
    # filtering just standalone letters
    letters_res = [letter for letter in letters if len(letter) == 1]
    # Returning label with dashes
    return '-'.join(letters_res)

def clean_label_test(label):
    letters = label.split(',')[0].split('-')
    letters_res = [letter for letter in letters if len(letter) == 1]
    # Returning label with dashes
    return '-'.join(letters_res)

# PRC method adjusted to KWS task
def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = [0 if score >= threshold else 1 for score in pred_scores]

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label=1)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label=1)
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls