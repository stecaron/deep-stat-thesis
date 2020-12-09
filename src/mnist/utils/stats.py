import numpy
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


def test_performances(p_values, index, alpha):

    y_adj = (numpy.arange(0, len(p_values), step=1) / len(p_values)) * alpha
    p_values_ord = numpy.argsort(p_values)

    # Find the first accepted observations
    # k = 0
    # while True:
    #     if p_values[p_values_ord][k] > alpha:
    #         cutoff = k
    #         break
    #     k += 1

    predictions = numpy.zeros(len(p_values))
    predictions[numpy.argwhere(p_values <= alpha)] = 1

    # total_outliers = numpy.sum(index[p_values_ord])
    # detected_outliers = numpy.sum(index[p_values_ord][0:cutoff])

    #precision = detected_outliers / (cutoff + 1)
    #recall = detected_outliers / total_outliers
    #f1_score = 2 * (precision * recall) / (precision + recall)
    precision = precision_score(index, predictions)
    recall = recall_score(index, predictions)
    f1_score_stats = f1_score(index, predictions)
    roc_auc = roc_auc_score(index, 1 - p_values)
    average_precision = average_precision_score(
        index, 1 - p_values)

    return precision, recall, f1_score_stats, average_precision, roc_auc
