import numpy


def test_performances(p_values, index, alpha):

    y_adj = (numpy.arange(0, len(p_values), step=1) / len(p_values)) * alpha
    p_values_ord = numpy.argsort(p_values)

    # Find the first accepted observations
    k = 0
    while True:
        if p_values[p_values_ord][k] > alpha:
            cutoff = k
            break
        k += 1

    total_outliers = numpy.sum(index[p_values_ord])
    detected_outliers = numpy.sum(index[p_values_ord][0:cutoff])

    precision = detected_outliers / (cutoff + 1)
    recall = detected_outliers / total_outliers

    return precision, recall
