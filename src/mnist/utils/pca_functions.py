from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import numpy


def my_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X, X_preimage)

def anomaly_scorer(estimator, X, y=None):
    X_reduced = estimator.transform(X)
    X_preimage = estimator.inverse_transform(X_reduced)

    # Score the set
    dist_train = numpy.linalg.norm(X - X_preimage, axis=1)
    train_probs = numpy.array(
        [numpy.sum(xi >= dist_train) / len(dist_train) for xi in dist_train],
        dtype=float)
    auc = roc_auc_score(y, train_probs)

    return auc