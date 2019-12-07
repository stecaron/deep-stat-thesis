import numpy
from scipy.stats import chi2
from scipy.spatial import distance


def compute_pvalues(dt, mean, sigma):
    """Function that compute the p-values for a normal distribution
        (chi-square test)

        Args:
            - dt: data on which to test each rows (rows: obs and cols: variabls)
            - mean: a vector of mean of the dist to be tested from
            - sigma: cov matrix of the normal dist to be tested from
    """

    loss = distance.cdist(dt,
                          mean,
                          'mahalanobis',
                          VI=numpy.linalg.inv(sigma))
    loss = loss**2
    pval = 1 - chi2.cdf(loss, df=mean.shape[1])  # we want 1 - Pr (X < x)

    return pval