import numpy
import torch

from scipy.stats import chi2


def hoeffing_test(mu, log_var):

    KLD = -0.5 * numpy.sum(1 + log_var - numpy.power(mu, 2) - numpy.exp(log_var), axis=1)
    stat = 2 * KLD
    pval = 1 - chi2.cdf(stat, df=mu.shape[1]-1)  # we want 1 - Pr (X < x)
    return(pval)