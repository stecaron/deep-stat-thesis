import numpy

def compute_kl_divergence(mu, log_var):

    KLD = -0.5 * numpy.sum(1 + log_var - numpy.power(mu, 2) - numpy.exp(log_var), axis=1)

    return(KLD)