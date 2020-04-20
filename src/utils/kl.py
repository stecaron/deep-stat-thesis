import numpy

def compute_kl_divergence(mu, log_var):

    KLD = -0.5 * numpy.sum(1 + log_var - numpy.power(mu, 2) - numpy.exp(log_var), axis=1)

    return(KLD)


def compute_kl_divergence_2_dist(mu, mu_ref, sigma, sigma_ref):

    # We assume sigma1 and sigma2 are diagonal matrix
    # Here are some operations on diagonal matrix D build from sigma vector
    # and D2 matrix built from sigma2 vector
    #   - det(D) = numpy.product(sigma)
    #   - D^-1 = 1/sigma
    #   - D * D2 = numpy.multiply(sigma, sigma2)

    # Reference: https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians

    a = numpy.log(numpy.product(sigma_ref)/numpy.product(sigma, axis=1))
    b = numpy.sum(numpy.multiply((1/sigma_ref), sigma), axis=1)
    c = numpy.sum(numpy.multiply(numpy.multiply((mu_ref - mu), (1/sigma_ref)), (mu_ref - mu)), axis=1)

    KLD = numpy.abs(0.5 * (a - mu.shape[1] + b + c))

    return(KLD)


