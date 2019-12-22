import numpy
import torch

from src.utils.kl import compute_kl_divergence


def compute_empirical_pval(dt_train, model, dt_test):

    # Encode train data
    dt_train_torch = torch.from_numpy(dt_train).float()
    dt_train_torch = torch.unsqueeze(dt_train_torch, 1)
    dt_train_torch = dt_train_torch.view(-1, 28 * 28)
    generated_train, mu_train, logvar_train, _ = model(dt_train_torch)
    mu_train = mu_train.detach().numpy()
    logvar_train = logvar_train.detach().numpy()

    # Encode test data
    dt_test_torch = torch.from_numpy(dt_test).float()
    dt_test_torch = torch.unsqueeze(dt_test_torch, 1)
    dt_test_torch = dt_test_torch.view(-1, 28 * 28)
    generated_test, mu_test, logvar_test, _ = model(dt_test_torch)
    mu_test = mu_test.detach().numpy()
    logvar_test = logvar_test.detach().numpy()

    # Compute KLD divergences for all train set
    kld_train = compute_kl_divergence(mu_train, logvar_train)

    # Compute p-values
    kld_test = compute_kl_divergence(mu_test, logvar_test)
    pvals = []
    for i in range(kld_test.shape[0]):
        all_values = numpy.concatenate((kld_train, kld_test[i].reshape(-1)))
        pvals.append(numpy.argwhere(kld_train >= kld_test[i]).shape[0]/all_values.shape[0]) 

    return(numpy.array(pvals), kld_train)