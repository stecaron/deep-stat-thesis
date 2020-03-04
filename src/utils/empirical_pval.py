import numpy
import torch

import torch.nn.functional as F

from src.utils.kl import compute_kl_divergence


def compute_empirical_pval(dt_train, model, dt_test):

    # Encode train data
    dt_train_torch = torch.from_numpy(dt_train).float()
    dt_train_torch.transpose_(1, 3)
    #dt_train_torch = torch.unsqueeze(dt_train, 1)
    #dt_train_torch = dt_train_torch.view(-1, 28 * 28)
    generated_train, mu_train, logvar_train, _ = model(dt_train_torch)
    mu_train = mu_train.detach().numpy()
    logvar_train = logvar_train.detach().numpy()

    # Encode test data
    dt_test_torch = torch.from_numpy(dt_test).float()
    dt_test_torch.transpose_(1, 3)
    #dt_test_torch = torch.unsqueeze(dt_test, 1)
    #dt_test_torch = dt_test_torch.view(-1, 28 * 28)
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


def compute_pval_loaders(train_loader, test_loader, model, device):

    model.eval()

    # Encode train data
    mu_train = []
    logvar_train = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        _, z_mu, z_var, _ = model(x, device=device)
        z_mu = z_mu.cpu()
        z_var = z_var.cpu()
        mu_train.append(z_mu.detach().numpy())
        logvar_train.append(z_var.detach().numpy())

    mu_train = numpy.concatenate(mu_train)
    logvar_train = numpy.concatenate(logvar_train)

    # Encode test data
    mu_test = []
    logvar_test = []
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        _, z_mu, z_var, _ = model(x, device=device)
        z_mu = z_mu.cpu()
        z_var = z_var.cpu()
        mu_test.append(z_mu.detach().numpy())
        logvar_test.append(z_var.detach().numpy())

    mu_test = numpy.concatenate(mu_test)
    logvar_test = numpy.concatenate(logvar_test)

    # Compute KLD divergences for all train set
    kld_train = compute_kl_divergence(mu_train, logvar_train)

    # Compute p-values
    kld_test = compute_kl_divergence(mu_test, logvar_test)
    pvals = []
    for i in range(kld_test.shape[0]):
        all_values = numpy.concatenate((kld_train, kld_test[i].reshape(-1)))
        pvals.append(numpy.argwhere(kld_train >= kld_test[i]).shape[0]/all_values.shape[0])

    return(numpy.array(pvals), kld_train)

def compute_reconstruction_pval(train_loader, model, test_loader, device):

    model.eval()

    error_train = []
    # Encode train data
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        with torch.no_grad():
            x_decoded, _, _, _ = model(x, device=device)
            x_error = torch.mean(F.mse_loss(x_decoded, x, reduce=False), axis=(1, 2, 3))
            x_error = x_error.cpu()
        error_train.append(x_error)

    error_test = []
    # Encode train data
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        with torch.no_grad():
            x_decoded, _, _, _ = model(x, device=device)
            x_error = torch.mean(F.mse_loss(x_decoded, x, reduce=False), axis=(1, 2, 3))
            x_error = x_error.cpu()
        error_test.append(x_error)

    error_train = numpy.concatenate(error_train)
    error_test = numpy.concatenate(error_test)

    # Compute p-values
    pvals = []
    for i in range(error_test.shape[0]):
        all_values = numpy.concatenate((error_train, error_test[i].reshape(-1)))
        pvals.append(numpy.argwhere(error_train >= error_test[i]).shape[0]/all_values.shape[0])

    return(numpy.array(pvals), error_train)