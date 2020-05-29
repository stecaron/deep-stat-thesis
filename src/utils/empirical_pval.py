import numpy
import torch
import pandas

import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from src.utils.kl import compute_kl_divergence
from src.utils.kl import compute_kl_divergence_2_dist


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
        pvals.append(
            numpy.argwhere(kld_train >= kld_test[i]).shape[0] /
            all_values.shape[0])

    return (numpy.array(pvals), kld_train)


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
        pvals.append(
            numpy.argwhere(kld_train >= kld_test[i]).shape[0] /
            all_values.shape[0])

    return (numpy.array(pvals), kld_train)


def compute_reconstruction_pval(train_loader, model, test_loader, device):

    model.eval()

    error_train = []
    # Encode train data
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        with torch.no_grad():
            x_decoded, _, _, _ = model(x, device=device)
            x_error = torch.mean(F.mse_loss(x_decoded, x, reduce=False),
                                 axis=(1, 2, 3))
            x_error = x_error.cpu()
        error_train.append(x_error)

    error_test = []
    # Encode train data
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        with torch.no_grad():
            x_decoded, _, _, _ = model(x, device=device)
            x_error = torch.mean(F.mse_loss(x_decoded, x, reduce=False),
                                 axis=(1, 2, 3))
            x_error = x_error.cpu()
        error_test.append(x_error)

    error_train = numpy.concatenate(error_train)
    error_test = numpy.concatenate(error_test)

    # Compute p-values
    pvals = []
    for i in range(error_test.shape[0]):
        all_values = numpy.concatenate(
            (error_train, error_test[i].reshape(-1)))
        pvals.append(
            numpy.argwhere(error_train >= error_test[i]).shape[0] /
            all_values.shape[0])

    return (numpy.array(pvals), error_train)


def compute_pval_loaders_mixture(train_loader, test_loader, model, device,
                                 method, experiment):

    model.eval()

    # Encode train data
    mu_train = []
    logvar_train = []
    ind_cat_train = []
    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        ind_cat_train.append(y)
        _, z_mu, z_var, _ = model(x, device=device)
        z_mu = z_mu.cpu()
        z_var = z_var.cpu()
        mu_train.append(z_mu.detach().numpy())
        logvar_train.append(z_var.detach().numpy())

    mu_train = numpy.concatenate(mu_train)
    logvar_train = numpy.concatenate(logvar_train)
    var_train = numpy.exp(logvar_train)

    # Encode test data
    ind_cat_test = []
    mu_test = []
    logvar_test = []
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        ind_cat_test.append(y)
        _, z_mu, z_var, _ = model(x, device=device)
        z_mu = z_mu.cpu()
        z_var = z_var.cpu()
        mu_test.append(z_mu.detach().numpy())
        logvar_test.append(z_var.detach().numpy())

    mu_test = numpy.concatenate(mu_test)
    logvar_test = numpy.concatenate(logvar_test)
    var_test = numpy.exp(logvar_test)

    ind_cat_train = numpy.concatenate(ind_cat_train)
    ind_cat_test = numpy.concatenate(ind_cat_test)

    #test
    ind_test = []
    for i, (x, y) in enumerate(train_loader):
        ind_test.append(numpy.argwhere(y == 1) + i * train_loader.batch_size)

    # Build train mixture
    if method == "mean":
        mu_all_train = numpy.mean(mu_train, axis=0)
        var_all_train = numpy.mean(numpy.exp(logvar_train), axis=0)

    # Compute KLD divergences for all train set
    kld_train = compute_kl_divergence_2_dist(mu_train, mu_all_train, numpy.sqrt(var_train),
                                             numpy.sqrt(var_all_train))


    pandas.DataFrame(mu_train[ind_cat_train.astype(bool) == False, :]).to_csv("mu_inliers.csv")
    pandas.DataFrame(mu_train[ind_cat_train.astype(bool) == True, :]).to_csv("mu_outliers.csv")
    pandas.DataFrame(var_train[ind_cat_train.astype(bool) == False, :]).to_csv("sigma_inliers.csv")
    pandas.DataFrame(var_train[ind_cat_train.astype(bool) == True, :]).to_csv("sigma_outliers.csv")

    # test
    stats = {
        'mu_train_inliers': numpy.mean(mu_train[ind_cat_train.astype(bool) == False, :]),
        'mu_train_outliers': numpy.mean(mu_train[ind_cat_train.astype(bool) == True, :]),
        'mu_train_inliers_sd': numpy.std(mu_train[ind_cat_train.astype(bool) == False, :]),
        'mu_train_outliers_sd': numpy.std(mu_train[ind_cat_train.astype(bool) == True, :]),
        'var_train_inliers': numpy.mean(var_train[ind_cat_train.astype(bool) == False, :]),
        'var_train_outliers': numpy.mean(var_train[ind_cat_train.astype(bool) == True, :]),
        'mu_all_train': numpy.mean(mu_all_train),
        'var_all_train': numpy.mean(var_all_train),
        'mu_test_inliers': numpy.mean(mu_test[ind_cat_test.astype(bool) == False, :]),
        'mu_test_outliers': numpy.mean(mu_test[ind_cat_test.astype(bool) == True, :]),
        'var_test_inliers': numpy.mean(var_test[ind_cat_test.astype(bool) == False, :]),
        'var_test_outliers': numpy.mean(var_test[ind_cat_test.astype(bool) == True, :])
    }

    experiment.log_metrics(stats)

    kwargs = dict(alpha=0.5, bins=30)

    # KLD distances between inliers and outliers
    kld_train_inliers = kld_train[ind_cat_train.astype(bool) == False]
    kld_train_outliers = kld_train[ind_cat_train.astype(bool) == True]
    plt.hist(kld_train_inliers, **kwargs, color='g', label='Cars')
    plt.hist(kld_train_outliers, **kwargs, color='r', label='Dogs')
    plt.gca().set(title='KLD train distribution', ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="kld_train", overwrite=True)
    plt.clf()

    # Average train mu distribution
    mu_cars = numpy.mean(mu_train[ind_cat_train.astype(bool) == False, :],
                         axis=1)
    mu_dogs = numpy.mean(mu_train[ind_cat_train.astype(bool) == True, :],
                         axis=1)
    kwargs = dict(alpha=0.5, bins=30)
    plt.hist(mu_cars, **kwargs, color='g', label='Cars')
    plt.hist(mu_dogs, **kwargs, color='r', label='Dogs')
    plt.gca().set(title='Average mu train distribution', ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="mu_train_dist", overwrite=True)
    plt.clf()

    # Average train var distribution
    var_cars = numpy.mean(var_train[ind_cat_train.astype(bool) == False, :],
                         axis=1)
    var_dogs = numpy.mean(var_train[ind_cat_train.astype(bool) == True, :],
                         axis=1)
    kwargs = dict(alpha=0.5, bins=30)
    plt.hist(var_cars, **kwargs, color='g', label='Cars')
    plt.hist(var_dogs, **kwargs, color='r', label='Dogs')
    plt.gca().set(title='Average var train distribution', ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="var_train_dist", overwrite=True)
    plt.clf()

    # Average train mu over dimensions distribution
    mu_cars = numpy.mean(mu_train[ind_cat_train.astype(bool) == False, :],
                         axis=0)
    mu_dogs = numpy.mean(mu_train[ind_cat_train.astype(bool) == True, :],
                         axis=0)
    kwargs = dict(alpha=0.5, bins=30)
    plt.hist(mu_cars, **kwargs, color='g', label='Cars')
    plt.hist(mu_dogs, **kwargs, color='r', label='Dogs')
    plt.hist(mu_all_train, **kwargs, color='b', label='All train')
    plt.gca().set(title='50-dimensions mu distribution over train',
                  ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="mu_train_dist_dims", overwrite=True)
    plt.clf()

    # Average train var over dimensions distribution
    var_cars = numpy.mean(var_train[ind_cat_train.astype(bool) == False, :],
                         axis=0)
    var_dogs = numpy.mean(var_train[ind_cat_train.astype(bool) == True, :],
                         axis=0)
    kwargs = dict(alpha=0.5, bins=30)
    plt.hist(var_cars, **kwargs, color='g', label='Cars')
    plt.hist(var_dogs, **kwargs, color='r', label='Dogs')
    plt.hist(var_all_train, **kwargs, color='b', label='All train')
    plt.gca().set(title='50-dimensions var distribution over train',
                  ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="var_train_dist_dims", overwrite=True)
    plt.clf()

    # Average test mu over dimensions distribution
    mu_cars = numpy.mean(mu_test[ind_cat_test.astype(bool) == False, :],
                         axis=0)
    mu_dogs = numpy.mean(mu_test[ind_cat_test.astype(bool) == True, :], axis=0)
    kwargs = dict(alpha=0.5, bins=30)
    plt.hist(mu_cars, **kwargs, color='g', label='Cars')
    plt.hist(mu_dogs, **kwargs, color='r', label='Dogs')
    plt.hist(mu_all_train, **kwargs, color='b', label='All train')
    plt.gca().set(title='50-dimensions mu distribution over test',
                  ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="mu_test_dist_dims", overwrite=True)
    plt.clf()

    # Average test var over dimensions distribution
    var_cars = numpy.mean(var_test[ind_cat_test.astype(bool) == False, :],
                         axis=0)
    var_dogs = numpy.mean(var_test[ind_cat_test.astype(bool) == True, :], axis=0)
    kwargs = dict(alpha=0.5, bins=30)
    plt.hist(var_cars, **kwargs, color='g', label='Cars')
    plt.hist(var_dogs, **kwargs, color='r', label='Dogs')
    plt.hist(var_all_train, **kwargs, color='b', label='All train')
    plt.gca().set(title='50-dimensions mu distribution over test',
                  ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="mu_test_dist_dims", overwrite=True)
    plt.clf()

    # First mu dimension distribution
    mu_cars = mu_train[ind_cat_train.astype(bool) == False, 0]
    mu_dogs = mu_train[ind_cat_train.astype(bool) == True, 0]
    kwargs = dict(alpha=0.5, bins=30)
    plt.hist(mu_cars, **kwargs, color='g', label='Cars')
    plt.hist(mu_dogs, **kwargs, color='r', label='Dogs')
    plt.hist(mu_all_train[0], **kwargs, color='b', label='All train')
    plt.gca().set(title='1st-dimension mu distribution over train',
                  ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="mu_train_1st_dim", overwrite=True)
    plt.clf()

    # Show 16 randoms dimensios
    random_dim = numpy.sort(random.sample(list(numpy.arange(0, 4)), 4))

    fig, axs = plt.subplots(1,
                            4,
                            figsize=(15, 6),
                            facecolor='w',
                            edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for i in range(4):
        dim = random_dim[i]
        mu_cars = mu_train[ind_cat_train.astype(bool) == False, dim]
        mu_dogs = mu_train[ind_cat_train.astype(bool) == True, dim]
        axs[i].hist(mu_cars, **kwargs, color='g', label='Cars')
        axs[i].hist(mu_dogs, **kwargs, color='r', label='Dogs')
        axs[i].axvline(mu_all_train[dim], color='b', label='All train')
        axs[i].set_title(str(dim))

    plt.xticks(fontsize=14)
    plt.legend()
    experiment.log_figure(figure_name="mu_train_16_random_dims", overwrite=True)
    plt.clf()


    # Show 16 randoms dimensios

    fig, axs = plt.subplots(1,
                            4,
                            figsize=(15, 6),
                            facecolor='w',
                            edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()

    for i in range(4):
        dim = random_dim[i]
        var_cars = var_train[ind_cat_train.astype(bool) == False, dim]
        var_dogs = var_train[ind_cat_train.astype(bool) == True, dim]
        axs[i].hist(var_cars, **kwargs, color='g', label='Cars')
        axs[i].hist(var_dogs, **kwargs, color='r', label='Dogs')
        axs[i].axvline(var_all_train[dim], color='b', label='All train')
        axs[i].set_title(str(dim))

    plt.xticks(fontsize=14)
    plt.legend()
    experiment.log_figure(figure_name="var_train_16_random_dims", overwrite=True)
    plt.clf()

    # d = {
    #     'mu1': mu_train[:, random_dim[0]],
    #     'mu2': mu_train[:, random_dim[1]],
    #     'mu3': mu_train[:, random_dim[2]],
    #     'outliers': ind_cat_train.astype(bool),
    # }
    # df = pandas.DataFrame(data=d)
    # groups = numpy.unique(ind_cat_train.astype(bool)).tolist()
    # colors = ("red", "blue")
    # group_names = ("inliers", "outliers")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for group, color, group_name in zip(groups, colors, group_names):
    #     dt_temp = df.loc[df["outliers"] == group]
    #     ax.scatter(dt_temp["mu1"],
    #                dt_temp["mu2"],
    #                dt_temp["mu3"],
    #                alpha=0.4,
    #                c=color,
    #                label=group_name)

    # ax.legend()
    # plt.show()

    # Compute p-values
    kld_test = compute_kl_divergence_2_dist(mu_test, mu_all_train, numpy.sqrt(var_test),
                                            numpy.sqrt(var_all_train))

    # KLD distances for test
    kld_test_inliers = kld_test[ind_cat_test.astype(bool) == False]
    kld_test_outliers = kld_test[ind_cat_test.astype(bool) == True]
    plt.hist(kld_test_inliers, **kwargs, color='g', label='Cars')
    plt.hist(kld_test_outliers, **kwargs, color='r', label='Dogs')
    plt.gca().set(title='KLD test distribution', ylabel='Frequency')
    plt.legend()
    experiment.log_figure(figure_name="kld_test", overwrite=True)
    plt.clf()

    pvals = []
    for i in range(kld_test.shape[0]):
        all_values = numpy.concatenate((kld_train, kld_test[i].reshape(-1)))
        pvals.append(
            numpy.argwhere(kld_train >= kld_test[i]).shape[0] /
            all_values.shape[0])

    return (numpy.array(pvals), kld_train)
