import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_loss(x, reconstructed_x, mean, log_var, beta, loss_type="binary", loss_network=None):
    # reconstruction loss
    if loss_type == "binary":
        #RCL = F.binary_cross_entropy(reconstructed_x, x, reduction="mean") * x.shape[2] * x.shape[2] * x.shape[1]
        RCL = F.binary_cross_entropy(reconstructed_x, x, reduction="sum") / x.shape[0]
    elif loss_type == "mse":
        RCL = F.mse_loss(reconstructed_x, x, reduction="sum") / x.shape[0]
    elif loss_type == "perceptual":
        features_y = loss_network(x)
        features_x = loss_network(reconstructed_x)
        RCL = F.mse_loss(features_x[0], features_y[0].detach(), reduction='sum') / x.shape[0]


    # kl divergence loss
    # Version sum
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return KLD * beta + RCL, KLD * beta


def perceptual_loss(target, decoded, loss_network, by_image=False):
    features_y = loss_network(target)
    features_x = loss_network(decoded)
    if by_image:
        RCL = F.mse_loss(features_y[0], features_x[0], reduction='none')
    else:
        RCL = F.mse_loss(features_y[0], features_x[0], reduction='sum') / target.shape[0]

    return(RCL)