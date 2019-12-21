import torch
import torch.nn.functional as F


def calculate_loss(x, reconstructed_x, mean, log_var, beta, loss_type="binary"):
    # reconstruction loss
    if loss_type == "binary":
        RCL = F.binary_cross_entropy(reconstructed_x, x, reduction="mean")
    elif loss_type == "mse":
        RCL = F.mse_loss(reconstructed_x, x, reduction="mean")

    # kl divergence loss
    # KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    KLD = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mean.pow(2) - 1 - log_var, 1))
    # Normalise by same number of elements as in reconstruction
    # KLD /= x.shape[0] * x.shape[1]
    KLD = beta * KLD

    return RCL + KLD