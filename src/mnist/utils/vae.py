import torch
import torch.nn.functional as F


def calculate_loss(x, reconstructed_x, mean, log_var, loss_type="binary"):
    # reconstruction loss
    if loss_type == "binary":
        RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    elif loss_type == "mse":
        RCL = F.mse_loss(reconstructed_x, x, size_average=False)

    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # Normalise by same number of elements as in reconstruction
    # KLD /= x.shape[0] * x.shape[1]

    return RCL + KLD