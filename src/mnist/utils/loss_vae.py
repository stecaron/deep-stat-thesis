import torch
import torch.nn.functional as F


def calculate_loss(x, reconstructed_x, mean, log_var, beta, loss_type="binary"):
    # reconstruction loss
    if loss_type == "binary":
        RCL = F.binary_cross_entropy(reconstructed_x, x, reduction="sum")
        #RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    elif loss_type == "mse":
        RCL = F.mse_loss(reconstructed_x, x, reduction="mean")

    # kl divergence loss
    # Version sum
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # Version mean
    #KLD = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mean.pow(2) - 1 - log_var, 1))
    KLD = beta * KLD

    print(KLD/(KLD + RCL))

    return RCL + KLD