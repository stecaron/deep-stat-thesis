import torch


def denormalize(x, mean, std, gpu):

    if gpu:
        mean = torch.FloatTensor(mean).cuda()
        std = torch.FloatTensor(std).cuda()
    
    x_new = x.new(*x.size())
    x_new[:, :, 0] = x[:, :, 0] * std[0] + mean[0]
    x_new[:, :, 1] = x[:, :, 1] * std[1] + mean[1]
    x_new[:, :, 2] = x[:, :, 2] * std[2] + mean[2]

    return(x_new)