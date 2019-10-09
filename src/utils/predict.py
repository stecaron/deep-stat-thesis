import torch


def predict(net, x_predict):

    # Set the model to predict
    net.eval()

    # Transform the data to predict to Tensor
    x = torch.from_numpy(x_predict).float()

    encoded, decoded = net(x)

    return encoded



