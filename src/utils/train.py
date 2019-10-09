import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def train(net, train_loader, optimizer, loss_function, num_epochs):
    """
    This function train the network.
    
    Args:
        pytorch_module (torch.nn.Module): The neural network to train.
    """
    print(net)

    # Train
    for epoch in range(num_epochs):
        net.train()
        for step, batch in enumerate(train_loader): 
            x = batch.float()
            target = batch.float() # output same as input
            
            # x = F.normalize(x, p=2, dim=1)

            encoded, decoded = net(x)

            loss = loss_function(decoded, target)
            # clear gradients for this training step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

