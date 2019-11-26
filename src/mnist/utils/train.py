import numpy


def train_mnist(train_loader, model, criterion, loss_func, n_epoch):

    for epoch in range(n_epoch):
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.float()
            outputs = inputs

            encoded, decoded = model(inputs)

            loss = loss_func(decoded, outputs)
            criterion.zero_grad()
            loss.backward()
            criterion.step()

        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())
