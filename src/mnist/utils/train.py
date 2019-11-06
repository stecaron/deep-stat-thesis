import numpy

def train_mnist(train_loader, model, criterion, loss_func, n_epoch):

    for epoch in range(n_epoch):
        for step, (x, target) in enumerate(train_loader):
            inputs = x.view(-1, 28*28)
            outputs = x.view(-1, 28*28)

            encoded, decoded = model(inputs)

            loss = loss_func(decoded, outputs)
            criterion.zero_grad()
            loss.backward()
            criterion.step()

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())