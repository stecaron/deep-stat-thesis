import numpy
import time

from src.mnist.utils.loss_vae import calculate_loss


def train_mnist(train_loader, model, criterion, loss_func, n_epoch,
                experiment):

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
        # log experiment result
        experiment.log_metric("train_loss", loss.data.numpy())


def train_mnist_vae(train_loader,
                    model,
                    criterion,
                    n_epoch,
                    experiment,
                    beta,
                    gpu=False,
                    loss_type="binary",
                    flatten=True):
    # set the train mode
    model.train()

    for epoch in range(n_epoch):
        train_loss = 0

        start = time.time()
        for i, (x, y) in enumerate(train_loader):
            # reshape the data into [batch_size, 784]
            if flatten:
                x = x.view(-1, 28 * 28)
            
            if gpu:
                x = x.cuda()
                y = y.cuda()


            criterion.zero_grad()
            reconstructed_x, z_mu, z_var, _ = model(x, gpu=gpu)
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var, loss_type=loss_type, beta=beta)
            loss.backward()
            train_loss += loss.item()
            criterion.step()

        train_loss /= len(train_loader)

        end = time.time()
        print(f'Epoch {epoch} ... Train Loss: {train_loss:.2f} ... time: {int(end - start)}')
        experiment.log_metric("train_loss", train_loss)