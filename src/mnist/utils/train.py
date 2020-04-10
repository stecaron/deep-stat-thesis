import numpy
import time
import torch

from src.mnist.utils.loss_vae import calculate_loss
from src.cars.loss.perceptual_loss import LossNetwork
from src.mnist.utils.loss_vae import perceptual_loss


def train_mnist(train_loader,
                model,
                criterion,
                n_epoch,
                experiment,
                device,
                model_name,
                loss_func=None,
                perceptual_ind=False):

    model.train()

    if perceptual_ind:
        loss_network = LossNetwork(device)

    for epoch in range(n_epoch):

        start = time.time()
        train_loss = 0.0

        for data in train_loader:
            inputs, targets = data
            inputs = inputs.float()
            outputs = inputs

            model.to(device)
            inputs = inputs.to(device)
            outputs = inputs.to(device)

            decoded = model(inputs)

            if perceptual_ind:
                loss = perceptual_loss(outputs, decoded, loss_network)
            else:
                loss = loss_func(decoded, outputs)

            train_loss += loss.item()
            criterion.zero_grad()
            loss.backward()
            criterion.step()

        train_loss = train_loss / len(train_loader) * train_loader.batch_size

        end = time.time()
        print(
            f'Epoch: {epoch} ... train loss: {train_loss} ... time: {int(end - start)}'
        )
        # log experiment result
        experiment.log_metric("train_loss", train_loss)

        if epoch == 0:
            best_loss = train_loss

        if train_loss <= best_loss:
            model.cpu()
            torch.save(model, f'{model_name}.pt')
            model.save_weights(f'./{model_name}.h5')


def train_mnist_vae(train_loader,
                    model,
                    criterion,
                    n_epoch,
                    experiment,
                    scheduler,
                    beta_list,
                    beta_epoch,
                    model_name,
                    device,
                    loss_type="binary",
                    flatten=True):
    # set the train mode
    model.train()
    if loss_type == "perceptual":
        loss_network = LossNetwork(device)
    else:
        loss_network = None



    for epoch in range(n_epoch):
        train_loss = 0.0

        list_mu_maj = []
        list_mu_min = []
        list_var_maj = []
        list_var_min = []

        step = 0
        for beta_step in beta_epoch:
            if epoch < beta_step:
                beta = beta_list[step]
                break
            step += 1

        start = time.time()
        print(f"beta: {beta}")

        for i, (x, y) in enumerate(train_loader):
            # reshape the data into [batch_size, 784]
            if flatten:
                x = x.view(-1, 28 * 28)

            model.to(device)
            x = x.to(device)
            y = y.to(device)

            criterion.zero_grad()
            reconstructed_x, z_mu, z_var, _ = model(x, device=device)
            loss, KLD = calculate_loss(x,
                                       reconstructed_x,
                                       z_mu,
                                       z_var,
                                       loss_type=loss_type,
                                       beta=beta,
                                       loss_network=loss_network)
            loss.backward()
            train_loss += loss.item()
            criterion.step()
            scheduler.step()

            z_mu = z_mu.cpu()
            z_var = z_var.cpu()
            y = y.cpu()

            list_mu_maj.append(torch.mean(torch.mean(z_mu[numpy.where(y == 0)], 0)))
            list_mu_min.append(torch.mean(torch.mean(z_mu[numpy.where(y != 0)], 0)))
            list_var_maj.append(torch.mean(torch.mean(torch.exp(z_var[numpy.where(y == 0)]), 0)))
            list_var_min.append(torch.mean(torch.mean(torch.exp(z_var[numpy.where(y != 0)]), 0)))


        train_loss = train_loss / len(train_loader) * train_loader.batch_size
        KLD_perc = numpy.around((KLD / loss).cpu().detach().numpy(), 2)

        mean_mu_maj = sum(list_mu_maj)/len(list_mu_maj)
        mean_mu_min = sum(list_mu_min)/len(list_mu_min)
        mean_var_maj = sum(list_var_maj)/len(list_var_maj)
        mean_var_min = sum(list_var_min)/len(list_var_min)

        experiment.log_metric("mean_mu_maj", mean_mu_maj.detach().cpu())
        experiment.log_metric("mean_mu_min", mean_mu_min.detach().cpu())
        experiment.log_metric("mean_var_maj", mean_var_maj.detach().cpu())
        experiment.log_metric("mean_var_min", mean_var_min.detach().cpu())

        end = time.time()
        print(
            f'Epoch {epoch} ... Train Loss: {train_loss:.2f} ... time: {int(end - start)}'
        )
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("kld_percentage", KLD_perc)

        if epoch == 0:
            best_loss = train_loss

        if train_loss <= best_loss:
            model.cpu()
            torch.save(model, f'{model_name}.pt')
            model.save_weights(f'./{model_name}.h5')
