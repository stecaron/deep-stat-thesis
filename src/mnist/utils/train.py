import numpy
import time
import torch
import pandas

from src.mnist.utils.loss_vae import calculate_loss
from src.cars.loss.perceptual_loss import LossNetwork
from src.mnist.utils.loss_vae import perceptual_loss
from src.cars.loss.sparse_loss import sparse_loss


def train_mnist(train_loader,
                model,
                criterion,
                n_epoch,
                experiment,
                device,
                model_name,
                loss_func=None,
                loss_type="perceptual"):

    model.train()

    if loss_type == "perceptual":
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

            encoded, decoded = model(inputs)

            if loss_type == "perceptual":
                loss = perceptual_loss(outputs, decoded, loss_network)
            elif loss_type == "sparsity":
                mse_loss = loss_func(decoded, outputs)
                l1_loss = sparse_loss(model, inputs)
                # add the sparsity penalty
                loss = mse_loss + 0.001 * l1_loss
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
                    #test_loader,
                    model,
                    criterion,
                    n_epoch,
                    experiment,
                    #scheduler,
                    beta_list,
                    beta_epoch,
                    model_name,
                    device,
                    #latent_dim,
                    loss_type="binary",
                    flatten=True):

    if loss_type == "perceptual":
        loss_network = LossNetwork(device)
    else:
        loss_network = None

    KLD_perc_list = []

    # cols_mu = ["mu_"+str(i) for i in range(latent_dim)]
    # cols_var = ["var_"+str(i) for i in range(latent_dim)]
    # cols_name = ["epoch", "outliers", "kld", "rcl", "pen"] + cols_mu + cols_var
    # df_test_monitoring = pandas.DataFrame(columns=cols_name)

    for epoch in range(n_epoch):
        train_loss = 0.0

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

            model.train()

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
            experiment.log_metric("KLD", KLD.detach().cpu())
            experiment.log_metric("RCL", loss.detach().cpu() - KLD.detach().cpu())
            loss.backward()
            train_loss += loss.item()
            criterion.step()

            z_mu = z_mu.cpu()
            z_var = z_var.cpu()
            y = y.cpu()

        # Test on test data loader
        # for i, (x, y) in enumerate(test_loader):
        #     # reshape the data into [batch_size, 784]
        #     if flatten:
        #         x = x.view(-1, 28 * 28)

        #     model.to(device)
        #     x = x.to(device)
        #     y = y.to(device)

        #     model.eval()
        #     reconstructed_x, z_mu, z_var, _ = model(x, device=device)
        #     loss, KLD, RCL = calculate_loss(x,
        #                                reconstructed_x,
        #                                z_mu,
        #                                z_var,
        #                                loss_type=loss_type,
        #                                beta=beta,
        #                                loss_network=loss_network)
        #     pen = loss - KLD - RCL
        #     data_epoch = numpy.concatenate((numpy.array(epoch).reshape(1), y.detach().cpu().numpy(), KLD.detach().cpu().numpy().reshape(1), RCL.detach().cpu().numpy().reshape(1), pen.detach().cpu().numpy().reshape(1), z_mu.detach().cpu().numpy().reshape(-1), numpy.exp(z_var.detach().cpu().reshape(-1))))
        #     df_test_monitoring = df_test_monitoring.append(pandas.DataFrame(data_epoch.reshape(1,-1), columns=cols_name), ignore_index=True)


        train_loss = train_loss / len(train_loader) * train_loader.batch_size
        KLD_perc = numpy.around((KLD / loss).cpu().detach().numpy(), 2)
        KLD_perc_list.append(KLD_perc)

        end = time.time()
        print(
            f'Epoch {epoch} ... Train Loss: {train_loss:.2f} ... time: {int(end - start)}'
        )
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("kld_percentage", KLD_perc)

        # df_test_monitoring.to_csv("test_loss.csv")

        if epoch == 0:
            best_loss = train_loss

        if train_loss <= best_loss:
            model.cpu()
            torch.save(model, f'{model_name}.pt')
            model.save_weights(f'./{model_name}.h5')

        # Save KLD percentage
        if epoch == (n_epoch-1):
            col_names = ["epoch", "kld_percentage"]
            df_results = pandas.DataFrame(columns=col_names)
            df_results["epoch"] = numpy.array(range(n_epoch))
            df_results["kld_percentage"] = numpy.array(KLD_perc_list)
            df_results.to_csv(f'{model_name}_kld_percentage.csv')
