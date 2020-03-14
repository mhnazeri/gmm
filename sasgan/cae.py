"""
Implementation of Contractive Autoencoder to compress feature vector into
latent feature vector
"""
import sys
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.loader import CAEDataset
from utils import config


class Encoder(nn.Module):
    """Encoder network of CAE"""

    def __init__(self, n_inputs=14, n_hidden=28, n_latent=7, activation="sigmoid"):
        super(Encoder, self).__init__()
        self.n_inputs = n_inputs
        self.n_latent = n_latent
        self.fc_1 = nn.Linear(self.n_inputs, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_latent)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise "{} function is not supported".format(self.activation)

    def forward(self, x):
        x = x.view(-1, self.n_inputs)
        x = self.activation(self.fc_1(x))
        return self.fc_2(x)


class Decoder(nn.Module):
    """Decoder network of CAE"""

    def __init__(self, n_inputs=14, n_hidden=28, n_latent=1, activation="sigmoid"):
        super(Decoder, self).__init__()
        self.n_latent = n_latent
        self.n_inputs = n_inputs

        # encoder network, two layer mlp
        self.fc_1 = nn.Linear(n_latent, n_hidden)
        self.fc_2 = nn.Linear(n_hidden, n_inputs)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise "{} function is not supported".format(self.activation)

    def forward(self, x):
        x = x.view(-1, self.n_latent)
        x = self.activation(self.fc_1(x))
        return self.fc_2(x)


def loss_function(output_encoder, outputs, inputs, lamda=1e-4):

    criterion = nn.BCEWithLogitsLoss()

    assert (
        outputs.shape == inputs.shape
    ), f"outputs.shape : {outputs.shape} != inputs.shape : {inputs.shape}"

    loss1 = criterion(outputs, inputs)

    output_encoder.backward(torch.ones(output_encoder.size()), retain_graph=True)

    inputs.grad.requires_grad = True
    # Frobenious norm, the square root of sum of all elements (square value)
    # in a jacobian matrix
    loss2 = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))
    inputs.grad.data.zero_()
    loss = loss1 + (lamda * loss2)
    return loss


def make_cae(
    dataloader_train,
    n_inputs=14,
    n_latent=1,
    n_hidden=28,
    batch=40,
    epochs=50,
    activation="sigmoid",
):
    """create the whole cae"""
    encoder = Encoder(n_inputs, n_hidden, n_latent, activation)
    decoder = Decoder(n_inputs, n_hidden, n_latent, activation)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.005
    )
    losses = []
    loss = torch.tensor([0])
    for e in range(1, epochs + 1):
        print(f"epoch: {e}")

        for i, samples in enumerate(dataloader_train, 1):
            # change (batch, 100, n_inputs) to (100, n_inputs) if use NuSceneDataloader
            # samples = samples[0].view(-1, n_inputs)
            samples = samples.view(-1, n_inputs)

            # remove zero rows
            # valid_rows = []
            # for row_idx in range(samples.size(0)):
            #     if not torch.all(samples[row_idx, :] == 0):
            #         valid_rows.append(row_idx)

            # samples = samples[valid_rows, :]

            samples.requires_grad = True
            samples.retain_grad()

            outputs_encoder = encoder(samples)

            outputs = decoder(outputs_encoder)
            loss = loss_function(outputs_encoder, outputs, samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss)
        print(f"epoch/epochs: {e}/{epochs} loss: {loss.item():.4f}")
        torch.save(encoder.state_dict(), f"./models_state/model_cae_{e}.pt")

    plt.plot(range(50), losses, "r-")
    plt.xlabel("# Epochs")
    plt.ylabel("Binary Cross Entropy with Logits Loss")
    plt.show()


if __name__ == "__main__":
    cae_config = config("CAE")
    root = config("Paths")["data_root"]
    torch.manual_seed(42)
    data = CAEDataset(root)
    data = DataLoader(data, batch_size=int(cae_config["batch_size"]), shuffle=True, num_workers=2, drop_last=True)

    make_cae(data, int(cae_config["input_dim"]), int(cae_config["embed_dim"]),
             int(cae_config["hidden_dim"]), int(cae_config["batch_size"]),
             int(cae_config["epochs"]), cae_config["activation"])
