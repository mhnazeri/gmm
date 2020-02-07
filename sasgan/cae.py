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


sys.path.append("/home/nao/Projects/sasgan/data")
from loader import CAEDataset


class Encoder(nn.Module):
    """Encoder network of CAE"""

    def __init__(self, n_inputs=14, n_hidden=1, activation="sigmoid"):
        super(Encoder, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.fc_1 = nn.Linear(self.n_inputs, 28)
        self.fc_2 = nn.Linear(28, n_hidden)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise f"{activation} function is not supported"

    def forward(self, x):
        x = self.activation(self.fc_1(x))
        return self.fc_2(x)


class Decoder(nn.Module):
    """Decoder network of CAE"""

    def __init__(self, n_inputs=1, n_output=14, activation="sigmoid"):
        super(Decoder, self).__init__()
        self.n_inputs = n_inputs
        self.n_output = n_output

        # encoder network, two layer mlp
        self.fc_1 = nn.Linear(n_inputs, 28)
        self.fc_2 = nn.Linear(28, n_output)

        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise f"{activation} function is not supported"

    def forward(self, x):
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
    n_output=14,
    n_hidden=1,
    batch=40,
    epochs=50,
    activation="sigmoid",
):
    """create the whole cae"""
    encoder = Encoder(n_inputs, n_hidden, activation).double()
    decoder = Decoder(n_hidden, n_output, activation).double()

    optimizer = optim.Adam(
        [{"params": encoder.parameters()}, {"params": decoder.parameters()}], lr=0.005
    )
    losses = []
    loss = torch.tensor([0])
    for e in range(epochs):
        print(f"epoch: {e}")

        for i_batch, samples in enumerate(dataloader_train):
            samples.requires_grad = True

            outputs_encoder = encoder(samples)

            samples.retain_grad()
            outputs = decoder(outputs_encoder)
            loss = loss_function(outputs_encoder, outputs, samples)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss)
        print(f"epoch/epochs: {e}/{epochs} loss: {loss.item():.4f}")
        torch.save(encoder.state_dict(), f"./models/model_{e}.pt")

    plt.plot(range(50), losses, "r-")
    plt.xlabel("# Epochs")
    plt.ylabel("Binary Cross Entropy with Logits Loss")
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    root = "/home/nao/Projects/sasgan"
    data = CAEDataset(
        os.path.join(root, "data/exported_json_data/scene-1100.json"),
        os.path.join("data/nuScene-mini"),
    )
    data = DataLoader(data, batch_size=40, shuffle=True, num_workers=2, drop_last=True)
    make_cae(data)
