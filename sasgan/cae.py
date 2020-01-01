"""
Implementation of Contractive Autoencoder to compress feature vector into
latent feature vector
"""
import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """Encoder network of CAE"""

    def __init__(self, n_inputs, n_hidden=1, batch=40, activation='sigmoid'):
        super(Encoder, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.batch = batch

        # encoder network, two layer mlp
        self.fc_1 = nn.Linear(n_inputs, 28)
        self.fc_2 = nn.Linear(28, n_hidden)

        if activation == 'sigmoid':
            self.activation = activation
        elif activation == 'tanh':
            self.activation = nn.Tanh
        else:
            raise f"{activation} function is not supported"

    def forward(self, x):
        x = self.activation(self.fc_1(x))
        return fc_2(x)


class Decoder(nn.Module):
    """Decoder network of CAE"""

    def __init__(self, n_inputs=1, n_output=14, batch=40, activation='sigmoid'):
        super(Decoder, self).__init__()
        self.n_inputs = n_inputs
        self.n_output = n_output
        self.batch = batch

        # encoder network, two layer mlp
        self.fc_1 = nn.Linear(n_inputs, 28)
        self.fc_2 = nn.Linear(28, n_output)

        if activation == 'sigmoid':
            self.activation = activation
        elif activation == 'tanh':
            self.activation = nn.Tanh
        else:
            raise f"{activation} function is not supported"

    def forward(self, x):
        x = self.activation(self.fc_1(x))
        return fc_2(x)


def loss_function(output_encoder, outputs, inputs, lamda = 1e-4):

    criterion = nn.CrossEntropyLoss()
    assert outputs.shape == inputs.shape, f'outputs.shape : {outputs.shape} != inputs.shape : {inputs.shape}'
    loss1 = criterion(outputs, inputs)

    output_encoder.backward(torch.ones(outputs_encoder.size()), retain_graph=True)
    # Frobenious norm, the square root of sum of all elements (square value)
    # in a jacobian matrix
    loss2 = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))
    inputs.grad.data.zero_()
    loss = loss1 + (lamda * loss2)
    return loss


def make_cae(n_inputs=14, n_output=14, n_hidden=1, batch=40, epochs=50, activation='sigmoid'):
    """create the whole cae"""
    encoder = Encoder(n_inputs, n_hidden, batch, activation)
    decoder = Decoder(n_inputs, n_output, batch, activation)

    optimizer = optim.Adam([{"params": encoder.parameters()},
                            {"params": decoder.parameters()}], lr=0.005)

    optimizer_cond = Adam(encoder.parameters(), lr=0.005)

    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        for i, (inputs, labels) in enumerate(dataloader_train):

            inputs.retain_grad()
            inputs.requires_grad_(True)

            outputs_encoder = encoder(inputs)
            outputs = decoder(outputs_encoder)
            loss = loss_function(outputs_encoder, outputs, inputs, lam)

            inputs.requires_grad_(False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch/epochs: {e}/{epochs} loss: {loss.item():.4f}')

