"""
Implementation of Contractive Autoencoder to compress feature vector into
latent feature vector
"""
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import load, save, tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from data.loader import CAEDataset
from utils import config, checkpoint_path
import seaborn as sns
import logging
from losses import cae_loss
import os
from numpy import inf, mean


logger = logging.getLogger(__name__)
sns.set(color_codes=True)

class Encoder(nn.Module):
    """Encoder network of CAE"""

    def __init__(self, n_inputs=13, n_hidden=28, n_latent=7, activation="sigmoid"):
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

    def __init__(self, n_inputs=13, n_hidden=28, n_latent=1, activation="sigmoid"):
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


def make_cae(dataloader_train, summary_writer):
    """
    The following function returns the required cae to be further used in the next sections of the model
    If there is any saved model of cae, It loads according to the loading strategy otherwise it just
        creates a model with the specified parameters and trains it from scratch.
    The model will train even though there is any saved model until there are iteratioins specified, In order to just
        retrieve a saved model, you should set the epochs to zero in the config file

    :param dataloader_train: the training data
    :return: The trained encoder and decoder model with frozen weights
    """
    save_dir = os.path.join(config("Directories")["save_model"], "cae")

    # Load the required parameters
    CAE = config("CAE")
    n_inputs = int(CAE["input_dim"])
    n_hidden = int(CAE["hidden_dim"])
    n_latent = int(CAE["latent_dimension"])
    iterations = int(CAE["epochs"])
    activation = str(CAE["activation"])
    learning_rate = float(CAE["learning_rate"])
    save_every_d_epochs = int(CAE["save_every_d_epochs"])
    ignore_first_epochs = int(CAE["ignore_first_epochs"])

    """create the whole cae"""
    encoder = Encoder(n_inputs, n_hidden, n_latent, activation)
    decoder = Decoder(n_inputs, n_hidden, n_latent, activation)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
    )

    # Load the CAE if available
    loading_path = checkpoint_path(save_dir)
    if loading_path is not None:
        logger.info(f"Loading the model...")
        saving_dictionary = load(loading_path)
        encoder.load_state_dict(saving_dictionary["encoder"])
        decoder.load_state_dict(saving_dictionary["decoder"])
        optimizer.load_state_dict(saving_dictionary["optimizer"])
        loss = saving_dictionary["loss"]
        start_epoch = saving_dictionary["epoch"] + 1
        step = saving_dictionary["step"]
        best_loss = saving_dictionary["best_loss"]
        logger.debug("Done")

    else:
        logger.info("No saved model, initializing...")
        start_epoch = 0
        loss = tensor([0])
        best_loss = inf
        step = 0
        logger.debug("Done")

    for epoch in range(start_epoch, start_epoch + iterations):
        logger.debug("Trining the CAE")
        losses = []
        for i, samples in enumerate(dataloader_train, 1):

            # change (batch, 100, n_inputs) to (100, n_inputs) if use NuSceneDataloader
            # samples = samples[0].view(-1, n_inputs)
            samples = samples.view(-1, n_inputs)
            samples = samples[[samples[j].sum() > 0 for j in range(samples.shape[0])]]

            samples = (samples - samples.mean()) / samples.std()

            # What are these for Mohammad??
            samples.requires_grad = True
            samples.retain_grad()

            outputs_encoder = encoder(samples)

            outputs = decoder(outputs_encoder)
            loss = cae_loss(outputs_encoder, outputs, samples)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            summary_writer.add_scalar("cae_loss", loss, step)

        logger.info(f"TRAINING [{(epoch + 1)} / {(start_epoch + iterations)}]\t loss: {mean(losses):.2f}")

        """
        The model will be saved in three circumstances:
          1. every d time steps specified on the config file
          2. when the loss was better in the previous iteration compared to best loss ever
          3. after the final iteration
        """

        if best_loss <= mean(losses) or \
                (epoch + 1) % save_every_d_epochs == 0 or \
                (epoch + 1) == start_epoch + iterations:
            logger.info("Saving the model....")
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "loss": loss,
            }
            if best_loss <= mean(losses) and epoch > ignore_first_epochs:
                best_loss = mean(losses)
                save(checkpoint, save_dir + "/best.pt")

            else:
                save(checkpoint, save_dir + "/checkpoint-" + str(epoch + 1) + ".pt")

    """
    Freeze the parameters on encoder and decoder
    > Note for mohammad, this does not freeze the waits necessarily, to do so you have to set requires_grad to false
    Check this plz: https://medium.com/jun-devpblog/pytorch-6-model-train-vs-model-eval-no-grad-hyperparameter-tuning-3812c216a3bd
    But later, when training the Whole module we should add the parameters with requires_grad to True to exclude the 
        parameters related to these submodeules (encoder and decoder)
    Check this plz: https://discuss.pytorch.org/t/correct-way-to-freeze-layers/26714
    """
    # encoder.eval()
    # decoder.eval()

    encoder.requires_grad_(False)
    decoder.requires_grad_(False)

    return encoder, decoder

if __name__ == "__main__":
    cae_config = config("CAE")
    root = config("Paths")["data_root"]
    data = CAEDataset(root)
    data = DataLoader(data, batch_size=int(cae_config["batch_size"]), shuffle=True, num_workers=2, drop_last=True)

    latents_list = list(range(1, 8  ))
    for i, latent in enumerate(latents_list):
        make_cae(data)
        plt.legend()
    plt.show()
