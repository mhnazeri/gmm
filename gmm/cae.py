"""
Implementation of Contractive Autoencoder to compress feature vector into
latent feature vector
"""
import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.loader import CAEDataset
from utils import *
from losses import cae_loss

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Encoder_Decoder(nn.Module):
    """Encoder network of CAE"""

    def __init__(
        self,
        Encoder: bool = True,
        input_size: int = 13,
        output_size: int = 7,
        structure: list = [128],
        latent_dimension: int = 7,
        dropout: float = 0.0,
        batch_normalization: bool = False,
        activation="Sigmoid",
    ):

        super(Encoder_Decoder, self).__init__()

        if (
            activation == "Sigmoid"
            or activation == "Tanh"
            or activation == "LeakyRelu"
            or activation == "Relu"
        ):
            pass

        else:
            raise "{} function is not supported".format(activation)

        if Encoder:
            structure.insert(0, input_size)

        else:
            structure.insert(0, latent_dimension)

        nn_layers = []
        for dim_in, dim_out in zip(structure[:-1], structure[1:]):
            nn_layers.append(nn.Linear(dim_in, dim_out))
            if activation == "Relu":
                nn_layers.append(nn.ReLU())
            elif activation == "LeakyRelu":
                nn_layers.append(nn.LeakyReLU())
            elif activation == "Sigmoid":
                nn_layers.append(nn.Sigmoid())
            elif activation == "Tanh":
                nn_layers.append(nn.Tanh())
            elif activation == "ELU":
                nn_layers.append(nn.ELU())
            if batch_normalization and dim_out != structure[-1]:
                nn_layers.append(nn.BatchNorm1d(dim_out))
            if dropout > 0:
                nn_layers.append(nn.Dropout(p=dropout))

        if Encoder:
            nn_layers.append(nn.Linear(structure[-1], latent_dimension))

        else:
            nn_layers.append(nn.Linear(structure[-1], output_size))

        self.mlp = nn.Sequential(*nn_layers)

    def forward(self, x):
        return self.mlp(x)


def make_cae(
    dataloader_train,
    summary_writer,
    save_dir: str = "save",
    encoder_structure: list = [128, 128],
    decoder_structure: list = [128, 128],
    dropout: float = 0.0,
    bn: bool = False,
    input_size: int = 13,
    output_size: int = 7,
    latent_dim: int = 16,
    iterations: int = 500,
    activation: str = "Relu",
    learning_rate: float = 0.001,
    save_every_d_epochs: int = 50,
    ignore_first_epochs: int = 300,
):
    """
    The following function returns the required cae to be further used in the next sections of the model
    If there is any saved model of cae, It loads according to the loading strategy otherwise it just
        creates a model with the specified parameters and trains it from scratch.
    The model will train even though there is any saved model until there are iterations specified, In order to just
        retrieve a saved model, you should set the epochs to zero in the config file

    :param dataloader_train: the training data iterator on Tensors with shape (batch_size, 520)
    :return: The trained encoder and decoder model with frozen weights
    """

    # create the whole cae
    encoder = Encoder_Decoder(
        Encoder=True,
        input_size=input_size,
        structure=encoder_structure,
        latent_dimension=latent_dim,
        dropout=dropout,
        activation=activation,
        batch_normalization=bn,
    )

    decoder = Encoder_Decoder(
        Encoder=False,
        input_size=input_size,
        output_size=output_size,
        structure=decoder_structure,
        latent_dimension=latent_dim,
        dropout=dropout,
        activation=activation,
        batch_normalization=bn,
    )

    # Get the suitable device to run the model on
    device = get_device(logger)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    tensor_type = get_tensor_type()
    encoder.type(tensor_type).train()
    decoder.type(tensor_type).train()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate
    )

    logger.debug("Here is the encoder...")
    logger.debug(encoder)

    logger.debug("Here is the decoder...")
    logger.debug(decoder)

    # Load the CAE if available
    loading_path = torch.load(save_dir)
    if loading_path is not None:
        logger.info(f"Loading the model in {loading_path}...")
        saving_dictionary = loading_path
        encoder.load_state_dict(saving_dictionary["encoder"])
        decoder.load_state_dict(saving_dictionary["decoder"])
        optimizer.load_state_dict(saving_dictionary["optimizer"])
        loss = saving_dictionary["loss"]
        start_epoch = saving_dictionary["epoch"] + 1
        step = saving_dictionary["step"]
        best_loss = saving_dictionary["best_loss"]
        logger.debug("Done")

    else:
        logger.info("No saved model for CAE, initializing...")
        start_epoch = 0
        loss = torch.tensor([0])
        best_loss = np.inf
        step = 0
        logger.debug("Done")
        encoder.apply(init_weights)
        decoder.apply(init_weights)

    for epoch in range(start_epoch, start_epoch + iterations):
        logger.debug("Training the CAE")
        losses = []
        for i, samples in enumerate(dataloader_train, 1):

            samples = samples.to(device)
            encoder.zero_grad()
            decoder.zero_grad()

            samples.requires_grad_(True)
            samples.retain_grad()

            outputs_encoder = encoder(samples)

            outputs = decoder(outputs_encoder)

            loss = cae_loss(outputs_encoder, outputs, samples, device)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1

            if torch.isnan(loss):
                print("outputs_encoder", outputs_encoder)
                print("outputs", outputs)
                print("samples", samples)
                break

        summary_writer.add_scalar("cae_loss", np.mean(losses), epoch)
        logger.info(
            f"TRAINING [{(epoch + 1):3d} / {(start_epoch + iterations)}]\t loss: {np.mean(losses):.2f}"
        )

        """
        The model will be saved in three circumstances: 
          1. every d time steps specified on the config file
          2. when the loss was better in the previous iteration compared to best loss ever
          3. after the final iteration
        """

        if (
            (best_loss <= np.mean(losses) and epoch > ignore_first_epochs)
            or (epoch + 1) % save_every_d_epochs == 0
            or (epoch + 1) == start_epoch + iterations
        ):
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "loss": loss,
            }
            if best_loss <= np.mean(losses) and epoch > ignore_first_epochs:
                logger.info("Saving the model (lowest loss)...")
                best_loss = np.mean(losses)
                torch.save(checkpoint, save_dir + "/best.pt")

            else:
                logger.info(f"Saving the model (intervals)...")
                torch.save(
                    checkpoint, save_dir + "/checkpoint-" + str(epoch + 1) + ".pt"
                )

    encoder.requires_grad_(False)
    decoder.requires_grad_(False)

    return encoder, decoder


if __name__ == "__main__":
    # This section is for experimenting about the best latent dimension
    DIRECTORIES = config("Directories")
    CAE = config("CAE")
    GENERAL = config("System")
    TRAINING = config("Training")

    root = DIRECTORIES["data_root"]
    cae_data = CAEDataset(root)
    data_loader = DataLoader(
        cae_data,
        batch_size=int(CAE["batch_size"]),
        num_workers=int(GENERAL["num_workers"]),
        shuffle=True,
        drop_last=True,
    )

    # Change this for experimenting other latent dims
    latent_dim_list = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]

    for latent_dim in latent_dim_list:
        summary_writer_cae = SummaryWriter(
            os.path.join(DIRECTORIES["log"], "cae_" + str(latent_dim))
        )
        temp = os.path.join(DIRECTORIES["save_model"], "cae/best.pt")
        save_dir = os.path.join(temp, str(latent_dim))

        make_cae(
            dataloader_train=data_loader,
            summary_writer=summary_writer_cae,
            save_dir=save_dir,
            encoder_structure=convert_str_to_list(CAE["encoder_structure"]),
            decoder_structure=convert_str_to_list(CAE["decoder_structure"]),
            dropout=float(CAE["dropout"]),
            bn=bool(CAE["batch_normalization"]),
            input_size=int(TRAINING["input_size"]),
            output_size=int(CAE["output_size"]),
            latent_dim=latent_dim,
            iterations=int(CAE["epochs"]),
            activation=str(CAE["activation"]),
            learning_rate=float(CAE["learning_rate"]),
            save_every_d_epochs=int(CAE["save_every_d_epochs"]),
            ignore_first_epochs=int(CAE["ignore_first_epochs"]),
        )
