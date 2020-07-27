"""
Implementation of Contractive Autoencoder to compress feature vector into
latent feature vector
"""
import os
import logging
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.loader import CFEXDataset
from utils import *
from losses import cae_loss

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Encoder(nn.Module):
    """Encoder network of CFEX"""

    def __init__(self):  # , model_arch: str="overfeat", pretrained: bool=True, refine: bool=True):
        """:return tensor (batch_size, 512, 12, 12)"""
        super(Encoder, self).__init__()
        # if model_arch == "overfeat":
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels=4, kernel_size=11, stride=4,
                                             out_channels=96)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=96, out_channels=256,
                                             kernel_size=5, stride=1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=256, out_channels=512,
                                             kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=512,
                                             kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Conv2d(in_channels=512, out_channels=512,
                                             kernel_size=3, stride=1, padding=1)),
        )

        # elif model_arch == "vgg":
        #     vgg = torchvision.models.vgg11(pretrained=pretrained).features
        #     for i, layer in enumerate(vgg.children()):
        #         if isinstance(layer, nn.MaxPool2d):
        #             vgg[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0,
        #                                   ceil_mode=False)
        #         elif isinstance(layer, nn.ReLU):
        #             vgg[i] = nn.LeakyReLU(inplace=True)
        #
        #     if refine:
        #         for param in vgg.parameters():
        #             param.requires_grad = True
        #     else:
        #         for param in vgg.parameters():
        #             param.requires_grad = False
        #
        #     self.net = vgg
        #
        # else:
        #     raise ValueError(f"Unrecognized model architecture {model_arch}")

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Encoder network of CFEX"""

    def __init__(self):
        super(Decoder, self).__init__()
        # if model_arch == "overfeat":
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=512, kernel_size=3, stride=2,
                                                      out_channels=512)),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                                      kernel_size=3, stride=2)),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                                      kernel_size=3, stride=1, padding=1)),
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=96,
                                                      kernel_size=3, stride=1, padding=1)),
            # nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=96, out_channels=96,
                                                      kernel_size=11, stride=2)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels=96, out_channels=4,
                                                      kernel_size=11, stride=2)),
        )

    def forward(self, x):
        return self.net(x)


def make_cfex(
        dataloader_train,
        summary_writer,
        save_dir: str = "save",
        iterations: int = 500,
        learning_rate: float = 0.001,
        save_every_d_epochs: int = 50,
        ignore_first_epochs: int = 300
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
    encoder = Encoder()

    decoder = Decoder()

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

    # Load the CFEX if available
    loading_path = checkpoint_path(save_dir)
    if loading_path is not None:
        logger.info(f"Loading the model in {loading_path}...")
        saving_dictionary = torch.load(loading_path)
        encoder.load_state_dict(saving_dictionary["encoder"])
        decoder.load_state_dict(saving_dictionary["decoder"])
        optimizer.load_state_dict(saving_dictionary["optimizer"])
        loss = saving_dictionary["loss"]
        start_epoch = saving_dictionary["epoch"] + 1
        step = saving_dictionary["step"]
        best_loss = saving_dictionary["best_loss"]
        logger.debug("Done")

    else:
        logger.info("No saved model for CFEX, initializing...")
        start_epoch = 0
        loss = torch.tensor([0])
        best_loss = np.inf
        step = 0
        logger.debug("Done")
        encoder.apply(init_weights)
        decoder.apply(init_weights)

    for epoch in range(start_epoch, start_epoch + iterations):
        logger.debug("Training the CFEX")
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
            logger.debug(f"Encoder Grad Norm: {check_grad_norm(encoder)}")
            logger.debug(f"Decoder Grad Norm: {check_grad_norm(decoder)}")
            optimizer.step()

            step += 1

            if torch.isnan(loss):
                print("outputs_encoder", outputs_encoder)
                print("outputs", outputs)
                print("samples", samples)
                break

        summary_writer.add_scalar("cfex_loss", np.mean(losses), epoch)
        logger.info(f"TRAINING [{(epoch + 1):3d} / {(start_epoch + iterations)}]\t loss: {np.mean(losses):.2f}")

        """
        The model will be saved in three circumstances: 
          1. every d time steps specified on the config file
          2. when the loss was better in the previous iteration compared to best loss ever
          3. after the final iteration
        """

        if (best_loss <= np.mean(losses) and epoch > ignore_first_epochs) or \
                (epoch + 1) % save_every_d_epochs == 0 or \
                (epoch + 1) == start_epoch + iterations:
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
                torch.save(checkpoint, save_dir + "/checkpoint-" + str(epoch + 1) + ".pt")

    encoder.requires_grad_(False)
    decoder.requires_grad_(False)

    return encoder, decoder


if __name__ == "__main__":
    # This section is for experimenting about the best latent dimension
    DIRECTORIES = config("Directories")
    CFEX = config("CFEX")
    GENERAL = config("System")
    TRAINING = config("Training")

    root = DIRECTORIES["data_root"]
    cfex_data = CFEXDataset(root)
    data_loader = DataLoader(cfex_data,
                             batch_size=int(CFEX["batch_size"]),
                             num_workers=int(GENERAL["num_workers"]),
                             shuffle=True,
                             drop_last=True)

    # for latent_dim in latent_dim_list:
    summary_writer_cfex = SummaryWriter(os.path.join(DIRECTORIES["log"], "cfex"))
    save_dir = os.path.join(DIRECTORIES["save_model"], "cfex_train")
    # save_dir = os.path.join(temp, str(latent_dim))
    make_cfex(dataloader_train=data_loader,
             summary_writer=summary_writer_cfex,
             save_dir=save_dir,
             iterations=int(CFEX["epochs"]),
             learning_rate=float(CFEX["learning_rate"]),
             save_every_d_epochs=int(CFEX["save_every_d_epochs"]),
             ignore_first_epochs=int(CFEX["ignore_first_epochs"]))
