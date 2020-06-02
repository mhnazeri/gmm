# Ready packages
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
import logging
import os

# Custom defined packages
from data.loader import *
from losses import bce_loss, displacement_error, final_displacement_error
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from cae import make_cae
from numpy import inf, mean
from models import \
    TrajectoryGenerator, \
    TrajectoryDiscriminator


##########################################################################################
#                          Getting the required configuration
##########################################################################################
parser = argparse.ArgumentParser()

# Keep these
parser.add_argument("--use_gpu", default=1, type=int)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
args = parser.parse_args()

# Getting the configuration for training
GENERAL = config("System")
DIRECTORIES = config("Directories")
CAE = config("CAE")
TRAINING = config("Training")
GENERATOR = config("Generator")
DISCRIMINATOR = config("Discriminator")

summary_writer_validation = SummaryWriter(os.path.join(DIRECTORIES["log"], "validation_loss"))
summary_writer_generator = SummaryWriter(os.path.join(DIRECTORIES["log"], "generator"))
summary_writer_discriminator = SummaryWriter(os.path.join(DIRECTORIES["log"], "discriminator"))
summary_writer_cae = SummaryWriter(os.path.join(DIRECTORIES["log"], "cae"))

def get_cae():
    """
    Implemented to load the dataset and run the make cae method to either train or load the cae
    :return: The trained encoder and decoder with frozen weights
    """

    # loading the dataset
    root = DIRECTORIES["data_root"]
    cae_data = CAEDataset(root)
    data_loader = DataLoader(cae_data,
                      batch_size=int(CAE["batch_size"]),
                      num_workers=int(GENERAL["num_workers"]),
                      shuffle=True,
                      drop_last=True)

    cae_encoder, cae_decoder = make_cae(dataloader_train=data_loader,
                                        summary_writer=summary_writer_cae,
                                        save_dir=os.path.join(DIRECTORIES["save_model"], "cae"),
                                        encoder_structure=convert_str_to_list(CAE["encoder_structure"]),
                                        decoder_structure=convert_str_to_list(CAE["decoder_structure"]),
                                        dropout=float(CAE["dropout"]),
                                        bn=bool(CAE["batch_normalization"]),
                                        input_size=int(TRAINING["input_size"]),
                                        latent_dim=int(CAE["embedding_dim"]),
                                        iterations=int(CAE["epochs"]),
                                        activation=str(CAE["activation"]),
                                        learning_rate=float(CAE["learning_rate"]),
                                        save_every_d_epochs=int(CAE["save_every_d_epochs"]),
                                        ignore_first_epochs=int(CAE["ignore_first_epochs"]))

    return cae_encoder, cae_decoder


def main():
    cae_encoder, cae_decoder = get_cae()
    logger.info("Preparing the dataloader for the main model...")

    nuscenes_data = NuSceneDataset(root_dir=DIRECTORIES["train_data"])

    data_loader = DataLoader(nuscenes_data,
                             batch_size=int(TRAINING["batch_size"]),
                             shuffle=True,
                             collate_fn=default_collate)

    embedder = None
    if bool(GENERATOR["use_cae_encoder"]):
        logger.info("Using the CAE enocder...")
        embedder = cae_encoder

    logger.info("Constructing the GAN...")

    # Construct the models
    g = TrajectoryGenerator(
        embedder=embedder,
        embedding_dim=int(CAE["embedding_dim"]),
        encoder_h_dim=int(GENERATOR["encoder_h_dim"]),
        decoder_h_dim=int(GENERATOR["decoder_h_dim"]),
        seq_length=int(GENERATOR["seq_length"]),
        input_size=int(TRAINING["input_size"]),
        decoder_mlp_structure=convert_str_to_list(GENERATOR["decoder_h2p_structure"]),
        decoder_mlp_activation=str(GENERATOR["decoder_h2p_activation"]),
        dropout=float(GENERATOR["dropout"]),
        fusion_pool_dim=int(GENERATOR["fusion_pool_dim"]),
        fusion_hidden_dim=int(GENERATOR["fusion_h_dim"]),
        context_feature_model_arch=str(GENERATOR["context_feature_model_arch"]),
        num_layers=int(GENERATOR["num_layers"])
    )
    logger.debug("Here is the generator...")
    logger.debug(g)

    d = TrajectoryDiscriminator(
        embedder=cae_encoder,
        input_size=int(TRAINING["input_size"]),
        embedding_dim=int(CAE["embedding_dim"]),
        num_layers=int(TRAINING["num_layers"]),
        encoder_h_dim=int(DISCRIMINATOR["encoder_h_dim"]),
        mlp_structure=convert_str_to_list(DISCRIMINATOR["mlp_structure"]),
        mlp_activation=DISCRIMINATOR["mlp_activation"],
        batch_normalization=bool(DISCRIMINATOR["batch_normalization"]),
        dropout=float(DISCRIMINATOR["dropout"])
        )

    logger.debug("Here is the discriminator...")
    logger.debug(d)

    # Initialize the weights
    g.apply(init_weights)
    d.apply(init_weights)

    # Get the device type
    device = get_device(logger, args.use_gpu)

    # Transfer the models to gpu
    g.to(device)
    d.to(device)

    # Change the tensor types to GPU if neccessary
    tensor_type = get_tensor_type(args.use_gpu)
    g.type(tensor_type)
    d.type(tensor_type)

    # defining the loss and optimizers for generator and discriminator
    d_optimizer = torch.optim.Adam(d.parameters(), lr=float(DISCRIMINATOR["learning_rate"]))
    g_optimizer = torch.optim.Adam(g.parameters(), lr=float(GENERATOR["learning_rate"]))

    # Loading the checkpoint if existing
    save_dir = os.path.join(DIRECTORIES["save_model"], "main_model")
    loading_path = checkpoint_path(save_dir)
    if loading_path is not None:
        logger.info(f"Loading the main model...")
        loaded_dictionary = torch.load(loading_path)
        g.load_state_dict(loaded_dictionary["generator"])
        d.load_state_dict(loaded_dictionary["discriminator"])
        g_optimizer.load_state_dict(loaded_dictionary["g_optimizer"])
        d_optimizer.load_state_dict(loaded_dictionary["d_optimizer"])
        start_epoch = loaded_dictionary["epoch"] + 1
        step = loaded_dictionary["step"]
        best_ADE_loss = loaded_dictionary["best_ADE_loss"]

        logger.info(f"Done loading the model in {loading_path}")

    else:
        logger.info(f"No saved checkpoint for GAN, Initializing...")
        step = 0
        start_epoch = 0
        best_ADE_loss = inf

    logger.debug("Training the model")
    for epoch in range(start_epoch, start_epoch + int(TRAINING["epochs"])):
        g.train()
        d.train()
        g_losses = []
        d_losses = []
        for batch in data_loader:
            d_steps_left = int(GENERATOR["steps"])
            g_steps_left = int(DISCRIMINATOR["steps"])

            batch_size = batch["past"].shape[1]
            true_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            logger.debug(f"step {step} started!")

            # Transferring the input to the suitable device
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            g.zero_grad()
            d.zero_grad()

            while g_steps_left > 0:
                ###################################################################
                #                 training the generator
                ###################################################################
                logger.debug("Training the generator")

                fake_traj = g(batch["past"], batch["rel_past"], batch["motion"])
                fake_prediction = d(fake_traj)

                g_loss = bce_loss(fake_prediction, true_labels)
                g_losses.append(g_loss.item())

                summary_writer_generator.add_scalar("GAN_loss", g_loss, step)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                g_steps_left -= 1

            while d_steps_left > 0:
                ###################################################################
                #                 training the discriminator
                ###################################################################
                logger.debug("Training the discriminator")

                real_predictions = d(batch["rel_past"])
                real_loss = bce_loss(real_predictions, true_labels)

                fake_traj = g(batch["past"], batch["rel_past"], batch["motion"])
                fake_prediction = d(fake_traj)
                fake_loss = bce_loss(fake_prediction, fake_labels)

                d_loss = fake_loss + real_loss
                d_losses.append(d_loss.item())

                summary_writer_discriminator.add_scalar("GAN_loss", d_loss, step)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_steps_left -= 1

            logger.debug(f"step {step} finished!")
            step += 1

        ##########################################################################################
        #                            evaluating the trained model
        ##########################################################################################
        with torch.no_grad():
            logger.debug("Evaluating the model")
            g.eval()
            d.eval()

            fake_traj = g(batch["past"], batch["rel_past"], batch["motion"])
            ADE_loss = displacement_error(fake_traj, batch["future"])[0].item()
            FDE_loss = final_displacement_error(fake_traj[-1], batch["future"][-1])[0].item()

            # Todo: show some qualitative results of the predictions to be shown in tensorboard

        if int(TRAINING["epochs"]) > 0:
            epochs = int(TRAINING["epochs"])
            logger.info(
                f"TRAINING[{epoch + 1}/{start_epoch + epochs}]\t"
                f"d_loss:{mean(d_losses):.2f}\t\t"
                f"g_loss:{mean(g_losses):.2f}\t\t"
                f"ADE_loss:{ADE_loss:.2f}\t\t"
                f"FDE_loss:{FDE_loss:.2f}")

        summary_writer_validation.add_scalar("ADE_loss", ADE_loss, epoch)
        summary_writer_validation.add_scalar("FDE_loss", FDE_loss, epoch)

        ##########################################################################################
        #                                   Saving the model
        ##########################################################################################
        if (ADE_loss <= best_ADE_loss and epoch > int(TRAINING["ignore_first_epochs"])) or \
                (epoch + 1) % int(TRAINING["save_every_d_steps"]) == 0 or \
                (epoch + 1) == start_epoch + int(TRAINING["epochs"]):

            checkpoint = {
                "epoch": epoch,
                "step": step,
                "generator": g.state_dict(),
                "discriminator": d.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "best_ADE_loss": best_ADE_loss,
            }
            if ADE_loss <= best_ADE_loss and epoch > int(TRAINING["ignore_first_epochs"]):
                logger.info("Saving the model(lowest ADE loss)...")
                best_ADE_loss = ADE_loss
                torch.save(checkpoint, save_dir + "/best.pt")

            else:
                logger.info(f"Saving the model(intervals)...")
                torch.save(checkpoint, save_dir + "/checkpoint-" + str(epoch + 1) + ".pt")

if __name__ == '__main__':
    main()
