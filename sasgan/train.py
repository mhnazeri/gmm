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
from losses import bce_loss
from utils import *
from torch.utils.data import DataLoader
from cae import make_cae
from models import \
    TrajectoryGenerator, \
    TrajectoryDiscriminator


##########################################################################################
#                          Getting the required configuration
##########################################################################################
parser = argparse.ArgumentParser()

# Keep these
parser.add_argument("--use_gpu", default=False, type=bool)

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

summary_writer_general = SummaryWriter(DIRECTORIES["log"])
summary_writer_generator = SummaryWriter(os.path.join(DIRECTORIES["log"], "generator"))
summary_writer_discriminator = SummaryWriter(os.path.join(DIRECTORIES["log"], "discriminator"))
summary_writer_cae = SummaryWriter(os.path.join(DIRECTORIES["log"], "cae"))

##########################################################################################
#                          Some useful functions
##########################################################################################

def init_weights(m):
    if m.__class__.__name__ == "Linear":
        nn.init.kaiming_uniform(m.weight)


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

<<<<<<< HEAD
<<<<<<< HEAD
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
||||||| parent of a9d089e... models finished
    cae_encoder, cae_decoder = make_cae(data_loader, summary_writer_cae)
=======
    cae_encoder, cae_decoder = make_cae(dataloader_train=data_loader,
                                        summary_writer=summary_writer_cae,
                                        save_dir=os.path.join(DIRECTORIES["save_model"], "cae"),
                                        encoder_structure=convert_str_to_list(CAE["encoder_structure"]),
                                        decoder_structure=convert_str_to_list(CAE["decoder_structure"]),
                                        dropout=float(CAE["dropout"]),
                                        bn=bool(CAE["batch_normalization"]),
                                        input_size=int(CAE["input_size"]),
                                        latent_dim=int(CAE["latent_dimension"]),
                                        iterations=int(CAE["epochs"]),
                                        activation=str(CAE["activation"]),
                                        learning_rate=float(CAE["learning_rate"]),
                                        save_every_d_epochs=int(CAE["save_every_d_epochs"]),
                                        ignore_first_epochs=int(CAE["ignore_first_epochs"]))

>>>>>>> a9d089e... models finished
||||||| 3dd7b53
    cae_encoder, cae_decoder = make_cae(data_loader, summary_writer_cae)
=======
    cae_encoder, cae_decoder = make_cae(dataloader_train=data_loader,
                                        summary_writer=summary_writer_cae,
                                        save_dir=os.path.join(DIRECTORIES["save_model"], "cae"),
                                        encoder_structure=convert_str_to_list(CAE["encoder_structure"]),
                                        decoder_structure=convert_str_to_list(CAE["decoder_structure"]),
                                        dropout=float(CAE["dropout"]),
                                        bn=bool(CAE["batch_normalization"]),
                                        input_size=int(CAE["input_size"]),
                                        latent_dim=int(CAE["latent_dimension"]),
                                        iterations=int(CAE["epochs"]),
                                        activation=str(CAE["activation"]),
                                        learning_rate=float(CAE["learning_rate"]),
                                        save_every_d_epochs=int(CAE["save_every_d_epochs"]),
                                        ignore_first_epochs=int(CAE["ignore_first_epochs"]))

>>>>>>> origin/lazy-dataloader

    logger.info("Done training/loading the CAE!")
    return cae_encoder, cae_decoder


def main():
    cae_encoder, cae_decoder = get_cae()

    # By now the Encoder and the Decoder have been loaded or trained

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    logger.info("Preparing the dataloader for the main model...")

    nuscenes_data = NuSceneDataset(root_dir=DIRECTORIES["nuscenes_json"],
                                   max_agent=int(TRAINING["max_agents"]),
                                   transform=image_transform)

    data_loader = DataLoader(nuscenes_data,
                             batch_size=int(TRAINING["batch_size"]),
                             shuffle=True)

<<<<<<< HEAD
<<<<<<< HEAD
    embedder = None
    if bool(GENERATOR["use_cae_encoder"]):
        embedder = cae_encoder

    logger.info("building the GAN...")
||||||| parent of a9d089e... models finished
=======



"""
>>>>>>> a9d089e... models finished
||||||| 3dd7b53
=======



"""
>>>>>>> origin/lazy-dataloader
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
    logger.info("Here is the generator")
    logger.info(g)

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

    logger.info("Here is the discriminator")
    logger.info(d)

    # Initialize the weights
    g.apply(init_weights)
    d.apply(init_weights)

    # Transfer the tensors to GPU if required
    tensor_type = get_tensor_type(args)
    g.type(tensor_type).train()
    d.type(tensor_type).train()

    # defining the loss and optimizers for generator and discriminator
    d_optimizer = torch.optim.Adam(d.parameters(), lr=float(DISCRIMINATOR["learning_rate"]))
    g_optimizer = torch.optim.Adam(g.parameters(), lr=float(GENERATOR["learning_rate"]))

    # Loading the checkpoint if existing

<<<<<<< HEAD
<<<<<<< HEAD
    save_dir = os.path.join(DIRECTORIES["save_model"], "main_model")
    loading_path = checkpoint_path(save_dir)
||||||| parent of a9d089e... models finished
||||||| 3dd7b53
=======



>>>>>>> origin/lazy-dataloader
    loading_path = checkpoint_path(DIRECTORIES["save_model"])
=======



    loading_path = checkpoint_path(DIRECTORIES["save_model"])
>>>>>>> a9d089e... models finished
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
        g_loss = loaded_dictionary["current_g_loss"]
        d_loss = loaded_dictionary["current_d_loss"]
        logger.info(f"Done loading the model in {loading_path}")

    else:
        logger.info(f"No saved checkpoint, Initializing...")
        step = 0
        start_epoch = 0
        best_ADE_loss = np.inf
        d_loss = np.inf
        g_loss = np.inf

    logger.debug("Training the model")
    for epoch in range(start_epoch, start_epoch + int(TRAINING["epochs"])):
        g.train()
        d.train()
        for batch in data_loader:
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps

            true_labels = torch.ones(batch.shape[0], 1)
            fake_labels = torch.zeros(batch.shape[0], 1)
            while g_steps_left > 0:
                ###################################################################
                #                 training the generator
                ###################################################################
                logger.debug("Training the generator")
                fake_traj = g(batch["past"], batch["past_rel"], batch["motion"])
                fake_prediction = d(fake_traj)

                g_loss = bce_loss(fake_prediction, true_labels)

                summary_writer_generator.add_scalar("Loss", g_loss, step)

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

                summary_writer_discriminator.add_scalar("Loss", d_loss, step)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_steps_left -= 1

            step += 1

        if args.iterations > 0:
            logger.info(
                f"TRAINING[{epoch + 1}/{start_epoch + args.iterations}]\td_loss:{d_loss:.2f}\tg_loss:{g_loss:.2f}")

            # Todo: show some qualitative results of the predictions

        with torch.no_grad():
            logger.debug("Evaluating the model")
            g.eval()
            d.eval()

            ADE_loss = 0

            # Todo: Calculate the FDE and ADE in this section for validation

            validation_loss = ADE_loss.item()

        # check if it is time to save a checkpoint of the model
        if validation_loss <= best_ADE_loss or \
                (epoch + 1) % int(TRAINING["save_every_d_steps"]) == 0 or \
                (epoch + 1) == start_epoch + int(TRAINING["epochs"]):
            logger.info("Saving the model....")
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "generator": g.state_dict(),
                "discriminator": d.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "best_ADE_loss": best_ADE_loss,
                "current_d_loss": d_loss,
                "current_g_loss": g_loss,
            }
            if validation_loss <= best_ADE_loss and epoch > args.ignore_first_iterations:
                best_ADE_loss = validation_loss
                torch.save(checkpoint, save_dir + "/best.pt")

            if (epoch + 1) % args.save_every_d_epochs == 0 or (epoch + 1) == start_epoch + args.iterations:
                torch.save(checkpoint, save_dir + "/checkpoint-" + str(epoch + 1) + ".pt")

if __name__ == '__main__':
    main()
