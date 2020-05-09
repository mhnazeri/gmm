# Ready packages
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import numpy as np
from torchvision.transforms import ToTensor
import logging
import os

# Custom defined packages
from data.loader import *
from utils import *
from torch.utils.data import DataLoader
from cae import make_cae
from models import Generator, TrajectoryDiscriminator, Encoder


##########################################################################################
#                          Getting the required configuration
##########################################################################################
# Getting the configuration for training
GENERAL = config("General")
DIRECTORIES = config("Directories")
CAE = config("CAE")
ENCODER = config("Encoder")
POOLING = config("Pooling")
TRAINING = config("Training")
GENERATOR = config("Generator")
DISCRIMINATOR = config("Discriminator")

parser = argparse.ArgumentParser()

# Keep these
parser.add_argument("--use_gpu", default=False, type=bool)



# General parameters
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--image_size", default=28 * 28, type=int)
parser.add_argument("--noise_size", default=100, type=int)
parser.add_argument("--iterations", default=10, type=int)

parser.add_argument("--models_dir", default="./save", type=str)
parser.add_argument("--loading_strategy", default="last", type=str)
parser.add_argument("--save_every_d_epochs", default=3, type=int)
parser.add_argument("--ignore_first_iterations", default=15, type=int)

# Generator parameters
parser.add_argument("--generator_structure", default=[100, 256, 256, 28 * 28], type=list)
parser.add_argument("--generator_activation", default="Relu", type=str)
parser.add_argument("--generator_dropout", default=0.0, type=float)
parser.add_argument("--g_steps", default=1, type=int)
parser.add_argument("--g_lr", default=0.002, type=float)

# Discriminator parameters
parser.add_argument("--disc_structure", default=[28 * 28, 256, 128], type=list)
parser.add_argument("--disc_activation", default="Relu", type=str)
parser.add_argument("--disc_dropout", default=0.0, type=float)
parser.add_argument("--d_steps", default=1, type=int)
parser.add_argument("--d_lr", default=0.0002, type=float)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    cae_encoder, cae_decoder = make_cae(data_loader, summary_writer_cae)

    logger.debug("Done training/loading the CAE")
    return cae_encoder, cae_decoder


def main(args):
    cae_encoder, cae_decoder = get_cae()

    ##########################################################################################
    #                          Training the rest of the model
    ##########################################################################################
    # creating the dataset and dataloader
    nuscenes_data = NuSceneDataset(root_dir=DIRECTORIES["nuscenes_json"],
                                   max_agent=int(GENERAL["max_agents"]))

    data_loader = DataLoader(nuscenes_data,
                             batch_size=int(GENERAL["batch_size"]),
                             shuffle=True)

    epoch = 0
    while epoch < int(TRAINING["num_epochs"]):
        for i, batch in enumerate(data_loader):
            rel_past = batch["rel_past"]
            past = batch[""]
            latent_trajectories = cae_encoder(batch)

        epoch += 1

    train_loader = DataLoader(nuscenes_data,
                              batch_size=args.batch_size,
                              shuffle=True)

"""
    # Construct the models
    logger.info("Here is the generator")
    g = Generator(args)
    logger.info(g)

    logger.info("Here is the discriminator")
    d = Discriminator(args)
    logger.info(d)

    # Initilize the weights
    g.apply(init_weights)
    d.apply(init_weights)

    # Transfer the tensors to GPU if required
    tensor_type = get_tensor_type(args)
    g.type(tensor_type).train()
    d.type(tensor_type).train()

    # defining the loss and optimizers for generator and discriminator
    d_optimizer = torch.optim.Adam(d.parameters(), lr=args.d_lr)
    g_optimizer = torch.optim.Adam(g.parameters(), lr=args.g_lr)

    # Loading the checkpoint if existing
    #   the loading strategy is based on the best accuracy and after every iteration interval

    loading_path = checkpoint_path(args)
    if loading_path is not None:
        logger.info(f"Loading the model in {loading_path}...")
        loaded_dictionary = torch.load(loading_path)
        g.load_state_dict(loaded_dictionary["generator"])
        d.load_state_dict(loaded_dictionary["discriminator"])
        g_optimizer.load_state_dict(loaded_dictionary["g_optimizer"])
        d_optimizer.load_state_dict(loaded_dictionary["d_optimizer"])
        start_epoch = loaded_dictionary["epoch"] + 1
        step = loaded_dictionary["step"]
        best_validation_loss = loaded_dictionary["best_validation_loss"]
        total_validation_loss = loaded_dictionary["total_validation_loss"]
        g_loss = loaded_dictionary["current_g_loss"]
        d_loss = loaded_dictionary["current_d_loss"]
        logger.debug(f"Done loading the model in {loading_path}")

    else:
        logger.info(f"No saved checkpoint, Initializing...")
        step = 0
        start_epoch = 0
        best_validation_loss = np.inf
        total_validation_loss = 0
        d_loss = np.inf
        g_loss = np.inf

    logger.debug("Training the model")
    for i in range(start_epoch, start_epoch + args.iterations):
        g.train()
        d.train()
        for x_train, _ in train_loader:
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            true_labels = torch.ones(x_train.shape[0], 1)
            fake_labels = torch.zeros(x_train.shape[0], 1)

            while d_steps_left > 0:
                ###################################################################
                #                 training the discriminator
                ###################################################################
                logger.debug("Training the discriminator")

                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))

                x_train = x_train.reshape(-1, args.image_size)
                real_predictions = d(x_train)
                real_loss = bce_loss(real_predictions, true_labels)

                fake_images = g(noise_tensor)
                fake_prediction = d(fake_images)
                fake_loss = bce_loss(fake_prediction, fake_labels)

                d_loss = fake_loss + real_loss

                summary_writer_discriminator.add_scalar("Loss", d_loss, i)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()
                d_steps_left -= 1

            while g_steps_left > 0:
                ###################################################################
                #                 training the generator
                ###################################################################
                logger.debug("Training the generator")
                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))
                fake_images = g(noise_tensor)
                fake_prediction = d(fake_images)

                g_loss = bce_loss(fake_prediction, true_labels)

                summary_writer_generator.add_scalar("Loss", g_loss, i)

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                g_steps_left -= 1

            step += 1

        if args.iterations > 0:
            logger.info(
                f"TRAINING[{i + 1}/{start_epoch + args.iterations}]\td_loss:{d_loss:.2f}\tg_loss:{g_loss:.2f}")
            summary_writer_general.add_images(f"samples_images_iteration_{i}",
                                              fake_images[np.random.randint(0, x_train.shape[0], 5)].squeeze(1).
                                              reshape([-1, 1, int(args.image_size ** 0.5),
                                                       int(args.image_size ** 0.5)]))

        with torch.no_grad():
            logger.debug("Evaluating the model")
            g.eval()
            d.eval()

            for x_test, _ in test_loader:
                # Discriminator part
                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))
                fake_labels = torch.zeros(x_train.shape[0])
                true_labels = torch.ones(x_train.shape[0])

                generated_images = g(noise_tensor)
                fake_scores = d(generated_images)
                fake_loss = bce_loss(fake_scores, fake_labels)

                x_test = x_test.reshape(-1, args.image_size)
                true_scores = d(x_test)
                true_loss = bce_loss(true_scores, true_labels)

                validation_d_loss = true_loss + fake_loss

                # Generator part
                noise_tensor = torch.normal(0.0, 1.0, (x_train.shape[0], args.noise_size))
                generated_images = g(noise_tensor)
                fake_scores = d(generated_images)

                validation_g_loss = bce_loss(fake_scores, true_labels)

                # Combine the losses
                total_validation_loss = validation_d_loss + validation_g_loss

            logger.info(f"VALIDATING\ttotal evaluation loss:"
                        f"{total_validation_loss:.2f}\tg_loss:{validation_g_loss:.2f}\td_loss:{validation_d_loss:.2f}")
            summary_writer_general.add_scalar("total_evaluation_loss", total_validation_loss, i)

        total_validation_loss = total_validation_loss.item()
        # check if it is time to save a checkpoint of the model
        if total_validation_loss <= best_validation_loss or \
                (i + 1) % args.save_every_d_epochs == 0 or \
                (i + 1) == start_epoch + args.iterations:
            logger.info("Saving the model....")
            checkpoint = {
                "epoch": i,
                "step": step,
                "generator": g.state_dict(),
                "discriminator": d.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "best_validation_loss": best_validation_loss,
                "current_d_loss": d_loss,
                "current_g_loss": g_loss,
                "total_validation_loss": total_validation_loss
            }
            if total_validation_loss <= best_validation_loss and i > args.ignore_first_iterations:
                best_validation_loss = total_validation_loss
                torch.save(checkpoint, args.model_dir + "/best.pt")

            if (i + 1) % args.save_every_d_epochs == 0 or (i + 1) == start_epoch + args.iterations:
                torch.save(checkpoint, args.model_dir + "/checkpoint-" + str(i + 1) + ".pt")

"""

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)








# May not be used

    #
    #
    #
    #

