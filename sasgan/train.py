import os
import logging
import random
import sys
import time
import argparse

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from data.loader import *
from losses import bce_loss, displacement_error, final_displacement_error, msd_error
from utils import *
from cae import make_cae
from models import \
    TrajectoryGenerator, \
    TrajectoryDiscriminator

##########################################################################################
#                          Getting the required configuration
##########################################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                                        input_size=int(CAE["input_size"]),
                                        output_size=int(CAE["output_size"]),
                                        latent_dim=int(CAE["embedding_dim"]),
                                        iterations=int(CAE["epochs"]),
                                        activation=str(CAE["activation"]),
                                        learning_rate=float(CAE["learning_rate"]),
                                        save_every_d_epochs=int(CAE["save_every_d_epochs"]),
                                        ignore_first_epochs=int(CAE["ignore_first_epochs"]))

    return cae_encoder, cae_decoder


def main():
    cae_encoder, cae_decoder = get_cae()
    max_agents = int(TRAINING["max_agents"])
    logger.info("Preparing the dataloader for the main model...")

    nuscenes_data = NuSceneDataset(root_dir=DIRECTORIES["train_data"])

    data_loader = DataLoader(nuscenes_data,
                             batch_size=int(TRAINING["batch_size"]),
                             shuffle=True,
                             collate_fn=default_collate)

    embedder = None
    if bool(GENERATOR["use_cae_encoder"]):
        logger.info("Using the CAE encoder...")
        if int(GENERATOR["refine_CAE"]) > 0:
            embedder = cae_encoder.requires_grad_(True)
        else:
            embedder = cae_encoder

    logger.info("Constructing the GAN...")

    # Construct the models
    g = TrajectoryGenerator(
        embedder=embedder,
        de_embedder=None,
        embedding_dim=int(CAE["embedding_dim"]),
        encoder_h_dim=int(GENERATOR["encoder_h_dim"]),
        decoder_h_dim=int(GENERATOR["decoder_h_dim"]),
        seq_length=int(GENERATOR["seq_length"]),
        input_size=int(TRAINING["input_size"]),
        output_size=int(GENERATOR["output_size"]),
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
        activation=DISCRIMINATOR["mlp_activation"],
        batch_normalization=bool(DISCRIMINATOR["batch_normalization"]),
        dropout=float(DISCRIMINATOR["dropout"])
        )
    logger.debug("Here is the discriminator...")
    logger.debug(d)

    # Initialize the weights
    # g.apply(init_weights)
    d.apply(init_weights)

    # Get the device type
    device = get_device(logger)

    clipping_threshold_d = int(TRAINING["gradient_clipping_d"])
    clipping_threshold_g = int(TRAINING["gradient_clipping_g"])

    # Transfer the models to gpu
    g.to(device)
    d.to(device)

    # Change the tensor types to GPU if neccessary
    # tensor_type = get_tensor_type()
    # g.type(tensor_type)
    # d.type(tensor_type)

    # defining the loss and optimizers for generator and discriminator
    d_optimizer = torch.optim.Adam(d.parameters(), lr=float(DISCRIMINATOR["learning_rate"]), betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(g.parameters(), lr=float(GENERATOR["learning_rate"]), betas=(0.5, 0.999))

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
        best_ADE_loss = np.inf

    logger.debug("Training the model")
    for epoch in range(start_epoch, start_epoch + int(TRAINING["epochs"])):
        g.train()
        d.train()
        g_losses = []
        d_losses = []
        for batch in data_loader:
            d_steps_left = int(DISCRIMINATOR["steps"])

            logger.debug(f"step {step} started!")
            # Transferring the input to the suitable device
            for key in batch.keys():
                batch[key] = batch[key].to(device)

            fake_labels = torch.ones(batch["past"].shape[1]).to(device) * random.uniform(0, 0.2)
            true_labels = torch.ones(batch["past"].shape[1]).to(device) * random.uniform(0.8, 1.0)
            ###################################################################
            #                 training the discriminator                       
            ###################################################################
            while d_steps_left > 0:
                # d.zero_grad()                                                    
                logger.debug("Training the discriminator")                         
                d_optimizer.zero_grad()

                # d_input_past = batch["past"].permute(1, 0, 2).contiguous().view(batch["past"].size()[1], -1)
                # d_input_future = batch["future"].permute(1, 0, 2).contiguous().view(batch["future"].size()[1], -1)
                # distort data to make d more robust
                # traj_gt_past_rel = batch["rel_past"] * torch.randn_like(batch["rel_past"])
                # traj_gt_future_rel = batch["rel_future"] * torch.randn_like(batch["rel_future"])
                # traj_gt = torch.cat([d_input_past, d_input_future], dim=1)     
                # real_predictions = d(batch["past"])
                # add some noise to real data to confuse discriminator
                # traj_gt = traj_gt * torch.randn_like(traj_gt)
                real_predictions = d(torch.cat([batch["rel_past"][:, :, :2], batch["rel_future"]], dim=0))
                real_loss = bce_loss(real_predictions, true_labels)
                # real_loss.backward()
                                                                                    
                fake_traj = g(batch["past"], batch["rel_past"], batch["motion"])
                # d_input_past = batch["past"].permute(1, 0, 2).contiguous().view(batch["past"].size()[1], -1)
                # d_input_future = fake_traj.detach().permute(1, 0, 2).contiguous().view(batch["future"].size()[1], -1)
                # traj_fake = torch.cat([d_input_past, d_input_future], dim=1)
                fake_prediction = d(torch.cat([batch["rel_past"][:, :, :2], fake_traj.detach()], dim=0))
                fake_loss = bce_loss(fake_prediction, fake_labels)
                # fake_loss.backward()

                d_loss = fake_loss + real_loss
                d_loss.backward()
                d_losses.append(d_loss.item())

                summary_writer_discriminator.add_scalar("GAN_loss", d_loss, step)

                if clipping_threshold_d > 0:
                    nn.utils.clip_grad_norm_(d.parameters(),
                                     clipping_threshold_d)

                d_optimizer.step()                                                 
                d_steps_left -= 1                                              

            ###################################################################
            #                 training the generator                           
            ###################################################################
            # g.zero_grad()
            logger.debug("Training the generator")
            g_optimizer.zero_grad()

            # d_input_past = batch["past"].permute(1, 0, 2).contiguous().view(batch["past"].size()[1], -1)
            # d_input_future_fake = fake_traj.permute(1, 0, 2).contiguous().view(batch["future"].size()[1], -1)
            # traj_fake = torch.cat([d_input_past, d_input_future_fake], dim=1)
            fake_prediction = d(torch.cat([batch["rel_past"][:, :, :2], fake_traj], dim=0))

            g_loss = bce_loss(fake_prediction, true_labels)
            g_loss.backward()
            g_losses.append(g_loss.item())

            # check if gradients of generator are changing and not exploding (>100)
            logger.debug(f"Generator Grad Total Norm: {check_grad_norm(g)}")

            summary_writer_generator.add_scalar("GAN_loss", g_loss, step)

            if clipping_threshold_g > 0:
                nn.utils.clip_grad_norm_(g.parameters(), clipping_threshold_g)

            g_optimizer.step()

            logger.debug(f"step {step} finished!")
            step += 1

        ##########################################################################################
        #                            evaluating the trained model
        ##########################################################################################
        with torch.no_grad():
            logger.debug("Evaluating the model")
            g.eval()
            d.eval()

            torch.cuda.synchronize(device=device)
            t1 = time.time()
            fake_traj = g(batch["past"][:,:max_agents,:], batch["rel_past"][:,:max_agents,:], batch["motion"][:max_agents])
            fake_traj = relative_to_abs(fake_traj, batch["past"][-1, :max_agents, :2])
            torch.cuda.synchronize(device=device)
            t2 = time.time()
            logger.debug(f"Trajectory Generation step took: {t2 - t1}")

            ADE_loss = displacement_error(fake_traj, batch["future"][:,:max_agents,:])[0].item()
            FDE_loss = final_displacement_error(fake_traj[-1], batch["future"][:,:max_agents,:])[0].item()

            msd = []
            for _ in range(int(TRAINING["k_samples"])):
                fake_traj = g(batch["past"][:,:max_agents,:], batch["rel_past"][:,:max_agents,:], batch["motion"][:max_agents])
                fake_traj = relative_to_abs(fake_traj, batch["past"][-1, :max_agents, :2])
                msd.append(msd_error(fake_traj, batch["future"][:,:max_agents,:])[0].item())

            MSD_loss = min(msd)

            # Todo: show some qualitative results of the predictions to be shown in tensorboard

        if int(TRAINING["epochs"]) > 0:
            epochs = int(TRAINING["epochs"])
            logger.info(
                f"TRAINING[{(epoch + 1):4}/{(start_epoch + epochs):4}]      "
                f"d_loss:{np.mean(d_losses):5.2f}      "
                f"g_loss:{np.mean(g_losses):5.2f}      "
                f"ADE_loss:{ADE_loss:8.2f}      "
                f"FDE_loss:{FDE_loss:8.2f}      "
                f"MSD_loss:{MSD_loss:8.2f}      ")

        validate(g, device)

        summary_writer_validation.add_scalar("ADE_loss", ADE_loss, epoch)
        summary_writer_validation.add_scalar("FDE_loss", FDE_loss, epoch)
        summary_writer_validation.add_scalar("MSD_loss", MSD_loss, epoch)

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


def validate(g, device=None):
    max_agents = int(TRAINING["max_agents"])
    nuscenes_data = NuSceneDataset(root_dir=DIRECTORIES["train_data"], test=True)
    data_loader = DataLoader(nuscenes_data,
                             batch_size=int(TRAINING["batch_size"]),
                             shuffle=True,
                             collate_fn=default_collate)
    if not g:
        embedder = None
        cae_encoder, cae_decoder = get_cae()
        if bool(GENERATOR["use_cae_encoder"]):
            logger.info("Using the CAE encoder...")
            embedder = cae_encoder

        logger.info("Constructing the GAN...")

        # Construct the models
        g = TrajectoryGenerator(
            embedder=embedder,
            de_embedder=None,
            embedding_dim=int(CAE["embedding_dim"]),
            encoder_h_dim=int(GENERATOR["encoder_h_dim"]),
            decoder_h_dim=int(GENERATOR["decoder_h_dim"]),
            seq_length=int(GENERATOR["seq_length"]),
            input_size=int(TRAINING["input_size"]),
            output_size=int(GENERATOR["output_size"]),
            decoder_mlp_structure=convert_str_to_list(GENERATOR["decoder_h2p_structure"]),
            decoder_mlp_activation=str(GENERATOR["decoder_h2p_activation"]),
            dropout=float(GENERATOR["dropout"]),
            fusion_pool_dim=int(GENERATOR["fusion_pool_dim"]),
            fusion_hidden_dim=int(GENERATOR["fusion_h_dim"]),
            context_feature_model_arch=str(GENERATOR["context_feature_model_arch"]),
            num_layers=int(GENERATOR["num_layers"]),
            pretrain=bool(GENERATOR["pretrain_vgg"]),
            refine=bool(GENERATOR["refine_vgg"])
        )
        logger.debug("Here is the generator...")
        logger.debug(g)
        # Get the device type
        device = get_device(logger)
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
            logger.info(f"No saved checkpoint for GAN")
            sys.exit(1)

    # Transfer the models to gpu
    # g.to(device)

    # Change the tensor types to GPU if neccessary
    # tensor_type = get_tensor_type()
    # g.type(tensor_type)

    ade_loss_list = []
    fde_loss_list = []
    msd_loss_list = []
    for batch in data_loader:
        # Transferring the input to the suitable device
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            logger.debug("Evaluating the model")
            g.eval()

            torch.cuda.synchronize(device=device)
            t1 = time.time()
            fake_traj = g(batch["past"][:, :max_agents, :], batch["rel_past"][:, :max_agents, :],
                          batch["motion"][:max_agents])
            fake_traj = relative_to_abs(fake_traj, batch["past"][-1, :max_agents, :2])
            torch.cuda.synchronize(device=device)
            t2 = time.time()

            ADE_loss = displacement_error(fake_traj, batch["future"][:, :max_agents, :])[0].item()
            FDE_loss = final_displacement_error(fake_traj[-1], batch["future"][:, :max_agents, :])[0].item()

            msd = []
            for _ in range(int(TRAINING["k_samples"])):
                fake_traj = g(batch["past"][:, :max_agents, :], batch["rel_past"][:, :max_agents, :],
                              batch["motion"][:max_agents])
                fake_traj = relative_to_abs(fake_traj, batch["past"][-1, :max_agents, :2])
                msd.append(msd_error(fake_traj, batch["future"][:, :max_agents, :])[0].item())

            MSD_loss = min(msd)
            ade_loss_list.append(ADE_loss)
            fde_loss_list.append(FDE_loss)
            msd_loss_list.append(MSD_loss)

    logger.info(
        f"VALIDATION:      "
        f"ADE_loss:{np.mean(ade_loss_list):8.2f}      "
        f"FDE_loss:{np.mean(fde_loss_list):8.2f}      "
        f"MSD_loss (mean):{np.mean(msd_loss_list):8.2f}      "
        f"MSD_loss (var):{np.std(msd_loss_list):8.2f}      ")
        # f"Trajectory Generation step took:{t2 - t1:8.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, action='store_true', help='train the model or test?')
    args = parser.parse_args()
    if args.train:
        main()
    else:
        validate(None)
