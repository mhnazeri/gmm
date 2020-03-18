from data.loader import *
from utils import *
from torch.utils.data import DataLoader
from cae import make_cae
from models import Generator, TrajectoryDiscriminator, Encoder

general_config = config("General")
path_config = config("Paths")
cae_config = config("CAE")
encoder_config = config("Encoder")
pooling_config = config("Pooling")
training_config = config("training")

if __name__ == '__main__':
    # This is for the first time run to train the CAE
    root = path_config["data_root"]
    torch.manual_seed(42)
    cae_data = CAEDataset(root)
    data = DataLoader(cae_data,
                      batch_size=int(cae_config["batch_size"]),
                      shuffle=True,
                      num_workers=2,
                      drop_last=True)

    latents_list = list(range(1, 8))
    for i, latent in enumerate(latents_list):
        plt.subplot(3, 3, i + 1)
        plt.title("latent_dimension: %d" % latent)
        cae_encoder, cae_decoder = make_cae(data, int(cae_config["input_dim"]), latent,
                                            int(cae_config["hidden_dim"]), int(cae_config["batch_size"]),
                                            int(cae_config["epochs"]), cae_config["activation"])

    # by now the CAE has been trained and the hidden dimensions are ready to be used in the next modules
    nuscenes_data = NuSceneDataset(root_dir=path_config["nuscenes_json"],
                                   max_agent=int(general_config["max_agents"]))

    data_loader = DataLoader(nuscenes_data,
                             batch_size=int(general_config["batch_size"]),
                             shuffle=True)

    epoch = 0
    while epoch < int(training_config["num_epochs"]):
        for i, batch in enumerate(data_loader):
            rel_past = batch["rel_past"]
            past = batch[""]
            latent_trajectories = cae_encoder(batch)




        epoch += 1
















