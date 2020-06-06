import torch
import random


def bce_loss(input, target):
    """According to the usage of binary CE on top of sigmoid, it could be summarized as below
    :param input: The input tensor of shape [Batch_size]
    :param target: The target tensor of shape [Btach_size]
    :return: The binary cross entropy loss
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    """
    Input:
    - scores_fake: Tensor of shape (N,) containing scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN generator loss
    """
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    """
    Input:
    - scores_real: Tensor of shape (N,) giving scores for real samples
    - scores_fake: Tensor of shape (N,) giving scores for fake samples

    Output:
    - loss: Tensor of shape (,) giving GAN discriminator loss
    """
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.ones_like(scores_fake) * random.uniform(0, 0.3)
    loss_real = bce_loss(scores_real, y_real)
    loss_fake = bce_loss(scores_fake, y_fake)
    return loss_real + loss_fake


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, output_size=7):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 13). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 13). Ground truth
    predictions.
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    batch_size = pred_traj.size(1)
    loss = pred_traj_gt.permute(1, 0, 2)[:, :, :output_size] - pred_traj.permute(1, 0, 2)[:, :, :output_size]
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    return torch.sum(loss) / batch_size, loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, output_size=7):
    """
    :param pred_pos: Tensor of shape (batch, 13). Predicted last pos.
    :param pred_pos_gt: Tensor of shape (batch, 13). Groudtruth last pos
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    batch_size = pred_pos.size(0)
    loss = pred_pos_gt[:, :output_size] - pred_pos[:, :output_size]
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    return torch.sum(loss) / batch_size, loss


def msd_error(pred_traj, pred_traj_gt, consider_ped=None, output_size=7):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 13). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 13). Ground truth
    predictions.
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    batch_size = pred_traj.size(1)
    loss = (pred_traj_gt.permute(1, 0, 2)[:, :, :output_size] - pred_traj.permute(1, 0, 2)[:, :, :output_size]).pow(2)

    if consider_ped is not None:
        loss = loss.sum(dim=2).sum(dim=1) * consider_ped
    else:
        loss = loss.sum(dim=2).sum(dim=1)

    return torch.sum(loss) / batch_size, loss


def cae_loss(output_encoder, outputs, inputs, device, lamda=1e-4):
    """Contractive auto-encoder loss
    :param output_encoder: output of encoder module
    :param outputs: output of decoder module
    :param inputs: ground-truth values
    :param lamda: coefficient for Frobenious norm
    :return: torch.IntTensor: conractive loss, torch.IntTensor of size 1
    """
    criterion = torch.nn.MSELoss()
    assert (
        outputs.shape == inputs.shape
    ), f"outputs.shape : {outputs.shape} != inputs.shape : {inputs.shape}"

    loss1 = criterion(outputs, inputs)
    output_encoder.backward(torch.ones(output_encoder.size()).to(device), retain_graph=True)
    inputs.grad.requires_grad = True
    loss2 = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))
    inputs.grad.data.zero_()
    loss = loss1 + (lamda * loss2)
    return loss
