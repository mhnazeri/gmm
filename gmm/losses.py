import torch


def bce_loss(input, target):
    """According to the usage of binary CE on top of sigmoid, it could be summarized as below
    :param input: The input tensor of shape [Batch_size]
    :param target: The target tensor of shape [Btach_size]
    :return: The binary cross entropy loss
    """
    bce = torch.nn.BCEWithLogitsLoss()
    loss = bce(input.squeeze(), target)
    return loss


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 13). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 13). Ground truth
    predictions.
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    batch_size = pred_traj.size(1)
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    # 10 is the prediction len
    return torch.sum(loss) / (batch_size * 10), loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None):
    """
    :param pred_pos: Tensor of shape (batch, 13). Predicted last pos.
    :param pred_pos_gt: Tensor of shape (batch, 13). Groudtruth last pos
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    batch_size = pred_pos.size(0)
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    return torch.sum(loss) / batch_size, loss


def msd_error(pred_traj, pred_traj_gt, consider_ped=None):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 13). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 13). Ground truth
    predictions.
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    batch_size = pred_traj.size(1)
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)).pow(2)

    if consider_ped is not None:
        loss = loss.sum(dim=2).sum(dim=1) * consider_ped
    else:
        loss = loss.sum(dim=2).sum(dim=1)
    # 10 is the prediction len
    return torch.sum(loss) / (batch_size * 10), loss


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
    output_encoder.backward(
        torch.ones(output_encoder.size()).to(device), retain_graph=True
    )
    inputs.grad.requires_grad = True
    loss2 = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))
    inputs.grad.data.zero_()
    loss = loss1 + (lamda * loss2)
    return loss
