import torch


def bce_loss(input, target):
    """According to the usage of binary CE on top of sigmoid, it could be summarized as below
    :param input: The input tensor of shape [Batch_size]
    :param target: The target tensor of shape [Btach_size]
    :return: The binary cross entropy loss
    """

    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None):
    """
    :param pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)

    return torch.sum(loss) / pred_traj.shape[1], loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None):
    """
    :param pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    :param pred_pos_gt: Tensor of shape (batch, 2). Groudtruth last pos
    :param consider_ped: Tensor of shape (batch)
    :return tuple: the average loss over batch,
             the loss tensor to be further used in qualitative results
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))

    return torch.sum(loss) / pred_pos.shape[1], loss


def cae_loss(output_encoder, outputs, inputs, lamda=1e-4):
    """Contractive auto-encoder loss
    :param output_encoder: output of encoder module
    :param outputs: output of decoder module
    :param inputs: ground-truth values
    :param lamda: coefficient for Frobenious norm
    :return: torch.IntTensor: conractive loss, torch.IntTensor of size 1
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    criterion = torch.nn.MSELoss()
    assert (
        outputs.shape == inputs.shape
    ), f"outputs.shape : {outputs.shape} != inputs.shape : {inputs.shape}"

    loss1 = criterion(outputs, inputs)
    output_encoder.backward(torch.ones(output_encoder.size()).to(device), retain_graph=True)
    inputs.grad.requires_grad = True
    # Frobenious norm, the square root of sum of all elements (square value)
    # in a jacobian matrix
    loss2 = torch.sqrt(torch.sum(torch.pow(inputs.grad, 2)))
    inputs.grad.data.zero_()
    loss = loss1 + (lamda * loss2)
    return loss
