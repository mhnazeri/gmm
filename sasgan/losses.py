import torch


def bce_loss(input, target):
    """
    According to the usage of binary CE on top of sigmoid, it could be summarized as below
    :param input: The input tensor of shape [Batch_size]
    :param target: The target tensor of shape [Btach_size]
    :return: The binary cross entropy loss
    """

    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    COPIED FROM SOICAL GAN CODE
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(
    pred_pos, pred_pos_gt, consider_ped=None, mode='sum'
):
    """
    COPIED FROM SOICAL GAN CODE
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def cae_loss(output_encoder, outputs, inputs, lamda=1e-4):
    """
    Will be described by mohammad
    :param output_encoder:
    :param outputs:
    :param inputs:
    :param lamda:
    :return:
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
