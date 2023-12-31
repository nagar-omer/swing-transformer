from typing import Callable

import torch
from torch import nn
from torch.nn import MultiheadAttention


def apply_layer_on_axis(x: torch.Tensor, layer: Callable, axis: int, in_dim: int, out_dim: int):
    """
    Apply linear layer on a specific axis of a tensor
    :param x: tensor to apply linear layer on
    :param layer: linear layer to apply
    :param axis: axis to apply linear layer on
    :param in_dim: input dimension of layer
    :param out_dim: output dimension of layer
    :return: tensor with linear layer applied on axis
    """
    assert type(x) == torch.Tensor, 'x must be a tensor'
    assert x.dim() >= axis, 'axis must be smaller than the number of dimensions of x'

    # get shape of x
    in_shape = list(x.shape)

    # calculate output shape after linear layer
    out_shape = in_shape.copy()
    del out_shape[axis]
    out_shape.append(out_dim)

    assert in_shape[axis] == in_dim, 'linear layer output must have the same size as x on axis'
    # move axis to the end
    permutation = list(range(len(in_shape)))
    permutation.remove(axis)
    permutation.append(axis)

    reshaped_tensor = x.permute(*permutation).contiguous().view(-1, in_dim)
    transformed = layer(reshaped_tensor)

    # view as original shape and back to original permutation
    reversed_permutation = list(range(len(in_shape)))
    reversed_permutation = reversed_permutation[:axis] + [len(in_shape) - 1] + reversed_permutation[axis:-1]

    x = transformed.view(*out_shape).permute(*reversed_permutation)

    return x


def patch_merging(images: torch.Tensor, patch_size: int):
    """
    Partition image into blocks of size block_size
    :param patch_size: size of each block - output will contain block_size x block_size
    :param images: image to partition
    :return: Tensor of shape n-samples x (n-channels * patch_size ** 2) x (height // patch_size) x (width // patch_size)
    """
    assert type(images) == torch.Tensor, 'images must be a tensor'
    assert images.dim() >= 4, 'images must be 4D tensor - n-samples x channels x height x width'
    assert images.shape[2] % patch_size == 0 and images.shape[3] % patch_size == 0, \
        'image width and height must be divisible by block_size'

    n_samples, n_channels, height, width = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.reshape(n_samples, n_channels * patch_size ** 2, height // patch_size, width // patch_size)
    return patches


def apply_window_msa(msa: MultiheadAttention, x: torch.Tensor, n_windows: int = 4, shift: bool = False):
    """
    Apply windowed multihead self attention
    :param msa: multi-head attention layer
    :param x: input tensor
    :param n_windows: number of windows (in each dimension n_windows / 2)
    :param shift: if True, shift the windows by 1/2 of the window size
    :return: output of multi-head attention
    """
    assert type(msa) == MultiheadAttention, 'mas must be a multihead attention layer'
    assert type(x) == torch.Tensor, 'x must be a tensor'
    assert n_windows ** 0.5 % 1 == 0, 'n_windows must be a perfect square'
    assert x.dim() == 4, 'x must be 4D tensor - n-samples x channels x height x width'
    assert x.shape[2] % (n_windows / 2) == 0 and x.shape[3] % (n_windows/2) == 0, \
        'image width and height must be divisible by window_size'

    # get shapes and window size
    n_windows_h, n_windows_w = int(n_windows ** 0.5), int(n_windows ** 0.5)
    n_samples, n_channels, height, width = x.shape
    window_size_h, window_size_w = int(height // (n_windows / 2)), int(width // (n_windows / 2))

    # roll and reshape - prepare for multi-head attention
    if shift:
        # pad to allow rolling
        pad_h_left, pad_h_right = window_size_h // 2, window_size_h // 2 + window_size_h % 2
        pad_w_left, pad_w_right = window_size_w // 2, window_size_w // 2 + window_size_w % 2

        x = torch.nn.functional.pad(x, (pad_w_left, pad_w_right, pad_h_left, pad_h_right))
        n_windows = int((n_windows ** 0.5 + 1) ** 2)
        n_windows_h, n_windows_w = int(n_windows ** 0.5), int(n_windows ** 0.5)
        height, width = x.shape[2], x.shape[3]

    x = x.unfold(2, window_size_h, window_size_w).unfold(3, window_size_h, window_size_w)
    x = x.reshape(n_samples, n_channels, n_windows, window_size_h * window_size_w).permute(2, 0, 3, 1)

    # TODO: parallelize
    # apply multi-head attention
    for i in range(n_windows):
        x[i], weights = msa(x[i], x[i], x[i])

    # return to original shape and roll back
    x = x.permute(1, 3, 0, 2).reshape(n_samples, n_channels, n_windows_h, n_windows_w, window_size_h, window_size_w)
    x = x.permute(0, 1, 2, 4, 3, 5).reshape(n_samples, n_channels, height, width)

    if shift:
        x = x[:, :, pad_h_left:-pad_h_right, pad_w_left:-pad_w_right]
    return x

