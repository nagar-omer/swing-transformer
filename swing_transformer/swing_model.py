import numpy as np
import torch
from torch import nn
from torch.nn import Linear, MultiheadAttention
from torch.utils.data import DataLoader
from swing_transformer.dataset_loaders.fashion_dataset_loader import KaggleFashionDataset, group_images


def apply_linear_on_axis(x: torch.Tensor, linear: nn.Linear, axis: int):
    """
    Apply linear layer on a specific axis of a tensor
    :param x: tensor to apply linear layer on
    :param linear: linear layer to apply
    :param axis: axis to apply linear layer on
    :return: tensor with linear layer applied on axis
    """
    assert type(x) == torch.Tensor, 'x must be a tensor'
    assert type(linear) == nn.Linear, 'linear must be a linear layer'
    assert x.dim() >= axis, 'axis must be smaller than the number of dimensions of x'

    # get shape of x
    in_shape = list(x.shape)
    # get shape of linear layer output
    linear_in_shape = linear.in_features
    linear_out_shape = linear.out_features

    # calculate output shape after linear layer
    out_shape = in_shape.copy()
    del out_shape[axis]
    out_shape.append(linear_out_shape)

    assert in_shape[axis] == linear_in_shape, 'linear layer output must have the same size as x on axis'
    # move axis to the end
    permutation = list(range(len(in_shape)))
    permutation.remove(axis)
    permutation.append(axis)

    reshaped_tensor = x.permute(*permutation).contiguous().view(-1, linear_in_shape)
    transformed = linear(reshaped_tensor)

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
    n_windows = 4
    assert type(msa) == MultiheadAttention, 'mas must be a multihead attention layer'
    assert type(x) == torch.Tensor, 'x must be a tensor'
    assert n_windows ** 0.5 % 1 == 0, 'n_windows must be a perfect square'
    assert x.dim() == 4, 'x must be 4D tensor - n-samples x channels x height x width'
    assert x.shape[2] % (n_windows / 2) == 0 and x.shape[3] % (n_windows/2) == 0, \
        'image width and height must be divisible by window_size'

    # get shapes and window size
    n_samples, n_channels, height, width = x.shape
    window_size_h, window_size_w = int(height // (n_windows / 2)), int(width // (n_windows / 2))

    # roll and reshape - prepare for multihead attention
    if shift:
        x = torch.roll(x, shifts=(window_size_h // 2, window_size_w // 2), dims=(2, 3))
    x = x.reshape(n_samples, n_windows, window_size_h * window_size_w, n_channels).permute(1, 0, 2, 3)

    # TODO: parallelize
    # apply multi-head attention
    for i in range(n_windows):
        weights, att = msa(x[i], x[i], x[i])
        x[i] = weights

    # return to original shape and roll back
    x = x.permute(1, 0, 2, 3).reshape(n_samples, n_channels, height, width)
    if shift:
        x = torch.roll(x, shifts=(-window_size_h // 2, -window_size_w // 2), dims=(2, 3))
    return x


class Swing(nn.Module):
    def __init__(self, linear_embedding_dim=96):
        super(Swing, self).__init__()
        # first image embedding - first patch-size is of size 3 x 4 x 4 = 48
        self._linear_embedding = Linear(in_features=48, out_features=linear_embedding_dim)
        self._w_msa = MultiheadAttention(embed_dim=linear_embedding_dim, num_heads=linear_embedding_dim)
        self._sw_msa = MultiheadAttention(embed_dim=linear_embedding_dim, num_heads=linear_embedding_dim)
        self._classifier = None

    def forward(self, x):
        patches_4x4 = patch_merging(x, patch_size=4)
        emb_4x4 = apply_linear_on_axis(patches_4x4, self._linear_embedding, axis=1)
        hidden_4x4 = apply_window_msa(self._w_msa, emb_4x4, n_windows=4)
        pass

    def _set_transformer(self):
        pass

    def _set_classifier(self):
        pass

    def _set_optimizer(self):
        pass


if __name__ == '__main__':
    ds = KaggleFashionDataset()
    dl = DataLoader(ds,
                    num_workers=4,
                    batch_size=16,
                    collate_fn=group_images)
    swing_trs = Swing()
    for i, (img, label) in enumerate(dl):
        print(i, img.shape, label)
        swing_trs(img)
