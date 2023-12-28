import torch
from torch import nn
from torch.utils.data import DataLoader
from swing_transformer.dataset_loaders.fashion_dataset_loader import KaggleFashionDataset, group_images


def partition_images(images: torch.Tensor, patch_size: int):
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


class Swing(nn.Module):
    def __init__(self):
        super(Swing, self).__init__()
        self._transformer = None
        self._classifier = None

    def forward(self, x):
        partition_images(x, patch_size=2)
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
