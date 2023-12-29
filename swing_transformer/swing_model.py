import numpy as np
import torch
from torch import nn
from torch.nn import Linear, MultiheadAttention
from torch.utils.data import DataLoader
from swing_transformer.dataset_loaders.fashion_dataset_loader import KaggleFashionDataset, group_images
from swing_transformer.utils import patch_merging, apply_linear_on_axis, apply_window_msa


class Swing(nn.Module):
    def __init__(self, linear_embedding_dim=96):
        super(Swing, self).__init__()
        # first image embedding - first patch-size is of size 3 x 4 x 4 = 48
        self._linear_embedding = Linear(in_features=48, out_features=linear_embedding_dim)
        self._w_msa = MultiheadAttention(embed_dim=linear_embedding_dim, num_heads=linear_embedding_dim, batch_first=True)
        self._sw_msa = MultiheadAttention(embed_dim=linear_embedding_dim, num_heads=linear_embedding_dim, batch_first=True)
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
    ds = KaggleFashionDataset(transform_type="resize-norm")
    dl = DataLoader(ds,
                    num_workers=4,
                    batch_size=16,
                    collate_fn=group_images)
    swing_trs = Swing()
    for i, (img, label) in enumerate(dl):
        print(i, img.shape, label)
        swing_trs(img)
