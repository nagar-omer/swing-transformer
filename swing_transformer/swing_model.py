import numpy as np
import torch
from torch import nn
from torch.nn import Linear, MultiheadAttention
from torch.utils.data import DataLoader
from swing_transformer.dataset_loaders.fashion_dataset_loader import KaggleFashionDataset, group_images
from swing_transformer.utils import patch_merging, apply_layer_on_axis, apply_window_msa


class MLP(nn.Module):
    """
    Multi-layer perceptron
    2 layers with gelu activation
    """
    def __init__(self, dim):
        super(MLP, self).__init__()
        self._linear_1 = Linear(in_features=dim, out_features=dim)
        self._linear_2 = Linear(in_features=dim, out_features=dim)

    def forward(self, x):
        x = self._linear_1(x)
        x = torch.relu(x)
        x = self._linear_2(x)
        return x


class SwingBlock(nn.Module):
    """
    Swing block x2
        step 1: linear -> layer-norm -> W-MSA -> (residual-connection) -> linear -> layer-norm -> MLP
        step 2: linear -> layer-norm -> SW-MSA -> (residual-connection) -> linear -> layer-norm -> MLP
    """
    def __init__(self, dim, n_heads=3, shift=False):
        super(SwingBlock, self).__init__()
        # first block params W-MSA
        #   linear -> layer-norm -> W-MSA -> (residual-connection) ->
        #   linear -> layer-norm -> MLP -> (residual-connection)
        self._dim = dim
        self._shifted_window = shift

        self._linear_msa = Linear(in_features=dim, out_features=dim)
        self._norm_msa = nn.LayerNorm(normalized_shape=dim)
        self._msa = MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self._linear_mlp = Linear(in_features=dim, out_features=dim)
        self._norm_mlp = nn.LayerNorm(normalized_shape=dim)
        self._mlp = MLP(dim=dim)

    def forward(self, x):
        # first block
        x_ = x.clone()
        x = apply_layer_on_axis(x, self._linear_msa, axis=1, in_dim=self._dim, out_dim=self._dim)
        x = apply_layer_on_axis(x, self._norm_msa, axis=1, in_dim=self._dim, out_dim=self._dim)  # check correct normalization
        x = x_ + apply_window_msa(self._msa, x, n_windows=4, shift=self._shifted_window)

        x_ = x.clone()
        x = apply_layer_on_axis(x, self._linear_mlp, axis=1, in_dim=self._dim, out_dim=self._dim)
        x = apply_layer_on_axis(x, self._norm_mlp, axis=1, in_dim=self._dim, out_dim=self._dim)
        x = x_ + apply_layer_on_axis(x, self._mlp, axis=1, in_dim=self._dim, out_dim=self._dim)

        return x


class SwingBlockStack(nn.Module):
    """
    Stack of Swing blocks
    """
    def __init__(self, dim, n_blocks, n_heads):
        super(SwingBlockStack, self).__init__()
        self._dim = dim
        self._n_blocks = n_blocks

        self._swing_blocks = nn.ModuleList([
            SwingBlock(dim=dim, shift=i % 2 == 0, n_heads=n_heads)
            for i in range(n_blocks)
        ])

    def forward(self, x):
        for block in self._swing_blocks:
            x = block(x)
        return x


class SwingTransformer(nn.Module):
    def __init__(self, transform_factor=2, window_size=2, n_blocks=None, n_heads=None):
        super(SwingTransformer, self).__init__()

        # number of blocks per stage
        assert n_blocks is None or len(n_blocks) == 4, 'n_blocks must be a list of length 4'
        assert n_blocks is None or all([n > 0 for n in n_blocks]), 'n_blocks must be a list of positive integers'
        self._n_blocks = [2, 2, 6, 2] if n_blocks is None else n_blocks
        self._n_heads = [3, 6, 12, 24] if n_heads is None else n_heads

        # embedding layer
        self._transform_factor = transform_factor
        self._window_size = window_size

        # embedding dimension first patch-size is of size 3 x 4 x 4 = 48
        self._c = window_size ** 2 * 3  # 3 channels per patch

        # Stage 1: linear embedding + Swing x 2
        self._stage_1_dim = self._c * transform_factor
        self._linear_embedding = Linear(in_features=self._c, out_features=self._stage_1_dim)
        self._stage_1_swing_block = SwingBlockStack(dim=self._stage_1_dim, n_blocks=self._n_blocks[0], n_heads=self._n_heads[0])

        # Stage 2: patch merging + Swing x 2
        self._stage_2_in_dim = self._stage_1_dim * (self._window_size ** 2)
        self._stage_2_dim = self._c * (2 * transform_factor)
        self._stage_2_expansion_layer = Linear(in_features=self._stage_2_in_dim, out_features=self._stage_2_dim)
        self._stage_2_swing_block = SwingBlockStack(dim=self._stage_2_dim, n_blocks=self._n_blocks[1], n_heads=self._n_heads[1])

        # Stage 3: patch merging + Swing x 6
        self._stage_3_in_dim = self._stage_2_dim * (self._window_size ** 2)
        self._stage_3_dim = self._c * (4 * transform_factor)
        self._stage_3_expansion_layer = Linear(in_features=self._stage_3_in_dim, out_features=self._stage_3_dim)
        self._stage_3_swing_block = SwingBlockStack(dim=self._stage_3_dim, n_blocks=self._n_blocks[2], n_heads=self._n_heads[2])

        # Stage 4: patch merging + Swing x 2
        self._stage_4_in_dim = self._stage_3_dim * (self._window_size ** 2)
        self._stage_4_dim = self._c * (8 * transform_factor)
        self._stage_4_expansion_layer = Linear(in_features=self._stage_4_in_dim, out_features=self._stage_4_dim)
        self._stage_4_swing_block = SwingBlockStack(dim=self._stage_4_dim, n_blocks=self._n_blocks[3], n_heads=self._n_heads[3])

        self._set_optimizer()

    def forward(self, x):
        # Patch partitioning
        patch_partition = patch_merging(x, patch_size=self._window_size)

        # Stage 1: linear embedding + Swing x 2
        emb1 = apply_layer_on_axis(patch_partition, self._linear_embedding, axis=1,
                                   in_dim=self._c, out_dim=self._stage_1_dim)
        hidden1 = self._stage_1_swing_block(emb1)

        # Stage 2: patch merging + Swing x 2
        emb2 = patch_merging(hidden1, patch_size=self._window_size)
        emb2 = apply_layer_on_axis(emb2, self._stage_2_expansion_layer, axis=1,
                                   in_dim=self._stage_2_in_dim, out_dim=self._stage_2_dim)
        hidden2 = self._stage_2_swing_block(emb2)

        # Stage 3: patch merging + Swing x 6
        emb3 = patch_merging(hidden2, patch_size=self._window_size)
        emb3 = apply_layer_on_axis(emb3, self._stage_3_expansion_layer, axis=1,
                                   in_dim=self._stage_3_in_dim, out_dim=self._stage_3_dim)
        hidden3 = self._stage_3_swing_block(emb3)

        # Stage 4: patch merging + Swing x 2
        emb4 = patch_merging(hidden3, patch_size=self._window_size)
        emb4 = apply_layer_on_axis(emb4, self._stage_4_expansion_layer, axis=1,
                                   in_dim=self._stage_4_in_dim, out_dim=self._stage_4_dim)
        out = self._stage_4_swing_block(emb4)

        # TODO: consider adding a CLS token + attention pooling
        out = out.reshape(out.shape[0], -1)
        return out

    def _set_optimizer(self):
        self._optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    ds = KaggleFashionDataset(transform_type="resize-norm")
    dl = DataLoader(ds,
                    num_workers=4,
                    batch_size=16,
                    collate_fn=group_images)
    swing_trf = SwingTransformer()
    for i, (img, label) in enumerate(dl):
        print(i, img.shape, label)
        swing_trf(img)
        if i==10:
            break
