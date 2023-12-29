import torch
from torch.nn import Linear

from swing_transformer.utils import apply_linear_on_axis, apply_window_msa


def test_apply_linear_on_axis():
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    # init linear layer with ones
    linear = Linear(in_features=3, out_features=1)

    # Initialize weights and biases with constant value 2
    with torch.no_grad():  # Disable gradient tracking
        linear.weight.fill_(2)  # Fill the weights with 2
        linear.bias.fill_(0)  # Fill the bias with 0
    axis = 1

    # x = n_samples x n_channels x height x width
    img_1 = torch.Tensor([
        [[1, 1],
        [1, 1]],
        [[2, 2],
        [2, 2]],
        [[3, 3],
        [3, 3]]
    ])
    img_2 = torch.Tensor([
        [[4, 4],
        [4, 4]],
        [[5, 5],
        [5, 5]],
        [[6, 6],
        [6, 6]]
    ])
    x = torch.stack([img_1, img_2], dim=0)

    # expected output is a tensor of shape n_samples x 1 x height x width
    expected_out = torch.stack([img_1.sum(axis=0) * 2, img_2.sum(axis=0) * 2], dim=0).unsqueeze(1)

    # apply and test
    out = apply_linear_on_axis(x, linear, 1)
    assert (expected_out == out).all().item(), 'apply_linear_on_axis failed'


def test_apply_window_msa():
    # x = n_samples x n_channels x height x width
    img_1 = torch.Tensor([
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 4, 4],
            [0, 0, 4, 4]
        ]
    ]
    )
    img_2 = torch.Tensor([
        [
            [5, 5, 6, 6],
            [5, 5, 6, 6],
            [7, 7, 8, 8],
            [7, 7, 8, 8]
        ]
    ]
    )
    x = torch.stack([img_1, img_2], dim=0)

    msa = torch.nn.MultiheadAttention(embed_dim=1, num_heads=1, bias=False, batch_first=True)
    with torch.no_grad():
        msa.in_proj_weight.fill_(1)
        msa.out_proj.weight.fill_(1)

    out = apply_window_msa(msa, x, n_windows=4)
    assert (out == x).all().item(), 'apply_window_msa failed'


if __name__ == '__main__':
    test_apply_linear_on_axis()
    test_apply_window_msa()