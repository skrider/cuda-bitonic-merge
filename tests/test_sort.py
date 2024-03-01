from cbm import sort
import pytest
import torch

torch.set_printoptions(profile="full")

# @pytest.mark.parametrize("d", [64, 1024, 4096])
# @pytest.mark.parametrize("b", [8, 16, 32])
# @pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("d", [256])
@pytest.mark.parametrize("b", [8])
@pytest.mark.parametrize("dim", [1])
@torch.no_grad()
def test_sort(d, b, dim):
    torch.random.manual_seed(0)
    output = torch.arange(0, d, device="cuda").flip(0).tile((b, 1)).half()
    input = output.clone()
    sort(output, dim=dim)
    __import__('pdb').set_trace()
    assert torch.all(torch.eq(output, torch.sort(input, dim=dim).values))
