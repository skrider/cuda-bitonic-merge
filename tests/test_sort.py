from cbm import sort
import pytest
import torch

torch.set_printoptions(profile="full")

# @pytest.mark.parametrize("d", [64, 1024, 4096])
# @pytest.mark.parametrize("b", [8, 16, 32])
# @pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("d", [4096])
@pytest.mark.parametrize("b", [8])
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("dtype", [torch.int16])
@torch.no_grad()
def test_sort(d, b, dim, dtype):
    torch.random.manual_seed(0)
    output = torch.randint(0, d, (1, d), device="cuda", dtype=dtype).flip(0).tile((b, 1))
    input = output.clone()
    sort(output, dim=dim)
    result = torch.eq(output, torch.sort(input, dim=dim).values)
    __import__('pdb').set_trace()
    assert torch.all(result)
