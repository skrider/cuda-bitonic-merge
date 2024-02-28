from cbm import sort
import pytest
import torch

@pytest.mark.parametrize("d", [64, 1024, 4096])
@pytest.mark.parametrize("b", [8, 16, 32])
@pytest.mark.parametrize("dim", [0, 1])
@torch.no_grad()
def test_sort(d, b, dim):
    torch.random.manual_seed(0)
    output = torch.randn(d, b, device="cuda")
    input = output.clone()
    sort(output, dim=dim)
    assert torch.all(torch.eq(output, torch.sort(input, dim=dim).values))
