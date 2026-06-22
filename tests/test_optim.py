import pytest
import torch

from fm_train.config import LRConfig, OptimizerConfig
from fm_train.optim import WarmupPlateauCooldown, build_optimizer, unscale_gradients


@pytest.mark.parametrize("name", ["sgd", "adamw"])
def test_optimizer_factory(name):
    parameter = torch.nn.Parameter(torch.ones(1))
    assert build_optimizer([parameter], OptimizerConfig(name=name)).param_groups


def test_adamw8bit_has_actionable_error_without_dependency():
    parameter = torch.nn.Parameter(torch.ones(1))
    try:
        optimizer = build_optimizer([parameter], OptimizerConfig(name="adamw8bit"))
    except RuntimeError as error:
        assert "bitsandbytes" in str(error)
    else:
        assert optimizer.param_groups


def test_warmup_and_terminal_zero():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = WarmupPlateauCooldown(
        optimizer, max_steps=20, config=LRConfig(warmup_ratio=0.1, cooldown_ratio=0.2)
    )
    scheduler.step()
    assert scheduler.get_last_lr() == [0.5]
    for _ in range(19):
        scheduler.step()
    assert scheduler.get_last_lr() == [0.0]


def test_loss_unscale():
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([1024.0])
    unscale_gradients([parameter], 1024)
    assert parameter.grad.item() == 1.0

