import torch

from fm_train.objective import FlowSchedule, flow_mse, flow_target, interpolate_flow, loss_weight


def test_flow_endpoints_and_target():
    clean = torch.tensor([[[[2.0]]]])
    noise = torch.tensor([[[[5.0]]]])
    assert torch.equal(interpolate_flow(clean, noise, torch.tensor([[[[0.0]]]])), clean)
    assert torch.equal(interpolate_flow(clean, noise, torch.tensor([[[[1.0]]]])), noise)
    assert torch.equal(flow_target(clean, noise), torch.tensor([[[[3.0]]]]))


def test_weighted_mse():
    prediction = torch.tensor([[[[1.0]]], [[[2.0]]]])
    target = torch.zeros_like(prediction)
    assert flow_mse(prediction, target, torch.ones_like(prediction)).item() == 2.5
    sigma = torch.tensor([0.5])
    assert loss_weight(sigma, "sigma_inverse").item() > 3.9


def test_flow_schedule_lookup_is_deterministic():
    schedule = FlowSchedule.create(shift=3.0, num_train_timesteps=100)
    first = schedule.sample(4, torch.device("cpu"), torch.float32, torch.Generator().manual_seed(9))
    second = schedule.sample(4, torch.device("cpu"), torch.float32, torch.Generator().manual_seed(9))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].min() >= 0 and first[0].max() <= 1
