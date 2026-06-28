import torch

from fm_train.objective import (
    FlowSchedule,
    flow_mse,
    flow_mse_per_sample,
    flow_target,
    interpolate_flow,
    loss_weight,
)


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
    assert torch.equal(
        flow_mse_per_sample(prediction, target, torch.ones_like(prediction)),
        torch.tensor([1.0, 4.0]),
    )
    sigma = torch.tensor([0.5])
    assert loss_weight(sigma, "sigma_inverse").item() > 3.9


def test_flow_schedule_lookup_is_deterministic():
    schedule = FlowSchedule.create(shift=3.0, num_train_timesteps=100)
    first = schedule.sample(4, torch.device("cpu"), torch.float32, torch.Generator().manual_seed(9))
    second = schedule.sample(4, torch.device("cpu"), torch.float32, torch.Generator().manual_seed(9))
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].min() >= 0 and first[0].max() <= 1


def test_flow_schedule_samples_continuous_t_roughly_uniformly():
    schedule = FlowSchedule(sigmas=torch.linspace(0, 1, 100), timesteps=torch.arange(100))
    _, timesteps, _ = schedule.sample(10_000, torch.device("cpu"), torch.float32, torch.Generator().manual_seed(11))
    t = timesteps.float() / timesteps.float().max()
    buckets = torch.clamp((t * 10).long(), max=9)
    counts = torch.bincount(buckets, minlength=10)
    assert counts.min() > 900
    assert counts.max() < 1100
    assert timesteps.unique().numel() > 1000


def test_flow_schedule_t_bias_favors_cleaner_samples():
    schedule = FlowSchedule(sigmas=torch.linspace(0, 1, 100), timesteps=torch.arange(100))
    generator = torch.Generator().manual_seed(12)
    _, uniform_timesteps, _ = schedule.sample(10_000, torch.device("cpu"), torch.float32, generator)
    generator = torch.Generator().manual_seed(12)
    _, biased_timesteps, _ = schedule.sample(
        10_000, torch.device("cpu"), torch.float32, generator, t_sampling_bias=2.0
    )
    uniform_t = uniform_timesteps.float() / uniform_timesteps.float().max()
    biased_t = biased_timesteps.float() / uniform_timesteps.float().max()
    assert biased_t.mean() < uniform_t.mean()
    assert (biased_t < 0.1).float().mean() > (uniform_t < 0.1).float().mean()
