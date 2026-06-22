import torch

from fm_train.data import EpochRandomSampler, apply_prompt_dropout
from fm_train.producer import BoundedProducer


def test_prompt_dropout_replaces_conditioning():
    embeds = torch.ones(3, 2, 4)
    masks = torch.ones(3, 2, dtype=torch.long)
    empty = torch.zeros(1, 2, 4)
    empty_mask = torch.zeros(1, 2, dtype=torch.long)
    result, result_mask, dropped = apply_prompt_dropout(embeds, masks, empty, empty_mask, 1.0)
    assert dropped.all()
    assert not result.any()
    assert not result_mask.any()


def test_bounded_producer_preserves_order():
    assert list(BoundedProducer(range(20), lambda value: value * 2, maxsize=2)) == list(range(0, 40, 2))


def test_epoch_sampler_is_repeatable_and_changes_by_epoch():
    sampler = EpochRandomSampler(list(range(20)), seed=7)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)
