import torch
from PIL import Image

from fm_train.data import EpochRandomSampler, apply_prompt_dropout, prepare_image
from collections import deque

from fm_train.precompute import online_indices, prune_consumed_keys
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


def test_prompt_dropout_trims_empty_conditioning_to_batch_length():
    embeds = torch.ones(1, 2, 4)
    masks = torch.ones(1, 2, dtype=torch.long)
    empty = torch.zeros(1, 5, 4)
    empty_mask = torch.zeros(1, 5, dtype=torch.long)

    result, result_mask, _ = apply_prompt_dropout(
        embeds, masks, empty, empty_mask, probability=1.0
    )

    assert result.shape == embeds.shape
    assert result_mask.shape == masks.shape


def test_bounded_producer_preserves_order():
    assert list(BoundedProducer(range(20), lambda value: value * 2, maxsize=2)) == list(range(0, 40, 2))


def test_epoch_sampler_is_repeatable_and_changes_by_epoch():
    sampler = EpochRandomSampler(list(range(20)), seed=7)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_epoch_sampler_can_include_epoch_in_indices():
    sampler = EpochRandomSampler(list(range(20)), seed=7, include_epoch=True)

    assert all(epoch == 0 for epoch, _ in sampler)
    sampler.set_epoch(1)
    assert all(epoch == 1 for epoch, _ in sampler)


def test_online_producer_order_matches_training_sampler_epochs():
    values = list(range(20))
    sampler = EpochRandomSampler(values, seed=7)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        expected = list(sampler)
        actual = online_indices(len(values), seed=7, epoch=epoch, shard_index=0, num_shards=1)
        assert actual == expected


def test_online_producer_drops_distributed_training_tail():
    values = list(range(29))
    sampler = EpochRandomSampler(values, seed=7)
    expected = list(sampler)[:24]

    actual = online_indices(
        len(values),
        seed=7,
        epoch=0,
        shard_index=0,
        num_shards=1,
        consumer_batch_size=8,
        consumer_processes=3,
    )

    assert actual == expected


def test_online_producer_tail_drop_happens_before_precompute_sharding():
    actual = online_indices(
        29,
        seed=7,
        epoch=0,
        shard_index=1,
        num_shards=2,
        consumer_batch_size=8,
        consumer_processes=3,
    )

    assert actual == online_indices(29, 7, 0, 0, 1, 8, 3)[1::2]


def test_online_window_prunes_consumed_keys_outside_fifo_order():
    class Cache:
        def __init__(self, existing: set[str]):
            self.existing = existing

        def contains(self, key: str) -> bool:
            return key in self.existing

    pending = deque(["kept-head", "consumed-middle", "kept-tail"])

    prune_consumed_keys(Cache({"kept-head", "kept-tail"}), pending)

    assert list(pending) == ["kept-head", "kept-tail"]


def test_prepare_image_makes_non_square_image_exactly_square():
    result = prepare_image(Image.new("RGB", (1200, 1600)), resolution=1024)

    assert result.shape == (3, 1024, 1024)
    assert result.is_contiguous()
