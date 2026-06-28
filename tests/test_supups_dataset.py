import warnings

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fm_train.example_dataset import SupUpsDataset, SupUpsDataset4Tier, _open_rgb_image


def test_supups_filters_scores_and_uses_both_caption_variants(tmp_path):
    (tmp_path / "metadata").mkdir()
    rows = []
    for index in range(40):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (48, 32)).save(path)
        rows.append({
            "path": str(path), "source": "test",
            "cap_short": f"short {index}", "cap_long": f"long {index}",
            "photo_score": 0.8, "general_score": 0.2, "is_bad_pool": False,
        })
    rows.append({
        "path": str(tmp_path / "low.png"), "source": "test",
        "cap_short": "low", "cap_long": None,
        "photo_score": 0.1, "general_score": 0.2, "is_bad_pool": False,
    })
    rows.append({
        "path": str(tmp_path / "bad.png"), "source": "test",
        "cap_short": None, "cap_long": "bad pool caption",
        "photo_score": 0.1, "general_score": 0.2, "is_bad_pool": True,
    })
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part.parquet")

    train = SupUpsDataset(str(tmp_path), split="train", validation_fraction=0.2)
    validation = SupUpsDataset(str(tmp_path), split="validation", validation_fraction=0.2)
    train_ids = {train.get_metadata(i)["id"] for i in range(len(train))}
    validation_ids = {validation.get_metadata(i)["id"] for i in range(len(validation))}

    assert train_ids.isdisjoint(validation_ids)
    assert len(train) + len(validation) == 41
    sample = next(
        train.get_metadata(i) for i in range(len(train))
        if not train.get_metadata(i)["is_bad_pool"]
    )
    assert sample["caption"].startswith("long ")
    assert sample["caption_short"].startswith("short ")
    assert sample["photo_score"] == 0.8


def test_supups_can_exclude_bad_pool(tmp_path):
    (tmp_path / "metadata").mkdir()
    path = tmp_path / "bad.png"
    Image.new("RGB", (32, 32)).save(path)
    pq.write_table(pa.Table.from_pylist([{
        "path": str(path), "cap_short": "bad", "cap_long": None,
        "photo_score": 0.1, "general_score": 0.1, "is_bad_pool": True,
    }]), tmp_path / "metadata/part.parquet")

    dataset = SupUpsDataset(
        str(tmp_path), split="train", validation_fraction=0.01, include_bad_pool=False
    )

    assert len(dataset) == 0


def test_supups_uses_long_captions_only_at_configured_rate(tmp_path):
    (tmp_path / "metadata").mkdir()
    rows = []
    for index in range(1000):
        rows.append({
            "path": str(tmp_path / f"{index}.png"),
            "cap_short": f"short {index}", "cap_long": f"long {index}",
            "photo_score": 0.8, "general_score": 0.8, "is_bad_pool": False,
        })
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part.parquet")
    dataset = SupUpsDataset(
        str(tmp_path), split="train", validation_fraction=0.01,
        long_caption_probability=0.1, caption_seed=7,
    )

    captions = [dataset.get_metadata(index)["caption"] for index in range(len(dataset))]
    long_count = sum(caption.startswith("long ") for caption in captions)

    assert 50 < long_count < 150
    assert captions == [dataset.get_metadata(index)["caption"] for index in range(len(dataset))]


def test_supups_uses_configured_caption_weights(tmp_path):
    (tmp_path / "metadata").mkdir()
    rows = []
    for index in range(2000):
        rows.append({
            "path": str(tmp_path / f"{index}.png"),
            "cap_very_short": f"very short {index}",
            "cap_short": f"short {index}",
            "cap_long": f"long {index}",
            "photo_score": 0.8,
            "general_score": 0.8,
            "is_bad_pool": False,
        })
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part.parquet")
    dataset = SupUpsDataset(
        str(tmp_path), split="train", validation_fraction=0.01,
        caption_weights={"very_short": 0.5, "short": 0.25, "long": 0.175},
        caption_seed=11,
    )

    captions = [dataset.get_metadata(index)["caption"] for index in range(len(dataset))]
    very_short_count = sum(caption.startswith("very short ") for caption in captions)
    short_count = sum(caption.startswith("short ") for caption in captions)
    long_count = sum(caption.startswith("long ") for caption in captions)

    assert 1000 < very_short_count < 1150
    assert 475 < short_count < 600
    assert 325 < long_count < 425
    assert captions == [dataset.get_metadata(index)["caption"] for index in range(len(dataset))]
    assert dataset.get_metadata(0)["caption_very_short"].startswith("very short ")


def test_supups_redistributes_weights_over_available_captions(tmp_path):
    (tmp_path / "metadata").mkdir()
    rows = []
    for index in range(2000):
        rows.append({
            "path": str(tmp_path / f"{index}.png"),
            "cap_short": f"short {index}",
            "cap_long": f"long {index}",
            "photo_score": 0.8,
            "general_score": 0.8,
            "is_bad_pool": False,
        })
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part.parquet")
    dataset = SupUpsDataset(
        str(tmp_path), split="train", validation_fraction=0.01,
        caption_weights={"very_short": 0.5, "short": 0.25, "long": 0.175},
        caption_seed=17,
    )

    captions = [dataset.get_metadata(index)["caption"] for index in range(len(dataset))]
    short_count = sum(caption.startswith("short ") for caption in captions)
    long_count = sum(caption.startswith("long ") for caption in captions)

    assert short_count + long_count == len(captions)
    assert 1100 < short_count < 1250
    assert 700 < long_count < 850


def test_supups_reads_caption_columns_added_to_later_fragments(tmp_path):
    (tmp_path / "metadata").mkdir()
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    common = {"photo_score": 0.8, "general_score": 0.8, "is_bad_pool": False}
    pq.write_table(pa.Table.from_pylist([{
        "path": str(first_path),
        "cap_short": "short first",
        "cap_long": "long first",
        **common,
    }]), tmp_path / "metadata/part-0000.parquet")
    pq.write_table(pa.Table.from_pylist([{
        "path": str(second_path),
        "cap_very_short": "very short second",
        "cap_short": "short second",
        "cap_long": "long second",
        **common,
    }]), tmp_path / "metadata/part-0001.parquet")

    dataset = SupUpsDataset(
        str(tmp_path), split="train", validation_fraction=0.01,
        caption_weights={"very_short": 1.0},
    )
    metadata = [dataset.get_metadata(index) for index in range(len(dataset))]

    assert {item["caption_very_short"] for item in metadata} == {None, "very short second"}
    assert any(item["caption"] == "very short second" for item in metadata)
    assert any(item["caption"] in {"short first", "long first"} for item in metadata)


def test_supups_4tier_adds_tags_with_configured_distribution(tmp_path):
    (tmp_path / "metadata").mkdir()
    rows = []
    for index in range(2000):
        rows.append({
            "path": str(tmp_path / f"{index}.png"),
            "cap_very_short": f"very short {index}",
            "cap_short": f"short {index}",
            "cap_long": f"long {index}",
            "photo_score": 0.8,
            "general_score": 0.8,
            "is_bad_pool": False,
            "tags": "photo, landscape",
        })
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part.parquet")
    dataset = SupUpsDataset4Tier(
        str(tmp_path),
        split="train",
        validation_fraction=0.01,
        caption_seed=23,
        tag_seed=29,
        p_very_short=1.0,
        p_short=0.0,
        p_long=0.0,
        p_no_tag=0.5,
        p_tag_suffix=0.25,
        p_tag_prefix=0.25,
    )

    metadata = [dataset.get_metadata(index) for index in range(len(dataset))]
    none_count = sum(item["caption_tag"] is None for item in metadata)
    prefix_count = sum(item["caption_tag_placement"] == "prefix" for item in metadata)
    suffix_count = sum(item["caption_tag_placement"] == "suffix" for item in metadata)

    assert 900 < none_count < 1100
    assert 425 < prefix_count < 575
    assert 425 < suffix_count < 575
    assert all(item["tags"] == ["photo", "landscape"] for item in metadata)
    assert metadata == [dataset.get_metadata(index) for index in range(len(dataset))]


def test_supups_4tier_can_force_tag_formatting_and_placement(tmp_path):
    (tmp_path / "metadata").mkdir()
    path = tmp_path / "sample.png"
    pq.write_table(pa.Table.from_pylist([{
        "path": str(path),
        "cap_very_short": "tiny caption",
        "cap_short": "short caption",
        "cap_long": "long caption",
        "photo_score": 0.8,
        "general_score": 0.8,
        "is_bad_pool": False,
        "tags": "photo",
    }]), tmp_path / "metadata/part.parquet")
    dataset = SupUpsDataset4Tier(
        str(tmp_path),
        split="train",
        validation_fraction=0.01,
        p_very_short=1.0,
        p_short=0.0,
        p_long=0.0,
        p_no_tag=0.0,
        p_tag_suffix=0.0,
        p_tag_prefix=1.0,
        p_tag_upper=1.0,
        p_tag_lower=0.0,
        p_tag_period=1.0,
    )

    sample = dataset.get_metadata(0)

    assert sample["caption"] == "PHOTO. tiny caption"
    assert sample["caption_tag"] == "PHOTO."
    assert sample["caption_tag_placement"] == "prefix"


def test_supups_can_deterministically_downsample_configured_tags(tmp_path):
    (tmp_path / "metadata").mkdir()
    rows = []
    for index in range(4000):
        tag = ("illustration", "photo", "illustration, portrait", "")[index % 4]
        rows.append({
            "path": str(tmp_path / f"{index}.png"),
            "cap_short": f"short {index}",
            "cap_long": f"long {index}",
            "photo_score": 0.8,
            "general_score": 0.8,
            "is_bad_pool": False,
            "tags": tag,
        })
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part.parquet")

    kwargs = {
        "split": "train",
        "validation_fraction": 0.01,
        "tag_keep_probabilities": {"ILLUSTRATION": 0.25},
        "untagged_keep_probability": 0.5,
        "tag_filter_seed": 71,
    }
    first = SupUpsDataset(str(tmp_path), **kwargs)
    second = SupUpsDataset(str(tmp_path), **kwargs)
    changed_seed = SupUpsDataset(str(tmp_path), **{**kwargs, "tag_filter_seed": 72})
    first_ids = [first.get_metadata(index)["id"] for index in range(len(first))]

    counts = {}
    for index in range(len(first)):
        tags = tuple(first.get_metadata(index)["tags"])
        counts[tags] = counts.get(tags, 0) + 1

    assert 200 < counts[("illustration",)] < 300
    assert 900 < counts[("photo",)] < 1000
    assert 200 < counts[("illustration", "portrait")] < 300
    assert 450 < counts[()] < 550
    assert first_ids == [second.get_metadata(index)["id"] for index in range(len(second))]
    assert first_ids != [
        changed_seed.get_metadata(index)["id"] for index in range(len(changed_seed))
    ]


def test_supups_rejects_unknown_tag_filter_keys(tmp_path):
    (tmp_path / "metadata").mkdir()
    pq.write_table(pa.Table.from_pylist([{
        "path": str(tmp_path / "sample.png"),
        "cap_short": "short",
        "cap_long": "long",
        "photo_score": 0.8,
        "general_score": 0.8,
        "is_bad_pool": False,
        "tags": "illustration",
    }]), tmp_path / "metadata/part.parquet")

    try:
        SupUpsDataset(
            str(tmp_path),
            tag_keep_probabilities={"art": 0.5},
        )
    except ValueError as error:
        assert "Unknown tag_keep_probabilities keys: art" in str(error)
        assert "observed tags: illustration" in str(error)
    else:
        raise AssertionError("Expected unknown tag filter key to be rejected")


def test_supups_balances_mutually_exclusive_primary_classes(tmp_path):
    (tmp_path / "metadata").mkdir()
    class_tags = {
        "anime": "anime",
        "art": "illustration",
        "nsfw": "nsfw",
        "portrait": "portrait",
        "landscape": "landscape",
        "photo": "photo",
    }
    rows = []
    for class_name, tag in class_tags.items():
        for index in range(100):
            tag_value = "anime, photo" if class_name == "anime" and index < 20 else tag
            rows.append({
                "path": str(tmp_path / f"{class_name}-{index}.png"),
                "cap_short": f"short {class_name} {index}",
                "cap_long": f"long {class_name} {index}",
                "photo_score": 0.8,
                "general_score": 0.8,
                "is_bad_pool": False,
                "tags": tag_value,
            })
    for index in range(20):
        rows.append({
            "path": str(tmp_path / f"other-{index}.png"),
            "cap_short": f"short other {index}",
            "cap_long": f"long other {index}",
            "photo_score": 0.8,
            "general_score": 0.8,
            "is_bad_pool": False,
            "tags": "other",
        })
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part.parquet")
    kwargs = {
        "split": "train",
        "validation_fraction": 0.01,
        "tag_balance_classes": {
            class_name: [tag] for class_name, tag in class_tags.items()
        },
        "tag_balance_target": 80,
        "tag_balance_seed": 91,
        "tag_balance_keep_unmatched": True,
        "tag_balance_rotate_each_epoch": True,
    }

    dataset = SupUpsDataset(str(tmp_path), **kwargs)
    repeated = SupUpsDataset(str(tmp_path), **kwargs)
    metadata = [dataset.get_metadata(index) for index in range(len(dataset))]
    balance_counts = {}
    for sample in metadata:
        balance_class = sample["balance_class"]
        balance_counts[balance_class] = balance_counts.get(balance_class, 0) + 1

    assert {name: balance_counts[name] for name in class_tags} == {
        name: 80 for name in class_tags
    }
    assert 15 <= balance_counts[None] <= 20
    assert all(
        sample["balance_class"] == "anime"
        for sample in metadata
        if "anime" in sample["tags"]
    )
    assert metadata == [
        repeated.get_metadata(index) for index in range(len(repeated))
    ]
    epoch_zero_ids = {sample["id"] for sample in metadata}
    dataset.set_epoch(1)
    repeated.set_epoch(1)
    epoch_one = [dataset.get_metadata(index) for index in range(len(dataset))]
    assert epoch_zero_ids != {sample["id"] for sample in epoch_one}
    assert epoch_one == [
        repeated.get_metadata(index) for index in range(len(repeated))
    ]
    assert {
        class_name: sum(sample["balance_class"] == class_name for sample in epoch_one)
        for class_name in class_tags
    } == {class_name: 80 for class_name in class_tags}


def test_image_loader_suppresses_bomb_warning_for_trusted_dataset(monkeypatch, tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (48, 32)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image = _open_rgb_image(path, draft_size=1024)

    assert image.mode == "RGB"
    assert not any(item.category is Image.DecompressionBombWarning for item in caught)
