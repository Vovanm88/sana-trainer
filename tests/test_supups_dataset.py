import warnings

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fm_train.example_dataset import SupUpsDataset, _open_rgb_image


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


def test_image_loader_suppresses_bomb_warning_for_trusted_dataset(monkeypatch, tmp_path):
    path = tmp_path / "large.png"
    Image.new("RGB", (48, 32)).save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        image = _open_rgb_image(path, draft_size=1024)

    assert image.mode == "RGB"
    assert not any(item.category is Image.DecompressionBombWarning for item in caught)
