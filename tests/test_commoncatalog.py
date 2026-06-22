from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from fm_train.example_dataset import CommonCatalogDataset


def test_commoncatalog_filters_and_splits(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "images").mkdir()
    rows = []
    for index in range(30):
        rel_path = f"images/{index}.png"
        Image.new("RGB", (32, 32)).save(tmp_path / rel_path)
        rows.append({"rel_path": rel_path, "blip2_caption": f"caption {index}", "decode_ok": True,
                     "is_bad": index == 0, "is_synthetic": False})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "metadata/part-000000.parquet")
    train = CommonCatalogDataset(str(tmp_path), split="train", validation_fraction=0.2)
    validation = CommonCatalogDataset(str(tmp_path), split="validation", validation_fraction=0.2)
    train_ids = {train.get_metadata(i)["id"] for i in range(len(train))}
    validation_ids = {validation.get_metadata(i)["id"] for i in range(len(validation))}
    assert train_ids.isdisjoint(validation_ids)
    assert len(train) + len(validation) == 29
    assert train[0]["image"].size == (32, 32)
