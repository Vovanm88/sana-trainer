from pathlib import Path

import pytest

from fm_train.config import Config, load_config


def test_example_configs_are_valid():
    root = Path(__file__).parents[1]
    assert load_config(root / "configs/cpt.yaml").training.profile == "cpt"
    assert load_config(root / "configs/sft.yaml").deepspeed.stage == 3


def test_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  typo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(path)


def test_rejects_invalid_ratios():
    config = Config()
    config.lr.warmup_ratio = 0.9
    config.lr.cooldown_ratio = 0.2
    with pytest.raises(ValueError, match="plateau phase"):
        config.validate()

