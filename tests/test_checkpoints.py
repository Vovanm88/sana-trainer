import json

from fm_train.checkpoints import latest_checkpoint, load_export_step, rotate_checkpoints


def test_checkpoint_rotation_preserves_milestones(tmp_path):
    for step in range(7):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "milestone-1").mkdir()
    removed = rotate_checkpoints(tmp_path, 5)
    assert len(removed) == 2
    assert len(list(tmp_path.glob("checkpoint-*"))) == 5
    assert (tmp_path / "milestone-1").exists()


def test_latest_and_export_step_include_milestones(tmp_path):
    checkpoint = tmp_path / "checkpoint-2"
    milestone = tmp_path / "milestone-5"
    checkpoint.mkdir()
    milestone.mkdir()
    (milestone / "export_manifest.json").write_text(json.dumps({"global_step": 5}), encoding="utf-8")
    assert latest_checkpoint(tmp_path) == milestone
    assert load_export_step(milestone) == 5
