"""Report tag and tag-combination frequencies for a configured SupUps split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from fm_train.config import load_config
from fm_train.data import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="SupUps training config")
    parser.add_argument(
        "--validation", action="store_true", help="Analyze validation_factory_args"
    )
    parser.add_argument("--top-combinations", type=int, default=30)
    parser.add_argument("--json", dest="json_path", help="Also write the full report as JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    factory_args = (
        config.data.validation_factory_args
        if args.validation
        else config.data.factory_args
    )
    dataset = build_dataset(config.data.factory, dict(factory_args))
    tags: Counter[str] = Counter()
    combinations: Counter[tuple[str, ...]] = Counter()
    balance_classes: Counter[str] = Counter()
    tagged_samples = 0
    for index in tqdm(range(len(dataset)), desc="Counting SupUps tags", unit="sample"):
        metadata = dataset.get_metadata(index)
        sample_tags = tuple(sorted({tag.casefold() for tag in metadata.get("tags", [])}))
        tags.update(sample_tags)
        combinations[sample_tags] += 1
        tagged_samples += bool(sample_tags)
        if metadata.get("balance_class") is not None:
            balance_classes[str(metadata["balance_class"])] += 1

    report = {
        "config": str(Path(args.config).resolve()),
        "split": "validation" if args.validation else "train",
        "samples": len(dataset),
        "tagged_samples": tagged_samples,
        "untagged_samples": len(dataset) - tagged_samples,
        "tags": [
            {
                "tag": tag,
                "count": count,
                "sample_fraction": count / max(len(dataset), 1),
            }
            for tag, count in tags.most_common()
        ],
        "combinations": [
            {"tags": list(combination), "count": count}
            for combination, count in combinations.most_common()
        ],
        "balance_classes": dict(balance_classes),
    }
    print(f"samples={report['samples']:,} tagged={tagged_samples:,}")
    print(f"{'tag':<20} {'count':>10} {'samples':>10}")
    for item in report["tags"]:
        print(
            f"{item['tag']:<20} {item['count']:>10,} "
            f"{item['sample_fraction']:>9.2%}"
        )
    print("\nMost common combinations:")
    for item in report["combinations"][: args.top_combinations]:
        label = ", ".join(item["tags"]) or "<untagged>"
        print(f"{item['count']:>10,}  {label}")
    if balance_classes:
        print("\nPrimary balance classes:")
        for class_name, count in balance_classes.items():
            print(f"{class_name:<20} {count:>10,}")

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON report to {path}")


if __name__ == "__main__":
    main()
