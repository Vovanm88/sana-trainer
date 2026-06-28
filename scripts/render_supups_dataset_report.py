from __future__ import annotations

import argparse
import importlib
import random
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml


PAGE_W = 1240
PAGE_H = 1754
MARGIN = 54
GAP = 24


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(30, bold=True)
FONT_META = load_font(21)
FONT_LABEL = load_font(22, bold=True)
FONT_TEXT = load_font(20)
FONT_SMALL = load_font(17)


def import_object(dotted: str) -> Any:
    module_name, object_name = dotted.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def text_height(lines: list[str], font: ImageFont.ImageFont, spacing: int) -> int:
    if not lines:
        return 0
    probe = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(probe)
    total = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        total += bbox[3] - bbox[1] + spacing
    return total - spacing


def wrap_text(text: object, width_chars: int, max_lines: int) -> list[str]:
    if text is None:
        return ["<missing>"]
    normalized = " ".join(str(text).split())
    if not normalized:
        return ["<empty>"]
    lines = textwrap.wrap(
        normalized,
        width=width_chars,
        break_long_words=True,
        replace_whitespace=True,
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    label: str,
    text: object,
    width_chars: int,
    max_lines: int,
) -> int:
    x, y = xy
    draw.text((x, y), label, fill=(20, 23, 28), font=FONT_LABEL)
    y += 30
    lines = wrap_text(text, width_chars, max_lines)
    for line in lines:
        draw.text((x, y), line, fill=(40, 44, 52), font=FONT_TEXT)
        y += 26
    return y + 14


def make_page(sample: dict[str, Any], image: Image.Image, number: int, total: int, source_index: int) -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), (250, 250, 248))
    draw = ImageDraw.Draw(page)

    title = f"SupUps SFT dataset sample {number}/{total}"
    draw.text((MARGIN, 36), title, fill=(18, 20, 24), font=FONT_TITLE)
    min_max = max(float(sample["photo_score"]), float(sample["general_score"]))
    meta = (
        f"index={source_index}  min_max={min_max:.4f}  "
        f"photo={float(sample['photo_score']):.4f}  general={float(sample['general_score']):.4f}  "
        f"bad_pool={bool(sample['is_bad_pool'])}"
    )
    draw.text((MARGIN, 78), meta, fill=(59, 64, 73), font=FONT_META)

    id_lines = wrap_text(sample["id"], 105, 2)
    y = 110
    for line in id_lines:
        draw.text((MARGIN, y), line, fill=(80, 84, 92), font=FONT_SMALL)
        y += 22

    image_area_top = y + GAP
    image_area_h = 850
    image_area_w = PAGE_W - 2 * MARGIN
    preview = image.copy()
    preview.thumbnail((image_area_w, image_area_h), Image.Resampling.LANCZOS)
    img_x = MARGIN + (image_area_w - preview.width) // 2
    img_y = image_area_top + (image_area_h - preview.height) // 2
    page.paste(preview, (img_x, img_y))
    draw.rectangle(
        (img_x - 1, img_y - 1, img_x + preview.width, img_y + preview.height),
        outline=(214, 216, 220),
        width=1,
    )

    y = image_area_top + image_area_h + 34
    y = draw_wrapped(draw, (MARGIN, y), "very_short", sample["caption_very_short"], 112, 3)
    y = draw_wrapped(draw, (MARGIN, y), "short", sample["caption_short"], 112, 5)
    draw_wrapped(draw, (MARGIN, y), "long", sample["caption_long"], 112, 10)
    return page


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/supups_v1_sft.yaml")
    parser.add_argument("--out", default="outputs/supups-v1-sft/dataset_sample_report.pdf")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    factory = import_object(config["data"]["factory"])
    dataset = factory(dict(config["data"]["factory_args"]))
    if len(dataset) < args.count:
        raise RuntimeError(f"Dataset has only {len(dataset)} samples, requested {args.count}")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), args.count)
    pages: list[Image.Image] = []
    for page_number, index in enumerate(indices, start=1):
        sample = dataset.get_metadata(index)
        image = dataset[index]["image"]
        pages.append(make_page(sample, image, page_number, args.count, index))
        print(f"{page_number:03d}/{args.count} index={index} id={sample['id']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(
        out,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=150.0,
        quality=85,
    )
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()
