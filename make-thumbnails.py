#!/usr/bin/env python3
#
# Generate social-preview thumbnails (og:image / twitter:image) from each post's
# home page card:
#   images/<folder>/index.png  ->  images/<folder>/thumbnail.jpg
#
# The index image is center-cropped to 1.91:1, resized to 1200x630 and saved as
# JPEG, lowering quality until it fits under 300 KB (WhatsApp drops previews
# above roughly that; LinkedIn/X/Facebook allow 5-8 MB).
#
# Usage:
#   ./make-thumbnails.py                 all folders with an index.png
#   ./make-thumbnails.py tanh DPT        only these folders

import sys
from pathlib import Path

from PIL import Image

W, H = 1200, 630
MAX_BYTES = 300 * 1024
QUALITIES = range(88, 59, -4)

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"


def make_thumbnail(folder: Path) -> None:
    src, dst = folder / "index.png", folder / "thumbnail.jpg"
    im = Image.open(src).convert("RGBA")

    # Flatten transparency onto white, JPEG has no alpha.
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")

    # Center-crop to the target aspect ratio, then resize.
    w, h = im.size
    if w / h > W / H:
        cw = round(h * W / H)
        box = ((w - cw) // 2, 0, (w - cw) // 2 + cw, h)
    else:
        ch = round(w * H / W)
        box = (0, (h - ch) // 2, w, (h - ch) // 2 + ch)
    im = im.crop(box).resize((W, H), Image.LANCZOS)

    for q in QUALITIES:
        im.save(dst, "JPEG", quality=q, optimize=True, progressive=True)
        if dst.stat().st_size <= MAX_BYTES:
            break
    else:
        print(f"  WARN    {dst.relative_to(ROOT)} still over {MAX_BYTES // 1024} KB", file=sys.stderr)

    print(f"  create  {dst.relative_to(ROOT)}  ({w}x{h} -> {W}x{H}, q={q}, {dst.stat().st_size // 1024} KB)")


def main() -> None:
    if len(sys.argv) > 1:
        folders = [IMAGES / name for name in sys.argv[1:]]
    else:
        folders = sorted(p.parent for p in IMAGES.glob("*/index.png"))

    for folder in folders:
        if not (folder / "index.png").exists():
            print(f"  skip    {folder.relative_to(ROOT)} (no index.png)", file=sys.stderr)
            continue
        make_thumbnail(folder)


if __name__ == "__main__":
    main()
