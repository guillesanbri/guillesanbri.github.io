#!/usr/bin/env bash
#
# Scaffold a new blog post:
#   _posts/YYYY-M-D-<slug>.md          front matter + skeleton
#   images/<folder>/index.png          1440x600 placeholder (home page card)
#   images/<folder>/thumbnail.jpg      1200x630 placeholder (og:image / twitter card;
#                                       regenerate from index.png with make-thumbnails.py)
#
# Usage:
#   ./new-post.sh "Post Title" [options]
#
# Options:
#   -s <slug>     filename slug          (default: derived from title)
#   -f <folder>   images/ folder name    (default: slug)
#   -t <tags>     comma-separated tags   (e.g. "CUDA, Deep Learning")
#   -d <date>     YYYY-MM-DD             (default: today)
#   -p            mark as pinned
#   -n            no table of contents
#   -h            show this help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INDEX_W=1440;  INDEX_H=600
THUMB_W=1200;  THUMB_H=630

usage() { sed -n '3,21p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[[ $# -ge 1 ]] || usage 1
[[ "$1" == "-h" || "$1" == "--help" ]] && usage 0

TITLE="$1"; shift
SLUG=""; FOLDER=""; TAGS=""; DATE="$(date +%F)"; PINNED=false; TOC=true

while getopts ":s:f:t:d:pnh" opt; do
  case "$opt" in
    s) SLUG="$OPTARG" ;;
    f) FOLDER="$OPTARG" ;;
    t) TAGS="$OPTARG" ;;
    d) DATE="$OPTARG" ;;
    p) PINNED=true ;;
    n) TOC=false ;;
    h) usage 0 ;;
    :) echo "Option -$OPTARG needs a value." >&2; exit 1 ;;
    *) echo "Unknown option -$OPTARG." >&2; usage 1 ;;
  esac
done

if [[ -z "$SLUG" ]]; then
  SLUG="$(printf '%s' "$TITLE" | iconv -f utf-8 -t ascii//TRANSLIT 2>/dev/null || printf '%s' "$TITLE")"
  SLUG="$(printf '%s' "$SLUG" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')"
fi
[[ -n "$SLUG" ]] || { echo "Could not derive a slug from the title; pass one with -s." >&2; exit 1; }
FOLDER="${FOLDER:-$SLUG}"

# Existing posts use non-padded dates in the filename (2026-5-2-...).
if ! date -d "$DATE" >/dev/null 2>&1; then
  echo "Invalid date: $DATE" >&2; exit 1
fi
FILE_DATE="$(date -d "$DATE" +%Y-%-m-%-d)"

POST="$ROOT/_posts/$FILE_DATE-$SLUG.md"
IMG_DIR="$ROOT/images/$FOLDER"

[[ -e "$POST" ]] && { echo "Post already exists: ${POST#$ROOT/}" >&2; exit 1; }
if compgen -G "$ROOT/_posts/*-$SLUG.md" >/dev/null; then
  echo "Warning: a post with slug '$SLUG' already exists (same permalink /$SLUG/)." >&2
fi

# Tags: "a, b,c" -> [a, b, c]
TAGS_YAML="$(printf '%s' "$TAGS" | sed -E 's/[[:space:]]*,[[:space:]]*/, /g; s/^[[:space:]]+|[[:space:]]+$//g')"

# Escape double quotes for the YAML title.
TITLE_YAML="${TITLE//\"/\\\"}"

cat > "$POST" <<EOF
---
layout: post
title: "$TITLE_YAML"
tags: [$TAGS_YAML]
folder: $FOLDER
pinned: $PINNED
toc: $TOC
# accent_dark: "#35A7FF"
# accent_light: "#eff2f7"
---
EOF

# Body is a quoted heredoc so $$ (math) and backticks stay literal;
# __FOLDER__ is substituted afterwards.
BODY="$(cat <<'EOF'

<!-- Only needed if the post uses math. -->

{% include mathjax.html %}

---

Intro paragraph. Link to [something](https://example.com), **bold**, *italics*, `inline code`.

---

# Section

Text.

## Subsection

<!-- ================================================================ -->

<!-- ELEMENT REFERENCE: copy what you need, delete the rest.          -->

<!-- ================================================================ -->

<!-- Callout (blockquote). Uses the post accent colors. -->

> A side note. Can hold **bold**, `code`, [links](https://example.com) and lists:
> - **First** item.
> - **Second** item.
>
> Closing sentence.

<!-- Green note box. -->

<div class="note">
<p>
Note text.
</p>
</div>

<!-- Red warning box. -->

<div class="warning">
<p>
Warning text.
</p>
</div>

<!-- Math: inline and display. Requires the mathjax include above. -->

Inline math like $$\alpha$$ or $$\hat{d_p}$$ sits within the sentence.

$$\text{DyT}(x) = \gamma * \text{tanh}(\alpha x) + \beta$$

$$
L(\hat{d}, d) = \frac{1}{n}\sum_{p} (\ln{\hat{d_p}} - \ln{d_p})^2 - \frac{\lambda}{n^2} \left( \sum_{p} (\ln{\hat{d_p}} - \ln{d_p}) \right)^2 \\
$$

<!-- Table (centered columns, bold the best result). -->

| Configuration | Metric A (ms) | Metric B (ms) |
|:-------------:|:-------------:|:-------------:|
| Baseline      | 0.112         | **0.053**     |
| Ours          | **0.075**     | 0.081         |

<!-- Code blocks. Highlighted languages: python, bash, c, cpp, diff. -->

```python
import torch

x = torch.randn(4, 4, device="cuda")
```

```bash
nvidia-smi
```

```cpp
matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
```

```diff
  auto start = chrono::steady_clock::now();
  matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, k, n);
+ cudaDeviceSynchronize();
- cout << duration.count() << "us" << endl;
+ cout << duration.count() / 1000.f << "ms" << endl;
```

<!-- Figure with caption. (Images point at the index.png placeholder; swap the filename.) -->

<figure align="center">
  <img src="./../images/__FOLDER__/index.png" alt="Description of the image."/>
  <figcaption>Figure 1: Caption.</figcaption>
</figure>

<!-- Figure with a border (screenshots, white-background diagrams). -->

<figure align="center">
  <img src="./../images/__FOLDER__/index.png" alt="Description of the image." style="border: 1px solid #ccc;"/>
</figure>

<!-- Image grid (Bootstrap rows/cols). -->

<div class="container">
  <div class="row">
    <div class="col"><img src="./../images/__FOLDER__/index.png" alt="Sample"/></div>
    <div class="col"><img src="./../images/__FOLDER__/index.png" alt="Sample"/></div>
    <div class="col"><img src="./../images/__FOLDER__/index.png" alt="Sample"/></div>
  </div>
</div>

<!-- Videos and embeds below are commented out because they need a real source to render.

Looping video (GIF replacement). Files live in /webm:
  ffmpeg -i in.gif -c:v libvpx-vp9 -b:v 0 -crf 35 -an -loop 0 webm/__FOLDER__.webm

<video src="./../webm/__FOLDER__.webm" style="width:100%;border-radius:0.25rem;" autoplay loop muted playsinline></video>

Video with controls:

<figure align="center">
  <video width="100%" controls>
    <source src="VIDEO_URL.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</figure>

Responsive embed (iframe):

<div style="width:100%;height:0;padding-bottom:56%;position:relative;"><iframe src="EMBED_URL" width="100%" height="100%" style="position:absolute;border-radius: 0.25rem;" frameBorder="0" allowFullScreen></iframe></div>

-->

<!-- ================================================================ -->

---

# Conclusion

Text.

---

# Related links

- [Link title](https://example.com)

---
EOF
)"
printf '%s\n' "${BODY//__FOLDER__/$FOLDER}" >> "$POST"

mkdir -p "$IMG_DIR"

# make_placeholder <path> <width> <height> <label>
make_placeholder() {
  local path="$1" w="$2" h="$3" label="$4"
  if [[ -e "$path" ]]; then
    echo "  skip    ${path#$ROOT/} (exists)"
    return
  fi
  if python3 -c "import PIL" 2>/dev/null; then
    python3 - "$path" "$w" "$h" "$label" <<'PY'
import sys
from PIL import Image, ImageDraw, ImageFont

path, w, h, label = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
img = Image.new("RGB", (w, h), (225, 229, 236))
d = ImageDraw.Draw(img)
d.rectangle([0, 0, w - 1, h - 1], outline=(150, 160, 175), width=4)
d.line([0, 0, w, h], fill=(200, 205, 215), width=2)
d.line([0, h, w, 0], fill=(200, 205, 215), width=2)

text = f"{label}\n{w} x {h}"
try:
    font = ImageFont.load_default(size=h // 8)
except TypeError:  # Pillow < 10.1
    font = ImageFont.load_default()
box = d.multiline_textbbox((0, 0), text, font=font, align="center")
tw, th = box[2] - box[0], box[3] - box[1]
pad = h // 20
x, y = (w - tw) / 2, (h - th) / 2
d.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(225, 229, 236))
d.multiline_text((x - box[0], y - box[1]), text, font=font, fill=(90, 100, 115), align="center")
img.save(path)
PY
  elif command -v magick >/dev/null || command -v convert >/dev/null; then
    "$(command -v magick || command -v convert)" -size "${w}x${h}" xc:'#e1e5ec' \
      -gravity center -fill '#5a6473' -pointsize $((h / 8)) \
      -annotate 0 "$label\n${w} x ${h}" "$path"
  else
    echo "  WARN    no Pillow or ImageMagick; create ${path#$ROOT/} (${w}x${h}) manually" >&2
    return
  fi
  echo "  create  ${path#$ROOT/}"
}

echo "  create  ${POST#$ROOT/}"
make_placeholder "$IMG_DIR/index.png"     "$INDEX_W" "$INDEX_H" "index.png"
make_placeholder "$IMG_DIR/thumbnail.jpg" "$THUMB_W" "$THUMB_H" "thumbnail.jpg"

echo
echo "Permalink: /$SLUG/"
