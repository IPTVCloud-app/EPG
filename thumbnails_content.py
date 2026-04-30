import json
import re
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

"""
Generate thumbnails.json for full path specification for the API.
"""

STREAMS_FILE = "streams.json"
OUTPUT_FILE = "thumbnails.json"
THUMBNAIL_DIR = Path("thumbnails")


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def get_image_metadata(path: Path):
    """
    Extract image metadata safely.
    """
    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format  # e.g. WEBP, PNG, JPG

            # Pillow does NOT reliably store "quality"
            # so we leave it null unless you embed it yourself later
            quality = None

            return {
                "width": width,
                "height": height,
                "format": fmt,
                "quality": quality
            }
    except Exception:
        return None


def main():
    with open(STREAMS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    streams = data.get("streams", [])

    thumbnails = []

    for item in streams:
        channel = item.get("channel")
        if not channel:
            continue

        slug = slugify(channel)
        path = THUMBNAIL_DIR / f"{slug}.webp"

        entry = {
            "channel": channel,
            "thumbnail_path": None,
            "width": None,
            "height": None,
            "format": None,
            "quality": None
        }

        if path.exists():
            meta = get_image_metadata(path)

            entry["thumbnail_path"] = str(path.as_posix())

            if meta:
                entry.update(meta)

        thumbnails.append(entry)

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(thumbnails),
        "thumbnails": thumbnails
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {OUTPUT_FILE} with {len(thumbnails)} entries.")


if __name__ == "__main__":
    main()
