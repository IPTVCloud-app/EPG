import json
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
 
"""
Generate thumbnails.json for path specification in API. 
"""

STREAMS_FILE = "streams.json"
OUTPUT_FILE = "thumbnails.json"

# absolute-style output path (as you requested)
THUMBNAIL_DIR = Path("thumbnails")


def get_image_metadata(path: Path):
    try:
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format

            return {
                "width": width,
                "height": height,
                "format": fmt,
                "quality": None  # not reliably extractable
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

        # ✅ EXACT filename (no slugify)
        file_path = THUMBNAIL_DIR / f"{channel}.webp"

        entry = {
            "channel": channel,
            "thumbnail_path": None,
            "width": None,
            "height": None,
            "format": None,
            "quality": None
        }

        if file_path.exists():
            entry["thumbnail_path"] = f"/thumbnails/{channel}.webp"

            meta = get_image_metadata(file_path)
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
