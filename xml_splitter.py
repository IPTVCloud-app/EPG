#!/usr/bin/env python3
"""
xml_splitter.py

Current structure:
./sites/site_url/site.xml

If file > 20MB:
./sites/site_url/site_part_001.xml
./sites/site_url/site_part_002.xml
...

Original ./sites/site_url/site.xml is deleted after successful split.

Usage:
    python xml_splitter.py ./sites
"""

import sys
import os
import math
import xml.etree.ElementTree as ET
from collections import OrderedDict

LIMIT_MB = 20
LIMIT_BYTES = LIMIT_MB * 1024 * 1024


def mb(x):
    return round(x / 1024 / 1024, 2)


def count_programmes(xml_file):
    count = 0
    context = ET.iterparse(xml_file, events=("end",))

    for _, elem in context:
        if elem.tag == "programme":
            count += 1
        elem.clear()

    return count


def write_part(index, channels, programmes, folder, site):
    file = os.path.join(
        folder,
        f"{site}_part_{index:03d}.xml"
    )

    root = ET.Element("tv")

    for ch in channels.values():
        root.append(ch)

    for prog in programmes:
        root.append(prog)

    ET.ElementTree(root).write(
        file,
        encoding="utf-8",
        xml_declaration=True
    )

    print(f"   ✅ {os.path.basename(file)}")


def split_file(xml_file):
    folder = os.path.dirname(xml_file)
    site = os.path.splitext(os.path.basename(xml_file))[0]

    size = os.path.getsize(xml_file)

    if size <= LIMIT_BYTES:
        print(f"⏭️  Skip {site}.xml ({mb(size)} MB)")
        return

    print(f"🔍 Analyzing {xml_file} ({mb(size)} MB)")

    total_programmes = count_programmes(xml_file)

    if total_programmes == 0:
        print("❌ No programme entries")
        return

    parts = math.ceil(size / LIMIT_BYTES)
    chunk = math.ceil(total_programmes / parts)

    print(f"📦 Estimated parts : {parts}")
    print(f"📺 Programmes      : {total_programmes}")
    print(f"✂️  Chunk/file     : {chunk}")

    channels = OrderedDict()
    programmes = []
    part = 1

    context = ET.iterparse(xml_file, events=("end",))

    for _, elem in context:
        if elem.tag == "channel":
            cid = elem.attrib.get("id")
            if cid and cid not in channels:
                channels[cid] = ET.fromstring(
                    ET.tostring(elem)
                )

        elif elem.tag == "programme":
            programmes.append(
                ET.fromstring(ET.tostring(elem))
            )

            if len(programmes) >= chunk:
                used = OrderedDict()

                for prog in programmes:
                    cid = prog.attrib.get("channel")
                    if cid in channels:
                        used[cid] = channels[cid]

                write_part(part, used, programmes, folder, site)
                programmes.clear()
                part += 1

        elem.clear()

    if programmes:
        used = OrderedDict()

        for prog in programmes:
            cid = prog.attrib.get("channel")
            if cid in channels:
                used[cid] = channels[cid]

        write_part(part, used, programmes, folder, site)

    os.remove(xml_file)
    print(f"🗑️  Deleted original {site}.xml")
    print(f"✅ Finished {site}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python split_epg.py ./sites")
        sys.exit(1)

    root = sys.argv[1]

    if not os.path.isdir(root):
        print("Invalid directory")
        sys.exit(1)

    xml_files = []

    # scan ./sites/*/*.xml
    for site_dir in sorted(os.listdir(root)):
        full_dir = os.path.join(root, site_dir)

        if not os.path.isdir(full_dir):
            continue

        xml_file = os.path.join(full_dir, f"{site_dir}.xml")

        if os.path.isfile(xml_file):
            xml_files.append(xml_file)

    if not xml_files:
        print("No XML files found")
        return

    print(f"📂 Found {len(xml_files)} XML files\n")

    for file in xml_files:
        try:
            split_file(file)
        except Exception as e:
            print(f"❌ Failed {file}: {e}\n")


if __name__ == "__main__":
    main()