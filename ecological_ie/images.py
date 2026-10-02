import base64
import html
import io
import re
from pathlib import Path

from PIL import Image

from .dump import Dump, nfc
from .toc import PageRef

IMG_TAG = re.compile(r"<img\b[^>]*>")
SCAN_ALT = re.compile(r'alt="Scan of ([^"]*)"')
DATA_SRC = re.compile(r'src="data:image/([a-z]+);base64,([A-Za-z0-9+/=]+)"')
PAGE_ID = re.compile(r"^\[(\d+)\].*?\b(A \d+(?: [IVX]+)?)\s*$")
SCAN_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def page_key(page_id: str) -> tuple[str, int] | None:
    match = PAGE_ID.match(nfc(page_id).strip())
    return (match.group(2), int(match.group(1))) if match else None


def viewer_images(viewer_html: str) -> dict[str, str]:
    images = {}
    for tag in IMG_TAG.finditer(viewer_html):
        alt, src = SCAN_ALT.search(tag.group(0)), DATA_SRC.search(tag.group(0))
        if alt and src:
            images[nfc(html.unescape(alt.group(1)))] = src.group(2)
    return images


def index_scans(scans_dir: Path) -> dict:
    index = {}
    for path in sorted(Path(scans_dir).rglob("*")):
        if path.suffix.lower() in SCAN_SUFFIXES and not path.name.startswith("."):
            index.setdefault(nfc(path.stem), path)
            index.setdefault(page_key(path.stem), path)
    index.pop(None, None)
    return index


def as_jpeg(data: bytes, max_side: int) -> tuple[bytes, tuple[int, int]]:
    image = Image.open(io.BytesIO(data))
    if image.format == "JPEG" and max(image.size) <= max_side:
        return data, image.size
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    out = io.BytesIO()
    image.save(out, "JPEG", quality=90)
    return out.getvalue(), image.size


class ImageResolver:
    def __init__(self, dump: Dump, scans_dir: Path | None = None, max_side: int = 3072):
        self.dump = dump
        self.scans = index_scans(scans_dir) if scans_dir else {}
        self.max_side = max_side
        self.viewers: dict[str, dict] = {}

    def viewer(self, run: str) -> dict:
        if run not in self.viewers:
            path = f"{run}/viewer.html"
            images = viewer_images(self.dump.read_text(path)) if self.dump.exists(path) else {}
            images.update({page_key(pid): data for pid, data in list(images.items())})
            images.pop(None, None)
            self.viewers[run] = images
        return self.viewers[run]

    def find(self, page: PageRef) -> tuple[bytes, tuple[int, int], str] | None:
        key = (page.sig, page.num)
        scan = self.scans.get(nfc(page.pid)) or self.scans.get(key)
        if scan:
            data, size = as_jpeg(scan.read_bytes(), self.max_side)
            return data, size, "scan"
        images = self.viewer(page.run)
        encoded = images.get(nfc(page.pid)) or images.get(key)
        if encoded:
            data, size = as_jpeg(base64.b64decode(encoded), self.max_side)
            return data, size, "viewer"
        return None
