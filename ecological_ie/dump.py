import fnmatch
import unicodedata
import zipfile
from pathlib import Path


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def common_root(names: list[str]) -> str:
    first = names[0].split("/", 1)[0] + "/"
    return first if all(name.startswith(first) for name in names) else ""


class Dump:
    # Forsteinrichtungsoperate_combined as folder or zip; paths are NFC because the macOS-made zip stores NFD names.
    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_file() and zipfile.is_zipfile(self.path):
            self.zip = zipfile.ZipFile(self.path)
            names = [name for name in self.zip.namelist() if not name.endswith("/")]
            root = common_root(names)
            self.entries = {nfc(name[len(root):]): name for name in names}
        else:
            self.zip = None
            self.entries = {nfc(p.relative_to(self.path).as_posix()): p for p in self.path.rglob("*") if p.is_file()}

    def exists(self, relative: str) -> bool:
        return nfc(relative) in self.entries

    def read_bytes(self, relative: str) -> bytes:
        entry = self.entries[nfc(relative)]
        return self.zip.read(entry) if self.zip else entry.read_bytes()

    def read_text(self, relative: str) -> str:
        return self.read_bytes(relative).decode("utf-8")

    def size(self, relative: str) -> int:
        entry = self.entries[nfc(relative)]
        return self.zip.getinfo(entry).file_size if self.zip else entry.stat().st_size

    def glob(self, pattern: str) -> list[str]:
        return sorted(name for name in self.entries if fnmatch.fnmatchcase(name, nfc(pattern)))
