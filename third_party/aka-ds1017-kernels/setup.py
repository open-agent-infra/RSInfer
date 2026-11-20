"""Keep editable source trees in Git and deduplicate identical headers in the wheel."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

from setuptools import setup
from setuptools.command.build_py import build_py


def source_files(root):
    """Restore each verified variant from shared upstream files and its own changes."""
    layout = json.loads((root / "layout.json").read_text())
    for variant, layers in sorted(layout.items()):
        paths = {}
        for layer in layers:
            directory = root / layer
            for path in directory.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                key = path.relative_to(directory)
                if key in paths:
                    raise ValueError(f"Duplicate source {variant}/{key}")
                paths[key] = path
        for key, path in sorted(paths.items()):
            yield f"{variant}/{key.as_posix()}", path


class BuildKernels(build_py):
    def run(self):
        super().run()
        package = Path(self.build_lib) / "aka_ds1017"
        sources = Path(__file__).parent / "native"
        seen = {}
        archive = package / "kernels.tar.gz"
        with archive.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for name, path in source_files(sources):
                    data = path.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    entry = tarfile.TarInfo(name)
                    entry.mode = 0o644
                    if digest in seen:
                        entry.type = tarfile.LNKTYPE
                        entry.linkname = seen[digest]
                        tar.addfile(entry)
                    else:
                        entry.size = len(data)
                        tar.addfile(entry, io.BytesIO(data))
                        seen[digest] = name
        proof = package / "provenance.json"
        record = json.loads(proof.read_text())
        record["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        proof.write_text(json.dumps(record, indent=2) + "\n")


setup(cmdclass={"build_py": BuildKernels})
