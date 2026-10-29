"""Validate release archives before uploading them to a package index."""

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import tarfile
import zipfile


def validate(directory, tag=None):
    wheels = list(directory.glob("stemflow-*.whl"))
    sources = list(directory.glob("stemflow-*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source archive.")
    wheel, source = wheels[0], sources[0]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Wheel must contain one package metadata file.")
        info = BytesParser().parsebytes(archive.read(metadata[0]))
        if any(name.startswith("tests/") for name in names):
            raise ValueError("The wheel must not install the repository's tests package.")
        if any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in names):
            raise ValueError("Wheel contains compiled Python caches.")
        for module in ["stemflow/version.py", "stemflow/model/AdaSTEM.py",
                       "stemflow/calibration/__init__.py", "stemflow/tuning.py",
                       "stemflow/utils/execution_monitor.py"]:
            if module not in names:
                raise ValueError(f"Required module is absent from wheel: {module}")
        if not {"monitoring", "tuning"}.issubset(info.get_all("Provides-Extra", [])):
            raise ValueError("Optional dependency groups are missing from wheel metadata.")
        if info["Requires-Python"] != ">=3.9.0":
            raise ValueError("Review Python-support metadata before changing the release check.")
    version = info["Version"]
    if info["Name"] != "stemflow" or not version:
        raise ValueError("Unexpected package name or missing version.")
    if tag and tag != f"v{version}":
        raise ValueError(f"Tag {tag!r} does not match package version {version!r}.")
    notes = Path(__file__).resolve().parents[2] / "resources" / f"RELEASE_NOTES_{version}.md"
    if not notes.is_file() or not notes.read_text().strip():
        raise ValueError(f"Prepare the release notes before publishing: {notes.name}")
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        root = f"stemflow-{version}"
        if any("__pycache__" in name or name.endswith((".pyc", ".pyo")) for name in names):
            raise ValueError("Source archive contains compiled Python caches.")
        for name in ["README.md", "LICENSE", "NEWS.md", "pyproject.toml", "requirements.txt"]:
            member = archive.getmember(f"{root}/{name}")
            if not member.isfile() or member.size == 0:
                raise ValueError(f"Source archive needs a nonempty regular file: {name}")
        source_info = BytesParser().parsebytes(archive.extractfile(f"{root}/PKG-INFO").read())
        if source_info["Version"] != version:
            raise ValueError("Wheel and source versions differ.")
    for path in [source, wheel]:
        print(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name} ({path.stat().st_size:,} bytes)")
    return version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    version = validate(args.directory, args.tag)
    if args.github_output:
        with args.github_output.open("a") as stream:
            stream.write(f"version={version}\n")
    print(f"Validated stemflow {version} wheel and source archive.")


if __name__ == "__main__":
    main()
