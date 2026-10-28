"""Reject packaging mistakes before publishing immutable release artifacts."""

from io import BytesIO
from pathlib import Path
import runpy
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
validate = runpy.run_path(str(ROOT / "scripts/07_release/validate_distribution.py"))["validate"]
render_recipe = runpy.run_path(str(ROOT / "scripts/07_release/prepare_conda_recipe.py"))["render"]


def archives(directory, wheel_extra=None, source_extra=None, readme_link=False):
    info = ("Metadata-Version: 2.1\nName: stemflow\nVersion: 2.0.0\n"
            "Requires-Python: >=3.9.0\nProvides-Extra: monitoring\nProvides-Extra: tuning\n")
    with zipfile.ZipFile(directory / "stemflow-2.0.0-py3-none-any.whl", "w") as wheel:
        wheel.writestr("stemflow-2.0.0.dist-info/METADATA", info)
        for name in ["stemflow/version.py", "stemflow/model/AdaSTEM.py",
                     "stemflow/calibration/__init__.py", "stemflow/tuning.py",
                     "stemflow/utils/execution_monitor.py"]:
            wheel.writestr(name, "# fixture\n")
        if wheel_extra:
            wheel.writestr(wheel_extra, "# unwanted file\n")
    with tarfile.open(directory / "stemflow-2.0.0.tar.gz", "w:gz") as source:
        for name in ["README.md", "LICENSE", "NEWS.md", "pyproject.toml", "requirements.txt", "PKG-INFO"]:
            data = info.encode() if name == "PKG-INFO" else b"fixture\n"
            member = tarfile.TarInfo(f"stemflow-2.0.0/{name}")
            if readme_link and name == "README.md":
                member.type, member.linkname = tarfile.SYMTYPE, "docs/index.md"
                source.addfile(member)
            else:
                member.size = len(data)
                source.addfile(member, BytesIO(data))
        if source_extra:
            member = tarfile.TarInfo(f"stemflow-2.0.0/{source_extra}")
            source.addfile(member, BytesIO())


def test_valid_archives_and_matching_tag(tmp_path):
    archives(tmp_path)
    assert validate(tmp_path, "v2.0.0") == "2.0.0"
    with pytest.raises(ValueError, match="does not match"):
        validate(tmp_path, "v2.0.1")


@pytest.mark.parametrize("extra", ["tests/__init__.py", "stemflow/__pycache__/version.pyc"])
def test_wheel_rejects_test_namespace_and_bytecode(tmp_path, extra):
    archives(tmp_path, wheel_extra=extra)
    with pytest.raises(ValueError):
        validate(tmp_path)


def test_source_readme_is_not_a_dangling_symlink(tmp_path):
    archives(tmp_path, readme_link=True)
    with pytest.raises(ValueError, match="regular file"):
        validate(tmp_path)


def test_source_rejects_compiled_caches(tmp_path):
    archives(tmp_path, source_extra="stemflow/__pycache__/version.pyc")
    with pytest.raises(ValueError, match="compiled Python"):
        validate(tmp_path)


def test_extra_archives_cannot_be_accidentally_uploaded(tmp_path):
    archives(tmp_path)
    (tmp_path / "stemflow-1.1.9.tar.gz").write_bytes(b"old build")
    with pytest.raises(ValueError, match="exactly one"):
        validate(tmp_path)


def test_feedstock_uses_published_checksum_and_guards_metadata():
    metadata = {"info": {"name": "stemflow", "version": "2.0.0"}, "urls": [dict(
        packagetype="sdist", filename="stemflow-2.0.0.tar.gz",
        url="https://files.pythonhosted.org/packages/example/stemflow-2.0.0.tar.gz",
        digests={"sha256": "a" * 64})]}
    recipe = render_recipe(metadata, "2.0.0")
    assert "sha256: " + "a" * 64 in recipe
    assert "duckdb >=1.1.3,<3" in recipe
    assert "numpy >=1.24.3\n" in recipe
    with pytest.raises(ValueError, match="does not match"):
        render_recipe(metadata, "2.0.1")
    metadata["urls"][0]["url"] = "https://example.org/unpublished.tar.gz"
    with pytest.raises(ValueError, match="hosted by PyPI"):
        render_recipe(metadata, "2.0.0")
