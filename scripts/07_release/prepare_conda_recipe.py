"""Generate a feedstock recipe from the exact published PyPI source artifact."""

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]


def render(metadata, version):
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Expected a stable three-component release version.")
    info = metadata["info"]
    if info["name"].lower() != "stemflow" or info["version"] != version:
        raise ValueError("PyPI metadata does not match the requested stemflow version.")
    sources = [record for record in metadata["urls"] if record["packagetype"] == "sdist"
               and record["filename"] == f"stemflow-{version}.tar.gz"]
    if len(sources) != 1:
        raise ValueError("Expected one published source archive.")
    source = sources[0]
    checksum = source["digests"]["sha256"]
    if not re.fullmatch(r"[a-f0-9]{64}", checksum):
        raise ValueError("Invalid source SHA256.")
    url = urlparse(source["url"])
    if url.scheme != "https" or url.hostname != "files.pythonhosted.org":
        raise ValueError("Expected an HTTPS source artifact hosted by PyPI.")
    text = (ROOT / "resources/configs/conda_forge/meta.yaml").read_text()
    return text.replace("@VERSION@", version).replace("@SHA256@", checksum).replace("@SOURCE_URL@", source["url"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2.0.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, help="Saved PyPI JSON for an offline preparation check.")
    args = parser.parse_args()
    if args.metadata:
        metadata = json.loads(args.metadata.read_text())
    else:
        if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
            parser.error("Expected a stable three-component release version.")
        with urlopen(f"https://pypi.org/pypi/stemflow/{args.version}/json", timeout=30) as response:
            metadata = json.load(response)
    recipe = render(metadata, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(recipe)
    print(f"Prepared {args.output} from PyPI metadata for stemflow {args.version}.")
    if args.metadata:
        print("Offline preparation only: verify against live PyPI metadata before submitting.")


if __name__ == "__main__":
    main()
