# Releasing stemflow

The source version lives in `stemflow/version.py`. GitHub, PyPI, documentation and conda-forge are separate destinations. Use the same reviewed, tested commit and version for the wheel/source release; a merge alone does not publish packages.

## Prepare and validate

1. Finalize `NEWS.md`, `docs/Releases/<version>.md`, the migration guide and `resources/RELEASE_NOTES_<version>.md`. Separate changes since the previous published release from changes already present in the main-branch baseline. Describe numerical changes and measured performance with their actual baselines.
2. Let the full pytest, DuckDB compatibility, cross-platform monitoring, documentation and release-package checks finish at the latest PR head. Do not substitute older green checks for current failures. The release install matrix covers Python 3.9–3.15; full pytest covers 3.11–3.15.
3. Build and inspect the artifacts locally if needed:

   ```bash
   python -m pip install build twine
   python -m build
   python -m twine check --strict dist/*
   python scripts/07_release/validate_distribution.py dist --tag v2.0.0
   ```

   The default build creates the wheel from an isolated source archive. The root README symlink is preserved in Git, but the source archive must contain a real README file. Cache files and the repository's top-level test package must not be installed. Remove stale `build/`, `dist/` and `stemflow.egg-info/` from your own previous builds before rebuilding; do not mix artifacts from different versions.

## Configure publishing once

In the existing stemflow project's [PyPI Publishing settings](https://pypi.org/manage/project/stemflow/settings/publishing/), add a GitHub Trusted Publisher:

| Setting | Value |
| --- | --- |
| Owner | `chenyangkang` |
| Repository | `stemflow` |
| Workflow filename | `publish-pypi.yml` |
| GitHub environment | `pypi` |

The publisher trusts the job identity; no long-lived PyPI token is stored in the repository. The repository administrator may configure approval rules for the GitHub `pypi` environment. Publishing cannot succeed until the PyPI project owner has configured this association. See [PyPI's instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Merge, tag and publish

1. Merge the reviewed PR after its checks pass. Verify main-branch pytest and documentation deployment before tagging. Preserve historical outputs and intentional scientific semantics described by the migration guide.
2. Fetch main, check out the exact release commit, and ensure no existing `v2.0.0` tag/release uses a different commit. Create and push the annotated tag:

   ```bash
   git tag -a v2.0.0 <verified-main-commit> -m 'stemflow 2.0.0'
   git push origin v2.0.0
   ```

3. `.github/workflows/publish-pypi.yml` verifies that the tag belongs to main and matches package metadata. It builds source/wheel artifacts, validates them, and runs installed-package checks outside the checkout. Core, optional tuning/monitoring and source installation must succeed across Python 3.9–3.15.
4. Only a pushed version tag in the upstream repository enables PyPI publication. PR and manual validation runs never publish. The workflow uploads the tested artifacts with Trusted Publishing, then creates a GitHub release using the same tag, files and prepared notes.
5. If publication fails, read the failed job before retrying. Rerun only failed jobs when earlier upload jobs already succeeded. Do not overwrite a published version with changed files; use a new patch version for a changed package.

## Update conda-forge

The retained recipe template updates dependency bounds and keeps optional XGBoost, Optuna and psutil out of the core runtime dependencies. Once PyPI contains the final source archive, generate its exact URL/checksum:

```bash
python scripts/07_release/prepare_conda_recipe.py --version 2.0.0 --output /path/to/stemflow-feedstock/recipe/meta.yaml
```

This command changes only the specified recipe file. It neither publishes nor pushes. Never submit a recipe with placeholder or locally guessed checksums. Review any conda-forge bot update instead of creating a duplicate version PR. For a manual update, use a fork, replace the recipe, rerender with conda-smithy when needed, and submit a PR to `conda-forge/stemflow-feedstock`. Preserve unrelated feedstock configuration. Build number resets to zero for the new version. See [conda-forge's update process](https://conda-forge.org/docs/maintainer/updating_pkgs/).

Conda's core package does not implement pip extras. Users needing the optional features can install `psutil xgboost optuna` from conda-forge alongside stemflow. Check that channel builds and clean-environment installs succeed before announcing conda availability.

## Verify the published result

Create separate fresh environments and test installation from the public channels:

```bash
python -m pip install 'stemflow[monitoring,tuning]==2.0.0'
python -m pip check
python -c "import stemflow; print(stemflow.__version__); print(stemflow.__file__)"
```

```bash
micromamba create -n stemflow_200 -c conda-forge python=3.11 stemflow=2.0.0 psutil xgboost optuna
micromamba run -n stemflow_200 python -c "import stemflow; print(stemflow.__version__); print(stemflow.__file__)"
```

Confirm API pages, executed examples, release links and installation guidance are live. Publish the final channel status with the migration notes. The 2.0.0 tag/release is distinct from publishing to PyPI or conda-forge; record any remaining blocked destination explicitly.
