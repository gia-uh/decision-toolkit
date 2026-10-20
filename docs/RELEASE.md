# Release

The package name is `decision-toolkit` (PyPI/TestPyPI returned 404 when this
file was written, so the name was available). The first release was `0.1.0`;
`0.1.1` was a metadata-only refresh; `0.1.2` validated Trusted Publishing;
`0.1.3` added release verification and docs publication; `0.1.4` migrated
package management and builds to uv but was published to TestPyPI only; `0.1.5`
fixes TestPyPI verification for uv.

## Local build and verification

The repository uses uv for every environment and build step. `uv.lock` must be
current before tagging.

```bash
uv sync --frozen
uv build
uv run --no-sync twine check dist/*
```

The wheel must contain:

- `decision_toolkit/` with `engine/`, `adapters/`, `contracts/`, `py.typed`;
- the `dtk` entry point;
- `LICENSE` metadata.

The sdist additionally includes `docs/`, `bench/`, `tests/`, CI and governance
files.

Clean install test (no `PYTHONPATH`):

```bash
uv venv --python 3.12 /tmp/dtk-wheel
uv pip install --python /tmp/dtk-wheel/bin/python \
  dist/decision_toolkit-0.1.5-py3-none-any.whl
uv pip check --python /tmp/dtk-wheel/bin/python
/tmp/dtk-wheel/bin/dtk --version
/tmp/dtk-wheel/bin/dtk models list --json
/tmp/dtk-wheel/bin/dtk doctor --json
```

With inference and server extras:

```bash
uv pip install --python /tmp/dtk-wheel/bin/python \
  "dist/decision_toolkit-0.1.5-py3-none-any.whl[onnx,server,tracing]"
/tmp/dtk-wheel/bin/dtk decide --model gliner2.5-decide ...
/tmp/dtk-wheel/bin/dtk serve --model gliner2.5-decide --warmup --port 8016
curl -s http://127.0.0.1:8016/health
```

## Trusted publishing

Configure two GitHub environments:

| Environment | Use |
| --- | --- |
| `testpypi` | TestPyPI publish |
| `pypi` | real PyPI publish; add required reviewers |

For each environment, configure a PyPI trusted publisher with:

- owner: `gia-uh`
- repository: `decision-toolkit`
- workflow: `release.yml`
- environment: `testpypi` or `pypi`

No long-lived PyPI token is needed; the workflow uses GitHub OIDC
(`id-token: write`).

## Release flow

1. Set the version in `pyproject.toml`, run `uv lock` to refresh `uv.lock`,
   and finalize `CHANGELOG.md`. Commit both files together.
2. Re-run `uv sync --frozen`, tests, benchmark smoke and `uv run --no-sync twine check`.
3. Commit the release preparation.
4. Create and push an annotated tag:

   ```bash
   git tag -a v0.1.5 -m "decision-toolkit 0.1.5"
   git push origin v0.1.5
   ```

5. `.github/workflows/release.yml` builds once, publishes to **TestPyPI**, then
   **PyPI**, then creates a GitHub Release. The `pypi` environment should require
   manual approval so a tag cannot publish unreviewed.
6. Verify TestPyPI before approval:

   ```bash
   uv venv --python 3.12 /tmp/dtk-testpypi
   uv pip install --python /tmp/dtk-testpypi/bin/python \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple \
     --index-strategy unsafe-best-match decision-toolkit
   /tmp/dtk-testpypi/bin/dtk --version
   ```

   PyPI and TestPyPI simple indexes may be cached for up to ten minutes. If the
   exact version is not visible yet, install the wheel URL from
   `https://pypi.org/pypi/decision-toolkit/<version>/json` (or the TestPyPI
   equivalent). The release workflow does this automatically.

7. After PyPI approval and publication:

   ```bash
   uv venv --python 3.12 /tmp/dtk-pypi
   uv pip install --python /tmp/dtk-pypi/bin/python decision-toolkit
   /tmp/dtk-pypi/bin/dtk --version
   /tmp/dtk-pypi/bin/dtk models list
   ```

## Version policy

- Semver starts at `0.1.0`; breaking changes before `1.0.0` still require a
  minor version bump and a changelog note.
- A published version is immutable: never re-upload or reuse a tag. Use a new
  patch version for fixes.
- Keep catalog/model changes separate from runtime fixes in release notes where
  possible.
- `dtk --version` reads the installed distribution version via
  `decision_toolkit.__version__`.

## Manual TestPyPI-only run

From GitHub Actions, run `release` with `target=testpypi`. It builds and uploads
to TestPyPI without publishing to PyPI or creating a GitHub release.

## Published release

`0.1.5` is published through Trusted Publishing on 2026-10-20:

<https://pypi.org/project/decision-toolkit/0.1.5/>

`0.1.0` was the first release; `0.1.1` refreshed the PyPI description; `0.1.2`
validated the OIDC TestPyPI → PyPI flow; `0.1.3` added post-publish install
verification and the GitHub Pages documentation site; `0.1.4` moved the
development, build, CI, release and documentation environments to uv but was
published to TestPyPI only; `0.1.5` fixes TestPyPI resolution during release
verification.

The tag-triggered workflow remains the preferred path for future releases once
the TestPyPI/PyPI trusted publishers and protected environments are configured.
