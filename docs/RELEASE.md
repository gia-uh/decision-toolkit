# Release

The package name is `decision-toolkit` (PyPI/TestPyPI returned 404 when this
file was written, so the name was available). The first release is `0.1.0`.

## Local build and verification

```bash
python -m build
python -m twine check dist/*
```

The wheel must contain:

- `decision_toolkit/` with `engine/`, `adapters/`, `contracts/`, `py.typed`;
- the `dtk` entry point;
- `LICENSE` metadata.

The sdist additionally includes `docs/`, `bench/`, `tests/`, CI and governance
files.

Clean install test (no `PYTHONPATH`):

```bash
python -m venv /tmp/dtk-wheel
. /tmp/dtk-wheel/bin/activate
pip install dist/decision_toolkit-0.1.0-py3-none-any.whl
dtk --version
dtk models list --json
dtk doctor --json
```

With inference and server extras:

```bash
pip install "decision_toolkit-0.1.0-py3-none-any.whl[onnx,server,tracing]"
dtk decide --model gliner2.5-decide ...
dtk serve --model gliner2.5-decide --warmup --port 8016
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

1. Set the version in `pyproject.toml` and finalize `CHANGELOG.md`.
2. Re-run tests, benchmark smoke and `twine check`.
3. Commit the release preparation.
4. Create and push an annotated tag:

   ```bash
   git tag -a v0.1.0 -m "decision-toolkit 0.1.0"
   git push origin v0.1.0
   ```

5. `.github/workflows/release.yml` builds once, publishes to **TestPyPI**, then
   **PyPI**, then creates a GitHub Release. The `pypi` environment should require
   manual approval so a tag cannot publish unreviewed.
6. Verify TestPyPI before approval:

   ```bash
   python -m venv /tmp/dtk-testpypi
   . /tmp/dtk-testpypi/bin/activate
   pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple decision-toolkit
   dtk --version
   ```

7. After PyPI approval and publication:

   ```bash
   python -m venv /tmp/dtk-pypi
   . /tmp/dtk-pypi/bin/activate
   pip install decision-toolkit
   dtk --version
   dtk models list
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

## Pre-release status

The repository can build and install cleanly, but **no artifact has been
uploaded yet**. The actual upload requires the PyPI/TestPyPI trusted publisher
configuration and GitHub environment approval described above.
