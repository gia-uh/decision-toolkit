"""``python -m decision_toolkit`` is the same entry point as the ``dtk`` script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
