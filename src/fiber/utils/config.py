"""Config loading.

One rule, deliberately dumb: every file named in `defaults:` is merged at the
ROOT level of the config, and a key defined in two defaults files is an error
(silent precedence between config files is how experiments become unreproducible).
Keys written in the leaf config always win over the defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - attribute protocol
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    defaults = raw.pop("defaults", {}) or {}
    merged: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for alias, filename in defaults.items():
        sub_path = (path.parent / filename).resolve()
        with open(sub_path) as fh:
            sub = yaml.safe_load(fh) or {}
        if "defaults" in sub:
            raise ValueError(f"nested defaults are not supported ({sub_path})")
        for key, value in sub.items():
            if key in merged:
                raise ValueError(
                    f"config key {key!r} defined by both {provenance[key]} and "
                    f"{filename} (alias {alias!r}); resolve it explicitly"
                )
            merged[key] = value
            provenance[key] = filename

    merged.update(raw)  # leaf config wins
    merged["_config_path"] = str(path)
    merged["_defaults"] = {k: str((path.parent / v).resolve()) for k, v in defaults.items()}
    return _wrap(merged)
