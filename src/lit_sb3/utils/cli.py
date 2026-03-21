"""Command-line interface argument parsing helpers."""

import argparse
from typing import Any


class StoreDict(argparse.Action):
    """Custom ``argparse`` action for storing key-value pairs in a dict.

    Enables ``key1:val1 key2:val2`` syntax on the CLI.
    Numeric strings are auto-converted to ``int`` or ``float``.
    Comma-separated values become lists.
    """

    def __init__(self, option_strings: list[str], dest: str, nargs: str = "+", **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | list[str] | None,
        option_string: str | None = None,
    ) -> None:  # ty:ignore[invalid-method-override]
        if values is None:
            return
        args: dict[str, Any] = {}
        for item in values:
            k, v = item.split(":")
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    if isinstance(v, str) and "," in v:
                        v = v.split(",")
            args[k] = v
        setattr(namespace, self.dest, args)
