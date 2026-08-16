"""GraphRoute-TS command-line entry point.

Thin Typer app. During the setup phase it only exposes read-only inspection
commands. It must never trigger downloads or training as a side effect.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console

from scalerag import __version__
from scalerag.reproducibility import RunContext

app = typer.Typer(
    add_completion=False,
    help="GraphRoute-TS command-line interface (setup-phase scaffolding).",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"scalerag {__version__}")


@app.command()
def env(json_out: bool = typer.Option(False, "--json", help="Emit JSON.")) -> None:
    """Print a best-effort runtime/hardware fingerprint (read-only)."""
    ctx = RunContext().to_dict()
    if json_out:
        console.print_json(json.dumps(ctx))
    else:
        for key, value in ctx.items():
            console.print(f"[bold]{key}[/bold]: {value}")


if __name__ == "__main__":
    app()
