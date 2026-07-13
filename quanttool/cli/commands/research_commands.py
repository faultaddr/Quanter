"""Commands for Serenity research scorecards."""

import json
from typing import Any, Tuple

import click
import typer


app = typer.Typer(help="Serenity research scorecards")


def _serenity_dependencies() -> Tuple[Any, Any, Any]:
    """Load the Serenity application boundary only when a command runs."""

    from pydantic import ValidationError

    from quanttool.application.serenity_service import SerenityService
    from quanttool.domain.models.serenity import SerenityScorecard

    return SerenityService, SerenityScorecard, ValidationError


def _load_input(input_path: str) -> str:
    """Read a UTF-8 JSON document from a path or standard input."""

    try:
        if input_path == "-":
            return click.get_text_stream("stdin", encoding="utf-8").read()

        with open(input_path, "r", encoding="utf-8") as input_file:
            return input_file.read()
    except (OSError, UnicodeError) as exc:
        raise click.ClickException(
            "Could not read input '{}': {}".format(input_path, exc)
        ) from exc


def _parse_scorecard(input_path: str) -> Any:
    """Parse and validate a Serenity scorecard document."""

    try:
        payload = json.loads(_load_input(input_path))
    except json.JSONDecodeError as exc:
        raise click.ClickException("Invalid JSON: {}".format(exc.msg)) from exc

    _, scorecard_model, validation_error = _serenity_dependencies()
    try:
        model_validate = getattr(scorecard_model, "model_validate", None)
        if model_validate is not None:
            return model_validate(payload)
        return scorecard_model.parse_obj(payload)
    except (validation_error, TypeError) as exc:
        raise click.ClickException("Invalid scorecard: {}".format(exc)) from exc


def _json_output(model: Any) -> str:
    """Serialize a Pydantic v1 or v2 model as UTF-8 JSON."""

    model_dump = getattr(model, "model_dump", None)
    if model_dump is not None:
        payload = model_dump(mode="json")
    else:
        payload = json.loads(model.json())
    return json.dumps(payload, ensure_ascii=False, indent=2)


@app.command()
def template() -> None:
    """Print a valid Serenity scorecard JSON template."""

    serenity_service, _, _ = _serenity_dependencies()
    typer.echo(_json_output(serenity_service().template()))


@app.command()
def scorecard(
    input_path: str = typer.Argument(..., metavar="INPUT", help="JSON file path or - for stdin"),
    output_format: str = typer.Option(
        "json",
        "--format",
        help="Output format: json, md, or both",
    ),
) -> None:
    """Score a Serenity research JSON document."""

    if output_format not in {"json", "md", "both"}:
        raise click.ClickException(
            "Unsupported format '{}'. Use json, md, or both.".format(output_format)
        )

    serenity_service, _, _ = _serenity_dependencies()
    service = serenity_service()
    result = service.score(_parse_scorecard(input_path))
    json_result = _json_output(result)

    if output_format == "json":
        typer.echo(json_result)
    elif output_format == "md":
        typer.echo(service.to_markdown(result))
    else:
        typer.echo("{}\n\n---\n\n{}".format(json_result, service.to_markdown(result)))


__all__ = ["app"]
