"""CLI entry point for the LoL stat tracker MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from lol_stat_tracker.features import build_dataset
from lol_stat_tracker.ingest import ingest_matches
from lol_stat_tracker.insights import build_last_game_report, build_weekly_summary
from lol_stat_tracker.train import train_win_model


def cmd_ingest(args: argparse.Namespace) -> None:
    saved = ingest_matches(
        game_name=args.game_name,
        tag_line=args.tag_line,
        region=args.region,
        api_key=args.api_key,
        count=args.count,
    )
    print(f"Ingest complete. New matches downloaded: {saved}")


def cmd_build_dataset(_: argparse.Namespace) -> None:
    output_path = build_dataset()
    print(f"Dataset written: {output_path}")


def cmd_train(_: argparse.Namespace) -> None:
    metrics_path = train_win_model()
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    print(f"Model trained. Metrics saved to: {metrics_path}")
    print(json.dumps(metrics, indent=2))


def cmd_report(_: argparse.Namespace) -> None:
    last_game_path = build_last_game_report()
    weekly_path = build_weekly_summary()
    print(f"Last game report: {last_game_path}")
    print(f"Weekly summary: {weekly_path}")


def cmd_serve_api(args: argparse.Namespace) -> None:
    uvicorn.run("lol_stat_tracker.api:app", host=args.host, port=args.port, reload=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoL Coach Tracker MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Download match data from Riot API")
    ingest_parser.add_argument("--game-name", required=True, help="Riot game name")
    ingest_parser.add_argument("--tag-line", required=True, help="Riot tagline")
    ingest_parser.add_argument("--region", default="americas", choices=["americas", "europe", "asia"])
    ingest_parser.add_argument("--api-key", default=None, help="Override RIOT_API_KEY env value")
    ingest_parser.add_argument("--count", type=int, default=100, help="Number of recent matches to fetch")
    ingest_parser.set_defaults(func=cmd_ingest)

    dataset_parser = subparsers.add_parser("build-dataset", help="Build processed matches CSV")
    dataset_parser.set_defaults(func=cmd_build_dataset)

    train_parser = subparsers.add_parser("train", help="Train Random Forest win model")
    train_parser.set_defaults(func=cmd_train)

    report_parser = subparsers.add_parser("report", help="Generate markdown coaching reports")
    report_parser.set_defaults(func=cmd_report)

    api_parser = subparsers.add_parser("serve-api", help="Run FastAPI backend for desktop app")
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.set_defaults(func=cmd_serve_api)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
