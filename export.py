"""
NFL Analytics — Export Module

Export analysis results to CSV, JSON, or Markdown.

Functions:
  to_csv(df, path)         — Export DataFrame to CSV
  to_json(df, path)        — Export DataFrame to JSON (records orient)
  to_markdown(df, path)    — Export DataFrame to GitHub-flavored Markdown table
  full_report(...)         — Generate a complete analysis report in all formats

Usage:
  from export import to_csv, full_report
  to_csv(rankings_df, "output/power_rankings.csv")
  full_report(output_dir="output/", season=2025)
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)


# ── Single-format exports ────────────────────────────────────

def to_csv(df: pd.DataFrame, path: str | Path, index: bool = True) -> Path:
    """Export DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    log.info(f"Exported CSV: {path} ({len(df)} rows)")
    return path


def to_json(df: pd.DataFrame, path: str | Path, orient: str = "records") -> Path:
    """Export DataFrame to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient=orient, indent=2)
    log.info(f"Exported JSON: {path} ({len(df)} rows)")
    return path


def to_markdown(df: pd.DataFrame, path: str | Path, title: str = "") -> Path:
    """Export DataFrame to a GitHub-flavored Markdown table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if title:
        lines.append(f"# {title}\n")

    lines.append(df.to_markdown())
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Exported Markdown: {path} ({len(df)} rows)")
    return path


# ── Predictions export ───────────────────────────────────────

def predictions_to_csv(predictions: list[dict], path: str | Path) -> Path:
    """Export Elo predictions list to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(predictions)
    df.to_csv(path, index=False)
    log.info(f"Exported predictions CSV: {path} ({len(df)} games)")
    return path


def predictions_to_json(predictions: list[dict], path: str | Path) -> Path:
    """Export Elo predictions list to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    log.info(f"Exported predictions JSON: {path} ({len(predictions)} games)")
    return path


# ── Full Report ──────────────────────────────────────────────

def full_report(
    output_dir: str | Path = "output",
    season: int | None = None,
    formats: list[str] | None = None,
) -> dict[str, list[Path]]:
    """
    Generate a complete analysis report with all metrics.

    Args:
        output_dir: Directory for output files
        season: NFL season year (defaults to latest in data)
        formats: List of formats to export ("csv", "json", "md"). Defaults to all.

    Returns:
        Dict mapping format -> list of exported file paths
    """
    from analyze import load_pbp, load_schedules, team_efficiency, qb_analysis, power_rankings, build_elo_ratings

    if formats is None:
        formats = ["csv", "json", "md"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pbp = load_pbp()
    sched = load_schedules()

    if season is None:
        season = int(pbp["season"].max())

    log.info(f"Generating full report for {season} season")

    # Run all analyses
    eff = team_efficiency(pbp, season=season)
    qbs = qb_analysis(pbp, season=season)
    ranks = power_rankings(pbp, sched, season=season)

    # Elo predictions
    elo = build_elo_ratings(sched)
    upcoming = sched[(sched["season"] == season) & (sched["result"].isna())]
    predictions = []
    if len(upcoming) > 0:
        next_week = upcoming["week"].min()
        for _, g in upcoming[upcoming["week"] == next_week].iterrows():
            predictions.append(elo.predict(g["home_team"], g["away_team"]))

    exported = {fmt: [] for fmt in formats}
    timestamp = datetime.now().strftime("%Y%m%d")

    datasets = {
        f"team_efficiency_{season}": eff,
        f"qb_rankings_{season}": qbs,
        f"power_rankings_{season}": ranks,
    }

    for name, df in datasets.items():
        if "csv" in formats:
            exported["csv"].append(to_csv(df, out / f"{name}.csv"))
        if "json" in formats:
            exported["json"].append(to_json(df, out / f"{name}.json"))
        if "md" in formats:
            title = name.replace("_", " ").title()
            exported["md"].append(to_markdown(df, out / f"{name}.md", title=title))

    if predictions:
        if "csv" in formats:
            exported["csv"].append(predictions_to_csv(predictions, out / f"predictions_week{next_week}_{season}.csv"))
        if "json" in formats:
            exported["json"].append(predictions_to_json(predictions, out / f"predictions_week{next_week}_{season}.json"))

    # Summary metadata
    meta = {
        "generated_at": datetime.now().isoformat(),
        "season": season,
        "teams_analyzed": len(eff),
        "qbs_qualified": len(qbs),
        "games_predicted": len(predictions),
        "files": {fmt: [str(p) for p in paths] for fmt, paths in exported.items()},
    }
    meta_path = out / f"report_meta_{timestamp}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    total_files = sum(len(v) for v in exported.values()) + 1  # +1 for meta
    log.info(f"Full report complete: {total_files} files in {out}/")

    return exported
