#!/usr/bin/env python3
"""
NFL Analytics — Unified CLI

Commands:
  fetch     — Download/refresh data from nfl_data_py
  teams     — Team efficiency rankings
  qbs       — QB analysis and rankings
  power     — Composite power rankings
  predict   — Elo-based game predictions
  elo       — Team Elo rating lookup
  viz       — Generate visualization charts
  export    — Export analysis to CSV/JSON/Markdown
  report    — Full analysis report (all formats)

Usage:
  python cli.py fetch --seasons 2023-2025
  python cli.py teams --season 2025 --top 10
  python cli.py qbs --season 2025
  python cli.py power --season 2025
  python cli.py predict --season 2025
  python cli.py elo KC BUF SF
  python cli.py viz scatter --season 2025 -o charts/
  python cli.py export power --season 2025 --format csv -o output/
  python cli.py report --season 2025 -o output/
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nfl-cli")


def cmd_fetch(args):
    """Download data from nfl_data_py."""
    from fetch_data import fetch_all, parse_seasons

    if args.seasons:
        seasons = parse_seasons(args.seasons)
    elif args.season:
        seasons = [args.season]
    else:
        y = datetime.now().year
        seasons = [y - 2, y - 1, y]

    fetch_all(seasons)


def cmd_teams(args):
    """Team efficiency analysis."""
    from analyze import load_pbp, team_efficiency

    pbp = load_pbp()
    season = args.season or int(pbp["season"].max())
    eff = team_efficiency(pbp, season=season)

    cols = ["plays", "epa_per_play", "epa_per_play_def", "success_rate", "explosive_rate", "pass_rate_early"]
    available = [c for c in cols if c in eff.columns]

    print(f"\n{'='*60}")
    print(f"  TEAM EFFICIENCY — {season} Season")
    print(f"{'='*60}\n")
    print(eff[available].head(args.top).to_string())
    print()


def cmd_qbs(args):
    """QB analysis and rankings."""
    from analyze import load_pbp, qb_analysis

    pbp = load_pbp()
    season = args.season or int(pbp["season"].max())
    qbs = qb_analysis(pbp, season=season, min_dropbacks=args.min_dropbacks)

    cols = ["team", "dropbacks", "epa_per_dropback", "cpoe", "comp_pct", "td_rate", "int_rate", "adot"]
    available = [c for c in cols if c in qbs.columns]

    print(f"\n{'='*60}")
    print(f"  QB RANKINGS — {season} Season (min {args.min_dropbacks} dropbacks)")
    print(f"{'='*60}\n")
    print(qbs[available].head(args.top).to_string())
    print()


def cmd_power(args):
    """Power rankings."""
    from analyze import load_pbp, load_schedules, power_rankings

    pbp = load_pbp()
    sched = load_schedules()
    season = args.season or int(pbp["season"].max())
    ranks = power_rankings(pbp, sched, season=season)

    print(f"\n{'='*60}")
    print(f"  POWER RANKINGS — {season} Season")
    print(f"{'='*60}\n")
    print(ranks.to_string())
    print()


def cmd_predict(args):
    """Elo game predictions."""
    from analyze import load_schedules, build_elo_ratings

    sched = load_schedules()
    season = args.season or int(sched["season"].max())
    elo = build_elo_ratings(sched)

    upcoming = sched[(sched["season"] == season) & (sched["result"].isna())]

    if len(upcoming) == 0:
        print(f"\nNo upcoming games found for {season} season.")
        return

    next_week = upcoming["week"].min()
    next_games = upcoming[upcoming["week"] == next_week]

    print(f"\n{'='*60}")
    print(f"  WEEK {next_week} PREDICTIONS — {season} Season")
    print(f"{'='*60}\n")

    predictions = []
    for _, g in next_games.iterrows():
        pred = elo.predict(g["home_team"], g["away_team"])
        predictions.append(pred)
        fav = pred["home"] if pred["home_win_prob"] > 0.5 else pred["away"]
        spread_str = f"{pred['spread']:+.1f}" if pred["spread"] != 0 else "PICK"
        wp = max(pred["home_win_prob"], pred["away_win_prob"])
        print(f"  {pred['away']:>3} @ {pred['home']:<3}  →  {fav} {spread_str}  ({wp:.1%})")

    print()
    return predictions


def cmd_elo(args):
    """Elo rating lookup for specific teams."""
    from analyze import load_schedules, build_elo_ratings

    sched = load_schedules()
    elo = build_elo_ratings(sched)

    print(f"\n{'='*40}")
    print(f"  ELO RATINGS")
    print(f"{'='*40}\n")

    for team in args.teams:
        t = team.upper()
        rating = elo.get_rating(t)
        diff = rating - 1500
        marker = "+" if diff >= 0 else ""
        print(f"  {t:>3}:  {rating:.0f}  ({marker}{diff:.0f})")
    print()


def cmd_viz(args):
    """Generate visualization charts."""
    from analyze import load_pbp, load_schedules, team_efficiency, qb_analysis, power_rankings, build_elo_ratings
    import viz

    pbp = load_pbp()
    sched = load_schedules()
    season = args.season or int(pbp["season"].max())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    chart_type = args.chart

    if chart_type in ("scatter", "all"):
        eff = team_efficiency(pbp, season=season)
        fig = viz.team_epa_scatter(eff, season=season)
        path = out / f"epa_scatter_{season}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    if chart_type in ("power", "all"):
        ranks = power_rankings(pbp, sched, season=season)
        fig = viz.power_rankings_bar(ranks)
        path = out / f"power_rankings_{season}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    if chart_type in ("qbs", "all"):
        qbs = qb_analysis(pbp, season=season)
        fig = viz.qb_rankings_bar(qbs, top_n=15)
        path = out / f"qb_rankings_{season}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    if chart_type in ("elo", "all"):
        teams = args.teams if args.teams else ["KC", "BUF", "SF", "BAL", "PHI"]
        fig = viz.elo_history(sched, teams=teams)
        path = out / f"elo_history.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    if chart_type in ("radar", "all") and args.team:
        eff = team_efficiency(pbp, season=season)
        fig = viz.team_efficiency_radar(eff, team=args.team.upper())
        path = out / f"radar_{args.team.upper()}_{season}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")

    if chart_type in ("predict", "all"):
        elo_sys = build_elo_ratings(sched)
        upcoming = sched[(sched["season"] == season) & (sched["result"].isna())]
        if len(upcoming) > 0:
            next_week = upcoming["week"].min()
            preds = []
            for _, g in upcoming[upcoming["week"] == next_week].iterrows():
                preds.append(elo_sys.predict(g["home_team"], g["away_team"]))
            fig = viz.weekly_spread_chart(preds)
            path = out / f"predictions_week{next_week}_{season}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {path}")

    print(f"\nCharts saved to {out}/")


def cmd_export(args):
    """Export specific analysis to file."""
    from analyze import load_pbp, load_schedules, team_efficiency, qb_analysis, power_rankings
    from export import to_csv, to_json, to_markdown

    pbp = load_pbp()
    sched = load_schedules()
    season = args.season or int(pbp["season"].max())
    out = Path(args.output)

    analysis_type = args.analysis

    if analysis_type == "teams":
        df = team_efficiency(pbp, season=season)
        name = f"team_efficiency_{season}"
    elif analysis_type == "qbs":
        df = qb_analysis(pbp, season=season)
        name = f"qb_rankings_{season}"
    elif analysis_type == "power":
        df = power_rankings(pbp, sched, season=season)
        name = f"power_rankings_{season}"
    else:
        print(f"Unknown analysis type: {analysis_type}")
        sys.exit(1)

    fmt = args.format
    if fmt == "csv":
        path = to_csv(df, out / f"{name}.csv")
    elif fmt == "json":
        path = to_json(df, out / f"{name}.json")
    elif fmt == "md":
        path = to_markdown(df, out / f"{name}.md", title=name.replace("_", " ").title())
    else:
        print(f"Unknown format: {fmt}")
        sys.exit(1)

    print(f"Exported: {path}")


def cmd_report(args):
    """Generate full analysis report."""
    from export import full_report

    season = args.season
    formats = args.formats.split(",") if args.formats else ["csv", "json", "md"]
    out = args.output

    exported = full_report(output_dir=out, season=season, formats=formats)

    total = sum(len(v) for v in exported.values())
    print(f"\nFull report: {total} files exported to {out}/")


# ── Argument Parser ──────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nfl-analytics",
        description="NFL Analytics Engine — Data, Analysis, Viz, Export",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Download NFL data")
    p_fetch.add_argument("--season", type=int, help="Single season (e.g., 2025)")
    p_fetch.add_argument("--seasons", type=str, help="Season range (e.g., 2020-2025)")

    # teams
    p_teams = sub.add_parser("teams", help="Team efficiency rankings")
    p_teams.add_argument("--season", type=int)
    p_teams.add_argument("--top", type=int, default=32, help="Number of teams to show")

    # qbs
    p_qbs = sub.add_parser("qbs", help="QB analysis")
    p_qbs.add_argument("--season", type=int)
    p_qbs.add_argument("--top", type=int, default=20)
    p_qbs.add_argument("--min-dropbacks", type=int, default=100)

    # power
    p_power = sub.add_parser("power", help="Power rankings")
    p_power.add_argument("--season", type=int)

    # predict
    p_predict = sub.add_parser("predict", help="Elo game predictions")
    p_predict.add_argument("--season", type=int)

    # elo
    p_elo = sub.add_parser("elo", help="Elo rating lookup")
    p_elo.add_argument("teams", nargs="+", help="Team abbreviations (e.g., KC BUF SF)")

    # viz
    p_viz = sub.add_parser("viz", help="Generate charts")
    p_viz.add_argument("chart", choices=["scatter", "power", "qbs", "elo", "radar", "predict", "all"],
                       help="Chart type")
    p_viz.add_argument("--season", type=int)
    p_viz.add_argument("--team", type=str, help="Team for radar chart")
    p_viz.add_argument("--teams", nargs="+", help="Teams for Elo history")
    p_viz.add_argument("-o", "--output", default="charts", help="Output directory")

    # export
    p_export = sub.add_parser("export", help="Export analysis to file")
    p_export.add_argument("analysis", choices=["teams", "qbs", "power"], help="Analysis to export")
    p_export.add_argument("--season", type=int)
    p_export.add_argument("--format", choices=["csv", "json", "md"], default="csv")
    p_export.add_argument("-o", "--output", default="output", help="Output directory")

    # report
    p_report = sub.add_parser("report", help="Full analysis report")
    p_report.add_argument("--season", type=int)
    p_report.add_argument("--formats", type=str, default="csv,json,md", help="Comma-separated formats")
    p_report.add_argument("-o", "--output", default="output", help="Output directory")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "fetch":   cmd_fetch,
        "teams":   cmd_teams,
        "qbs":     cmd_qbs,
        "power":   cmd_power,
        "predict": cmd_predict,
        "elo":     cmd_elo,
        "viz":     cmd_viz,
        "export":  cmd_export,
        "report":  cmd_report,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
