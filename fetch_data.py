"""
NFL Analytics — Data Fetcher

Downloads and caches NFL data using nfl_data_py:
  - Play-by-play data (main analysis source)
  - Roster data (player info, positions, teams)
  - Schedule/game results
  - Team descriptions

Usage:
  python fetch_data.py                    # Current + last 2 seasons
  python fetch_data.py --season 2025      # Specific season
  python fetch_data.py --seasons 2020-2025  # Range
"""

import sys
import logging
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_pbp(seasons: list[int]) -> pd.DataFrame:
    """
    Fetch play-by-play data for given seasons.
    This is the core dataset — every play from every game.
    ~50k rows per season.
    """
    log.info(f"Fetching play-by-play for seasons: {seasons}")
    pbp = nfl.import_pbp_data(seasons, downcast=True)
    path = DATA_DIR / "pbp.parquet"
    pbp.to_parquet(path)
    log.info(f"Saved {len(pbp):,} plays to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return pbp


def fetch_rosters(seasons: list[int]) -> pd.DataFrame:
    """Fetch weekly roster data (player, team, position, status)."""
    log.info(f"Fetching rosters for seasons: {seasons}")
    rosters = nfl.import_weekly_rosters(seasons)
    path = DATA_DIR / "rosters.parquet"
    rosters.to_parquet(path)
    log.info(f"Saved {len(rosters):,} roster entries to {path}")
    return rosters


def fetch_schedules(seasons: list[int]) -> pd.DataFrame:
    """Fetch game schedule and results."""
    log.info(f"Fetching schedules for seasons: {seasons}")
    sched = nfl.import_schedules(seasons)
    path = DATA_DIR / "schedules.parquet"
    sched.to_parquet(path)
    log.info(f"Saved {len(sched):,} games to {path}")
    return sched


def fetch_team_desc() -> pd.DataFrame:
    """Fetch team descriptions (abbreviations, colors, logos)."""
    log.info("Fetching team descriptions")
    teams = nfl.import_team_desc()
    path = DATA_DIR / "teams.parquet"
    teams.to_parquet(path)
    log.info(f"Saved {len(teams)} teams to {path}")
    return teams


def fetch_all(seasons: list[int]):
    """Fetch all datasets."""
    log.info(f"\n{'='*60}\nNFL DATA FETCH — Seasons: {seasons}\n{'='*60}")
    fetch_pbp(seasons)
    fetch_rosters(seasons)
    fetch_schedules(seasons)
    fetch_team_desc()
    log.info("All data fetched successfully.")


def parse_seasons(arg: str) -> list[int]:
    """Parse season argument: '2025' or '2020-2025'."""
    if "-" in arg:
        start, end = arg.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(arg)]


if __name__ == "__main__":
    if "--season" in sys.argv:
        idx = sys.argv.index("--season")
        seasons = [int(sys.argv[idx + 1])]
    elif "--seasons" in sys.argv:
        idx = sys.argv.index("--seasons")
        seasons = parse_seasons(sys.argv[idx + 1])
    else:
        # Default: current season + 2 prior
        from datetime import datetime
        current_year = datetime.now().year
        seasons = [current_year - 2, current_year - 1, current_year]

    fetch_all(seasons)
