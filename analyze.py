"""
NFL Analytics — Core Analysis Engine

Provides:
  1. Team efficiency metrics (EPA/play, success rate, explosive play rate)
  2. QB analysis (CPOE, EPA, air yards, pressure handling)
  3. Elo ratings and game predictions
  4. Weekly power rankings

Usage:
  python analyze.py                  # Full analysis for latest cached data
  python analyze.py --team KC        # Single-team deep dive
  python analyze.py --predict week17 # Predict upcoming week
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"


# ── Data Loading ─────────────────────────────────────────────────

def load_pbp() -> pd.DataFrame:
    """Load cached play-by-play data."""
    path = DATA_DIR / "pbp.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No PBP data at {path}. Run fetch_data.py first.")
    return pd.read_parquet(path)


def load_schedules() -> pd.DataFrame:
    """Load cached schedule data."""
    path = DATA_DIR / "schedules.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No schedule data at {path}. Run fetch_data.py first.")
    return pd.read_parquet(path)


# ── 1. Team Efficiency ───────────────────────────────────────────

def team_efficiency(pbp: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """
    Compute per-team offensive and defensive efficiency metrics.

    Metrics:
      - epa_per_play:     Expected Points Added per play (offense)
      - epa_per_play_def: EPA/play allowed (defense, lower = better)
      - success_rate:     % of plays with positive EPA (offense)
      - explosive_rate:   % of plays with EPA > 1.5
      - pass_rate:        Pass play % on early downs
      - rush_epa:         EPA/rush
      - pass_epa:         EPA/pass

    Returns a DataFrame indexed by posteam (team abbreviation).
    """
    df = pbp.copy()

    # Filter to regular plays (no kickoffs, penalties, etc.)
    df = df[
        (df["play_type"].isin(["pass", "run"]))
        & (df["epa"].notna())
        & (df["posteam"].notna())
    ]

    if season:
        df = df[df["season"] == season]

    # Offensive metrics
    off = df.groupby("posteam").agg(
        plays=("epa", "count"),
        epa_per_play=("epa", "mean"),
        success_rate=("epa", lambda x: (x > 0).mean()),
        explosive_rate=("epa", lambda x: (x > 1.5).mean()),
        pass_epa=("epa", lambda x: x[df.loc[x.index, "play_type"] == "pass"].mean()
                  if len(x[df.loc[x.index, "play_type"] == "pass"]) > 0 else 0),
        rush_epa=("epa", lambda x: x[df.loc[x.index, "play_type"] == "run"].mean()
                  if len(x[df.loc[x.index, "play_type"] == "run"]) > 0 else 0),
    ).round(4)

    # Simpler pass/rush EPA calculation
    pass_plays = df[df["play_type"] == "pass"].groupby("posteam")["epa"].mean().rename("pass_epa_clean")
    rush_plays = df[df["play_type"] == "run"].groupby("posteam")["epa"].mean().rename("rush_epa_clean")

    # Defensive metrics
    defense = df.groupby("defteam").agg(
        def_plays=("epa", "count"),
        epa_per_play_def=("epa", "mean"),
        def_success_rate=("epa", lambda x: (x > 0).mean()),
    ).round(4)

    # Early-down pass rate (1st & 2nd down, non-garbage time)
    early = df[(df["down"].isin([1, 2])) & (df["wp"].between(0.1, 0.9))]
    pass_rate = early.groupby("posteam")["play_type"].apply(
        lambda x: (x == "pass").mean()
    ).rename("pass_rate_early").round(4)

    # Merge
    result = off.join(pass_plays).join(rush_plays).join(defense, how="left").join(pass_rate)

    # Sort by EPA/play
    result = result.sort_values("epa_per_play", ascending=False)

    return result


# ── 2. QB Analysis ───────────────────────────────────────────────

def qb_analysis(pbp: pd.DataFrame, season: int | None = None, min_dropbacks: int = 100) -> pd.DataFrame:
    """
    QB-level analysis using play-by-play data.

    Metrics:
      - epa_per_dropback: EPA per pass attempt
      - cpoe:             Completion % Over Expected
      - adot:             Average Depth of Target (air yards)
      - pressure_rate:    How often the QB is pressured
      - sack_rate:        Sack rate when pressured
      - scramble_rate:    How often the QB scrambles
    """
    df = pbp.copy()
    df = df[
        (df["play_type"] == "pass")
        & (df["epa"].notna())
        & (df["passer_player_name"].notna())
    ]

    if season:
        df = df[df["season"] == season]

    qbs = df.groupby("passer_player_name").agg(
        team=("posteam", "first"),
        dropbacks=("epa", "count"),
        epa_per_dropback=("epa", "mean"),
        total_epa=("epa", "sum"),
        cpoe=("cpoe", "mean"),
        adot=("air_yards", "mean"),
        comp_pct=("complete_pass", "mean"),
        interceptions=("interception", "sum"),
        sacks=("sack", "sum"),
        pass_tds=("pass_touchdown", "sum"),
    ).round(3)

    # Filter to qualified QBs
    qbs = qbs[qbs["dropbacks"] >= min_dropbacks]

    # Derived metrics
    qbs["td_rate"] = (qbs["pass_tds"] / qbs["dropbacks"]).round(4)
    qbs["int_rate"] = (qbs["interceptions"] / qbs["dropbacks"]).round(4)
    qbs["sack_rate"] = (qbs["sacks"] / (qbs["dropbacks"] + qbs["sacks"])).round(4)

    # Sort by EPA/dropback
    qbs = qbs.sort_values("epa_per_dropback", ascending=False)

    return qbs


# ── 3. Elo Ratings ───────────────────────────────────────────────

class EloSystem:
    """
    Simple Elo rating system for NFL teams.

    Parameters:
      K = 20      (standard NFL Elo K-factor)
      HFA = 48    (home-field advantage in Elo points, ~2.5 pts spread)
      REGRESS = 1/3  (regress toward mean between seasons)
    """

    def __init__(self, K=20, HFA=48, MEAN=1500, REGRESS=1/3):
        self.K = K
        self.HFA = HFA
        self.MEAN = MEAN
        self.REGRESS = REGRESS
        self.ratings = {}  # team -> current Elo

    def _expected(self, elo_a: float, elo_b: float) -> float:
        """Expected win probability for team A."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

    def _mov_multiplier(self, mov: int, elo_diff: float) -> float:
        """Margin-of-victory multiplier (FiveThirtyEight formula)."""
        return np.log(abs(mov) + 1) * (2.2 / ((elo_diff if elo_diff > 0 else -elo_diff) * 0.001 + 2.2))

    def get_rating(self, team: str) -> float:
        """Get current Elo for a team (default 1500)."""
        return self.ratings.get(team, self.MEAN)

    def new_season(self):
        """Regress all ratings toward the mean between seasons."""
        for team in self.ratings:
            self.ratings[team] = self.MEAN + (self.ratings[team] - self.MEAN) * (1 - self.REGRESS)

    def update(self, home: str, away: str, home_score: int, away_score: int):
        """
        Update Elo ratings after a game result.
        Returns the new ratings for both teams.
        """
        home_elo = self.get_rating(home) + self.HFA
        away_elo = self.get_rating(away)

        expected_home = self._expected(home_elo, away_elo)

        if home_score > away_score:
            result = 1.0
        elif home_score < away_score:
            result = 0.0
        else:
            result = 0.5  # Ties are rare but happen

        mov = home_score - away_score
        elo_diff = home_elo - away_elo
        mult = self._mov_multiplier(mov, elo_diff)

        shift = self.K * mult * (result - expected_home)

        self.ratings[home] = self.get_rating(home) + shift
        self.ratings[away] = self.get_rating(away) - shift

        return self.ratings[home], self.ratings[away]

    def predict(self, home: str, away: str) -> dict:
        """
        Predict game outcome.
        Returns: home_win_prob, away_win_prob, spread (in points).
        """
        home_elo = self.get_rating(home) + self.HFA
        away_elo = self.get_rating(away)

        home_wp = self._expected(home_elo, away_elo)
        spread = (home_elo - away_elo) / 25  # Elo diff to point spread (~25 Elo per point)

        return {
            "home": home,
            "away": away,
            "home_elo": round(self.get_rating(home)),
            "away_elo": round(self.get_rating(away)),
            "home_win_prob": round(home_wp, 3),
            "away_win_prob": round(1 - home_wp, 3),
            "spread": round(spread, 1),
        }


def build_elo_ratings(schedules: pd.DataFrame) -> EloSystem:
    """
    Build Elo ratings from historical game results.
    Processes games chronologically.
    """
    elo = EloSystem()

    # Sort by season, week, gameday
    games = schedules.sort_values(["season", "week", "gameday"]).copy()

    # Filter to completed games
    games = games[games["result"].notna()]

    current_season = None

    for _, game in games.iterrows():
        season = game["season"]

        # Regress at season boundaries
        if current_season is not None and season != current_season:
            elo.new_season()
        current_season = season

        home = game["home_team"]
        away = game["away_team"]
        home_score = game.get("home_score", 0)
        away_score = game.get("away_score", 0)

        if pd.notna(home_score) and pd.notna(away_score):
            elo.update(home, away, int(home_score), int(away_score))

    return elo


# ── 4. Power Rankings ────────────────────────────────────────────

def power_rankings(pbp: pd.DataFrame, schedules: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compute weekly power rankings combining:
      - Elo rating (40%)
      - Offensive EPA/play (25%)
      - Defensive EPA/play (25%)
      - Win % (10%)

    Returns a sorted DataFrame with composite power score.
    """
    # Get Elo ratings
    elo = build_elo_ratings(schedules)

    # Get team efficiency for the season
    eff = team_efficiency(pbp, season=season)

    # Get win records
    games = schedules[(schedules["season"] == season) & (schedules["result"].notna())]

    wins = {}
    for _, g in games.iterrows():
        home, away = g["home_team"], g["away_team"]
        if home not in wins:
            wins[home] = {"w": 0, "l": 0, "t": 0}
        if away not in wins:
            wins[away] = {"w": 0, "l": 0, "t": 0}

        hs, as_ = g.get("home_score", 0), g.get("away_score", 0)
        if pd.notna(hs) and pd.notna(as_):
            if hs > as_:
                wins[home]["w"] += 1
                wins[away]["l"] += 1
            elif as_ > hs:
                wins[away]["w"] += 1
                wins[home]["l"] += 1
            else:
                wins[home]["t"] += 1
                wins[away]["t"] += 1

    win_df = pd.DataFrame(wins).T
    win_df["win_pct"] = (win_df["w"] + 0.5 * win_df["t"]) / (win_df["w"] + win_df["l"] + win_df["t"])

    # Build power rankings
    teams = eff.index.tolist()
    rows = []

    for team in teams:
        team_elo = elo.get_rating(team)
        off_epa = eff.loc[team, "epa_per_play"] if team in eff.index else 0
        def_epa = eff.loc[team, "epa_per_play_def"] if team in eff.index else 0
        wp = win_df.loc[team, "win_pct"] if team in win_df.index else 0.5
        record = f"{int(win_df.loc[team, 'w'])}-{int(win_df.loc[team, 'l'])}" if team in win_df.index else "0-0"

        # Normalize components to 0-100 scale
        elo_norm = min(max((team_elo - 1300) / 400 * 100, 0), 100)
        off_norm = min(max((off_epa + 0.3) / 0.6 * 100, 0), 100)  # EPA range roughly -0.3 to +0.3
        def_norm = min(max((-def_epa + 0.3) / 0.6 * 100, 0), 100)  # Invert (lower = better)
        wp_norm = wp * 100

        composite = (
            elo_norm * 0.40
            + off_norm * 0.25
            + def_norm * 0.25
            + wp_norm * 0.10
        )

        rows.append({
            "team": team,
            "record": record,
            "elo": round(team_elo),
            "off_epa": round(off_epa, 3),
            "def_epa": round(def_epa, 3),
            "win_pct": round(wp, 3),
            "power_score": round(composite, 1),
        })

    rankings = pd.DataFrame(rows).sort_values("power_score", ascending=False).reset_index(drop=True)
    rankings.index = rankings.index + 1  # 1-based ranking
    rankings.index.name = "rank"

    return rankings


# ── Main ─────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("NFL ANALYTICS ENGINE — Starting analysis")
    log.info("=" * 60)

    pbp = load_pbp()
    sched = load_schedules()

    # Determine latest season
    latest_season = int(pbp["season"].max())
    log.info(f"Latest season in data: {latest_season}")

    # 1. Team efficiency
    log.info("\n--- TEAM EFFICIENCY ---")
    eff = team_efficiency(pbp, season=latest_season)
    print("\nTop 10 Teams by EPA/Play:")
    print(eff[["plays", "epa_per_play", "success_rate", "explosive_rate", "pass_rate_early"]].head(10).to_string())

    # 2. QB rankings
    log.info("\n--- QB ANALYSIS ---")
    qbs = qb_analysis(pbp, season=latest_season)
    print("\nTop 10 QBs by EPA/Dropback:")
    print(qbs[["team", "dropbacks", "epa_per_dropback", "cpoe", "comp_pct", "td_rate", "int_rate"]].head(10).to_string())

    # 3. Power rankings
    log.info("\n--- POWER RANKINGS ---")
    rankings = power_rankings(pbp, sched, season=latest_season)
    print(f"\nPower Rankings — {latest_season} Season:")
    print(rankings.to_string())

    # 4. Elo predictions (if upcoming games exist)
    elo = build_elo_ratings(sched)
    upcoming = sched[(sched["season"] == latest_season) & (sched["result"].isna())]

    if len(upcoming) > 0:
        next_week = upcoming["week"].min()
        next_games = upcoming[upcoming["week"] == next_week]
        log.info(f"\n--- WEEK {next_week} PREDICTIONS ---")
        predictions = []
        for _, g in next_games.iterrows():
            pred = elo.predict(g["home_team"], g["away_team"])
            predictions.append(pred)
            fav = pred["home"] if pred["home_win_prob"] > 0.5 else pred["away"]
            spread_str = f"{pred['spread']:+.1f}" if pred["spread"] != 0 else "PICK"
            print(f"  {pred['away']} @ {pred['home']}: {fav} {spread_str} (WP: {max(pred['home_win_prob'], pred['away_win_prob']):.1%})")

    log.info("\nAnalysis complete.")


if __name__ == "__main__":
    main()
