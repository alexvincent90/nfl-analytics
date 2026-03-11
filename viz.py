"""
NFL Analytics — Visualization Module

Charts:
  1. team_epa_scatter      — Offense vs Defense EPA (quadrant chart)
  2. power_rankings_bar    — Horizontal bar chart of composite power scores
  3. qb_rankings_bar       — Top QBs by EPA/dropback
  4. elo_history            — Elo rating over time for selected teams
  5. team_efficiency_radar  — Radar/spider chart for a single team
  6. weekly_spread_chart    — Predicted spreads for upcoming games

All functions return a matplotlib Figure so callers can save or show.

Usage:
  from viz import team_epa_scatter
  fig = team_epa_scatter(efficiency_df)
  fig.savefig("epa_scatter.png", dpi=150, bbox_inches="tight")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Style defaults ────────────────────────────────────────────

sns.set_theme(style="whitegrid", font_scale=0.95)

COLORS = {
    "primary": "#2563EB",
    "accent":  "#00A699",
    "red":     "#ef4444",
    "green":   "#22c55e",
    "amber":   "#f59e0b",
    "navy":    "#1a2332",
    "gray":    "#94a3b8",
}

# NFL team colors (subset, for team-specific charts)
TEAM_COLORS = {
    "KC":  "#E31837", "BUF": "#00338D", "BAL": "#241773", "SF":  "#AA0000",
    "PHI": "#004C54", "DAL": "#003594", "DET": "#0076B6", "MIA": "#008E97",
    "CIN": "#FB4F14", "JAX": "#006778", "HOU": "#03202F", "CLE": "#FF3C00",
    "NYJ": "#125740", "NE":  "#002244", "PIT": "#FFB612", "IND": "#002C5F",
    "LAC": "#0080C6", "DEN": "#FB4F14", "LV":  "#000000", "TEN": "#4B92DB",
    "GB":  "#203731", "MIN": "#4F2683", "CHI": "#C83803", "TB":  "#D50A0A",
    "NO":  "#D3BC8D", "ATL": "#A71930", "CAR": "#0085CA", "SEA": "#002244",
    "LAR": "#003594", "ARI": "#97233F", "NYG": "#0B2265", "WAS": "#5A1414",
}


def _get_team_color(team: str) -> str:
    return TEAM_COLORS.get(team, COLORS["primary"])


# ── 1. EPA Scatter (Offense vs Defense) ───────────────────────

def team_epa_scatter(efficiency: pd.DataFrame, season: int | None = None) -> plt.Figure:
    """
    Quadrant scatter: x = Offensive EPA/play, y = Defensive EPA/play.
    Top-right quadrant = elite teams (good offense, good defense).
    Defense is inverted so UP = better defense.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    off = efficiency["epa_per_play"]
    # Invert defense so up = good
    defe = -efficiency["epa_per_play_def"]

    colors = [_get_team_color(t) for t in efficiency.index]

    ax.scatter(off, defe, c=colors, s=120, edgecolors="white", linewidth=1.5, zorder=5)

    for team in efficiency.index:
        ax.annotate(
            team,
            (off[team], defe[team]),
            fontsize=8, fontweight="bold", ha="center", va="bottom",
            xytext=(0, 8), textcoords="offset points",
        )

    # Quadrant lines at league average
    ax.axhline(defe.mean(), color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(off.mean(), color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.6)

    # Quadrant labels
    props = dict(fontsize=9, color=COLORS["gray"], alpha=0.5, fontstyle="italic")
    ax.text(off.max(), defe.max(), "ELITE", ha="right", va="top", **props)
    ax.text(off.min(), defe.max(), "Defensive", ha="left", va="top", **props)
    ax.text(off.max(), defe.min(), "Offensive", ha="right", va="bottom", **props)
    ax.text(off.min(), defe.min(), "Rebuilding", ha="left", va="bottom", **props)

    title = "Team Efficiency — Offense vs Defense EPA/Play"
    if season:
        title += f" ({season})"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Offensive EPA/Play →", fontsize=11)
    ax.set_ylabel("← Defensive EPA/Play (inverted, up = better)", fontsize=11)

    fig.tight_layout()
    return fig


# ── 2. Power Rankings Bar Chart ──────────────────────────────

def power_rankings_bar(rankings: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of composite power scores (top 32 teams)."""
    fig, ax = plt.subplots(figsize=(10, 12))

    df = rankings.head(32).iloc[::-1]  # Reverse for horizontal bar (top team at top)

    colors = [_get_team_color(t) for t in df["team"]]

    bars = ax.barh(df["team"], df["power_score"], color=colors, edgecolor="white", height=0.7)

    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{row["power_score"]:.1f}  ({row["record"]})',
            va="center", fontsize=9, fontweight="500",
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Power Score", fontsize=11)
    ax.set_title("NFL Power Rankings", fontsize=14, fontweight="bold", pad=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))

    fig.tight_layout()
    return fig


# ── 3. QB Rankings Bar Chart ─────────────────────────────────

def qb_rankings_bar(qbs: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Top QBs by EPA/dropback with CPOE annotation."""
    fig, ax = plt.subplots(figsize=(10, 8))

    df = qbs.head(top_n).iloc[::-1]

    # Color by EPA tier
    bar_colors = []
    for epa in df["epa_per_dropback"]:
        if epa >= 0.15:
            bar_colors.append(COLORS["green"])
        elif epa >= 0.05:
            bar_colors.append(COLORS["primary"])
        elif epa >= 0:
            bar_colors.append(COLORS["amber"])
        else:
            bar_colors.append(COLORS["red"])

    labels = [f'{name} ({row["team"]})' for name, row in df.iterrows()]
    bars = ax.barh(labels, df["epa_per_dropback"], color=bar_colors, edgecolor="white", height=0.7)

    for bar, (_, row) in zip(bars, df.iterrows()):
        cpoe = row.get("cpoe", 0)
        cpoe_str = f"+{cpoe:.1f}" if cpoe and not np.isnan(cpoe) and cpoe > 0 else f"{cpoe:.1f}" if cpoe and not np.isnan(cpoe) else ""
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{row["epa_per_dropback"]:.3f}  CPOE: {cpoe_str}',
            va="center", fontsize=8,
        )

    ax.axvline(0, color=COLORS["navy"], linewidth=1)
    ax.set_xlabel("EPA / Dropback", fontsize=11)
    ax.set_title("QB Rankings — EPA per Dropback", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


# ── 4. Elo History ───────────────────────────────────────────

def elo_history(schedules: pd.DataFrame, teams: list[str], seasons: list[int] | None = None) -> plt.Figure:
    """
    Plot Elo rating over time for selected teams.
    Rebuilds Elo from schedules and tracks history per game.
    """
    from analyze import EloSystem

    elo = EloSystem()
    games = schedules.sort_values(["season", "week", "gameday"]).copy()
    games = games[games["result"].notna()]

    if seasons:
        games = games[games["season"].isin(seasons)]

    # Track history for requested teams
    history = {t: [] for t in teams}
    game_indices = {t: [] for t in teams}

    current_season = None
    game_num = 0

    for _, game in games.iterrows():
        season = game["season"]
        if current_season is not None and season != current_season:
            elo.new_season()
        current_season = season

        home, away = game["home_team"], game["away_team"]
        hs, as_ = game.get("home_score", 0), game.get("away_score", 0)

        if pd.notna(hs) and pd.notna(as_):
            elo.update(home, away, int(hs), int(as_))

        game_num += 1
        for t in teams:
            if t in (home, away):
                history[t].append(elo.get_rating(t))
                game_indices[t].append(game_num)

    fig, ax = plt.subplots(figsize=(12, 6))

    for t in teams:
        color = _get_team_color(t)
        ax.plot(game_indices[t], history[t], label=t, color=color, linewidth=2)

    ax.axhline(1500, color=COLORS["gray"], linestyle="--", linewidth=0.8, alpha=0.5, label="Average (1500)")
    ax.set_xlabel("Game Number", fontsize=11)
    ax.set_ylabel("Elo Rating", fontsize=11)
    ax.set_title("Elo Rating History", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


# ── 5. Team Efficiency Radar ─────────────────────────────────

def team_efficiency_radar(efficiency: pd.DataFrame, team: str) -> plt.Figure:
    """
    Spider/radar chart for a single team's efficiency profile.
    Metrics: Off EPA, Def EPA (inv), Success Rate, Explosive Rate, Pass Rate, Rush EPA.
    """
    if team not in efficiency.index:
        raise ValueError(f"Team '{team}' not found in efficiency data")

    row = efficiency.loc[team]
    all_teams = efficiency

    # Metrics to plot (normalized to 0-100 percentile)
    metrics = {
        "Off EPA":        "epa_per_play",
        "Def EPA":        "epa_per_play_def",
        "Success %":      "success_rate",
        "Explosive %":    "explosive_rate",
        "Early Pass %":   "pass_rate_early",
    }

    labels = list(metrics.keys())
    values = []

    for label, col in metrics.items():
        if col not in all_teams.columns or pd.isna(row.get(col)):
            values.append(50)
            continue
        col_data = all_teams[col].dropna()
        if col == "epa_per_play_def":
            # Invert: lower defensive EPA = better
            pctile = 100 - (col_data.rank(pct=True).get(team, 0.5) * 100)
        else:
            pctile = col_data.rank(pct=True).get(team, 0.5) * 100
        values.append(pctile)

    # Close the radar
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    color = _get_team_color(team)
    ax.fill(angles, values, color=color, alpha=0.2)
    ax.plot(angles, values, color=color, linewidth=2.5)
    ax.scatter(angles[:-1], values[:-1], color=color, s=60, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25th", "50th", "75th", "100th"], fontsize=8, color=COLORS["gray"])

    ax.set_title(f"{team} — Efficiency Profile (Percentile)", fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    return fig


# ── 6. Weekly Predictions ────────────────────────────────────

def weekly_spread_chart(predictions: list[dict]) -> plt.Figure:
    """
    Horizontal diverging bar chart showing predicted spreads for upcoming games.
    Bars extend left (away favored) or right (home favored).
    """
    if not predictions:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No upcoming games to predict", ha="center", va="center", fontsize=14)
        return fig

    fig, ax = plt.subplots(figsize=(10, max(4, len(predictions) * 0.6)))

    labels = [f'{p["away"]} @ {p["home"]}' for p in predictions]
    spreads = [p["spread"] for p in predictions]

    bar_colors = [COLORS["green"] if s > 0 else COLORS["red"] for s in spreads]

    ax.barh(labels, spreads, color=bar_colors, edgecolor="white", height=0.6)

    for i, pred in enumerate(predictions):
        fav = pred["home"] if pred["spread"] > 0 else pred["away"]
        wp = max(pred["home_win_prob"], pred["away_win_prob"])
        ax.text(
            pred["spread"] + (0.3 if pred["spread"] >= 0 else -0.3),
            i,
            f'{fav} {abs(pred["spread"]):.1f} ({wp:.0%})',
            va="center",
            ha="left" if pred["spread"] >= 0 else "right",
            fontsize=9,
        )

    ax.axvline(0, color=COLORS["navy"], linewidth=1.5)
    ax.set_xlabel("Predicted Spread (+ = Home favored)", fontsize=11)
    ax.set_title("Game Predictions — Elo Model", fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig
