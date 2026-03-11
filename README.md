# NFL Analytics Toolkit

Python-based NFL statistical analysis framework. Uses `nfl_data_py` for
historical play-by-play data, roster info, and schedule data.

## Features

### Phase 1 (Current)
- **Data Pipeline**: Fetch and cache NFL play-by-play, roster, and schedule data
- **Team Analytics**: Offensive/defensive efficiency, EPA analysis, success rate
- **Player Value Model**: Expected Points Added per play, CPOE for QBs
- **Game Predictor**: Simple Elo-based game outcome predictions

### Phase 2 (Future)
- Prop bet modeling (player performance distributions)
- Draft value analysis (surplus value by pick)
- Injury impact modeling
- Weekly power rankings with confidence intervals

## Setup

```bash
pip install -r requirements.txt
python fetch_data.py --season 2025
python analyze.py
```

## Data Sources
- `nfl_data_py` — comprehensive NFL data (play-by-play, rosters, schedules)
- Pro Football Reference (supplemental)
- No API keys required for base functionality
