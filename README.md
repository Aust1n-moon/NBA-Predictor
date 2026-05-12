# NBA Predictor

Machine learning models for predicting NBA game outcomes, including moneyline, spread, team totals, and over/under lines. Uses 2025-26 season data scraped from the NBA API and Basketball Reference.

## Features

- **Moneyline** — win probability for each team (Logistic Regression, Random Forest, Gradient Boosting)
- **Spread** — predicted point margin
- **Team Totals** — predicted points per team
- **Over/Under** — predicted combined game total
- **Top Picks** — ranks the best bets across all games on a given date
- **Injury Awareness** — suppresses predictions when impact players (25+ MPG) are confirmed out

## Project Structure

```
collected data/
  Data.ipynb                  # Scrapes game stats and schedule from nba_api / Basketball Reference
  nba_schedule_2025-2026.csv  # Remaining schedule
  injuries.csv                # Current injury report
  player_minutes.csv          # Per-game minutes for impact player detection
  2025-26 season/             # Per-team game logs (one CSV per team)

modelling/
  scikitlearn-model.py        # Model training and evaluation (CLI)
  dashboard.py                # Tkinter GUI for date-based predictions
```

## Setup

```bash
pip install pandas numpy scikit-learn nba_api
```

## Usage

### Collect Data

Run all cells in `collected data/Data.ipynb` to scrape the latest game logs and schedule. This calls the NBA API and Basketball Reference, so expect rate-limiting delays.

### Train & Evaluate (CLI)

```bash
python modelling/scikitlearn-model.py
```

Prints MAE, RMSE, R², and accuracy for each model across all four prediction targets.

### Dashboard (GUI)

```bash
python modelling/dashboard.py
```

1. Wait for models to load (status bar turns green)
2. Enter a date and click **Search** to list that day's games
3. Select a game and click **Predict** for full predictions
4. Click **Top Picks** for the 3 highest-confidence bets of the day

## How It Works

Rolling 5-game averages of team stats (points, efficiency ratings, shooting percentages, pace, etc.) are computed for both the team and opponent, shifted by one game to prevent data leakage. These features, along with home/away status, are fed into three model types per prediction target:

- Linear/Logistic Regression
- Random Forest
- Gradient Boosting

Predictions are averaged across models for the final output.
