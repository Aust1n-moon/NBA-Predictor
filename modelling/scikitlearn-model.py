import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ============================================================
# Teams Dictionary
# ============================================================
TEAMS = {
    1610612737: "Atlanta_Hawks",
    1610612738: "Boston_Celtics",
    1610612751: "Brooklyn_Nets",
    1610612766: "Charlotte_Hornets",
    1610612741: "Chicago_Bulls",
    1610612739: "Cleveland_Cavaliers",
    1610612742: "Dallas_Mavericks",
    1610612743: "Denver_Nuggets",
    1610612765: "Detroit_Pistons",
    1610612744: "Golden_State_Warriors",
    1610612745: "Houston_Rockets",
    1610612754: "Indiana_Pacers",
    1610612746: "Los_Angeles_Clippers",
    1610612747: "Los_Angeles_Lakers",
    1610612763: "Memphis_Grizzlies",
    1610612748: "Miami_Heat",
    1610612749: "Milwaukee_Bucks",
    1610612750: "Minnesota_Timberwolves",
    1610612740: "New_Orleans_Pelicans",
    1610612752: "New_York_Knicks",
    1610612760: "Oklahoma_City_Thunder",
    1610612753: "Orlando_Magic",
    1610612755: "Philadelphia_76ers",
    1610612756: "Phoenix_Suns",
    1610612757: "Portland_Trail_Blazers",
    1610612758: "Sacramento_Kings",
    1610612759: "San_Antonio_Spurs",
    1610612761: "Toronto_Raptors",
    1610612762: "Utah_Jazz",
    1610612764: "Washington_Wizards",
}

#to read the csv files
def read_csv():
    folder = Path(__file__).parent.parent / "collected data" / "2025-26 season"
    all_teams = []

    for team_name in TEAMS.values():
        df = pd.read_csv(folder / f"{team_name}_2025-26.csv")
        all_teams.append(df)

    return pd.concat(all_teams, ignore_index=True)


def get_teams_missing_impact_players(min_mpg=25):
    """Return {team_display_name: [player_names]} for teams with impact players confirmed out."""
    base = Path(__file__).parent.parent / "collected data"
    injuries_path = base / "injuries.csv"
    minutes_path = base / "player_minutes.csv"

    if not injuries_path.exists() or not minutes_path.exists():
        return {}

    injuries = pd.read_csv(injuries_path)
    minutes = pd.read_csv(minutes_path)

    # Impact players: >= min_mpg minutes per game
    impact = minutes[minutes['MIN'] >= min_mpg].copy()

    # Map TEAM_ID -> display name (space-separated)
    team_id_to_display = {k: v.replace("_", " ") for k, v in TEAMS.items()}
    impact['team_display'] = impact['TEAM_ID'].map(team_id_to_display)

    # Filter injuries for confirmed "Out"
    out_mask = injuries['Description'].str.startswith('Out', na=False)
    injured_out = injuries[out_mask]

    # Cross-reference by player name
    result = {}
    for _, row in injured_out.iterrows():
        player = row['Player']
        match = impact[impact['PLAYER_NAME'] == player]
        if not match.empty:
            team = match.iloc[0]['team_display']
            result.setdefault(team, []).append(player)

    return result


# ============================================================
# Feature Engineering
# ============================================================

df = read_csv()

# Keep one row per game per team (both starters/bench rows share the same full-game uppercase columns)
df = df[df['startersBench'] == 'Starters'].reset_index(drop=True)

# Parse dates and sort
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

# Derived columns
df['is_home'] = df['MATCHUP'].str.contains('vs.').astype(int)
df['WIN'] = (df['WL'] == 'W').astype(int)
df['OPP_PTS'] = (df['PTS'] - df['PLUS_MINUS']).astype(int)
df['GAME_TOTAL'] = df['PTS'] + df['OPP_PTS']

# Rolling averages of key stats (shifted by 1 to prevent data leakage)
ROLLING_WINDOW = 5
ROLLING_COLS = [
    'PTS', 'FG_PCT', 'FG3_PCT', 'FTM', 'FTA',
    'OREB', 'DREB', 'AST', 'STL', 'BLK', 'TOV',
    'offensiveRating', 'defensiveRating', 'pace',
    'effectiveFieldGoalPercentage', 'trueShootingPercentage',
    'PLUS_MINUS',
]

for col in ROLLING_COLS:
    df[f'avg_{col}'] = df.groupby('TEAM_ID')[col].transform(
        lambda x: x.shift(1).rolling(window=ROLLING_WINDOW, min_periods=3).mean()
    )

# Build opponent rolling features by matching on GAME_ID
avg_cols = [f'avg_{col}' for col in ROLLING_COLS]
opp = df[['GAME_ID', 'TEAM_ID'] + avg_cols].copy()
opp.columns = ['GAME_ID', 'OPP_TEAM_ID'] + [f'opp_{c}' for c in avg_cols]

df = df.merge(opp, on='GAME_ID')
df = df[df['TEAM_ID'] != df['OPP_TEAM_ID']].reset_index(drop=True)

# Drop rows where rolling averages haven't accumulated yet
opp_avg_cols = [f'opp_{c}' for c in avg_cols]
df = df.dropna(subset=avg_cols + opp_avg_cols).reset_index(drop=True)

# ============================================================
# Model Building
# ============================================================

feature_cols = ['is_home'] + avg_cols + [f'opp_{c}' for c in avg_cols]
X = df[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Helper to train and evaluate regression models
def evaluate_regression(name, y, X_scaled):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    }
    print(f"=== {name} ===")
    for label, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"  [{label}]")
        print(f"    MAE:  {mean_absolute_error(y_test, preds):.2f}")
        print(f"    RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.2f}")
        print(f"    R²:   {r2_score(y_test, preds):.3f}")
    print()

# Helper to train and evaluate classification models
def evaluate_classification(name, y, X_scaled):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    }
    print(f"=== {name} ===")
    for label, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"  [{label}]")
        print(f"    Accuracy: {accuracy_score(y_test, preds):.3f}")
        print(classification_report(y_test, preds, target_names=['Loss', 'Win']))
    print()

# --- Spread Model (predict PLUS_MINUS) ---
evaluate_regression('Spread Model', df['PLUS_MINUS'], X_scaled)

# --- Moneyline Model (predict Win/Loss) ---
evaluate_classification('Moneyline Model', df['WIN'], X_scaled)

# --- Team Total Points Model (predict PTS) ---
evaluate_regression('Team Total Points Model', df['PTS'], X_scaled)

# --- Game Total Points Over/Under Model (predict combined score) ---
evaluate_regression('Game Total Points (Over/Under) Model', df['GAME_TOTAL'], X_scaled)
