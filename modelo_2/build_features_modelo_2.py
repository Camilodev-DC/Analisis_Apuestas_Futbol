import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_MATCHES = ROOT / "data" / "raw" / "matches.csv"
RAW_EVENTS = ROOT / "data" / "raw" / "events_rich.csv"
SHOT_FEATURES = ROOT / "data" / "processed" / "features_modelo1_a_j.csv"
MODEL_1 = ROOT / "modelo 1" / "artifacts" / "modelo_1_xg_logistic.joblib"
ARTIFACTS_DIR = ROOT / "modelo_2" / "artifacts"
FEATURES_FILE = ARTIFACTS_DIR / "features_modelo_2.csv"
ROLLING_WINDOW = 5

MODEL_1_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "is_big_chance",
    "defensive_pressure",
    "buildup_passes",
    "buildup_unique_players",
    "buildup_decentralized",
    "first_touch",
]

MATCH_ROLLING_SOURCE_COLS = [
    "goals_for",
    "goals_against",
    "shots_for",
    "shots_against",
    "sot_for",
    "sot_against",
    "corners_for",
    "cards_for",
    "points",
]

EVENT_ROLLING_SOURCE_COLS = [
    "xg_for",
    "pass_accuracy",
    "progressive_passes",
    "big_chances",
    "high_press_pct",
    "crosses",
]


def load_matches():
    matches = pd.read_csv(RAW_MATCHES)
    matches["date_dt"] = pd.to_datetime(matches["date"], dayfirst=True, errors="coerce")
    matches = matches.sort_values(["date_dt", "id"]).reset_index(drop=True)
    return matches


def build_xg_by_team_match():
    shots = pd.read_csv(SHOT_FEATURES)
    model = joblib.load(MODEL_1)
    shots["xg_modelo1"] = model.predict_proba(shots[MODEL_1_FEATURES])[:, 1]
    grouped = shots.groupby(["match_id", "team_name"], as_index=False).agg(
        xg_for=("xg_modelo1", "sum"),
        big_chances=("is_big_chance", "sum"),
        shots_model1=("id", "count"),
        mean_xg_per_shot=("xg_modelo1", "mean"),
    )
    return grouped


def build_event_team_aggregates():
    usecols = [
        "match_id",
        "team_name",
        "event_type",
        "outcome",
        "x",
        "end_x",
        "qualifiers",
    ]
    events = pd.read_csv(RAW_EVENTS, usecols=usecols)
    events["qualifiers"] = events["qualifiers"].fillna("[]").astype(str)

    is_pass = events["event_type"] == "Pass"
    is_pass_ok = is_pass & (events["outcome"] == "Successful")
    is_progressive = is_pass_ok & ((events["end_x"] - events["x"]) >= 15) & (events["end_x"] > events["x"])
    is_cross = is_pass & events["qualifiers"].str.contains("Cross", na=False)

    defensive_actions = events["event_type"].isin(
        ["Tackle", "Interception", "BallRecovery", "BlockedPass", "Clearance"]
    )
    high_press_actions = defensive_actions & (events["x"] > 60)

    events = events.assign(
        pass_attempt=is_pass.astype(int),
        pass_completed=is_pass_ok.astype(int),
        progressive_pass=is_progressive.astype(int),
        cross=is_cross.astype(int),
        defensive_action=defensive_actions.astype(int),
        high_press_action=high_press_actions.astype(int),
    )

    agg = events.groupby(["match_id", "team_name"], as_index=False).agg(
        passes_attempted=("pass_attempt", "sum"),
        passes_completed=("pass_completed", "sum"),
        progressive_passes=("progressive_pass", "sum"),
        crosses=("cross", "sum"),
        defensive_actions=("defensive_action", "sum"),
        high_press_actions=("high_press_action", "sum"),
    )

    agg["pass_accuracy"] = agg["passes_completed"] / agg["passes_attempted"].replace(0, np.nan)
    agg["high_press_pct"] = agg["high_press_actions"] / agg["defensive_actions"].replace(0, np.nan)
    agg["pass_accuracy"] = agg["pass_accuracy"].fillna(0)
    agg["high_press_pct"] = agg["high_press_pct"].fillna(0)
    return agg


def build_team_match_long(matches, team_events):
    home = matches[
        [
            "match_id",
            "date_dt",
            "referee",
            "home_team",
            "away_team",
            "fthg",
            "ftag",
            "hs",
            "as_",
            "hst",
            "ast",
            "hc",
            "hy",
            "hr",
            "ftr",
        ]
    ].copy()
    home.columns = [
        "match_id",
        "date_dt",
        "referee",
        "team_name",
        "opponent_name",
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "sot_for",
        "sot_against",
        "corners_for",
        "yellow_for",
        "red_for",
        "ftr",
    ]
    home["is_home"] = 1

    away = matches[
        [
            "match_id",
            "date_dt",
            "referee",
            "away_team",
            "home_team",
            "ftag",
            "fthg",
            "as_",
            "hs",
            "ast",
            "hst",
            "ac",
            "ay",
            "ar",
            "ftr",
        ]
    ].copy()
    away.columns = [
        "match_id",
        "date_dt",
        "referee",
        "team_name",
        "opponent_name",
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_against",
        "sot_for",
        "sot_against",
        "corners_for",
        "yellow_for",
        "red_for",
        "ftr",
    ]
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)
    long_df["cards_for"] = long_df["yellow_for"] + 2 * long_df["red_for"]

    long_df["points"] = np.select(
        [
            (long_df["is_home"] == 1) & (long_df["ftr"] == "H"),
            (long_df["is_home"] == 0) & (long_df["ftr"] == "A"),
            long_df["ftr"] == "D",
        ],
        [3, 3, 1],
        default=0,
    )

    long_df = long_df.merge(team_events, how="left", on=["match_id", "team_name"])
    fill_zero_cols = [
        "xg_for",
        "big_chances",
        "shots_model1",
        "mean_xg_per_shot",
        "passes_attempted",
        "passes_completed",
        "progressive_passes",
        "crosses",
        "defensive_actions",
        "high_press_actions",
        "pass_accuracy",
        "high_press_pct",
    ]
    for col in fill_zero_cols:
        if col in long_df.columns:
            long_df[col] = long_df[col].fillna(0)
    return long_df.sort_values(["team_name", "date_dt", "match_id"]).reset_index(drop=True)


def add_team_rollings(team_match):
    grouped = team_match.groupby("team_name", sort=False)
    for col in MATCH_ROLLING_SOURCE_COLS + EVENT_ROLLING_SOURCE_COLS:
        team_match[f"{col}_avg{ROLLING_WINDOW}"] = grouped[col].transform(
            lambda x: x.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()
        )
    return team_match


def add_referee_prematch_features(matches):
    stats = {}
    home_win_rate = []
    avg_total_goals = []
    card_bias = []

    for row in matches.itertuples():
        ref = row.referee
        current = stats.get(ref, {"matches": 0, "home_wins": 0, "total_goals": 0.0, "card_bias": 0.0})
        if current["matches"] == 0:
            home_win_rate.append(0.0)
            avg_total_goals.append(matches["total_goals"].mean())
            card_bias.append(0.0)
        else:
            home_win_rate.append(current["home_wins"] / current["matches"])
            avg_total_goals.append(current["total_goals"] / current["matches"])
            card_bias.append(current["card_bias"] / current["matches"])

        current["matches"] += 1
        current["home_wins"] += int(row.ftr == "H")
        current["total_goals"] += float(row.total_goals)
        current["card_bias"] += float(row.ay - row.hy)
        stats[ref] = current

    matches["ref_home_win_rate_pre"] = home_win_rate
    matches["ref_avg_total_goals_pre"] = avg_total_goals
    matches["ref_card_bias_pre"] = card_bias
    return matches


def add_strength_features(matches):
    table = {}
    home_ppg = []
    away_ppg = []
    home_gdpg = []
    away_gdpg = []
    home_strength = []
    away_strength = []

    teams = pd.unique(matches[["home_team", "away_team"]].values.ravel("K"))
    for team in teams:
        table[team] = {"played": 0, "points": 0.0, "goal_diff": 0.0}

    for row in matches.itertuples():
        h = table[row.home_team]
        a = table[row.away_team]

        h_ppg = h["points"] / h["played"] if h["played"] else 0.0
        a_ppg = a["points"] / a["played"] if a["played"] else 0.0
        h_gdpg = h["goal_diff"] / h["played"] if h["played"] else 0.0
        a_gdpg = a["goal_diff"] / a["played"] if a["played"] else 0.0

        home_ppg.append(h_ppg)
        away_ppg.append(a_ppg)
        home_gdpg.append(h_gdpg)
        away_gdpg.append(a_gdpg)
        home_strength.append(h_ppg + 0.25 * h_gdpg)
        away_strength.append(a_ppg + 0.25 * a_gdpg)

        h_points = 3 if row.ftr == "H" else 1 if row.ftr == "D" else 0
        a_points = 3 if row.ftr == "A" else 1 if row.ftr == "D" else 0
        h["played"] += 1
        a["played"] += 1
        h["points"] += h_points
        a["points"] += a_points
        h["goal_diff"] += row.fthg - row.ftag
        a["goal_diff"] += row.ftag - row.fthg

    matches["home_points_per_game_pre"] = home_ppg
    matches["away_points_per_game_pre"] = away_ppg
    matches["home_goal_diff_per_game_pre"] = home_gdpg
    matches["away_goal_diff_per_game_pre"] = away_gdpg
    matches["home_strength_rating"] = home_strength
    matches["away_strength_rating"] = away_strength
    matches["strength_rating_diff"] = matches["home_strength_rating"] - matches["away_strength_rating"]
    return matches


def pivot_team_rollings(matches, team_match):
    cols = [c for c in team_match.columns if c.endswith(f"_avg{ROLLING_WINDOW}")]

    home_roll = team_match[["match_id", "team_name"] + cols].copy()
    home_roll = home_roll.rename(columns={"team_name": "home_team", **{c: f"home_{c}" for c in cols}})

    away_roll = team_match[["match_id", "team_name"] + cols].copy()
    away_roll = away_roll.rename(columns={"team_name": "away_team", **{c: f"away_{c}" for c in cols}})

    matches = matches.merge(home_roll, how="left", on=["match_id", "home_team"])
    matches = matches.merge(away_roll, how="left", on=["match_id", "away_team"])
    return matches


def finalize_features(matches):
    matches["bookmaker_spread_home"] = matches[["b365h", "bwh", "maxh", "avgh"]].std(axis=1)
    matches["bookmaker_spread_draw"] = matches[["b365d", "bwd", "maxd", "avgd"]].std(axis=1)
    matches["bookmaker_spread_away"] = matches[["b365a", "bwa", "maxa", "avga"]].std(axis=1)

    prob_cols = ["implied_prob_h", "implied_prob_d", "implied_prob_a"]
    prob_sum = matches[prob_cols].sum(axis=1).replace(0, np.nan)
    norm_probs = matches[prob_cols].div(prob_sum, axis=0).clip(lower=1e-9)
    matches["market_entropy"] = -(norm_probs * np.log(norm_probs)).sum(axis=1)

    diff_pairs = [
        ("goals_for_avg5", "goals_form_diff"),
        ("goals_against_avg5", "goals_conceded_diff"),
        ("sot_for_avg5", "sot_form_diff"),
        ("xg_for_avg5", "xg_form_diff"),
        ("pass_accuracy_avg5", "pass_accuracy_diff"),
        ("progressive_passes_avg5", "progressive_passes_diff"),
        ("big_chances_avg5", "big_chances_diff"),
        ("high_press_pct_avg5", "high_press_pct_diff"),
        ("crosses_avg5", "crosses_diff"),
        ("points_avg5", "points_form_diff"),
    ]
    for base_name, diff_name in diff_pairs:
        matches[diff_name] = matches[f"home_{base_name}"] - matches[f"away_{base_name}"]

    feature_cols = [
        "implied_prob_h",
        "implied_prob_d",
        "implied_prob_a",
        "bookmaker_spread_home",
        "bookmaker_spread_draw",
        "bookmaker_spread_away",
        "market_entropy",
        "ref_home_win_rate_pre",
        "ref_avg_total_goals_pre",
        "ref_card_bias_pre",
        "home_strength_rating",
        "away_strength_rating",
        "strength_rating_diff",
        "home_points_per_game_pre",
        "away_points_per_game_pre",
        "home_goal_diff_per_game_pre",
        "away_goal_diff_per_game_pre",
        "home_goals_for_avg5",
        "away_goals_for_avg5",
        "home_goals_against_avg5",
        "away_goals_against_avg5",
        "home_sot_for_avg5",
        "away_sot_for_avg5",
        "home_pass_accuracy_avg5",
        "away_pass_accuracy_avg5",
        "home_progressive_passes_avg5",
        "away_progressive_passes_avg5",
        "home_big_chances_avg5",
        "away_big_chances_avg5",
        "home_high_press_pct_avg5",
        "away_high_press_pct_avg5",
        "home_crosses_avg5",
        "away_crosses_avg5",
        "home_xg_for_avg5",
        "away_xg_for_avg5",
        "goals_form_diff",
        "goals_conceded_diff",
        "sot_form_diff",
        "xg_form_diff",
        "pass_accuracy_diff",
        "progressive_passes_diff",
        "big_chances_diff",
        "high_press_pct_diff",
        "crosses_diff",
        "points_form_diff",
    ]

    keep_cols = [
        "match_id",
        "date_dt",
        "home_team",
        "away_team",
        "referee",
        "ftr",
        "total_goals",
    ] + feature_cols

    final_df = matches[keep_cols].rename(columns={"date_dt": "match_date"})
    return final_df, feature_cols


def build_features_dataset():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    matches = matches.rename(columns={"id": "match_id"})

    xg_by_team = build_xg_by_team_match()
    event_team = build_event_team_aggregates()
    team_events = event_team.merge(xg_by_team, how="left", on=["match_id", "team_name"])
    for col in ["xg_for", "big_chances", "shots_model1", "mean_xg_per_shot"]:
        team_events[col] = team_events[col].fillna(0)

    team_match = build_team_match_long(matches, team_events)
    team_match = add_team_rollings(team_match)

    matches = add_referee_prematch_features(matches)
    matches = add_strength_features(matches)
    matches = pivot_team_rollings(matches, team_match)

    features_df, feature_cols = finalize_features(matches)
    features_df.to_csv(FEATURES_FILE, index=False)
    return features_df, feature_cols


def main():
    features_df, feature_cols = build_features_dataset()
    meta = {
        "rows": int(len(features_df)),
        "feature_count": int(len(feature_cols)),
        "features_file": str(FEATURES_FILE),
    }
    (ARTIFACTS_DIR / "features_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Features Modelo 2 guardadas en: {FEATURES_FILE}")
    print(f"Filas: {len(features_df)} | Features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
