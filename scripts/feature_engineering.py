import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MATCHES_FILE = RAW_DIR / "matches.csv"
RICH_EVENTS_FILE = RAW_DIR / "events_rich.csv"
FEATURES_OUTPUT = PROCESSED_DIR / "features_modelo1_a_j.csv"
BASE_URL = "https://premier.72-60-245-2.sslip.io"
MAX_WORKERS = 8


def fetch_events_for_match(match_row):
    match_id = int(match_row["id"])
    response = requests.get(f"{BASE_URL}/matches/{match_id}/events", timeout=30)
    response.raise_for_status()
    payload = response.json()
    events = payload["events"] if isinstance(payload, dict) else payload
    for event in events:
        qualifiers = event.get("qualifiers", [])
        event["qualifiers"] = json.dumps(qualifiers, ensure_ascii=True)
        event["home_team"] = match_row["home_team"]
        event["away_team"] = match_row["away_team"]
        event["match_date"] = match_row["date"]
        event["referee"] = match_row["referee"]
    return events


def ensure_rich_events(matches):
    if RICH_EVENTS_FILE.exists():
        cached = pd.read_csv(RICH_EVENTS_FILE)
        if "qualifiers" in cached.columns:
            return cached

    all_events = []
    records = matches.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_events_for_match, row) for row in records]
        for future in as_completed(futures):
            all_events.extend(future.result())

    events = pd.DataFrame(all_events)
    RICH_EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(RICH_EVENTS_FILE, index=False)
    return events


def qualifier_names(cell):
    try:
        parsed = json.loads(cell) if isinstance(cell, str) else cell
    except json.JSONDecodeError:
        parsed = []
    if not isinstance(parsed, list):
        return set()
    names = set()
    for item in parsed:
        display_name = ((item or {}).get("type") or {}).get("displayName")
        if display_name:
            names.add(display_name)
    return names


def build_standard_features(shots):
    shots["distance_to_goal"] = np.sqrt((100 - shots["x"]) ** 2 + (50 - shots["y"]) ** 2)
    shots["angle_to_goal"] = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))
    shots["dist_squared"] = shots["distance_to_goal"] ** 2
    shots["dist_angle"] = shots["distance_to_goal"] * shots["angle_to_goal"]
    shots["is_in_area"] = (shots["x"] > 83).astype(int)
    shots["is_central"] = shots["y"].between(33, 67, inclusive="both").astype(int)

    qualifier_map = {
        "is_big_chance": "BigChance",
        "is_header": "Head",
        "is_right_foot": "RightFoot",
        "is_left_foot": "LeftFoot",
        "is_counter": "FastBreak",
        "from_corner": "FromCorner",
        "is_penalty": "Penalty",
        "is_volley": "Volley",
        "first_touch": "FirstTouch",
        "is_set_piece": "SetPiece",
    }
    for column, token in qualifier_map.items():
        shots[column] = shots["qualifier_names"].apply(lambda names, t=token: int(t in names))

    shots["team_is_home"] = (shots["team_name"] == shots["home_team"]).astype(int)
    return shots


def compute_defensive_pressure(events, shots, radius=10):
    grouped = {}
    for (match_id, minute), chunk in events.groupby(["match_id", "minute"], sort=False):
        team_coords = {}
        for team_name, team_chunk in chunk.groupby("team_name", sort=False):
            coords = team_chunk[["x", "y"]].dropna().to_numpy(dtype=float)
            team_coords[team_name] = coords
        grouped[(match_id, minute)] = team_coords

    pressure = []
    for row in shots.itertuples():
        team_coords = grouped.get((row.match_id, row.minute), {})
        nearby_count = 0
        for team_name, coords in team_coords.items():
            if team_name == row.team_name or len(coords) == 0:
                continue
            dx = np.abs(coords[:, 0] - row.x)
            dy = np.abs(coords[:, 1] - row.y)
            nearby_count += int(((dx < radius) & (dy < radius)).sum())
        pressure.append(nearby_count)
    return pressure


def compute_buildup_features(events, shots, window_secs=60):
    result = pd.DataFrame(index=shots.index)
    result["buildup_passes"] = 0
    result["buildup_unique_players"] = 0

    events = events.copy()
    shots = shots.copy()
    events["event_seconds"] = events["minute"].fillna(0) * 60 + events["second"].fillna(0)
    shots["event_seconds"] = shots["minute"].fillna(0) * 60 + shots["second"].fillna(0)

    for (match_id, team_name), team_events in events.groupby(["match_id", "team_name"], sort=False):
        team_shots = shots[(shots["match_id"] == match_id) & (shots["team_name"] == team_name)]
        if team_shots.empty:
            continue

        team_events = team_events.sort_values(["event_seconds", "id"]).copy()
        event_seconds = team_events["event_seconds"].to_numpy(dtype=float)
        successful_pass = (
            (team_events["event_type"] == "Pass")
            & (team_events["outcome"] == "Successful")
        ).to_numpy()
        player_ids = team_events["player_id"].to_numpy()
        event_ids = team_events["id"].to_numpy()

        for shot in team_shots.sort_values(["event_seconds", "id"]).itertuples():
            mask = (
                (event_seconds < shot.event_seconds)
                & (event_seconds >= shot.event_seconds - window_secs)
                & (event_ids != shot.id)
            )
            result.at[shot.Index, "buildup_passes"] = int((successful_pass & mask).sum())
            players_window = player_ids[mask]
            players_window = players_window[~pd.isna(players_window)]
            result.at[shot.Index, "buildup_unique_players"] = int(pd.Series(players_window).nunique())

    result["buildup_decentralized"] = (result["buildup_unique_players"] > 3).astype(int)
    return result


def classify_goalmouth_zone(row):
    if pd.isna(row["goal_mouth_y"]) or pd.isna(row["goal_mouth_z"]):
        return "unknown"

    if row["goal_mouth_y"] < 33:
        y_zone = "left"
    elif row["goal_mouth_y"] > 67:
        y_zone = "right"
    else:
        y_zone = "center"

    if row["goal_mouth_z"] < 33:
        z_zone = "low"
    elif row["goal_mouth_z"] > 67:
        z_zone = "high"
    else:
        z_zone = "mid"

    return f"{y_zone}_{z_zone}"


def build_provisional_xg(shots):
    model_features = [
        "distance_to_goal",
        "angle_to_goal",
        "dist_squared",
        "dist_angle",
        "is_in_area",
        "is_central",
        "is_big_chance",
        "is_header",
        "is_right_foot",
        "is_left_foot",
        "is_counter",
        "from_corner",
        "is_penalty",
        "is_volley",
        "first_touch",
        "is_set_piece",
        "defensive_pressure",
        "buildup_passes",
        "buildup_unique_players",
        "buildup_decentralized",
        "porteria_zone",
    ]

    numeric_features = [col for col in model_features if col != "porteria_zone"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["porteria_zone"],
            ),
        ]
    )
    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    clf.fit(shots[model_features], shots["is_goal"].astype(int))
    return clf.predict_proba(shots[model_features])[:, 1]


def build_team_match_frame(matches):
    base_columns = [
        "match_id",
        "date",
        "referee",
        "team_name",
        "opponent_team",
        "goals_for",
        "goals_against",
        "yellow_for",
        "yellow_against",
    ]
    home = matches[
        ["id", "date", "referee", "home_team", "away_team", "fthg", "ftag", "hy", "ay"]
    ].copy()
    home.columns = base_columns
    home["team_is_home"] = 1

    away = matches[
        ["id", "date", "referee", "away_team", "home_team", "ftag", "fthg", "ay", "hy"]
    ].copy()
    away.columns = base_columns
    away["team_is_home"] = 0

    team_match = pd.concat([home, away], ignore_index=True)
    team_match["date"] = pd.to_datetime(team_match["date"], dayfirst=True, errors="coerce")
    return team_match.sort_values(["team_name", "date", "match_id"]).reset_index(drop=True)


def compute_team_context(matches, events, shots):
    team_match = build_team_match_frame(matches)

    match_shot_summary = shots.groupby(["match_id", "team_name"]).agg(
        provisional_xg=("xg_provisional", "sum"),
        shots_total=("id", "count"),
    )

    successful_passes = events[
        (events["event_type"] == "Pass") & (events["outcome"] == "Successful")
    ]
    pass_decentralization = successful_passes.groupby(["match_id", "team_name"])["player_id"].nunique()

    altitude = events[
        events["event_type"].isin(["Pass", "BallRecovery", "Tackle", "Interception"])
    ].groupby(["match_id", "team_name"])["x"].mean()

    goal_events = events[events["is_goal"].astype(bool)].copy()
    late_goals = goal_events[goal_events["minute"] >= 75].groupby(["match_id", "team_name"]).size()
    early_goals = goal_events[goal_events["minute"] < 75].groupby(["match_id", "team_name"]).size()

    ppda_rows = []
    for match_id, match_events in events.groupby("match_id", sort=False):
        teams = match_events["team_name"].dropna().unique().tolist()
        for team in teams:
            opp_passes = match_events[
                (match_events["team_name"] != team)
                & (match_events["event_type"] == "Pass")
                & (match_events["outcome"] == "Successful")
                & (match_events["x"] > 40)
            ]
            def_actions = match_events[
                (match_events["team_name"] == team)
                & (
                    match_events["event_type"].isin(
                        ["Tackle", "Interception", "BlockedPass", "Clearance"]
                    )
                )
                & (match_events["x"] > 40)
            ]
            ppda_rows.append(
                {
                    "match_id": match_id,
                    "team_name": team,
                    "ppda_proxy": len(opp_passes) / max(len(def_actions), 1),
                }
            )
    ppda = pd.DataFrame(ppda_rows).set_index(["match_id", "team_name"])

    team_match = team_match.merge(
        match_shot_summary, how="left", left_on=["match_id", "team_name"], right_index=True
    )
    team_match = team_match.merge(
        pass_decentralization.rename("pass_decentralization"),
        how="left",
        left_on=["match_id", "team_name"],
        right_index=True,
    )
    team_match = team_match.merge(
        altitude.rename("altitude_of_play"),
        how="left",
        left_on=["match_id", "team_name"],
        right_index=True,
    )
    team_match = team_match.merge(
        late_goals.rename("late_goals"),
        how="left",
        left_on=["match_id", "team_name"],
        right_index=True,
    )
    team_match = team_match.merge(
        early_goals.rename("early_goals"),
        how="left",
        left_on=["match_id", "team_name"],
        right_index=True,
    )
    team_match = team_match.merge(
        ppda,
        how="left",
        left_on=["match_id", "team_name"],
        right_index=True,
    )

    fill_zero = [
        "provisional_xg",
        "shots_total",
        "pass_decentralization",
        "altitude_of_play",
        "late_goals",
        "early_goals",
        "ppda_proxy",
    ]
    team_match[fill_zero] = team_match[fill_zero].fillna(0)

    ref_stats = matches.groupby("referee").agg(
        home_win_rate=("ftr", lambda x: (x == "H").mean()),
        home_bias=("ay", "mean"),
        referee_home_yellow=("hy", "mean"),
    )
    ref_stats["home_bias"] = ref_stats["home_bias"] - ref_stats["referee_home_yellow"]
    ref_stats = ref_stats.drop(columns=["referee_home_yellow"])

    team_match = team_match.merge(ref_stats, how="left", left_on="referee", right_index=True)
    grouped = team_match.groupby("team_name", sort=False)
    team_match["home_xg_debt_5"] = grouped["provisional_xg"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean()
    ) - grouped["goals_for"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    team_match["momentum"] = grouped["goals_for"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ) - grouped["goals_for"].transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())

    late_roll = grouped["late_goals"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).sum())
    early_roll = grouped["early_goals"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).sum())
    team_match["clutch_ratio"] = late_roll / early_roll.replace(0, np.nan)
    team_match["ppda"] = team_match["ppda_proxy"]

    return team_match[
        [
            "match_id",
            "team_name",
            "home_xg_debt_5",
            "ppda",
            "pass_decentralization",
            "momentum",
            "home_win_rate",
            "home_bias",
            "altitude_of_play",
            "clutch_ratio",
        ]
    ]


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    matches = pd.read_csv(MATCHES_FILE)
    events = ensure_rich_events(matches)

    events["qualifiers"] = events["qualifiers"].fillna("[]").astype(str)
    events["qualifier_names"] = events["qualifiers"].apply(qualifier_names)

    shots = events[events["is_shot"].astype(bool)].copy()
    shots = build_standard_features(shots)

    shots["defensive_pressure"] = compute_defensive_pressure(events, shots)

    buildup = compute_buildup_features(events, shots)
    shots = shots.join(buildup)

    shots["porteria_zone"] = shots.apply(classify_goalmouth_zone, axis=1)
    porteria_dummies = pd.get_dummies(shots["porteria_zone"], prefix="porteria_zone", dtype=int)
    shots = pd.concat([shots, porteria_dummies], axis=1)

    max_distance = max(shots["distance_to_goal"].max(), 1)
    shots["shot_quality_index"] = (
        shots["is_big_chance"] * 0.38
        + shots["is_in_area"] * 0.18
        + shots["is_counter"] * 0.12
        + shots["is_central"] * 0.10
        + (1 - shots["distance_to_goal"] / max_distance) * 0.22
    )

    shots["xg_provisional"] = build_provisional_xg(shots)

    team_context = compute_team_context(matches, events, shots)
    shots = shots.merge(team_context, how="left", on=["match_id", "team_name"])

    output = shots.copy()
    if "qualifier_names" in output.columns:
        output["qualifier_names"] = output["qualifier_names"].apply(lambda x: "|".join(sorted(x)))

    output.to_csv(FEATURES_OUTPUT, index=False)
    print(f"Tabla consolidada creada en: {FEATURES_OUTPUT}")
    print(f"Filas: {len(output)} | Columnas: {len(output.columns)}")


if __name__ == "__main__":
    main()
