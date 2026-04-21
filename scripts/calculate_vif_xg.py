import pandas as pd
import numpy as np
import requests
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# 1. Download sample of events to calculate VIF
BASE_URL = "https://premier.72-60-245-2.sslip.io"
resp = requests.get(f"{BASE_URL}/matches?limit=50", timeout=30)
matches_data = resp.json()
match_ids = [m["id"] for m in matches_data.get("matches", [])]

all_events = []
for match_id in match_ids:
    r = requests.get(f"{BASE_URL}/matches/{match_id}/events?limit=5000", timeout=30)
    data = r.json()
    all_events.extend(data.get("events", []))

events = pd.DataFrame(all_events)
shots = events[events["is_shot"] == True].copy()

# 2. Feature Engineering
shots["distance_to_goal"] = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
shots["angle_to_goal"]    = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))
shots["dist_squared"]     = shots["distance_to_goal"] ** 2
shots["dist_angle"]       = shots["distance_to_goal"] * shots["angle_to_goal"]
shots["is_in_area"]       = (shots["x"] > 83).astype(int)
shots["is_central"]       = shots["y"].between(33, 67).astype(int)

# Qualifiers
def parse_qualifiers(q_list):
    if not isinstance(q_list, list):
        return ""
    return str(q_list)

shots["qualifiers_str"] = shots["qualifiers"].apply(parse_qualifiers)
q = shots["qualifiers_str"]

shots["is_big_chance"]  = q.str.contains("BigChance",  na=False).astype(int)
shots["is_header"]      = q.str.contains("'Head'",     na=False).astype(int)
shots["is_right_foot"]  = q.str.contains("RightFoot",  na=False).astype(int)
shots["is_left_foot"]   = q.str.contains("LeftFoot",   na=False).astype(int)
shots["is_counter"]     = q.str.contains("FastBreak",  na=False).astype(int)
shots["from_corner"]    = q.str.contains("FromCorner", na=False).astype(int)
shots["is_penalty"]     = q.str.contains("'Penalty'",  na=False).astype(int)
shots["is_volley"]      = q.str.contains("Volley",     na=False).astype(int)
shots["first_touch"]    = q.str.contains("FirstTouch", na=False).astype(int)
shots["is_set_piece"]   = q.str.contains("SetPiece",   na=False).astype(int)

# Select features for VIF
features = [
    "distance_to_goal", "angle_to_goal", "dist_squared", "dist_angle", 
    "is_in_area", "is_central", "is_big_chance", "is_header", 
    "is_right_foot", "is_left_foot", "is_counter", "from_corner", 
    "is_penalty", "is_volley", "first_touch", "is_set_piece"
]

X = shots[features].dropna()
X = add_constant(X)

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data.sort_values(by="VIF", ascending=False).to_string(index=False))
