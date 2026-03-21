import csv
import json
import os

def clean_team(team):
    team_map = {
        "Nott'm Forest": "Nottingham Forest",
        "Spurs": "Tottenham",
        "Tottenham": "Tottenham",
        "Nottingham Forest": "Nottingham Forest"
    }
    return team_map.get(team, team)

def create_player_mapping(players_path, events_path):
    print(f"Reading players from {players_path}...")
    players_data = []
    with open(players_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['team_clean'] = clean_team(row.get('team', ''))
            row['first_name'] = row.get('first_name', '')
            row['second_name'] = row.get('second_name', '')
            row['full_name'] = f"{row['first_name']} {row['second_name']}".strip()
            players_data.append(row)
            
    print(f"Reading events from {events_path} (unique players only)...")
    event_players = {} # whoscored_id -> {name, team}
    with open(events_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row.get('player_id')
            if eid and eid not in event_players:
                event_players[eid] = {
                    "name": row.get('player_name'),
                    "team": clean_team(row.get('team_name'))
                }
    
    mapping = {}
    found = 0
    not_found = []

    print(f"Mapping {len(event_players)} players found in events...")
    for eid, info in event_players.items():
        e_name = info['name']
        e_team = info['team']
        
        # Priority 1: Exact Match Full Name + Team
        matches = [p for p in players_data if p['full_name'] == e_name and p['team_clean'] == e_team]
        
        # Priority 2: Exact Match Web Name + Team
        if not matches:
            matches = [p for p in players_data if p['web_name'] == e_name and p['team_clean'] == e_team]
            
        # Priority 3: Case-insensitive Match Full Name + Team
        if not matches:
            matches = [p for p in players_data if p['full_name'].lower() == e_name.lower() and p['team_clean'] == e_team]

        # Priority 4: "Second Name" in WhoScored Name + Team (e.g. Gabriel Magalhães -> Gabriel)
        if not matches:
            matches = [p for p in players_data if p['team_clean'] == e_team and 
                      (p['second_name'] in e_name and p['second_name'] != '')]

        # Priority 5: Partial match as fallback
        if not matches:
            matches = [p for p in players_data if p['team_clean'] == e_team and 
                      (e_name in p['full_name'] or p['full_name'] in e_name)]

        if matches:
            # If multiple matches, pick the one with the most similar name length or first one
            p = matches[0]
            # Special case for Gabriel Jesus / Gabriel Magalhaes if ambiguous
            if len(matches) > 1:
                # pick the one where full name shares more words
                e_words = set(e_name.lower().split())
                best_match = matches[0]
                best_score = -1
                for cand in matches:
                    cand_words = set(cand['full_name'].lower().split())
                    score = len(e_words.intersection(cand_words))
                    if score > best_score:
                        best_score = score
                        best_match = cand
                p = best_match

            mapping[eid] = {
                "fpl_id": p['id'],
                "fpl_name": p['web_name'],
                "fpl_full_name": p['full_name'],
                "whoscored_name": e_name,
                "team": e_team
            }
            found += 1
        else:
            not_found.append({"id": eid, "name": e_name, "team": e_team})
            
    output_path = "/home/camilo/proyectos/Modelo_apuestas_Futbol/data/processed/player_id_map.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
        
    print(f"Finished: {found} players mapped. {len(not_found)} not found.")
    return output_path

if __name__ == "__main__":
    players_csv = "/home/camilo/proyectos/Modelo_apuestas_Futbol/data/raw/players.csv"
    events_csv = "/home/camilo/proyectos/Modelo_apuestas_Futbol/data/raw/events.csv"
    create_player_mapping(players_csv, events_csv)
