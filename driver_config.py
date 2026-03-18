"""
Single source of truth for all driver/team data.
Edit this file to update ratings — never edit individual prediction modules.
"""

# 2025 Championship standings — update after each race
CURRENT_STANDINGS = {
    'NOR': 390,
    'PIA': 366,
    'VER': 341,   # README had 341; app.py had 366 — using README value (correct)
    'RUS': 294,
    'LEC': 226,
    'HAM': 152,
    'ANT': 137,
    'ALB': 73,
    'HAD': 51,
    'HUL': 49,
    'SAI': 48,
    'BEA': 41,
    'ALO': 40,
    'LAW': 36,
    'OCO': 32,
    'STR': 32,
    'TSU': 28,
    'GAS': 22,
    'BOR': 19,   # Gabriel Bortoleto (Kick Sauber) — was misspelled "Borghesi" in README
    'DOO': 8,    # Jack Doohan (Alpine) — was missing from app.py
}

# 2025 driver → team mapping (single source)
DRIVER_TEAM_MAP = {
    'HAM': 'Ferrari',        'LEC': 'Ferrari',
    'VER': 'Red Bull Racing', 'LAW': 'Red Bull Racing',
    'NOR': 'McLaren',        'PIA': 'McLaren',
    'RUS': 'Mercedes',       'ANT': 'Mercedes',
    'ALO': 'Aston Martin',   'STR': 'Aston Martin',
    'GAS': 'Alpine',         'DOO': 'Alpine',
    'OCO': 'Haas F1 Team',   'BEA': 'Haas F1 Team',
    'TSU': 'RB',             'HAD': 'RB',
    'SAI': 'Williams',       'ALB': 'Williams',
    'HUL': 'Kick Sauber',    'BOR': 'Kick Sauber',
}

# Race performance ratings (0.0–1.0).
# These are used by every prediction module — one edit updates all of them.
DRIVER_PERFORMANCE = {
    'NOR': 0.95,
    'PIA': 0.93,
    'LEC': 0.92,
    'HAM': 0.91,
    'VER': 0.90,
    'RUS': 0.88,
    'SAI': 0.85,
    'ALO': 0.84,
    'GAS': 0.82,
    'STR': 0.81,
    'LAW': 0.79,
    'OCO': 0.78,
    'TSU': 0.76,
    'ALB': 0.75,
    'HUL': 0.73,
    'ANT': 0.72,
    'BEA': 0.70,
    'DOO': 0.69,
    'HAD': 0.68,
    'BOR': 0.65,
}

# Qualifying-specific skill ratings
DRIVER_QUALI_SKILL = {
    'NOR': 0.98, 'LEC': 0.97, 'PIA': 0.95, 'HAM': 0.94,
    'VER': 0.93, 'RUS': 0.92, 'SAI': 0.89, 'ALO': 0.88,
    'GAS': 0.87, 'STR': 0.85, 'LAW': 0.84, 'OCO': 0.83,
    'TSU': 0.82, 'ALB': 0.81, 'HUL': 0.80, 'ANT': 0.78,
    'BEA': 0.75, 'DOO': 0.73, 'HAD': 0.71, 'BOR': 0.68,
}

# Team qualifying strength (affects grid predictions)
TEAM_QUALI_STRENGTH = {
    'McLaren': 0.95,
    'Ferrari': 0.93,
    'Red Bull Racing': 0.91,
    'Mercedes': 0.89,
    'Aston Martin': 0.80,
    'Alpine': 0.75,
    'Haas F1 Team': 0.72,
    'RB': 0.70,
    'Williams': 0.68,
    'Kick Sauber': 0.63,
}

# Team reliability (lower = more likely to DNF)
TEAM_RELIABILITY = {
    'McLaren': 0.03,
    'Ferrari': 0.04,
    'Red Bull Racing': 0.04,
    'Mercedes': 0.05,
    'Aston Martin': 0.07,
    'Alpine': 0.08,
    'Haas F1 Team': 0.08,
    'RB': 0.09,
    'Williams': 0.09,
    'Kick Sauber': 0.10,
}

# Estimated starting grid positions by team (for race-winner predictor baseline)
TEAM_GRID_BASE = {
    'McLaren': 2,
    'Ferrari': 3,
    'Red Bull Racing': 4,
    'Mercedes': 6,
    'Aston Martin': 9,
    'Alpine': 12,
    'Haas F1 Team': 13,
    'RB': 15,
    'Williams': 17,
    'Kick Sauber': 19,
}

F1_POINTS_SYSTEM = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
