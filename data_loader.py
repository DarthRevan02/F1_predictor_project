import fastf1
import pandas as pd
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

# Resolve cache path relative to this file — works on any machine (fixes flaw #1)
_CACHE_DIR = Path(__file__).parent / 'f1_cache'
_CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(_CACHE_DIR))


class F1DataLoader:
    """Handles loading F1 race data from FastF1 API"""

    def __init__(self, year=2024):
        self.year = year
        self.schedule = None

    def get_completed_races(self):
        """
        Get list of completed race round numbers for this season.

        Returns:
            list[int]: Round numbers of completed races.
        """
        schedule = fastf1.get_event_schedule(self.year)
        self.schedule = schedule

        current_time = datetime.now(timezone.utc)
        completed_races = []

        for _, event in schedule.iterrows():
            if pd.notna(event['EventDate']) and event['EventFormat'] != 'testing':
                event_date = event['EventDate']
                if event_date.tzinfo is None:
                    event_date = event_date.replace(tzinfo=timezone.utc)
                if event_date < current_time:
                    completed_races.append(event['RoundNumber'])

        return completed_races

    def load_race_data(self, num_races=None, load_all=False):
        """
        Load race results from completed races.

        Args:
            num_races (int | None): Number of most-recent races to load.
            load_all (bool): If True, load all completed races.

        Returns:
            pd.DataFrame: Race results, or empty DataFrame if none found.
        """
        all_data = []

        print(f"Loading F1 data for {self.year}...")

        completed_races = self.get_completed_races()

        if not completed_races:
            print("⚠️  No completed races found for this season yet")
            return pd.DataFrame()

        print(f"✓ Found {len(completed_races)} completed race(s)")

        if load_all or num_races is None:
            races_to_load = completed_races
        else:
            races_to_load = completed_races[-num_races:]

        print(f"📊 Loading {len(races_to_load)} race(s)...")

        for round_num in races_to_load:
            try:
                session = fastf1.get_session(self.year, round_num, 'R')
                session.load()
                results = session.results
                event_name = session.event['EventName']

                for _, row in results.iterrows():
                    all_data.append({
                        'Race': event_name,
                        'RoundNumber': round_num,
                        'Driver': row['Abbreviation'],
                        'Team': row['TeamName'],
                        'GridPosition': row['GridPosition'],
                        'Position': row['Position'],
                        'Points': row['Points'],
                    })
                print(f"✓ Loaded Round {round_num}: {event_name}")
            except Exception as e:
                print(f"✗ Error loading round {round_num}: {e}")
                continue

        df = pd.DataFrame(all_data)
        print(f"\n✓ Loaded {len(df)} results from {len(races_to_load)} race(s)")
        return df

    def get_schedule(self):
        """Return the event schedule for this year."""
        if self.schedule is None:
            self.schedule = fastf1.get_event_schedule(self.year)
        return self.schedule
