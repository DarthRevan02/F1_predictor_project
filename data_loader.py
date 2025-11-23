import fastf1
import pandas as pd
import warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

# Enable FastF1 cache
fastf1.Cache.enable_cache('C:\\Users\\Aadi Jain\\Desktop\\AI-ML\\f1_predictor_project\\f1_cache')

class F1DataLoader:
    """Handles loading F1 race data from FastF1 API"""
    
    def __init__(self, year=2024):
        self.year = year
        self.schedule = None
        
    def get_completed_races(self):
        """
        Get list of completed races (races that have already happened)
        
        Returns:
            list: List of completed race round numbers
        """
        schedule = fastf1.get_event_schedule(self.year)
        self.schedule = schedule
        
        current_time = datetime.now(timezone.utc)
        completed_races = []
        
        for idx, event in schedule.iterrows():
            # Check if the race has happened (EventDate is in the past)
            if pd.notna(event['EventDate']):
                event_date = event['EventDate']
                
                # Make event_date timezone-aware if it isn't
                if event_date.tzinfo is None:
                    event_date = event_date.replace(tzinfo=timezone.utc)
                
                if event_date < current_time:
                    # Check if it's actually a race (not testing)
                    if event['EventFormat'] != 'testing':
                        completed_races.append(event['RoundNumber'])
        
        return completed_races
    
    def load_race_data(self, num_races=None, load_all=False):
        """
        Load race results from completed races
        
        Args:
            num_races (int): Number of recent races to load (None = all completed)
            load_all (bool): If True, loads all completed races regardless of num_races
            
        Returns:
            pd.DataFrame: DataFrame with race results
        """
        all_data = []
        
        print(f"Loading F1 data for {self.year}...")
        
        # Get completed races
        completed_races = self.get_completed_races()
        
        if not completed_races:
            print("⚠️  No completed races found for this season yet")
            return pd.DataFrame()
        
        print(f"✓ Found {len(completed_races)} completed race(s)")
        
        # Determine which races to load
        if load_all or num_races is None:
            races_to_load = completed_races
            print(f"📊 Loading ALL {len(races_to_load)} completed races...")
        else:
            # Load the most recent N races
            races_to_load = completed_races[-num_races:] if len(completed_races) > num_races else completed_races
            print(f"📊 Loading last {len(races_to_load)} race(s)...")
        
        # Load data from races
        for round_num in races_to_load:
            try:
                session = fastf1.get_session(self.year, round_num, 'R')
                session.load()
                
                results = session.results
                event_name = session.event['EventName']
                
                for idx, row in results.iterrows():
                    all_data.append({
                        'Race': event_name,
                        'RoundNumber': round_num,
                        'Driver': row['Abbreviation'],
                        'Team': row['TeamName'],
                        'GridPosition': row['GridPosition'],
                        'Position': row['Position'],
                        'Points': row['Points']
                    })
                print(f"✓ Loaded Round {round_num}: {event_name}")
            except Exception as e:
                print(f"✗ Error loading round {round_num}: {e}")
                continue
        
        df = pd.DataFrame(all_data)
        print(f"\n✓ Successfully loaded {len(df)} race results from {len(races_to_load)} race(s)")
        
        return df
    
    def get_schedule(self):
        """Get the race schedule for the year"""
        if self.schedule is None:
            self.schedule = fastf1.get_event_schedule(self.year)
        return self.schedule
    
    def get_next_race(self):
        """
        Get information about the next upcoming race
        
        Returns:
            dict: Information about next race
        """
        schedule = self.get_schedule()
        current_time = datetime.now(timezone.utc)
        
        for idx, event in schedule.iterrows():
            if pd.notna(event['EventDate']):
                event_date = event['EventDate']
                
                # Make event_date timezone-aware if it isn't
                if event_date.tzinfo is None:
                    event_date = event_date.replace(tzinfo=timezone.utc)
                
                if event_date > current_time and event['EventFormat'] != 'testing':
                    return {
                        'name': event['EventName'],
                        'date': event['EventDate'],
                        'location': event['Location'],
                        'round': event['RoundNumber']
                    }
        
        return None
