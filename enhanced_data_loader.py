import fastf1
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timezone
warnings.filterwarnings('ignore')

# Enable FastF1 cache
fastf1.Cache.enable_cache('f1_predictor_project\\f1_cache')

class EnhancedF1DataLoader:
    """
    Advanced F1 data loader with weather, tire, and qualifying data
    """
    
    def __init__(self, years=None):
        self.years = years or [2022, 2023, 2024, 2025]
        self.circuit_characteristics = self._initialize_circuit_data()
        
    def _initialize_circuit_data(self):
        """Circuit characteristics database"""
        return {
            'Monaco Grand Prix': {
                'type': 'street',
                'length': 3.337,
                'corners': 19,
                'avg_speed': 160,
                'overtaking_difficulty': 10,
                'tire_wear': 'medium'
            },
            'Italian Grand Prix': {
                'type': 'permanent',
                'length': 5.793,
                'corners': 11,
                'avg_speed': 264,
                'overtaking_difficulty': 3,
                'tire_wear': 'low'
            },
            'Singapore Grand Prix': {
                'type': 'street',
                'length': 5.063,
                'corners': 23,
                'avg_speed': 172,
                'overtaking_difficulty': 8,
                'tire_wear': 'high'
            },
            'Belgian Grand Prix': {
                'type': 'permanent',
                'length': 7.004,
                'corners': 19,
                'avg_speed': 235,
                'overtaking_difficulty': 4,
                'tire_wear': 'high'
            },
            'Las Vegas Grand Prix': {
                'type': 'street',
                'length': 6.201,
                'corners': 17,
                'avg_speed': 240,
                'overtaking_difficulty': 5,
                'tire_wear': 'medium'
            },
            'Qatar Grand Prix': {
                'type': 'permanent',
                'length': 5.380,
                'corners': 16,
                'avg_speed': 230,
                'overtaking_difficulty': 6,
                'tire_wear': 'high'
            },
            'Abu Dhabi Grand Prix': {
                'type': 'permanent',
                'length': 5.281,
                'corners': 16,
                'avg_speed': 195,
                'overtaking_difficulty': 5,
                'tire_wear': 'medium'
            }
        }
    
    def load_comprehensive_race_data(self):
        """
        Load complete dataset with all features
        """
        all_data = []
        
        print("=" * 60)
        print("🏎️  ENHANCED F1 DATA LOADER")
        print("=" * 60)
        print(f"\n📊 Loading data from years: {self.years}")
        print(f"⏳ This will take 5-10 minutes on first run...\n")
        
        for year in self.years:
            print(f"\n{'='*60}")
            print(f"📅 LOADING {year} SEASON")
            print(f"{'='*60}")
            
            try:
                schedule = fastf1.get_event_schedule(year)
                completed_races = self._get_completed_races(schedule)
                
                if not completed_races:
                    print(f"⚠️  No completed races found for {year}")
                    continue
                
                print(f"✓ Found {len(completed_races)} completed race(s)")
                
                for round_num in completed_races:
                    try:
                        race_data = self._load_race_with_all_features(year, round_num)
                        if race_data:
                            all_data.extend(race_data)
                            print(f"  ✓ Round {round_num}: {len(race_data)} results loaded")
                    except Exception as e:
                        print(f"  ✗ Round {round_num} failed: {str(e)[:50]}")
                        continue
                        
            except Exception as e:
                print(f"✗ Error loading {year} season: {e}")
                continue
        
        if not all_data:
            print("\n❌ No data loaded!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        
        print("\n" + "="*60)
        print("✅ DATA LOADING COMPLETE")
        print("="*60)
        print(f"📊 Total records: {len(df)}")
        print(f"📅 Years covered: {df['Year'].unique()}")
        print(f"🏁 Unique races: {df['Race'].nunique()}")
        print(f"👨‍✈️ Unique drivers: {df['Driver'].nunique()}")
        print(f"🏢 Unique teams: {df['Team'].nunique()}")
        print("="*60 + "\n")
        
        return df
    
    def _get_completed_races(self, schedule):
        """Get completed races from schedule"""
        current_time = datetime.now(timezone.utc)
        completed = []
        
        for idx, event in schedule.iterrows():
            if pd.notna(event['EventDate']):
                event_date = event['EventDate']
                if event_date.tzinfo is None:
                    event_date = event_date.replace(tzinfo=timezone.utc)
                
                if event_date < current_time and event['EventFormat'] != 'testing':
                    completed.append(event['RoundNumber'])
        
        return completed
    
    def _load_race_with_all_features(self, year, round_num):
        """
        Load race with ALL features:
        - Weather data
        - Tire data
        - Qualifying data
        - Circuit characteristics
        - Historical performance
        """
        race_session = fastf1.get_session(year, round_num, 'R')
        race_session.load()
        
        # Try to load qualifying
        try:
            quali_session = fastf1.get_session(year, round_num, 'Q')
            quali_session.load()
            has_quali = True
        except:
            has_quali = False
        
        results = race_session.results
        event_name = race_session.event['EventName']
        
        # Get weather data
        weather = self._extract_weather_data(race_session)
        
        # Get circuit characteristics
        circuit_data = self.circuit_characteristics.get(
            event_name, 
            self._get_default_circuit_data()
        )
        
        race_data = []
        
        for idx, row in results.iterrows():
            driver_code = row['Abbreviation']
            
            # Base race data
            record = {
                'Year': year,
                'Race': event_name,
                'RoundNumber': round_num,
                'Driver': driver_code,
                'Team': row['TeamName'],
                'GridPosition': row['GridPosition'],
                'Position': row['Position'],
                'Points': row['Points'],
                'Status': row['Status'],
                
                # Weather features
                'AirTemp': weather['air_temp'],
                'TrackTemp': weather['track_temp'],
                'Humidity': weather['humidity'],
                'Rainfall': weather['rainfall'],
                
                # Circuit features
                'CircuitType': circuit_data['type'],
                'CircuitLength': circuit_data['length'],
                'NumCorners': circuit_data['corners'],
                'AvgSpeed': circuit_data['avg_speed'],
                'OvertakingDifficulty': circuit_data['overtaking_difficulty'],
                'TireWear': circuit_data['tire_wear'],
                
                # Tire data (if available)
                'TireCompound': self._get_tire_compound(row),
                'NumPitStops': self._count_pit_stops(race_session, driver_code),
                
                # Qualifying data (if available)
                'QualifyingTime': self._get_quali_time(quali_session, driver_code) if has_quali else None,
                'GapToPole': self._get_gap_to_pole(quali_session, driver_code) if has_quali else None,
                
                # Race metrics
                'FastestLap': row.get('FastestLap', None),
                'AvgLapTime': self._calculate_avg_lap_time(race_session, driver_code),
                'Consistency': self._calculate_consistency(race_session, driver_code),
                
                # DNF indicator
                'DidNotFinish': 0 if row['Status'] == 'Finished' else 1
            }
            
            race_data.append(record)
        
        return race_data
    
    def _extract_weather_data(self, session):
        """Extract weather data from session"""
        try:
            weather = session.weather_data
            return {
                'air_temp': weather['AirTemp'].mean(),
                'track_temp': weather['TrackTemp'].mean(),
                'humidity': weather['Humidity'].mean() if 'Humidity' in weather else 50,
                'rainfall': int(weather['Rainfall'].any()) if 'Rainfall' in weather else 0
            }
        except:
            # Default weather if not available
            return {
                'air_temp': 25.0,
                'track_temp': 35.0,
                'humidity': 50.0,
                'rainfall': 0
            }
    
    def _get_tire_compound(self, row):
        """Get tire compound used (simplified)"""
        try:
            compound = row.get('Compound', 'MEDIUM')
            return compound if compound else 'MEDIUM'
        except:
            return 'MEDIUM'
    
    def _count_pit_stops(self, session, driver):
        """Count pit stops for driver"""
        try:
            laps = session.laps.pick_driver(driver)
            pit_stops = laps[laps['PitInTime'].notna()]
            return len(pit_stops)
        except:
            return 1  # Default assumption
    
    def _get_quali_time(self, quali_session, driver):
        """Get qualifying time"""
        try:
            driver_quali = quali_session.results[
                quali_session.results['Abbreviation'] == driver
            ]
            if not driver_quali.empty:
                q3_time = driver_quali.iloc[0].get('Q3')
                if pd.notna(q3_time):
                    return q3_time.total_seconds()
            return None
        except:
            return None
    
    def _get_gap_to_pole(self, quali_session, driver):
        """Get gap to pole position in seconds"""
        try:
            results = quali_session.results
            pole_time = results.iloc[0].get('Q3')
            driver_time = results[results['Abbreviation'] == driver].iloc[0].get('Q3')
            
            if pd.notna(pole_time) and pd.notna(driver_time):
                gap = (driver_time - pole_time).total_seconds()
                return gap
            return None
        except:
            return None
    
    def _calculate_avg_lap_time(self, session, driver):
        """Calculate average lap time"""
        try:
            laps = session.laps.pick_driver(driver)
            valid_laps = laps[laps['LapTime'].notna()]
            if not valid_laps.empty:
                avg_time = valid_laps['LapTime'].mean().total_seconds()
                return avg_time
            return None
        except:
            return None
    
    def _calculate_consistency(self, session, driver):
        """Calculate lap time consistency (lower = more consistent)"""
        try:
            laps = session.laps.pick_driver(driver)
            valid_laps = laps[laps['LapTime'].notna()]
            if len(valid_laps) > 3:
                lap_times = [lap.total_seconds() for lap in valid_laps['LapTime']]
                consistency = np.std(lap_times)
                return consistency
            return None
        except:
            return None
    
    def _get_default_circuit_data(self):
        """Default circuit characteristics"""
        return {
            'type': 'permanent',
            'length': 5.0,
            'corners': 16,
            'avg_speed': 200,
            'overtaking_difficulty': 5,
            'tire_wear': 'medium'
        }