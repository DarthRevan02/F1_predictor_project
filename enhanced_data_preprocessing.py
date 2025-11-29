import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class EnhancedF1DataPreprocessor:
    """
    Advanced preprocessing with comprehensive feature engineering
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.driver_stats = {}
        self.team_stats = {}
        
    def preprocess_and_engineer_features(self, df):
        """
        Complete preprocessing pipeline with all feature engineering
        """
        print("\n" + "="*60)
        print("🔧 ADVANCED FEATURE ENGINEERING")
        print("="*60)
        
        # Remove invalid rows
        df = df.dropna(subset=['GridPosition', 'Position'])
        df = df[df['Position'] <= 20]  # Valid finishing positions
        
        print(f"\n1️⃣  Creating Historical Performance Features...")
        df = self._add_historical_features(df)
        
        print(f"2️⃣  Calculating Recent Form Features...")
        df = self._add_recent_form_features(df)
        
        print(f"3️⃣  Engineering Tire Strategy Features...")
        df = self._add_tire_features(df)
        
        print(f"4️⃣  Creating Circuit-Specific Features...")
        df = self._add_circuit_specific_features(df)
        
        print(f"5️⃣  Adding Weather Interaction Features...")
        df = self._add_weather_interactions(df)
        
        print(f"6️⃣  Creating Championship Pressure Features...")
        df = self._add_championship_features(df)
        
        print(f"7️⃣  Engineering Qualifying Features...")
        df = self._add_qualifying_features(df)
        
        print(f"8️⃣  Creating Team Dynamics Features...")
        df = self._add_team_dynamics(df)
        
        print(f"9️⃣  Adding Temporal Features...")
        df = self._add_temporal_features(df)
        
        print(f"🔟 Encoding Categorical Variables...")
        df = self._encode_categorical_variables(df)
        
        print(f"\n✅ Feature Engineering Complete!")
        print(f"📊 Total features created: {len(df.columns)}")
        print(f"📈 Training samples: {len(df)}")
        print("="*60 + "\n")
        
        return df, self.label_encoders
    
    def _add_historical_features(self, df):
        """Add historical performance at each circuit"""
        # Driver's average finish at this circuit (excluding current race)
        df['DriverCircuitAvg'] = df.groupby(['Driver', 'Race'])['Position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Team's average finish at this circuit
        df['TeamCircuitAvg'] = df.groupby(['Team', 'Race'])['Position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        # Driver's best finish at this circuit
        df['DriverCircuitBest'] = df.groupby(['Driver', 'Race'])['Position'].transform(
            lambda x: x.expanding().min().shift(1)
        )
        
        # Fill NaN with reasonable defaults
        df['DriverCircuitAvg'] = df['DriverCircuitAvg'].fillna(10)
        df['TeamCircuitAvg'] = df['TeamCircuitAvg'].fillna(10)
        df['DriverCircuitBest'] = df['DriverCircuitBest'].fillna(10)
        
        print(f"   ✓ Historical circuit performance features added")
        return df
    
    def _add_recent_form_features(self, df):
        """Add recent form (last 3, 5 races)"""
        # Sort by year and round to ensure chronological order
        df = df.sort_values(['Year', 'RoundNumber'])
        
        # Last 3 races average position
        df['RecentForm3'] = df.groupby('Driver')['Position'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
        
        # Last 5 races average position
        df['RecentForm5'] = df.groupby('Driver')['Position'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
        # Points in last 3 races
        df['RecentPoints3'] = df.groupby('Driver')['Points'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum().shift(1)
        )
        
        # Momentum: improving or declining
        df['FormTrend'] = df.groupby('Driver')['Position'].transform(
            lambda x: x.rolling(window=3, min_periods=2).apply(
                lambda y: y.iloc[0] - y.iloc[-1] if len(y) >= 2 else 0
            ).shift(1)
        )
        
        # Fill NaN
        df['RecentForm3'] = df['RecentForm3'].fillna(10)
        df['RecentForm5'] = df['RecentForm5'].fillna(10)
        df['RecentPoints3'] = df['RecentPoints3'].fillna(0)
        df['FormTrend'] = df['FormTrend'].fillna(0)
        
        print(f"   ✓ Recent form features (3 and 5 race averages) added")
        return df
    
    def _add_tire_features(self, df):
        """Add tire strategy features"""
        # Encode tire compounds
        tire_degradation = {'SOFT': 1.5, 'MEDIUM': 1.0, 'HARD': 0.7, 'INTERMEDIATE': 0.8, 'WET': 0.9}
        df['TireDegradation'] = df['TireCompound'].map(tire_degradation).fillna(1.0)
        
        # Pit stop impact (each stop costs ~20-25 seconds)
        df['PitStopImpact'] = df['NumPitStops'] * 22
        
        # Optimal strategy indicator (1-2 stops is usually optimal)
        df['OptimalStrategy'] = df['NumPitStops'].apply(
            lambda x: 1 if 1 <= x <= 2 else 0
        )
        
        # Tire wear interaction with circuit
        tire_wear_map = {'low': 0.7, 'medium': 1.0, 'high': 1.3}
        df['CircuitTireWear'] = df['TireWear'].map(tire_wear_map).fillna(1.0)
        
        df['TireCircuitInteraction'] = df['TireDegradation'] * df['CircuitTireWear']
        
        print(f"   ✓ Tire strategy and degradation features added")
        return df
    
    def _add_circuit_specific_features(self, df):
        """Add circuit characteristic features"""
        # Encode circuit type
        circuit_type_map = {'street': 1, 'permanent': 0}
        df['CircuitType_Encoded'] = df['CircuitType'].map(circuit_type_map).fillna(0)
        
        # Normalized circuit features
        df['NormalizedCircuitLength'] = df['CircuitLength'] / df['CircuitLength'].max()
        df['NormalizedCorners'] = df['NumCorners'] / df['NumCorners'].max()
        df['NormalizedSpeed'] = df['AvgSpeed'] / df['AvgSpeed'].max()
        
        # Overtaking factor (affects position changes)
        df['OvertakingFactor'] = 11 - df['OvertakingDifficulty']  # Invert scale
        
        # Circuit complexity score
        df['CircuitComplexity'] = (
            df['NumCorners'] * 0.4 + 
            df['OvertakingDifficulty'] * 0.3 + 
            df['CircuitType_Encoded'] * 10 * 0.3
        )
        
        print(f"   ✓ Circuit-specific features engineered")
        return df
    
    def _add_weather_interactions(self, df):
        """Add weather interaction features"""
        # Temperature effects on tire performance
        df['TempTireInteraction'] = df['TrackTemp'] * df['TireDegradation']
        
        # Rain significantly affects results
        df['RainImpact'] = df['Rainfall'] * 5  # Rain is a major factor
        
        # Hot weather and high tire wear compounds difficulty
        df['WeatherTireStress'] = (
            (df['TrackTemp'] / 50) * df['CircuitTireWear']
        )
        
        # Humidity can affect grip
        df['HumidityEffect'] = df['Humidity'] / 100
        
        print(f"   ✓ Weather interaction features created")
        return df
    
    def _add_championship_features(self, df):
        """Add championship pressure features"""
        # Championship position at time of race
        df['ChampionshipPosition'] = df.groupby(['Year', 'RoundNumber'])['Points'].rank(
            ascending=False, method='min'
        )
        
        # Points gap to leader
        df['PointsGapToLeader'] = df.groupby(['Year', 'RoundNumber'])['Points'].transform(
            lambda x: x.max() - x
        )
        
        # Championship pressure (top 3 drivers have more pressure)
        df['ChampionshipPressure'] = df['ChampionshipPosition'].apply(
            lambda x: 10 if x <= 3 else 5 if x <= 5 else 0
        )
        
        # Late season pressure multiplier
        df['LateSeasonMultiplier'] = df['RoundNumber'] / df.groupby('Year')['RoundNumber'].transform('max')
        
        df['TotalPressure'] = df['ChampionshipPressure'] * df['LateSeasonMultiplier']
        
        print(f"   ✓ Championship pressure features calculated")
        return df
    
    def _add_qualifying_features(self, df):
        """Add qualifying-related features"""
        # Gap to pole (already have, but ensure it's filled)
        if 'GapToPole' in df.columns:
            df['GapToPole'] = df['GapToPole'].fillna(
                df.groupby('Race')['GapToPole'].transform('median')
            )
        else:
            df['GapToPole'] = 0
        
        # Qualifying performance relative to teammate
        df['QualiVsTeammate'] = df.groupby(['Race', 'Team', 'Year'])['GridPosition'].transform(
            lambda x: x - x.mean()
        )
        
        # Grid position relative to team average
        df['GridVsTeamAvg'] = df.groupby(['Team', 'Year'])['GridPosition'].transform(
            lambda x: x - x.mean()
        )
        
        print(f"   ✓ Qualifying features engineered")
        return df
    
    def _add_team_dynamics(self, df):
        """Add team-level features"""
        # Team's recent form
        df['TeamRecentForm'] = df.groupby('Team')['Position'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        ).fillna(10)
        
        # Team reliability (DNF rate)
        df['TeamReliability'] = df.groupby('Team')['DidNotFinish'].transform(
            lambda x: 1 - x.rolling(window=10, min_periods=1).mean().shift(1)
        ).fillna(0.95)
        
        # Faster than teammate indicator
        df['FasterThanTeammate'] = df.groupby(['Race', 'Team', 'Year'])['GridPosition'].transform(
            lambda x: (x < x.mean()).astype(int)
        )
        
        print(f"   ✓ Team dynamics features added")
        return df
    
    def _add_temporal_features(self, df):
        """Add time-based features"""
        # Season progression (early vs late season affects reliability)
        df['SeasonProgression'] = df['RoundNumber'] / df.groupby('Year')['RoundNumber'].transform('max')
        
        # Race number in career (experience proxy)
        df['DriverExperience'] = df.groupby('Driver').cumcount() + 1
        
        print(f"   ✓ Temporal features created")
        return df
    
    def _encode_categorical_variables(self, df):
        """Encode all categorical variables"""
        # Driver encoding
        self.label_encoders['Driver'] = LabelEncoder()
        df['Driver_Encoded'] = self.label_encoders['Driver'].fit_transform(df['Driver'])
        
        # Team encoding
        self.label_encoders['Team'] = LabelEncoder()
        df['Team_Encoded'] = self.label_encoders['Team'].fit_transform(df['Team'])
        
        # Race encoding (circuit learning)
        self.label_encoders['Race'] = LabelEncoder()
        df['Race_Encoded'] = self.label_encoders['Race'].fit_transform(df['Race'])
        
        # Tire compound encoding
        self.label_encoders['TireCompound'] = LabelEncoder()
        df['TireCompound_Encoded'] = self.label_encoders['TireCompound'].fit_transform(
            df['TireCompound'].fillna('MEDIUM')
        )
        
        print(f"   ✓ Categorical variables encoded")
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns for training"""
        return [
            # Basic features
            'Driver_Encoded', 'Team_Encoded', 'Race_Encoded', 'GridPosition',
            
            # Historical performance
            'DriverCircuitAvg', 'TeamCircuitAvg', 'DriverCircuitBest',
            
            # Recent form
            'RecentForm3', 'RecentForm5', 'RecentPoints3', 'FormTrend',
            'TeamRecentForm',
            
            # Tire features
            'TireDegradation', 'PitStopImpact', 'OptimalStrategy',
            'TireCircuitInteraction', 'TireCompound_Encoded',
            
            # Circuit features
            'CircuitType_Encoded', 'NormalizedCircuitLength', 'NormalizedCorners',
            'NormalizedSpeed', 'OvertakingFactor', 'CircuitComplexity', 'CircuitTireWear',
            
            # Weather features
            'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',
            'TempTireInteraction', 'WeatherTireStress', 'HumidityEffect',
            
            # Championship features
            'ChampionshipPosition', 'PointsGapToLeader', 'TotalPressure',
            
            # Qualifying features
            'GapToPole', 'QualiVsTeammate', 'GridVsTeamAvg',
            
            # Team dynamics
            'TeamReliability', 'FasterThanTeammate',
            
            # Temporal features
            'SeasonProgression', 'DriverExperience'
        ]
    
    def get_drivers(self):
        """Get list of all drivers"""
        return list(self.label_encoders['Driver'].classes_)
    
    def get_teams(self):
        """Get list of all teams"""
        return list(self.label_encoders['Team'].classes_)