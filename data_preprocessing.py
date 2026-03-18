import pandas as pd
from sklearn.preprocessing import LabelEncoder
from driver_config import DRIVER_TEAM_MAP


class F1DataPreprocessor:
    """Handles data cleaning and preprocessing."""

    def __init__(self):
        self.label_encoders = {}

    def preprocess_data(self, df):
        """
        Preprocess race data for ML.

        The encoders are fitted on the union of actual training data labels
        AND the full 2025 grid from driver_config.  This prevents a
        ValueError when a 2025 driver who raced in only a few rounds is
        asked for in prediction — fixes flaw #8.

        Returns:
            (pd.DataFrame, dict): cleaned frame + label encoders
        """
        df = df.dropna(subset=['GridPosition', 'Position']).copy()

        # Build exhaustive label sets so unseen 2025 drivers don't crash
        all_known_drivers = sorted(
            set(df['Driver'].tolist()) | set(DRIVER_TEAM_MAP.keys())
        )
        all_known_teams = sorted(
            set(df['Team'].tolist()) | set(DRIVER_TEAM_MAP.values())
        )

        self.label_encoders['Driver'] = LabelEncoder()
        self.label_encoders['Driver'].fit(all_known_drivers)

        self.label_encoders['Team'] = LabelEncoder()
        self.label_encoders['Team'].fit(all_known_teams)

        df['Driver_Encoded'] = self.label_encoders['Driver'].transform(df['Driver'])
        df['Team_Encoded'] = self.label_encoders['Team'].transform(df['Team'])

        print(f"✓ Preprocessed {len(df)} race results")
        print(f"✓ Encoder covers {len(self.label_encoders['Driver'].classes_)} drivers "
              f"({len(df['Driver'].unique())} in training data)")
        print(f"✓ Encoder covers {len(self.label_encoders['Team'].classes_)} teams")

        return df, self.label_encoders

    def safe_encode_driver(self, driver):
        """
        Encode a driver abbreviation, returning None if still unknown.
        Callers should handle None rather than crashing (fixes flaw #8).
        """
        enc = self.label_encoders.get('Driver')
        if enc is None:
            return None
        if driver not in enc.classes_:
            return None
        return int(enc.transform([driver])[0])

    def safe_encode_team(self, team):
        """Same safe wrapper for teams."""
        enc = self.label_encoders.get('Team')
        if enc is None:
            return None
        if team not in enc.classes_:
            return None
        return int(enc.transform([team])[0])

    def get_drivers(self):
        return list(self.label_encoders['Driver'].classes_)

    def get_teams(self):
        return list(self.label_encoders['Team'].classes_)
