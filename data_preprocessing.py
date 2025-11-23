import pandas as pd
from sklearn.preprocessing import LabelEncoder

class F1DataPreprocessor:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self):
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        """
        Preprocess the race data for machine learning
        
        Args:
            df (pd.DataFrame): Raw race data
            
        Returns:
            pd.DataFrame: Preprocessed data
            dict: Label encoders for categorical variables
        """
        # Remove rows with missing values
        df = df.dropna(subset=['GridPosition', 'Position'])
        
        # Encode categorical variables
        self.label_encoders['Driver'] = LabelEncoder()
        self.label_encoders['Team'] = LabelEncoder()
        
        df['Driver_Encoded'] = self.label_encoders['Driver'].fit_transform(df['Driver'])
        df['Team_Encoded'] = self.label_encoders['Team'].fit_transform(df['Team'])
        
        print(f"✓ Preprocessed {len(df)} race results")
        print(f"✓ Found {len(self.label_encoders['Driver'].classes_)} unique drivers")
        print(f"✓ Found {len(self.label_encoders['Team'].classes_)} unique teams")
        
        return df, self.label_encoders
    
    def get_drivers(self):
        """Get list of all drivers"""
        return list(self.label_encoders['Driver'].classes_)
    
    def get_teams(self):
        """Get list of all teams"""
        return list(self.label_encoders['Team'].classes_)