import numpy as np
from driver_config import DRIVER_TEAM_MAP

_TOP_TEAMS = {'McLaren', 'Ferrari', 'Red Bull Racing', 'Mercedes'}


class StrategyPredictor:
    """Forecasts pit stop strategies and tyre choices."""

    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders

    def predict_race_strategy(self, race_name="Next Race", race_distance=58):
        drivers = list(self.label_encoders['Driver'].classes_)[:10]
        predictions = []

        for driver in drivers:
            team = DRIVER_TEAM_MAP.get(driver, 'Unknown')
            strat = self._driver_strategy(team, race_distance)
            predictions.append({
                'driver': driver,
                'team': team,
                'predicted_strategy': strat['strategy_type'],
                'num_pit_stops': strat['num_stops'],
                'first_stop_lap': strat['first_stop'],
                'tire_compounds': strat['compounds'],
                'strategy_risk': strat['risk_level'],
                'undercut_probability': strat['undercut_prob'],
            })

        return {
            'race_name': race_name,
            'race_distance': race_distance,
            'strategy_predictions': predictions,
            'most_aggressive': max(predictions, key=lambda x: x['strategy_risk']),
            'most_conservative': min(predictions, key=lambda x: x['strategy_risk']),
        }

    def _driver_strategy(self, team, laps):
        if laps < 50:
            stops = int(np.random.choice([1, 2], p=[0.7, 0.3]))
        else:
            stops = int(np.random.choice([1, 2, 3], p=[0.4, 0.5, 0.1]))

        if stops == 1:
            return dict(strategy_type="One-Stop", num_stops=1,
                        compounds=["Medium", "Hard"],
                        first_stop=int(np.random.randint(laps // 3, laps // 2)),
                        risk_level=30, undercut_prob=70 if team in _TOP_TEAMS else 40)
        if stops == 2:
            return dict(strategy_type="Two-Stop", num_stops=2,
                        compounds=["Soft", "Medium", "Hard"],
                        first_stop=int(np.random.randint(laps // 4, laps // 3)),
                        risk_level=50, undercut_prob=70 if team in _TOP_TEAMS else 40)
        return dict(strategy_type="Three-Stop (Aggressive)", num_stops=3,
                    compounds=["Soft", "Soft", "Medium", "Hard"],
                    first_stop=int(np.random.randint(10, 20)),
                    risk_level=75, undercut_prob=70 if team in _TOP_TEAMS else 40)
