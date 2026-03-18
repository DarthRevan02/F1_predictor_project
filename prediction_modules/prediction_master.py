from prediction_modules.prediction_race_winner import RaceWinnerPredictor
from prediction_modules.prediction_podium import PodiumPredictor
from prediction_modules.prediction_qualifying import QualifyingPredictor
from prediction_modules.prediction_fastest_lap import FastestLapPredictor
from prediction_modules.prediction_points_finish import PointsFinishPredictor
from prediction_modules.prediction_dnf_probability import DNFPredictor
from prediction_modules.prediction_overtakes import OvertakePredictor
from prediction_modules.prediction_championship_impact import ChampionshipImpactPredictor
from prediction_modules.prediction_team_performance import TeamPerformancePredictor
from prediction_modules.prediction_strategy import StrategyPredictor


class MasterF1Predictor:
    """
    Coordinator that exposes all specialised prediction modules.

    Note: this class intentionally does NOT define a predict_race_winner()
    method — the previous version shadowed the sub-predictor's method with
    a different signature, causing confusion (flaw #15).  Callers should use
    master.race_winner_predictor.predict_race_winner(race_name) directly,
    which is what app.py already does.
    """

    def __init__(self, model, label_encoders, current_standings=None):
        self.model = model
        self.label_encoders = label_encoders
        self.current_standings = current_standings or {}

        self.race_winner_predictor = RaceWinnerPredictor(model, label_encoders)
        self.podium_predictor = PodiumPredictor(model, label_encoders)
        self.qualifying_predictor = QualifyingPredictor(model, label_encoders)
        self.fastest_lap_predictor = FastestLapPredictor(model, label_encoders)
        self.points_predictor = PointsFinishPredictor(model, label_encoders)
        self.dnf_predictor = DNFPredictor(model, label_encoders)
        self.overtake_predictor = OvertakePredictor(model, label_encoders)
        self.championship_predictor = ChampionshipImpactPredictor(
            model, label_encoders, current_standings
        )
        self.team_predictor = TeamPerformancePredictor(model, label_encoders)
        self.strategy_predictor = StrategyPredictor(model, label_encoders)

    def get_comprehensive_predictions(self, race_name="Next Race", race_distance=58):
        """Run all 10 prediction modules and return combined results."""
        print(f"\n🏁 Generating comprehensive predictions for {race_name}...")
        predictions = {'race_name': race_name, 'predictions': {}}

        steps = [
            ('race_winner',        lambda: self.race_winner_predictor.predict_race_winner(race_name)),
            ('podium',             lambda: self.podium_predictor.predict_podium(race_name)),
            ('qualifying',         lambda: self.qualifying_predictor.predict_qualifying(race_name)),
            ('fastest_lap',        lambda: self.fastest_lap_predictor.predict_fastest_lap(race_name)),
            ('points_finishers',   lambda: self.points_predictor.predict_points_finishers(race_name)),
            ('dnf_risk',           lambda: self.dnf_predictor.predict_dnf_risk(race_name)),
            ('overtakes',          lambda: self.overtake_predictor.predict_position_changes(race_name)),
            ('championship_impact',lambda: self.championship_predictor.predict_championship_impact(race_name)),
            ('team_performance',   lambda: self.team_predictor.predict_team_performance(race_name)),
            ('strategy',           lambda: self.strategy_predictor.predict_race_strategy(race_name, race_distance)),
        ]

        for i, (key, fn) in enumerate(steps, 1):
            print(f"  [{i}/{len(steps)}] {key}...")
            predictions['predictions'][key] = fn()

        print("✓ All predictions generated!\n")
        return predictions

    def get_prediction_summary(self, race_name="Next Race"):
        """Concise summary of key predictions."""
        c = self.get_comprehensive_predictions(race_name)
        p = c['predictions']
        return {
            'race_name': race_name,
            'predicted_winner':    p['race_winner']['winner'].get('driver'),
            'predicted_podium':    [x['driver'] for x in p['race_winner']['podium']],
            'pole_position':       p['qualifying']['pole_position']['driver'],
            'fastest_lap_favorite':p['fastest_lap']['most_likely']['driver'],
            'top_team':            p['team_performance']['top_team']['team'],
            'highest_dnf_risk':    p['dnf_risk']['highest_risk'][0]['driver'],
            'best_overtaker':      p['overtakes']['best_overtakers'][0]['driver'],
        }

    def compare_drivers(self, driver1, driver2, race_name="Next Race"):
        """Head-to-head driver comparison."""
        c = self.get_comprehensive_predictions(race_name)
        p = c['predictions']

        def extract(driver):
            metrics = {
                'predicted_position': None,
                'podium_probability': 0.0,
                'quali_position': None,
                'dnf_risk': 0.0,
                'overtaking_rating': 0.0,
            }
            for pred in p['race_winner']['predictions']:
                if pred['driver'] == driver:
                    metrics['predicted_position'] = pred['predicted_position']
            for pred in p['podium']['top_contenders']:
                if pred['driver'] == driver:
                    metrics['podium_probability'] = pred['podium_probability']
            for pred in p['qualifying']['predicted_grid']:
                if pred['driver'] == driver:
                    metrics['quali_position'] = pred['predicted_grid']
            for pred in p['dnf_risk']['predictions']:
                if pred['driver'] == driver:
                    metrics['dnf_risk'] = pred['dnf_probability']
            for pred in p['overtakes']['predictions']:
                if pred['driver'] == driver:
                    metrics['overtaking_rating'] = pred['overtaking_rating']
            return metrics

        m1, m2 = extract(driver1), extract(driver2)
        score1 = sum([
            (m1['predicted_position'] or 20) < (m2['predicted_position'] or 20),
            m1['podium_probability'] > m2['podium_probability'],
        ])
        score2 = 2 - score1

        return {
            'race_name': race_name,
            'driver1': driver1, 'driver2': driver2,
            'comparison': {driver1: m1, driver2: m2},
            'advantage': driver1 if score1 >= score2 else driver2,
        }
