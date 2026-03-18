import numpy as np
import pandas as pd
from driver_config import DRIVER_PERFORMANCE


class WDCSimulator:
    """Monte Carlo World Drivers Championship simulator."""

    def __init__(self, drivers, current_standings=None):
        self.drivers = drivers
        self.points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        self.current_standings = current_standings or {}
        self.qualifying_predictor = None

    def set_qualifying_predictor(self, predictor):
        self.qualifying_predictor = predictor

    def set_current_standings(self, standings):
        self.current_standings = standings

    def simulate_season(self, target_driver, remaining_races, num_simulations=1_000_000,
                        seed=None):
        """
        Run Monte Carlo championship simulation.

        Args:
            seed (int | None): Optional RNG seed for reproducibility (flaw #12).
                               Pass an integer to get deterministic results.

        Returns:
            (probability, wins, stats)
        """
        rng = np.random.default_rng(seed)   # isolated RNG — never touches global state

        print(f"🏁 Championship Simulation")
        print(f"   Target: {target_driver}  |  Races: {len(remaining_races)}"
              f"  |  Sims: {num_simulations:,}  |  Seed: {seed}")

        driver_tiers = self._get_driver_performance_tiers()
        predicted_grids = self._predict_grids_for_races(remaining_races)

        wins = 0
        pos_counts = [0] * 4   # [wins, podiums, top5, top10]
        final_pts_list = []

        batch = 10_000
        for batch_start in range(0, num_simulations, batch):
            size = min(batch, num_simulations - batch_start)
            for _ in range(size):
                total_points = dict(self.current_standings)
                for d in self.drivers:
                    total_points.setdefault(d, 0)

                prev_results = None
                for race_idx, race_name in enumerate(remaining_races):
                    grid = (
                        predicted_grids.get(race_name)
                        if race_idx == 0
                        else self._grid_from_prev(prev_results, driver_tiers, rng)
                    )
                    results = self._simulate_race(driver_tiers, grid, rng)
                    prev_results = results
                    for pos, driver in enumerate(results):
                        if pos < len(self.points_system):
                            total_points[driver] += self.points_system[pos]

                sorted_drivers = sorted(total_points.items(), key=lambda x: x[1], reverse=True)
                if sorted_drivers[0][0] == target_driver:
                    wins += 1

                target_pos = next(
                    i for i, (d, _) in enumerate(sorted_drivers) if d == target_driver
                )
                pos_counts[0] += target_pos == 0
                pos_counts[1] += target_pos < 3
                pos_counts[2] += target_pos < 5
                pos_counts[3] += target_pos < 10
                final_pts_list.append(total_points[target_driver])

            if batch_start and batch_start % 100_000 == 0:
                print(f"   {batch_start:,}/{num_simulations:,} "
                      f"({wins / batch_start * 100:.2f}% win rate)")

        prob = wins / num_simulations * 100
        arr = np.array(final_pts_list)
        stats = {
            'wins': wins,
            'probability': prob,
            'podium_rate': pos_counts[1] / num_simulations * 100,
            'top5_rate': pos_counts[2] / num_simulations * 100,
            'avg_points': float(arr.mean()),
            'median_points': float(np.median(arr)),
            'current_points': self.current_standings.get(target_driver, 0),
        }
        print(f"\n✓ Simulation complete! Win probability: {prob:.2f}%")
        return prob, wins, stats

    # ------------------------------------------------------------------
    def _get_driver_performance_tiers(self):
        return {d: DRIVER_PERFORMANCE.get(d, 0.50) for d in self.drivers}

    def _predict_grids_for_races(self, remaining_races):
        grids = {}
        for name in remaining_races:
            actual = self._actual_qualifying(name)
            if actual:
                grids[name] = actual
                print(f"   ✓ Actual qualifying grid loaded for {name}")
            elif self.qualifying_predictor:
                try:
                    q = self.qualifying_predictor.predict_qualifying(name)
                    grids[name] = {p['driver']: p['predicted_grid'] for p in q['predicted_grid']}
                    print(f"   ✓ Predicted grid for {name}")
                except Exception as e:
                    print(f"   ⚠️  Grid prediction failed for {name}: {e}")
                    grids[name] = None
            else:
                grids[name] = None
        return grids

    def _actual_qualifying(self, race_name, year=2025):
        try:
            import fastf1
            from datetime import datetime, timezone

            schedule = fastf1.get_event_schedule(year)
            current_time = datetime.now(timezone.utc)

            for _, event in schedule.iterrows():
                if event['EventName'] != race_name:
                    continue
                event_date = event['EventDate']
                if event_date.tzinfo is None:
                    event_date = event_date.replace(tzinfo=timezone.utc)
                if (event_date - current_time).days > 1:
                    return None

                qualifying = fastf1.get_session(year, int(event['RoundNumber']), 'Q')
                qualifying.load()
                results = qualifying.results
                if results.empty:
                    return None
                grid = {}
                for _, row in results.iterrows():
                    if pd.notna(row['Position']):
                        grid[row['Abbreviation']] = int(row['Position'])
                return grid or None
        except Exception:
            return None

    def _grid_from_prev(self, prev_results, driver_tiers, rng):
        if not prev_results:
            return None
        grid = {}
        for driver, skill in driver_tiers.items():
            if driver in prev_results:
                fp = prev_results.index(driver) + 1
                var = int(rng.integers(-3, 4))
                if skill > 0.85:
                    var //= 2
                grid[driver] = int(max(1, min(20, fp + var)))
            else:
                if skill > 0.85:
                    grid[driver] = int(rng.integers(1, 8))
                elif skill > 0.75:
                    grid[driver] = int(rng.integers(6, 14))
                else:
                    grid[driver] = int(rng.integers(12, 21))
        return grid

    def _simulate_race(self, driver_tiers, starting_grid, rng):
        drivers = list(driver_tiers.keys())
        weights = []
        for driver in drivers:
            base = driver_tiers[driver]
            if starting_grid and driver in starting_grid:
                grid_bonus = (21 - starting_grid[driver]) * 0.01
            else:
                grid_bonus = 0.15 if base > 0.85 else (0.05 if base > 0.75 else -0.05)

            variance = float(rng.uniform(0.85, 1.15))
            w = (base + grid_bonus) * variance
            if rng.random() < 0.05:   # 5% DNF
                w = 0.0
            weights.append(w)

        perf = sorted(zip(drivers, weights), key=lambda x: x[1], reverse=True)
        return [d for d, _ in perf]

    def get_remaining_races_info(self, year=2025):
        try:
            import fastf1
            from datetime import datetime, timezone

            schedule = fastf1.get_event_schedule(year)
            now = datetime.now(timezone.utc)
            remaining = []
            for _, event in schedule.iterrows():
                if pd.notna(event['EventDate']) and event['EventFormat'] != 'testing':
                    ed = event['EventDate']
                    if ed.tzinfo is None:
                        ed = ed.replace(tzinfo=timezone.utc)
                    if ed > now:
                        remaining.append(event['EventName'])
            return remaining
        except Exception:
            return ['Qatar Grand Prix', 'Abu Dhabi Grand Prix']
