import numpy as np
import pandas as pd
from datetime import datetime, timezone

class EnhancedWDCSimulator:
    """
    Advanced WDC simulator with:
    - Dynamic driver ratings
    - Safety car modeling
    - Weather effects
    - Reliability simulation
    - Track-specific performance
    """
    
    def __init__(self, drivers, current_standings, historical_data=None):
        self.drivers = drivers
        self.points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        self.current_standings = current_standings or {}
        self.historical_data = historical_data
        
        # Calculate dynamic performance ratings
        self.driver_ratings = self._calculate_dynamic_ratings()
        self.team_reliability = self._calculate_team_reliability()
        
    def _calculate_dynamic_ratings(self, last_n_races=5):
        """
        Calculate driver performance based on recent results
        """
        if self.historical_data is None:
            # Fallback to static ratings
            return {
                'NOR': 0.95, 'PIA': 0.93, 'LEC': 0.92, 'HAM': 0.91,
                'VER': 0.90, 'RUS': 0.88, 'SAI': 0.85, 'ALO': 0.84,
                'GAS': 0.82, 'STR': 0.81, 'LAW': 0.79, 'OCO': 0.78,
                'TSU': 0.76, 'ALB': 0.75, 'HUL': 0.73, 'ANT': 0.72,
                'BEA': 0.70, 'DOR': 0.69, 'HAD': 0.68, 'BOR': 0.65
            }
        
        ratings = {}
        
        # Get recent races
        recent_data = self.historical_data.sort_values(['Year', 'RoundNumber']).tail(
            last_n_races * len(self.drivers)
        )
        
        for driver in self.drivers:
            driver_data = recent_data[recent_data['Driver'] == driver]
            
            if len(driver_data) > 0:
                # Calculate performance score
                avg_position = driver_data['Position'].mean()
                points_scored = driver_data['Points'].sum()
                
                # Convert to rating (1.0 = best, 0.0 = worst)
                position_score = 1.0 - (avg_position / 20.0)
                points_score = points_scored / (last_n_races * 25)  # Normalize by max possible
                
                # Weighted average
                rating = position_score * 0.6 + points_score * 0.4
                
                # Apply recent form weight (more recent = more weight)
                if len(driver_data) >= 3:
                    recent_3 = driver_data.tail(3)['Position'].mean()
                    momentum = 1.0 - (recent_3 / 20.0)
                    rating = rating * 0.7 + momentum * 0.3
                
                ratings[driver] = np.clip(rating, 0.5, 0.98)
            else:
                ratings[driver] = 0.70  # Default for new drivers
        
        return ratings
    
    def _calculate_team_reliability(self):
        """
        Calculate team reliability from historical DNF rates
        """
        if self.historical_data is None:
            # Fallback reliability scores
            return {
                'McLaren': 0.96, 'Ferrari': 0.94, 'Red Bull Racing': 0.95,
                'Mercedes': 0.94, 'Aston Martin': 0.90, 'Alpine': 0.88,
                'Haas F1 Team': 0.87, 'RB': 0.89, 'Williams': 0.86,
                'Kick Sauber': 0.84
            }
        
        reliability = {}
        
        for team in self.historical_data['Team'].unique():
            team_data = self.historical_data[self.historical_data['Team'] == team]
            finish_rate = 1.0 - (team_data['DidNotFinish'].mean())
            reliability[team] = np.clip(finish_rate, 0.80, 0.98)
        
        return reliability
    
    def simulate_season_advanced(self, target_driver, remaining_races, num_simulations=1000000):
        """
        Advanced Monte Carlo simulation with all enhancements
        """
        print(f"\n{'='*60}")
        print(f"🏁 ADVANCED CHAMPIONSHIP SIMULATION")
        print(f"{'='*60}")
        print(f"   Target Driver: {target_driver}")
        print(f"   Remaining Races: {len(remaining_races)}")
        print(f"   Simulations: {num_simulations:,}")
        print(f"\n   Dynamic Driver Ratings (Recent Form):")
        
        # Print top 10 driver ratings
        sorted_ratings = sorted(
            self.driver_ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (driver, rating) in enumerate(sorted_ratings, 1):
            bar = "█" * int(rating * 30)
            print(f"   {i:2d}. {driver}: {bar} {rating:.3f}")
        
        print(f"\n⏳ Running simulations...\n")
        
        # Simulation tracking
        wins = 0
        batch_size = 10000
        
        # Detailed statistics
        position_counts = {driver: [0] * 4 for driver in self.drivers}
        final_points_distribution = {driver: [] for driver in self.drivers}
        race_wins_distribution = {driver: 0 for driver in self.drivers}
        
        for batch_start in range(0, num_simulations, batch_size):
            current_batch_size = min(batch_size, num_simulations - batch_start)
            
            for sim_num in range(current_batch_size):
                # Start with current standings
                total_points = dict(self.current_standings)
                
                # Ensure all drivers have entry
                for driver in self.drivers:
                    if driver not in total_points:
                        total_points[driver] = 0
                
                # Simulate each remaining race
                for race_name in remaining_races:
                    race_result = self._simulate_single_race_advanced(race_name)
                    
                    # Award points
                    for pos, driver in enumerate(race_result):
                        if pos < len(self.points_system):
                            total_points[driver] += self.points_system[pos]
                    
                    # Track race wins
                    if race_result[0] in self.drivers:
                        race_wins_distribution[race_result[0]] += 1
                
                # Determine championship winner
                sorted_drivers = sorted(
                    total_points.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                championship_winner = sorted_drivers[0][0]
                
                if championship_winner == target_driver:
                    wins += 1
                
                # Track statistics
                target_final_points = total_points[target_driver]
                final_points_distribution[target_driver].append(target_final_points)
                
                # Position tracking
                target_position = next(
                    i for i, (d, p) in enumerate(sorted_drivers) if d == target_driver
                )
                
                if target_position == 0:
                    position_counts[target_driver][0] += 1  # Win
                if target_position < 3:
                    position_counts[target_driver][1] += 1  # Podium
                if target_position < 5:
                    position_counts[target_driver][2] += 1  # Top 5
                if target_position < 10:
                    position_counts[target_driver][3] += 1  # Top 10
            
            # Progress update
            if batch_start % 100000 == 0 and batch_start > 0:
                current_prob = (wins / batch_start) * 100
                print(f"   Progress: {batch_start:,}/{num_simulations:,} ({current_prob:.2f}% win rate)")
        
        # Calculate final statistics
        probability = (wins / num_simulations) * 100
        
        stats = {
            'wins': wins,
            'probability': probability,
            'podium_rate': (position_counts[target_driver][1] / num_simulations) * 100,
            'top5_rate': (position_counts[target_driver][2] / num_simulations) * 100,
            'top10_rate': (position_counts[target_driver][3] / num_simulations) * 100,
            'avg_points': np.mean(final_points_distribution[target_driver]),
            'median_points': np.median(final_points_distribution[target_driver]),
            'min_points': np.min(final_points_distribution[target_driver]),
            'max_points': np.max(final_points_distribution[target_driver]),
            'current_points': self.current_standings.get(target_driver, 0),
            'expected_race_wins': race_wins_distribution.get(target_driver, 0) / num_simulations * len(remaining_races)
        }
        
        print(f"\n{'='*60}")
        print(f"✅ SIMULATION COMPLETE")
        print(f"{'='*60}\n")
        
        return probability, wins, stats
    
    def _simulate_single_race_advanced(self, race_name):
        """
        Simulate a single race with:
        - Dynamic performance
        - Weather effects
        - Safety cars
        - DNFs
        - Track-specific performance
        """
        # Base performance from recent form
        race_performance = {}
        
        for driver in self.drivers:
            base_rating = self.driver_ratings.get(driver, 0.70)
            
            # Add race-day variance (form, luck, mistakes)
            variance = np.random.normal(1.0, 0.15)  # 15% standard deviation
            
            # Weather impact (random weather for simulation)
            weather_impact = self._simulate_weather_impact()
            
            # Track-specific performance variation
            track_variation = np.random.uniform(0.85, 1.15)
            
            # Calculate final performance
            performance = base_rating * variance * weather_impact * track_variation
            
            race_performance[driver] = performance
        
        # Simulate DNFs based on reliability
        finishing_drivers = self._simulate_dnfs(list(race_performance.keys()))
        
        # Sort finishing drivers by performance
        final_result = sorted(
            [(d, race_performance[d]) for d in finishing_drivers],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Simulate safety car (30% chance)
        if np.random.random() < 0.30:
            final_result = self._apply_safety_car_effect(final_result)
        
        return [driver for driver, _ in final_result]
    
    def _simulate_weather_impact(self):
        """
        Simulate weather effects on performance
        """
        # 15% chance of rain
        if np.random.random() < 0.15:
            # Rain adds more randomness
            return np.random.uniform(0.7, 1.3)
        else:
            # Dry race - less variance
            return np.random.uniform(0.95, 1.05)
    
    def _simulate_dnfs(self, drivers):
        """
        Simulate DNFs based on team reliability
        """
        driver_team_map = {
            'HAM': 'Ferrari', 'LEC': 'Ferrari',
            'VER': 'Red Bull Racing', 'LAW': 'Red Bull Racing',
            'NOR': 'McLaren', 'PIA': 'McLaren',
            'RUS': 'Mercedes', 'ANT': 'Mercedes',
            'ALO': 'Aston Martin', 'STR': 'Aston Martin',
            'GAS': 'Alpine', 'DOR': 'Alpine',
            'OCO': 'Haas F1 Team', 'BEA': 'Haas F1 Team',
            'TSU': 'RB', 'HAD': 'RB',
            'SAI': 'Williams', 'ALB': 'Williams',
            'HUL': 'Kick Sauber', 'BOR': 'Kick Sauber'
        }
        
        finishing_drivers = []
        
        for driver in drivers:
            team = driver_team_map.get(driver, 'Unknown')
            reliability = self.team_reliability.get(team, 0.90)
            
            # Random DNF check
            if np.random.random() < reliability:
                finishing_drivers.append(driver)
        
        return finishing_drivers
    
    def _apply_safety_car_effect(self, race_result):
        """
        Simulate safety car impact (shuffles order slightly)
        """
        # Safety car can promote 1-3 drivers and demote 1-3 drivers
        num_to_shuffle = np.random.randint(2, 6)
        
        if len(race_result) <= num_to_shuffle:
            return race_result
        
        # Shuffle middle pack
        shuffled = race_result.copy()
        middle_start = np.random.randint(3, min(10, len(race_result) - 3))
        middle_end = min(middle_start + num_to_shuffle, len(race_result))
        
        middle_section = shuffled[middle_start:middle_end]
        np.random.shuffle(middle_section)
        shuffled[middle_start:middle_end] = middle_section
        
        return shuffled
    
    def compare_drivers(self, driver1, driver2, remaining_races, num_simulations=100000):
        """
        Head-to-head comparison of two drivers
        """
        print(f"\n{'='*60}")
        print(f"⚔️  HEAD-TO-HEAD COMPARISON")
        print(f"{'='*60}")
        print(f"   {driver1} vs {driver2}")
        print(f"   Simulations: {num_simulations:,}\n")
        
        driver1_wins = 0
        driver2_wins = 0
        
        for _ in range(num_simulations):
            total_points = dict(self.current_standings)
            
            for driver in self.drivers:
                if driver not in total_points:
                    total_points[driver] = 0
            
            # Simulate remaining races
            for race_name in remaining_races:
                race_result = self._simulate_single_race_advanced(race_name)
                
                for pos, driver in enumerate(race_result):
                    if pos < len(self.points_system):
                        total_points[driver] += self.points_system[pos]
            
            # Compare final points
            if total_points[driver1] > total_points[driver2]:
                driver1_wins += 1
            elif total_points[driver2] > total_points[driver1]:
                driver2_wins += 1
        
        driver1_prob = (driver1_wins / num_simulations) * 100
        driver2_prob = (driver2_wins / num_simulations) * 100
        
        print(f"{'='*60}")
        print(f"📊 RESULTS")
        print(f"{'='*60}")
        print(f"   {driver1}: {driver1_prob:.2f}% to finish ahead")
        print(f"   {driver2}: {driver2_prob:.2f}% to finish ahead")
        print(f"{'='*60}\n")
        
        return {
            'driver1': driver1,
            'driver2': driver2,
            'driver1_probability': driver1_prob,
            'driver2_probability': driver2_prob
        }