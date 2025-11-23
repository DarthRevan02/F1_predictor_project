import numpy as np
import pandas as pd

class WDCSimulator:
    """Runs Monte Carlo simulations for World Drivers Championship"""
    
    def __init__(self, drivers, current_standings=None):
        self.drivers = drivers
        self.points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        self.current_standings = current_standings or {}
        
    def set_current_standings(self, standings):
        """
        Set the current championship standings
        
        Args:
            standings (dict): Dictionary of driver: points
        """
        self.current_standings = standings
        
    def simulate_season(self, target_driver, remaining_races, num_simulations=1000000):
        """
        Run Monte Carlo simulation for championship probability based on remaining races
        
        Args:
            target_driver (str): Driver to calculate probability for
            remaining_races (list): List of remaining race names
            num_simulations (int): Number of simulations to run
            
        Returns:
            tuple: (probability, number of wins, detailed stats)
        """
        num_remaining = len(remaining_races)
        
        print(f"🏁 Championship Simulation")
        print(f"   Target Driver: {target_driver}")
        print(f"   Remaining Races: {num_remaining}")
        print(f"   Races: {', '.join(remaining_races)}")
        print(f"   Simulations: {num_simulations:,}")
        print(f"\n⏳ Running simulations...")
        
        wins = 0
        batch_size = 10000
        
        # Driver performance tiers (based on 2024 season)
        driver_tiers = self._get_driver_performance_tiers()
        
        # Track detailed statistics
        position_counts = {driver: [0] * 4 for driver in self.drivers}  # Wins, Podiums, Top5, Top10
        final_points = {driver: [] for driver in self.drivers}
        
        for batch_start in range(0, num_simulations, batch_size):
            current_batch_size = min(batch_size, num_simulations - batch_start)
            
            # Run simulations in this batch
            for sim_num in range(current_batch_size):
                # Start with current standings
                total_points = dict(self.current_standings)
                
                # Ensure all drivers have an entry
                for driver in self.drivers:
                    if driver not in total_points:
                        total_points[driver] = 0
                
                # Simulate remaining races
                for race_idx, race_name in enumerate(remaining_races):
                    # Generate race results based on driver performance tiers
                    race_results = self._simulate_single_race(driver_tiers)
                    
                    # Award points
                    for pos, driver in enumerate(race_results):
                        if pos < len(self.points_system):
                            total_points[driver] += self.points_system[pos]
                
                # Determine championship winner
                sorted_drivers = sorted(total_points.items(), key=lambda x: x[1], reverse=True)
                championship_winner = sorted_drivers[0][0]
                
                if championship_winner == target_driver:
                    wins += 1
                
                # Track statistics for target driver
                target_final_points = total_points[target_driver]
                final_points[target_driver].append(target_final_points)
                
                # Count positions
                target_position = next(i for i, (d, p) in enumerate(sorted_drivers) if d == target_driver)
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
                print(f"   Progress: {batch_start:,}/{num_simulations:,} simulations ({current_prob:.2f}% win rate)")
        
        probability = (wins / num_simulations) * 100
        
        # Calculate detailed statistics
        stats = {
            'wins': wins,
            'probability': probability,
            'podium_rate': (position_counts[target_driver][1] / num_simulations) * 100,
            'top5_rate': (position_counts[target_driver][2] / num_simulations) * 100,
            'avg_points': np.mean(final_points[target_driver]) if final_points[target_driver] else 0,
            'median_points': np.median(final_points[target_driver]) if final_points[target_driver] else 0,
            'current_points': self.current_standings.get(target_driver, 0)
        }
        
        print(f"\n✓ Simulation complete!")
        
        return probability, wins, stats
    
    def _get_driver_performance_tiers(self):
        """
        Define driver performance tiers based on 2025 season expectations
        Higher weight = better performance
        """
        performance = {
            # 2025 Season Performance Ratings
            'NOR': 0.95, 'PIA': 0.93, 'LEC': 0.92, 'HAM': 0.91,
            'VER': 0.90, 'RUS': 0.88, 'SAI': 0.85, 'ALO': 0.84,
            'GAS': 0.82, 'STR': 0.81, 'LAW': 0.79, 'OCO': 0.78,
            'TSU': 0.76, 'ALB': 0.75, 'HUL': 0.73, 'ANT': 0.72,
            'BEA': 0.70, 'DOR': 0.69, 'HAD': 0.68, 'BOR': 0.65
        }
        
        # Return performance for all drivers, defaulting to 0.50 for unknown
        return {driver: performance.get(driver, 0.50) for driver in self.drivers}
    
    def _simulate_single_race(self, driver_tiers):
        """
        Simulate a single race with weighted random results
        
        Args:
            driver_tiers (dict): Performance weights for each driver
            
        Returns:
            list: Ordered list of drivers (winner first)
        """
        # Get performance weights
        drivers = list(driver_tiers.keys())
        weights = [driver_tiers[d] for d in drivers]
        
        # Add randomness to weights (race incidents, luck, etc.)
        race_weights = []
        for weight in weights:
            # Add random variance (±20%)
            variance = np.random.uniform(0.8, 1.2)
            race_weights.append(weight * variance)
        
        # Sort drivers by their race performance
        driver_performance = list(zip(drivers, race_weights))
        driver_performance.sort(key=lambda x: x[1], reverse=True)
        
        return [driver for driver, _ in driver_performance]
    
    def get_remaining_races_info(self, year=2024):
        """
        Get information about remaining races in the season
        
        Returns:
            list: List of remaining race names
        """
        try:
            import fastf1
            from datetime import datetime, timezone
            
            schedule = fastf1.get_event_schedule(year)
            current_time = datetime.now(timezone.utc)
            
            remaining = []
            for idx, event in schedule.iterrows():
                if pd.notna(event['EventDate']):
                    event_date = event['EventDate']
                    
                    if event_date.tzinfo is None:
                        event_date = event_date.replace(tzinfo=timezone.utc)
                    
                    if event_date > current_time and event['EventFormat'] != 'testing':
                        remaining.append(event['EventName'])
            
            return remaining
        except:
            # Fallback for 2024 end of season
            return ['Las Vegas Grand Prix', 'Qatar Grand Prix', 'Abu Dhabi Grand Prix']