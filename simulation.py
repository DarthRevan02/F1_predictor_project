import numpy as np
import pandas as pd

class WDCSimulator:
    """Runs Monte Carlo simulations for World Drivers Championship"""
    
    def __init__(self, drivers, current_standings=None):
        self.drivers = drivers
        self.points_system = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
        self.current_standings = current_standings or {}
        self.qualifying_predictor = None
        
    def set_qualifying_predictor(self, predictor):
        """Set qualifying predictor for grid position predictions"""
        self.qualifying_predictor = predictor
        
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
        
        # Driver performance tiers (based on 2025 season)
        driver_tiers = self._get_driver_performance_tiers()
        
        # Track detailed statistics
        position_counts = {driver: [0] * 4 for driver in self.drivers}  # Wins, Podiums, Top5, Top10
        final_points = {driver: [] for driver in self.drivers}
        
        # Get predicted starting grids for each race
        predicted_grids = self._predict_grids_for_races(remaining_races)
        
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
                
                # Track race results for grid prediction
                previous_race_results = None
                
                # Simulate remaining races
                for race_idx, race_name in enumerate(remaining_races):
                    # Get starting grid for this race
                    if race_idx == 0:
                        # First race: use predicted qualifying positions
                        starting_grid = predicted_grids.get(race_name, None)
                    else:
                        # Subsequent races: use previous race results with variance
                        starting_grid = self._predict_grid_from_previous_race(
                            previous_race_results, driver_tiers
                        )
                    
                    # Generate race results based on starting grid and performance
                    race_results = self._simulate_single_race(
                        driver_tiers, 
                        starting_grid=starting_grid
                    )
                    
                    # Store for next iteration
                    previous_race_results = race_results
                    
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
    
    def _predict_grids_for_races(self, remaining_races):
        """
        Get actual or predicted qualifying grids for remaining races
        Tries FastF1 API first for completed qualifying, falls back to predictions
        
        Returns:
            dict: {race_name: {driver: grid_position}}
        """
        predicted_grids = {}
        
        print(f"   📊 Getting starting grids for {len(remaining_races)} race(s)...")
        
        for race_name in remaining_races:
            # Try to get actual qualifying results from FastF1
            actual_grid = self._get_actual_qualifying_from_fastf1(race_name)
            
            if actual_grid:
                predicted_grids[race_name] = actual_grid
                print(f"   ✓ Loaded ACTUAL qualifying grid for {race_name} from FastF1")
            elif self.qualifying_predictor:
                # Fall back to prediction if qualifying hasn't happened yet
                try:
                    quali_prediction = self.qualifying_predictor.predict_qualifying(race_name)
                    grid = {}
                    for pred in quali_prediction['predicted_grid']:
                        grid[pred['driver']] = pred['predicted_grid']
                    predicted_grids[race_name] = grid
                    print(f"   ✓ Predicted grid for {race_name} (qualifying not yet completed)")
                except Exception as e:
                    print(f"   ⚠️  Could not predict grid for {race_name}: {e}")
                    predicted_grids[race_name] = None
            else:
                print(f"   ⚠️  No data available for {race_name}, using performance-based grid")
                predicted_grids[race_name] = None
        
        return predicted_grids
    
    def _get_actual_qualifying_from_fastf1(self, race_name, year=2025):
        """
        Get actual qualifying results from FastF1 API if available
        
        Args:
            race_name (str): Name of the race
            year (int): Season year
            
        Returns:
            dict: {driver: grid_position} or None if not available
        """
        try:
            import fastf1
            from datetime import datetime, timezone
            
            # Get the race schedule
            schedule = fastf1.get_event_schedule(year)
            current_time = datetime.now(timezone.utc)
            
            # Find the race in schedule
            race_event = None
            for idx, event in schedule.iterrows():
                if event['EventName'] == race_name:
                    race_event = event
                    break
            
            if race_event is None:
                return None
            
            # Check if qualifying has happened (event date has passed)
            event_date = race_event['EventDate']
            if event_date.tzinfo is None:
                event_date = event_date.replace(tzinfo=timezone.utc)
            
            # If race hasn't happened yet, qualifying likely hasn't either
            if current_time < event_date:
                # Check if we're close to race day (qualifying might be done)
                # Qualifying usually happens 1 day before race
                days_until_race = (event_date - current_time).days
                
                if days_until_race > 1:
                    # Too early, qualifying definitely hasn't happened
                    return None
            
            # Try to load qualifying session
            try:
                round_num = race_event['RoundNumber']
                qualifying = fastf1.get_session(year, round_num, 'Q')
                qualifying.load()
                
                # Get qualifying results
                results = qualifying.results
                
                if results.empty:
                    return None
                
                # Extract grid positions
                grid = {}
                for idx, row in results.iterrows():
                    driver_code = row['Abbreviation']
                    # Use Position from qualifying results
                    if pd.notna(row['Position']):
                        grid[driver_code] = int(row['Position'])
                
                if len(grid) > 0:
                    return grid
                else:
                    return None
                    
            except Exception as e:
                # Qualifying session not available or not loaded yet
                return None
                
        except Exception as e:
            # Any error in fetching, return None to fall back to prediction
            return None
    
    def _predict_grid_from_previous_race(self, previous_results, driver_tiers):
        """
        Predict next race starting grid based on previous race results
        Good performers likely to qualify well again
        
        Args:
            previous_results (list): List of drivers in finish order
            driver_tiers (dict): Driver performance ratings
            
        Returns:
            dict: {driver: estimated_grid_position}
        """
        if not previous_results:
            return None
        
        # Drivers who finished well are more likely to qualify well
        # But add variance for track-specific performance
        grid = {}
        
        for driver in driver_tiers.keys():
            if driver in previous_results:
                # Find finish position
                finish_pos = previous_results.index(driver) + 1
                
                # Grid position correlates with race finish but with variance
                # Top finishers likely to qualify well again (±3 positions variance)
                base_grid = finish_pos
                variance = np.random.randint(-3, 4)
                
                # Better drivers have less variance (more consistent)
                driver_skill = driver_tiers.get(driver, 0.5)
                if driver_skill > 0.85:
                    variance = variance // 2  # Top drivers more consistent
                
                predicted_grid = max(1, min(20, base_grid + variance))
                grid[driver] = predicted_grid
            else:
                # Driver DNF'd last race, estimate based on performance tier
                driver_skill = driver_tiers.get(driver, 0.5)
                if driver_skill > 0.85:
                    grid[driver] = np.random.randint(1, 8)
                elif driver_skill > 0.75:
                    grid[driver] = np.random.randint(6, 14)
                else:
                    grid[driver] = np.random.randint(12, 21)
        
        return grid
    
    def _simulate_single_race(self, driver_tiers, starting_grid=None):
        """
        Simulate a single race with weighted random results
        
        Args:
            driver_tiers (dict): Performance weights for each driver
            starting_grid (dict): Starting positions {driver: grid_position}
            
        Returns:
            list: Ordered list of drivers (winner first)
        """
        # Get performance weights
        drivers = list(driver_tiers.keys())
        
        race_weights = []
        for driver in drivers:
            base_weight = driver_tiers[driver]
            
            # Grid position influence (starting further forward is advantage)
            if starting_grid and driver in starting_grid:
                grid_pos = starting_grid[driver]
                # Convert grid position to weight bonus (P1 = +0.15, P20 = -0.05)
                grid_bonus = (21 - grid_pos) * 0.01  # Max +0.20, Min -0.00
            else:
                # No grid info, use performance tier to estimate
                if base_weight > 0.85:
                    grid_bonus = 0.15
                elif base_weight > 0.75:
                    grid_bonus = 0.05
                else:
                    grid_bonus = -0.05
            
            # Add randomness for race incidents, strategy, luck (±20%)
            race_variance = np.random.uniform(0.85, 1.15)
            
            # Calculate final race weight
            final_weight = (base_weight + grid_bonus) * race_variance
            
            # DNF probability (5% base, higher for incidents)
            dnf_chance = 0.05
            if np.random.random() < dnf_chance:
                final_weight = 0  # DNF = worst position
            
            race_weights.append(final_weight)
        
        # Sort drivers by their race performance
        driver_performance = list(zip(drivers, race_weights))
        driver_performance.sort(key=lambda x: x[1], reverse=True)
        
        return [driver for driver, _ in driver_performance]
    
    def get_remaining_races_info(self, year=2025):
        """
        Get information about remaining races in the season
        
        Args:
            year (int): Year to check (default: 2025)
        
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
            # Fallback for 2025 end of season
            return ['Qatar Grand Prix', 'Abu Dhabi Grand Prix']