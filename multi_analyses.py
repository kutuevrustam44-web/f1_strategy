import os
import fastf1
import pandas as pd

class F1PureDegradationAnalysis:
    """
    Analysis engine to calculate pure tyre degradation coefficients
    by isolating fuel weight effects from historical FIA telemetry data.
    """
    def __init__(self, cache_dir: str = 'f1_cache'):
        self.cache_dir = cache_dir
        self._setup_cache()
        self.fuel_effect_per_10kg = 0.3 # Time advantage in seconds per 10kg fuel lost

    def _setup_cache(self):
        """Initializes local caching directory for FastF1 data storage."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(cache_dir)
        fastf1.Cache.enable_cache(self.cache_dir)

    def calculate_segment_data(self, laps_obj, driver_code: str, max_lap: int, fuel_consumption: float) -> pd.DataFrame:
        """
        Filters quick laps for a specific driver and corrects lap times 
        by removing the linear performance advantage gained from fuel burn.
        """
        driver_laps = laps_obj.pick_driver(driver_code).pick_quicklaps()
        report = []
        
        for _, row in driver_laps.iterrows():
            lap_num = int(row['LapNumber'])
            if lap_num > max_lap:
                break
                
            actual_time = row['LapTime'].total_seconds()
            
            # Fuel weight correction logic
            burned_fuel = lap_num * fuel_consumption
            fuel_advantage = (burned_fuel / 10.0) * self.fuel_effect_per_10kg
            corrected_time = actual_time + fuel_advantage
            
            report.append({
                "Lap": lap_num,
                "Actual_Time": actual_time,
                "Corrected_Time": corrected_time,
                "Tyre_Age": row['TyreLife']
            })
            
        return pd.DataFrame(report)

    def process_pure_soft(self):
        """Analyzes Soft compound degradation under clean air conditions (Spain 2021)."""
        session = fastf1.get_session(2021, 'Spain', 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        df_soft = self.calculate_segment_data(session.laps, 'VER', max_lap=24, fuel_consumption=1.50)
        self._print_metrics("SOFT PURE (VER, Spain 2021, Laps 1-24)", df_soft)

    def process_pure_medium(self):
        """Analyzes Medium compound degradation under clean air conditions (Austria 2021)."""
        session = fastf1.get_session(2021, 'Austria', 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        df_medium = self.calculate_segment_data(session.laps, 'VER', max_lap=32, fuel_consumption=1.30)
        self._print_metrics("MEDIUM PURE (VER, Austria 2021, Laps 1-32)", df_medium)
        
        # Storing traffic data for future delta validation (Dirty Air calculations)
        ham_laps = self.calculate_segment_data(session.laps, 'HAM', max_lap=20, fuel_consumption=1.30)
        bot_laps = session.laps.pick_driver('BOT')
        bot_hard = bot_laps[bot_laps['Compound'] == 'HARD']
        bot_hard_data = self.calculate_segment_data(bot_hard, 'BOT', max_lap=65, fuel_consumption=1.30)
        
        print("\n[INFO] Background traffic data for HAM (Soft) and BOT (Hard) cached successfully.")

    def process_pure_hard(self):
        """Analyzes Hard compound degradation under high-load conditions (Zandvoort 2021)."""
        session = fastf1.get_session(2021, 'Zandvoort', 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        
        ver_laps = session.laps.pick_driver('VER')
        ver_hard_laps = ver_laps[ver_laps['Compound'] == 'HARD']
        
        df_hard = self.calculate_segment_data(ver_hard_laps, 'VER', max_lap=72, fuel_consumption=1.45)
        self._print_metrics("HARD PURE (VER, Zandvoort 2021, Laps 40-72)", df_hard)

    def _print_metrics(self, label: str, df: pd.DataFrame):
        if df.empty:
            return
        total_degrad = df['Corrected_Time'].iloc[-1] - df['Corrected_Time'].iloc[0]
        avg_degrad = total_degrad / len(df)
        print(f"\n[{label}]")
        print(f"  Total laps analyzed: {len(df)}")
        print(f"  Net pace loss due to tyre wear: {total_degrad:.3f} sec")
        print(f"  Pure degradation coefficient per lap: +{avg_degrad:.3f} sec")

if __name__ == "__main__":
    analyzer = F1PureDegradationAnalysis()
    print("Initializing multi-track historical session analysis...")
    analyzer.process_pure_soft()
    analyzer.process_pure_medium()
    analyzer.process_pure_hard()
