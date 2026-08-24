import os
import fastf1
import pandas as pd

class F1RainEngine:
    """
    Analysis engine to extract pure degradation coefficients for wet weather 
    compounds (Intermediate and Full Wet) from historical 2021 FIA telemetry.
    """
    def __init__(self, cache_dir: str = 'f1_cache'):
        self.cache_dir = cache_dir
        self._setup_cache()
        self.fuel_effect_per_10kg = 0.3 # Time advantage in seconds per 10kg lost

    def _setup_cache(self):
        """Initializes local caching directory for FastF1 data storage."""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        fastf1.Cache.enable_cache(self.cache_dir)

    def calculate_rain_segment(self, laps_obj, driver_code: str, max_lap: int, fuel_consumption: float) -> pd.DataFrame:
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
            
            # Fuel weight correction logic (isolating clean tyre wear)
            burned_fuel = lap_num * fuel_consumption
            fuel_advantage = (burned_fuel / 10.0) * self.fuel_effect_per_10kg
            corrected_time = actual_time + fuel_advantage
            
            report.append({
                "Lap": lap_num,
                "Actual_Time": actual_time,
                "Corrected_Time": corrected_time,
                "Compound": row['Compound']
            })
            
        return pd.DataFrame(report)

    def analyze_intermediates(self):
        """Analyzes Intermediate compound degradation under damp conditions (Turkey 2021)."""
        session = fastf1.get_session(2021, 'Turkey', 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        df_inter = self.calculate_rain_segment(session.laps, 'BOT', max_lap=35, fuel_consumption=1.35)
        self._print_metrics("INTERMEDIATE PURE (BOT, Turkey 2021, Laps 1-35)", df_inter)

    def analyze_full_wets(self):
        """Analyzes Full Wet compound degradation under heavy rain conditions (Imola 2021)."""
        # FIXED: Changed specific schedule name string to match exact historical database schema
        session = fastf1.get_session(2021, 'EmiliaRomagna', 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        df_wet = self.calculate_rain_segment(session.laps, 'HAM', max_lap=9, fuel_consumption=1.35)
        self._print_metrics("FULL WET PURE (HAM, Imola 2021, Laps 1-9)", df_wet)

    def _print_metrics(self, label: str, df: pd.DataFrame):
        if df.empty:
            print(f"\n[{label}] Error: Failed to extract session data.")
            return
            
        # FIXED: Enforced explicit scalar scalar position extraction to avoid object indexing errors
        total_degrad = df['Corrected_Time'].iloc[-1] - df['Corrected_Time'].iloc[0]
        avg_degrad = total_degrad / len(df)
        
        print(f"\n[{label}]")
        print(f"  Total laps analyzed: {len(df)}")
        print(f"  Net pace delta over segment: {total_degrad:.3f} sec")
        print(f"  Degradation coefficient per lap: {avg_degrad:.3f} sec")

if __name__ == "__main__":
    analyzer = F1RainEngine()
    print("Initializing wet weather historical session analysis...")
    analyzer.analyze_intermediates()
    analyzer.analyze_full_wets()
