from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =============================================================================
# CONFIGURATION
# =============================================================================

PIRELLI_COLORS = {
    "Soft": "#E60000",
    "Medium": "#FCD700",
    "Hard": "#808080",
    "Intermediate": "#39B54A",
    "Wet": "#0072CE",
}

SLICK_COMPOUNDS = ("Soft", "Medium", "Hard")
RAIN_COMPOUNDS = ("Intermediate", "Wet")
ALL_COMPOUNDS = SLICK_COMPOUNDS + RAIN_COMPOUNDS
WEATHER_MODES = ("Dry", "Damp", "Wet")

BASE_DEGRADATION = {
    "Soft": 1.80,
    "Medium": 1.20,
    "Hard": 0.80,
    "Intermediate": 1.00,
    "Wet": 1.10,
}

BASE_LAP_DELTA = {
    "Soft": 0.00,
    "Medium": 0.60,
    "Hard": 1.20,
    "Intermediate": 3.50,
    "Wet": 4.50,
}

OUT_LAP_PENALTY = {
    "Soft": 1.5,
    "Medium": 2.2,
    "Hard": 3.5,
    "Intermediate": 2.0,
    "Wet": 4.0,
}

DEFAULT_INVENTORY_COUNTS = {
    "Soft": 8,
    "Medium": 3,
    "Hard": 2,
    "Intermediate": 4,
    "Wet": 3,
}

SLICK_WET_TIME_PENALTY = 22.0
SLICK_DAMP_TIME_PENALTY = 8.5

SLICK_WET_WEAR_MULTIPLIER = 5.0
SLICK_DAMP_WEAR_MULTIPLIER = 2.0

WING_REPAIR_PENALTY = 10.0

# Dirty air (running in another car's turbulent wake) increases tyre
# wear on top of the fixed pace penalty already modelled per lap.
DIRTY_AIR_WEAR_MULTIPLIER = 1.15


# =============================================================================
# CIRCUIT DATABASE
# =============================================================================

@dataclass(frozen=True)
class TrackSpec:
    name: str
    base_lap_time: float
    pit_loss: float
    fuel_burn: float
    tyre_stress: float
    total_laps: int
    pit_time: float


TRACKS_DATABASE: Dict[str, TrackSpec] = {
    "Bahrain": TrackSpec("Bahrain", 94.0, 22.5, 1.45, 1.55, 57, 22.5),
    "Imola": TrackSpec("Imola", 77.5, 21.0, 1.30, 1.40, 63, 21.0),
    "Portimao": TrackSpec("Portimao", 81.0, 22.0, 1.35, 1.50, 66, 22.0),
    "Barcelona": TrackSpec("Barcelona", 80.0, 22.5, 1.35, 1.60, 66, 22.5),
    "Monaco": TrackSpec("Monaco", 75.0, 20.0, 1.10, 1.75, 78, 20.0),
    "Baku": TrackSpec("Baku", 105.0, 21.5, 1.40, 1.35, 51, 21.5),
    "Paul Ricard": TrackSpec("Paul Ricard", 95.0, 22.0, 1.40, 1.45, 53, 22.0),
    "Red Bull Ring": TrackSpec("Red Bull Ring", 66.5, 20.5, 1.25, 1.55, 71, 20.5),
    "Silverstone": TrackSpec("Silverstone", 89.0, 23.0, 1.45, 1.70, 52, 23.0),
    "Hungaroring": TrackSpec("Hungaroring", 81.0, 21.5, 1.30, 1.55, 70, 21.5),
    "Spa": TrackSpec("Spa", 109.0, 24.0, 1.60, 1.50, 44, 24.0),
    "Zandvoort": TrackSpec("Zandvoort", 73.0, 21.0, 1.25, 1.65, 72, 21.0),
    "Monza": TrackSpec("Monza", 82.5, 21.5, 1.35, 1.25, 53, 21.5),
    "Sochi": TrackSpec("Sochi", 95.5, 22.0, 1.35, 1.40, 53, 22.0),
    "Istanbul": TrackSpec("Istanbul", 92.0, 22.5, 1.40, 1.60, 58, 22.5),
    "COTA": TrackSpec("COTA", 98.0, 23.0, 1.45, 1.55, 56, 23.0),
    "Mexico": TrackSpec("Mexico", 81.5, 21.5, 1.20, 1.45, 71, 21.5),
    "Interlagos": TrackSpec("Interlagos", 72.0, 22.0, 1.35, 1.50, 71, 22.0),
    "Lusail": TrackSpec("Lusail", 84.5, 22.0, 1.40, 1.50, 57, 22.0),
    "Jeddah": TrackSpec("Jeddah", 90.5, 22.5, 1.50, 1.45, 50, 22.5),
    "Yas Marina": TrackSpec("Yas Marina", 88.0, 22.0, 1.35, 1.40, 58, 22.0),
    "Suzuka": TrackSpec("Suzuka", 92.0, 23.0, 1.45, 1.75, 53, 23.0),
    "Marina Bay": TrackSpec("Marina Bay", 101.0, 23.5, 1.40, 1.65, 61, 23.5),
    "Shanghai": TrackSpec("Shanghai", 94.5, 23.0, 1.42, 1.60, 56, 23.0),
    "Miami": TrackSpec("Miami", 91.5, 22.0, 1.38, 1.45, 57, 22.0),
    "Las Vegas": TrackSpec("Las Vegas", 96.0, 24.0, 1.52, 1.20, 50, 24.0),
    "Generic Sprint": TrackSpec("Generic Sprint", 85.0, 22.0, 1.35, 1.50, 60, 22.0),
}


# =============================================================================
# FORMATTING
# =============================================================================

def format_lap_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:06.3f}"


def format_race_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    rem = seconds - hours * 3600
    minutes = int(rem // 60)
    secs = rem - minutes * 60
    return f"{hours}h {minutes:02d}m {secs:05.1f}s"


# =============================================================================
# DYNAMIC TRACK CONDITIONS RADAR
# =============================================================================

@dataclass(frozen=True)
class WeatherRadar:
    """
    Rain onset is configured in seconds after the race start.

    Each lap uses accumulated race time at the start of that lap.
    Rain duration is configured in laps. After the primary rain window,
    an automatic 11-lap Damp drying profile is generated.
    """

    rain_start_seconds: float = 999999.0
    rain_duration_laps: int = 0
    rain_intensity: str = "Wet"
    drying_laps: int = 11

    def weather_for_lap(
        self,
        lap: int,
        race_elapsed_seconds: float,
        base_lap_time: float,
    ) -> str:
        if self.rain_duration_laps <= 0:
            return "Dry"

        if race_elapsed_seconds < self.rain_start_seconds:
            return "Dry"

        approx_rain_start_lap = max(
            1,
            int(self.rain_start_seconds // max(base_lap_time, 1.0)) + 1,
        )

        rain_end_lap = (
            approx_rain_start_lap
            + self.rain_duration_laps
            - 1
        )

        drying_end_lap = rain_end_lap + self.drying_laps

        if approx_rain_start_lap <= lap <= rain_end_lap:
            return self.rain_intensity

        if rain_end_lap < lap <= drying_end_lap:
            return "Damp"

        return "Dry"

    def timeline(self, total_laps: int, base_lap_time: float) -> List[str]:
        result: List[str] = []
        elapsed = 0.0

        for lap in range(1, int(total_laps) + 1):
            weather = self.weather_for_lap(
                lap=lap,
                race_elapsed_seconds=elapsed,
                base_lap_time=base_lap_time,
            )
            result.append(weather)
            elapsed += base_lap_time

        return result


# =============================================================================
# TYRE PHYSICS
# =============================================================================

@dataclass
class TyrePhysics:
    composition: str
    health: float = 100.0
    age: int = 0
    base_lap_time: float = 90.0
    tyre_stress: float = 1.0
    weather: str = "Dry"
    dirty_air: bool = False
    wing_damage: str = "None"
    track_temperature: float = 35.0
    PIT_OUT_LAP: bool = False

    CLIFF_THRESHOLD = 30.0
    BREAK_THRESHOLD = 15.0

    def _temperature_factor(self) -> float:
        temp = float(self.track_temperature)

        if self.composition in RAIN_COMPOUNDS:
            if temp < 20.0:
                return 0.85
            if temp <= 30.0:
                return 1.0
            if temp <= 40.0:
                return 1.25
            return 1.80

        if temp < 20.0:
            return 0.80
        if temp < 25.0:
            return 0.90
        if temp <= 40.0:
            return 1.0
        if temp <= 45.0:
            return 1.15
        if temp <= 50.0:
            return 1.35
        return 1.60

    def _temperature_comp_factor(self) -> float:
        temp = float(self.track_temperature)

        if self.composition == "Soft":
            if temp > 45.0:
                return 1.35
            if temp > 40.0:
                return 1.15

        if self.composition == "Medium":
            if temp > 48.0:
                return 1.20

        if self.composition == "Hard":
            if temp > 50.0:
                return 1.10

        return 1.0

    def degradation_per_lap(self) -> float:
        base = (
            BASE_DEGRADATION[self.composition]
            * float(self.tyre_stress)
            * self._temperature_factor()
            * self._temperature_comp_factor()
        )

        if self.composition == "Intermediate":
            if self.weather == "Wet":
                base *= 1.60
            elif self.weather == "Damp":
                base *= 0.75 if self.health >= 40.0 else 2.0
            else:
                base *= 2.50

        elif self.composition == "Wet":
            if self.weather == "Wet":
                base *= 1.00
            elif self.weather == "Damp":
                base *= 2.50
            else:
                base *= 6.00

        elif self.composition in SLICK_COMPOUNDS:
            if self.weather == "Wet":
                base *= SLICK_WET_WEAR_MULTIPLIER
            elif self.weather == "Damp":
                base *= SLICK_DAMP_WEAR_MULTIPLIER

        if self.dirty_air:
            base *= DIRTY_AIR_WEAR_MULTIPLIER

        return float(base)

    def _temperature_time_delta(self) -> float:
        temp = float(self.track_temperature)

        if self.composition in RAIN_COMPOUNDS:
            if temp < 15.0:
                return 0.8
            if temp <= 30.0:
                return 0.0
            if temp <= 40.0:
                return 0.5
            return 1.2

        if temp < 20.0:
            return 1.5
        if temp < 25.0:
            return 0.7
        if temp <= 40.0:
            return 0.0
        if temp <= 45.0:
            return 0.3
        if temp <= 50.0:
            return 0.8
        return 1.5

    def _wear_time_coeff(self) -> float:
        return {
            "Soft": 3.5,
            "Medium": 2.8,
            "Hard": 2.2,
            "Intermediate": 2.5,
            "Wet": 2.0,
        }[self.composition]

    def lap_time_delta(self) -> float:
        delta = float(BASE_LAP_DELTA[self.composition])

        wear_factor = ((100.0 - self.health) / 100.0) ** 1.6
        delta += wear_factor * self._wear_time_coeff()

        if self.health < self.CLIFF_THRESHOLD:
            delta += 2.5

        # Hard puncture physics. Once the tyre drops below the break
        # threshold, the carcass structure starts to fail and the lap
        # time gets a heavy fixed penalty on top of normal wear loss.
        if 0.0 < self.health < self.BREAK_THRESHOLD:
            delta += 15.0
        elif self.health == 0.0:
            delta += 30.0

        delta += self._temperature_time_delta()

        if self.composition in SLICK_COMPOUNDS:
            if self.weather == "Wet":
                delta += SLICK_WET_TIME_PENALTY
            elif self.weather == "Damp":
                delta += SLICK_DAMP_TIME_PENALTY

        if self.composition == "Intermediate":
            if self.weather == "Wet":
                delta -= 1.0
            elif self.weather == "Damp":
                delta -= 0.6
            else:
                delta += 2.5

        if self.composition == "Wet":
            if self.weather == "Wet":
                delta -= 1.5
            elif self.weather == "Damp":
                delta += 0.35 * float(self.age)
            else:
                delta += 5.0

        if self.wing_damage == "Minor":
            delta += 0.4
        elif self.wing_damage == "Critical":
            delta += 1.2

        if self.dirty_air:
            delta += 0.25

        return float(delta)

    def compute_lap_time(self, track_evolution: float = 1.0) -> float:
        lap_time = (
            float(self.base_lap_time)
            * float(track_evolution)
            + self.lap_time_delta()
        )

        if self.PIT_OUT_LAP:
            lap_time += OUT_LAP_PENALTY[self.composition]

        return float(lap_time)

    def apply_lap(self) -> None:
        self.health = max(
            0.0,
            float(self.health) - self.degradation_per_lap(),
        )
        self.age = int(self.age) + 1
        self.PIT_OUT_LAP = False

    def is_cliff(self) -> bool:
        return float(self.health) < self.CLIFF_THRESHOLD

    def is_break(self) -> bool:
        return float(self.health) < self.BREAK_THRESHOLD


def simulate_inventory_pre_wear(
    compound: str,
    age_laps: int,
    tyre_stress: float,
    track_temperature: float,
) -> float:
    """
    Background pre-wear calculation for an allocated tyre set.

    Runs `age_laps` hidden qualifying laps
    using the physical degradation model (BASE_DEGRADATION *
    circuit tyre stress * temperature coefficient) and
    returns final tyre life as a percentage.
    """

    scratch_tyre = TyrePhysics(
        composition=compound,
        health=100.0,
        age=0,
        tyre_stress=float(tyre_stress),
        weather="Dry",
        track_temperature=float(track_temperature),
    )

    for _ in range(max(0, int(age_laps))):
        scratch_tyre.apply_lap()

    return float(scratch_tyre.health)


# =============================================================================
# CAR STATE
# =============================================================================

@dataclass
class CarState:
    track: TrackSpec
    current_tyre: TyrePhysics
    fuel: float

    lap: int = 1
    total_time: float = 0.0

    laps_history: List[Dict[str, Any]] = field(default_factory=list)
    pit_stops: List[Dict[str, Any]] = field(default_factory=list)

    tyre_sets_inventory: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=dict
    )

    weather_radar: WeatherRadar = field(default_factory=WeatherRadar)

    pending_pit: bool = False
    pending_set_id: Optional[str] = None

    dirty_air: bool = False
    wing_damage: str = "None"

    track_temperature: float = 35.0

    time_penalty: float = 0.0
    penalty_served: bool = False

    force_wing_replacement_next_pit: bool = False
    race_started: bool = False

    safety_car_active: bool = False

    @property
    def remaining_laps(self) -> int:
        return max(
            0,
            int(self.track.total_laps) - int(self.lap) + 1,
        )

    @property
    def track_evolution(self) -> float:
        if self.track.total_laps <= 1:
            return 1.0

        progress = (
            float(self.lap - 1)
            / float(self.track.total_laps - 1)
        )

        return 1.05 - progress * 0.10

    def weather_for_current_lap(self) -> str:
        return self.weather_radar.weather_for_lap(
            lap=int(self.lap),
            race_elapsed_seconds=float(self.total_time),
            base_lap_time=float(self.track.base_lap_time),
        )

    def record_lap(
        self,
        lap_time: float,
        health_before: float,
        weather: str,
        is_in_lap: bool,
        wing_repair: bool,
    ) -> None:
        self.laps_history.append(
            {
                "lap": int(self.lap),
                "time": float(lap_time),
                "composition": str(self.current_tyre.composition),
                "health_before": float(health_before),
                "health_after": float(self.current_tyre.health),
                "weather": str(weather),
                "temperature": float(self.track_temperature),
                "is_in_lap": bool(is_in_lap),
                "is_out_lap": bool(
                    self.current_tyre.PIT_OUT_LAP
                ),
                "wing_repair": bool(wing_repair),
            }
        )

        self.total_time += float(lap_time)

        self.fuel = max(
            0.0,
            float(self.fuel) - float(self.track.fuel_burn),
        )


# =============================================================================
# SHARED SIMULATION FUNCTIONS
# =============================================================================

def evolution_at(lap: int, total_laps: int) -> float:
    if total_laps <= 1:
        return 1.0

    progress = (
        float(lap - 1)
        / float(total_laps - 1)
    )

    return 1.05 - progress * 0.10


def weather_for_simulation_lap(
    radar: WeatherRadar,
    lap: int,
    elapsed_seconds: float,
    track: TrackSpec,
) -> str:
    return radar.weather_for_lap(
        lap=int(lap),
        race_elapsed_seconds=float(elapsed_seconds),
        base_lap_time=float(track.base_lap_time),
    )


# =============================================================================
# GLOBAL STRATEGY OPTIMIZER
# =============================================================================

class StrategyOptimizer:
    """
    Multi-stop brute-force strategy evaluation from the current race state.

    A plan is represented as:
    [(pit_lap_1, compound_1), (pit_lap_2, compound_2), ...]

    Monaco and Monza are limited to 0-stop and 1-stop plans. All other
    circuits evaluate 0 to 3 stops with a mandatory six-lap separation.
    """

    LAP_STEP = 5
    TOP_N = 5
    MIN_STOP_GAP = 6
    MAX_STOPS = 3

    def __init__(self, car: CarState, lap_step: int = 5):
        self.car = car
        self.lap_step = max(1, int(lap_step))

    def _fresh_inventory(self) -> Dict[str, List[Dict[str, Any]]]:
        return copy.deepcopy(self.car.tyre_sets_inventory)

    def _consume_set(
        self,
        inventory: Dict[str, List[Dict[str, Any]]],
        compound: str,
    ) -> Optional[Dict[str, Any]]:
        if not inventory.get(compound):
            return None
        return inventory[compound].pop(0)

    def _max_stops(self) -> int:
        if self.car.track.name in ("Monaco", "Monza"):
            return 1
        return self.MAX_STOPS

    def _weather_at_lap(self, lap: int) -> str:
        track = self.car.track
        elapsed = float(max(0, lap - 1)) * float(track.base_lap_time)
        return self.car.weather_radar.weather_for_lap(
            lap=int(lap),
            race_elapsed_seconds=elapsed,
            base_lap_time=float(track.base_lap_time),
        )

    def _track_condition_changed(self, lap_a: int, lap_b: int) -> bool:
        """True if the forecast track condition differs between two laps."""
        return self._weather_at_lap(lap_a) != self._weather_at_lap(lap_b)

    def _pit_lap_candidates(self) -> List[int]:
        start_lap = max(1, int(self.car.lap))
        finish = int(self.car.track.total_laps)

        # Under Safety Car / VSC conditions an immediate pit stop on the
        # current lap is a real strategic option (the field is bunched up
        # and slow, so track position is barely lost), so the current lap
        # itself becomes a valid pit-lap candidate.
        if bool(getattr(self.car, "safety_car_active", False)):
            first = start_lap
        else:
            first = start_lap + 1

        if first >= finish:
            return []
        return list(range(first, finish, self.lap_step))

    def _generate_plans(self) -> List[List[Tuple[int, str]]]:
        """
        Generate legal pit-stop strategies with a bounded iterative search.

        The search space is intentionally restricted to:
        - 0-stop strategies
        - 1-stop strategies
        - 2-stop strategies

        Every pair of consecutive pit stops must have a minimum six-lap gap.
        Monaco and Monza are restricted to 0-stop and 1-stop strategies only.
        """
        pit_laps = self._pit_lap_candidates()
        max_stops = min(2, self._max_stops())
        compounds = tuple(ALL_COMPOUNDS)
        plans: List[List[Tuple[int, str]]] = [[]]

        # One-stop layer.
        if max_stops >= 1:
            for lap_1 in pit_laps:
                for comp_1 in compounds:
                    plans.append([(int(lap_1), str(comp_1))])

        # Two-stop layer.
        if max_stops >= 2:
            for index_1, lap_1 in enumerate(pit_laps):
                for lap_2 in pit_laps[index_1 + 1:]:
                    if int(lap_2) - int(lap_1) < self.MIN_STOP_GAP:
                        continue

                    for comp_1 in compounds:
                        for comp_2 in compounds:
                            if (
                                comp_1 == comp_2
                                and not self._track_condition_changed(
                                    int(lap_1),
                                    int(lap_2),
                                )
                            ):
                                continue

                            plans.append(
                                [
                                    (int(lap_1), str(comp_1)),
                                    (int(lap_2), str(comp_2)),
                                ]
                            )

        return plans

    def _simulate_strategy(
        self,
        start_comp: str,
        plan: List[Tuple[int, str]],
    ) -> Optional[Dict[str, Any]]:
        """
        Simulate the complete remaining race for a multi-stop plan.

        Planned stops are executed sequentially. A sporting penalty is served
        on the first available in-lap using a strict hands-off procedure:
        penalty seconds are added, tyre and wing service are deferred, and the
        penalty is then cleared.
        """
        track = self.car.track
        total_laps = int(track.total_laps)
        start_lap = max(1, int(self.car.lap))
        inventory = self._fresh_inventory()

        live_start = int(self.car.lap) > 1 or bool(
            getattr(self.car, "race_started", False)
        )

        if live_start:
            if str(self.car.current_tyre.composition) != str(start_comp):
                return None
            start_set = {
                "id": "LIVE_CURRENT_SET",
                "health": float(self.car.current_tyre.health),
                "age": int(self.car.current_tyre.age),
            }
        else:
            start_set = self._consume_set(inventory, str(start_comp))
            if start_set is None:
                return None

        tyre = TyrePhysics(
            composition=str(start_comp),
            health=float(start_set["health"]),
            age=int(start_set["age"]),
            base_lap_time=float(track.base_lap_time),
            tyre_stress=float(track.tyre_stress),
            weather="Dry",
            track_temperature=float(self.car.track_temperature),
        )

        plan = [(int(lap), str(comp)) for lap, comp in plan]
        plan.sort(key=lambda item: item[0])

        if len({lap for lap, _ in plan}) != len(plan):
            return None

        for index in range(1, len(plan)):
            if plan[index][0] - plan[index - 1][0] < self.MIN_STOP_GAP:
                return None

        if any(
            lap < start_lap or lap >= total_laps
            for lap, _ in plan
        ):
            return None

        total_time = 0.0
        lap_history: List[Dict[str, Any]] = []
        transitions = [{
            "lap": int(start_lap),
            "composition": str(start_comp),
            "note": "Active tyre set" if live_start else "Starting tyre set",
        }]

        current_comp = str(start_comp)
        current_health = float(start_set["health"])
        current_age = int(start_set["age"])
        out_lap_next = False
        plan_index = 0

        penalty = float(self.car.time_penalty)
        penalty_served = bool(self.car.penalty_served) or penalty <= 0.0
        wing_damage = str(self.car.wing_damage)
        force_wing = bool(self.car.force_wing_replacement_next_pit)

        for lap in range(start_lap, total_laps + 1):
            weather = weather_for_simulation_lap(
                radar=self.car.weather_radar,
                lap=lap,
                elapsed_seconds=total_time,
                track=track,
            )

            scheduled_stop = (
                plan_index < len(plan)
                and lap == int(plan[plan_index][0])
            )

            tyre = TyrePhysics(
                composition=str(current_comp),
                health=float(current_health),
                age=int(current_age),
                base_lap_time=float(track.base_lap_time),
                tyre_stress=float(track.tyre_stress),
                weather=str(weather),
                dirty_air=bool(self.car.dirty_air),
                wing_damage=str(wing_damage),
                track_temperature=float(self.car.track_temperature),
                PIT_OUT_LAP=bool(out_lap_next),
            )

            out_lap_next = False
            was_out_lap = bool(tyre.PIT_OUT_LAP)
            health_before = float(tyre.health)
            lap_time = tyre.compute_lap_time(
                evolution_at(lap, total_laps)
            )

            if lap == 1 and not live_start:
                lap_time += 3.0

            wing_repair = False
            penalty_served_this_lap = False
            fitted_compound: Optional[str] = None

            if scheduled_stop:
                # If Safety Car / VSC is active and this plan pits on the
                # very lap the car is currently on, only that first stop
                # gets the halved SC pit loss. Every later planned stop on
                # a future lap still pays the full green-flag pit loss.
                if (
                    bool(getattr(self.car, "safety_car_active", False))
                    and lap == start_lap
                ):
                    lap_time += float(track.pit_loss) / 2.0
                else:
                    lap_time += float(track.pit_loss)

                if not penalty_served and penalty > 0.0:
                    lap_time += float(penalty)
                    penalty = 0.0
                    penalty_served = True
                    penalty_served_this_lap = True
                    transitions.append({
                        "lap": int(lap),
                        "composition": str(current_comp),
                        "note": "Penalty service: hands-off",
                    })
                else:
                    if wing_damage in ("Minor", "Critical") or force_wing:
                        lap_time += WING_REPAIR_PENALTY
                        wing_damage = "None"
                        force_wing = False
                        wing_repair = True

            tyre.apply_lap()

            lap_history.append({
                "lap": int(lap),
                "time": float(lap_time),
                "health_before": float(health_before),
                "health_after": float(tyre.health),
                "composition": str(current_comp),
                "weather": str(weather),
                "is_in_lap": bool(scheduled_stop),
                "is_out_lap": bool(was_out_lap),
                "wing_repair": bool(wing_repair),
                "penalty_served": bool(penalty_served_this_lap),
            })

            total_time += float(lap_time)
            current_health = float(tyre.health)
            current_age = int(tyre.age)

            if scheduled_stop:
                planned_compound = str(plan[plan_index][1])
                plan_index += 1

                if not penalty_served_this_lap:
                    next_set = self._consume_set(
                        inventory,
                        planned_compound,
                    )
                    if next_set is None:
                        return None

                    current_comp = planned_compound
                    current_health = float(next_set["health"])
                    current_age = int(next_set["age"])
                    out_lap_next = True
                    fitted_compound = planned_compound

                    transitions.append({
                        "lap": int(lap),
                        "composition": str(planned_compound),
                        "note": "Pit stop",
                    })

                else:
                    transitions.append({
                        "lap": int(lap),
                        "composition": str(planned_compound),
                        "note": "Tyre service deferred after penalty",
                    })

        if plan_index != len(plan):
            return None

        return {
            "total_time": float(total_time),
            "start_compound": str(start_comp),
            "plan": plan,
            "pit_lap": int(plan[0][0]) if plan else None,
            "second_compound": str(plan[0][1]) if plan else None,
            "start_lap": int(start_lap),
            "transitions": transitions,
            "lap_history": lap_history,
        }

    def _dry_race_requires_two_slick_sets(self, result: Dict[str, Any]) -> bool:
        """
        For fully dry races, only show strategies using two different slick
        compounds across the WHOLE race, not just the remaining segment.

        Pit stops already completed earlier in the race count toward the
        two-compound rule, so a strategy that already used two compounds
        does not get forced into another unnecessary stop.
        """
        for lap in range(1, int(self.car.track.total_laps) + 1):
            if self._weather_at_lap(lap) != "Dry":
                return False

        compounds: List[str] = []

        # Compounds already fitted during real pit stops so far this race.
        for pit_stop in self.car.pit_stops:
            compounds.append(str(pit_stop.get("composition", "")))

        # The very first compound the car started the race on.
        if self.car.laps_history:
            compounds.append(str(self.car.laps_history[0].get("composition", "")))

        # The compound of the segment being simulated, plus every future
        # planned stop in this candidate strategy.
        compounds.append(str(result.get("start_compound", "")))
        compounds.extend(str(compound) for _, compound in result.get("plan", []))

        slicks = [compound for compound in compounds if compound in SLICK_COMPOUNDS]
        return len(set(slicks)) >= 2

    def optimize(self) -> List[Dict[str, Any]]:
        inventory = self._fresh_inventory()
        results: List[Dict[str, Any]] = []

        live_start = int(self.car.lap) > 1 or bool(
            getattr(self.car, "race_started", False)
        )

        if live_start:
            start_candidates = [str(self.car.current_tyre.composition)]
        else:
            start_candidates = [
                compound for compound in ALL_COMPOUNDS
                if inventory.get(compound)
            ]

        plans = self._generate_plans()

        for start_comp in start_candidates:
            for plan in plans:
                result = self._simulate_strategy(
                    start_comp=str(start_comp),
                    plan=list(plan),
                )
                if result is not None:
                    results.append(result)

        results.sort(key=lambda item: float(item["total_time"]))

        if all(
            self._weather_at_lap(lap) == "Dry"
            for lap in range(1, int(self.car.track.total_laps) + 1)
        ):
            results = [
                item for item in results
                if self._dry_race_requires_two_slick_sets(item)
            ]

        return results[: self.TOP_N]




# =============================================================================
# LIVE RACE SIMULATION
# =============================================================================

def _drive_one_lap(car: CarState) -> str:
    if car.lap > car.track.total_laps:
        return "Race session complete."

    messages: List[str] = []

    weather = car.weather_for_current_lap()
    car.current_tyre.weather = weather
    car.current_tyre.track_temperature = (
        car.track_temperature
    )
    car.current_tyre.wing_damage = car.wing_damage
    car.current_tyre.dirty_air = car.dirty_air

    # Critical tyre life warning. The system no longer auto-selects or
    # reserves a tyre set from the warehouse inventory on the engineer's
    # behalf. The car remains on track on the degraded set until the
    # strategy engineer manually selects a tyre set and confirms the
    # box call via the "BOX, BOX" control.
    if (
        car.current_tyre.is_break()
        and not car.pending_pit
    ):
        messages.append(
            "CRITICAL TYRE LIFE WARNING: Active compound health dropped "
            "below 15%! Schedule an urgent box window manually via Pit "
            "Wall Center to avoid catastrophic failure."
        )

    is_in_lap = bool(car.pending_pit)

    # The penalty must be served exclusively during a pit stop.
    # Until fully served, mechanics must not perform
    # tyre or front wing service, modelling the FIA protocol where
    # the team remains hands-off.
    penalty_outstanding = bool(
        is_in_lap
        and car.time_penalty > 0.0
        and not car.penalty_served
    )

    health_before = float(
        car.current_tyre.health
    )

    lap_time = car.current_tyre.compute_lap_time(
        car.track_evolution
    )

    if car.lap == 1:
        lap_time += 3.0

    # Under Safety Car / VSC conditions the whole field is bunched up
    # behind the pace car, so every lap on track is driven much slower.
    if car.safety_car_active:
        lap_time *= 1.4

    wing_repair_this_lap = False
    penalty_served_this_lap = False
    pit_under_safety_car = False

    if is_in_lap:

        if car.safety_car_active:
            # Track position is barely lost pitting behind the pace car,
            # so the net pit loss is cut in half versus a green-flag stop.
            lap_time += float(car.track.pit_loss) / 2.0
            pit_under_safety_car = True
        else:
            lap_time += float(car.track.pit_loss)

        if penalty_outstanding:

            # This pit stop is used exclusively to serve the
            # sporting penalty. Tyres and the front wing are not
            # serviced and the planned replacement is
            # deferred to the next pit stop.
            penalty = float(car.time_penalty)

            lap_time += penalty

            car.penalty_served = True
            car.time_penalty = 0.0

            st.session_state.time_penalty = 0.0

            penalty_served_this_lap = True

            messages.append(
                f"Driver successfully served the "
                f"{penalty:.1f} second penalty during the "
                "pit stop. Tyre and front wing service "
                "is deferred to the next pit stop."
            )

        else:

            # Repair the damaged front wing.
            if car.wing_damage in (
                "Minor",
                "Critical",
            ):
                lap_time += WING_REPAIR_PENALTY

                wing_repair_this_lap = True

                messages.append(
                    "Front wing repair completed: "
                    f"+{WING_REPAIR_PENALTY:.1f} sec."
                )

                car.wing_damage = "None"

                st.session_state.wing_damage = "None"
                st.session_state.wing_damage_select = (
                    "None"
                )

            # Force a preventive replacement of the
            # front wing even if it was undamaged.
            if car.force_wing_replacement_next_pit:

                if not wing_repair_this_lap:
                    lap_time += WING_REPAIR_PENALTY

                    messages.append(
                        "Preventive front wing replacement "
                        f"completed: +{WING_REPAIR_PENALTY:.1f} sec."
                    )

                wing_repair_this_lap = True

                car.wing_damage = "None"

                st.session_state.wing_damage = "None"
                st.session_state.wing_damage_select = (
                    "None"
                )

                car.force_wing_replacement_next_pit = (
                    False
                )
                st.session_state.force_wing_next_pit = (
                    False
                )
                st.session_state[
                    "force_wing_next_pit_checkbox"
                ] = False

    # Lap telemetry must be recorded before the tyre change.
    car.current_tyre.apply_lap()

    car.record_lap(
        lap_time=float(lap_time),
        health_before=float(health_before),
        weather=str(weather),
        is_in_lap=bool(is_in_lap),
        wing_repair=bool(
            wing_repair_this_lap
        ),
    )

    # Change the tyre set after the in-lap is completed.
    # Until the penalty is served, mechanics must not
    # touch the tyres; the selected set remains
    # scheduled for the next pit stop.
    if (
        is_in_lap
        and car.pending_set_id
        and not penalty_outstanding
    ):

        found_set = None
        found_comp = None

        for compound, sets in (
            car.tyre_sets_inventory.items()
        ):
            for tyre_set in sets:
                if (
                    tyre_set["id"]
                    == car.pending_set_id
                ):
                    found_set = tyre_set
                    found_comp = compound
                    break

            if found_set is not None:
                break

        if (
            found_set is not None
            and found_comp is not None
        ):
            car.tyre_sets_inventory[
                found_comp
            ].remove(found_set)

            car.current_tyre = TyrePhysics(
                composition=str(found_comp),
                health=float(found_set["health"]),
                age=int(found_set["age"]),
                base_lap_time=float(
                    car.track.base_lap_time
                ),
                tyre_stress=float(
                    car.track.tyre_stress
                ),
                weather=weather,
                dirty_air=car.dirty_air,
                wing_damage=car.wing_damage,
                track_temperature=float(
                    car.track_temperature
                ),
                PIT_OUT_LAP=True,
            )

            car.pit_stops.append(
                {
                    "lap": int(car.lap),
                    "composition": str(found_comp),
                    "set_id": str(found_set["id"]),
                    "time": float(
                        car.track.pit_loss
                    ),
                    "weather": str(weather),
                    "wing_repair": bool(
                        wing_repair_this_lap
                    ),
                    "penalty_served": bool(
                        penalty_served_this_lap
                    ),
                }
            )

            messages.append(
                f"Pit stop completed: fitted "
                f"{found_comp}."
            )

            if pit_under_safety_car:
                messages.append(
                    "Strategic pit stop executed under Safety Car "
                    "conditions! Net pit loss reduced by 50%."
                )

        else:
            messages.append(
                "Scheduled tyre set "
                "was not found in the allocation inventory."
            )

        car.pending_pit = False
        car.pending_set_id = None

    car.lap += 1

    if car.current_tyre.is_cliff():
        messages.append(
            "Tyre Cliff: "
            f"{car.current_tyre.health:.1f}%."
        )

    return " ".join(messages)


# =============================================================================
# STREAMLIT STATE
# =============================================================================

def _make_inventory() -> Dict[
    str,
    List[Dict[str, Any]],
]:
    inventory: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for compound in ALL_COMPOUNDS:
        inventory[compound] = [
            {
                "id": f"{compound}_{index}",
                "health": 100.0,
                "age": 0,
            }
            for index in range(
                DEFAULT_INVENTORY_COUNTS[
                    compound
                ]
            )
        ]

    return inventory


def _init_state() -> None:
    defaults = {
        "initialized": True,
        "car": None,
        "last_message": "",
        "track_temperature": 35.0,
        "time_penalty": 0.0,
        "wing_damage": "None",
        "force_wing_next_pit": False,
        "tyre_sets_inventory": _make_inventory(),
        "weather_start_seconds": 999999.0,
        "weather_duration_laps": 0,
        "weather_intensity": "Wet",
        "baseline_initial_time": None,
        "baseline_initial_strategy": None,
        "dirty_air": False,
        "safety_car_active": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# =============================================================================
# CAR INITIALIZATION
# =============================================================================

def _build_car(
    track_name: str,
    start_set_id: str,
    tyre_sets_inventory: Dict[str, List[Dict[str, Any]]],
    weather_radar: WeatherRadar,
    track_temperature: float,
    time_penalty: float,
    dirty_air: bool = False,
) -> Optional[CarState]:
    """Build the live car from the exact tyre set selected in the allocation inventory."""

    track = TRACKS_DATABASE[track_name]
    inventory = tyre_sets_inventory

    starting_set: Optional[Dict[str, Any]] = None
    start_comp: Optional[str] = None

    for compound, sets in inventory.items():
        for tyre_set in sets:
            if str(tyre_set["id"]) == str(start_set_id):
                starting_set = tyre_set
                start_comp = str(compound)
                break
        if starting_set is not None:
            break

    if starting_set is None or start_comp is None:
        return None

    inventory[start_comp].remove(starting_set)

    health = float(starting_set["health"])
    age = int(starting_set["age"])

    first_weather = weather_radar.weather_for_lap(
        lap=1,
        race_elapsed_seconds=0.0,
        base_lap_time=float(track.base_lap_time),
    )

    tyre = TyrePhysics(
        composition=start_comp,
        health=health,
        age=age,
        base_lap_time=float(track.base_lap_time),
        tyre_stress=float(track.tyre_stress),
        weather=first_weather,
        dirty_air=bool(dirty_air),
        track_temperature=float(track_temperature),
    )

    return CarState(
        track=track,
        current_tyre=tyre,
        fuel=110.0,
        tyre_sets_inventory=inventory,
        weather_radar=weather_radar,
        track_temperature=float(track_temperature),
        time_penalty=float(time_penalty),
        dirty_air=bool(dirty_air),
    )


def _build_preview_car(
    track_name: str,
    weather_radar: WeatherRadar,
    track_temperature: float,
    tyre_sets_inventory: Dict[
        str,
        List[Dict[str, Any]],
    ],
    dirty_air: bool = False,
) -> CarState:
    """
    Builds a temporary car object for the pre-race
    briefing. It uses an inventory copy so no real
    tyre set is consumed before the race
    starts.
    """

    track = TRACKS_DATABASE[track_name]

    inventory_copy = copy.deepcopy(
        tyre_sets_inventory
    )

    dummy_tyre = TyrePhysics(
        composition="Medium",
        health=100.0,
        age=0,
        base_lap_time=track.base_lap_time,
        tyre_stress=track.tyre_stress,
        weather="Dry",
        dirty_air=bool(dirty_air),
        track_temperature=track_temperature,
    )

    return CarState(
        track=track,
        current_tyre=dummy_tyre,
        fuel=110.0,
        tyre_sets_inventory=inventory_copy,
        weather_radar=weather_radar,
        track_temperature=track_temperature,
        dirty_air=bool(dirty_air),
    )


def _first_rain_lap(
    weather_radar: WeatherRadar,
    track: TrackSpec,
) -> Tuple[Optional[int], Optional[str]]:

    timeline = weather_radar.timeline(
        track.total_laps,
        track.base_lap_time,
    )

    for lap, weather in enumerate(
        timeline,
        start=1,
    ):
        if weather != "Dry":
            return lap, weather

    return None, None


def _render_prerace_briefing(
    config: Dict[str, Any],
) -> None:
    """
    Pre-race strategic engineering briefing module.

    Runs on the opening screen before the
    race is started and executes a complete strategy
    evaluation before presenting the
    engineer's recommendation.
    """

    st.markdown(
        "### Pre-Race Strategic Engineering Briefing"
    )

    preview_car = _build_preview_car(
        track_name=config["track_name"],
        weather_radar=config["weather_radar"],
        track_temperature=config[
            "track_temperature"
        ],
        tyre_sets_inventory=(
            st.session_state.tyre_sets_inventory
        ),
        dirty_air=bool(config.get("dirty_air", False)),
    )

    optimizer = StrategyOptimizer(
        car=preview_car,
        lap_step=3,
    )

    strategies = optimizer.optimize()

    if not strategies:
        st.warning(
            "Insufficient tyre sets are available to calculate the engineering briefing."
        )
        return

    best = strategies[0]

    chain = Dashboard.format_strategy_chain(
        best,
        preview_car.track.total_laps,
    )

    lines = [
        f"Recommended opening strategy: {chain}.",
        "Predicted Race Time: "
        f"{format_race_time(float(best['total_time']))}.",
    ]

    rain_lap, rain_weather = _first_rain_lap(
        config["weather_radar"],
        preview_car.track,
    )

    if rain_lap is not None:
        lines.append(
            "The radar predicts "
            f"{rain_weather} conditions starting from Lap {rain_lap}."
        )

    baseline = optimizer._simulate_strategy(
        start_comp=config["start_comp"],
        plan=[],
    )

    if baseline is not None:

        baseline_time = float(
            baseline["total_time"]
        )
        best_time = float(best["total_time"])
        delta = baseline_time - best_time

        selected_matches_best = (
            best["start_compound"] == config["start_comp"]
            and not best.get("plan")
        )

        if delta > 0.05 and not selected_matches_best:
            lines.append(
                "WARNING: the selected opening compound "
                f"{config['start_comp']} with no pit stop under the configured track conditions profile loses {delta:.1f} seconds versus the recommended strategy."
            )

    st.info(" ".join(lines))


# =============================================================================
# UI
# =============================================================================

class Dashboard:

    @staticmethod
    def render_header(car: CarState) -> None:
        st.title("ApexStrategy Engine")

        weather = car.weather_for_current_lap()

        cols = st.columns(5)

        cols[0].metric(
            "Lap",
            f"{car.lap} / {car.track.total_laps}",
        )

        cols[1].metric(
            "Race Time",
            format_race_time(car.total_time),
        )

        cols[2].metric(
            "Fuel Mass (kg)",
            f"{car.fuel:.1f} kg",
        )

        cols[3].metric(
            "Track Conditions",
            weather,
        )

        cols[4].metric(
            "Track Temp",
            f"{car.track_temperature:.0f} °C",
        )

        st.markdown(
            "### Tyre Allocation Inventory"
        )

        cols = st.columns(5)

        for index, compound in enumerate(
            ALL_COMPOUNDS
        ):
            cols[index].metric(
                compound,
                f"{len(car.tyre_sets_inventory.get(compound, []))}",
            )

    @staticmethod
    def render_current_tyre(
        car: CarState,
    ) -> None:

        tyre = car.current_tyre
        color = PIRELLI_COLORS[
            tyre.composition
        ]

        weather = car.weather_for_current_lap()

        st.markdown(
            "### Active Car Allocation Profile"
        )

        st.markdown(
            f"""
            <div style="
                padding:18px;
                border-radius:8px;
                background:#FFFFFF;
                color:#000000;
                border:2px solid {color};
                border-left:10px solid {color};
            ">
                <div style="color:#000000;">
                    <b>Compound:</b> {tyre.composition}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Tyre Life (%):</b>
                    {float(tyre.health):.1f}%
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Tyre Age (Laps):</b>
                    {int(tyre.age)} Laps
                    <br><br>
                    <b>Track Conditions:</b>
                    {weather}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Front Wing Damage:</b>
                    {car.wing_damage}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Driver Sporting Time Penalty:</b>
                    {car.time_penalty:.1f} sec
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_visualization(
        car: CarState,
    ) -> None:

        if not car.laps_history:
            st.info(
                "Run at least one lap to generate telemetry."
            )
            return

        # Strict datatype casting prevents categorical axis duplication.
        laps = [
            int(item["lap"])
            for item in car.laps_history
        ]

        times = [
            float(item["time"])
            for item in car.laps_history
        ]

        healths = [
            float(item["health_after"])
            for item in car.laps_history
        ]

        comps = [
            str(item["composition"])
            for item in car.laps_history
        ]

        weather = [
            str(item["weather"])
            for item in car.laps_history
        ]

        colors = [
            PIRELLI_COLORS[compound]
            for compound in comps
        ]

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "Lap Time",
                "Tyre Life",
                "Track Conditions Radar",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=laps,
                y=times,
                mode="lines+markers",
                line=dict(
                    color="#808080",
                    width=3,
                ),
                marker=dict(
                    color=colors,
                    size=9,
                    line=dict(
                        color="#000000",
                        width=1,
                    ),
                ),
                customdata=list(
                    zip(
                        comps,
                        weather,
                        healths,
                    )
                ),
                hovertemplate=(
                    "Lap %{x}<br>"
                    "Race Time: %{y:.3f} sec<br>"
                    "Compound: %{customdata.0}<br>"
                    "Track Conditions: %{customdata.1}<br>"
                    "Tyre Life (%): %{customdata.2:.1f}%"
                    "<extra></extra>"
                ),
                name="Lap Time",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=laps,
                y=healths,
                mode="lines+markers",
                line=dict(
                    color="#808080",
                    width=3,
                ),
                marker=dict(
                    color=colors,
                    size=9,
                    line=dict(
                        color="#000000",
                        width=1,
                    ),
                ),
                hovertemplate=(
                    "Lap %{x}<br>"
                    "Tyre Life (%): %{y:.1f}%"
                    "<extra></extra>"
                ),
                name="Tyre Life",
            ),
            row=2,
            col=1,
        )

        weather_y = {
            "Dry": 0,
            "Damp": 1,
            "Wet": 2,
        }

        fig.add_trace(
            go.Scatter(
                x=laps,
                y=[
                    weather_y[item]
                    for item in weather
                ],
                mode="lines+markers",
                marker=dict(size=7),
                hovertemplate=(
                    "Lap %{x}<br>"
                    "Track Conditions Index: %{y}"
                    "<extra></extra>"
                ),
                name="Track Conditions",
            ),
            row=3,
            col=1,
        )

        fig.update_xaxes(
            type="linear",
            tickmode="linear",
            dtick=1,
            row=3,
            col=1,
        )

        fig.update_yaxes(
            type="linear",
            tickformat=".2f",
            row=1,
            col=1,
        )

        fig.update_yaxes(
            range=[0.0, 105.0],
            tickformat=".1f",
            row=2,
            col=1,
        )

        fig.update_yaxes(
            tickmode="array",
            tickvals=[0, 1, 2],
            ticktext=[
                "Dry",
                "Damp",
                "Wet",
            ],
            range=[-0.3, 2.3],
            row=3,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=820,
            showlegend=False,
            margin=dict(
                l=60,
                r=30,
                t=70,
                b=50,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        rows = []

        for item in car.laps_history[-10:]:
            rows.append(
                {
                    "Lap": int(item["lap"]),
                    "Lap Time": format_lap_time(
                        float(item["time"])
                    ),
                    "Compound": str(
                        item["composition"]
                    ),
                    "Tyre Life (%)": (
                        f"{float(item['health_after']):.1f}%"
                    ),
                    "Track Conditions": str(
                        item["weather"]
                    ),
                    "In-Lap": bool(
                        item["is_in_lap"]
                    ),
                    "Repair": bool(
                        item["wing_repair"]
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    @staticmethod
    def format_strategy_chain(
        strategy: Dict[str, Any],
        total_laps: int,
    ) -> str:
        start = str(strategy["start_compound"])
        plan = [
            (int(lap), str(compound))
            for lap, compound in strategy.get("plan", [])
        ]

        if not plan:
            return (
                f"Start: {start} | No Pit Stop | "
                f"Finish: Lap {int(total_laps)}"
            )

        stops = " | ".join(
            f"Pit {lap}: {compound}"
            for lap, compound in plan
        )
        return (
            f"Start: {start} | {stops} | "
            f"Finish: Lap {int(total_laps)}"
        )

    @staticmethod
    def render_strategy_graph(
        lap_history: List[Dict[str, Any]],
        key: str,
    ) -> None:
        if not lap_history:
            return

        laps = [int(item["lap"]) for item in lap_history]
        times = [float(item["time"]) for item in lap_history]
        healths = [float(item["health_after"]) for item in lap_history]
        comps = [str(item["composition"]) for item in lap_history]

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=("Predicted Lap Time", "Predicted Tyre Life"),
        )

        # One trace per contiguous stint keeps each Pirelli colour continuous.
        start_index = 0
        while start_index < len(laps):
            compound = comps[start_index]
            end_index = start_index + 1
            while end_index < len(laps) and comps[end_index] == compound:
                end_index += 1

            segment_laps = laps[start_index:end_index]
            segment_times = times[start_index:end_index]
            segment_healths = healths[start_index:end_index]
            colour = PIRELLI_COLORS[compound]

            fig.add_trace(
                go.Scatter(
                    x=[int(value) for value in segment_laps],
                    y=[float(value) for value in segment_times],
                    mode="lines+markers",
                    line=dict(color=colour, width=2.5),
                    marker=dict(color=colour, size=7),
                    text=[str(compound)] * len(segment_laps),
                    hovertemplate=(
                        "Lap %{x}<br>Race Time: %{y:.3f} sec<br>"
                        "Compound: %{text}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=[int(value) for value in segment_laps],
                    y=[float(value) for value in segment_healths],
                    mode="lines+markers",
                    line=dict(color=colour, width=2.5),
                    marker=dict(color=colour, size=7),
                    text=[str(compound)] * len(segment_laps),
                    hovertemplate=(
                        "Lap %{x}<br>Tyre Life: %{y:.1f}%<br>"
                        "Compound: %{text}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            start_index = end_index

        fig.update_xaxes(
            type="linear",
            tickmode="linear",
            dtick=1,
            row=2,
            col=1,
        )
        fig.update_yaxes(type="linear", tickformat=".2f", row=1, col=1)
        fig.update_yaxes(
            type="linear",
            range=[0.0, 105.0],
            tickformat=".1f",
            row=2,
            col=1,
        )
        fig.update_layout(
            template="plotly_dark",
            height=480,
            showlegend=False,
            margin=dict(l=60, r=30, t=70, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key=key)

    @staticmethod
    def render_strategy_cards(car: CarState) -> None:
        st.markdown("### Global Strategy Brute-Force: TOP-5 Permutations")

        strategies = StrategyOptimizer(car=car, lap_step=7).optimize()

        if not strategies:
            st.warning("No legal strategies were found.")
            return

        baseline = st.session_state.get("baseline_initial_time")

        # The optimizer projects only the remaining race from the current lap.
        # Add the already completed race time so the displayed Total Time is
        # always the projected time for the entire race.
        best_time = float(car.total_time) + float(strategies[0]["total_time"])

        for index, strategy in enumerate(strategies, start=1):
            remaining_time = float(strategy["total_time"])
            total_time = float(car.total_time) + remaining_time
            delta_to_best = total_time - best_time
            chain = Dashboard.format_strategy_chain(
                strategy,
                car.track.total_laps,
            )

            # Live progression-aware delta: the optimizer's "total_time"
            # field is the remaining projected time from the current lap
            # onward, not the full race total. To compare cleanly against
            # the Lap 0 baseline (a full-race projection) the elapsed race
            # time already banked in car.total_time must be added back in
            # before differencing against the baseline.
            if baseline is None:
                evolution_text = "Baseline pending"
                evolution_color = "#FFFFFF"
            else:
                evolution = total_time - float(baseline)

                if evolution < -0.0005:
                    evolution_text = (
                        f"Faster by {evolution:.3f} sec "
                        "versus opening baseline"
                    )
                    evolution_color = PIRELLI_COLORS["Intermediate"]
                elif evolution > 0.0005:
                    evolution_text = (
                        f"Pace loss delta: +{evolution:.3f} sec "
                        "slower versus initial forecast"
                    )
                    evolution_color = PIRELLI_COLORS["Soft"]
                else:
                    evolution_text = "On baseline"
                    evolution_color = "#FFFFFF"

            st.markdown(
                f"""
                <div style="
                    padding:16px;
                    margin-bottom:10px;
                    border:1px solid #555;
                    border-radius:7px;
                ">
                    <b>Strategy #{index}</b><br>
                    <b>Total Time:</b> {format_race_time(total_time)}
                    &nbsp; | &nbsp;
                    <b>Delta to Current Best:</b> +{delta_to_best:.3f} sec
                    <br><br>
                    <div style="
                        font-size:1.15rem;
                        font-weight:800;
                        color:{evolution_color};
                        background:#1A1A1A;
                        padding:10px;
                        border-left:5px solid {evolution_color};
                    ">
                        Evolution versus Initial Baseline: {evolution_text}
                    </div>
                    <br>
                    <b>Strategy Chain:</b> {chain}
                </div>
                """,
                unsafe_allow_html=True,
            )

        initial_strategy = st.session_state.get(
            "baseline_initial_strategy"
        )
        if initial_strategy is not None:
            initial_chain = Dashboard.format_strategy_chain(
                initial_strategy,
                car.track.total_laps,
            )
            initial_time = float(
                initial_strategy["total_time"]
            )
            st.markdown(
                f"""
                <div style="
                    padding:16px;
                    margin-top:10px;
                    margin-bottom:10px;
                    border:2px solid #FFFFFF;
                    border-radius:7px;
                ">
                    <b>Initial Algorithm Recommendation</b><br>
                    <b>Predicted Race Time:</b>
                    {format_race_time(initial_time)}
                    <br><br>
                    <b>Strategy Chain:</b> {initial_chain}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Predictive Telemetry Strategy Charts")
        for index, strategy in enumerate(strategies, start=1):
            with st.expander(
                f"Strategy #{index}",
                expanded=index == 1,
            ):
                Dashboard.render_strategy_graph(
                    strategy["lap_history"],
                    key=f"strategy_graph_{index}",
                )


# =============================================================================
# SIDEBAR
# =============================================================================

def _render_sidebar() -> Dict[str, Any]:
    st.sidebar.header("Session Configuration")

    track_name = st.sidebar.selectbox(
        "Circuit",
        list(TRACKS_DATABASE.keys()),
    )

    track_temperature = float(
        st.sidebar.slider(
            "Track Temp (°C)",
            min_value=10,
            max_value=60,
            value=int(st.session_state.track_temperature),
        )
    )
    st.session_state.track_temperature = track_temperature

    st.sidebar.markdown("---")
    st.sidebar.subheader("Track Conditions Forecast")

    start_seconds = float(
        st.sidebar.number_input(
            "Rain Onset After Start (sec)",
            min_value=0.0,
            value=float(st.session_state.weather_start_seconds),
            step=30.0,
        )
    )
    duration_laps = int(
        st.sidebar.number_input(
            "Rain Duration (Laps)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.weather_duration_laps),
            step=1,
        )
    )
    intensity = st.sidebar.selectbox(
        "Rain Intensity",
        ["Damp", "Wet"],
        index=0 if st.session_state.weather_intensity == "Damp" else 1,
    )

    st.session_state.weather_start_seconds = start_seconds
    st.session_state.weather_duration_laps = duration_laps
    st.session_state.weather_intensity = intensity

    radar = WeatherRadar(
        rain_start_seconds=start_seconds,
        rain_duration_laps=duration_laps,
        rain_intensity=intensity,
        drying_laps=11,
    )

    track = TRACKS_DATABASE[track_name]
    preview = radar.timeline(track.total_laps, track.base_lap_time)
    weather_counts = {mode: preview.count(mode) for mode in WEATHER_MODES}
    st.sidebar.caption(
        "Track Conditions Radar: "
        f"Dry {weather_counts['Dry']} | "
        f"Damp {weather_counts['Damp']} | "
        f"Wet {weather_counts['Wet']}"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Aerodynamic Conditions")

    dirty_air = bool(
        st.sidebar.checkbox(
            "Dirty Air (Running in Turbulent Wake)",
            value=bool(st.session_state.get("dirty_air", False)),
            key="dirty_air_checkbox",
        )
    )
    st.session_state.dirty_air = dirty_air

    st.sidebar.markdown("---")
    st.sidebar.subheader("Sporting Penalty")

    penalty = float(
        st.sidebar.number_input(
            "Driver Sporting Time Penalty (sec)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.time_penalty),
            step=0.5,
            key="sport_penalty_input",
        )
    )
    st.session_state.time_penalty = penalty

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tyre Allocation Inventory")

    for compound in ALL_COMPOUNDS:
        current_sets = st.session_state.tyre_sets_inventory.get(compound, [])

        desired_count = int(
            st.sidebar.number_input(
                f"{compound} Allocation Count",
                min_value=0,
                max_value=20,
                value=len(current_sets),
                step=1,
                key=f"inventory_count_{compound}",
            )
        )

        while len(st.session_state.tyre_sets_inventory[compound]) < desired_count:
            index = len(st.session_state.tyre_sets_inventory[compound])
            st.session_state.tyre_sets_inventory[compound].append(
                {
                    "id": f"{compound}_{index}_new",
                    "health": 100.0,
                    "age": 0,
                }
            )

        while len(st.session_state.tyre_sets_inventory[compound]) > desired_count:
            st.session_state.tyre_sets_inventory[compound].pop()

        with st.sidebar.expander(
            f"{compound}: {len(st.session_state.tyre_sets_inventory[compound])} Sets",
            expanded=False,
        ):
            for index, tyre_set in enumerate(
                st.session_state.tyre_sets_inventory[compound]
            ):
                col1, col2 = st.columns(2)
                health_key = f"{compound}_health_{index}"
                age_key = f"{compound}_age_{index}"
                prev_age_key = f"_prev_age_{tyre_set['id']}"

                new_age = int(
                    col2.number_input(
                        "Tyre Age (Laps)",
                        min_value=0,
                        max_value=100,
                        value=int(tyre_set["age"]),
                        step=1,
                        key=age_key,
                    )
                )

                prev_age = st.session_state.get(
                    prev_age_key,
                    int(tyre_set["age"]),
                )

                if new_age != prev_age:
                    simulated_health = simulate_inventory_pre_wear(
                        compound=compound,
                        age_laps=new_age,
                        tyre_stress=float(track.tyre_stress),
                        track_temperature=float(track_temperature),
                    )
                    tyre_set["health"] = simulated_health
                    st.session_state[health_key] = simulated_health

                st.session_state[prev_age_key] = new_age
                tyre_set["age"] = new_age

                tyre_set["health"] = float(
                    col1.number_input(
                        "Tyre Life (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(tyre_set["health"]),
                        step=1.0,
                        key=health_key,
                    )
                )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Starting Tyre Selection")

    starting_options: List[Tuple[str, str, str]] = []
    for compound, sets in st.session_state.tyre_sets_inventory.items():
        for tyre_set in sets:
            label = (
                f"{tyre_set['id']} | {compound} | "
                f"Tyre Life {float(tyre_set['health']):.1f}% | "
                f"Tyre Age {int(tyre_set['age'])} Laps"
            )
            starting_options.append(
                (label, str(tyre_set["id"]), str(compound))
            )

    if starting_options:
        selected_start_label = st.sidebar.selectbox(
            "Starting Tyre Set",
            [item[0] for item in starting_options],
        )
        selected_start = next(
            item for item in starting_options
            if item[0] == selected_start_label
        )
        start_set_id = selected_start[1]
        start_comp = selected_start[2]
    else:
        st.sidebar.error("No tyre sets are available for session start.")
        start_set_id = ""
        start_comp = ""

    return {
        "track_name": track_name,
        "start_comp": start_comp,
        "start_set_id": start_set_id,
        "weather_radar": radar,
        "track_temperature": track_temperature,
        "time_penalty": penalty,
        "dirty_air": dirty_air,
    }




# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    st.set_page_config(
        page_title="ApexStrategy Engine",
        layout="wide",
    )

    _init_state()

    config = _render_sidebar()

    if st.session_state.car is None:

        st.title("ApexStrategy Engine")

        st.markdown(
            """
            Dynamic race strategy command platform.
            Track conditions are calculated lap by lap,
            and the optimizer evaluates strategy permutations
            from the opening lap of the race.
            """
        )

        _render_prerace_briefing(config)

        if st.button(
            "Start New Race",
            type="primary",
            use_container_width=True,
        ):

            car = _build_car(
                track_name=config["track_name"],
                start_set_id=config["start_set_id"],
                tyre_sets_inventory=
                    st.session_state.tyre_sets_inventory,
                weather_radar=
                    config["weather_radar"],
                track_temperature=
                    config["track_temperature"],
                time_penalty=
                    config["time_penalty"],
                dirty_air=
                    bool(config.get("dirty_air", False)),
            )

            if car is None:
                st.error(
                    "The selected starting tyre set is no longer available."
                )
                return

            # Capture the immutable Lap 0 strategy benchmark before any
            # live lap is simulated.
            car.race_started = True
            baseline_optimizer = StrategyOptimizer(car=car, lap_step=3)
            baseline_strategies = baseline_optimizer.optimize()
            st.session_state.baseline_initial_time = (
                float(baseline_strategies[0]["total_time"])
                if baseline_strategies
                else None
            )
            st.session_state.baseline_initial_strategy = (
                copy.deepcopy(baseline_strategies[0])
                if baseline_strategies
                else None
            )
            car.race_started = True
            st.session_state.car = car

            _safe_rerun()

        return

    car: CarState = st.session_state.car

    # After race start, update only dynamic
    # environmental parameters and the radar.
    car.weather_radar = (
        config["weather_radar"]
    )
    car.track_temperature = (
        config["track_temperature"]
    )
    car.dirty_air = bool(
        config.get("dirty_air", False)
    )

    car.safety_car_active = bool(
        st.session_state.get("safety_car_active", False)
    )

    if not car.penalty_served:
        car.time_penalty = float(
            st.session_state.time_penalty
        )

    Dashboard.render_header(car)

    st.markdown("---")

    Dashboard.render_current_tyre(car)

    st.markdown("---")

    st.markdown(
        "### Front Wing Status"
    )

    wing_damage = st.selectbox(
        "Front Wing Damage",
        ["None", "Minor", "Critical"],
        index=[
            "None",
            "Minor",
            "Critical",
        ].index(car.wing_damage),
        key="wing_damage_select",
    )

    car.wing_damage = wing_damage
    st.session_state.wing_damage = (
        wing_damage
    )

    force_wing_next_pit = st.checkbox(
        "Schedule Front Wing Replacement at Next Pit Stop",
        value=bool(
            st.session_state.get(
                "force_wing_next_pit",
                False,
            )
        ),
        key="force_wing_next_pit_checkbox",
    )

    car.force_wing_replacement_next_pit = (
        force_wing_next_pit
    )
    st.session_state.force_wing_next_pit = (
        force_wing_next_pit
    )

    st.markdown(
        "### Pit Wall Command Center"
    )

    safety_car_active = st.checkbox(
        "SAFETY CAR / VSC PERIOD ACTIVE",
        value=bool(
            st.session_state.get(
                "safety_car_active",
                False,
            )
        ),
        key="safety_car_toggle",
        help=(
            "While active, every lap on track is 40% slower and any "
            "pit stop taken this lap has its net pit loss cut in half."
        ),
    )

    st.session_state.safety_car_active = safety_car_active
    car.safety_car_active = safety_car_active

    if safety_car_active:
        st.warning(
            "Safety Car / VSC period is ACTIVE: lap times are down "
            "40% and the current pit loss is halved."
        )

    st.markdown(
        "### Pit Stop Planning"
    )

    available_sets: List[
        Tuple[str, str]
    ] = []

    for compound, sets in (
        car.tyre_sets_inventory.items()
    ):
        for tyre_set in sets:
            available_sets.append(
                (
                    f"{compound} | "
                    f"{tyre_set['health']:.1f}% | "
                    f"{tyre_set['age']} Laps",
                    str(tyre_set["id"]),
                )
            )

    selected_set_id = None

    if available_sets:
        selected_label = st.selectbox(
            "Select Tyre Set for Next Box",
            [
                item[0]
                for item in available_sets
            ],
        )

        selected_set_id = next(
            item[1]
            for item in available_sets
            if item[0] == selected_label
        )

    control_cols = st.columns(4)

    with control_cols[0]:

        if st.button(
            "BOX, BOX",
            use_container_width=True,
        ):
            if selected_set_id is None:
                st.warning(
                    "No tyre set is available."
                )
            else:
                car.pending_pit = True
                car.pending_set_id = (
                    selected_set_id
                )

                st.session_state.last_message = (
                    f"Pit stop scheduled for Lap {car.lap}."
                )

    with control_cols[1]:

        if st.button(
            "Run 1 Lap",
            use_container_width=True,
        ):
            st.session_state.last_message = (
                _drive_one_lap(car)
            )

    with control_cols[2]:

        if st.button(
            "Run 5 Laps",
            use_container_width=True,
        ):
            messages = []

            for _ in range(5):

                if car.lap > car.track.total_laps:
                    break

                message = _drive_one_lap(car)

                if message:
                    messages.append(message)

            st.session_state.last_message = (
                " ".join(messages)
            )

    with control_cols[3]:

        if st.button(
            "Reset Session",
            use_container_width=True,
        ):
            st.session_state.car = None
            st.session_state.last_message = ""
            st.session_state.baseline_initial_time = None
            st.session_state.tyre_sets_inventory = (
                _make_inventory()
            )
            _safe_rerun()

    if st.session_state.last_message:
        st.info(
            st.session_state.last_message
        )

    progress = min(
        1.0,
        float(car.lap - 1)
        / max(
            1,
            car.track.total_laps,
        ),
    )

    st.progress(progress)

    st.caption(
        f"Progress: {car.lap - 1} / "
        f"{car.track.total_laps} Laps | "
        f"Race Time: "
        f"{format_race_time(car.total_time)}"
    )

    st.markdown("---")

    Dashboard.render_visualization(car)

    st.markdown("---")

    Dashboard.render_strategy_cards(car)


if __name__ == "__main__":
    main()