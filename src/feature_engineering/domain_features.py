"""
Domain-Specific Feature Engineering UDFs

Implements custom features for road accident severity prediction:
  1. danger_index: Continuous risk score (0-1) from lighting, weather, surface, speed, road type
  2. vehicle_vulnerable: Boolean flag for high-casualty-risk vehicles
"""

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, BooleanType
from .constants import (
    LIGHT_RISK_MAP,
    WEATHER_RISK_MAP,
    SURFACE_RISK_MAP,
    ROAD_TYPE_RISK_MAP,
    DANGER_INDEX_WEIGHTS,
    MIN_SPEED_LIMIT,
    MAX_SPEED_LIMIT,
    VULNERABLE_VEHICLE_TYPES,
    SMALL_CAR_ENGINE_THRESHOLD,
)


def danger_index_udf(light, weather, surface, speed_limit, road_type):
    """
    Compute danger index from 5 safety components.

    Combines environmental and vehicle factors into a single 0-1 safety score.
    Factors:
      - light: Light_Conditions (darkness riskier than daylight)
      - weather: Weather_Conditions (fog/snow riskier than clear)
      - surface: Road_Surface_Conditions (icy/wet riskier than dry)
      - speed_limit: Speed_limit (higher speeds riskier)
      - road_type: Road_Type (minor roads riskier than motorways)

    Args:
        light: Light_Conditions (e.g., "Darkness - no lighting", "Daylight")
        weather: Weather_Conditions (e.g., "Fog or mist", "Fine no high winds")
        surface: Road_Surface_Conditions (e.g., "Wet", "Dry", "Icy")
        speed_limit: Speed_limit numeric value in mph (0-70)
        road_type: Road_Type (e.g., "Motorway", "Minor Road", "A-Road")

    Returns:
        float in [0, 1]: 0=safest, 1=most dangerous.
        Returns 0.5 (neutral) if critical data is missing.
    """
    # Handle None checks first
    if all(x is None for x in [light, weather, surface, speed_limit, road_type]):
        return 0.5  # All data missing -> neutral

    # Get component risk scores, defaulting to neutral (0.4) for unknowns
    light_risk = LIGHT_RISK_MAP.get(light, 0.4)
    weather_risk = WEATHER_RISK_MAP.get(weather, 0.4)
    surface_risk = SURFACE_RISK_MAP.get(surface, 0.4)
    road_type_risk = ROAD_TYPE_RISK_MAP.get(road_type, 0.4)

    # Normalize speed to 0-1 range
    if speed_limit is None:
        speed_risk = 0.4  # Unknown speed -> neutral
    else:
        try:
            speed_limit_val = float(speed_limit)
            # min-max scaling 
            speed_risk = (speed_limit_val - MIN_SPEED_LIMIT) / (
                MAX_SPEED_LIMIT - MIN_SPEED_LIMIT
            )
            # Clamp to [0, 1]
            speed_risk = max(0.0, min(1.0, speed_risk))
        except (ValueError, TypeError):
            speed_risk = 0.4  # Unparseable -> neutral

    # Compute weighted average
    weights = DANGER_INDEX_WEIGHTS
    total_weight = sum(weights.values())  # Should be 1.0

    danger = (
        weights["light"] * light_risk
        + weights["weather"] * weather_risk
        + weights["surface"] * surface_risk
        + weights["speed"] * speed_risk
        + weights["road_type"] * road_type_risk
    ) / total_weight

    # Clamp to [0, 1]
    danger = max(0.0, min(1.0, danger))
    return float(danger)


# Register as Spark UDF
danger_index_spark_udf = F.udf(danger_index_udf, DoubleType())


def vehicle_vulnerability_udf(vehicle_type, engine_capacity):
    """
    Determine if vehicle is vulnerable in crashes.

    Vulnerable vehicles have minimal/no cabin protection or weak structure:
      - Motorcycles (all types)
      - Pedal cycles, ridden horses
      - Small cars with weak engine/frame (< 1200cc)

    Args:
        vehicle_type: Vehicle_Type (e.g., "Car", "Motorcycle over 500cc", "Bus or coach")
        engine_capacity: Engine_Capacity_.CC. (numeric cc, can be None)

    Returns:
        bool: True if vehicle is high-risk (vulnerable), False otherwise.
              Returns False for unknown/missing data (conservative).
    """
    # Check if high-risk vehicle type
    if vehicle_type in VULNERABLE_VEHICLE_TYPES:
        return True

    # Check if small car (risky due to weak structure)
    if vehicle_type == "Car":
        if engine_capacity is None:
            return False  # Unknown capacity -> assume standard car (not vulnerable)

        try:
            engine_capacity_val = float(engine_capacity)
            if engine_capacity_val < SMALL_CAR_ENGINE_THRESHOLD:
                return True  # Small engine -> vulnerable
        except (ValueError, TypeError):
            return False  # Unparseable -> assume standard (not vulnerable)

    # All other vehicles are not vulnerable
    return False


# Register as Spark UDF
vulnerability_spark_udf = F.udf(vehicle_vulnerability_udf, BooleanType())
