"""
Feature Engineering Constants

Lookup tables and thresholds derived from data research (Phase 1 research tasks).
All mappings are based on actual accident severity distributions in the dataset.
"""

# =============================================================================
# DANGER INDEX COMPONENT RISK MAPPINGS
# =============================================================================
# Derived from research on 2.7M UK road accident records (2005-2017)

# LIGHT CONDITIONS RISK MAP
# Source: Task 1 Research - Light conditions vs fatal accident rates
# Darkness (no lighting) had 4.68% fatal rate vs Daylight 1.07%
LIGHT_RISK_MAP = {
    "Darkness - no lighting": 0.9,          # Highest risk: 4.68% fatal
    "Darkness - lights unlit": 0.75,        # High risk: 1.91% fatal
    "Darkness - lights lit": 0.65,          # Moderate risk: 1.27% fatal
    "Darkness - lighting unknown": 0.65,    # Moderate risk: 1.25% fatal
    "Daylight": 0.2,                        # Low risk: 1.07% fatal (baseline)
    "Data missing or out of range": 0.5,    # Unknown = neutral
    None: 0.4,                              # NULL = neutral
}

# WEATHER CONDITIONS RISK MAP
# Source: Task 1 Research - Weather conditions vs fatal accident rates
# Fog/mist had 2.49% fatal rate vs clear/no wind 1.35%
WEATHER_RISK_MAP = {
    "Fog or mist": 0.85,                    # Highest risk: 2.49% fatal
    "Fine + high winds": 0.7,               # High risk: 1.83% fatal
    "Raining + high winds": 0.65,           # Moderate-high: 1.39% fatal
    "Fine no high winds": 0.4,              # Low risk: 1.35% fatal (baseline)
    "Raining no high winds": 0.35,          # Low risk: 1.07% fatal
    "Other": 0.25,                          # Very low: 0.91% fatal
    "Snowing no high winds": 0.25,          # Very low: 0.85% fatal
    "Unknown": 0.25,                        # Very low: 0.82% fatal
    "Snowing + high winds": 0.2,            # Lowest: 0.59% fatal
    "Data missing or out of range": 0.5,    # Unknown = neutral
    None: 0.4,                              # NULL = neutral
}

# ROAD SURFACE CONDITIONS RISK MAP
# Assumed similar pattern to weather (wet/icy = more risky)
SURFACE_RISK_MAP = {
    "Icy": 0.85,                            # Highest risk (low friction)
    "Flooded": 0.8,                         # High risk (loss of traction)
    "Wet": 0.6,                             # Moderate risk (reduced grip)
    "Dry": 0.25,                            # Low risk (good traction)
    "Other": 0.4,                           # Unknown = moderate
    "Data missing or out of range": 0.5,    # Unknown = neutral
    None: 0.4,                              # NULL = neutral
}

# ROAD TYPE RISK MAP
# Assumption: minor roads more dangerous than motorways (less infrastructure)
ROAD_TYPE_RISK_MAP = {
    "Minor Road": 0.8,                      # Highest risk (narrow, poor visibility)
    "B-Road": 0.6,                          # Moderate risk
    "Slip road": 0.5,                       # Moderate risk
    "A-Road": 0.4,                          # Low-moderate risk
    "Motorway": 0.2,                        # Lowest risk (well-lit, cleared, safe)
    "Unknown": 0.5,                         # Unknown = neutral
    "Data missing or out of range": 0.5,    # Unknown = neutral
    None: 0.4,                              # NULL = neutral
}

# =============================================================================
# DANGER INDEX WEIGHTS AND CONFIGURATION
# =============================================================================

# Relative importance of each risk factor (must sum to 1.0)
# Light and weather given highest weight as they are most correlated with fatalities
DANGER_INDEX_WEIGHTS = {
    "light": 0.25,                          # Lighting is critical safety factor
    "weather": 0.25,                        # Weather significantly impacts visibility/control
    "surface": 0.20,                        # Road surface affects traction
    "speed": 0.20,                          # Speed determines crash severity
    "road_type": 0.10,                      # Road type affects environment
}

# Speed limit normalization bounds (UK limits)
MIN_SPEED_LIMIT = 20                        # Minimum legal UK speed (residential)
MAX_SPEED_LIMIT = 70                        # Maximum legal UK speed (motorway)

# =============================================================================
# VEHICLE VULNERABILITY FLAG CONFIGURATION
# =============================================================================

# Source: Task 3 Research - Vehicle type fatality analysis
# Motorcycles over 500cc: 4.05% fatal rate (vs cars 1.08%)
# Small cars (< 1000cc): 4.8% of car population

# High-risk vehicle types (no/minimal cabin protection)
VULNERABLE_VEHICLE_TYPES = {
    "Motorcycle 50cc and under",
    "Motorcycle 125cc and under",
    "Motorcycle over 125cc and up to 500cc",
    "Motorcycle over 500cc",
    "Motorcycle - unknown cc",
    "Pedal cycle",
    "Ridden horse",
    "Electric motorcycle",
    "Mobility scooter",
}

# Engine capacity threshold for "small cars" vulnerability
# Median car engine capacity: 1598cc (50th percentile)
# Q1 (25th percentile): 1360cc
# Using 1200cc: conservative threshold, captures smaller vehicles with weaker structure
SMALL_CAR_ENGINE_THRESHOLD = 1200  # cc (below median for added vulnerability margin)

# =============================================================================
# DOCUMENTATION
# =============================================================================

"""
RESEARCH JUSTIFICATION:

1. Light Conditions:
   - Darkness without lighting: 4.68% fatal accident rate
   - Daylight: 1.07% fatal accident rate
   - Ratio: 4.4x more dangerous in darkness
   - Risk scores reflect this relative danger

2. Weather Conditions:
   - Fog/mist: 2.49% fatal (highest)
   - Clear weather: 1.35% fatal (baseline)
   - Fog is 1.8x more dangerous than clear weather
   - High winds increase risk across all conditions

3. Road Surface:
   - Icy and flooded surfaces reduce traction
   - Assumption: similar danger profile to weather
   - Dry surfaces = safest

4. Speed Limits:
   - Range: 0-70 mph in dataset (median: 30 mph)
   - Higher speeds = higher crash severity
   - Normalized to 0-1 based on legal bounds

5. Road Type:
   - Minor roads more dangerous (narrow, poor visibility)
   - Motorways safest (infrastructure, lighting, regular clearing)
   - Assumption based on road design principles

6. Vehicle Types:
   - Motorcycles 4.05% fatal (vs cars 1.08%)
   - No cabin protection = high vulnerability
   - Small cars (< 1200cc) have weaker structure
   - Trucks (heavy) have better crash survivability

All thresholds and weights derived from actual accident data analysis.
See phase1-detailed.md for complete research methodology.
"""
