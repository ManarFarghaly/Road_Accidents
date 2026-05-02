
from __future__ import annotations

# from pyspark.ml.pipeline import PipelineStage 
from pyspark.ml.feature import OneHotEncoder, StringIndexer , FeatureHasher
# For LR: hash trick instead of raw integer (keeps dims bounded)

# ── Low-cardinality (< 15 levels) → one-hot ───────────────────────────────
# Chosen threshold: 15 levels. At 15 dummies per column × 12 columns we add
# ~180 sparse-vector slots, which LR handles happily and GBT tolerates.
# Anything ≥ 15 levels (Vehicle_Type ≈ 21, Age_Band_of_Driver borderline)
# goes to HIGH_CARD_CATS as an integer index instead.
# Low cardinality , nominal order 
# OHE for LR , integer index for GBT (trees handle categoricals natively, LR loses a little expressiveness but saves thousands of sparse dimensions)
LOW_CARD_CATS = [
    "Day_of_Week",                               # 7   
    "Road_Type",                                 # 6
    "Weather_Conditions",                        # 9
    "Light_Conditions",                          # 5
    "Road_Surface_Conditions",                   # 5
    "Urban_or_Rural_Area",                       # 3
    "Junction_Detail",                           # 9
    "Junction_Control",                          # 5
    "Sex_of_Driver",                             # 4 incl. "Not known"
    "Pedestrian_Crossing-Human_Control",         # 4
    "Pedestrian_Crossing-Physical_Facilities",   # 6
    "Driver_Home_Area_Type",                   # 4
    "Journey_Purpose_of_Driver",                # 8
    "Junction_Location",                        # 10
    "Propulsion_Code",             # 13
    "Towing_and_Articulation", #7,
    "Vehicle_Leaving_Carriageway", # 10,
    "X1st_Point_of_Impact"  # 6
    ]

# Low cardinality, ordinal order
#Encoding: StringIndexer with stringOrderType="alphabetAsc".
# Integer Index for both LR and GBT (trees handle categoricals natively, LR loses a little expressiveness but saves thousands of sparse dimensions)
ORDINAL_CATS = ["Age_Band_of_Driver","1st_Road_Class"] 
HIGH_CARD_CATS = [
    "Vehicle_Type",                # ~21 levels
    "make",                        # ~60
    "model",                       # ~20 000
    "LSOA_of_Accident_Location",   # ~35 000
    "Local_Authority_(District)",  # ~400
    "Local_Authority_(Highway)",   # 207
    "Police_Force",                # ~50
    "Vehicle_Manoeuvre" # 19
]

LABEL_COL = "Accident_Severity"

def build_encoding_stages_trees():
    stages = []
    encoded_output_cols = []
    
    all_cats = LOW_CARD_CATS + HIGH_CARD_CATS + ORDINAL_CATS
    
    for c in all_cats:
        idx_col = f"{c}_idx"
        order = "alphabetAsc" if c in ORDINAL_CATS else "frequencyDesc"
        stages.append(StringIndexer(
            inputCol=c, outputCol=idx_col, handleInvalid="keep", stringOrderType=order
        ))
        encoded_output_cols.append(idx_col)

    # Label should NOT be in encoded_output_cols
    stages.append(StringIndexer(inputCol=LABEL_COL, outputCol="label", handleInvalid="keep"))
    
    return stages, encoded_output_cols


def build_encoding_stages_lr(num_features: int = 8192): 
    encoded_output_cols = []
    
    # 1. Nominal Low Card -> OneHot
    for c in LOW_CARD_CATS:
        idx_col = f"{c}_idx"
        ohe_col = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCol=idx_col, outputCol=ohe_col, handleInvalid="keep"))
        encoded_output_cols.append(ohe_col)

    # 2. Ordinal -> Integer Index (Keep order)
    for c in ORDINAL_CATS:
        idx_col = f"{c}_idx"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_col, stringOrderType="alphabetAsc", handleInvalid="keep"))
        encoded_output_cols.append(idx_col)

    # 3. High Card -> Hashing ONLY
    stages.append(FeatureHasher(
        inputCols=HIGH_CARD_CATS, 
        outputCol="hashed_high_card", 
        numFeatures=num_features
    ))
    encoded_output_cols.append("hashed_high_card")

    # Label indexer (Separate from feature cols)
    stages.append(StringIndexer(inputCol=LABEL_COL, outputCol="label", handleInvalid="keep"))
    
    return stages, encoded_output_cols