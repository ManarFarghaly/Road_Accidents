from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from pyspark.sql import functions as F
from src.config import PROJECT_ROOT, get_spark
from src.models.save_load import load_model
from src.preprocessing.clean import clean


DEMO_DIR = PROJECT_ROOT / "demo"
DEMO_INPUT_PATH = DEMO_DIR / "accident_input1.json"
DEMO_OUTPUT_PATH = DEMO_DIR / "accident_prediction.json"
MODEL_PATH = PROJECT_ROOT / "models" / "gbt"


DEMO_RECORD_TEMPLATE: dict[str, Any] = {
	"Accident_Index": "DEMO-0001",
	"Accident_Severity": "Slight",
	"Date": "2017-06-21 08:30:00",
	"Time": "08:30",
	"1st_Road_Class": "A",
	"Road_Type": "Single carriageway",
	"Weather_Conditions": "Fine no high winds",
	"Light_Conditions": "Daylight",
	"Road_Surface_Conditions": "Dry",
	"Urban_or_Rural_Area": "Urban",
	"Junction_Detail": "T or staggered junction",
	"Junction_Control": "Give way or uncontrolled",
	"Sex_of_Driver": "Male",
	"Pedestrian_Crossing-Human_Control": "None within 50 metres",
	"Pedestrian_Crossing-Physical_Facilities": "No physical crossing within 50 metres",
	"Driver_Home_Area_Type": "Urban area",
	"Journey_Purpose_of_Driver": "Commuting to/from work",
	"Junction_Location": "Not at or within 20 metres of junction",
	"Propulsion_Code": "Petrol",
	"Towing_and_Articulation": "No tow/articulation",
	"Vehicle_Leaving_Carriageway": "Did not leave carriageway",
	"X1st_Point_of_Impact": "Front",
	"Age_Band_of_Driver": "26 - 35",
	"Vehicle_Type": "Car",
	"make": "Ford",
	"model": "Focus",
	"Local_Authority_(District)": "Demo District",
	"Local_Authority_(Highway)": "Demo Highway",
	"Police_Force": "Demo Force",
	"Vehicle_Manoeuvre": "Going ahead other",
	"LSOA_of_Accident_Location": "E01000001",
	"Speed_limit": 30.0,
	"Number_of_Vehicles": 2,
	"Latitude": 51.5074,
	"Longitude": -0.1278,
	"Age_of_Vehicle": 5.0,
	"Engine_Capacity_.CC.": 1600.0,
	"Driver_IMD_Decile": 5.0,
	"InScotland": "No",
	"Was_Vehicle_Left_Hand_Drive": "No",
	"temp": 18.0,
	"tmin": 12.0,
	"tmax": 20.0,
	"wspd": 9.0,
	"pres": 1015.0,
	"rhum": 62.0,
}


PREDICTION_LABELS = {
	0.0: "Slight",
	1.0: "Serious",
	2.0: "Fatal",
}


def ensure_demo_input(path: Path) -> None:
	if path.exists():
		return

	DEMO_DIR.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps([DEMO_RECORD_TEMPLATE], indent=2), encoding="utf-8")


def load_records(path: Path) -> list[dict[str, Any]]:
	payload = json.loads(path.read_text(encoding="utf-8"))
	if isinstance(payload, dict):
		return [payload]
	if isinstance(payload, list):
		return payload
	raise ValueError("Input JSON must be an object or a list of objects.")


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
	normalized: list[dict[str, Any]] = []
	for record in records:
		merged = dict(DEMO_RECORD_TEMPLATE)
		merged.update(record)
		normalized.append(merged)
	return normalized


def prepare_inference_frame(spark, records: list[dict[str, Any]]):
	df = spark.createDataFrame(normalize_records(records))

	if "Date" in df.columns:
		df = df.withColumn("Date", F.to_timestamp("Date"))
		if "Day_of_Week" not in df.columns:
			df = df.withColumn("Day_of_Week", F.date_format(F.col("Date"), "EEEE"))

	if "Accident_Severity" not in df.columns:
		df = df.withColumn(
			"Accident_Severity",
			F.lit(DEMO_RECORD_TEMPLATE["Accident_Severity"]),
		)

	return df


def add_neutral_target_encoding(df):
	demo_number = F.regexp_extract(F.col("Accident_Index"), r"(\d+)$", 1).cast("int")
	encoding_value = F.when(F.pmod(demo_number, F.lit(2)) == 0, F.lit(1.0)).otherwise(F.lit(0.0))

	for column_name in ("model_te", "LSOA_of_Accident_Location_te"):
		if column_name in df.columns:
			df = df.fillna({column_name: 0.0})
			df = df.withColumn(column_name, encoding_value)
		else:
			df = df.withColumn(column_name, encoding_value)
	return df


def collect_predictions(predictions_df):
	rows = []
	for row in predictions_df.collect():
		item = row.asDict(recursive=True)
		prediction_value = float(item.get("prediction", 0.0))
		item["predicted_case"] = PREDICTION_LABELS.get(
			prediction_value,
			str(prediction_value),
		)
		rows.append(item)
	return rows


def main() -> None:
	ensure_demo_input(DEMO_INPUT_PATH)
	DEMO_DIR.mkdir(parents=True, exist_ok=True)

	spark = get_spark("road-accidents-inference")
	spark.sparkContext.setLogLevel("WARN")

	model = load_model(MODEL_PATH)
	records = load_records(DEMO_INPUT_PATH)
	input_df = prepare_inference_frame(spark, records)
	cleaned_df = add_neutral_target_encoding(clean(input_df))

	predictions_df = model.transform(cleaned_df).select(
		"Accident_Index",
		"prediction",
	)
	output_rows = collect_predictions(predictions_df)

	DEMO_OUTPUT_PATH.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")

	for row in output_rows:
		print(
			f"Accident case for {row.get('Accident_Index', 'unknown')}: "
			f"{row.get('predicted_case', 'unknown')}"
		)

	spark.stop()


if __name__ == "__main__":
	main()
