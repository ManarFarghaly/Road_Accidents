"""
Unit Tests for Domain Features

Tests for danger_index and vehicle_vulnerable features:
  1. Output range verification
  2. Logic correctness
  3. NULL/missing value handling
  4. No data leakage
  5. End-to-end pipeline integration
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from src.config import get_spark, MERGED_PARQUET
from src.feature_engineering.domain_features import (
    danger_index_udf,
    vehicle_vulnerability_udf,
)
from src.feature_engineering import (
    DangerIndexTransformer,
    VehicleVulnerabilityTransformer,
)


# =============================================================================
# DANGER INDEX UDF TESTS
# =============================================================================


class TestDangerIndexUDF:
    """Tests for danger_index_udf function."""

    def test_output_range_high_danger(self):
        """Test: Maximum danger scenario returns close to 1.0."""
        result = danger_index_udf(
            light="Darkness - no lighting",
            weather="Fog or mist",
            surface="Icy",
            speed_limit=70,
            road_type="Minor Road",
        )
        assert isinstance(result, float), "Output must be float"
        assert 0.0 <= result <= 1.0, f"Danger index out of range: {result}"
        assert result > 0.6, f"High danger should be > 0.6, got {result}"
        print(f"✓ Max danger scenario: {result:.3f}")

    def test_output_range_low_danger(self):
        """Test: Minimum danger scenario returns close to 0.0."""
        result = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=20,
            road_type="Motorway",
        )
        assert isinstance(result, float), "Output must be float"
        assert 0.0 <= result <= 1.0, f"Danger index out of range: {result}"
        assert result < 0.4, f"Low danger should be < 0.4, got {result}"
        print(f"✓ Min danger scenario: {result:.3f}")

    def test_darkness_riskier_than_daylight(self):
        """Test: Darkness increases danger_index vs daylight (all else equal)."""
        # Same weather, surface, speed, road -> only light differs
        darkness = danger_index_udf(
            light="Darkness - no lighting",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=30,
            road_type="A-Road",
        )
        daylight = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=30,
            road_type="A-Road",
        )
        assert darkness > daylight, f"Darkness ({darkness:.3f}) should be > Daylight ({daylight:.3f})"
        print(f"✓ Darkness > Daylight: {darkness:.3f} > {daylight:.3f}")

    def test_fog_riskier_than_clear_weather(self):
        """Test: Fog increases danger_index vs clear weather (all else equal)."""
        fog = danger_index_udf(
            light="Daylight",
            weather="Fog or mist",
            surface="Dry",
            speed_limit=30,
            road_type="A-Road",
        )
        clear = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=30,
            road_type="A-Road",
        )
        assert fog > clear, f"Fog ({fog:.3f}) should be > Clear ({clear:.3f})"
        print(f"✓ Fog > Clear: {fog:.3f} > {clear:.3f}")

    def test_icy_riskier_than_dry(self):
        """Test: Icy road increases danger_index vs dry road (all else equal)."""
        icy = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Icy",
            speed_limit=30,
            road_type="A-Road",
        )
        dry = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=30,
            road_type="A-Road",
        )
        assert icy > dry, f"Icy ({icy:.3f}) should be > Dry ({dry:.3f})"
        print(f"✓ Icy > Dry: {icy:.3f} > {dry:.3f}")

    def test_higher_speed_riskier(self):
        """Test: Higher speed increases danger_index (all else equal)."""
        high_speed = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=70,
            road_type="A-Road",
        )
        low_speed = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=20,
            road_type="A-Road",
        )
        assert (
            high_speed > low_speed
        ), f"70 mph ({high_speed:.3f}) should be > 20 mph ({low_speed:.3f})"
        print(f"✓ 70 mph > 20 mph: {high_speed:.3f} > {low_speed:.3f}")

    def test_minor_road_riskier_than_motorway(self):
        """Test: Minor roads increase danger_index vs motorway (all else equal)."""
        minor = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=30,
            road_type="Minor Road",
        )
        motorway = danger_index_udf(
            light="Daylight",
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=30,
            road_type="Motorway",
        )
        assert (
            minor > motorway
        ), f"Minor ({minor:.3f}) should be > Motorway ({motorway:.3f})"
        print(f"✓ Minor > Motorway: {minor:.3f} > {motorway:.3f}")

    def test_null_light_condition(self):
        """Test: NULL light condition handled gracefully."""
        result = danger_index_udf(
            light=None,
            weather="Fine no high winds",
            surface="Dry",
            speed_limit=30,
            road_type="A-Road",
        )
        assert isinstance(result, float), "Must return float for NULL input"
        assert 0.0 <= result <= 1.0, "Must be in valid range"
        print(f"✓ NULL light condition: {result:.3f}")

    def test_null_weather(self):
        """Test: NULL weather condition handled gracefully."""
        result = danger_index_udf(
            light="Daylight",
            weather=None,
            surface="Dry",
            speed_limit=30,
            road_type="A-Road",
        )
        assert isinstance(result, float), "Must return float for NULL input"
        assert 0.0 <= result <= 1.0, "Must be in valid range"
        print(f"✓ NULL weather: {result:.3f}")

    def test_all_nulls(self):
        """Test: All NULL inputs returns neutral 0.5."""
        result = danger_index_udf(
            light=None, weather=None, surface=None, speed_limit=None, road_type=None
        )
        assert result == 0.5, f"All NULL should return 0.5 (neutral), got {result}"
        print(f"✓ All NULL inputs: {result:.3f}")


# =============================================================================
# VEHICLE VULNERABILITY UDF TESTS
# =============================================================================


class TestVehicleVulnerabilityUDF:
    """Tests for vehicle_vulnerability_udf function."""

    def test_motorcycle_vulnerable(self):
        """Test: All motorcycle types flagged as vulnerable."""
        motorcycles = [
            "Motorcycle 50cc and under",
            "Motorcycle 125cc and under",
            "Motorcycle over 125cc and up to 500cc",
            "Motorcycle over 500cc",
            "Motorcycle - unknown cc",
        ]
        for moto in motorcycles:
            result = vehicle_vulnerability_udf(vehicle_type=moto, engine_capacity=None)
            assert result is True, f"Motorcycle type '{moto}' should be vulnerable"
        print(f"✓ All {len(motorcycles)} motorcycle types flagged vulnerable")

    def test_pedal_cycle_vulnerable(self):
        """Test: Pedal cycles flagged as vulnerable."""
        result = vehicle_vulnerability_udf(
            vehicle_type="Pedal cycle", engine_capacity=None
        )
        assert result is True, "Pedal cycle should be vulnerable"
        print("✓ Pedal cycle flagged vulnerable")

    def test_ridden_horse_vulnerable(self):
        """Test: Ridden horses flagged as vulnerable."""
        result = vehicle_vulnerability_udf(
            vehicle_type="Ridden horse", engine_capacity=None
        )
        assert result is True, "Ridden horse should be vulnerable"
        print("✓ Ridden horse flagged vulnerable")

    def test_small_car_vulnerable(self):
        """Test: Cars with engine < 1200cc flagged as vulnerable."""
        result = vehicle_vulnerability_udf(vehicle_type="Car", engine_capacity=999)
        assert result is True, "Small car (999cc) should be vulnerable"
        print("✓ Small car (999cc) flagged vulnerable")

    def test_large_car_not_vulnerable(self):
        """Test: Cars with engine >= 1200cc NOT flagged as vulnerable."""
        result = vehicle_vulnerability_udf(vehicle_type="Car", engine_capacity=1500)
        assert result is False, "Large car (1500cc) should NOT be vulnerable"
        print("✓ Large car (1500cc) NOT flagged vulnerable")

    def test_bus_not_vulnerable(self):
        """Test: Buses NOT flagged as vulnerable."""
        result = vehicle_vulnerability_udf(
            vehicle_type="Bus or coach (17 or more pass seats)", engine_capacity=5000
        )
        assert result is False, "Bus should NOT be vulnerable"
        print("✓ Bus NOT flagged vulnerable")

    def test_van_not_vulnerable(self):
        """Test: Vans NOT flagged as vulnerable."""
        result = vehicle_vulnerability_udf(
            vehicle_type="Van / Goods 3.5 tonnes mgw or under", engine_capacity=2000
        )
        assert result is False, "Van should NOT be vulnerable"
        print("✓ Van NOT flagged vulnerable")

    def test_car_with_null_engine_not_vulnerable(self):
        """Test: Car with NULL engine capacity assumed standard (not vulnerable)."""
        result = vehicle_vulnerability_udf(vehicle_type="Car", engine_capacity=None)
        assert result is False, "Car with unknown engine should NOT be vulnerable"
        print("✓ Car with NULL engine NOT flagged vulnerable")

    def test_unknown_vehicle_type_not_vulnerable(self):
        """Test: Unknown vehicle types NOT flagged as vulnerable (conservative)."""
        result = vehicle_vulnerability_udf(
            vehicle_type="Unknown type", engine_capacity=None
        )
        assert (
            result is False
        ), "Unknown vehicle type should NOT be vulnerable (conservative)"
        print("✓ Unknown vehicle type NOT flagged vulnerable")


# =============================================================================
# END-TO-END TRANSFORMER TESTS
# =============================================================================


def test_danger_index_transformer_on_full_data():
    """Test: DangerIndexTransformer works on full dataset."""
    spark = get_spark("test-danger-transformer")
    df = spark.read.parquet(str(MERGED_PARQUET)).limit(100)

    transformer = DangerIndexTransformer()
    df_transformed = transformer.transform(df)

    # Verify column exists
    assert "danger_index" in df_transformed.columns, "danger_index column not created"

    # Verify all values in range [0, 1]
    danger_vals = df_transformed.select("danger_index").collect()
    for row in danger_vals:
        val = row["danger_index"]
        assert (
            0.0 <= val <= 1.0
        ), f"danger_index value out of range: {val}"

    print(f"✓ DangerIndexTransformer: 100 rows processed, all values in [0, 1]")
    spark.stop()


def test_vulnerability_transformer_on_full_data():
    """Test: VehicleVulnerabilityTransformer works on full dataset."""
    spark = get_spark("test-vuln-transformer")
    df = spark.read.parquet(str(MERGED_PARQUET)).limit(100)

    transformer = VehicleVulnerabilityTransformer()
    df_transformed = transformer.transform(df)

    # Verify column exists
    assert (
        "vehicle_vulnerable" in df_transformed.columns
    ), "vehicle_vulnerable column not created"

    # Verify all values are boolean
    vuln_vals = df_transformed.select("vehicle_vulnerable").collect()
    for row in vuln_vals:
        val = row["vehicle_vulnerable"]
        assert isinstance(val, bool), f"vehicle_vulnerable should be bool, got {type(val)}"

    print(f"✓ VehicleVulnerabilityTransformer: 100 rows processed, all values boolean")
    spark.stop()


def test_no_data_leakage():
    """Test: Features don't use Accident_Severity (target column)."""
    # This is a code inspection test rather than runtime test
    # Both UDFs don't have Accident_Severity in their function signatures

    import inspect
    from src.feature_engineering.domain_features import (
        danger_index_udf,
        vehicle_vulnerability_udf,
    )

    # Check function signatures
    danger_sig = inspect.signature(danger_index_udf)
    vuln_sig = inspect.signature(vehicle_vulnerability_udf)

    # Neither should have 'severity' or 'target' in parameters
    assert "severity" not in str(danger_sig), "danger_index_udf uses target!"
    assert "target" not in str(danger_sig), "danger_index_udf uses target!"
    assert "severity" not in str(vuln_sig), "vehicle_vulnerability_udf uses target!"
    assert "target" not in str(vuln_sig), "vehicle_vulnerability_udf uses target!"

    print("✓ No data leakage: Features don't depend on Accident_Severity")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING UNIT TESTS FOR DOMAIN FEATURES")
    print("=" * 80)

    # Danger Index tests
    print("\n[DANGER INDEX TESTS]")
    test_danger = TestDangerIndexUDF()
    test_danger.test_output_range_high_danger()
    test_danger.test_output_range_low_danger()
    test_danger.test_darkness_riskier_than_daylight()
    test_danger.test_fog_riskier_than_clear_weather()
    test_danger.test_icy_riskier_than_dry()
    test_danger.test_higher_speed_riskier()
    test_danger.test_minor_road_riskier_than_motorway()
    test_danger.test_null_light_condition()
    test_danger.test_null_weather()
    test_danger.test_all_nulls()

    # Vehicle Vulnerability tests
    print("\n[VEHICLE VULNERABILITY TESTS]")
    test_vuln = TestVehicleVulnerabilityUDF()
    test_vuln.test_motorcycle_vulnerable()
    test_vuln.test_pedal_cycle_vulnerable()
    test_vuln.test_ridden_horse_vulnerable()
    test_vuln.test_small_car_vulnerable()
    test_vuln.test_large_car_not_vulnerable()
    test_vuln.test_bus_not_vulnerable()
    test_vuln.test_van_not_vulnerable()
    test_vuln.test_car_with_null_engine_not_vulnerable()
    test_vuln.test_unknown_vehicle_type_not_vulnerable()

    # Integration tests
    print("\n[INTEGRATION TESTS]")
    test_danger_index_transformer_on_full_data()
    test_vulnerability_transformer_on_full_data()
    test_no_data_leakage()

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
