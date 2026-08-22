import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from gw_simulator.recharge import (
    compute_storage_deficit_recharge,
    expand_pml_composites_to_daily,
    load_recharge_raster_manifest,
    validate_water_balance,
)


class RechargeTests(unittest.TestCase):
    def test_storage_deficit_recharge_conserves_daily_water_balance(self):
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        et = pd.DataFrame(
            {"ET_interpolated": [2.0, 2.0, 2.0, 2.0]}, index=dates
        )
        precipitation = pd.DataFrame(
            {"ppt_mm": [0.0, 1.0, 10.0, 0.0]}, index=dates
        )

        result = compute_storage_deficit_recharge(et, precipitation)

        np.testing.assert_allclose(result["Deficit"], [2.0, 3.0, 0.0, 2.0])
        np.testing.assert_allclose(result["Recharge"], [0.0, 0.0, 5.0, 0.0])

    def test_expand_pml_composites_preserves_mean_rate(self):
        composites = pd.DataFrame(
            {"ET_mm_day": [2.0, 4.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-09"]),
        )

        result = expand_pml_composites_to_daily(
            composites, start_date="2024-01-01", end_date="2024-01-16"
        )

        self.assertEqual(result.loc["2024-01-01":"2024-01-08", "ET"].sum(), 16.0)
        self.assertEqual(result.loc["2024-01-09":"2024-01-16", "ET"].sum(), 32.0)

    def test_expand_pml_composites_rejects_unsupported_tail(self):
        composites = pd.DataFrame(
            {"ET_mm_day": [2.0]}, index=pd.to_datetime(["2024-01-01"])
        )
        with self.assertRaisesRegex(ValueError, "only supports"):
            expand_pml_composites_to_daily(composites, end_date="2024-01-09")

    def test_water_balance_validator(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        et = pd.DataFrame({"ET": [2.0, 1.0, 1.0]}, index=dates)
        precipitation = pd.DataFrame({"P": [1.0, 3.0, 0.0]}, index=dates)
        result = compute_storage_deficit_recharge(et, precipitation)

        summary = validate_water_balance(result)

        self.assertEqual(summary["max_daily_balance_error_mm"], 0.0)
        self.assertEqual(summary["recharge_mm"], 1.0)

    def test_raster_manifest_supports_multiband_file_and_relative_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raster_path = root / "recharge.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=3,
                height=3,
                count=2,
                dtype="float32",
                crs="EPSG:26910",
                transform=from_origin(0.0, 3.0, 1.0, 1.0),
            ) as destination:
                destination.write(np.ones((3, 3), dtype="float32"), 1)
                destination.write(np.full((3, 3), 2.0, dtype="float32"), 2)
            manifest_path = root / "recharge_manifest.csv"
            pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-02"],
                    "raster_path": ["recharge.tif", "recharge.tif"],
                    "band": [1, 2],
                    "units": ["mm/day", "mm/day"],
                }
            ).to_csv(manifest_path, index=False)

            result = load_recharge_raster_manifest(
                manifest_path,
                "2020-01-01",
                "2020-01-02",
                inspect_rasters=True,
            )

            self.assertEqual(
                result["raster_path"].tolist(),
                [raster_path.resolve(), raster_path.resolve()],
            )
            self.assertEqual(result["band"].tolist(), [1, 2])

    def test_raster_manifest_rejects_missing_daily_record(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raster_path = root / "recharge.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=1,
                height=1,
                count=1,
                dtype="float32",
                crs="EPSG:26910",
                transform=from_origin(0.0, 1.0, 1.0, 1.0),
            ) as destination:
                destination.write(np.ones((1, 1), dtype="float32"), 1)
            manifest_path = root / "recharge_manifest.csv"
            pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-03"],
                    "raster_path": ["recharge.tif", "recharge.tif"],
                }
            ).to_csv(manifest_path, index=False)

            with self.assertRaisesRegex(ValueError, "missing 1 requested day"):
                load_recharge_raster_manifest(
                    manifest_path,
                    "2020-01-01",
                    "2020-01-03",
                )
