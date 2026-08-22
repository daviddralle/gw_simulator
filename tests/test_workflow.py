from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from gw_simulator.workflow import (
    load_workflow_config,
    recharge_input_path,
    recharge_source,
)


class WorkflowRechargeTests(unittest.TestCase):
    @staticmethod
    def _write_config(root: Path, values: dict) -> Path:
        path = root / "workflow.yml"
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return path

    @staticmethod
    def _base_values() -> dict:
        return {
            "version": 1,
            "name": "example_basin",
            "boundary": "boundary.gpkg",
            "dem": "dem.tif",
            "groundwater": {
                "start_date": "2020-10-01",
                "end_date": "2021-09-30",
            },
        }

    def test_default_recharge_source_creates_earth_engine_output_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_workflow_config(
                self._write_config(root, self._base_values())
            )

            self.assertEqual(recharge_source(config), "earth_engine_deficit")
            self.assertEqual(
                recharge_input_path(config),
                (root / "data/forcing/example_basin/recharge.csv").resolve(),
            )
            self.assertEqual(
                config.path_value("forcing_cache"),
                (root / "data/forcing/example_basin/pml_v22a_prism").resolve(),
            )

    def test_user_csv_source_requires_and_resolves_csv_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = self._base_values()
            values["recharge"] = {"source": "csv"}
            values["recharge_csv"] = "forcing/recharge.csv"
            config = load_workflow_config(self._write_config(root, values))

            self.assertEqual(recharge_source(config), "csv")
            self.assertEqual(
                recharge_input_path(config),
                (root / "forcing/recharge.csv").resolve(),
            )

    def test_spatial_source_requires_manifest(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = self._base_values()
            values["recharge"] = {"source": "raster_manifest"}

            with self.assertRaisesRegex(ValueError, "recharge_raster_manifest"):
                load_workflow_config(self._write_config(root, values))


if __name__ == "__main__":
    unittest.main()
