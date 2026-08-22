import unittest

import numpy as np
import pandas as pd

from gw_simulator.reach_plots import (
    ReachVisualizationConfig,
    add_rolling_depletion_fraction,
    select_representative_water_years,
    water_year,
)


class ReachPlotTests(unittest.TestCase):
    def test_water_year_changes_on_october_first(self):
        dates = pd.to_datetime(["2020-09-30", "2020-10-01", "2021-01-01"])
        np.testing.assert_array_equal(water_year(dates), [2020, 2021, 2021])

    def test_rolling_fraction_uses_window_volumes_and_keeps_negative_values(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        frame = pd.DataFrame(
            {
                "date": np.repeat(dates, 2),
                "reach_id": np.tile([1, 2], 3),
                "routed_unimpaired_total_streamflow_m3d": [10, 5, 10, 5, 10, 5],
                "routed_total_streamflow_depletion_m3d": [2, -1, 4, -2, 6, -3],
            }
        )

        result = add_rolling_depletion_fraction(
            frame,
            ReachVisualizationConfig(rolling_days=2),
        )
        last = result[result["date"] == dates[-1]].set_index("reach_id")

        self.assertAlmostEqual(
            last.loc[1, "rolling_routed_depletion_fraction_pct"], 50.0
        )
        self.assertAlmostEqual(
            last.loc[2, "rolling_routed_depletion_fraction_pct"], -50.0
        )

    def test_representative_years_are_complete_driest_and_wettest(self):
        dates = pd.date_range("2018-10-01", "2021-09-30", freq="D")
        frame = pd.DataFrame({"date": dates, "recharge_m3d": 2.0})
        frame.loc[water_year(frame["date"]) == 2020, "recharge_m3d"] = 0.5
        frame.loc[water_year(frame["date"]) == 2021, "recharge_m3d"] = 4.0

        selected, recharge = select_representative_water_years(frame)

        self.assertEqual(selected, [2020, 2021])
        self.assertLess(recharge[2020], recharge[2019])
        self.assertGreater(recharge[2021], recharge[2019])


if __name__ == "__main__":
    unittest.main()
