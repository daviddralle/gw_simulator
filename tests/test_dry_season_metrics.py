import unittest

import numpy as np
import pandas as pd

from gw_simulator.dry_season_metrics import (
    DrySeasonMetricConfig,
    extract_basin_dry_season_metrics,
    extract_reach_dry_season_metrics,
    summarize_reach_dry_season_metrics,
)


def _basin_frame(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    size = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "unimpaired_total_streamflow_m3d": 10.0,
            "pumped_total_streamflow_m3d": 5.0,
            "total_streamflow_depletion_m3d": 5.0,
            "scheduled_pumping_m3d": 4.0,
            "allocated_pumping_m3d": 3.5,
            "modeled_extraction_m3d": 3.0,
            "source_capacity_shortfall_m3d": 0.5,
            "daily_aquifer_storage_depletion_m3": np.ones(size),
        }
    )


class DrySeasonMetricTests(unittest.TestCase):
    def test_basin_metrics_use_integrated_fraction_and_complete_calendar_season(self):
        metrics = extract_basin_dry_season_metrics(
            _basin_frame("2020-06-01", "2020-10-31")
        ).iloc[0]

        self.assertTrue(metrics["complete_season"])
        self.assertEqual(metrics["observed_days"], 153)
        self.assertAlmostEqual(
            metrics["integrated_streamflow_depletion_fraction_pct"], 50.0
        )
        self.assertAlmostEqual(metrics["pumped_streamflow_minimum_30day_mean_m3d"], 5.0)
        self.assertEqual(metrics["days_with_at_least_50pct_depletion"], 153)

    def test_incomplete_edge_season_is_flagged_and_not_ranked(self):
        frame = pd.concat(
            [
                _basin_frame("2020-06-01", "2020-10-31"),
                _basin_frame("2021-06-01", "2021-09-30"),
            ],
            ignore_index=True,
        )
        metrics = extract_basin_dry_season_metrics(frame).set_index("dry_season_year")

        self.assertTrue(metrics.loc[2020, "complete_season"])
        self.assertFalse(metrics.loc[2021, "complete_season"])
        self.assertTrue(
            pd.isna(metrics.loc[2021, "depletion_fraction_rank_complete_seasons"])
        )

    def test_reach_metrics_rank_fraction_and_volume_separately(self):
        dates = pd.date_range("2020-06-01", "2020-10-31", freq="D")
        rows = []
        for reach_id, natural, pumped in ((1, 100.0, 90.0), (2, 10.0, 1.0)):
            for date in dates:
                rows.append(
                    {
                        "date": date,
                        "reach_id": reach_id,
                        "unimpaired_local_total_streamflow_m3d": natural,
                        "local_total_streamflow_depletion_m3d": natural - pumped,
                        "routed_unimpaired_total_streamflow_m3d": natural,
                        "routed_pumped_total_streamflow_m3d": pumped,
                        "routed_total_streamflow_depletion_m3d": natural - pumped,
                    }
                )
        metrics = extract_reach_dry_season_metrics(pd.DataFrame(rows)).set_index(
            "reach_id"
        )

        self.assertEqual(metrics.loc[1, "depletion_volume_rank_within_season"], 1)
        self.assertEqual(metrics.loc[2, "depletion_fraction_rank_within_season"], 1)
        summary = summarize_reach_dry_season_metrics(metrics.reset_index())
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary["complete_season_count"].unique().tolist(), [1])


if __name__ == "__main__":
    unittest.main()
