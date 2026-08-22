import numpy as np
import pandas as pd
import pytest

from gw_simulator.hydrogeology_estimation import (
    RecessionFilter,
    aggregate_streamflow_to_daily,
    block_bootstrap_kirchner_signature,
    convert_streamflow_to_mm_day,
    derive_parameter_triplet,
    estimate_kirchner_signature,
    prepare_recession_pairs,
    rorabaugh_diffusivity_m2_day,
)


def test_convert_streamflow_to_mm_day() -> None:
    area_m2 = 86_400_000.0
    converted = convert_streamflow_to_mm_day(
        np.array([1.0]), units="m3_s", basin_area_m2=area_m2
    )
    np.testing.assert_allclose(converted, [1.0])


def test_aggregate_streamflow_to_daily_enforces_subdaily_coverage() -> None:
    first_day = pd.date_range("2020-01-01", periods=96, freq="15min")
    second_day = pd.date_range("2020-01-02", periods=48, freq="15min")
    frame = pd.DataFrame(
        {
            "when": first_day.append(second_day),
            "q": np.concatenate((np.ones(96), np.full(48, 2.0))),
        }
    )
    frame = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)

    daily, quality = aggregate_streamflow_to_daily(
        frame,
        date_column="when",
        flow_column="q",
        min_daily_coverage=0.80,
    )

    assert daily.loc[0, "flow"] == pytest.approx(1.0)
    assert np.isnan(daily.loc[1, "flow"])
    assert quality["modal_sampling_interval_minutes"] == 15.0
    assert quality["minimum_valid_observations_per_day"] == 77
    assert quality["duplicate_timestamp_count"] == 1


def test_prepare_recession_pairs_excludes_dry_season_and_high_et() -> None:
    dates = pd.date_range("2020-01-01", "2020-07-31", freq="D")
    flow = pd.DataFrame(
        {"date": dates, "q_mm_day": 2.0 - np.arange(len(dates)) * 0.001}
    )
    forcing = pd.DataFrame(
        {
            "date": dates,
            "P": 0.0,
            "ET": 1.0,
            "Recharge": 0.0,
        }
    )
    forcing.loc[forcing["date"].eq("2020-01-20"), "ET"] = 2.0
    pumping = pd.DataFrame(
        {
            "date": dates,
            "pumping_m3_day": np.where(dates.month <= 4, 10.0, 100.0),
        }
    )

    pairs, summary = prepare_recession_pairs(
        flow,
        forcing=forcing,
        pumping=pumping,
        config=RecessionFilter(max_pumping_quantile=0.8),
    )

    assert set(pairs["date"].dt.month).issubset({1, 2, 3, 4, 11, 12})
    assert pd.Timestamp("2020-01-20") not in set(pairs["date"])
    assert summary["eligible_pair_count"] == len(pairs)


def test_kirchner_signature_recovers_power_law_exponent() -> None:
    q = np.geomspace(0.02, 4.0, 600)
    exponent = 1.55
    minus_dqdt = 0.025 * q**exponent * (
        1.0 + 0.03 * np.sin(np.arange(len(q)))
    )
    pairs = pd.DataFrame(
        {"q_mid_mm_day": q, "minus_dqdt_mm_day2": minus_dqdt}
    )

    bins, summary = estimate_kirchner_signature(
        pairs, bin_count=12, min_bin_count=20
    )

    assert len(bins) == 12
    assert summary["power_law_b"] == pytest.approx(exponent, abs=0.02)
    assert summary["power_law_r_squared"] > 0.99
    assert summary["dynamic_storage_over_fitted_q_range_mm"] > 0.0


def test_water_year_block_bootstrap_returns_intervals() -> None:
    frames = []
    for water_year in range(2016, 2022):
        q = np.geomspace(0.03, 3.0, 60)
        frames.append(
            pd.DataFrame(
                {
                    "date": pd.date_range(f"{water_year - 1}-11-01", periods=60),
                    "q_mid_mm_day": q,
                    "minus_dqdt_mm_day2": 0.03 * q**1.6,
                }
            )
        )
    result = block_bootstrap_kirchner_signature(
        pd.concat(frames, ignore_index=True),
        replicates=20,
        min_pairs_per_water_year=10,
        min_bin_count=5,
    )
    assert result["successful_replicates"] == 20
    assert result["power_law_b"]["p50"] == pytest.approx(1.6, abs=0.02)


def test_rorabaugh_and_parameter_combination() -> None:
    diffusivity = rorabaugh_diffusivity_m2_day(100.0, 500.0)
    assert diffusivity == pytest.approx(2332.5)

    result = derive_parameter_triplet(
        transmissivity_m2_day=5.0,
        diffusivity_m2_day=diffusivity,
        dynamic_storage_mm=100.0,
    )
    assert result["specific_yield"] == pytest.approx(5.0 / 2332.5)
    assert result["effective_depth_m"] == pytest.approx(
        0.1 / (5.0 / 2332.5)
    )
