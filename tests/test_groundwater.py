from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from landlab import RasterModelGrid
from rasterio.transform import from_origin

from gw_simulator.groundwater import (
    GroundwaterState,
    HYDROGRAPH_DISPLAY_FLOOR_M3D,
    PumpingForcing,
    _format_hydrograph_axis,
    _plot_hydrograph_series,
    _run_daily_step,
    _topographic_source_zones,
    apply_specific_yield_floor,
    apply_well_aquifer_depth,
    build_depletion_table,
    build_reach_daily_table,
    route_reach_daily_table,
    build_reach_network,
    load_recharge_data,
    load_recharge_raster_field,
    run_scenarios,
)
from gw_simulator.stream_limiter import (
    RoutedStreamLimitedGroundwaterDupuitPercolator,
    route_streamflow_with_availability,
)


class GroundwaterTests(unittest.TestCase):
    def test_log_hydrograph_keeps_dry_days_at_labeled_display_floor(self):
        fig, axis = plt.subplots()
        try:
            line = _plot_hydrograph_series(
                axis,
                pd.date_range("2020-01-01", periods=3),
                [0.0, 0.5, 10.0],
            )[0]
            _format_hydrograph_axis(axis)

            np.testing.assert_allclose(
                line.get_ydata(),
                [HYDROGRAPH_DISPLAY_FLOOR_M3D, HYDROGRAPH_DISPLAY_FLOOR_M3D, 10.0],
            )
            self.assertEqual(axis.get_yscale(), "log")
            self.assertEqual(axis.get_ylim()[0], HYDROGRAPH_DISPLAY_FLOOR_M3D)
            self.assertIn("display floor", axis.texts[0].get_text())
        finally:
            plt.close(fig)

    @staticmethod
    def _topographic_test_state() -> GroundwaterState:
        grid = RasterModelGrid((4, 4), xy_spacing=1.0)
        grid.add_field(
            "water_table__elevation",
            np.full(grid.number_of_nodes, 2.0),
            at="node",
        )
        grid.add_field(
            "aquifer_base__elevation",
            np.zeros(grid.number_of_nodes),
            at="node",
        )
        grid.add_field(
            "aquifer__porosity",
            np.full(grid.number_of_nodes, 0.5),
            at="node",
        )
        grid.add_field(
            "database_aquifer__porosity",
            np.full(grid.number_of_nodes, 0.5),
            at="node",
        )
        grid.add_field(
            "aquifer__hydraulic_conductivity",
            np.ones(grid.number_of_nodes),
            at="node",
        )
        return GroundwaterState(
            grid=grid,
            dem_coarse=None,
            dx=1.0,
            dy=1.0,
            origin_x=0.0,
            origin_y=0.0,
            outlet_node=0,
            stream_indices=np.array([], dtype=int),
            active_nodes=np.isin(np.arange(grid.number_of_nodes), grid.core_nodes),
            cell_area=1.0,
            basin_area_m2=float(grid.core_nodes.size),
        )

    @staticmethod
    def _reach_test_state() -> GroundwaterState:
        grid = RasterModelGrid((5, 5), xy_spacing=1.0)
        grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        stream_nodes = np.array([6, 8, 12, 17, 22])
        grid.status_at_node[stream_nodes] = grid.BC_NODE_IS_FIXED_VALUE
        receiver = np.arange(grid.number_of_nodes)
        receiver[stream_nodes] = [12, 12, 17, 22, 22]
        receiver[[7, 11, 13, 16, 18]] = [6, 6, 8, 12, 17]
        drainage_area = np.zeros(grid.number_of_nodes)
        drainage_area[stream_nodes] = [1.0, 1.1, 3.0, 4.0, 5.0]
        grid.add_field("flow__receiver_node", receiver, at="node")
        grid.add_field("drainage_area", drainage_area, at="node")
        return GroundwaterState(
            grid=grid,
            dem_coarse=None,
            dx=1.0,
            dy=1.0,
            origin_x=0.0,
            origin_y=0.0,
            outlet_node=22,
            stream_indices=stream_nodes,
            active_nodes=np.isin(np.arange(grid.number_of_nodes), grid.core_nodes),
            cell_area=1.0,
            basin_area_m2=float(grid.core_nodes.size),
            stream_area_threshold=1.0,
        )

    def test_well_depth_changes_only_local_storage_geometry(self):
        grid = RasterModelGrid((4, 4), xy_spacing=50.0)
        grid.add_field(
            "topographic__elevation", np.full(grid.number_of_nodes, 100.0), at="node"
        )
        grid.add_field(
            "database_aquifer__thickness",
            np.full(grid.number_of_nodes, 10.0),
            at="node",
        )
        grid.add_field(
            "modeled_aquifer__thickness",
            np.full(grid.number_of_nodes, 10.0),
            at="node",
        )
        grid.add_field(
            "aquifer_base__elevation",
            np.full(grid.number_of_nodes, 90.0),
            at="node",
        )
        grid.add_field(
            "hydraulic_conductivity",
            np.full(grid.number_of_links, 1.0e-5),
            at="link",
        )
        state = GroundwaterState(
            grid=grid,
            dem_coarse=None,
            dx=50.0,
            dy=50.0,
            origin_x=0.0,
            origin_y=0.0,
            outlet_node=0,
            stream_indices=np.array([], dtype=int),
            active_nodes=np.isin(np.arange(grid.number_of_nodes), grid.core_nodes),
            cell_area=2500.0,
            basin_area_m2=10000.0,
        )
        well_node = int(grid.core_nodes[0])
        conductivity_before = grid.at_link["hydraulic_conductivity"].copy()

        applied = apply_well_aquifer_depth(state, [well_node], 25.0)

        np.testing.assert_array_equal(applied, [well_node])
        self.assertEqual(
            grid.at_node["modeled_aquifer__thickness"][well_node], 35.0
        )
        self.assertEqual(grid.at_node["aquifer_base__elevation"][well_node], 65.0)
        self.assertEqual(grid.at_node["database_aquifer__thickness"][well_node], 10.0)
        np.testing.assert_array_equal(
            grid.at_link["hydraulic_conductivity"], conductivity_before
        )
        untouched = np.setdiff1d(np.arange(grid.number_of_nodes), [well_node])
        np.testing.assert_array_equal(
            grid.at_node["modeled_aquifer__thickness"][untouched], 10.0
        )

    def test_load_recharge_data_rejects_missing_days(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "recharge.csv"
            pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-03"],
                    "Recharge": [1.0, 2.0],
                }
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing 1 requested day"):
                load_recharge_data(path, "2020-01-01", "2020-01-03")

    def test_spatial_recharge_is_applied_at_model_nodes_and_volume_weighted(self):
        state = self._topographic_test_state()
        state.grid.add_zeros("average_surface_water__specific_discharge", at="node")
        state.grid.add_zeros("recharge_rate", at="node")
        recharge = np.zeros(state.grid.number_of_nodes)
        recharge[state.grid.core_nodes] = [1.0, 2.0, 3.0, 4.0]

        class GroundwaterFlux:
            volume_m3 = 0.0

            def reset(self):
                self.volume_m3 = 0.0

        class Solver:
            number_of_substeps = 1

            def run_with_adaptive_time_step_solver(self, _duration):
                return None

        record = _run_daily_step(
            state,
            Solver(),
            GroundwaterFlux(),
            recharge_mm_day=recharge,
            previous_storage_m3=4.0,
        )

        self.assertAlmostEqual(record["recharge_mm_day"], 2.5)
        self.assertAlmostEqual(record["recharge_m3d"], 0.01)
        np.testing.assert_allclose(
            state.grid.at_node["recharge_rate"][state.grid.core_nodes]
            * 86400.0
            * 1000.0,
            [1.0, 2.0, 3.0, 4.0],
        )

    def test_uniform_spatial_field_matches_legacy_basin_scalar(self):
        def prepared_state():
            state = self._topographic_test_state()
            state.grid.add_zeros(
                "average_surface_water__specific_discharge", at="node"
            )
            state.grid.add_zeros("recharge_rate", at="node")
            return state

        class GroundwaterFlux:
            volume_m3 = 0.0

            def reset(self):
                self.volume_m3 = 0.0

        class Solver:
            number_of_substeps = 1

            def run_with_adaptive_time_step_solver(self, _duration):
                return None

        scalar_state = prepared_state()
        scalar_record = _run_daily_step(
            scalar_state,
            Solver(),
            GroundwaterFlux(),
            recharge_mm_day=2.5,
            previous_storage_m3=4.0,
        )
        spatial_state = prepared_state()
        spatial_values = np.full(spatial_state.grid.number_of_nodes, 2.5)
        spatial_record = _run_daily_step(
            spatial_state,
            Solver(),
            GroundwaterFlux(),
            recharge_mm_day=spatial_values,
            previous_storage_m3=4.0,
        )

        self.assertEqual(
            scalar_record["recharge_mm_day"], spatial_record["recharge_mm_day"]
        )
        self.assertEqual(
            scalar_record["recharge_m3d"], spatial_record["recharge_m3d"]
        )
        np.testing.assert_array_equal(
            scalar_state.grid.at_node["recharge_rate"],
            spatial_state.grid.at_node["recharge_rate"],
        )

    def test_recharge_raster_is_aligned_and_converted_to_node_order(self):
        state = self._topographic_test_state()
        match = xr.DataArray(
            np.zeros((4, 4)),
            coords={"y": [3.0, 2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0, 3.0]},
            dims=("y", "x"),
        )
        match = match.rio.write_crs("EPSG:26910").rio.write_transform(
            from_origin(-0.5, 3.5, 1.0, 1.0)
        )
        state.dem_coarse = match
        north_up = np.arange(16, dtype="float32").reshape(4, 4)

        with TemporaryDirectory() as directory:
            raster_path = Path(directory) / "recharge.tif"
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                width=4,
                height=4,
                count=1,
                dtype="float32",
                crs="EPSG:26910",
                transform=from_origin(-0.5, 3.5, 1.0, 1.0),
            ) as destination:
                destination.write(north_up, 1)

            result = load_recharge_raster_field(state, raster_path)

        np.testing.assert_allclose(result, np.flipud(north_up).ravel())

    def test_run_scenarios_rejects_snapshot_outside_simulation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "recharge.csv"
            pd.DataFrame(
                {"date": ["2020-01-01"], "Recharge": [0.0]}
            ).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "Snapshot date.*outside"):
                run_scenarios(
                    self._topographic_test_state(),
                    path,
                    start_date="2020-01-01",
                    end_date="2020-01-01",
                    snapshot_dates=["2020-01-02"],
                )

    def test_specific_yield_floor_changes_only_selected_low_values(self):
        state = self._topographic_test_state()
        nodes = state.grid.core_nodes[:2]
        state.grid.at_node["aquifer__porosity"][nodes] = [0.0002, np.float32(0.01)]
        state.grid.at_node["database_aquifer__porosity"][:] = state.grid.at_node[
            "aquifer__porosity"
        ]
        database_before = state.grid.at_node["database_aquifer__porosity"].copy()

        changed = apply_specific_yield_floor(state, 0.01)

        np.testing.assert_array_equal(changed, nodes[:1])
        np.testing.assert_allclose(
            state.grid.at_node["aquifer__porosity"][nodes], [0.01, 0.01]
        )
        np.testing.assert_array_equal(
            state.grid.at_node["database_aquifer__porosity"], database_before
        )

    def test_topographic_pumping_delivers_exact_volume_with_storage_caps(self):
        state = self._topographic_test_state()
        nodes = state.grid.core_nodes[:3]
        state.grid.at_node["water_table__elevation"][nodes] = [2.0, 1.0, 1.0e-12]
        forcing = PumpingForcing(
            mode="climatology",
            flux_fields={1: np.zeros(state.grid.number_of_nodes)},
            zero_flux=np.zeros(state.grid.number_of_nodes),
            source_mode="topographic",
            source_zones={12: nodes},
            zone_demands_m3d={1: {12: 0.6}},
            source_nodes=nodes,
        )

        flux = forcing.flux_for_date(
            "2020-01-01", state=state, source_zone_storage_fraction=0.5
        )
        withdrawn = -flux[nodes] * state.grid.cell_area_at_node[nodes] * 86400.0

        self.assertAlmostEqual(float(np.sum(withdrawn)), 0.6)
        np.testing.assert_allclose(withdrawn, [0.4, 0.2, 0.0])
        self.assertLessEqual(withdrawn[0], 0.5)
        self.assertLessEqual(withdrawn[1], 0.25)
        self.assertEqual(withdrawn[2], 0.0)
        self.assertTrue(np.all(flux[np.setdiff1d(np.arange(16), nodes)] == 0.0))

    def test_topographic_pumping_fails_when_zone_storage_is_insufficient(self):
        state = self._topographic_test_state()
        nodes = state.grid.core_nodes[:2]
        state.grid.at_node["water_table__elevation"][nodes] = [2.0, 1.0]
        forcing = PumpingForcing(
            mode="climatology",
            flux_fields={1: np.zeros(state.grid.number_of_nodes)},
            zero_flux=np.zeros(state.grid.number_of_nodes),
            source_mode="topographic",
            source_zones={12: nodes},
            zone_demands_m3d={1: {12: 0.8}},
            source_nodes=nodes,
        )

        with self.assertRaisesRegex(
            RuntimeError, "2020-01-01.*zone 12.*0.750 m3/day"
        ):
            forcing.flux_for_date(
                "2020-01-01", state=state, source_zone_storage_fraction=0.5
            )

    def test_topographic_pumping_clips_to_capacity_when_not_strict(self):
        state = self._topographic_test_state()
        nodes = state.grid.core_nodes[:2]
        state.grid.at_node["water_table__elevation"][nodes] = [2.0, 1.0]
        point_flux = np.zeros(state.grid.number_of_nodes)
        point_flux[nodes[0]] = -0.8 / 86400.0
        forcing = PumpingForcing(
            mode="climatology",
            flux_fields={1: point_flux},
            zero_flux=np.zeros(state.grid.number_of_nodes),
            source_mode="topographic",
            source_zones={12: nodes},
            zone_demands_m3d={1: {12: 0.8}},
            source_nodes=nodes,
        )

        flux = forcing.flux_for_date(
            "2020-01-01",
            state=state,
            source_zone_storage_fraction=0.5,
            strict_pumping_supply=False,
        )
        allocated = -np.sum(flux * state.grid.cell_area_at_node) * 86400.0

        self.assertAlmostEqual(allocated, 0.75)
        self.assertAlmostEqual(
            forcing.scheduled_volume_for_date("2020-01-01", state),
            0.8,
        )

    def test_coarser_source_network_merges_fine_reach_zones(self):
        grid = RasterModelGrid((5, 5), xy_spacing=1.0)
        receiver = np.arange(grid.number_of_nodes)
        receiver[[16, 11, 18, 13, 12, 6, 8, 7]] = [11, 12, 13, 12, 7, 8, 7, 2]
        drainage_area = np.zeros(grid.number_of_nodes)
        drainage_area[[11, 13, 12, 8]] = 0.3
        drainage_area[7] = 0.6
        drainage_area[2] = 0.7
        grid.add_field("flow__receiver_node", receiver, at="node")
        grid.add_field("drainage_area", drainage_area, at="node")
        state = GroundwaterState(
            grid=grid,
            dem_coarse=None,
            dx=1.0,
            dy=1.0,
            origin_x=0.0,
            origin_y=0.0,
            outlet_node=2,
            stream_indices=np.flatnonzero(drainage_area >= 0.25),
            active_nodes=np.isin(np.arange(grid.number_of_nodes), grid.core_nodes),
            cell_area=1.0,
            basin_area_m2=float(grid.core_nodes.size),
            stream_area_threshold=0.25,
        )

        fine_zones, fine_mapping = _topographic_source_zones(
            state,
            np.array([16, 6]),
            0.25,
        )
        coarse_zones, coarse_mapping = _topographic_source_zones(
            state,
            np.array([16, 6]),
            0.5,
        )

        self.assertEqual(len(fine_zones), 2)
        self.assertNotEqual(fine_mapping[16], fine_mapping[6])
        self.assertEqual(len(coarse_zones), 1)
        self.assertEqual(coarse_mapping[16], coarse_mapping[6])

    def test_reach_network_is_disjoint_and_excludes_upstream_contributions(self):
        state = self._reach_test_state()

        network = build_reach_network(state)

        self.assertEqual(network.number_of_reaches, 3)
        np.testing.assert_array_equal(network.downstream_reach, [2, 2, -1])
        self.assertTrue(
            np.all(network.reach_at_core_node[state.grid.core_nodes] >= 0)
        )
        np.testing.assert_allclose(network.incremental_area_m2, [2.0, 1.0, 2.0])
        self.assertAlmostEqual(
            float(np.sum(network.incremental_area_m2)), state.basin_area_m2
        )
        # The confluence reach contains only its own two incremental hillslope
        # cells; its upstream area separately includes both tributary reaches.
        self.assertEqual(network.incremental_area_m2[2], 2.0)
        self.assertEqual(network.upstream_area_m2[2], 5.0)

    def test_reach_daily_table_pairs_total_flow_without_clipping(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="D")

        def scenario(values):
            return pd.DataFrame(
                {
                    "date": np.repeat(dates, 3),
                    "reach_id": np.tile([1, 2, 3], 2),
                    "local_total_streamflow_m3d": np.asarray(values) + 1.0,
                }
            )

        natural = scenario([2.0, 3.0, 4.0, 3.0, 4.0, 5.0])
        pumped = scenario([1.0, 3.5, 2.0, 2.0, 4.5, 4.0])

        result = build_reach_daily_table(
            {"Unimpaired (Natural)": natural, "With Pumping": pumped}
        )

        np.testing.assert_allclose(
            result["local_total_streamflow_depletion_m3d"],
            [1.0, -0.5, 2.0, 1.0, -0.5, 1.0],
        )
        self.assertLess(result["local_total_streamflow_depletion_m3d"].min(), 0.0)
        self.assertFalse(any("groundwater" in column for column in result.columns))
        self.assertFalse(any("saturation" in column for column in result.columns))

    def test_reach_daily_routing_integrates_all_upstream_total_flow(self):
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"] * 3),
                "reach_id": [1, 2, 3],
                "unimpaired_local_total_streamflow_m3d": [3.0, 4.0, 5.0],
                "pumped_local_total_streamflow_m3d": [2.0, 3.0, 4.0],
                "local_total_streamflow_depletion_m3d": [1.0, 1.0, 1.0],
                "local_streamflow_depletion_fraction_pct": [100 / 3, 25.0, 20.0],
            }
        )

        result = route_reach_daily_table(daily, np.array([2, 2, -1]))

        np.testing.assert_allclose(
            result["routed_unimpaired_total_streamflow_m3d"], [3.0, 4.0, 12.0]
        )
        np.testing.assert_allclose(
            result["routed_pumped_total_streamflow_m3d"], [2.0, 3.0, 9.0]
        )
        np.testing.assert_allclose(
            result["routed_total_streamflow_depletion_m3d"], [1.0, 1.0, 3.0]
        )
        self.assertAlmostEqual(
            result.loc[result["reach_id"] == 3, "routed_streamflow_depletion_fraction_pct"].iloc[0],
            25.0,
        )

    def test_limited_reach_routing_zeroes_only_floating_point_negatives(self):
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"] * 2),
                "reach_id": [1, 2],
                "unimpaired_local_total_streamflow_m3d": [1.0, -1.0],
                "pumped_local_total_streamflow_m3d": [1.0, -1.0 - 1.0e-14],
                "local_total_streamflow_depletion_m3d": [0.0, 1.0e-14],
                "local_streamflow_depletion_fraction_pct": [0.0, np.nan],
            }
        )

        result = route_reach_daily_table(
            daily,
            np.array([1, -1]),
            enforce_nonnegative=True,
        )

        self.assertEqual(result["routed_pumped_total_streamflow_m3d"].min(), 0.0)
        self.assertEqual(
            result["routed_streamflow_depletion_fraction_pct"].dropna().max(),
            0.0,
        )

    def test_stream_availability_is_enforced_in_topological_order(self):
        downstream = np.array([2, 2, -1])
        routed, unavailable = route_streamflow_with_availability(
            np.array([-5.0, 4.0, -10.0]),
            downstream,
        )

        np.testing.assert_allclose(routed, [0.0, 4.0, 0.0])
        np.testing.assert_allclose(unavailable, [5.0, 0.0, 6.0])
        self.assertAlmostEqual(
            routed[2], -11.0 + float(np.sum(unavailable))
        )
        self.assertTrue(np.all(routed >= 0.0))

    def test_dry_stream_cannot_recharge_aquifer_through_fixed_head(self):
        grid = RasterModelGrid((3, 4), xy_spacing=1.0)
        grid.set_closed_boundaries_at_grid_edges(True, True, True, True)
        stream_node = 7
        grid.status_at_node[stream_node] = grid.BC_NODE_IS_FIXED_VALUE
        grid.add_field(
            "topographic__elevation", np.ones(grid.number_of_nodes), at="node"
        )
        grid.add_field(
            "aquifer_base__elevation", np.zeros(grid.number_of_nodes), at="node"
        )
        water_table = np.zeros(grid.number_of_nodes)
        water_table[grid.core_nodes] = 0.1
        water_table[stream_node] = 1.0
        grid.add_field("water_table__elevation", water_table, at="node")
        grid.add_field("recharge_rate", np.zeros(grid.number_of_nodes), at="node")
        active = grid.active_link_dirs_at_node[stream_node] != 0
        boundary_links = grid.links_at_node[stream_node][active]
        boundary_directions = grid.active_link_dirs_at_node[stream_node][active]
        reach_at_core = np.full(grid.number_of_nodes, -1, dtype=int)
        reach_at_core[grid.core_nodes] = 0
        initial_heads = water_table[grid.core_nodes].copy()

        component = RoutedStreamLimitedGroundwaterDupuitPercolator(
            grid,
            reach_at_core_node=reach_at_core,
            boundary_links=boundary_links,
            boundary_link_directions=boundary_directions,
            boundary_link_reaches=np.zeros(boundary_links.size, dtype=int),
            downstream_reach=np.array([-1]),
            hydraulic_conductivity=1.0e-4,
            recharge_rate="recharge_rate",
            porosity=0.2,
        )
        component.run_with_adaptive_time_step_solver(100.0)

        np.testing.assert_allclose(
            grid.at_node["water_table__elevation"][grid.core_nodes], initial_heads
        )
        np.testing.assert_allclose(
            grid.at_link["groundwater__specific_discharge"][boundary_links], 0.0
        )
        self.assertGreater(component.unavailable_stream_loss_m3, 0.0)

    def test_timeseries_pumping_is_zero_only_outside_coverage(self):
        january = np.array([-1.0, 0.0])
        zero = np.zeros(2)
        forcing = PumpingForcing(
            mode="timeseries",
            flux_fields={pd.Period("2010-01", freq="M"): january},
            zero_flux=zero,
            coverage_start=pd.Period("2010-01", freq="M"),
            coverage_end=pd.Period("2010-02", freq="M"),
        )

        np.testing.assert_array_equal(forcing.flux_for_date("2009-12-01"), zero)
        np.testing.assert_array_equal(forcing.flux_for_date("2010-01-01"), january)
        np.testing.assert_array_equal(forcing.flux_for_date("2010-03-01"), zero)
        with self.assertRaisesRegex(ValueError, "2010-02"):
            forcing.flux_for_date("2010-02-01")

    def test_depletion_table_reports_total_streamflow_only(self):
        dates = pd.date_range("2020-01-01", periods=2, freq="D")
        natural = pd.DataFrame(
            {
                "date": dates,
                "total_streamflow_m3d": [10.0, 12.0],
                "groundwater_to_stream_m3d": [6.0, 7.0],
                "saturation_excess_m3d": [4.0, 5.0],
                "pumping_m3d": [0.0, 0.0],
                "storage_m3": [100.0, 98.0],
                "initial_storage_m3": [100.0, 100.0],
            }
        )
        pumped = pd.DataFrame(
            {
                "date": dates,
                "total_streamflow_m3d": [8.0, 9.0],
                "groundwater_to_stream_m3d": [5.0, 5.5],
                "saturation_excess_m3d": [3.0, 3.5],
                "pumping_m3d": [3.0, 3.0],
                "scheduled_pumping_m3d": [4.0, 4.0],
                "storage_m3": [99.0, 96.0],
                "initial_storage_m3": [98.0, 98.0],
            }
        )

        result = build_depletion_table(natural, pumped)

        np.testing.assert_allclose(
            result["total_streamflow_depletion_m3d"], [2.0, 3.0]
        )
        self.assertNotIn("groundwater_depletion_m3d", result)
        self.assertNotIn("saturation_excess_depletion_m3d", result)
        np.testing.assert_allclose(
            result["impaired_streamflow_fraction_pct"], [80.0, 75.0]
        )
        np.testing.assert_allclose(
            result["instantaneous_streamflow_response_fraction_pct"],
            [200.0 / 3.0, 100.0],
        )
        np.testing.assert_allclose(
            result["instantaneous_storage_response_fraction_pct"],
            [-100.0 / 3.0, 100.0 / 3.0],
        )
        np.testing.assert_allclose(
            result["cumulative_capture_fraction_pct"], [50.0, 62.5]
        )
        np.testing.assert_allclose(result["modeled_extraction_m3d"], [1.0, 4.0])
        np.testing.assert_allclose(result["pumping_balance_gap_m3d"], [2.0, -1.0])
        np.testing.assert_allclose(
            result["cumulative_modeled_capture_fraction_pct"],
            [200.0, 100.0],
        )
        np.testing.assert_allclose(
            result["cumulative_schedule_fulfillment_pct"],
            [25.0, 62.5],
        )
        np.testing.assert_allclose(
            result["cumulative_source_capacity_shortfall_m3"],
            [1.0, 2.0],
        )
        np.testing.assert_allclose(
            result["cumulative_source_allocation_fulfillment_pct"],
            [75.0, 75.0],
        )
        np.testing.assert_allclose(
            result["cumulative_unmodeled_supply_requirement_m3"],
            [3.0, 3.0],
        )
