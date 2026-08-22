"""Routed surface-water limits for losing fixed-head groundwater boundaries."""

from __future__ import annotations

import numpy as np
from landlab.components import GroundwaterDupuitPercolator
from landlab.components.groundwater.dupuit_percolator import _update_thickness
from landlab.grid.mappers import map_value_at_max_node_to_link


def topological_reach_order(downstream_reach: np.ndarray) -> np.ndarray:
    """Return headwater-to-outlet order for a one-downstream-link reach graph."""
    downstream = np.asarray(downstream_reach, dtype=int)
    n_reaches = downstream.size
    if np.any((downstream < -1) | (downstream >= n_reaches)):
        raise ValueError("Downstream reach indices are outside the reach graph.")
    if np.any(downstream == np.arange(n_reaches)):
        raise ValueError("A reach cannot route to itself.")
    donor_count = np.bincount(
        downstream[downstream >= 0], minlength=n_reaches
    ).astype(int)
    queue = [int(value) for value in np.flatnonzero(donor_count == 0)]
    order: list[int] = []
    while queue:
        reach = queue.pop(0)
        order.append(reach)
        next_reach = int(downstream[reach])
        if next_reach >= 0:
            donor_count[next_reach] -= 1
            if donor_count[next_reach] == 0:
                queue.append(next_reach)
    if len(order) != n_reaches:
        raise ValueError("Reach routing graph is cyclic or incomplete.")
    return np.asarray(order, dtype=int)


def route_streamflow_with_availability(
    local_total_volume: np.ndarray,
    downstream_reach: np.ndarray,
    *,
    topological_order: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Route local volumes while preventing flow through any link from going negative.

    Returns routed outflow and the potential losing-stream volume rejected at each
    reach because neither upstream inflow nor local generation was available.
    """
    local = np.asarray(local_total_volume, dtype=float)
    downstream = np.asarray(downstream_reach, dtype=int)
    if local.shape != downstream.shape:
        raise ValueError("Local streamflow and downstream arrays must have equal shape.")
    order = (
        topological_reach_order(downstream)
        if topological_order is None
        else np.asarray(topological_order, dtype=int)
    )
    incoming = np.zeros_like(local)
    routed = np.zeros_like(local)
    unavailable = np.zeros_like(local)
    for reach in order:
        available = incoming[reach] + local[reach]
        if available < 0.0:
            unavailable[reach] = -available
            routed[reach] = 0.0
        else:
            routed[reach] = available
        next_reach = int(downstream[reach])
        if next_reach >= 0:
            incoming[next_reach] += routed[reach]
    return routed, unavailable


class RoutedStreamLimitedGroundwaterDupuitPercolator(
    GroundwaterDupuitPercolator
):
    """Landlab GDP with losing-stream flux capped by routed surface-water supply.

    The groundwater equation and adaptive timestep criteria match Landlab 2.10.1.
    At each substep, potential exchange across fixed-head stream links is combined
    with locally generated surface water and routed from headwaters to the outlet.
    Any potential loss that would make a reach outflow negative is removed before
    updating aquifer storage. Gaining exchange is never restricted.
    """

    def __init__(
        self,
        grid,
        *,
        reach_at_core_node: np.ndarray,
        boundary_links: np.ndarray,
        boundary_link_directions: np.ndarray,
        boundary_link_reaches: np.ndarray,
        downstream_reach: np.ndarray,
        limiter_tolerance_m3: float = 1.0e-6,
        limiter_max_iterations: int = 25,
        **kwargs,
    ):
        super().__init__(grid, **kwargs)
        self._limiter_reach_at_core = np.asarray(
            reach_at_core_node[grid.core_nodes], dtype=int
        )
        self._limiter_boundary_links = np.asarray(boundary_links, dtype=int)
        self._limiter_boundary_directions = np.asarray(
            boundary_link_directions, dtype=float
        )
        self._limiter_boundary_reaches = np.asarray(
            boundary_link_reaches, dtype=int
        )
        self._limiter_downstream = np.asarray(downstream_reach, dtype=int)
        self._limiter_order = topological_reach_order(self._limiter_downstream)
        self._limiter_n_reaches = self._limiter_downstream.size
        self._limiter_face_widths = grid.length_of_face[
            grid.face_at_link[self._limiter_boundary_links]
        ]
        self._limiter_tolerance_m3 = float(limiter_tolerance_m3)
        self._limiter_max_iterations = int(limiter_max_iterations)
        if self._limiter_tolerance_m3 < 0.0:
            raise ValueError("Stream-limiter tolerance cannot be negative.")
        if self._limiter_max_iterations < 1:
            raise ValueError("Stream-limiter iteration count must be positive.")
        if np.any(self._limiter_reach_at_core < 0):
            raise ValueError("Every core node must belong to a stream reach.")
        if np.any(self._limiter_boundary_reaches < 0):
            raise ValueError("Every stream boundary link must belong to a reach.")
        self.reset_stream_limiter_diagnostics()

    def reset_stream_limiter_diagnostics(self) -> None:
        self.unavailable_stream_loss_m3 = 0.0
        self.unavailable_stream_loss_by_reach_m3 = np.zeros(
            self._limiter_n_reaches, dtype=float
        )
        self.stream_limiter_substeps = 0
        self.stream_limiter_iteration_total = 0
        self.stream_limiter_max_iterations_used = 0
        self.stream_limiter_max_dry_reaches = 0
        self.stream_limiter_numerical_clip_m3 = 0.0

    def _surface_volume_from_update(
        self,
        thickness_start: np.ndarray,
        thickness_end: np.ndarray,
        dqdx: np.ndarray,
        substep_dt: float,
        *,
        validate_negative: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        surface_rate = np.zeros(self._grid.number_of_nodes, dtype=float)
        cores = self._cores
        surface_rate[cores] = (
            self._recharge[cores]
            - dqdx[cores]
            - self._n[cores]
            * (thickness_end[cores] - thickness_start[cores])
            / substep_dt
        )
        scale = np.maximum(
            np.maximum(np.abs(self._recharge[cores]), np.abs(dqdx[cores])),
            1.0e-15,
        )
        negative_tolerance = np.maximum(1.0e-16, 1.0e-8 * scale)
        invalid = surface_rate[cores] < -negative_tolerance
        negative_nodes = cores[surface_rate[cores] < 0.0]
        negative_volume_m3 = float(
            np.sum(
                -surface_rate[negative_nodes]
                * self._grid.cell_area_at_node[negative_nodes]
                * substep_dt
            )
        )
        if validate_negative and np.any(invalid) and negative_volume_m3 > max(
            1.0e-1, 100000.0 * self._limiter_tolerance_m3
        ):
            minimum = float(np.min(surface_rate[cores][invalid]))
            raise RuntimeError(
                "Stream limiter derived negative surface-water generation "
                f"({minimum:.6g} m/s; {negative_volume_m3:.6g} m3 over the "
                f"{substep_dt:.6g}-second substep at {negative_nodes.size} node(s))."
            )
        surface_rate[cores] = np.maximum(surface_rate[cores], 0.0)
        local_volume = np.bincount(
            self._limiter_reach_at_core,
            weights=(
                surface_rate[cores]
                * self._grid.cell_area_at_node[cores]
                * substep_dt
            ),
            minlength=self._limiter_n_reaches,
        ).astype(float)
        return surface_rate, local_volume, negative_volume_m3

    def _limited_boundary_flux(
        self,
        potential_q: np.ndarray,
        local_surface_volume_m3: np.ndarray,
        substep_dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        links = self._limiter_boundary_links
        potential_link_volume = (
            potential_q[links]
            * self._limiter_boundary_directions
            * self._limiter_face_widths
            * substep_dt
        )
        potential_groundwater_by_reach = np.bincount(
            self._limiter_boundary_reaches,
            weights=potential_link_volume,
            minlength=self._limiter_n_reaches,
        ).astype(float)
        _, unavailable = route_streamflow_with_availability(
            local_surface_volume_m3 + potential_groundwater_by_reach,
            self._limiter_downstream,
            topological_order=self._limiter_order,
        )

        actual_link_volume = potential_link_volume.copy()
        losing = potential_link_volume < 0.0
        for reach in np.flatnonzero(unavailable > self._limiter_tolerance_m3):
            reach_losing = losing & (self._limiter_boundary_reaches == reach)
            loss_magnitude = float(-np.sum(potential_link_volume[reach_losing]))
            if loss_magnitude <= 0.0:
                raise RuntimeError(
                    "A reach has unavailable stream loss but no losing boundary link."
                )
            actual_link_volume[reach_losing] += (
                unavailable[reach]
                * (-potential_link_volume[reach_losing])
                / loss_magnitude
            )
        actual_q = potential_q.copy()
        actual_q[links] = actual_link_volume / (
            self._limiter_boundary_directions
            * self._limiter_face_widths
            * substep_dt
        )
        return actual_q, unavailable

    def run_with_adaptive_time_step_solver(self, dt):
        """Advance groundwater while enforcing routed stream-water availability."""
        if (self._wtable > self._elev).any():
            self._wtable[self._wtable > self._elev] = self._elev[
                self._wtable > self._elev
            ]
            self._thickness[self._cores] = (self._wtable - self._base)[self._cores]

        self._base_grad[self._grid.active_links] = self._grid.calc_grad_at_link(
            self._base
        )[self._grid.active_links]
        cosa = np.cos(np.arctan(self._base_grad))
        reg_thickness = self._elev - self._base
        qs_cumulative = np.zeros_like(self._elev)
        remaining_time = float(dt)
        self._num_substeps = 0

        while remaining_time > 0.0:
            self._hydr_grad[self._grid.active_links] = (
                self._grid.calc_grad_at_link(self._wtable) * cosa
            )[self._grid.active_links]
            self._vel[:] = -self._K * self._hydr_grad
            hlink = (
                map_value_at_max_node_to_link(
                    self._grid, "water_table__elevation", "aquifer__thickness"
                )
                * cosa
            )
            potential_q = hlink * self._vel
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                dt_vn = self._vn_coefficient * np.min(
                    np.divide(
                        self._n_link * self._grid.length_of_link**2,
                        4 * self._K * hlink,
                        where=hlink > 0,
                        out=np.ones_like(potential_q) * 1.0e15,
                    )
                )
                dt_courant = self._courant_coefficient * np.min(
                    np.divide(
                        self._grid.length_of_link,
                        np.abs(self._vel / self._n_link),
                        where=np.abs(self._vel) > 0,
                        out=np.ones_like(potential_q) * 1.0e15,
                    )
                )
            substep_dt = min(dt_courant, dt_vn, remaining_time)
            if not np.isfinite(substep_dt) or substep_dt <= 0.0:
                raise RuntimeError("Stream-limited groundwater timestep is not positive.")

            thickness_start = self._thickness.copy()
            actual_q = potential_q.copy()
            previous_unavailable = None
            for iteration in range(1, self._limiter_max_iterations + 1):
                dqdx = self._grid.calc_flux_div_at_node(actual_q)
                thickness_end = thickness_start.copy()
                thickness_end[self._cores] = _update_thickness(
                    substep_dt,
                    thickness_start,
                    reg_thickness,
                    self._recharge,
                    dqdx,
                    self._n,
                    self._r,
                )[self._cores]
                thickness_end[thickness_end < 0.0] = 0.0
                surface_rate, local_surface_volume, _ = self._surface_volume_from_update(
                    thickness_start,
                    thickness_end,
                    dqdx,
                    substep_dt,
                    validate_negative=False,
                )
                new_q, unavailable = self._limited_boundary_flux(
                    potential_q,
                    local_surface_volume,
                    substep_dt,
                )
                if previous_unavailable is not None and np.max(
                    np.abs(unavailable - previous_unavailable)
                ) <= self._limiter_tolerance_m3:
                    actual_q = new_q
                    break
                actual_q = new_q
                previous_unavailable = unavailable
            else:
                raise RuntimeError(
                    "Routed stream-loss limiter did not converge within "
                    f"{self._limiter_max_iterations} iterations."
                )

            # Recompute the accepted state once from the beginning of the substep.
            dqdx = self._grid.calc_flux_div_at_node(actual_q)
            thickness_end = thickness_start.copy()
            thickness_end[self._cores] = _update_thickness(
                substep_dt,
                thickness_start,
                reg_thickness,
                self._recharge,
                dqdx,
                self._n,
                self._r,
            )[self._cores]
            thickness_end[thickness_end < 0.0] = 0.0
            surface_rate, _, numerical_clip_m3 = self._surface_volume_from_update(
                thickness_start,
                thickness_end,
                dqdx,
                substep_dt,
            )
            self._q[:] = actual_q
            self._thickness[:] = thickness_end
            self._wtable[:] = self._base + self._thickness
            self._qs[:] = surface_rate
            qs_cumulative += surface_rate * substep_dt

            self.unavailable_stream_loss_m3 += float(np.sum(unavailable))
            self.unavailable_stream_loss_by_reach_m3 += unavailable
            self.stream_limiter_numerical_clip_m3 += numerical_clip_m3
            dry_reaches = int(np.sum(unavailable > self._limiter_tolerance_m3))
            self.stream_limiter_substeps += 1
            self.stream_limiter_iteration_total += iteration
            self.stream_limiter_max_iterations_used = max(
                self.stream_limiter_max_iterations_used, iteration
            )
            self.stream_limiter_max_dry_reaches = max(
                self.stream_limiter_max_dry_reaches, dry_reaches
            )

            remaining_time = max(0.0, remaining_time - substep_dt)
            self._num_substeps += 1
            self._callback_fun(
                self._grid, self._recharge, substep_dt, **self._callback_kwds
            )

        self._qsavg[:] = qs_cumulative / dt
