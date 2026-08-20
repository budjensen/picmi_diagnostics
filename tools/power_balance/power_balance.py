"""
Power-balance diagnostics for capacitively- and/or inductively-coupled PIC simulations.

These tools compute the different pieces of a plasma power balance from
``Analysis`` diagnostics data:

- ``compute_P_wall`` / ``compute_P_wall_phase_resolved``: power (I*V) delivered
  through a biased wall boundary, from the total current density (particle +
  displacement). Not every discharge has one (e.g. a purely inductive source
  with a grounded/unbiased boundary) -- functions downstream treat this as optional.
- ``compute_P_in`` / ``compute_P_in_phase_resolved``: power deposited into the
  plasma species, from the capacitive power-density diagnostic (``P_C``).
- ``compute_P_I_in`` / ``compute_P_I_in_phase_resolved``: power deposited into
  the plasma species, from the inductive power-density diagnostic (``P_I``).
- ``integrate_collisional_power``: power lost to collisions, integrated over space.
- ``compute_wall_flux_power`` / ``compute_wall_flux_power_from_intervals``: power
  deposited into the wall by particle flux (as opposed to I*V).
- ``spatial_power_profile``: spatial profile of input vs. collisional power density.
- ``power_balance_components``: convenience wrapper that runs all of the above,
  auto-detecting which channels (capacitive/inductive heating, biased wall) are
  present, and returns a single flat dict, suitable for tabulating/plotting
  across many runs.
- ``compare_power_balance`` / ``plot_power_balance_comparison``: build and plot
  a grouped-bar comparison of the power balance across several simulations.

All functions assume a single electron species and a single ion species (their
names are passed in via the ``electrons``/``ions`` keyword arguments, so the same
functions work across simulations that name their species differently) and a
periodic drive (e.g. RF) at a known frequency, so a single cycle can be
time-averaged. Power is reported per unit area (W/m^2), consistent with the 1D
diagnostics produced by ``Analysis``.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from .analysis import Analysis
except ImportError:
    from analysis import Analysis

VoltageFunc = Callable[[np.ndarray], np.ndarray]


def _resolve_rf_freq(diag: Analysis, rf_freq: Optional[float]) -> float:
    """Fall back to the diagnostic's own interval period if no frequency is given."""
    if rf_freq is not None:
        return rf_freq
    if hasattr(diag, 'interval_period'):
        return 1.0 / diag.interval_period
    raise ValueError(
        "rf_freq was not given and diag has no 'interval_period' to infer it from "
        "(only set when time-resolved diagnostics are present). Pass rf_freq explicitly."
    )


def _wall_current_density_intervals(diag: Analysis, electrons: str, ions: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Total current density (particle + displacement, interpolated to nodes) at
    each interval time, for use in wall I*V power calculations.

    Returns
    -------
    interval_times : np.ndarray
        Fraction of the drive cycle (0 to 1) at which each interval was sampled.
    J_t : list[np.ndarray]
        Total current density on the node grid, one array per interval time.
    """
    diag.avg_intervals(f'Jz_{electrons}')
    diag.avg_intervals(f'Jz_{ions}')
    diag.avg_intervals('J_d')

    n_intervals = len(diag.in_times)
    J_d_nodes = [np.interp(diag.nodes, diag.cells, diag.avg_in_data['J_d'][i]) for i in range(n_intervals)]
    J_t = [
        J_d_nodes[i] + diag.avg_in_data[f'Jz_{electrons}'][i] + diag.avg_in_data[f'Jz_{ions}'][i]
        for i in range(n_intervals)
    ]
    return diag.in_times, J_t


def compute_P_wall_phase_resolved(
    diag: Analysis,
    voltage: VoltageFunc,
    rf_freq: Optional[float] = None,
    electrons: str = 'electrons',
    ions: str = 'He',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Phase-resolved power (I*V) delivered by the wall over one drive cycle.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    voltage : Callable[[np.ndarray], np.ndarray]
        A function that takes an array of times (seconds) and returns the
        corresponding applied wall voltage.
    rf_freq : float, optional
        Drive frequency in Hz. If not given, falls back to
        ``1 / diag.interval_period`` (requires time-resolved diagnostics).
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'Jz_{electrons}'``).

    Returns
    -------
    times : np.ndarray
        Times within the cycle, in seconds.
    P_wall : np.ndarray
        Instantaneous wall power (I*V) at each time, in W/m^2.
    """
    freq = _resolve_rf_freq(diag, rf_freq)
    interval_times, J_t = _wall_current_density_intervals(diag, electrons, ions)
    times = interval_times / freq
    voltages = voltage(times)

    P_wall = np.array([-np.average(J) * V for J, V in zip(J_t, voltages)])
    return times, P_wall


def compute_P_wall(
    diag: Analysis,
    voltage: VoltageFunc,
    rf_freq: Optional[float] = None,
    electrons: str = 'electrons',
    ions: str = 'He',
) -> float:
    """
    Cycle-averaged power (I*V) delivered by the wall, using the total current
    density (particle + displacement) at the wall over interval slices.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    voltage : Callable[[np.ndarray], np.ndarray]
        A function that takes an array of times (seconds) and returns the
        corresponding applied wall voltage.
    rf_freq : float, optional
        Drive frequency in Hz. If not given, falls back to
        ``1 / diag.interval_period`` (requires time-resolved diagnostics).
    electrons, ions : str
        Names of the electron and ion species, as used in field names.

    Returns
    -------
    P_wall : float
        Cycle-averaged wall power, in W/m^2.
    """
    freq = _resolve_rf_freq(diag, rf_freq)
    times, P_wall_phase = compute_P_wall_phase_resolved(diag, voltage, freq, electrons, ions)
    dt = times[1] - times[0]
    return np.sum(P_wall_phase) * dt * freq


def _species_power_in_phase_resolved(
    diag: Analysis, prefix: str, rf_freq: Optional[float], electrons: str, ions: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-resolved power delivered to each species from a `f'{prefix}_<species>'` power-density field."""
    freq = _resolve_rf_freq(diag, rf_freq)
    diag.avg_intervals(f'{prefix}_{electrons}')
    diag.avg_intervals(f'{prefix}_{ions}')

    times = diag.in_times / freq
    P_in_e = np.array([np.sum(p * diag.dz) for p in diag.avg_in_data[f'{prefix}_{electrons}']])
    P_in_i = np.array([np.sum(p * diag.dz) for p in diag.avg_in_data[f'{prefix}_{ions}']])
    return times, P_in_e, P_in_i


def _species_power_in(diag: Analysis, prefix: str, electrons: str, ions: str) -> Tuple[float, float]:
    """Cycle-averaged power delivered to each species from a `f'{prefix}_<species>'` power-density field."""
    diag.avg_time_averaged(f'{prefix}_{electrons}')
    diag.avg_time_averaged(f'{prefix}_{ions}')

    P_in_e = np.sum(diag.avg_ta_data[f'{prefix}_{electrons}'] * diag.dz)
    P_in_i = np.sum(diag.avg_ta_data[f'{prefix}_{ions}'] * diag.dz)
    return P_in_e, P_in_i


def compute_P_in_phase_resolved(
    diag: Analysis,
    rf_freq: Optional[float] = None,
    electrons: str = 'electrons',
    ions: str = 'He',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-resolved capacitive power delivered to each plasma species over one
    drive cycle, from the capacitive power-density diagnostic (``P_C``)
    integrated over space.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    rf_freq : float, optional
        Drive frequency in Hz. If not given, falls back to
        ``1 / diag.interval_period`` (requires time-resolved diagnostics).
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'P_C_{electrons}'``).

    Returns
    -------
    times : np.ndarray
        Times within the cycle, in seconds.
    P_in_e, P_in_i : np.ndarray
        Instantaneous capacitive power delivered to each species at each
        time, in W/m^2.
    """
    return _species_power_in_phase_resolved(diag, 'P_C', rf_freq, electrons, ions)


def compute_P_in(diag: Analysis, electrons: str = 'electrons', ions: str = 'He') -> Tuple[float, float]:
    """
    Cycle-averaged capacitive power delivered to each plasma species, in
    W/m^2, from the time-averaged capacitive power-density diagnostic
    (``P_C``) integrated over space.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'P_C_{electrons}'``).

    Returns
    -------
    P_in_e, P_in_i : float
        Cycle-averaged capacitive power delivered to the electrons and ions, in W/m^2.
    """
    return _species_power_in(diag, 'P_C', electrons, ions)


def compute_P_I_in_phase_resolved(
    diag: Analysis,
    rf_freq: Optional[float] = None,
    electrons: str = 'electrons',
    ions: str = 'He',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-resolved inductive power delivered to each plasma species over one
    drive cycle, from the inductive power-density diagnostic (``P_I``)
    integrated over space.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    rf_freq : float, optional
        Drive frequency in Hz. If not given, falls back to
        ``1 / diag.interval_period`` (requires time-resolved diagnostics).
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'P_I_{electrons}'``).

    Returns
    -------
    times : np.ndarray
        Times within the cycle, in seconds.
    P_in_e, P_in_i : np.ndarray
        Instantaneous inductive power delivered to each species at each
        time, in W/m^2.
    """
    return _species_power_in_phase_resolved(diag, 'P_I', rf_freq, electrons, ions)


def compute_P_I_in(diag: Analysis, electrons: str = 'electrons', ions: str = 'He') -> Tuple[float, float]:
    """
    Cycle-averaged inductive power delivered to each plasma species, in
    W/m^2, from the time-averaged inductive power-density diagnostic
    (``P_I``) integrated over space.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'P_I_{electrons}'``).

    Returns
    -------
    P_in_e, P_in_i : float
        Cycle-averaged inductive power delivered to the electrons and ions, in W/m^2.
    """
    return _species_power_in(diag, 'P_I', electrons, ions)


def _field_available(diag: Analysis, field: str) -> bool:
    """Whether `field` is among this diagnostics object's time-averaged fields."""
    return field in getattr(diag, 'ta_fields', [])


def _discover_collisional_fields(diag: Analysis, electrons: str, ions: str) -> Tuple[List[str], List[str]]:
    """Find the per-collision-type ``coll-energy_*`` fields for each species."""
    ta_dir = diag.ta_colls[next(iter(diag.ta_colls))]
    coll_energy = [name.split('.npy')[0] for name in os.listdir(ta_dir) if name.startswith('coll-energy')]

    electron_fields = [name for name in coll_energy if name.startswith(f'coll-energy_{electrons}_')]
    ion_fields = [name for name in coll_energy if name.startswith(f'coll-energy_{ions}_')]
    return electron_fields, ion_fields


def integrate_collisional_power(diag: Analysis, electrons: str = 'electrons', ions: str = 'He') -> Tuple[float, float]:
    """
    Integrate the cycle-averaged collisional power loss over the spatial
    domain, for each species, in W/m^2.

    Discovers whichever ``coll-energy_<species>_<collision type>`` fields are
    present for each species (e.g. elastic, excitation, ionization), so it
    adapts automatically to the collision processes configured in a given run.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'coll-energy_{electrons}_elastic'``).

    Returns
    -------
    electron_power, ion_power : float
        Cycle-averaged collisional power loss for each species, in W/m^2.
    """
    electron_fields, ion_fields = _discover_collisional_fields(diag, electrons, ions)

    for field in electron_fields + ion_fields:
        diag.avg_time_averaged(field)

    electron_power = sum(np.sum(diag.avg_ta_data[field] * diag.dz) for field in electron_fields)
    ion_power = sum(np.sum(diag.avg_ta_data[field] * diag.dz) for field in ion_fields)
    return electron_power, ion_power


def compute_wall_flux_power(diag: Analysis, electrons: str = 'electrons', ions: str = 'He') -> Tuple[float, float]:
    """
    Cycle-averaged power deposited into the wall via particle flux (as opposed
    to I*V), in W/m^2, from the time-averaged ``Pw`` diagnostic.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'Pw_{electrons}'``).

    Returns
    -------
    P_wall_e, P_wall_i : float
        Cycle-averaged wall-flux power for each species, in W/m^2.
    """
    diag.avg_time_averaged(f'Pw_{electrons}')
    diag.avg_time_averaged(f'Pw_{ions}')

    P_wall_e = np.sum(diag.avg_ta_data[f'Pw_{electrons}'])
    P_wall_i = np.sum(diag.avg_ta_data[f'Pw_{ions}'])
    return P_wall_e, P_wall_i


def compute_wall_flux_power_from_intervals(
    diag: Analysis,
    rf_freq: Optional[float] = None,
    electrons: str = 'electrons',
    ions: str = 'He',
) -> Tuple[float, float]:
    """
    Cycle-averaged power deposited into the wall via particle flux, computed
    from the interval-sampled ``Pw`` diagnostic instead of the time-averaged
    one. Useful as a cross-check against ``compute_wall_flux_power``.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    rf_freq : float, optional
        Drive frequency in Hz. If not given, falls back to
        ``1 / diag.interval_period`` (requires time-resolved diagnostics).
    electrons, ions : str
        Names of the electron and ion species, as used in field names
        (e.g. ``f'Pw_{electrons}'``).

    Returns
    -------
    P_wall_e, P_wall_i : float
        Cycle-averaged wall-flux power for each species, in W/m^2.
    """
    freq = _resolve_rf_freq(diag, rf_freq)
    diag.avg_intervals(f'Pw_{electrons}')
    diag.avg_intervals(f'Pw_{ions}')

    times = diag.in_times / freq
    dt = times[1] - times[0]

    P_wall_e = sum(np.sum(Pw) * dt * freq for Pw in diag.avg_in_data[f'Pw_{electrons}'])
    P_wall_i = sum(np.sum(Pw) * dt * freq for Pw in diag.avg_in_data[f'Pw_{ions}'])
    return P_wall_e, P_wall_i


def spatial_power_profile(
    diag: Analysis, electrons: str = 'electrons', ions: str = 'He'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Spatial profile of cycle-averaged input power density vs. collisional
    power-loss density, for comparing where in the domain particles gain vs.
    lose energy.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    electrons, ions : str
        Names of the electron and ion species, as used in field names.

    Returns
    -------
    z : np.ndarray
        Cell-center positions, in meters.
    P_in_density : np.ndarray
        Power density delivered to the plasma species (``P_C``), in W/m^3.
    coll_density : np.ndarray
        Power density lost to collisions, in W/m^3.
    """
    diag.avg_time_averaged(f'P_C_{electrons}')
    diag.avg_time_averaged(f'P_C_{ions}')
    P_in_density = diag.avg_ta_data[f'P_C_{electrons}'] + diag.avg_ta_data[f'P_C_{ions}']

    electron_fields, ion_fields = _discover_collisional_fields(diag, electrons, ions)
    for field in electron_fields + ion_fields:
        diag.avg_time_averaged(field)
    coll_density = sum((diag.avg_ta_data[field] for field in electron_fields + ion_fields), np.zeros_like(P_in_density))

    return diag.cells, P_in_density, coll_density


def power_balance_components(
    diag: Analysis,
    voltage: Optional[VoltageFunc] = None,
    rf_freq: Optional[float] = None,
    electrons: str = 'electrons',
    ions: str = 'He',
) -> Dict[str, float]:
    """
    Compute a full power balance for one simulation: power in to the plasma
    species (capacitive and/or inductive, whichever are present), power lost
    to collisions and wall flux, and optionally the I*V power measured at a
    biased wall. Useful for tabulating/plotting across many runs (see
    ``compare_power_balance``).

    Which heating channels are included is auto-detected from the fields
    present in `diag` -- a purely capacitive discharge contributes only
    ``P_C_*`` terms, a purely inductive one only ``P_I_*``, and a hybrid
    discharge contributes both.

    Parameters
    ----------
    diag : Analysis
        The diagnostics object containing the simulation data.
    voltage : Callable[[np.ndarray], np.ndarray], optional
        A function that takes an array of times (seconds) and returns the
        corresponding applied wall voltage. Omit for discharges with no
        biased wall (e.g. a purely inductive source) -- ``'P_wall'`` is then
        left out of the returned dict entirely.
    rf_freq : float, optional
        Drive frequency in Hz. If not given, falls back to
        ``1 / diag.interval_period`` (requires time-resolved diagnostics).
    electrons, ions : str
        Names of the electron and ion species, as used in field names.

    Returns
    -------
    components : dict[str, float]
        ``P_C_in_e``/``P_C_in_i`` (capacitive) and/or ``P_I_in_e``/``P_I_in_i``
        (inductive), whichever channels are present; ``P_in_e``, ``P_in_i``,
        ``P_in`` (their totals); ``P_coll_e``, ``P_coll_i`` (collisional
        loss); ``P_wallflux_e``, ``P_wallflux_i`` (wall-flux loss, if the
        ``Pw_*`` diagnostic is present -- older runs may lack it); ``P_loss``
        (sum of whichever loss channels are present); and ``P_wall`` (I*V
        power from a biased wall) if `voltage` was given. All in W/m^2.
    """
    has_capacitive = _field_available(diag, f'P_C_{electrons}')
    has_inductive = _field_available(diag, f'P_I_{electrons}')
    if not (has_capacitive or has_inductive):
        raise ValueError(
            f"Neither 'P_C_{electrons}' nor 'P_I_{electrons}' were found among this "
            "simulation's diagnostics -- no input-power channel to compute."
        )

    components: Dict[str, float] = {}
    if has_capacitive:
        components['P_C_in_e'], components['P_C_in_i'] = compute_P_in(diag, electrons, ions)
    if has_inductive:
        components['P_I_in_e'], components['P_I_in_i'] = compute_P_I_in(diag, electrons, ions)

    P_in_e = components.get('P_C_in_e', 0.0) + components.get('P_I_in_e', 0.0)
    P_in_i = components.get('P_C_in_i', 0.0) + components.get('P_I_in_i', 0.0)
    components['P_in_e'] = P_in_e
    components['P_in_i'] = P_in_i
    components['P_in'] = P_in_e + P_in_i

    P_coll_e, P_coll_i = integrate_collisional_power(diag, electrons, ions)
    components.update(P_coll_e=P_coll_e, P_coll_i=P_coll_i)
    P_loss = P_coll_e + P_coll_i

    if _field_available(diag, f'Pw_{electrons}'):
        P_wallflux_e, P_wallflux_i = compute_wall_flux_power(diag, electrons, ions)
        components.update(P_wallflux_e=P_wallflux_e, P_wallflux_i=P_wallflux_i)
        P_loss += P_wallflux_e + P_wallflux_i
    components['P_loss'] = P_loss

    if voltage is not None:
        freq = _resolve_rf_freq(diag, rf_freq)
        components['P_wall'] = compute_P_wall(diag, voltage, freq, electrons, ions)

    return components


# Species colors (categorical slots 1 & 2 of the shared palette; validated as an
# adjacent CVD-safe pair) and a neutral gray for the (non species-resolved) wall
# power bar. Hatch texture, not color, distinguishes each bar's two mechanisms.
_SPECIES_COLORS = {'electrons': '#2a78d6', 'ions': '#eb6834'}
_WALL_COLOR = '#52514e'
_PRIMARY_HATCH = ''
_SECONDARY_HATCH = '///'
_SEGMENT_EDGE = '#fcfcfb'
_NET_MARKER_COLOR = '#0b0b0b'

# Mechanism colors for the species-agnostic views (categorical slots 1-4;
# validated as adjacent-safe pairs 1-2 and 3-4). `_FALLBACK_COLOR` marks a bar
# whose breakdown was skipped because a component went net negative -- a
# lighter neutral than `_WALL_COLOR` so the two "not a mechanism" grays stay
# visually distinguishable from each other.
_MECH_COLORS = {
    'capacitive': '#2a78d6',
    'inductive': '#eb6834',
    'collisional': '#1baf7a',
    'wallflux': '#eda100',
}
_FALLBACK_COLOR = '#898781'


def _balance_pct(P_in: float, P_out: float) -> float:
    """Percent difference between P_in and P_out, robust to negative/near-zero values.

    Dividing by `P_in` directly (the naive `(P_in-P_out)/P_in*100`) blows up
    or flips sign in confusing ways once `P_in` can be negative or small.
    Normalizing by the larger magnitude of the two keeps the result bounded
    (|pct| <= 200) and sign-meaningful (positive => P_in exceeds P_out).
    """
    denom = max(abs(P_in), abs(P_out))
    return 100.0 * (P_in - P_out) / denom if denom > 0 else 0.0


_COMPONENT_KEYS = [
    'P_C_in_e', 'P_C_in_i', 'P_I_in_e', 'P_I_in_i',
    'P_coll_e', 'P_coll_i', 'P_wallflux_e', 'P_wallflux_i', 'P_wall',
]


def _all_components_nonnegative(results: List[Dict]) -> bool:
    """Whether every source/sink component across all cases is >= 0 (missing treated as 0)."""
    return all(r.get(key, 0.0) >= 0.0 for r in results for key in _COMPONENT_KEYS)


def _mechanism_legend_handles(include_fallback: bool = False, include_wall: bool = False) -> List:
    """Shared legend swatches for the mechanism colors, used by the mechanism-based
    comparison chart and the per-species sources/sinks chart."""
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=_MECH_COLORS['capacitive'], edgecolor='none', label='Capacitive'),
        Patch(facecolor=_MECH_COLORS['inductive'], edgecolor='none', label='Inductive'),
        Patch(facecolor=_MECH_COLORS['collisional'], edgecolor='none', label='Collisional'),
        Patch(facecolor=_MECH_COLORS['wallflux'], edgecolor='none', label='Wall flux'),
    ]
    if include_fallback:
        handles.append(Patch(facecolor=_FALLBACK_COLOR, edgecolor='none', label='Net (breakdown skipped)'))
    if include_wall:
        handles.append(Patch(facecolor=_WALL_COLOR, edgecolor='none', label='$P_{wall}$ (I·V)'))
    return handles


def compare_power_balance(
    cases: List[Dict],
    quiet_startup: bool = True,
    **plot_kwargs,
) -> Tuple['plt.Figure', 'plt.Axes', List[Dict]]:
    """
    Compute and plot a power balance comparison across several simulations.

    This is the "plug in a list of directories" entry point: it loads each
    simulation, computes its power_balance_components(), and hands the
    results to plot_power_balance_comparison().

    Parameters
    ----------
    cases : list[dict]
        One entry per simulation to compare, in the order they should appear
        on the x-axis. Each dict accepts:

        - ``'directory'`` : str -- diagnostics directory to load. Alternatively,
          pass an already-constructed ``'diag'`` (an ``Analysis`` instance) to
          skip loading, e.g. to reuse a diagnostics object across multiple analyses.
        - ``'label'`` : str, optional -- name shown on the x-axis (default:
          the directory string).
        - ``'electrons'``, ``'ions'`` : str, optional -- species names
          (default ``'electrons'``, ``'He'``).
        - ``'voltage'`` : Callable[[np.ndarray], np.ndarray], optional --
          applied wall voltage vs. time. Omit for simulations with no biased
          wall (e.g. a purely inductive discharge); the P_wall bar is then
          skipped for that case.
        - ``'rf_freq'`` : float, optional -- drive frequency in Hz, falls
          back to ``diag.interval_period`` if omitted.
    quiet_startup : bool, default True
        Passed to ``Analysis()`` when constructing diagnostics from a directory.
    **plot_kwargs
        Passed through to ``plot_power_balance_comparison`` (e.g. ``ax``,
        ``dpi``, ``figsize``, ``annotate_balance``).

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    results : list[dict]
        The computed ``power_balance_components()`` dict for each case, with
        a ``'label'`` key added -- handy for saving to JSON, tabulating, or
        re-plotting later via ``plot_power_balance_comparison(results)``.
    """
    results = []
    for case in cases:
        diag = case.get('diag')
        if diag is None:
            diag = Analysis(case['directory'], quiet_startup=quiet_startup)
        electrons = case.get('electrons', 'electrons')
        ions = case.get('ions', 'He')

        components = power_balance_components(
            diag,
            voltage=case.get('voltage'),
            rf_freq=case.get('rf_freq'),
            electrons=electrons,
            ions=ions,
        )
        components['label'] = case.get('label') or case.get('directory', '?')
        results.append(components)

    fig, ax = plot_power_balance_comparison(results, **plot_kwargs)
    return fig, ax, results


def plot_power_balance_comparison(
    results: List[Dict],
    ax=None,
    dpi: int = 130,
    figsize: Optional[Tuple[float, float]] = None,
    annotate_balance: bool = True,
) -> Tuple['plt.Figure', 'plt.Axes']:
    """
    Grouped bar chart comparing power balance across simulations -- dispatches
    to whichever of the two renderings below suits the data:

    - If every source/sink component across all `results` is non-negative,
      uses ``plot_power_balance_by_species``: it's the more detailed
      comparison (breaks each bar down by species too), and with nothing
      negative there's no cancellation for that detail to obscure.
    - If any component is negative anywhere (e.g. capacitive power flowing
      back into the field at some case), falls back to
      ``plot_power_balance_by_mechanism``, which only breaks a bar down when
      its two totals are both non-negative and otherwise shows a single
      net-total bar -- see that function's docstring for why.

    Call either one directly to force a specific rendering regardless of sign.

    Parameters
    ----------
    results : list[dict]
        Per-simulation power balance dicts, as returned by
        ``power_balance_components()`` (with a ``'label'`` key added -- see
        ``compare_power_balance()``).
    ax, dpi, figsize, annotate_balance
        Passed through to whichever rendering is selected.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """
    if _all_components_nonnegative(results):
        return plot_power_balance_by_species(results, ax=ax, dpi=dpi, figsize=figsize, annotate_balance=annotate_balance)
    return plot_power_balance_by_mechanism(results, ax=ax, dpi=dpi, figsize=figsize, annotate_balance=annotate_balance)


def plot_power_balance_by_species(
    results: List[Dict],
    ax=None,
    dpi: int = 130,
    figsize: Optional[Tuple[float, float]] = None,
    annotate_balance: bool = True,
) -> Tuple['plt.Figure', 'plt.Axes']:
    """
    Grouped bar chart comparing power balance across simulations, broken down
    by species as well as by mechanism -- the detailed rendering that
    ``plot_power_balance_comparison`` uses automatically when nothing is negative.

    For each simulation, draws up to three bars:

    - ``P_wall``: I*V power measured at a biased wall (solid, single bar --
      not species-resolved). Skipped for simulations whose dict has no
      ``'P_wall'`` entry (e.g. an unbiased/purely inductive discharge).
    - ``P_in``: power delivered to the plasma species, stacked by species
      (color) and by heating mechanism, capacitive vs. inductive (hatch texture).
    - ``P_out``: power lost to collisions and wall flux, stacked the same
      way (species by color, collisional vs. wall-flux by hatch texture).

    A given component (e.g. capacitive power to ions) can time-average to a
    net negative value -- energy flowing from particles back into the field,
    or numerical noise near zero at a coarse dt. Rather than hide that, each
    stack splits into a positive part (stacking upward from zero) and a
    negative part (stacking downward from zero), so a canceling component is
    visible as a dip below the axis instead of silently vanishing or
    corrupting the rest of the stack. A short horizontal tick marks the true
    net total (the algebraic sum, ``P_in``/``P_loss`` -- the same value used
    for the balance annotation), so the gap between that tick and the top of
    the positive stack is a direct visual read of how much cancellation
    happened. This detailed, per-species view is the clearest comparison when
    everything is non-negative; once cancellation is actually happening
    somewhere, ``plot_power_balance_by_mechanism`` is usually easier to read.

    Two legends are drawn: one for species color, one for the hatch texture
    (which doubles as the P_wall bar's color key and the net-total tick's key,
    since those are also a "channel" of sorts).

    Parameters
    ----------
    results : list[dict]
        Per-simulation power balance dicts, as returned by
        ``power_balance_components()`` (with a ``'label'`` key added -- see
        ``compare_power_balance()``). Missing components (e.g. no inductive
        channel, or no P_wall) are treated as zero/absent, so simulations
        with different available channels can be compared side by side.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    dpi, figsize
        Passed to ``plt.subplots()`` when `ax` is None.
    annotate_balance : bool, default True
        Annotate each case with the percent difference between P_in and
        P_out, as a self-consistency check.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    n = len(results)
    x = np.arange(n)
    width = 0.26
    gap = 0.03
    offset_wall, offset_in, offset_out = -(width + gap), 0.0, (width + gap)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (1.7 * n + 3, 5.5), dpi=dpi)
    else:
        fig = ax.figure

    def stacked_signed(offset, segments):
        """segments: list of (heights, color, hatch).

        Positive-height entries stack upward from a running positive
        baseline; negative-height entries stack downward from a running
        negative baseline -- independent of segment order and of each other.
        A component that time-averages to a net negative value (energy
        flowing back into the field, or numerical noise near zero at a
        coarse dt) is drawn as a dip below the axis rather than silently
        dropped or corrupting the rest of the stack.

        Returns (pos_top, neg_bottom): the top of the positive stack and the
        bottom of the negative stack, per case. `pos_top + neg_bottom` equals
        the net (algebraic) total.
        """
        pos_bottom = np.zeros(n)
        neg_bottom = np.zeros(n)
        for heights, color, hatch in segments:
            heights = np.asarray(heights, dtype=float)
            pos_mask = heights > 0
            neg_mask = heights < 0
            if np.any(pos_mask):
                ax.bar(
                    x[pos_mask] + offset, heights[pos_mask], width, bottom=pos_bottom[pos_mask],
                    color=color, hatch=hatch, edgecolor=_SEGMENT_EDGE, linewidth=0.6,
                )
                pos_bottom[pos_mask] += heights[pos_mask]
            if np.any(neg_mask):
                ax.bar(
                    x[neg_mask] + offset, heights[neg_mask], width, bottom=neg_bottom[neg_mask],
                    color=color, hatch=hatch, edgecolor=_SEGMENT_EDGE, linewidth=0.6,
                )
                neg_bottom[neg_mask] += heights[neg_mask]
        return pos_bottom, neg_bottom

    def net_marker(offset, net_values):
        """An unfilled box outline from 0 to the true (algebraic) total for each
        case -- easier to read at a glance than a bare tick, since the box
        itself spans the value being reported rather than just pointing at it."""
        for xj, netj in zip(x + offset, net_values):
            ax.add_patch(Rectangle(
                (xj - 0.5 * width, min(0.0, netj)), width, abs(netj),
                fill=False, edgecolor=_NET_MARKER_COLOR, linewidth=1.8, zorder=6,
            ))

    e, i = _SPECIES_COLORS['electrons'], _SPECIES_COLORS['ions']

    pos_top_in, _ = stacked_signed(offset_in, [
        ([r.get('P_C_in_e', 0.0) for r in results], e, _PRIMARY_HATCH),
        ([r.get('P_C_in_i', 0.0) for r in results], i, _PRIMARY_HATCH),
        ([r.get('P_I_in_e', 0.0) for r in results], e, _SECONDARY_HATCH),
        ([r.get('P_I_in_i', 0.0) for r in results], i, _SECONDARY_HATCH),
    ])
    pos_top_out, _ = stacked_signed(offset_out, [
        ([r.get('P_coll_e', 0.0) for r in results], e, _PRIMARY_HATCH),
        ([r.get('P_coll_i', 0.0) for r in results], i, _PRIMARY_HATCH),
        ([r.get('P_wallflux_e', 0.0) for r in results], e, _SECONDARY_HATCH),
        ([r.get('P_wallflux_i', 0.0) for r in results], i, _SECONDARY_HATCH),
    ])

    P_in_total = np.array([r.get('P_in', 0.0) for r in results])
    P_out_total = np.array([r.get('P_loss', 0.0) for r in results])
    net_marker(offset_in, P_in_total)
    net_marker(offset_out, P_out_total)

    has_wall = np.array(['P_wall' in r for r in results])
    if np.any(has_wall):
        P_wall = np.array([r.get('P_wall', 0.0) for r in results])
        ax.bar(
            x[has_wall] + offset_wall, P_wall[has_wall], width,
            color=_WALL_COLOR, edgecolor=_SEGMENT_EDGE, linewidth=0.6,
        )

    ax.axhline(0, color='#c3c2b7', linewidth=1.0, zorder=0.5)

    if annotate_balance:
        # Position above the taller of the positive stacks (which, per
        # stacked_signed, always upper-bounds the net total too).
        drawn_top = np.maximum(pos_top_in, pos_top_out)
        for j in range(n):
            if P_in_total[j] == 0 and P_out_total[j] == 0:
                continue
            top = max(drawn_top[j], P_in_total[j], P_out_total[j], 0.0)
            pct = _balance_pct(P_in_total[j], P_out_total[j])
            ax.text(
                x[j] + offset_in, top * 1.03 + 0.01, f"{pct:+.1f}%",
                ha='center', va='bottom', fontsize=8, color=_WALL_COLOR,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r.get('label', str(idx)) for idx, r in enumerate(results)], rotation=15, ha='right')
    ax.set_ylabel('Power [W/m$^2$]')
    ax.set_title('Power balance comparison')
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.margins(x=0.02)

    species_handles = [
        Patch(facecolor=e, edgecolor='none', label='Electrons'),
        Patch(facecolor=i, edgecolor='none', label='Ions'),
    ]
    channel_handles = [
        Patch(facecolor='0.6', edgecolor=_SEGMENT_EDGE, hatch=_PRIMARY_HATCH, label='Capacitive / Collisional'),
        Patch(facecolor='0.6', edgecolor=_SEGMENT_EDGE, hatch=_SECONDARY_HATCH, label='Inductive / Wall flux'),
        Patch(facecolor='none', edgecolor=_NET_MARKER_COLOR, linewidth=1.8, label='Net total'),
    ]
    if np.any(has_wall):
        channel_handles.append(Patch(facecolor=_WALL_COLOR, edgecolor=_SEGMENT_EDGE, label='$P_{wall}$ (I·V)'))

    legend_species = ax.legend(
        handles=species_handles, title='Species', loc='upper left',
        bbox_to_anchor=(1.02, 1.0), fontsize=9, title_fontsize=9, frameon=False,
    )
    ax.add_artist(legend_species)
    ax.legend(
        handles=channel_handles, title='Channel', loc='upper left',
        bbox_to_anchor=(1.02, 1.0 - 0.09 * (len(species_handles) + 1.5)), fontsize=9, title_fontsize=9, frameon=False,
    )

    fig.tight_layout()
    return fig, ax


def plot_power_balance_by_mechanism(
    results: List[Dict],
    ax=None,
    dpi: int = 130,
    figsize: Optional[Tuple[float, float]] = None,
    annotate_balance: bool = True,
) -> Tuple['plt.Figure', 'plt.Axes']:
    """
    Grouped bar chart comparing the net power balance across simulations,
    broken down by mechanism only (not by species) -- the fallback rendering
    ``plot_power_balance_comparison`` uses automatically once any component is
    negative somewhere.

    For each simulation, draws up to three bars:

    - ``P_wall``: I*V power measured at a biased wall (solid, neutral color).
      Skipped for simulations whose dict has no ``'P_wall'`` entry.
    - ``P_in``: split into total capacitive and total inductive power (each
      summed over species).
    - ``P_out``: split into total collisional and total wall-flux power loss.

    Each bar breaks down into its two components *only if both are
    non-negative*. If either total is negative (e.g. capacitive power nets
    negative because a species is giving energy back to the field), the
    breakdown for that bar is skipped entirely and a single bar shows just
    the net total -- reconstructing a signed stack across already-aggregated
    mechanism totals tends to read as more confusing than informative once
    species-level detail isn't there to explain *why* it's negative (that's
    what ``plot_species_sources_sinks`` is for). Because of this rule, the
    bar's drawn height always equals the net total exactly, in every case --
    no separate net-total marker is needed here.

    Parameters
    ----------
    results : list[dict]
        Per-simulation power balance dicts, as returned by
        ``power_balance_components()`` (with a ``'label'`` key added -- see
        ``compare_power_balance()``).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    dpi, figsize
        Passed to ``plt.subplots()`` when `ax` is None.
    annotate_balance : bool, default True
        Annotate each case with the percent difference between P_in and
        P_out, as a self-consistency check.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """
    import matplotlib.pyplot as plt

    n = len(results)
    x = np.arange(n)
    width = 0.26
    gap = 0.03
    offset_wall, offset_in, offset_out = -(width + gap), 0.0, (width + gap)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (1.7 * n + 3, 5.5), dpi=dpi)
    else:
        fig = ax.figure

    def mechanism_bar(offset, total_a, color_a, total_b, color_b, net_total):
        """Two-color stack where both totals are >= 0; a single net-total bar elsewhere."""
        total_a = np.asarray(total_a, dtype=float)
        total_b = np.asarray(total_b, dtype=float)
        net_total = np.asarray(net_total, dtype=float)
        breakdown = (total_a >= 0) & (total_b >= 0)
        fallback = ~breakdown

        if np.any(breakdown):
            ax.bar(
                x[breakdown] + offset, total_a[breakdown], width,
                color=color_a, edgecolor=_SEGMENT_EDGE, linewidth=0.6,
            )
            ax.bar(
                x[breakdown] + offset, total_b[breakdown], width, bottom=total_a[breakdown],
                color=color_b, edgecolor=_SEGMENT_EDGE, linewidth=0.6,
            )
        if np.any(fallback):
            ax.bar(
                x[fallback] + offset, net_total[fallback], width,
                color=_FALLBACK_COLOR, edgecolor=_SEGMENT_EDGE, linewidth=0.6,
            )
        return fallback

    P_C_total = np.array([r.get('P_C_in_e', 0.0) + r.get('P_C_in_i', 0.0) for r in results])
    P_I_total = np.array([r.get('P_I_in_e', 0.0) + r.get('P_I_in_i', 0.0) for r in results])
    P_in_total = np.array([r.get('P_in', 0.0) for r in results])

    P_coll_total = np.array([r.get('P_coll_e', 0.0) + r.get('P_coll_i', 0.0) for r in results])
    P_wf_total = np.array([r.get('P_wallflux_e', 0.0) + r.get('P_wallflux_i', 0.0) for r in results])
    P_out_total = np.array([r.get('P_loss', 0.0) for r in results])

    in_fallback = mechanism_bar(offset_in, P_C_total, _MECH_COLORS['capacitive'], P_I_total, _MECH_COLORS['inductive'], P_in_total)
    out_fallback = mechanism_bar(offset_out, P_coll_total, _MECH_COLORS['collisional'], P_wf_total, _MECH_COLORS['wallflux'], P_out_total)

    has_wall = np.array(['P_wall' in r for r in results])
    if np.any(has_wall):
        P_wall = np.array([r.get('P_wall', 0.0) for r in results])
        ax.bar(
            x[has_wall] + offset_wall, P_wall[has_wall], width,
            color=_WALL_COLOR, edgecolor=_SEGMENT_EDGE, linewidth=0.6,
        )

    ax.axhline(0, color='#c3c2b7', linewidth=1.0, zorder=0.5)

    if annotate_balance:
        # The drawn bar height always equals the net total exactly (see
        # docstring), so it doubles directly as the annotation position.
        all_vals = np.concatenate([P_in_total, P_out_total, [0.0]])
        pad = 0.04 * max(np.max(np.abs(all_vals)), 1e-9)
        for j in range(n):
            if P_in_total[j] == 0 and P_out_total[j] == 0:
                continue
            pct = _balance_pct(P_in_total[j], P_out_total[j])
            top = max(P_in_total[j], P_out_total[j], 0.0)
            ax.text(
                x[j] + 0.5 * (offset_in + offset_out), top + pad, f"{pct:+.1f}%",
                ha='center', va='bottom', fontsize=8, color=_WALL_COLOR,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([r.get('label', str(idx)) for idx, r in enumerate(results)], rotation=15, ha='right')
    ax.set_ylabel('Power [W/m$^2$]')
    ax.set_title('Power balance comparison')
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.margins(x=0.02)

    handles = _mechanism_legend_handles(
        include_fallback=bool(np.any(in_fallback) or np.any(out_fallback)),
        include_wall=bool(np.any(has_wall)),
    )
    ax.legend(
        handles=handles, title='Source', loc='upper left',
        bbox_to_anchor=(1.02, 1.0), fontsize=9, title_fontsize=9, frameon=False,
    )

    fig.tight_layout()
    return fig, ax


def plot_species_sources_sinks(
    components: Dict,
    axes=None,
    dpi: int = 130,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Tuple['plt.Figure', Tuple['plt.Axes', 'plt.Axes']]:
    """
    Diverging horizontal bar chart of one simulation's power sources and
    sinks, faceted by species: electrons in one panel, ions in the other.

    Each panel lists the same four mechanisms (capacitive, inductive,
    collisional, wall flux) as bars extending right of zero for a net gain
    and left of zero for a net loss/sink -- collisional and wall-flux power
    are always losses (drawn negative), while capacitive and inductive power
    are drawn with whatever sign they actually have (capacitive power can
    itself net negative for a species -- see ``plot_power_balance_by_mechanism``).
    A label at the end of each bar gives its exact value.

    Parameters
    ----------
    components : dict
        A single simulation's ``power_balance_components()`` dict. An
        optional ``'label'`` is used as the figure title if `title` isn't given.
    axes : tuple[matplotlib.axes.Axes, matplotlib.axes.Axes], optional
        The (electrons, ions) axes to plot on. If None, creates a new figure.
    dpi, figsize
        Passed to ``plt.subplots()`` when `axes` is None.
    title : str, optional
        Figure title. Defaults to ``components.get('label')`` if present.

    Returns
    -------
    fig, axes : matplotlib Figure, tuple of two Axes (electrons, ions)
    """
    import matplotlib.pyplot as plt

    rows = [
        ('Capacitive', 'P_C_in_{s}', _MECH_COLORS['capacitive'], False),
        ('Inductive', 'P_I_in_{s}', _MECH_COLORS['inductive'], False),
        ('Collisional', 'P_coll_{s}', _MECH_COLORS['collisional'], True),
        ('Wall flux', 'P_wallflux_{s}', _MECH_COLORS['wallflux'], True),
    ]

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize or (9, 4), dpi=dpi, sharex=True)
    else:
        fig = axes[0].figure

    y = np.arange(len(rows))

    # Shared symmetric x-range across both panels, so bar lengths are directly comparable.
    max_abs = 0.0
    per_panel_values = []
    for suffix in ('e', 'i'):
        values = np.array([
            (-components.get(key.format(s=suffix), 0.0) if is_sink else components.get(key.format(s=suffix), 0.0))
            for _, key, _, is_sink in rows
        ])
        per_panel_values.append(values)
        max_abs = max(max_abs, np.max(np.abs(values)) if values.size else 0.0)
    max_abs = max(max_abs, 1e-9)
    label_pad = 0.02 * max_abs

    for ax, species_label, values in zip(axes, ('Electrons', 'Ions'), per_panel_values):
        colors = [c for _, _, c, _ in rows]
        ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.6, height=0.62, zorder=3)
        ax.axvline(0, color='#c3c2b7', linewidth=1, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels([label for label, *_ in rows])
        ax.invert_yaxis()
        ax.set_title(species_label, fontsize=11)
        ax.set_xlabel('Power [W/m$^2$]')
        ax.set_xlim(-1.35 * max_abs, 1.35 * max_abs)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', color='#e1e0d9', linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

        for yi, v in zip(y, values):
            ha = 'left' if v >= 0 else 'right'
            ax.text(
                v + (label_pad if v >= 0 else -label_pad), yi, f"{v:+.2f}",
                va='center', ha=ha, fontsize=8.5, color='#52514e', zorder=4,
            )

    fig.suptitle(title or components.get('label') or 'Sources (+) and sinks (-) by species', y=1.02)
    fig.legend(
        handles=_mechanism_legend_handles(), title='Mechanism', loc='upper left',
        bbox_to_anchor=(1.0, 0.92), fontsize=9, title_fontsize=9, frameon=False,
    )

    fig.tight_layout()
    return fig, tuple(axes)


def plot_power_balance_convergence(
    results: List[Dict],
    ax=None,
    dpi: int = 130,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple['plt.Figure', 'plt.Axes']:
    """
    Line plot of net P_in / P_loss (and P_wall, if present) across a set of
    simulations, in the order given -- e.g. a dt-convergence sweep -- so
    convergence shows up directly as the lines coming together, without a
    separate imbalance-percent panel.

    Unlike ``plot_power_balance_comparison``, the x-axis here is purely
    categorical (one tick per case, in input order -- exactly like the bar
    charts' x-axis), not the actual timestep value; if you want dt on a log
    axis instead, build that directly from ``Analysis.dt`` per case.

    Parameters
    ----------
    results : list[dict]
        Per-simulation power balance dicts, as returned by
        ``power_balance_components()`` (with a ``'label'`` key added -- see
        ``compare_power_balance()``), in the order they should appear on the
        x-axis.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    dpi, figsize
        Passed to ``plt.subplots()`` when `ax` is None.

    Returns
    -------
    fig, ax : matplotlib Figure, Axes
    """
    import matplotlib.pyplot as plt

    n = len(results)
    x = np.arange(n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (1.6 * n + 3, 5), dpi=dpi)
    else:
        fig = ax.figure

    P_in = np.array([r.get('P_in', 0.0) for r in results])
    P_out = np.array([r.get('P_loss', 0.0) for r in results])
    has_wall = np.array(['P_wall' in r for r in results])

    ax.plot(x, P_in, 'o-', color=_MECH_COLORS['capacitive'], linewidth=2, markersize=6, label='$P_{in}$', zorder=3)
    ax.plot(x, P_out, 'o-', color=_MECH_COLORS['inductive'], linewidth=2, markersize=6, label='$P_{loss}$', zorder=3)
    if np.any(has_wall):
        P_wall = np.array([r.get('P_wall', np.nan) if keep else np.nan for r, keep in zip(results, has_wall)])
        ax.plot(x, P_wall, 'o--', color=_WALL_COLOR, linewidth=1.6, markersize=5, label='$P_{wall}$', zorder=2)

    ax.axhline(0, color='#c3c2b7', linewidth=1.0, zorder=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([r.get('label', str(idx)) for idx, r in enumerate(results)], rotation=15, ha='right')
    ax.set_ylabel('Power [W/m$^2$]')
    ax.set_title('Power balance convergence')
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(x=0.05)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    return fig, ax
