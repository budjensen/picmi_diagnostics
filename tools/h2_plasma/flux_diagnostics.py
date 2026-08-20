'''
Reproducible tools for computing ion, electron, and H-radical flux at the
anode from picmi_diagnostics.Analysis output, across the DC/RF discharge
sweeps in this study (dc_base*, f30_p28*, f60_p28*, f30_p18*, f60_p18*, ...).

Anode convention
-----------------
Every inputs_*.py in this study grounds the lower z boundary
(warpx_potential_lo_z=0.0) and applies the DC bias (plus any RF term) to the
upper boundary (warpx_potential_hi_z=self.voltage, voltage=-300 V by
default). The lower boundary is therefore the anode, and it corresponds to
the 'lw' ("left wall") key used throughout Analysis (e.g.
get_avg_wall_edf_data(separate_rl=True)[species]['lw']). 'rw' is the biased
cathode. Re-check this against warpx_potential_lo_z/hi_z before reusing
these tools on a differently-wired input file.

Ion/electron flux at the anode: EDF weight normalization
-----------------------------------------------------------
main.py's calculate_wall_eadf (_get_wall_ieadf/_get_wall_eeadf) histograms
the scraped-particle weights as `weights=w/self.dz`, so
sum(avg_wall_edf_data[species][wall]) equals <raw scraped weight over one
collection window>/dz, not the raw weight itself. Analysis' own
normalize_wall_edf divides by the EDF's own trapezoidal integral, so this
1/dz factor cancels there and has gone unnoticed in every existing shape-only
plot -- but it matters for an absolute flux. charged_species_flux() below
multiplies the EDF sum back by dz before dividing by the collection time
(dz_correction=True, the default). This was checked against J_w (a second,
independently-normalized wall diagnostic sampled via
get_particle_scraped_this_step * q_e/dt, with no dz term) on
dc_base_1/diags_p28_V300: dz_correction=True gives
q_e*(electron_flux - ion_flux) at the anode within ~8% of J_w's own value,
while the uncorrected sum/collection_time is off by ~1/dz (~1e6). Pass
dz_correction=False to recover that uncorrected number if needed, and use
wall_current_from_J_w() to re-check this on new data.

Anode particle-collection undercount
-------------------------------------
As of Aug 2026 the simulations here only record about half of the particles
that actually reach the anode in the boundary-scraping diagnostic (bug not
yet fixed at the WarpX/diagnostics level). undercount_factor (default 2.0)
compensates for this in the ion/electron flux functions. It does not apply
to the H-radical flux, which comes from the reaction-rate-driven diffusion
solve below, not from wall-EDF particle scraping.
'''
import numpy as np
from scipy.sparse import diags as _diags_matrix
from scipy.sparse.linalg import spsolve

try:
    from .analysis import Analysis
except ImportError:
    from analysis import Analysis

# Physical constants (SI), copied from WarpX/Python/pywarpx/picmi.py for
# consistency with the rest of this codebase (see reactor_calc_paper.ipynb).
Q_E = 1.602176634e-19
M_E = 9.1093837015e-31
M_P = 1.67262192369e-27
K_B = 1.380649e-23

ANODE_WALL = 'lw'


def get_diagnostic_collection_time(diag) -> float:
    '''
    Duration [s] over which the wall-EDF/EADF boundary-scraping buffer
    accumulates before each lw_/rw_*.npy snapshot is written (the buffer is
    cleared at the start of a diagnostic output and saved at its end -- see
    main.py's do_diagnostics/clear_wall_eadf_buffers). Parsed from
    'Diagnostic time [s]=' in diagnostic_times.dat.

    Parameters
    ----------
    diag : picmi_diagnostics.analysis.Analysis

    Returns
    -------
    float
    '''
    with open(f'{diag.directory}/diagnostic_times.dat', 'r') as f:
        for line in f:
            if line.startswith('Diagnostic time [s]='):
                return float(line.split('=')[1])
    raise ValueError(
        f"'Diagnostic time [s]=' not found in {diag.directory}/diagnostic_times.dat")


def charged_species_flux(diag,
                          species: str,
                          wall: str = ANODE_WALL,
                          undercount_factor: float = 2.0,
                          dz_correction: bool = True) -> float:
    '''
    Real-particle flux [m^-2 s^-1] of `species` at `wall`, from the
    collection-averaged wall EDF.

    Parameters
    ----------
    diag : picmi_diagnostics.analysis.Analysis
    species : str
        Must be one of diag.wall_eadf_dir (e.g. 'electrons', 'H3p').
    wall : str, default='lw'
        'lw' (anode, grounded lower boundary in this study) or 'rw' (cathode).
    undercount_factor : float, default=2.0
        Multiplicative correction for the anode boundary-scraping undercount
        bug (see module docstring). Set to 1.0 once that bug is fixed, or to
        apply no correction at the cathode.
    dz_correction : bool, default=True
        Multiply the EDF sum back by diag.dz before dividing by the
        collection time (see module docstring). Set False to reproduce the
        raw, uncorrected sum/collection_time estimate instead.

    Returns
    -------
    float
    '''
    if not diag.wall_eadf_bool:
        raise ValueError('Wall EADF data not found for this Analysis object')
    edfs = diag.get_avg_wall_edf_data(separate_rl=True)
    if species not in edfs:
        raise ValueError(f'Species must be one of: {", ".join(edfs.keys())}')
    if wall not in edfs[species]:
        raise ValueError(f'Wall must be one of: {", ".join(edfs[species].keys())}')

    weight_sum = np.sum(edfs[species][wall])
    if dz_correction:
        weight_sum = weight_sum * diag.dz

    collection_time = get_diagnostic_collection_time(diag)
    return undercount_factor * weight_sum / collection_time


def all_charged_species_fluxes(diag,
                                wall: str = ANODE_WALL,
                                undercount_factor: float = 2.0,
                                dz_correction: bool = True) -> dict:
    '''
    charged_species_flux() for every species with wall-EDF data at `wall`.

    Returns
    -------
    dict[str, float]
        Flux [m^-2 s^-1] keyed by species name.
    '''
    edfs = diag.get_avg_wall_edf_data(separate_rl=True)
    return {
        species: charged_species_flux(diag, species, wall=wall,
                                       undercount_factor=undercount_factor,
                                       dz_correction=dz_correction)
        for species in edfs if wall in edfs[species]
    }


def wall_current_from_J_w(diag, wall: str = ANODE_WALL) -> float:
    '''
    Time-averaged net current density [A/m^2] at `wall` from the J_w field,
    a second, independently-sampled wall diagnostic (per-step scraped
    samples * q_e/dt, no dz term -- see main.py's update_J_w). Useful as a
    quick cross-check on charged_species_flux()'s dz_correction: compare
    q_e * (electron_flux - ion_flux) at 'lw' (or ion - electron at 'rw')
    against this value.

    Parameters
    ----------
    diag : picmi_diagnostics.analysis.Analysis
    wall : str, default='lw'

    Returns
    -------
    float
    '''
    if 'J_w' not in getattr(diag, 'ta_fields', []):
        raise ValueError('J_w time-averaged data not found for this Analysis object')
    diag.avg_time_averaged('J_w')
    idx = 0 if wall == 'lw' else 1
    return diag.avg_ta_data['J_w'][idx]


# --- H radical density / flux ------------------------------------------
#
# Reproduces Mar_2026/dc_paper/reactor_calc_paper.ipynb's H-production
# mechanism and steady-state diffusion solve, generalized to any Analysis
# object with the same collision-rate fields.

# field name -> (H atoms produced per reaction, required)
# electron dissociation (excitation7/8) and ion dissociation (H3p
# excitation1-3, e- + H2 -> H2+ + 2e-, H2+ + H2 -> H3+ + H) coefficients
# match reactor_calc_paper.ipynb exactly, including the ion-dissociation
# 2 * (...) / 2 that cancels to a coefficient of 1.0 per channel (the
# cross-section for that channel is doubled for transport reasons in the
# underlying MCC data, and halved back out here).
DEFAULT_H_REACTIONS = {
    'coll-rate_electrons_excitation7': (2.0, True),
    'coll-rate_electrons_excitation8': (2.0, True),
    'coll-rate_H3p_excitation1': (1.0, False),
    'coll-rate_H3p_excitation2': (1.0, False),
    'coll-rate_H3p_excitation3': (1.0, False),
    'coll-rate_e_h3_recombination': (3.0, True),
    'coll-rate_electrons_ionization': (1.0, True),
}


def h_production_rate_profile(diag, reactions: dict = None) -> np.ndarray:
    '''
    Collection-averaged H-atom production rate [m^-3 s^-1] profile (on
    diag.cells), summed over `reactions`.

    Parameters
    ----------
    diag : picmi_diagnostics.analysis.Analysis
    reactions : dict, default=None
        Maps a coll-rate field name to (H atoms per reaction, required). If
        None, uses DEFAULT_H_REACTIONS. A channel marked non-required is
        silently skipped (contributes zero) if its field isn't present in
        diag.ta_fields; a required channel raises if missing.

    Returns
    -------
    np.ndarray
    '''
    if reactions is None:
        reactions = DEFAULT_H_REACTIONS
    C_H_total = np.zeros_like(diag.cells)
    for field, (coeff, required) in reactions.items():
        if field not in diag.ta_fields:
            if required:
                raise ValueError(
                    f'Required collision-rate field {field!r} not found in {diag.directory}')
            continue
        diag.avg_time_averaged(field)
        C_H_total = C_H_total + coeff * diag.avg_ta_data[field]
    return C_H_total


def solve_h_density_profile(diag,
                             pressure_torr: float,
                             reactions: dict = None,
                             gas_temp_K: float = 1000.0,
                             D_H_ref_m2s: float = 0.091,
                             D_H_ref_torr: float = 18.0,
                             recomb_coeff: float = 0.1) -> np.ndarray:
    '''
    Steady-state atomic hydrogen density profile n_H(z) [m^-3] on
    diag.cells, solving D_H * d2n_H/dz2 + C_H(z) = 0 with a recombination
    boundary condition dn_H/dz = recomb_coeff * n_H / D_H at both walls.
    Reproduces reactor_calc_paper.ipynb's solve_for_H_profile.

    Parameters
    ----------
    diag : picmi_diagnostics.analysis.Analysis
    pressure_torr : float
        Gas pressure, used to scale the H diffusivity from its reference
        value (D_H_ref_m2s at D_H_ref_torr).
    reactions : dict, default=None
        Passed to h_production_rate_profile.
    gas_temp_K : float, default=1000.0
        Neutral gas temperature, used for the H mean thermal velocity in the
        boundary condition.
    D_H_ref_m2s, D_H_ref_torr : float
        Reference H-in-H2 diffusivity and the pressure it was measured at;
        D_H = D_H_ref_m2s * D_H_ref_torr / pressure_torr.
    recomb_coeff : float, default=0.1
        Wall recombination probability for H atoms.

    Returns
    -------
    np.ndarray
        n_H [m^-3] on diag.cells.
    '''
    C_H_total = h_production_rate_profile(diag, reactions)

    D_H = D_H_ref_m2s * D_H_ref_torr / pressure_torr
    N = len(diag.cells)
    dz = diag.dz
    v_m_H = np.sqrt(8 * gas_temp_K * K_B / (np.pi * M_P))

    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N - 1)
    laplacian = _diags_matrix([off_diag, main_diag, off_diag], [-1, 0, 1]).tolil()
    laplacian[0, 0] = 1 + recomb_coeff * v_m_H * dz / D_H / 4
    laplacian[0, 1] = -1
    laplacian[-1, -1] = 1 + recomb_coeff * v_m_H * dz / D_H / 4
    laplacian[-1, -2] = -1
    laplacian = laplacian.tocsr()

    rhs = -C_H_total * dz**2 / D_H
    rhs[0] = 0
    rhs[-1] = 0

    return spsolve(laplacian, rhs)


def h_radical_flux(diag,
                    pressure_torr: float,
                    wall: str = ANODE_WALL,
                    reactions: dict = None,
                    gas_temp_K: float = 1000.0,
                    D_H_ref_m2s: float = 0.091,
                    D_H_ref_torr: float = 18.0,
                    recomb_coeff: float = 0.1,
                    n_H_profile: np.ndarray = None) -> float:
    '''
    Effusive H-radical flux [m^-2 s^-1] at `wall`: n_H(wall) * v_mean / 4.

    Parameters
    ----------
    diag, pressure_torr, reactions, gas_temp_K, D_H_ref_m2s, D_H_ref_torr,
    recomb_coeff
        Passed to solve_h_density_profile if n_H_profile is not given.
    wall : str, default='lw'
        'lw' -> diag.cells[0] (z=0 boundary); 'rw' -> diag.cells[-1].
    n_H_profile : np.ndarray, default=None
        Precomputed result of solve_h_density_profile, to avoid re-solving
        the diffusion equation when checking both walls or scanning
        parameters.

    Returns
    -------
    float
    '''
    if n_H_profile is None:
        n_H_profile = solve_h_density_profile(
            diag, pressure_torr, reactions=reactions, gas_temp_K=gas_temp_K,
            D_H_ref_m2s=D_H_ref_m2s, D_H_ref_torr=D_H_ref_torr,
            recomb_coeff=recomb_coeff)
    v_m_H = np.sqrt(8 * gas_temp_K * K_B / (np.pi * M_P))
    idx = 0 if wall == 'lw' else -1
    return n_H_profile[idx] * v_m_H / 4


def summarize_anode_fluxes(diag,
                            pressure_torr: float,
                            undercount_factor: float = 2.0,
                            wall: str = ANODE_WALL,
                            dz_correction: bool = True,
                            reactions: dict = None) -> dict:
    '''
    One-call summary of ion, electron, and H-radical flux at `wall` (default:
    the grounded anode, 'lw').

    Parameters
    ----------
    diag : picmi_diagnostics.analysis.Analysis
    pressure_torr : float
        Gas pressure for the H-radical diffusion solve.
    undercount_factor : float, default=2.0
        See charged_species_flux. Applied to the ion/electron flux only.
    wall, dz_correction : see charged_species_flux.
    reactions : dict, default=None
        See h_production_rate_profile.

    Returns
    -------
    dict
        {'wall', 'undercount_factor', 'electron_flux_m2s',
         'ion_flux_m2s' (dict, one entry per non-electron species),
         'H_flux_m2s'}, all fluxes in m^-2 s^-1.
    '''
    electron_name = diag.species_names[0]
    charged = all_charged_species_fluxes(diag, wall=wall,
                                          undercount_factor=undercount_factor,
                                          dz_correction=dz_correction)
    h_flux = h_radical_flux(diag, pressure_torr, wall=wall, reactions=reactions)

    return {
        'wall': wall,
        'undercount_factor': undercount_factor,
        'electron_flux_m2s': charged.get(electron_name),
        'ion_flux_m2s': {s: f for s, f in charged.items() if s != electron_name},
        'H_flux_m2s': h_flux,
    }


def print_flux_summary(diag, summary: dict, label: str = None):
    '''
    Pretty-print the output of summarize_anode_fluxes.

    Parameters
    ----------
    diag : picmi_diagnostics.analysis.Analysis
    summary : dict
        Output of summarize_anode_fluxes.
    label : str, default=None
        Optional case label to print above the summary. If None, uses
        diag.directory.
    '''
    print(f'--- {label or diag.directory} ---')
    print(f"  Wall: {summary['wall']} (undercount_factor={summary['undercount_factor']})")
    print(f"  Electron flux: {summary['electron_flux_m2s']:.3e} m^-2 s^-1")
    for species, flux in summary['ion_flux_m2s'].items():
        print(f'  Ion flux ({species}): {flux:.3e} m^-2 s^-1')
    print(f"  H flux: {summary['H_flux_m2s']:.3e} m^-2 s^-1")


# --- comparison across many simulations ----------------------------------
#
# Mirrors the compare_*()/plot_*_comparison() pattern in
# July_2026/power_balance/inductive/base_tests/picmi_diagnostics/power_balance.py:
# a loader ("plug in a list of directories") that computes per-case results
# and hands them to a renderer, so the renderer can also be called directly
# on already-computed results (e.g. to re-plot without re-loading).

_ELECTRON_COLOR = '#2a78d6'
_H_COLOR = '#3fa34d'
_ION_PALETTE = ['#eb6834', '#c94f9c', '#8a5fd1', '#d1b400']
_SEGMENT_EDGE = '#fcfcfb'
_GRID_COLOR = '#e1e0d9'
_SPINE_COLOR = '#c3c2b7'


def compare_anode_fluxes(cases,
                          pressure_torr=None,
                          undercount_factor: float = 2.0,
                          wall: str = ANODE_WALL,
                          dz_correction: bool = True,
                          reactions: dict = None,
                          quiet_startup: bool = True,
                          **plot_kwargs):
    '''
    Compute and plot ion, electron, and H-radical anode flux across several
    simulations. This is the "plug in a list of directories" entry point: it
    loads each simulation, computes summarize_anode_fluxes(), and hands the
    results to plot_anode_flux_comparison().

    Parameters
    ----------
    cases : list[str] or list[dict]
        One entry per simulation, in the order they should appear on the
        x-axis. Either a plain list of diagnostics directories, e.g.
        ``['dc_base_1/diags_p28_V300', 'f30_p28_1/diags_V300_RF50', ...]``,
        or a list of dicts for per-case control, each accepting:

        - ``'directory'`` : str -- diagnostics directory to load.
          Alternatively pass an already-constructed ``'diag'`` (an
          ``Analysis`` instance) to skip loading, e.g. to reuse a
          diagnostics object across multiple analyses.
        - ``'label'`` : str, optional -- name shown on the x-axis (default:
          the directory string).
        - ``'pressure_torr'`` : float, optional -- overrides `pressure_torr`
          for this case only (e.g. when mixing 18 and 28 Torr cases).
    pressure_torr : float or list[float], optional
        Gas pressure(s) [Torr] for the H-radical diffusion solve. Either a
        single value applied to every case, or a list with one entry per
        case (matching `cases`' order). A case dict's own ``'pressure_torr'``
        (if given) takes precedence. Required, one way or another, for every
        case.
    undercount_factor, wall, dz_correction, reactions
        Passed to summarize_anode_fluxes for every case (unless a case dict
        overrides them individually -- not currently supported per-case
        except for 'pressure_torr').
    quiet_startup : bool, default=True
        Passed to Analysis() when constructing diagnostics from a directory.
    **plot_kwargs
        Passed through to plot_anode_flux_comparison (e.g. ``axes``,
        ``dpi``, ``figsize``, ``labels``, ``yscale``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of 3 matplotlib.axes.Axes
        [H, electron, ion] panels.
    results : list[dict]
        The summarize_anode_fluxes() dict for each case, with a ``'label'``
        key added -- handy for tabulating or re-plotting later via
        plot_anode_flux_comparison(results).
    '''
    normalized_cases = [{'directory': case} if isinstance(case, str) else dict(case)
                         for case in cases]

    if pressure_torr is not None and isinstance(pressure_torr, (list, tuple, np.ndarray)):
        if len(pressure_torr) != len(normalized_cases):
            raise ValueError(
                f'pressure_torr has length {len(pressure_torr)} but {len(normalized_cases)} cases were given')
        pressure_list = list(pressure_torr)
    else:
        pressure_list = [pressure_torr] * len(normalized_cases)

    results = []
    for case, default_pressure in zip(normalized_cases, pressure_list):
        diag = case.get('diag')
        if diag is None:
            diag = Analysis(case['directory'], quiet_startup=quiet_startup)
        case_pressure = case.get('pressure_torr', default_pressure)
        if case_pressure is None:
            raise ValueError(
                f"No pressure_torr given for case {case.get('directory', case.get('label', '?'))!r}; "
                "pass pressure_torr to compare_anode_fluxes, or give this case its own 'pressure_torr'.")

        summary = summarize_anode_fluxes(
            diag, pressure_torr=case_pressure, undercount_factor=undercount_factor,
            wall=wall, dz_correction=dz_correction, reactions=reactions)
        summary['label'] = case.get('label') or case.get('directory', diag.directory)
        results.append(summary)

    fig, axes = plot_anode_flux_comparison(results, **plot_kwargs)
    return fig, axes, results


def plot_anode_flux_comparison(results,
                                axes=None,
                                dpi: int = 130,
                                figsize=None,
                                labels=None,
                                yscale: str = 'linear'):
    '''
    Three-panel bar chart comparing H-radical, electron, and ion anode flux
    across simulations.

    Parameters
    ----------
    results : list[dict]
        Per-simulation flux dicts, as returned by summarize_anode_fluxes()
        (with a ``'label'`` key added -- see compare_anode_fluxes()).
    axes : array-like of 3 matplotlib.axes.Axes, optional
        Axes to plot [H, electron, ion] on. If None, creates a new 1x3 figure.
    dpi, figsize
        Passed to plt.subplots() when `axes` is None.
    labels : list[str], optional
        Overrides each result's own ``'label'`` for the x-tick text (must be
        the same length as `results`).
    yscale : str, default='linear'
        Passed to each panel's set_yscale (e.g. 'log'). A case with exactly
        zero flux (see module docstring -- some directories genuinely have
        no recorded ion flux at the anode) simply draws no visible bar under
        'log', rather than raising.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of 3 matplotlib.axes.Axes
    '''
    import matplotlib.pyplot as plt

    n = len(results)
    x = np.arange(n)
    tick_labels = labels if labels is not None else [r.get('label', str(idx)) for idx, r in enumerate(results)]
    if len(tick_labels) != n:
        raise ValueError(f'labels must have length {n} (one per case), got {len(tick_labels)}')

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize or (4.6 * 3, 5.0), dpi=dpi)
    else:
        fig = axes[0].figure

    ax_H, ax_e, ax_i = axes

    H_flux = np.array([r['H_flux_m2s'] for r in results])
    ax_H.bar(x, H_flux, color=_H_COLOR, edgecolor=_SEGMENT_EDGE, linewidth=0.6)
    ax_H.set_title('H Radical Flux')

    e_flux = np.array([r['electron_flux_m2s'] for r in results])
    ax_e.bar(x, e_flux, color=_ELECTRON_COLOR, edgecolor=_SEGMENT_EDGE, linewidth=0.6)
    ax_e.set_title('Electron Flux')

    ion_species = sorted({species for r in results for species in r['ion_flux_m2s']})
    ion_colors = {species: _ION_PALETTE[idx % len(_ION_PALETTE)] for idx, species in enumerate(ion_species)}
    bottom = np.zeros(n)
    for species in ion_species:
        heights = np.array([r['ion_flux_m2s'].get(species, 0.0) for r in results])
        ax_i.bar(x, heights, bottom=bottom, color=ion_colors[species],
                  edgecolor=_SEGMENT_EDGE, linewidth=0.6, label=species)
        bottom += heights
    ax_i.set_title('Ion Flux')
    if len(ion_species) > 1:
        ax_i.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=30, ha='right')
        ax.set_ylabel('Flux [m$^{-2}$ s$^{-1}$]')
        ax.set_yscale(yscale)
        ax.grid(axis='y', color=_GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(_SPINE_COLOR)
        ax.spines['bottom'].set_color(_SPINE_COLOR)
        ax.margins(x=0.05)

    fig.tight_layout()
    return fig, axes
