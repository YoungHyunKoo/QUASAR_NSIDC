"""
dynamic_tiepoints.py
====================

Daily (dynamic) tie points for passive-microwave sea ice concentration, plus a
gradient-ratio weather filter. Designed as an add-on to sicci_ic_algs.py.

Idea
----
1. Compute an *initial* SIC from the daily-mean TBs with the fixed (ATBD) tie
   points.
2. Use it to pick two populations of grid cells from the same day:
     open water (OW) : ocean cells with no land in the footprint, initial SIC
                       below 5 %, inside a band 100-500 km from the ice edge
                       (cold water near the ice; strongly weather-affected
                       cells are excluded by the low initial SIC)
     consolidated ice (CI) : cells with initial NASA Team SIC above 95 %
3. OW tie point = median TB of the OW cells, per channel.
   Ice tie points = the two ends of the CI cloud along its main axis
   (first principal component of 18V/18H/36V/36H). The end with the higher
   gradient ratio GR(36V/18V) is labelled first-year (type A in the south),
   the other multiyear (type B in the south).
4. These tie points replace the fixed ones in the 36-value list used by
   sicci_ic_algs.py, so every algorithm there (NASA Team, Bootstrap_f/p,
   Bristol, OSI SAF, ...) uses them without changes to its formula.
   The same daily tie points are then applied to every swath of that day.

If a population has too few cells, or the result fails a sanity check, the
fixed tie points are kept for that surface type and a note is returned.

Tie-point index convention (same as sicci_ic_algs.py)
    [0]  w37v   [1]  my37v  [2]  fy37v  [3]  w37h   [4]  my37h  [5]  fy37h
    [6]  w18v   [7]  my18v  [8]  fy18v  [9]  w18h   [10] my18h  [11] fy18h
    [30] my85v  [31] fy85v  [32] my85h  [33] fy85h  [34] w85v   [35] w85h
"""

import warnings
import numpy as np

# tie-point list positions for each gridded channel
TP_INDEX = {
    'tb36v': {'w': 0,  'my': 1,  'fy': 2},
    'tb36h': {'w': 3,  'my': 4,  'fy': 5},
    'tb18v': {'w': 6,  'my': 7,  'fy': 8},
    'tb18h': {'w': 9,  'my': 10, 'fy': 11},
    'tb89v': {'w': 34, 'my': 30, 'fy': 31},
    'tb89h': {'w': 35, 'my': 32, 'fy': 33},
}

LOW_FREQ = ('tb18v', 'tb18h', 'tb36v', 'tb36h')

DEFAULTS = dict(
    ow_sic_max=0.05,         # initial SIC (fraction) below this -> OW candidate
    ow_alg='bootstrap_f',    # algorithm used to select OW cells
    ci_sic_min=0.95,         # initial SIC above this -> CI candidate
    ci_alg='nasa_team',      # algorithm used to select CI cells (as in OSI SAF)
    ice_edge_sic=0.15,       # ice edge for the OW distance band
    ow_dist_min_km=100.0,    # OW cells at least this far from the ice edge ...
    ow_dist_max_km=500.0,    # ... and at most this far
    ci_end_fraction=0.10,    # fraction of CI cells at each end of the main axis
    min_cells=100,           # minimum cells per population
    min_ice_separation_K=3.0,  # minimum distance between the two ice tie points
    min_ice_water_18v_K=20.0,  # ice must be this much warmer than water at 18.7V
    update_89=True,          # also update the 89 GHz tie points
)


def gradient_ratio(tb_hi, tb_lo):
    """GR = (TB_hi - TB_lo) / (TB_hi + TB_lo)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return (tb_hi - tb_lo) / (tb_hi + tb_lo)


def _distance_to_ice_km(ice, cell_km):
    """Distance (km) from each cell to the nearest ice cell."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        warnings.warn('scipy not available: OW distance band is not applied')
        return None
    if not ice.any():
        return None
    return distance_transform_edt(~ice) * cell_km


def daily_tiepoints(tb, fixed_tiepts, compute_sic, hemisphere, sensor='AMSR2',
                    land_fraction=None, cell_km=12.5, **options):
    """
    Determine daily tie points from daily-mean TBs.

    tb            : dict 'tb18v', 'tb18h', 'tb36v', 'tb36h' (+ optional 'tb89v',
                    'tb89h'), 2D arrays in K, NaN = missing
    fixed_tiepts  : 36-value fixed tie-point list (e.g. TIEPOINTS[sensor][hemisphere])
    compute_sic   : sicci_ic_algs.compute_sic
    land_fraction : 2D land fraction in % (NaN = unknown); cells with any land
                    are excluded. None = no land screening.
    cell_km       : grid spacing of `tb` (for the OW distance band)

    Returns (tiepts, info): a new 36-value list and a dict with diagnostics.
    """
    opt = dict(DEFAULTS, **options)
    tp = [float(v) for v in fixed_tiepts]
    info = {'notes': []}

    missing = [c for c in LOW_FREQ if c not in tb]
    if missing:
        info['notes'].append('missing channels %s: fixed tie points kept' % missing)
        return tp, info

    channels = [c for c in TP_INDEX if c in tb]
    if not opt['update_89']:
        channels = [c for c in channels if c not in ('tb89v', 'tb89h')]

    # cells with all low-frequency TBs and no land in the footprint
    good = np.all([np.isfinite(tb[c]) for c in LOW_FREQ], axis=0)
    if land_fraction is not None:
        good &= np.isfinite(land_fraction) & (land_fraction == 0)

    # 1. initial SIC with the fixed tie points
    sic0 = compute_sic(tb, tp, hemisphere, sensor)
    for key in (opt['ow_alg'], opt['ci_alg']):
        if key not in sic0:
            raise KeyError("compute_sic returned no '%s' (available: %s)"
                           % (key, sorted(sic0)))
    c_ow = np.asarray(sic0[opt['ow_alg']], dtype=float)
    c_ci = np.asarray(sic0[opt['ci_alg']], dtype=float)

    # 2a. open water population
    ow = good & (c_ow < opt['ow_sic_max'])
    dist = _distance_to_ice_km(good & (c_ci >= opt['ice_edge_sic']), cell_km)
    if dist is not None:
        ow &= (dist >= opt['ow_dist_min_km']) & (dist <= opt['ow_dist_max_km'])
    info['n_ow'] = int(ow.sum())

    # 2b. consolidated ice population
    ci = good & (c_ci > opt['ci_sic_min'])
    info['n_ci'] = int(ci.sum())

    # 3a. open-water tie points: median per channel
    if info['n_ow'] >= opt['min_cells']:
        for c in channels:
            vals = tb[c][ow]
            vals = vals[np.isfinite(vals)]
            if vals.size >= opt['min_cells']:
                tp[TP_INDEX[c]['w']] = float(np.median(vals))
    else:
        info['notes'].append('too few OW cells (%d): fixed OW tie points kept' % info['n_ow'])

    # 3b. ice tie points: ends of the CI cloud along its first principal axis
    if info['n_ci'] >= opt['min_cells']:
        X = np.column_stack([tb[c][ci] for c in LOW_FREQ])
        Xc = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ vt[0]
        lo, hi = np.quantile(proj, [opt['ci_end_fraction'], 1 - opt['ci_end_fraction']])
        ends = []
        for sel in (proj <= lo, proj >= hi):
            ends.append({c: float(np.nanmean(tb[c][ci][sel])) for c in channels})

        gr_end = [gradient_ratio(e['tb36v'], e['tb18v']) for e in ends]
        fy, my = (ends[0], ends[1]) if gr_end[0] >= gr_end[1] else (ends[1], ends[0])

        # Sanity checks.
        # (1) the two ends must be clearly apart (distance in 18V/18H/36V/36H),
        #     otherwise the ice line / NASA Team system is ill-defined;
        # (2) ice must be clearly warmer than open water at 18.7V.
        # 36.5V is NOT tested: in the Antarctic, type-B ice at 36.5V is only a
        # few K warmer than open water (or colder), which is physically normal.
        sep = float(np.sqrt(sum((fy[c] - my[c])**2 for c in LOW_FREQ)))
        margin18 = min(fy['tb18v'], my['tb18v']) - tp[TP_INDEX['tb18v']['w']]
        info['ice_end_separation_K'] = sep
        if sep >= opt['min_ice_separation_K'] and margin18 >= opt['min_ice_water_18v_K']:
            for c in channels:
                tp[TP_INDEX[c]['fy']] = fy[c]
                tp[TP_INDEX[c]['my']] = my[c]
        else:
            info['notes'].append(
                'degenerate ice cloud (ends %.1f K apart, need >= %.1f; ice-water 18V '
                '%.1f K, need >= %.1f): fixed ice tie points kept'
                % (sep, opt['min_ice_separation_K'], margin18, opt['min_ice_water_18v_K']))
    else:
        info['notes'].append('too few CI cells (%d): fixed ice tie points kept' % info['n_ci'])

    return tp, info


def weather_filter(sic, tb, gr3719_max=0.05, gr2219_max=0.045, algorithms=None):
    """
    Gradient-ratio weather filter (NASA Team style). Calm open water already
    has a high GR(36V/18V), while ice has GR near or below 0, so cells above the
    threshold are treated as open water and spurious 'ice' caused by
    atmospheric water vapour, cloud liquid water or wind is removed.
    Set SIC to 0 where
        GR(36.5V/18.7V) > gr3719_max   [and, if 'tb23v' is in tb,
        GR(23.8V/18.7V) > gr2219_max]

    sic        : dict name -> SIC fraction (output of compute_sic); modified copy returned
    tb         : dict of TBs for the same cells
    algorithms : names to filter (None = all)

    Returns (filtered_sic, filtered_mask): mask is True where SIC was set to 0.
    """
    gr37 = gradient_ratio(tb['tb36v'], tb['tb18v'])
    weather = gr37 > gr3719_max
    if 'tb23v' in tb:
        weather |= gradient_ratio(tb['tb23v'], tb['tb18v']) > gr2219_max
    weather &= np.isfinite(gr37)

    out = dict(sic)
    for name in (algorithms or list(sic)):
        if name in out:
            out[name] = np.where(weather, 0.0, out[name])
    return out, weather


def describe(tp, info=None):
    """Short text summary of the 18/36 GHz tie points (for logs)."""
    s = []
    for c in ('tb18v', 'tb18h', 'tb36v', 'tb36h', 'tb89v', 'tb89h'):
        i = TP_INDEX[c]
        s.append('%s  OW %6.1f  FY %6.1f  MY %6.1f' % (c, tp[i['w']], tp[i['fy']], tp[i['my']]))
    if info:
        s.append('cells: OW %s, CI %s' % (info.get('n_ow'), info.get('n_ci')))
        s += ['note: ' + n for n in info['notes']]
    return '\n'.join(s)
