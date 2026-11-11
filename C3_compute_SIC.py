#!/usr/bin/env python3
"""
Sea ice concentration (SIC) algorithms from the ESA SICCI-P2 ATBD
(D2.1, SICCI-P2-ATBD(SIC), Issue 1.0, 22 September 2017), Section 5.4.1,
revised to run on the daily gridded AMSR2 files written by
amsr2_l1r_daily_grid.py (AMSR2_L1R_TB_PS_<N|S>_<YYYYMMDD>.nc).

Changes from the ATBD code
--------------------------
* All algorithms are vectorised: they accept NumPy arrays (whole grids) as
  well as scalars. Missing TBs (NaN) simply propagate to NaN SIC.
* Algorithms that need 6.9 GHz or 10.7 GHz (One_channel, P10) are removed.
  All remaining algorithms use only 18.7, 36.5 and 89 GHz.
* The RRDP text-file driver is replaced by a NetCDF reader/writer.
* The degenerate-geometry sentinel -9.98 in Bootstrap_p is returned as NaN.
* Tie points are determined daily from the daily-mean TBs (dynamic_tiepoints.py)
  and applied to the daily SIC and to every swath of that day. Use
  --fixed-tiepoints for the fixed ATBD tie points.
* A GR(36.5V/18.7V) > 0.05 weather filter sets SIC to 0 (--no-weather-filter
  to switch it off). Not applied on the 89-GHz-only 6.25 km grid.

Which SIC is computed on which grid
-----------------------------------
    6.25 km : 89 GHz algorithms  (ASI, Near90GHz, Near90_lin_dyn, P90)
    12.5 km : 18.7/36.5 GHz algorithms
              + 89 GHz and hybrid algorithms, using 89 GHz TBs block-averaged
                from the 6.25 km grid (2x2 cells)
    25 km   : same as 12.5 km, with 89 GHz block-averaged 4x4
The 89 GHz upscaling can be switched off with --no-89-upscale.

The grids nest exactly (same NSIDC extents), so a 12.5 km cell is exactly
2x2 cells of 6.25 km. The block average is the mean of the valid fine cells.

Output
------
One NetCDF-4 file per input file, with the same groups as the input
(grid_6p25km, grid_12p5km, grid_25km). SIC is stored in percent, NOT clipped
to 0-100 %, as in the ATBD/RRDP evaluation (values slightly below 0 % or above
100 % are expected and informative for algorithm comparison).

    --mode daily   SIC from the *_daily_mean TBs -> sic_<alg>(y, x)
    --mode swath   SIC for each swath           -> sic_<alg>(swath, y, x)
    --mode both    both: sic_<alg>_daily(y, x) and sic_<alg>_swath(swath, y, x)

Usage
-----
    python C3_compute_SIC.py /data/daily/AMSR2_L1R_TB_PS_N_20171112.nc --output-dir /data/sic
    python C3_compute_SIC.py "/data/daily/AMSR2_L1R_TB_PS_*.nc" --mode both

The algorithm functions can also be imported and used on your own arrays:
    from sicci_ic_algs import bootstrap_f, TIEPOINTS

Tie-point index convention (tiepts list, 36 values):
    [0]  w37v   [1]  my37v  [2]  fy37v  [3]  w37h   [4]  my37h  [5]  fy37h
    [6]  w18v   [7]  my18v  [8]  fy18v  [9]  w18h   [10] my18h  [11] fy18h
    [12] w6v    [13] my6v   [14] fy6v   [15] w6h    [16] my6h   [17] fy6h
    [18] w10v   [19] my10v  [20] fy10v  [21] w10h   [22] my10h  [23] fy10h
    [24] w22v   [25] my22v  [26] fy22v  [27] w22h   [28] my22h  [29] fy22h
    [30] my85v  [31] fy85v  [32] my85h  [33] fy85h  [34] w85v   [35] w85h
(The 6, 10 and 22 GHz entries are unused here but kept so the indices match
the ATBD.)
"""

import argparse
import glob
import os
import warnings
from datetime import datetime, timezone

import netCDF4
import numpy as np
from dynamic_tiepoints import daily_tiepoints, weather_filter, describe, LOW_FREQ

# ---------------------------------------------------------------------------
# Tie points (from the ATBD code)
# ---------------------------------------------------------------------------

TIEPOINTS = {
    # .xls files in the e-mail from Leif 16 Oct 2015 21:54:15 CEST,
    # Subject: RE: AMSR2 TPs for RRDP2_v0.9
    'AMSR2': {
        'north': [215.71, 191.70, 254.91, 152.80, 178.15, 241.86,
                  190.71, 227.11, 260.96, 114.08, 204.34, 244.51,
                  162.68, 250.07, 259.51, 82.76, 224.60, 240.67,
                  171.29, 245.54, 261.26, 90.29, 219.95, 244.00,
                  207.78, 213.99, 260.24, 145.43, 195.45, 246.14,
                  191.37, 238.09, 180.97, 228.58, 249.23, 210.55],
        'south': [215.23, 219.68, 251.23, 153.39, 197.66, 232.68,
                  190.03, 244.08, 260.73, 114.11, 212.37, 239.19,
                  161.52, 256.38, 260.58, 83.08, 225.74, 238.20,
                  170.67, 254.78, 262.38, 91.06, 223.55, 241.31,
                  205.70, 236.81, 259.00, 142.84, 208.80, 239.51,
                  211.59, 241.11, 200.12, 229.20, 246.66, 207.92],
    },
    # AMSR-E: water from Tiepoints_20130117.xlsx (N) / Tiepoints_latest_S.xlsx (S),
    # ice from hard copy 8 Apr 2013, 2007-2011.
    'AMSRE': {
        'north': [209.81, 196.91, 247.13, 145.29, 184.94, 235.01,
                  183.72, 226.26, 252.15, 108.46, 207.78, 237.54,
                  161.35, 246.04, 251.99, 82.13, 221.19, 232.08,
                  167.34, 239.61, 251.34, 88.26, 216.31, 234.01,
                  196.41, 216.67, 250.87, 128.23, 199.6, 236.72,
                  187.6, 232.01, 178.9, 222.39, 243.2, 196.94],
        'south': [212.57, 226.51, 253.84, 149.07, 204.66, 239.96,
                  185.34, 246.1, 258.58, 110.83, 217.65, 242.8,
                  159.69, 254.18, 257.04, 80.15, 225.37, 236.52,
                  166.31, 251.65, 257.23, 86.62, 221.47, 238.5,
                  201.53, 240.65, 257.56, 137.19, 213.79, 242.61,
                  210.22, 242.81, 197.78, 232.4, 247.59, 207.2],
    },
}


def norsex_TPs(sensor_name, hemisphere):
    """NORSEX tie points (values produced by Leif, described in PVASR).

    Returns TB_w_19v, TB_w_37v, TB_fy_19v, TB_fy_37v, TB_my_19v, TB_my_37v.
    """
    tps = {
        'north': {
            'AMSR':  (170.01, 193.19, 251.17, 244.47, 222.11, 184.02),
            'SSMI':  (171.56, 191.87, 251.91, 241.53, 219.20, 175.93),
            'SMMR':  (162.61, 190.80, 251.17, 244.47, 222.11, 184.02),
        },
        'south': {
            'AMSR':  (171.86, 196.65, 258.41, 252.57, 244.39, 219.62),
            'SSMI':  (171.52, 192.94, 259.93, 253.25, 244.59, 219.59),
            'SMMR':  (160.77, 190.92, 258.41, 252.57, 244.39, 219.62),
        },
    }
    key = 'AMSR' if sensor_name in ('AMSRE', 'AMSR2') else sensor_name
    return tps[hemisphere][key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cubic_coefficients(P0, P1):
    """Coefficients d of C = d0*P^3 + d1*P^2 + d2*P + d3 used by ASI/Near90.

    Solves the same 4x4 system as the ATBD code (np.matrix A.I * b.T).
    """
    A = np.array([[P1**3.0, P1**2.0, P1, 1.0],
                  [P0**3.0, P0**2.0, P0, 1.0],
                  [3.0 * P1**3.0, 2.0 * P1**2.0, P1, 0.0],
                  [3.0 * P0**3.0, 2.0 * P0**2.0, P0, 0.0]])
    b = np.array([1.0, 0.0, -0.14, -1.14])
    return np.linalg.solve(A, b)


def _safe_div(num, den):
    """Division returning NaN where the denominator is zero."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den == 0, np.nan, num / np.where(den == 0, 1.0, den))


# ---------------------------------------------------------------------------
# Near-90 GHz algorithms
# ---------------------------------------------------------------------------

def asi(tb85v, tb85h):
    # ASI. The coefficients are checked with Christian Melsheimer (e-mail to
    # Leif, May 24, 2013 17:50). Same values for N and S, and AMSR and SSM/I.
    # d = [1.64e-5, -0.0016, 0.0192, 0.971]
    d = _cubic_coefficients(P0=47.0, P1=11.7)
    P = tb85v - tb85h
    return d[0] * P**3 + d[1] * P**2 + d[2] * P + d[3]


def near90(tb85v, tb85h, tiepts):
    P0 = tiepts[34] - tiepts[35]   # w85v - w85h
    P1 = tiepts[31] - tiepts[33]   # fy85v - fy85h
    d = _cubic_coefficients(P0, P1)
    P = tb85v - tb85h
    return d[0] * P**3 + d[1] * P**2 + d[2] * P + d[3]


def near90_linear_dyn(tb85v, tb85h, tiepts):
    tmy85v, tfy85v, tmy85h, tfy85h, tw85v, tw85h = tiepts[30:36]
    PFY = tfy85v - tfy85h
    PMY = tmy85v - tmy85h
    PW = tw85v - tw85h
    PI = (PFY + PMY) / 2
    P = tb85v - tb85h
    return (P - PW) / (PI - PW)


def P90(tb85v, tb85h):
    X = tb85v - tb85h
    P = (X - 2.63) / 0.752
    d3, d2, d1, d0 = 1.64 / 100000.0, -0.0016, 0.0192, 0.971
    c1 = d3 * P**3.0 + d2 * P**2.0 + d1 * P + d0
    c = c1 + (P - 8) / 700           # to adjust near SIC0
    c = np.where(P > 48, -0.026, c)  # to prevent large P85 giving ice
    c = np.where(P < 8.5, 1.03, c)   # to prevent low P85 losing ice
    return np.where(np.isnan(P), np.nan, c)


# ---------------------------------------------------------------------------
# 18.7 / 36.5 GHz algorithms
# ---------------------------------------------------------------------------

def bootstrap_f(tb18v, tb37v, tiepts):
    tw37v, tmy37v, tfy37v = tiepts[0], tiepts[1], tiepts[2]
    tw18v, tmy18v, tfy18v = tiepts[6], tiepts[7], tiepts[8]

    af = (tfy37v - tmy37v) / (tfy18v - tmy18v)
    bf = tmy37v - af * tmy18v
    qf = _safe_div(tb37v - tw37v, tb18v - tw18v)
    wf = tw37v - qf * tw18v
    ti18vf = _safe_div(bf - wf, qf - af)
    cf = _safe_div(tb18v - tw18v, ti18vf - tw18v)
    # open water: TB equal to the water tie point
    return np.where(tb18v - tw18v == 0, 0.0, cf)


def bootstrap_p(tb37v, tb37h, tiepts):
    tw37v, tmy37v, tfy37v = tiepts[0], tiepts[1], tiepts[2]
    tw37h, tmy37h, tfy37h = tiepts[3], tiepts[4], tiepts[5]

    ap = (tfy37v - tmy37v) / (tfy37h - tmy37h)
    bp = tmy37v - ap * tmy37h
    qp = _safe_div(tb37v - tw37v, tb37h - tw37h)
    wp = tw37v - qp * tw37h
    ti37hp = _safe_div(bp - wp, qp - ap)        # NaN where ATBD returns -9.98
    ti37vp = ap * ti37hp + bp
    cp = _safe_div(tb37v - tw37v, ti37vp - tw37v)
    return np.where(tb37h - tw37h == 0, 0.0, cp)


def bristol(tb18v, tb37v, tb37h, tiepts):
    # Bristol ice concentration algorithm
    tw37v, tmy37v, tfy37v = tiepts[0], tiepts[1], tiepts[2]
    tw37h, tmy37h, tfy37h = tiepts[3], tiepts[4], tiepts[5]
    tw18v, tmy18v, tfy18v = tiepts[6], tiepts[7], tiepts[8]

    xa = tmy37v + (1.045 * tmy37h) + (0.525 * tmy18v)
    xd = tfy37v + (1.045 * tfy37h) + (0.525 * tfy18v)
    xh = tw37v + (1.045 * tw37h) + (0.525 * tw18v)
    xt = tb37v + (1.045 * tb37h) + (0.525 * tb18v)

    ya = (0.9164 * tmy18v) - tmy37v + (0.4965 * tmy37h)
    yd = (0.9164 * tfy18v) - tfy37v + (0.4965 * tfy37h)
    yh = (0.9164 * tw18v) - tw37v + (0.4965 * tw37h)
    yt = (0.9164 * tb18v) - tb37v + (0.4965 * tb37h)

    a_ht = _safe_div(yt - yh, xt - xh)
    b_ht = yh - (a_ht * xh)
    a_da = (ya - yd) / (xa - xd)
    b_da = yd - (a_da * xd)

    xi = _safe_div(b_da - b_ht, a_ht - a_da)
    return _safe_div(xt - xh, xi - xh)


def calval(tb37v, tb18v, tiepts):
    tw37v, tmy37v, tfy37v = tiepts[0], tiepts[1], tiepts[2]
    tw18v, tmy18v, tfy18v = tiepts[6], tiepts[7], tiepts[8]
    A = np.array([[tw37v, tw18v, 1.0],
                  [tfy37v, tfy18v, 1.0],
                  [tmy37v, tmy18v, 1.0]])
    b = np.array([0.0, 1.0, 1.0])
    d = np.linalg.solve(A, b)
    return d[0] * tb37v + d[1] * tb18v + d[2]


def nasa(tb18v, tb18h, tb37v, tiepts):
    # NASA Team ice concentration algorithm. Returns (total, multiyear).
    ow37v, my37v, fy37v = tiepts[0], tiepts[1], tiepts[2]
    ow18v, my18v, fy18v = tiepts[6], tiepts[7], tiepts[8]
    ow18h, my18h, fy18h = tiepts[9], tiepts[10], tiepts[11]

    a0 = - ow18v + ow18h
    a1 = ow18v + ow18h
    a2 = my18v - my18h - ow18v + ow18h
    a3 = - my18v - my18h + ow18v + ow18h
    a4 = fy18v - fy18h - ow18v + ow18h
    a5 = - fy18v - fy18h + ow18v + ow18h

    b0 = - ow37v + ow18v
    b1 = ow37v + ow18v
    b2 = my37v - my18v - ow37v + ow18v
    b3 = - my37v - my18v + ow37v + ow18v
    b4 = fy37v - fy18v - ow37v + ow18v
    b5 = - fy37v - fy18v + ow37v + ow18v

    gr = _safe_div(tb37v - tb18v, tb37v + tb18v)
    pr_ = _safe_div(tb18v - tb18h, tb18v + tb18h)

    d0 = (-a2 * b4) + (a4 * b2)
    d1 = (-a3 * b4) + (a5 * b2)
    d2 = (-a2 * b5) + (a4 * b3)
    d3 = (-a3 * b5) + (a5 * b3)
    dd = d0 + d1 * pr_ + d2 * gr + d3 * pr_ * gr

    f0 = (a0 * b2) - (a2 * b0)
    f1 = (a1 * b2) - (a3 * b0)
    f2 = (a0 * b3) - (a2 * b1)
    f3 = (a1 * b3) - (a3 * b1)
    m0 = (-a0 * b4) + (a4 * b0)
    m1 = (-a1 * b4) + (a5 * b0)
    m2 = (-a0 * b5) + (a4 * b1)
    m3 = (-a1 * b5) + (a5 * b1)

    cf = _safe_div(f0 + f1 * pr_ + f2 * gr + f3 * pr_ * gr, dd)
    cm = _safe_div(m0 + m1 * pr_ + m2 * gr + m3 * pr_ * gr, dd)
    return cm + cf, cm


def norsex(tb18v, tb37v, sensor_name, hemisphere):
    SAT = 260.0
    T_sa = 270.0
    T_a = 250.0
    To = 272.0

    tau_sa19v = 0.0610
    tau_sa37v = 0.1000
    tau_a19v = 0.0440
    tau_a37v = 0.0700
    (TB_w_19v, TB_w_37v, TB_fy_19v, TB_fy_37v,
     TB_my_19v, TB_my_37v) = norsex_TPs(sensor_name, hemisphere)

    # Constants to be used in computing ice concentrations:
    a11 = TB_fy_19v - TB_w_19v
    a21 = TB_fy_37v - TB_w_37v
    a12 = TB_my_19v - TB_w_19v
    a22 = TB_my_37v - TB_w_37v
    d_coef = a11 * a22 - a12 * a21

    t_atm_surf = SAT  # Initialize atmospheric surface temperature
    for _ in range(2):
        # Interpolate opacity between arctic and subarctic values:
        tau19v = tau_a19v + (t_atm_surf - T_a) * (tau_sa19v - tau_a19v) / (T_sa - T_a)
        tau37v = tau_a37v + (t_atm_surf - T_a) * (tau_sa37v - tau_a37v) / (T_sa - T_a)

        # Emitted brightness temperature at the surface, corrected for the atmosphere:
        TB_surf_19v = ((tb18v - t_atm_surf * (2.0 * tau19v - tau19v**2.0 + 0.01))
                       / (1.0 - 2.0 * tau19v + tau19v**2.0 - 0.01))
        TB_surf_37v = ((tb37v - t_atm_surf * (2.0 * tau37v - tau37v**2.0 + 0.01))
                       / (1.0 - 2.0 * tau37v + tau37v**2.0 - 0.01))

        # Solve for first-year and multi-year ice concentrations, then update
        # the atmospheric surface temperature.
        c1 = TB_surf_19v - TB_w_19v
        c2 = TB_surf_37v - TB_w_37v
        Cmy = (a11 * c2 - a21 * c1) / d_coef
        Cfy = (a22 * c1 - a12 * c2) / d_coef
        CT = Cfy + Cmy

        t_atm_surf = To + (SAT - To) * CT

    return CT


def P37(tb37v, tb37h, tiepts):
    # Instead of NRL
    PW = tiepts[0] - tiepts[3]
    PMY = tiepts[1] - tiepts[4]
    PFY = tiepts[2] - tiepts[5]
    PI = (PFY + PMY) / 2
    return (tb37v - tb37h - PW) / (PI - PW)


def P18(tb18v, tb18h, tiepts):
    PW = tiepts[6] - tiepts[9]
    PMY = tiepts[7] - tiepts[10]
    PFY = tiepts[8] - tiepts[11]
    PI = (PFY + PMY) / 2
    return (tb18v - tb18h - PW) / (PI - PW)


def pr(tb18v, tb18h, tb37v, tb37h, tiepts):
    # Simple polarization ratio algorithm. Returns (c, PR18, PR37).
    ow37v, my37v, fy37v = tiepts[0], tiepts[1], tiepts[2]
    ow37h, my37h, fy37h = tiepts[3], tiepts[4], tiepts[5]
    ow18v, my18v, fy18v = tiepts[6], tiepts[7], tiepts[8]
    ow18h, my18h, fy18h = tiepts[9], tiepts[10], tiepts[11]

    i18v = (fy18v + my18v) / 2
    i18h = (fy18h + my18h) / 2
    i37v = (fy37v + my37v) / 2
    i37h = (fy37h + my37h) / 2

    PR18 = _safe_div(tb18v - tb18h, tb18v + tb18h)
    PR37 = _safe_div(tb37v - tb37h, tb37v + tb37h)

    c18 = _safe_div(ow18v * (1 - PR18) - ow18h * (1 + PR18),
                    PR18 * (i18v + i18h - ow18v - ow18h) - (i18v - i18h - ow18v + ow18h))
    c37 = _safe_div(ow37v * (1 - PR37) - ow37h * (1 + PR37),
                    PR37 * (i37v + i37h - ow37v - ow37h) - (i37v - i37h - ow37v + ow37h))
    c_old = (c18 + c37) / 2
    c = _safe_div(c_old, 2 - c_old)
    return c, PR18, PR37


def UMass(tb18v, tb37v, tiepts):
    # Based on C. T. Swift, L. S. Fedor, and R. O. Ramseier, "An Algorithm to
    # Measure Sea Ice Concentration With Microwave Radiometers," J. Geophys.
    # Res., 90(C1), 1087-1099, 1985. Solution of eqs. (11)-(12), using
    # brightness temperatures instead of emissivities (Rasmus 2012).
    tw37v, tmy37v, tfy37v = tiepts[0], tiepts[1], tiepts[2]
    tw19v, tmy19v, tfy19v = tiepts[6], tiepts[7], tiepts[8]

    a1 = (tfy19v - tb18v) / (tfy19v - tw19v)
    a2 = (tfy19v - tmy19v) / (tfy19v - tw19v)
    a3 = (tfy37v - tb37v) / (tfy37v - tmy37v)
    a4 = (tfy37v - tw37v) / (tfy37v - tmy37v)
    fw = (a1 - a2 * a3) / (1.0 - a2 * a4)

    Cmy = a3 - fw * a4
    Cfy = 1.0 - fw - Cmy
    return Cmy + Cfy


# ---------------------------------------------------------------------------
# Blended / hybrid algorithms
# ---------------------------------------------------------------------------

def osisaf(c0, c1, t):
    # Blend of Bootstrap_f (c0) and Bristol (c1), threshold t
    wc = (np.abs(t - c0) + t - c0) / (2.0 * t)
    c = c1 * (1.0 - wc) + wc * c0
    return np.where(c0 < 0, c0, c)


def sicci(c0, c1):
    # Bootstrap_f (c0) below 70 %, Bristol (c1) above 90 %, linear in between
    wCF = np.where(c0 < 0.7, 1.0,
                   np.where(c0 < 0.9, 1.0 - (c0 - 0.7) / (0.9 - 0.7), 0.0))
    wBR = 1.0 - wCF
    return c0 * wCF + c1 * wBR


def tud(c85, cf):
    # TUD ice concentration algorithm.
    # NOTE (kept as in the ATBD): cf is a fraction (0-1), so the test cf > 10
    # is never true and TUD returns cf (Bootstrap_f) unchanged. If cf was meant
    # in percent, the threshold would be cf > 0.1 for fractions.
    with np.errstate(invalid='ignore'):
        return np.where((c85 > 0) & (cf > 10), np.sqrt(cf * c85), cf)


# ---------------------------------------------------------------------------
# Algorithm selection
# ---------------------------------------------------------------------------

# name -> (description, channels used)
ALG_INFO = {
    'near90_lin_dyn': ('Near 90 GHz linear, dynamic tie points', '89V 89H'),
    'near90ghz':      ('Near 90 GHz (ASI form with tie points)', '89V 89H'),
    'asi':            ('ASI, fixed tie points (no weather correction)', '89V 89H'),
    'p90':            ('P90 polarisation difference', '89V 89H'),
    'p37':            ('P37 polarisation difference', '36V 36H'),
    'bootstrap_p':    ('Bootstrap polarisation mode', '36V 36H'),
    'p18':            ('P18 polarisation difference', '18V 18H'),
    'bristol':        ('Bristol', '18V 36V 36H'),
    'pr':             ('Polarisation ratio', '18V 18H 36V 36H'),
    'nasa_team':      ('NASA Team (total)', '18V 18H 36V'),
    'norsex':         ('NORSEX', '18V 36V'),
    'bootstrap_f':    ('Bootstrap frequency mode', '18V 36V'),
    'calval':         ('CalVal', '18V 36V'),
    'umass_aes':      ('UMass-AES (Swift et al. 1985)', '18V 36V'),
    'tud':            ('TUD (see note in code)', '18V 36V 89V 89H'),
    'bf_nt':          ('mean(Bootstrap_f, NASA Team)', '18V 18H 36V'),
    'bf_nt_n90ld':    ('mean(Bootstrap_f, NASA Team, Near90_lin_dyn)', '18V 18H 36V 89V 89H'),
    'p37_n90ld':      ('mean(P37, Near90_lin_dyn)', '36V 36H 89V 89H'),
    'p37_n90ld_bf':   ('mean(P37, Near90_lin_dyn, Bootstrap_f)', '18V 36V 36H 89V 89H'),
    'bf2_n90ld':      ('Bootstrap_f/Near90_lin_dyn blend, weight bf^2', '18V 36V 89V 89H'),
    'bf3_n90ld':      ('Bootstrap_f/Near90_lin_dyn blend, weight bf^3', '18V 36V 89V 89H'),
    'bf_n90ld':       ('mean(Bootstrap_f, Near90_lin_dyn)', '18V 36V 89V 89H'),
    'bf_bfn90ld':     ('Bootstrap_f/Near90_lin_dyn blend, weight bf', '18V 36V 89V 89H'),
    'osisaf':         ('OSI SAF blend of Bootstrap_f and Bristol', '18V 36V 36H'),
    'sicci':          ('SICCI blend of Bootstrap_f and Bristol', '18V 36V 36H'),
}


def compute_sic(tb, tiepts, hemisphere, sensor='AMSR2'):
    """
    Compute all SIC algorithms possible with the channels in `tb`.

    tb : dict with any of 'tb18v', 'tb18h', 'tb36v', 'tb36h', 'tb89v', 'tb89h'
         (arrays in K, NaN = missing, all the same shape).
    Returns dict name -> SIC as a fraction (1.0 = 100 %).
    """
    have_low = all(k in tb for k in ('tb18v', 'tb18h', 'tb36v', 'tb36h'))
    have_89 = all(k in tb for k in ('tb89v', 'tb89h'))
    out = {}

    if have_89:
        v89, h89 = tb['tb89v'], tb['tb89h']
        n90ld = near90_linear_dyn(v89, h89, tiepts)
        out['near90_lin_dyn'] = n90ld
        out['near90ghz'] = near90(v89, h89, tiepts)
        out['asi'] = asi(v89, h89)
        out['p90'] = P90(v89, h89)

    if have_low:
        v18, h18, v36, h36 = tb['tb18v'], tb['tb18h'], tb['tb36v'], tb['tb36h']
        bf = bootstrap_f(v18, v36, tiepts)
        br = bristol(v18, v36, h36, tiepts)
        nt, _cmy = nasa(v18, h18, v36, tiepts)
        p37 = P37(v36, h36, tiepts)

        out['p37'] = p37
        out['bootstrap_p'] = bootstrap_p(v36, h36, tiepts)
        out['p18'] = P18(v18, h18, tiepts)
        out['bristol'] = br
        out['pr'] = pr(v18, h18, v36, h36, tiepts)[0]
        out['nasa_team'] = nt
        out['norsex'] = norsex(v18, v36, sensor, hemisphere)
        out['bootstrap_f'] = bf
        out['calval'] = calval(v36, v18, tiepts)
        out['umass_aes'] = UMass(v18, v36, tiepts)
        out['bf_nt'] = (bf + nt) / 2.0
        out['osisaf'] = osisaf(bf, br, 0.4)
        out['sicci'] = sicci(bf, br)

        if have_89:
            out['tud'] = tud(n90ld, bf)
            out['bf_nt_n90ld'] = (bf + nt + n90ld) / 3.0
            out['p37_n90ld'] = (p37 + n90ld) / 2.0
            out['p37_n90ld_bf'] = (p37 + n90ld + bf) / 3.0
            out['bf2_n90ld'] = (bf + bf**2.0 * n90ld) / (1.0 + bf**2.0)
            out['bf3_n90ld'] = (bf + bf**3.0 * n90ld) / (1.0 + bf**3.0)
            out['bf_n90ld'] = (bf + n90ld) / 2.0
            out['bf_bfn90ld'] = _safe_div(bf + bf * n90ld, 1.0 + bf)

    # keep a stable, documented order
    return {k: out[k] for k in ALG_INFO if k in out}


# ---------------------------------------------------------------------------
# NetCDF input / output
# ---------------------------------------------------------------------------

# TB_NAMES = ('tb18v', 'tb18h', 'tb36v', 'tb36h', 'tb89v', 'tb89h')
TB_NAMES = ('tb18v', 'tb18h', 'tb23v', 'tb23h', 'tb36v', 'tb36h', 'tb89v', 'tb89h')

FINE_GROUP = 'grid_6p25km'
SIC_FILL = np.float32(-999.0)


def read_var(var, index=None):
    """Read a (packed) TB variable as float32 with NaN for missing."""
    data = var[:] if index is None else var[index]
    return np.ma.filled(np.ma.asarray(data, dtype=np.float32), np.nan)


def block_mean(a, factor):
    """Mean of valid (non-NaN) values in factor x factor blocks of the last two axes."""
    ny, nx = a.shape[-2:]
    shp = a.shape[:-2] + (ny // factor, factor, nx // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN blocks
        return np.nanmean(a.reshape(shp), axis=(-3, -1))


def read_group_tbs(nc, gname, suffix, index=None, upscale_89=True):
    """Collect the TBs needed for SIC on one grid group.

    suffix '_daily_mean' for daily TBs, '' for per-swath TBs (with index).
    89 GHz TBs are block-averaged from the 6.25 km group for coarser grids.
    """
    grp = nc[gname]
    tb = {}
    for name in TB_NAMES:
        if name + suffix in grp.variables:
            tb[name] = read_var(grp[name + suffix], index)

    if upscale_89 and gname != FINE_GROUP and FINE_GROUP in nc.groups \
            and 'tb89v' not in tb:
        fine = nc[FINE_GROUP]
        factor = int(round(grp.resolution_km / fine.resolution_km))
        ny, nx = grp.dimensions['y'].size, grp.dimensions['x'].size
        if (fine.dimensions['y'].size == ny * factor
                and fine.dimensions['x'].size == nx * factor):
            for name in ('tb89v', 'tb89h'):
                tb[name] = block_mean(read_var(fine[name + suffix], index), factor)
    return tb


def copy_var(src_grp, dst_grp, name):
    src = src_grp[name]
    kw = {'zlib': True} if src.ndim >= 2 else {}
    dst = dst_grp.createVariable(name, src.dtype, src.dimensions, **kw)
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs() if k != '_FillValue'})
    dst[:] = src[:]


def create_sic_var(grp, varname, alg, dims, chunks, upscaled):
    desc, channels = ALG_INFO[alg]
    v = grp.createVariable(varname, 'f4', dims, zlib=True, complevel=4,
                           chunksizes=chunks, fill_value=SIC_FILL,
                           least_significant_digit=2)
    v.units = '%'
    v.long_name = 'sea ice concentration, ' + desc
    v.standard_name = 'sea_ice_area_fraction'
    v.channels = channels
    v.grid_mapping = 'crs'
    v.coordinates = 'lat lon'
    v.comment = 'Not clipped to 0-100 %.'
    if upscaled and '89' in channels:
        v.comment += ' 89 GHz TBs block-averaged from the 6.25 km grid.'
    return v


def to_percent(frac):
    return np.ma.masked_invalid(np.asarray(frac, dtype=np.float32) * 100.0)


# ---------------------------------------------------------------------------
# Drop-in replacement for process_file() in sicci_ic_algs.py
# (also add near the top of sicci_ic_algs.py:)
#
#     from dynamic_tiepoints import daily_tiepoints, weather_filter, describe, LOW_FREQ
# ---------------------------------------------------------------------------

TIEPOINT_GROUP = 'grid_25km'   # grid used to determine the daily tie points


def _read_land_fraction(grp):
    if 'land_fraction' not in grp.variables:
        return None
    return np.ma.filled(np.ma.asarray(grp['land_fraction'][:], dtype=np.float32), np.nan)


def _apply_weather(sic, tb, use_weather):
    """Weather filter where 18.7/36.5 GHz exist (not on the 89-GHz-only grid)."""
    if use_weather and all(c in tb for c in ('tb18v', 'tb36v')):
        sic, _ = weather_filter(sic, tb)
    return sic


def process_file(in_path, output_dir, mode='daily', sensor='AMSR2',
                 upscale_89=True, dynamic=True, use_weather=True, algorithms=None):
    """
    dynamic     : True  -> daily tie points from the daily-mean TBs (applied to
                           the daily SIC and to every swath of that day)
                  False -> fixed ATBD tie points
    use_weather : apply the GR(36.5V/18.7V) weather filter (SIC set to 0)
    algorithms  : list of algorithm names to write (None = all), e.g.
                  ['nasa_team', 'bootstrap_f', 'bootstrap_p', 'osisaf']
    """
    nc = netCDF4.Dataset(in_path)
    hemisphere = nc.getncattr('hemisphere')
    tiepts_fixed = TIEPOINTS[sensor][hemisphere]
    n_swath = nc.dimensions['swath'].size

    # --- daily tie points (once per file = once per day) -------------------
    tiepts = list(tiepts_fixed)
    tp_info = {'notes': ['fixed tie points (dynamic=False)']}
    if dynamic:
        gname_tp = TIEPOINT_GROUP if TIEPOINT_GROUP in nc.groups else \
            next(g for g in nc.groups if g != FINE_GROUP)
        grp_tp = nc[gname_tp]
        tb_d = read_group_tbs(nc, gname_tp, '_daily_mean', None, upscale_89)
        tiepts, tp_info = daily_tiepoints(
            tb_d, tiepts_fixed, compute_sic, hemisphere, sensor,
            land_fraction=_read_land_fraction(grp_tp),
            cell_km=float(grp_tp.resolution_km))
        del tb_d
        print('Daily tie points from %s:\n%s' % (gname_tp, describe(tiepts, tp_info)))

    def sic_for(tb):
        sic = compute_sic(tb, tiepts, hemisphere, sensor)
        if algorithms is not None:
            sic = {k: v for k, v in sic.items() if k in algorithms}
        return _apply_weather(sic, tb, use_weather)

    base = os.path.basename(in_path).replace('_TB_', '_SIC_')
    out_path = os.path.join(output_dir, base)
    out = netCDF4.Dataset(out_path, 'w', format='NETCDF4')
    out.Conventions = 'CF-1.8'
    out.title = 'AMSR2 sea ice concentration, SICCI-P2 ATBD algorithm suite'
    out.source_file = os.path.basename(in_path)
    out.hemisphere = hemisphere
    out.date = nc.getncattr('date')
    out.sensor_tiepoints = sensor
    out.tiepoint_method = 'dynamic (daily)' if dynamic else 'fixed'
    out.tiepoints_fixed = np.array(tiepts_fixed, dtype=np.float32)
    out.tiepoints_used = np.array(tiepts, dtype=np.float32)
    out.tiepoint_n_open_water = tp_info.get('n_ow', -1)
    out.tiepoint_n_consolidated_ice = tp_info.get('n_ci', -1)
    out.tiepoint_notes = '; '.join(tp_info['notes']) or 'none'
    out.weather_filter = 'GR(36.5V/18.7V) > 0.05 -> SIC = 0' if use_weather else 'none'
    out.algorithm_reference = ('ESA SICCI-P2 ATBD D2.1 (SICCI-P2-ATBD(SIC)), '
                               'Issue 1.0, 22 Sep 2017, Section 5.4.1')
    out.history = 'Created %s by sicci_ic_algs.py' % datetime.now(timezone.utc).isoformat(timespec='seconds')

    # swath metadata
    out.createDimension('swath', None)
    for name in ('swath_file', 'swath_start_time', 'orbit_direction'):
        if name in nc.variables:
            copy_var(nc, out, name)

    for gname, grp in nc.groups.items():
        og = out.createGroup(gname)
        og.resolution_km = grp.resolution_km
        og.createDimension('y', grp.dimensions['y'].size)
        og.createDimension('x', grp.dimensions['x'].size)
        for name in ('crs', 'x', 'y', 'lat', 'lon', 'land_fraction', 'land_mask'):
            if name in grp.variables:
                copy_var(grp, og, name)
        ny, nx = grp.dimensions['y'].size, grp.dimensions['x'].size
        chunks2 = (min(ny, 512), min(nx, 512))
        upscaled = gname != FINE_GROUP and upscale_89
        daily_suffix = '_daily' if mode == 'both' else ''
        swath_suffix = '_swath' if mode == 'both' else ''

        # --- daily-mean SIC ------------------------------------------------
        if mode in ('daily', 'both'):
            tb = read_group_tbs(nc, gname, '_daily_mean', None, upscale_89)
            sic = sic_for(tb)
            for name, frac in sic.items():
                v = create_sic_var(og, 'sic_' + name + daily_suffix, name,
                                   ('y', 'x'), chunks2, upscaled)
                v[:] = to_percent(frac)
            print('  %s daily: %d algorithms' % (gname, len(sic)))

        # --- per-swath SIC (same daily tie points) -------------------------
        if mode in ('swath', 'both') and n_swath > 0:
            if 'obs_time' in grp.variables:
                copy_var(grp, og, 'obs_time')
            chunks3 = (1,) + chunks2
            vars_ = {}
            for i in range(n_swath):
                tb = read_group_tbs(nc, gname, '', i, upscale_89)
                sic = sic_for(tb)
                for name, frac in sic.items():
                    if name not in vars_:
                        vars_[name] = create_sic_var(
                            og, 'sic_' + name + swath_suffix, name,
                            ('swath', 'y', 'x'), chunks3, upscaled)
                    vars_[name][i] = to_percent(frac)
            print('  %s per swath: %d algorithms x %d swaths' % (gname, len(vars_), n_swath))

    nc.close()
    out.close()
    print('Wrote %s' % out_path)
    return out_path




def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+',
                    help='daily TB files from amsr2_l1r_daily_grid.py (globs allowed)')
    ap.add_argument('--output-dir', default='.', help='directory for SIC files')
    ap.add_argument('--mode', choices=['daily', 'swath', 'both'], default='both',
                    help='SIC from daily-mean TBs, from each swath, or both')
    ap.add_argument('--sensor', choices=sorted(TIEPOINTS), default='AMSR2',
                    help='tie-point set to use')
    ap.add_argument('--upscale-89', dest='upscale_89', default=True,
                    action=argparse.BooleanOptionalAction, help='use block-averaged 89 GHz on the 12.5/25 km grids')
    ap.add_argument('--dynamic', default=True,
                    action=argparse.BooleanOptionalAction, help='daily dynamic tie points (--dynamic False: fixed ATBD tie points)')
    ap.add_argument('--weather-filter', dest='use_weather', default=True,
                    action=argparse.BooleanOptionalAction, help='apply the GR(36.5V/18.7V) weather filter')
    ap.add_argument('--algorithms', nargs='+', default=None,
                    help='only write these algorithms, e.g. nasa_team bootstrap_f osisaf')
    ap.add_argument('--datefirst', help='first processing date (YYYYMMDD)', default='00000000')
    ap.add_argument('--datelast', help='last processing date (YYYYMMDD)', default='99999999')
    args = ap.parse_args()

    files = sorted({f for pat in args.inputs for f in glob.glob(pat)})
    if not files:
        raise SystemExit('No input files found.')
    os.makedirs(args.output_dir, exist_ok=True)
    
    for f in files:
        
        f_date = os.path.basename(f).split("_")[-1][:8]
        
        if f_date >= args.datefirst and f_date <= args.datelast:
            print(os.path.basename(f))
            process_file(f, args.output_dir, args.mode, args.sensor,
                         upscale_89=args.upscale_89,
                         dynamic=args.dynamic,
                         use_weather=args.use_weather,
                         algorithms=args.algorithms)


if __name__ == '__main__':
    main()
