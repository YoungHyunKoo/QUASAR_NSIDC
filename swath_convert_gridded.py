#!/usr/bin/env python3
"""
Grid AMSR2 L1R swath brightness temperatures (GW1AM2_*_L1SGRTBR_*.nc) onto the
NSIDC Sea Ice Polar Stereographic grids and combine all swaths from one day
into a single CF-compliant NetCDF-4 file, together with daily-mean fields.

Channels and output grids
-------------------------
    89.0 GHz H/V  -> 6.25 km   (89A and 89B scans, each with its own geolocation)
    18.7 GHz H/V  -> 12.5 km and 25 km
    36.5 GHz H/V  -> 12.5 km and 25 km

Output file layout (one file per day and hemisphere)
----------------------------------------------------
    /                     swath (unlimited) dimension + per-swath metadata
    /grid_6p25km          x, y, crs, lat, lon,
                          tb89h, tb89v              (swath, y, x)  per-swath grids
                          obs_time                  (swath, y, x)  mean obs. time per cell
                          tb89h_daily_mean, ...     (y, x)
                          tb89h_daily_nobs, ...     (y, x)         footprints used
    /grid_12p5km          same, for tb18h, tb18v, tb36h, tb36v
    /grid_25km            same, for tb18h, tb18v, tb36h, tb36v

Gridding method
---------------
Drop-in-the-bucket: each swath footprint is assigned to the grid cell that
contains its centre, and all footprints in a cell are averaged. The daily mean
is the average of all footprints from all swaths of the day (i.e. weighted by
the number of footprints, not a mean of swath means).

Usage
-----
    python amsr2_l1r_daily_grid.py --input-dir /data/amsr2/l1r \
        --output-dir /data/amsr2/daily --hemisphere north
    python amsr2_l1r_daily_grid.py --input-dir ... --date 20171112 --hemisphere both

Requirements: numpy, netCDF4, pyproj
"""

import argparse
import glob
import os
from collections import defaultdict
from datetime import datetime, timezone

import netCDF4
import numpy as np
from pyproj import CRS, Transformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAT_89A = 'Latitude of Observation Point for 89A'
LON_89A = 'Longitude of Observation Point for 89A'
LAT_89B = 'Latitude of Observation Point for 89B'
LON_89B = 'Longitude of Observation Point for 89B'

# Geolocation keys:
#   'A'   = 89 GHz A-scan footprints (486 per scan)
#   'B'   = 89 GHz B-scan footprints (486 per scan)
#   'LOW' = 6.9-36.5 GHz footprints (243 per scan). In L1R the low-frequency
#           channels are co-registered to the odd-numbered (1-based) 89A
#           footprints, i.e. latA[:, ::2] in 0-based Python indexing.
#
# L1R resampled options for the low-frequency channels:
#   res06 / res10 / res23 / res36 = resampled to the footprint of the
#   6.9 / 10.65 / 23.8 / 36.5 GHz channel.
# Defaults below use res23 for 18.7 GHz (closest to its native footprint) and
# res36 for 36.5 GHz (native footprint). For footprint-matched 18.7/36.5 pairs
# (e.g. Bootstrap/NASA Team style algorithms) switch 36.5 GHz to res23.
CHANNELS = {
    'tb89h': {'res_km': [6.25],
              'sources': [('Brightness Temperature (original,89GHz-A,H)', 'A'),
                          ('Brightness Temperature (original,89GHz-B,H)', 'B')]},
    'tb89v': {'res_km': [6.25],
              'sources': [('Brightness Temperature (original,89GHz-A,V)', 'A'),
                          ('Brightness Temperature (original,89GHz-B,V)', 'B')]},
    'tb18h': {'res_km': [12.5, 25.0],
              'sources': [('Brightness Temperature (res23,18.7GHz,H)', 'LOW')]},
    'tb18v': {'res_km': [12.5, 25.0],
              'sources': [('Brightness Temperature (res23,18.7GHz,V)', 'LOW')]},
    'tb36h': {'res_km': [12.5, 25.0],
              'sources': [('Brightness Temperature (res36,36.5GHz,H)', 'LOW')]},
    'tb36v': {'res_km': [12.5, 25.0],
              'sources': [('Brightness Temperature (res36,36.5GHz,V)', 'LOW')]},
    'tb23h': {'res_km': [12.5, 25.0],
              'sources': [('Brightness Temperature (res23,23.8GHz,H)', 'LOW')]},
    'tb23v': {'res_km': [12.5, 25.0],
              'sources': [('Brightness Temperature (res23,23.8GHz,V)', 'LOW')]}
}

CHANNEL_META = {
    'tb89h': (89.0, 'H'), 'tb89v': (89.0, 'V'),
    'tb18h': (18.7, 'H'), 'tb18v': (18.7, 'V'),
    'tb36h': (36.5, 'H'), 'tb36v': (36.5, 'V'),
    'tb23h': (23.8, 'H'), 'tb23v': (23.8, 'V'),
}

# Layers: 'Land_Ocean Flag 6 to 36' = [6.9, 10.65, 23.8, 36.5 GHz footprints],
#         'Land_Ocean Flag 89'      = [89A, 89B].
# 12.5/25 km use the 23.8 GHz footprint (= res23 footprint of 18.7 GHz), the
# largest of the gridded channels, which gives a conservative mask.
LAND_SOURCES = {
    6.25: [('land_89A', 'A'), ('land_89B', 'B')],
    12.5: [('land_23', 'LOW')],
    25.0: [('land_23', 'LOW')],
}
LAND_MASK_THRESHOLD = 0.1  # land_mask = 1 where land_fraction > this (%)

# NSIDC Sea Ice Polar Stereographic grids (outer edges of the grid, metres).
# EPSG:3413/3976 are the WGS84 versions; use 3411/3412 (Hughes 1980 ellipsoid) instead to match older NSIDC products exactly.
# WGS 84 Arctic/Antarctic Polar Stereographic
# EPSG:3995/3031 for matching with BAS ice masks (S1, MODIS)

GRIDS = {
    'north': dict(epsg=3413, x_min=-3850e3, x_max=3750e3,
                  y_min=-5350e3, y_max=5850e3),
    'south': dict(epsg=3976, x_min=-3950e3, x_max=3950e3,
                  y_min=-3950e3, y_max=4350e3),
}

TB_SCALE = 0.01          # packing for output TBs (K per count)
TB_FILL = 0              # packed fill value
TB_VALID_RANGE = (20.0, 360.0)   # physically plausible TB range (K)
RAW_INVALID_MIN = 65534  # L1R fill / error codes for uint16 TBs
TIME_FILL = -9999.0


def res_group_name(res_km):
    return 'grid_' + ('%g' % res_km).replace('.', 'p') + 'km'


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

class PolarGrid:
    """One NSIDC polar stereographic grid at a given resolution."""

    def __init__(self, hemisphere, res_km):
        g = GRIDS[hemisphere]
        self.hemisphere = hemisphere
        self.res_km = res_km
        self.res = res_km * 1000.0
        self.crs = CRS.from_epsg(g['epsg'])
        self.x_min, self.y_max = g['x_min'], g['y_max']
        self.nx = int(round((g['x_max'] - g['x_min']) / self.res))
        self.ny = int(round((g['y_max'] - g['y_min']) / self.res))
        # cell-centre coordinates; rows run from top (north/y_max) to bottom
        self.x = self.x_min + (np.arange(self.nx) + 0.5) * self.res
        self.y = self.y_max - (np.arange(self.ny) + 0.5) * self.res
        self.to_xy = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        self.to_ll = Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)

    @property
    def size(self):
        return self.nx * self.ny

    def cell_index(self, x, y):
        """Flat cell index for projected coordinates, -1 where outside."""
        col = np.floor((x - self.x_min) / self.res).astype(np.int64)
        row = np.floor((self.y_max - y) / self.res).astype(np.int64)
        inside = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        idx = np.full(x.shape, -1, dtype=np.int64)
        idx[inside] = row[inside] * self.nx + col[inside]
        return idx

    def latlon(self):
        xx, yy = np.meshgrid(self.x, self.y)
        lon, lat = self.to_ll.transform(xx, yy)
        return lat.astype(np.float32), lon.astype(np.float32)


# ---------------------------------------------------------------------------
# Reading one swath
# ---------------------------------------------------------------------------

def parse_utc(s):
    """Parse e.g. '2017-11-12T07:44:41.550Z'."""
    return datetime.strptime(s.rstrip('Z'), '%Y-%m-%dT%H:%M:%S.%f').replace(tzinfo=timezone.utc)


def read_swath(path, day_start):
    """Read geolocation, scan times and the required TBs from one L1R file."""
    needed = {src for ch in CHANNELS.values() for src, _ in ch['sources']}
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)

        latA, lonA = ds[LAT_89A][:], ds[LON_89A][:]
        latB, lonB = ds[LAT_89B][:], ds[LON_89B][:]
        geo = {'A': (latA, lonA),
               'B': (latB, lonB),
               'LOW': (latA[:, ::2], lonA[:, ::2])}

        # Scan Time is TAI93 seconds; anchor it to the UTC start time in the
        # global attributes so leap seconds do not need to be handled.
        scan_time = ds['Scan Time'][:].astype(np.float64)
        start_utc = parse_utc(ds.getncattr('ObservationStartDateTime'))
        t0 = (start_utc - day_start).total_seconds()
        scan_sec = t0 + (scan_time - scan_time[0])   # seconds since day start

        tbs = {}
        for name in needed:
            var = ds[name]
            raw = var[:]
            scale = float(var.getncattr('SCALE FACTOR'))
            tb = raw.astype(np.float32) * scale
            bad = ((raw >= RAW_INVALID_MIN)
                   | (tb < TB_VALID_RANGE[0]) | (tb > TB_VALID_RANGE[1]))
            tb[bad] = np.nan
            tbs[name] = tb
            
        land = ds['Land_Ocean Flag 6 to 36'][:].astype(np.float32)
        land89 = ds['Land_Ocean Flag 89'][:].astype(np.float32)
        for arr in (land, land89):
            arr[arr > 100] = np.nan
        tbs['land_23'] = land[2]
        tbs['land_89A'], tbs['land_89B'] = land89[0], land89[1]

        meta = {'file': os.path.basename(path),
                'start_sec': t0,
                'orbit_direction': str(ds.getncattr('OrbitDirection'))}
    return geo, scan_sec, tbs, meta


# ---------------------------------------------------------------------------
# Gridding
# ---------------------------------------------------------------------------

def grid_swath(geo, scan_sec, tbs, grids):
    """
    Grid one swath onto every grid.

    Returns {res_km: {'sum': {ch: arr}, 'cnt': {ch: arr}, 'time': arr}} where
    arrays are flat (ny*nx). 'time' is the mean observation time per cell
    (seconds since day start), NaN where the swath has no footprints.
    """
    out = {}
    for res_km, grid in grids.items():
        north = grid.hemisphere == 'north'
        # project each geolocation set once per grid
        idx_cache = {}
        for key, (lat, lon) in geo.items():
            lat = lat.astype(np.float64)
            lon = lon.astype(np.float64)
            valid = (lat > 0) if north else (lat < 0)
            valid &= np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
            x, y = grid.to_xy.transform(lon, lat)
            idx = grid.cell_index(x, y)
            idx[~valid] = -1
            idx_cache[key] = idx

        sums, cnts = {}, {}
        tsum = np.zeros(grid.size)
        tcnt = np.zeros(grid.size)
        for ch, cfg in CHANNELS.items():
            if res_km not in cfg['res_km']:
                continue
            s = np.zeros(grid.size)
            c = np.zeros(grid.size)
            for src, key in cfg['sources']:
                tb = tbs[src]
                idx = idx_cache[key]
                times = np.broadcast_to(scan_sec[:, None], tb.shape)
                ok = (idx >= 0) & np.isfinite(tb)
                i = idx[ok]
                s += np.bincount(i, weights=tb[ok], minlength=grid.size)
                c += np.bincount(i, minlength=grid.size)
                tsum += np.bincount(i, weights=times[ok], minlength=grid.size)
                tcnt += np.bincount(i, minlength=grid.size)
            sums[ch], cnts[ch] = s, c
            
        lsum = np.zeros(grid.size)
        lcnt = np.zeros(grid.size)
        for src, key in LAND_SOURCES.get(res_km, []):
            idx, frac = idx_cache[key], tbs[src]
            ok = (idx >= 0) & np.isfinite(frac)
            lsum += np.bincount(idx[ok], weights=frac[ok], minlength=grid.size)
            lcnt += np.bincount(idx[ok], minlength=grid.size)

        with np.errstate(invalid='ignore', divide='ignore'):
            tmean = np.where(tcnt > 0, tsum / tcnt, np.nan)
        out[res_km] = {'sum': sums, 'cnt': cnts, 'time': tmean, 'land_sum': lsum, 'land_cnt': lcnt}
    return out


def pack_tb(tb):
    """float K (NaN = missing) -> uint16 counts with fill value 0."""
    packed = np.full(tb.shape, TB_FILL, dtype=np.uint16)
    ok = np.isfinite(tb)
    packed[ok] = np.round(tb[ok] / TB_SCALE).astype(np.uint16)
    return packed


# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------

def add_tb_var(grp, name, dims, chunks, long_name, extra=None):
    freq, pol = CHANNEL_META[name.split('_')[0]]
    v = grp.createVariable(name, 'u2', dims, zlib=True, complevel=4,
                           chunksizes=chunks, fill_value=TB_FILL)
    v.scale_factor = np.float32(TB_SCALE)
    v.add_offset = np.float32(0.0)
    v.units = 'K'
    v.long_name = long_name
    v.standard_name = 'brightness_temperature'
    v.frequency_GHz = freq
    v.polarization = pol
    v.grid_mapping = 'crs'
    # v.coordinates = 'lat lon'
    v.valid_min = np.uint16(TB_VALID_RANGE[0] / TB_SCALE)
    v.valid_max = np.uint16(TB_VALID_RANGE[1] / TB_SCALE)
    if extra:
        for k, val in extra.items():
            v.setncattr(k, val)
    v.set_auto_maskandscale(False)   # we write pre-packed integers
    return v


def create_output(path, day, hemisphere, grids, write_latlon=True):
    nc = netCDF4.Dataset(path, 'w', format='NETCDF4')
    nc.Conventions = 'CF-1.8'
    nc.title = 'AMSR2 L1R brightness temperatures gridded to NSIDC polar stereographic grids'
    nc.source = 'GCOM-W1 AMSR2 Level 1R (L1SGRTBR) swath data, JAXA'
    nc.hemisphere = hemisphere
    nc.date = day.strftime('%Y-%m-%d')
    nc.gridding_method = ('drop-in-the-bucket: mean of all footprint centres falling in each cell; '
                          'daily mean weighted by number of footprints')
    nc.low_frequency_geolocation = 'odd-numbered (1-based) 89A footprints, i.e. latA[:, ::2]'
    nc.history = 'Created %s by amsr2_l1r_daily_grid.py' % datetime.now(timezone.utc).isoformat(timespec='seconds')

    day_units = 'seconds since %s 00:00:00' % day.strftime('%Y-%m-%d')
    nc.createDimension('swath', None)
    v = nc.createVariable('swath_file', str, ('swath',))
    v.long_name = 'source L1R file name'
    v = nc.createVariable('swath_start_time', 'f8', ('swath',))
    v.units = day_units
    v.long_name = 'swath observation start time (UTC)'
    v = nc.createVariable('orbit_direction', str, ('swath',))
    v.long_name = 'Ascending or Descending'

    for res_km, grid in grids.items():
        g = nc.createGroup(res_group_name(res_km))
        g.resolution_km = res_km
        g.createDimension('y', grid.ny)
        g.createDimension('x', grid.nx)

        crs = g.createVariable('crs', 'i4')
        for k, val in grid.crs.to_cf().items():
            crs.setncattr(k, val)
        crs.epsg_code = 'EPSG:%d' % grid.crs.to_epsg()

        vx = g.createVariable('x', 'f8', ('x',))
        vx[:] = grid.x
        vx.standard_name = 'projection_x_coordinate'
        vx.units = 'm'
        vy = g.createVariable('y', 'f8', ('y',))
        vy[:] = grid.y
        vy.standard_name = 'projection_y_coordinate'
        vy.units = 'm'

        if write_latlon:
            lat, lon = grid.latlon()
            vlat = g.createVariable('lat', 'f4', ('y', 'x'), zlib=True)
            vlat[:] = lat
            vlat.standard_name, vlat.units = 'latitude', 'degrees_north'
            vlon = g.createVariable('lon', 'f4', ('y', 'x'), zlib=True)
            vlon[:] = lon
            vlon.standard_name, vlon.units = 'longitude', 'degrees_east'

        chunks3 = (1, min(grid.ny, 512), min(grid.nx, 512))
        chunks2 = chunks3[1:]
        for ch, cfg in CHANNELS.items():
            if res_km not in cfg['res_km']:
                continue
            srcs = '; '.join(s for s, _ in cfg['sources'])
            add_tb_var(g, ch, ('swath', 'y', 'x'), chunks3,
                       'per-swath gridded brightness temperature', {'source_variables': srcs})
            add_tb_var(g, ch + '_daily_mean', ('y', 'x'), chunks2,
                       'daily mean brightness temperature (all swaths)', {'source_variables': srcs})
            n = g.createVariable(ch + '_daily_nobs', 'u2', ('y', 'x'), zlib=True, chunksizes=chunks2)
            n.long_name = 'number of footprints in daily mean'
            n.grid_mapping = 'crs'
            
        # Land fraction variable
        lf = g.createVariable('land_fraction', 'u1', ('y', 'x'), zlib=True,
                              chunksizes=chunks2, fill_value=np.uint8(255))
        lf.units = '%'
        lf.long_name = 'mean land fraction of footprints in cell (all swaths of the day)'
        lf.grid_mapping = 'crs'
        lm = g.createVariable('land_mask', 'u1', ('y', 'x'), zlib=True,
                              chunksizes=chunks2, fill_value=np.uint8(255))
        lm.flag_values = np.array([0, 1], dtype=np.uint8)
        lm.flag_meanings = 'ocean land'
        lm.long_name = 'land mask (land_fraction > %d %%)' % LAND_MASK_THRESHOLD
        lm.grid_mapping = 'crs'
        
        # Observation time variable
        t = g.createVariable('obs_time', 'f4', ('swath', 'y', 'x'), zlib=True,
                             chunksizes=chunks3, fill_value=np.float32(TIME_FILL))
        t.units = day_units
        t.long_name = 'mean observation time of footprints in cell for this swath (UTC)'
        t.comment = 'Values > 86400 are possible for swaths that cross midnight.'
        t.grid_mapping = 'crs'
    return nc


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def file_date(path):
    """GW1AM2_201711120744_016A_... -> '20171112' (date of swath start)."""
    return os.path.basename(path).split('_')[1][:8]


def process_day(files, day_str, hemisphere, output_dir, write_latlon=True):
    day = datetime.strptime(day_str, '%Y%m%d').replace(tzinfo=timezone.utc)
    res_list = sorted({r for c in CHANNELS.values() for r in c['res_km']})
    grids = {r: PolarGrid(hemisphere, r) for r in res_list}

    out_path = os.path.join(output_dir, 'AMSR2_L1R_TB_PS_%s_%s.nc' % (hemisphere[0].upper(), day_str))
    nc = create_output(out_path, day, hemisphere, grids, write_latlon)

    daily_sum = {r: defaultdict(lambda r=r: np.zeros(grids[r].size)) for r in res_list}
    daily_cnt = {r: defaultdict(lambda r=r: np.zeros(grids[r].size)) for r in res_list}
    daily_land = {r: [np.zeros(grids[r].size), np.zeros(grids[r].size)] for r in res_list}

    i_sw = 0
    for path in sorted(files):
        try:
            geo, scan_sec, tbs, meta = read_swath(path, day)
        except Exception as exc:  # corrupt or unexpected file: skip, keep going
            print('  ! skipping %s (%s)' % (os.path.basename(path), exc))
            continue
        gridded = grid_swath(geo, scan_sec, tbs, grids)

        n_cells = sum(int((c > 0).sum()) for r in gridded for c in gridded[r]['cnt'].values())
        if n_cells == 0:
            print('  - %s: no coverage in %s hemisphere grid' % (meta['file'], hemisphere))
            continue

        nc['swath_file'][i_sw] = meta['file']
        nc['swath_start_time'][i_sw] = meta['start_sec']
        nc['orbit_direction'][i_sw] = meta['orbit_direction']

        for r, res in gridded.items():
            g = nc[res_group_name(r)]
            grid = grids[r]
            for ch, s in res['sum'].items():
                c = res['cnt'][ch]
                with np.errstate(invalid='ignore', divide='ignore'):
                    mean = np.where(c > 0, s / c, np.nan)
                g[ch][i_sw] = pack_tb(mean.reshape(grid.ny, grid.nx))
                daily_sum[r][ch] += s
                daily_cnt[r][ch] += c
            
            daily_land[r][0] += res['land_sum']
            daily_land[r][1] += res['land_cnt']
            
            t = res['time'].reshape(grid.ny, grid.nx)
            g['obs_time'][i_sw] = np.where(np.isfinite(t), t, TIME_FILL).astype(np.float32)

        print('  + %s (%s) -> swath %d' % (meta['file'], meta['orbit_direction'], i_sw))
        i_sw += 1

    # daily means
    for r in res_list:
        g = nc[res_group_name(r)]
        grid = grids[r]
        for ch in daily_sum[r]:
            s, c = daily_sum[r][ch], daily_cnt[r][ch]
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(c > 0, s / c, np.nan)
            g[ch + '_daily_mean'][:] = pack_tb(mean.reshape(grid.ny, grid.nx))
            g[ch + '_daily_nobs'][:] = np.minimum(c, 65535).astype(np.uint16).reshape(grid.ny, grid.nx)
            
        lsum, lcnt = daily_land[r]
        with np.errstate(invalid='ignore', divide='ignore'):
            lfrac = np.where(lcnt > 0, lsum / lcnt, np.nan).reshape(grid.ny, grid.nx)
        ok = np.isfinite(lfrac)
        g['land_fraction'][:] = np.ma.masked_where(~ok, np.round(np.where(ok, lfrac, 0)).astype(np.uint8))
        g['land_mask'][:] = np.ma.masked_where(~ok, (lfrac > LAND_MASK_THRESHOLD).astype(np.uint8))

    nc.number_of_swaths = i_sw
    nc.close()
    print('Wrote %s (%d swaths)' % (out_path, i_sw))
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', required=True, help='directory with L1R swath files')
    ap.add_argument('--output-dir', default='.', help='directory for daily output files')
    ap.add_argument('--pattern', default='GW1AM2_*_L1SGRTBR_*.nc', help='glob pattern for input files')
    ap.add_argument('--date', help='process only this date (YYYYMMDD); default: all dates found')
    ap.add_argument('--hemisphere', choices=['north', 'south', 'both'], default='south')
    ap.add_argument('--no-latlon', action='store_true', help='do not store 2-D lat/lon arrays')
    ap.add_argument('--datefirst', help='first processing date (YYYYMMDD)', default='00000000')
    ap.add_argument('--datelast', help='last processing date (YYYYMMDD)', default='99999999')
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.input_dir, args.pattern))
    by_day = defaultdict(list)
    for f in files:
        if file_date(f) >= args.datefirst and file_date(f) <= args.datelast:
            by_day[file_date(f)].append(f)
    if args.date:
        by_day = {args.date: by_day.get(args.date, [])}
    if not any(by_day.values()):
        raise SystemExit('No input files found.')
        
    # print(by_day)

    os.makedirs(args.output_dir, exist_ok=True)
    hemis = ['north', 'south'] if args.hemisphere == 'both' else [args.hemisphere]
    for day_str in sorted(by_day):
        for hemi in hemis:
            print('%s  %s hemisphere: %d files' % (day_str, hemi, len(by_day[day_str])))
            process_day(by_day[day_str], day_str, hemi, args.output_dir,
                        write_latlon=not args.no_latlon)


if __name__ == '__main__':
    main()