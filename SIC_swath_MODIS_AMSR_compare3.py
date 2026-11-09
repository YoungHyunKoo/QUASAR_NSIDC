import xarray as xr
import fsspec
import glob, os
import numpy as np
import pickle
import pandas as pd
import time
import netCDF4

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# from matplotlib.colors import Normalize
# import cartopy
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

import rioxarray
from rasterio.enums import Resampling
import yaml

import h5py

import rasterio
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm
from pyproj import Proj, transform, Transformer

import warnings
warnings.filterwarnings('ignore')

def read_config(fn):
    from yaml.scanner import ScannerError

    try:
        with open(fn, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'Could not file config file: {fn}')
    except ScannerError:
        raise ValueError(f'Bad yaml file: {fn}')

    if config is None:
        raise ValueError(f'config is None read from: {fn}')

    return config


#####################################################################
########## FUNCTIONS ##########

def get_S1_array(x0, y0, img_id, dp = 100, distance = 100000, proj = "EPSG:6932"):

    geometry = ee.Geometry.Point([x0, y0], proj = proj)
    
    extent = geometry.buffer(distance = distance, proj = proj).bounds(proj = proj)        
    extent_coords = np.array(extent.coordinates().getInfo()[0])
    
    # Define necessary functions with the defined bbox --------------------------
    def collection_addbands(img):
        bands = img.bandNames() # First band ('HH' or 'VV')
        band = [ee.Algorithms.If(bands.contains('HH'), 'HH', 'VV')]
        norm = img.select(band).divide(img.select('angle')).rename('norm')
        img2 = img.addBands(norm, overwrite=True).select('norm')
        return img2
    def add_coverage(img):    
        tol = 10000
        overlap = img.geometry().intersection(extent, tol)
        ratio = overlap.area(tol).divide(extent.area(tol))
        return img.set({'coverage_ratio': ratio})
    # calculate coverage area of image to roi
    def coverage(img):
        tol = 10000
        overlap = img.geometry().intersection(extent, tol)
        ratio = overlap.area(tol).divide(extent.area(tol))
        return ratio.getInfo()
    # -------------------------------------------------------------------------

    k = "COPERNICUS/S1_GRD/" + img_id
        
    img = collection_addbands(ee.Image(k)).setDefaultProjection(proj)
    band_coord = ee.Image.pixelCoordinates(proj)
    img = img.addBands(band_coord)
    
    img = img.clip(extent)                  
    
    img2 = img.select("norm")
    xx2 = img.select("x")
    yy2 = img.select("y")
        
    a = geemap.ee_to_numpy(img2, region = extent, scale = dp)[:, :, 0]
    xx = geemap.ee_to_numpy(xx2, region = extent, scale = dp)[:, :, 0]
    yy = geemap.ee_to_numpy(yy2, region = extent, scale = dp)[:, :, 0]

    return a, xx, yy 

def draw_map(modis_fn, xxs, yys, array, xx, yy, sic, sic2, ii, jj, S1_id, AMSR_id, step = 16, sensor = "sar"):
    s = int(step/2)+1
    time_amsr = AMSR_id.split("_")[1]
    time_s1 = S1_id.split("_")[4].replace("T", "")[:-2]
    year = AMSR_id.split("_")[1][:4]

    fig, ax = plt.subplots(1,1, figsize = (4,4), dpi = 200, num = 1, clear = True)
    ax.set_aspect('equal', adjustable='box')
    if sensor == "sar":
        ax.pcolormesh(xxs, yys, array, vmin = -1, vmax = 0, cmap = "gray")
    elif sensor == "modis":
        ax.pcolormesh(xxs, yys, array, vmin = 0, vmax = 1, cmap = "gray")
    pm = ax.pcolormesh(xx[ii-s:ii+s, jj-s:jj+s], yy[ii-s:ii+s, jj-s:jj+s], sic[ii-s:ii+s, jj-s:jj+s], alpha = 0.3, vmin = 0, vmax = 100)
    # plt.colorbar(sar, shrink = 0.5)
    CS = ax.contour(xx[ii-s:ii+s, jj-s:jj+s], yy[ii-s:ii+s, jj-s:jj+s], sic[ii-s:ii+s, jj-s:jj+s], np.arange(0, 100, 5),
                   colors='yellow', linewidths = 0.7, linestyles = "dashed", alpha = 0.5, zorder = 1)
    ax.clabel(CS, fontsize=7)
    
    xmin, xmax = xxs.min(), xxs.max()
    ymin, ymax = yys.min(), yys.max()
    ax.set_xlim(xxs.min(), xxs.max())
    ax.set_ylim(yys.min(), yys.max())
    
    # colorbar
    cbaxes = inset_axes(ax, width="40%", height="3%", loc='lower center') 
    cb = fig.colorbar(pm, cax=cbaxes, orientation='horizontal')
    cbaxes.xaxis.set_ticks_position('top')
    
    ax2 = inset_axes(ax, width="40%", height="40%", bbox_to_anchor = (1,0,1,1), bbox_transform = ax.transAxes, loc='lower left') 
    ax2.pcolormesh(xx[20:-20, 20:-20], yy[20:-20, 20:-20], sic2[20:-20, 20:-20], vmin = 0, vmax = 100)
    sic0 = np.where(sic > 0, 1, np.nan)
    # [sic <= 0] = np.nan
    ax2.pcolormesh(xx[20:-20, 20:-20], yy[20:-20, 20:-20], sic0[20:-20, 20:-20], vmin=0, vmax = 1, cmap = "jet", alpha = 0.5)
    ax2.scatter(xx[ii, jj], yy[ii, jj], c = "r")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ax.annotate(f"S1: {time_s1}", xy=(0.02,0.88), xycoords='axes fraction', fontsize = 9, color = "white",
                  path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    ax.annotate(f"AMSR: {time_amsr}", xy=(0.02,0.94), xycoords='axes fraction', fontsize = 9, color = "yellow",
                  path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.annotate("20 km", xy=(xmin + 30000, ymin + 19000), fontsize = 10, ha = "center", va = "bottom", color = "k")
    ax.plot([xmin + 20000, xmin + 40000], [ymin + 15000, ymin + 15000], color = "k")
    os.makedirs(f"{folder_fig}/{modis_fn}", exist_ok=True)
    plt.savefig(f"{folder_fig}/{modis_fn}/AMSR_SIC_{time_amsr}_{time_s1}.png", bbox_inches = "tight")
    fig.clear()
    plt.close(fig)

def draw_map_ref(modis_fn, xxs, yys, array, xx, yy, sic, sic2, ii, jj, time_s1, time_amsr, dsec, step = 16, sensor = "sar", version = ""):
    s = int(step/2)+1

    fig, ax = plt.subplots(1,1, figsize = (4,4), dpi = 200, num = 1, clear = True)
    ax.set_aspect('equal', adjustable='box')
    if sensor == "sar":
        ax.pcolormesh(xxs[::s, ::s], yys[::s, ::s], array[::s, ::s], vmin = -1, vmax = 0, cmap = "gray")
    elif sensor == "modis":
        ax.pcolormesh(xxs[::s, ::s], yys[::s, ::s], array[::s, ::s], vmin = -1, vmax = 1, cmap = "gray_r")
    pm = ax.pcolormesh(xx, yy, sic, alpha = 0.3, vmin = 0, vmax = 100)
    CS = ax.contour(xx, yy, sic, np.arange(0, 101, 10),
                   colors='yellow', linewidths = 0.5, linestyles = "dashed", alpha = 0.5, zorder = 1)
    ax.clabel(CS, fontsize=7)
    
    xmin, xmax = xxs.min(), xxs.max()
    ymin, ymax = yys.min(), yys.max()
    ax.set_xlim(xxs.min(), xxs.max())
    ax.set_ylim(yys.min(), yys.max())
    
    # colorbar
    cbaxes = inset_axes(ax, width="40%", height="3%", loc='lower center') 
    cb = fig.colorbar(pm, cax=cbaxes, orientation='horizontal')
    cbaxes.xaxis.set_ticks_position('top')
    
    ax2 = inset_axes(ax, width="40%", height="40%", bbox_to_anchor = (1,0,1,1), bbox_transform = ax.transAxes, loc='lower left') 
    ax2.pcolormesh(xx, yy, sic2, vmin = 0, vmax = 100)

    ax2.scatter(xx[ii, jj], yy[ii, jj], c = "r")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ax.annotate(f"MODIS: {time_s1}", xy=(0.02,0.90), xycoords='axes fraction', fontsize = 9, color = "white",\
                  path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    ax.annotate(f"AMSR: {time_amsr}", xy=(0.02,0.95), xycoords='axes fraction', fontsize = 9, color = "yellow",\
                  path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    ax.annotate(f"dt: {dsec:.0f} seconds", xy=(0.02,0.85), xycoords='axes fraction', fontsize = 9, color = "cyan",\
                 path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.annotate("100 km", xy=(xmin + 100000, ymin + 19000), fontsize = 10, ha = "center", va = "bottom", color = "k")
    ax.plot([xmin + 50000, xmin + 150000], [ymin + 15000, ymin + 15000], color = "k")
    os.makedirs(f"{folder_fig}/{modis_fn}", exist_ok=True)
    plt.savefig(f"{folder_fig}/{modis_fn}/AMSR_SIC_{time_amsr}_{time_s1}_{version}_map.png", bbox_inches = "tight")
    fig.clear()
    plt.close(fig)

def draw_map_raw(modis_fn, xxs, yys, array, xx, yy, sic, sic2, ii, jj, time_s1, time_amsr, dsec, step = 10, sensor = "sar", version = ""):
    s = int(step/2)+1

    fig, ax = plt.subplots(1,1, figsize = (4,4), dpi = 200, num = 1, clear = True)
    ax.set_aspect('equal', adjustable='box')
    if sensor == "sar":
        ax.pcolormesh(xxs[::s, ::s], yys[::s, ::s], array[::s, ::s], vmin = 0, vmax = 40, cmap = "viridis")
    elif sensor == "modis":
        ax.pcolormesh(xxs[::s, ::s], yys[::s, ::s], array[::s, ::s], vmin = 0, vmax = 100, cmap = "viridis")
    # pm = ax.pcolormesh(xx, yy, sic, alpha = 0.3, vmin = 0, vmax = 100)
    # CS = ax.contour(xx, yy, sic, np.arange(0, 101, 10),
    #                colors='yellow', linewidths = 0.5, linestyles = "dashed", alpha = 0.5, zorder = 1)
    # ax.clabel(CS, fontsize=7)
    
    xmin, xmax = xxs.min(), xxs.max()
    ymin, ymax = yys.min(), yys.max()
    ax.set_xlim(xxs.min(), xxs.max())
    ax.set_ylim(yys.min(), yys.max())
    
    # colorbar
    # cbaxes = inset_axes(ax, width="40%", height="3%", loc='lower center') 
    # cb = fig.colorbar(pm, cax=cbaxes, orientation='horizontal')
    # cbaxes.xaxis.set_ticks_position('top')
    
    ax.annotate(f"MODIS: {time_s1}", xy=(0.02,0.90), xycoords='axes fraction', fontsize = 9, color = "white",\
                  path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    ax.annotate(f"AMSR: {time_amsr}", xy=(0.02,0.95), xycoords='axes fraction', fontsize = 9, color = "yellow",\
                  path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    ax.annotate(f"dt: {dsec:.0f} seconds", xy=(0.02,0.85), xycoords='axes fraction', fontsize = 9, color = "cyan",\
                path_effects=[patheffects.withStroke(linewidth=2, foreground='k', capstyle="round")])
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.annotate("100 km", xy=(xmin + 100000, ymin + 19000), fontsize = 10, ha = "center", va = "bottom", color = "k")
    ax.plot([xmin + 50000, xmin + 150000], [ymin + 15000, ymin + 15000], color = "k")
    os.makedirs(f"{folder_fig}/{modis_fn}", exist_ok=True)
    plt.savefig(f"{folder_fig}/{modis_fn}/AMSR_SIC_{time_amsr}_{time_s1}_{version}_raw_{sensor}.png", bbox_inches = "tight")
    fig.clear()
    plt.close(fig)

def draw_map_resampled(modis_fn, amsr2_xx, amsr2_yy, modis_resampled, valid_ratio, time_s1, time_amsr, amsr2, version = 0):
    fig, ax = plt.subplots(1,4, sharex = True, sharey = True, figsize = (12,4), dpi = 200, num = 1, clear = True)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    map_msic = ax[0].pcolormesh(amsr2_xx, amsr2_yy, modis_resampled.data, vmin = 0, vmax = 100)
    fig.colorbar(map_msic, orientation = "horizontal", shrink = 0.5, pad = 0.1, label = "MODIS SIC (resampled) [%]")
    map_mv = ax[1].pcolormesh(amsr2_xx, amsr2_yy, valid_ratio, vmin = 0, vmax = 1, cmap = "Reds")
    fig.colorbar(map_mv, orientation = "horizontal", shrink = 0.5, pad = 0.1, label = "Valid MODIS pixel ratio")
    map_asic = ax[2].pcolormesh(amsr2_xx, amsr2_yy, amsr2.data, vmin = 0, vmax = 100)
    fig.colorbar(map_asic, orientation = "horizontal", shrink = 0.5, pad = 0.1, label = "AMSR2 swath SIC [%]")
    map_dsic = ax[3].pcolormesh(amsr2_xx, amsr2_yy, amsr2.data - modis_resampled.data, vmin = -100, vmax = 100, cmap = "RdBu")
    fig.colorbar(map_dsic, orientation = "horizontal", shrink = 0.5, pad = 0.1, label = "AMSR2 SIC - MODIS SIC [%]")
    ax[0].set_title(f"S1: {time_s1}; AMSR: {time_amsr}", ha = "left")
    for a in ax:
        a.grid(ls = ":", lw = 0.5)
    os.makedirs(f"{folder_fig}/{modis_fn}", exist_ok=True)
    plt.savefig(f"{folder_fig}/{modis_fn}/AMSR_SIC_{time_amsr}_{time_s1}_{version}_resampled.png", bbox_inches = "tight")
    fig.clear()
    plt.close(fig)

def draw_scatter(modis_fn, sic_sar, sic_amsr, sic_valid, time_s1, time_amsr, version = 0):

    # All data (valid ratio > 0.5)
    idx1 = abs(sic_sar * sic_amsr) >= 0
    sic_sar1 = sic_sar[idx1]
    sic_amsr1 = sic_amsr[idx1]
    sic_valid1 = sic_valid[idx1]
    
    mae1 = np.nanmean(abs(sic_sar1 - sic_amsr1))
    corr1 = np.corrcoef(sic_sar1, sic_amsr1)[0][1]

    # Full-ratio data (valid ratio = 1.0)
    idx2 = (sic_valid == 1) & idx1
    sic_sar2 = sic_sar[idx2]
    sic_amsr2 = sic_amsr[idx2]
    sic_valid2 = sic_valid[idx2]
    
    mae2 = np.nanmean(abs(sic_sar2 - sic_amsr2))
    corr2 = np.corrcoef(sic_sar2, sic_amsr2)[0][1]

    fig, ax = plt.subplots(1,1, figsize = (4,3.5), dpi = 200, num = 1, clear = True)    
    
    ax.annotate(f"S1: {time_s1}", xy=(0.02,0.95), xycoords='axes fraction', fontsize = 9, color = "k")
    ax.annotate(f"AMSR: {time_amsr}", xy=(0.02,0.90), xycoords='axes fraction', fontsize = 9, color = "k")
    ax.annotate(f"Corr$_{{1.0}}$: {corr2:.2f}", xy=(0.02,0.85), xycoords='axes fraction', fontsize = 9, color = "brown")
    ax.annotate(f"MAE$_{{1.0}}$: {mae2:.2f}", xy=(0.02,0.80), xycoords='axes fraction', fontsize = 9, color = "brown")
    ax.annotate(f"N$_{{1.0}}$: {len(sic_sar2)}", xy=(0.02,0.75), xycoords='axes fraction', fontsize = 9, color = "brown")
    ax.annotate(f"Corr$_{{0.5}}$: {corr1:.2f}", xy=(0.02,0.70), xycoords='axes fraction', fontsize = 9, color = "r")
    ax.annotate(f"MAE$_{{0.5}}$: {mae1:.2f}", xy=(0.02,0.65), xycoords='axes fraction', fontsize = 9, color = "r")
    ax.annotate(f"N$_{{0.5}}$: {len(sic_sar1)}", xy=(0.02,0.60), xycoords='axes fraction', fontsize = 9, color = "r")
    
    sc = ax.scatter(sic_sar, sic_amsr, c = sic_valid, s = 2, vmin = 0.3, vmax = 1.0, cmap = "Reds")
    fig.colorbar(sc, ax=ax, shrink = 0.5, label = "Valid ratio")
    ax.grid(ls = ":", lw = 0.5)
    ax.set_xlabel("SAR SIC (%)")
    ax.set_ylabel("PMW SIC (%)")
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.plot([0, 100], [0, 100], ls = "--", lw = 0.5, color = "gray")
    
    os.makedirs(f"{folder_fig}/{modis_fn}", exist_ok=True)
    plt.savefig(f"{folder_fig}/{modis_fn}/AMSR_SIC_{time_amsr}_{time_s1}_{version}_scatter.png", bbox_inches = "tight")
    fig.clear()
    plt.close(fig)

import rasterio
import shapely.geometry
import geopandas as gpd

import gportal
import requests
from gportal import http_client

def _get_new_gportal(path, params=None, timeout=None):
    # gportal 0.4.0 builds search URLs on the old server; send them to the new CSW server
    url = "https://csw.gportal.jaxa.jp/csw" if path.rstrip("/").endswith("csw/csw") \
          else "https://www.gportal.jaxa.jp/" + path.lstrip("/")
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()

http_client.get = _get_new_gportal

def find_amsr_for_tiff(tif_file, time0, dh = 1):

    # Open your GeoTIFF
    with rasterio.open(tif_file) as src:
        bounds = src.bounds          # (left, bottom, right, top)
        crs = src.crs
    
    # Create a box from the raster bounds
    raster_box = shapely.geometry.box(*bounds)
    
    # If the raster is NOT in EPSG:4326, reproject it to match the swath CRS
    if crs.to_epsg() != 4326:
        raster_gdf = gpd.GeoDataFrame(geometry=[raster_box], crs=crs)
        raster_gdf = raster_gdf.to_crs("EPSG:4326")
        raster_box = raster_gdf.geometry.iloc[0]
    
    bounds_4326 = raster_box.bounds  # (minx, miny, maxx, maxy)
    amsr_cnt = 0

    while (amsr_cnt < 1) and (dh <= 12):
        start_time = time0 - timedelta(seconds = 3600*dh)
        end_time = time0 + timedelta(seconds = 3600*dh)
        
        res = gportal.search(
            dataset_ids=['11001002'], # 'L2-Sea Ice Concentration（SIC)': ['11002013', '11002006']; 'L1R-Brightness temperature（TB）': ['11001002'] L1R Brightness temp: ['11001002'],  # l1r - see gportal.datasets()
            start_time=start_time,
            end_time=end_time,
            bbox=bounds_4326
        )

        if res.matched() > 0:
            features = [p.to_dict(flatten_properties=True) for p in res.products()]
            swath_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
            swath_gdf['geometry'] = swath_gdf['geometry'].buffer(0)
            
            # Spatial relationship options
            intersecting  = swath_gdf[swath_gdf.intersects(raster_box)]
            full_coverage  = swath_gdf[swath_gdf.covers(raster_box)]     # swath covers entire raster
                
            amsr_cnt = len(intersecting) #res.matched()
            # print(f"Matched for {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}: {amsr_cnt}")
            
        dh += 2
        
        # print(dh)
    
    if dh <= 12:
        swath_gdf["overlap_area"] = swath_gdf.geometry.intersection(raster_box).area
        swath_gdf["coverage_pct"] = (
            swath_gdf["overlap_area"] / raster_box.area * 100
        ).round(1)    
    
        swath_gdf = swath_gdf[swath_gdf["coverage_pct"] > 50].reset_index(drop = True)
        swath_gdf['time_str'] = [s.split("_")[1] for s in swath_gdf['identifier']]
    
        return swath_gdf
        
    else:
        return []

def convert_to_rio(data, x, y, proj = "EPSG:6932"):
    
    data2 = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y, "x": x}
    )
    
    # Assign CRS and write nodata
    data2 = data2.rio.write_crs(proj)
    data2 = data2.rio.write_nodata(np.nan)
    data2 = data2.rio.set_spatial_dims(x_dim="x", y_dim="y")

    return data2
    
    
def process_file(in_path, output_dir, mode='daily', sensor='AMSR2',
                 upscale_89=True):
    nc = netCDF4.Dataset(in_path)
    hemisphere = nc.getncattr('hemisphere')
    tiepts = TIEPOINTS[sensor][hemisphere]
    n_swath = nc.dimensions['swath'].size

    base = os.path.basename(in_path).replace('_TB_', '_SIC_')
    out_path = os.path.join(output_dir, base)
    out = netCDF4.Dataset(out_path, 'w', format='NETCDF4')
    out.Conventions = 'CF-1.8'
    out.title = 'AMSR2 sea ice concentration, SICCI-P2 ATBD algorithm suite'
    out.source_file = os.path.basename(in_path)
    out.hemisphere = hemisphere
    out.date = nc.getncattr('date')
    out.sensor_tiepoints = sensor
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
        # variable names: sic_<alg> for one mode, sic_<alg>_daily/_swath for both
        daily_suffix = '_daily' if mode == 'both' else ''
        swath_suffix = '_swath' if mode == 'both' else ''

        # --- daily-mean SIC ------------------------------------------------
        if mode in ('daily', 'both'):
            tb = read_group_tbs(nc, gname, '_daily_mean', None, upscale_89)
            sic = compute_sic(tb, tiepts, hemisphere, sensor)
            for name, frac in sic.items():
                v = create_sic_var(og, 'sic_' + name + daily_suffix, name,
                                   ('y', 'x'), chunks2, upscaled)
                v[:] = to_percent(frac)
            print('  %s daily: %d algorithms' % (gname, len(sic)))

        # --- per-swath SIC -------------------------------------------------
        if mode in ('swath', 'both') and n_swath > 0:
            if 'obs_time' in grp.variables:
                copy_var(grp, og, 'obs_time')
            chunks3 = (1,) + chunks2
            vars_ = {}
            for i in range(n_swath):
                tb = read_group_tbs(nc, gname, '', i, upscale_89)
                sic = compute_sic(tb, tiepts, hemisphere, sensor)
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


def generate_modis_hdf(fn, identifiers, times_amsr2, times_modis2, sic_amsr2, sic_sar2, sic_valid2, sic_counts2, tb_amsr2, xxs2, yys2, flags, sol_zenith2, sen_zenith2, raw_modis2, raw_modis_max2, raw_modis_min2, fields_sic, fields_tb, amsr_crs = "EPSG:3031"):

    N = len(sic_amsr2)
    
    os.makedirs(f"{folder_out}/MODIS/{year}/{month}", exist_ok=True)
    output = f"{folder_out}/MODIS/{year}/{month}/" + os.path.basename(fn.replace(".tif", ".h5"))

    with h5py.File(output, "w") as f:
        for n in range(0, N):
            grp = f.create_group(identifiers[n])
            # grp.create_dataset("TIME_AMSR", data=times_amsr2[n])
            grp.create_dataset("XX_AMSR", data=xxs2[n])
            grp.create_dataset("YY_AMSR", data=yys2[n])   

            grp.create_dataset("SIC_MODIS", data=sic_sar2[n], compression = "gzip")
            grp.create_dataset("SIC_MODIS_COUNT", data=sic_counts2[n], compression = "gzip")
            grp.create_dataset("SIC_MODIS_VALID", data=sic_valid2[n], compression = "gzip")
            grp.create_dataset("RAW_MODIS_MAX", data=raw_modis_max2[n], compression = "gzip")
            grp.create_dataset("RAW_MODIS_MIN", data=raw_modis_min2[n], compression = "gzip")
            grp.create_dataset("RAW_MODIS_MEAN", data=raw_modis2[n], compression = "gzip")
            
            for field_sic in fields_sic:
                grp.create_dataset(f"{field_sic}", data=sic_amsr2[n][field_sic], compression = "gzip")
            
            for field_tb in fields_tb:
                grp.create_dataset(f"{field_tb}", data=tb_amsr2[n][field_tb], compression = "gzip")

            grp.attrs["FLAG"] = flags[n]
            grp.attrs["AMSR_FILE"] = identifiers[n]
            grp.attrs["solar_zenith"] = sol_zenith2[n]
            grp.attrs["sensor_zenith"] = sen_zenith2[n]

        # Attributes (metadata) at file, group, and dataset level
        f.attrs["projection"] = amsr_crs
        f.attrs["gridid"] = gridid
        f.attrs["MODIS_FILE"] = os.path.basename(fn)

    print(f"HDF5 file '{output}' created successfully.")



 
def get_tb_from_nc(fn, field, nt = 0, group = None):
    # Read tb from netcdf file
    # MEMORY FIX: slice the swath *before* converting to numpy, so only one
    # swath is read from disk instead of all swaths in the file.

    if group:
        field = group + "/" + field
        field_x = group + "/x"
        field_y = group + "/y"

    with netCDF4.Dataset(fn, "r") as ds:
        if "_daily" in field:
            value = np.array(ds[field][:])
        else:
            value = np.array(ds[field][nt])
        x = np.array(ds[field_x])
        y = np.array(ds[field_y])
        xx, yy = np.meshgrid(x, y)

    return xx, yy, value


from functools import lru_cache
import gc


def _read_fields(fn, fields, nt, group):
    """Read several fields from one netCDF file in a single open.
    Swath fields read only swath `nt` (not every swath in the file)."""
    out = {}
    with netCDF4.Dataset(fn, "r") as ds:
        x = np.array(ds[f"{group}/x"])
        y = np.array(ds[f"{group}/y"])
        landmask = np.array(ds[grid]['land_mask']) # 0: ocean; 1: land
        oceanmask = np.where(landmask == 0, 1, np.nan).astype("float32")
        
        for field in fields:
            var = ds[f"{group}/{field}"]
            if "_daily" in field:
                out[field] = np.array(var[:]) * oceanmask
            else:
                out[field] = np.array(var[nt]) * oceanmask
    return x, y, out


@lru_cache(maxsize = 4)
def read_amsr_swath(fn_tb, fn_sic, nt, group):
    """SPEED FIX: many MODIS granules on one day match the same AMSR2 swath.
    The result is cached, so each (day, swath) is read from disk only once.
    The cache is cleared at the start of every new date.
    The returned dicts are shared through the cache: never modify them."""
    x1, y1, tb = _read_fields(fn_tb, fields_tb, nt, group)
    _, _, sic = _read_fields(fn_sic, fields_sic, nt, group)
    with netCDF4.Dataset(fn_tb, "r") as ds:
        amsr_crs = ds['grid_25km']['crs'].epsg_code
    return x1, y1, tb, sic, amsr_crs


def load_modis(tif_path, proj, block_rows = 256):
    """MEMORY FIX: read the MODIS ice mask and raw image once per file.
    - float32 instead of float64
    - only the bands that are actually used (rgb2, rgb3, cloud/land masks were never used)
    - zenith angles reduced to their mean right away
    - 1D coordinates only; the extent in `proj` is computed exactly, in row blocks."""

    ### MODIS ice mask
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        tags = src.tags()
        # discard (0), good (1), thin ice (2) or cloud (3)
        flag = int(tags['QUALITY_FLAG']) if 'QUALITY_FLAG' in tags else -1

    ok = (data >= 0) & (data <= 1)          # water/ice pixels
    valid = ok.astype(np.float32)           # Valid = 1; invalid (cloud and land) = 0
    data2 = data.astype(np.float32)
    data2[~ok] = np.nan
    del data

    ### MODIS raw image (band 1 + zenith angles)
    with rasterio.open(tif_path.replace("uint8_land", "float32")) as src:
        H, W = src.height, src.width
        tr = src.transform
        src_crs = src.crs

        rgb1 = src.read(1).astype(np.float32, copy = False)
        rgb1[~ok] = np.nan
        sen_zenith_mean = np.nanmean(src.read(4))
        sol_zenith_mean = np.nanmean(src.read(5))
    del ok

    # 1D pixel-center coordinates (same values as xs[0, :] and ys[:, 0] before)
    modis_x = np.array(rasterio.transform.xy(tr, np.zeros(W, dtype=int), np.arange(W))[0])
    modis_y = np.array(rasterio.transform.xy(tr, np.arange(H), np.zeros(H, dtype=int))[1])

    x0 = (modis_x.min() + modis_x.max()) / 2
    y0 = (modis_y.min() + modis_y.max()) / 2

    # Extent of all pixel centers in the target CRS, block by block
    transformer = Transformer.from_crs(src_crs, proj, always_xy=True)
    xmin2, xmax2, ymin2, ymax2 = np.inf, -np.inf, np.inf, -np.inf
    for r0 in range(0, H, block_rows):
        xx_b, yy_b = np.meshgrid(modis_x, modis_y[r0:r0 + block_rows])
        xb, yb = transformer.transform(xx_b, yy_b)
        xb, yb = np.asarray(xb), np.asarray(yb)
        xmin2, xmax2 = min(xmin2, xb.min()), max(xmax2, xb.max())
        ymin2, ymax2 = min(ymin2, yb.min()), max(ymax2, yb.max())
        del xx_b, yy_b, xb, yb

    # Source DataArrays for resampling, built once per file
    modis = convert_to_rio(data2, modis_x, modis_y, proj = src_crs)
    valid_mask = convert_to_rio(valid, modis_x, modis_y, proj = src_crs)
    modis_rgb = convert_to_rio(rgb1, modis_x, modis_y, proj = src_crs)

    return dict(flag = flag, modis_x = modis_x, x0 = x0, y0 = y0,
                xmin2 = xmin2, xmax2 = xmax2, ymin2 = ymin2, ymax2 = ymax2,
                modis = modis, valid_mask = valid_mask, modis_rgb = modis_rgb,
                sen_zenith_mean = sen_zenith_mean, sol_zenith_mean = sol_zenith_mean)


def resample_modis(m, amsr2_x, amsr2_y, amsr_crs, sic_template):
    """Resample the MODIS data onto one AMSR2 target grid."""
    amsr2 = convert_to_rio(sic_template, amsr2_x, amsr2_y, proj = amsr_crs)

    modis_resampled = m["modis"].rio.reproject_match(amsr2, resampling=Resampling.sum)
    valid_count = m["valid_mask"].rio.reproject_match(amsr2, resampling=Resampling.sum)
    sic_sar0 = np.asarray((modis_resampled / valid_count * 100).data)

    raw_modis0 = np.asarray(m["modis_rgb"].rio.reproject_match(amsr2, resampling=Resampling.average).data)
    raw_modis0_max = np.asarray(m["modis_rgb"].rio.reproject_match(amsr2, resampling=Resampling.max).data)
    raw_modis0_min = np.asarray(m["modis_rgb"].rio.reproject_match(amsr2, resampling=Resampling.min).data)

    valid_count_arr = valid_count.values.squeeze()

    # maximum valid count for AMSR2 pixel
    modis_x = m["modis_x"]
    maximum_count = (np.nanmedian(amsr2_x[1:] - amsr2_x[:-1]) / np.nanmedian(modis_x[1:] - modis_x[:-1])) ** 2
    sic_valid0 = valid_count_arr / maximum_count

    return sic_sar0, sic_valid0, valid_count_arr, raw_modis0, raw_modis0_max, raw_modis0_min


def process_modis_file(fn, files_amsr_raw, files_amsr, amsr_list):
    """All work for one MODIS file. Large arrays are freed when it returns."""

    print("##### MODIS ice mask: ", os.path.basename(fn))

    times_amsr2 = []
    times_modis2 = []
    identifiers = []
    sic_amsr2 = []
    sic_sar2 = []
    sic_valid2 = []
    sic_counts2 = []
    raw_modis2 = []
    raw_modis_max2 = []
    raw_modis_min2 = []
    sol_zenith2 = []
    sen_zenith2 = []
    tb_amsr2 = []
    xxs2 = []
    yys2 = []
    flags = []

    t_modis_str = os.path.basename(fn).split("021KM.A")[1][:12]
    t_modis = datetime.strptime(t_modis_str, "%Y%j.%H%M")

    gdf = find_amsr_for_tiff(fn, t_modis, dh = 6)

    print(fn, t_modis, len(gdf))

    if len(gdf) == 0:
        return

    ### Read MODIS images once per file
    m = load_modis(fn, proj)

    # SPEED FIX: the AMSR2 target grid is the same for every swath of a day,
    # so the (expensive) MODIS resampling is done once per grid and reused.
    resampled_cache = {}

    for k in range(0, len(gdf)):
        t_amsr_str = gdf.loc[k, 'time_str']
        t_amsr = datetime.strptime(t_amsr_str, "%Y%m%d%H%M")
        amsr_id = gdf.loc[k, "identifier"]

        dsec = abs(t_modis - t_amsr).total_seconds()

        time1 = time.time()

        if not (dsec <= 3600 * 6 and amsr_id in amsr_list):
            continue

        nt = list(amsr_list).index(amsr_id)

        ##### Load brightness temperature and SIC (only swath nt, cached) #####
        x1, y1, tb_12km, sic_12km, amsr_crs = read_amsr_swath(files_amsr_raw, files_amsr, nt, gridid)

        ##### Geometry compare for MODIS & AMSR2 #####
        xx1, yy1 = np.meshgrid(x1, y1)
        dist = ((xx1 - m["x0"])**2 + (yy1 - m["y0"])**2) ** 0.5
        del xx1, yy1

        if not (np.nanmin(dist) <= 200000 and resample):
            continue

        xind = np.where((x1 > m["xmin2"]) & (x1 < m["xmax2"]))[0]
        yind = np.where((y1 > m["ymin2"]) & (y1 < m["ymax2"]))[0]
        j1, j2 = xind[0], xind[-1]
        i1, i2 = yind[0], yind[-1]

        amsr2_y = y1[i1+1:i2]
        amsr2_x = x1[j1+1:j2]
        amsr2_xx, amsr2_yy = np.meshgrid(amsr2_x, amsr2_y)

        # .copy() the small subsets so the full AMSR grids are not kept alive
        tb_rep = {f: tb_12km[f][i1+1:i2, j1+1:j2].copy() for f in fields_tb}
        sic_rep = {f: sic_12km[f][i1+1:i2, j1+1:j2].copy() for f in fields_sic}

        key = (str(amsr_crs), int(i1), int(i2), int(j1), int(j2))
        if key not in resampled_cache:
            resampled_cache[key] = resample_modis(m, amsr2_x, amsr2_y, amsr_crs, sic_rep[fields_sic[-1]])
        sic_sar0, sic_valid0, valid_count_arr, raw_modis0, raw_modis0_max, raw_modis0_min = resampled_cache[key]

        times_amsr2.append(t_amsr)
        times_modis2.append(t_modis)
        sic_amsr2.append(sic_rep)
        tb_amsr2.append(tb_rep)

        sic_sar2.append(sic_sar0)
        sic_valid2.append(sic_valid0)
        sic_counts2.append(valid_count_arr)
        xxs2.append(amsr2_xx)
        yys2.append(amsr2_yy)
        identifiers.append(amsr_id)
        flags.append(m["flag"])

        # MODIS raw data parameters
        raw_modis2.append(raw_modis0)
        raw_modis_max2.append(raw_modis0_max)
        raw_modis_min2.append(raw_modis0_min)

        sol_zenith2.append(m["sol_zenith_mean"])
        sen_zenith2.append(m["sen_zenith_mean"])

        elapsed_time = time.time() - time1
        print("MODIS:", t_modis, " AMSR:", t_amsr, " dt:", abs(t_modis - t_amsr), f" << {elapsed_time:.1f} seconds")

    if len(times_amsr2) > 0:
        generate_modis_hdf(fn, identifiers, times_amsr2, times_modis2, sic_amsr2, sic_sar2, sic_valid2,
                           sic_counts2, tb_amsr2, xxs2, yys2, flags, sol_zenith2, sen_zenith2,
                           raw_modis2, raw_modis_max2, raw_modis_min2, fields_sic, fields_tb, amsr_crs)
                           

import os, resource

def print_mem(label=""):
    """Print current and peak memory of this Python process (Linux)."""
    try:
        import psutil
        current = psutil.Process(os.getpid()).memory_info().rss / 1e9
    except ImportError:
        # Fallback without psutil: read current memory from /proc
        with open("/proc/self/status") as f:
            line = next(l for l in f if l.startswith("VmRSS"))
        current = int(line.split()[1]) / 1e6           # kB -> GB
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # kB -> GB on Linux
    print(f"[MEM] {label}: current {current:.2f} GB | peak {peak:.2f} GB", flush=True)


#########################################################################

#########################################################################

##### PARAMETERS ####################################################
# Configuration file
fn = "./config.yaml"
config = read_config(fn)

proj = config['crs'] #'EPSG:3413' #'EPSG:3408'

import sys

# Start and end date
startdate = str(sys.argv[1])
enddate = str(sys.argv[2])

year = startdate[:4]
month = startdate[4:6]
# startdate = config['times_date0'] #'2025-01-01'
# enddate = config['times_date1'] #'2026-01-01'

gridid = config["gridid"]

folder_amsr = config['dir_amsr']
folder_swsic = config['dir_swsic']
folder_tb = config['dir_tb']
folder_dasic = config['dir_dasic']
folder_modis = config['dir_modis']
folder_anc = config['dir_anc']

## Output figures and HDF files
folder_fig = config['dir_fig'].replace("/gridid/", f"/{gridid}/")
folder_out = config['dir_out'].replace("/gridid/", f"/{gridid}/")


print(folder_amsr)

# global folder_amsr, folder_swsic, folder_dasic, folder_modis, folder_s1, folder_fig, folder_anc, folder_out

print(gridid, year, month, folder_modis, os.path.exists(folder_modis))

fields_sic = ['sic_nasa_team_swath', 'sic_bootstrap_f_swath', 'sic_bootstrap_p_swath', 'sic_bristol_swath', 'sic_osisaf_swath', 'sic_sicci_swath', 'sic_bf_nt_swath',
              'sic_nasa_team_daily', 'sic_bootstrap_f_daily', 'sic_bootstrap_p_daily', 'sic_bristol_daily', 'sic_osisaf_daily', 'sic_sicci_daily', 'sic_bf_nt_daily']
              
fields_tb = ['tb18h', 'tb18v', 'tb36h', 'tb36v',
             'tb18h_daily_mean', 'tb18v_daily_mean', 'tb36h_daily_mean', 'tb36v_daily_mean']

########################################################################

##### MAIN TASK #####
### Use Pyresample (updated on July 15, 2026)
from pyresample import geometry, kd_tree

resample = True #False
mapping = True

# dates = [datetime(2017,12,31), datetime(2018,11,19), datetime(2019,11,7), datetime(2019,11,23), datetime(2022,12,11), datetime(2023,11,9)]
s1_dirs = glob.glob(f"{folder_modis}/*021KM*")

dates = []
adates = []

for s in s1_dirs:

    adate = os.path.basename(s).split("021KM.A")[1][:7]
    date = datetime.strftime(datetime.strptime(adate, "%Y%j"), "%Y%m%d")

    if (date not in dates) and (date >= startdate) and (date <= enddate):
        dates.append(date)
        adates.append(adate)

dates = sorted(dates)
adates = sorted(adates)

print(dates)
print(adates)

for n, d in enumerate(dates[:]):
    adate = adates[n]

    date0 = d

    print("\n", date0)
    files_amsr_raw = f"{folder_tb}/AMSR2_L1R_TB_PS_S_{date0}.nc"
    files_amsr = f"{folder_swsic}/AMSR2_L1R_SIC_PS_S_{date0}.nc"
    files_modis = glob.glob(f"{folder_modis}/*.A{adate}.*.tif")

    fn_amsr = f"{folder_swsic}/AMSR2_L1R_SIC_PS_S_{date0}.nc"

    # New date: drop cached AMSR2 swaths from the previous date
    read_amsr_swath.cache_clear()

    if os.path.exists(fn_amsr):
        with netCDF4.Dataset(fn_amsr, "r") as ds:
            amsr_list = [f.replace(".nc", "") for f in ds['swath_file']]
    else:
        files_modis = []

    for fn in files_modis[:]:
        process_modis_file(fn, files_amsr_raw, files_amsr, amsr_list)
        gc.collect()   # make sure everything from this file is released
        print_mem(f"after  {os.path.basename(fn)}")

    print("========================================================")
