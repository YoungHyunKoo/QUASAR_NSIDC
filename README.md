# QUASAR_NSIDC
QUASAR workflow pipeline for NSIDC
- Code 1: Download AMSR2 L1R raw files over the Southern Ocean
  ```
  python C1_download_AMSR.py 20200101 20200131
  ```
- Code 2: Reproject AMSR2 L1R files (in input directory) onto a polar stereographic grid (output saved in output directory)
  ```
  python C2_convert_grid.py --input-dir /data/amsr2 --output-dir /data/gridded --hemisphere north --datefirst 20200101 --datelast 20200131
  ```
  
- Code 3: Calculate sea ice concentration (individual-swath and/or daily-mean): Bootstrap (BT), NASA Team (NT), OSI-SAF, etc.
  ```
  python C3_compute_SIC.py --input-dir /data/gridded/AMSR2_L1R_TB_PS_S*.nc --output-dir /data/sic --datefirst 20200101 --datelast 20200131
  ```
  
- Code 4: Compare AMSR2 sea ice concentration with high-resolution ice concentration from MODIS and Sentinel-1 (pixel-wise SIC comparison)
  **Sentinel-1 comparison**
  ```
  python C4_S1_AMSR_compare.py 20200101 20200131
  ```
  **MODIS comparison**
  ```
  python C4_MODIS_AMSR_compare.py 20200101 20200131
  ```
  
