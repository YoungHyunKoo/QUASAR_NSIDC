# QUASAR_NSIDC
QUASAR workflow pipeline for NSIDC
- Code 1: Download AMSR2 L1R raw files
  ```
  Python Download_AMSR2.py
  ```
- Code 2: Reproject AMSR2 L1R files onto an Antarctic stereographic grid
- Code 3: Calculate sea ice concentration (individual-swath and/or daily-mean): Bootstrap (BT), NASA Team (NT), OSI-SAF, etc.
- Code 4: Compare AMSR2 sea ice concentration with high-resolution ice concentration from MODIS and Sentinel-1
