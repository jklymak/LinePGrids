# GridLineP

Routines for gridding LineP data.

These routines make netcdf files that are organized by depth, cruise number,
and station for the nominal Line-P stations.  Data starts in 1968.

A CTD cast is considered at a LineP station if it is within 7.2 nm of the
nominal station location.

A cruise is considered a LineP cruise if it has at least four occupied
stations, and it made it as far offshore as P10.

## download raw netcdf files

```
wget -i LineP/wget_netcdf_file_download_list.csv
```

to download all the data from waterproerties.ca.  Note you may need an account to access this data.

To get this list, go to https://www.waterproperties.ca/data/

and search the map under "Defined Area": "Line P", and then file suffix "ctd", and then under "Download": "WGET netCDF"

## Process:

- run `MakeLinePGrid.py` to make `LinePGrid.nc` (and `LinePStations.nc`).
- run `MakeLinePIsoGrid.py` to make `LinePIsoGrid.nc`.




## Getting data:

See http://206.12.89.152:8080/thredds/catalog/otherdata/catalog.html

```
<xarray.Dataset> Size: 2GB
Dimensions:                   (depth: 5000, mission_ind: 204, station_ind: 27)
Coordinates:
  * depth                     (depth) float64 40kB 0.5 1.5 ... 4.998e+03 5e+03
  * mission_ind               (mission_ind) int64 2kB 1 4 5 7 ... 503 511 513
  * station_ind               (station_ind) int64 216B 0 1 2 3 4 ... 23 24 25 26
Data variables: (12/26)
    temperature               (depth, mission_ind, station_ind) float64 220MB ...
    pressure                  (depth, mission_ind, station_ind) float64 220MB ...
    salinity                  (depth, mission_ind, station_ind) float64 220MB ...
    oxygen                    (depth, mission_ind, station_ind) float64 220MB ...
    conductivity              (depth, mission_ind, station_ind) float64 220MB ...
    time                      (mission_ind, station_ind) datetime64[ns] 44kB ...
    ...                        ...
    mission_time              (mission_ind) datetime64[ns] 2kB ...
    station_name              (station_ind) <U3 324B ...
    potential_density         (depth, mission_ind, station_ind) float64 220MB ...
    potential_temperature     (depth, mission_ind, station_ind) float64 220MB ...
    station_lon               (station_ind) float64 216B ...
    station_lat               (station_ind) float64 216B ...
Attributes: (12/15)
    Conventions:               CF-1.11
    title:                     Line P cruise grid
    institution:               Institute of Ocean Sciences, Sidney, BC, Canada
    creator:                   Jody Klymak
    creator_institution:       University of Victoria, BC, Canada
    email:                     jklymak@uvic.ca
    ...                        ...
    references:                https://github.com/jklymak/LinePGrid/
    keywords:                  CTD, Oceans, Ocean Pressure, Water Pressure, O...
    keywords_vocabulary:       GCMD Science Keywords
    license:                   This data may be redistributed and used withou...
    sea_name:                  Coastal Waters of Southeast Alaska and British...
    standard_name_vocabulary:  CF STandard Name Table v85
```

and

```
<xarray.Dataset> Size: 2GB
Dimensions:                   (isodepths: 5000, mission_ind: 204,
                               station_ind: 27)
Coordinates:
  * isodepths                 (isodepths) float64 40kB 0.5 1.5 ... 5e+03
  * mission_ind               (mission_ind) int64 2kB 1 4 5 7 ... 503 511 513
  * station_ind               (station_ind) int64 216B 0 1 2 3 4 ... 23 24 25 26
Data variables: (12/28)
    mean_potential_density    (isodepths) float64 40kB ...
    temperature               (isodepths, mission_ind, station_ind) float64 220MB ...
    pressure                  (isodepths, mission_ind, station_ind) float64 220MB ...
    salinity                  (isodepths, mission_ind, station_ind) float64 220MB ...
    oxygen                    (isodepths, mission_ind, station_ind) float64 220MB ...
    conductivity              (isodepths, mission_ind, station_ind) float64 220MB ...
    ...                        ...
    station_name              (station_ind) <U3 324B ...
    potential_density         (isodepths, mission_ind, station_ind) float64 220MB ...
    potential_temperature     (isodepths, mission_ind, station_ind) float64 220MB ...
    station_lon               (station_ind) float64 216B ...
    station_lat               (station_ind) float64 216B ...
    isopycnal_depth           (isodepths, mission_ind, station_ind) float64 220MB ...
Attributes: (12/15)
    Conventions:               CF-1.11
    title:                     Line P cruise grid, on isopycnals
    institution:               Institute of Ocean Sciences, Sidney, BC, Canada
    creator:                   Jody Klymak
    creator_institution:       University of Victoria, BC, Canada
    email:                     jklymak@uvic.ca
    ...                        ...
    references:                https://github.com/jklymak/LinePGrid/
    keywords:                  CTD, Oceans, Ocean Pressure, Water Pressure, O...
    keywords_vocabulary:       GCMD Science Keywords
    license:                   This data may be redistributed and used withou...
    sea_name:                  Coastal Waters of Southeast Alaska and British...
    standard_name_vocabulary:  CF STandard Name Table v85
```

## Requirements:

- xarray
- numpy
- seawater
