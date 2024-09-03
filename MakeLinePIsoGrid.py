import xarray as xr
import numpy as np

mintime = np.datetime64('2000-01-01')
maxtime = np.datetime64('2020-01-01')
with xr.open_dataset('./LinePGrid.nc') as grid:
    inds = np.where((grid.mission_time >= mintime)  & (grid.mission_time <= maxtime))[0]
    dd = grid.potential_density[:,inds].mean(dim=('mission_ind', 'station_ind'))
    dd = dd.sortby(dd)
    grid['meanpden'] = ('depth', dd.values)
    isogrid = xr.Dataset(coords={'isodepths':grid.depth.values, 'mission_ind': grid.mission_ind.values, 'station_ind': grid.station_ind.values})
    goodpd = np.isfinite(grid.meanpden).values
    isogrid['mean_potential_density'] = ('isodepths', grid.meanpden.values)
    for k in grid.keys():
        if ('depth' in grid[k].dims) and (grid[k].ndim > 1):
            print('Yes')
            isogrid[k] = (('isodepths', 'mission_ind', 'station_ind'), np.nan * grid[k].data)
            for nn in range(len(grid.mission_ind)):
                for mm in range(len(grid.station_ind)):
                    good = (np.isfinite(grid[k][:, nn, mm] + grid.potential_density[:, nn, mm])).values
                    if np.sum(good) > 1:
                        isogrid[k][goodpd, nn, mm] = np.interp(grid.meanpden[goodpd],
                                                      grid.potential_density[good, nn, mm].values,
                                                      grid[k][good, nn, mm].values, left=np.nan, right=np.nan)
        elif k not in ['depth', 'meanpden']:
            isogrid[k] = grid[k]
    # get the depth
    isogrid['isopycnal_depth'] = (('isodepths', 'mission_ind', 'station_ind'), np.nan * isogrid['temperature'].data)
    for nn in range(len(grid.mission_ind)):
        for mm in range(len(grid.station_ind)):
            good = (np.isfinite(grid.potential_density[:, nn, mm])).values
            if np.sum(good) > 1:
                isogrid['isopycnal_depth'][goodpd, nn, mm] = np.interp(grid.meanpden[goodpd],
                                              grid.potential_density[good, nn, mm].values,
                                              grid['depth'][good].values, left=np.nan, right=np.nan)
    #isogrid = isogrid.reset_coords(['depth'])

    isogrid.attrs = grid.attrs.copy()
    isogrid.attrs['title'] = 'Line P cruise grid, on isopycnals'
    isogrid.attrs['history'] += ' Interpolated onto isopycnals in MakeIsoGrid.ipynb'
    isogrid.attrs['comment'] += (f' Isopycnal mean profile is average of all cruises between '
                                 f'{mintime:10s} and {maxtime:10s}')
    isogrid.isopycnal_depth.attrs = {
        'standard_name': 'depth',
        'long_name': 'depth of isopycnal [m]',
        'units': 'm',
        'positive': 'down',
        'comment': 'Depth of the isopycnal in the original CTD cast'
    }
    isogrid.isodepths.attrs = {
        'standard_name': 'depth',
        'long_name': 'depth of isopycnal [m]',
        'units': 'm',
        'positive': 'down',
        'comment': f'Depth of the isopycnal the mean.  Mean is all cruises between {mintime:10s} and {maxtime:10s}'
    }
    isogrid.mission_ind.attrs = {
        'long_name': 'mission index',
        'comment': 'index into the mission list'
    }
    isogrid.station_ind.attrs = {
        'long_name': 'mission index',
        'comment': 'index in the station list.  Note that P1 is 0, P25 is 24 '
                   'P35 is 25 and P26 is 26.'
    }
    isogrid.temperature.attrs = {
        'standard_name': 'sea_water_temperature',
        'units': 'Celsius',
        'long_name': 'Temperature [C]',
        'comment': 'From various CTDs so variable accuracy.  Interpolated onto mean isopycnals'
    }
    isogrid.pressure.attrs = {
        'standard_name': 'sea_water_pressure',
        'units': 'dbar',
        'long_name': 'Pressure [dbar]',
        'comment': 'Interpolated onto mean isopycnals'
    }
    isogrid.salinity.attrs = {
        'standard_name': 'sea_water_practical_salinity',
        'units': 'psu',
        'long_name': 'Salinity [psu]',
        'comment': 'From various CTDs, probably from EOS80, Interpolated onto mean isopycnals'
    }
    isogrid.oxygen.attrs = {
        'standard_name': 'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water',
        'units': 'umol l-1',
        'long_name': 'O2 concentration [umol l^-1]',
        'comment': 'Many older CTDs did not have O2, Interpolated onto mean isopycnals'
    }
    isogrid.conductivity.attrs = {
        'standard_name': 'sea_water_electrical_conductivity',
        'units': 'S m-1',
        'long_name': 'Conductivity [S m^-1]',
        'comment': 'Conductivty not reported for older CTDs, Interpolated onto mean isopycnals'
    }
    isogrid.potential_density.attrs = {
        'standard_name': 'sea_water_potential_density',
        'long_name': 'potential density [kg m-3]',
        'comment': 'potential density relative to 0 dbar, using EOS80. Interpolated onto mean isopycnals',
        'units': 'kg m-3'}

    isogrid.potential_temperature.attrs = {
        'standard_name': 'sea_water_potential_temperature',
        'long_name': 'potential temperature [C]',
        'units': 'Celsius',
        'comment': 'potential temperature relative to 0 dbar, using EOS80. Interpolated onto mean isopycnals'
    }

    isogrid.time.attrs['standard_name'] = 'time'
    isogrid.time.attrs['long_name'] = 'CTD cast time'
    isogrid.mission_time.attrs['standard_name'] = 'time'
    isogrid.mission_time.attrs['long_name'] = 'Average mission time'

    isogrid.maxdepth.attrs['long_name'] = 'Maximum Cast depth'
    isogrid.maxdepth.attrs['positive'] = 'down'
    isogrid.maxdepth.attrs['comment'] = 'Maximum cast depth for this station for this mission'

    isogrid.alongx.attrs['long_name'] = 'distance along line P [km]'
    isogrid.alongx.attrs['units'] = 'km'
    isogrid.alongx.attrs['comment'] = ('km along LineP; P1=0, P26=-1423; negative '
                             'because lineP goes to the west')

    isogrid.station_alongx.attrs['long_name'] = 'distance along line P [km]'
    isogrid.station_alongx.attrs['units'] = 'km'
    isogrid.station_alongx.attrs['comment'] = ('Nominal station km along LineP; P1=0, P26=-1423; negative '
                             'because lineP goes to the west')

    isogrid.latitude.attrs = {'long_name': 'latitude [N]',
                           'standard_name': 'latitude',
                           'units':        'degrees_north'}

    isogrid.longitude.attrs = {'long_name': 'longitude [E]',
                           'standard_name': 'longitude',
                           'units':        'degrees_east',
                           'comment': 'west is less than zero'}

    isogrid.station_id.attrs = {'long_name': 'Station Name'}
    isogrid.mission.attrs = {'long_name': 'Mission Name',
                         'comment': 'IOS mission name. Note these '
                                    'are not necessarily in chronological '
                                    'order'}
    isogrid.mission.attrs = {'long_name': 'Mission Name',
                         'comment': 'IOS mission name. Note these '
                                    'are not necessarily in chronological '
                                    'order'}

    # get station lon and lat
    with xr.open_dataset('LinePStations.nc') as stations:
        isogrid['station_lon'] = ('station_ind', stations.lon.data)
        isogrid.station_lon.attrs = {'long_name': 'longitude [E]',
                           'standard_name': 'longitude',
                           'units':        'degrees_east',
                           'comment': 'Nominal station location. west is less than zero'}

        isogrid['station_lat'] = ('station_ind', stations.lat.data)
        isogrid.station_lat.attrs = {'long_name': 'latitude [N]',
                           'standard_name': 'latitude',
                           'units':        'degrees_north',
                           'comment': 'Nominal station location'}
    isogrid['mean_potential_density'].attrs = {
        'long_name': 'mean potential density [kg m^-3]',
        'units': 'kg m-3',
        'comment': f'density of this isopycnal:  Mean from {mintime:10s} to {maxtime:10s}'
    }
    isogrid.to_netcdf('LinePIsoGrid.nc')