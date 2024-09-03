import xarray as xr
import numpy as np
import seawater
import glob


if True:
    names = glob.glob(f'LineP/*.nc')
    names.sort()

    station = xr.Dataset(coords={'depth':np.arange(0.5,5000.4), 'cast': np.arange(len(names))})
    Nd = len(station.depth)
    Nc = len(station.cast)
    station['temperature'] = (('depth', 'cast'), np.zeros((Nd, Nc))*np.nan)

    station['pressure'] = (('depth', 'cast'), np.zeros((Nd, Nc))*np.nan)
    station['salinity'] = (('depth', 'cast'), np.zeros((Nd, Nc))*np.nan)
    station['oxygen'] = (('depth', 'cast'), np.zeros((Nd, Nc))*np.nan)
    station.oxygen.attrs['units'] = 'mL/L'

    station['conductivity'] = (('depth', 'cast'), np.zeros((Nd, Nc))*np.nan)
    station.conductivity.attrs['units'] = 'S/m'

    station['time'] = (('cast'), np.zeros(Nc, dtype='datetime64[ns]'))
    station['longitude'] = (('cast'), np.zeros(Nc))
    station['latitude'] = (('cast'), np.zeros(Nc))
    station['mission_id'] = (('cast'), Nc * ['              '])
    station['project'] = (('cast'), Nc * ['              '])
    station['instrument_model'] = (('cast'), Nc * ['              '])
    station['instrument_serial_number'] = (('cast'), Nc * ['              '])
    station['filename'] = (('cast'), Nc * ['              '])

    for nn, name in enumerate(names):
        print(name)
        with xr.open_dataset(name) as ds:
            for td in ['latitude', 'longitude', 'time', 'project', 'mission_id', 'instrument_model', 'instrument_serial_number', 'filename']:
                if td in ds:
                    station[td][nn] = ds[td]
            keys = ''
            for key in ds.keys():
                keys += ' ' + key
            # print(keys)
            if 'sea_water_temperature' in ds:
                station['temperature'][:, nn] = np.interp(station.depth, ds.depth, ds['sea_water_temperature'],
                                                        right=np.nan, left=np.nan)
            else:
                for k in ds.keys():
                    if 'TEMP' in k:
                        station['temperature'][:, nn] = np.interp(station.depth, ds.depth, ds[k],
                                                        right=np.nan, left=np.nan)

            if 'sea_water_practical_salinity' in ds:
                station['salinity'][:, nn] = np.interp(station.depth, ds.depth, ds['sea_water_practical_salinity'],
                                                        right=np.nan, left=np.nan)
            else:
                for k in ds.keys():
                    if 'SALT' in k:
                        station['salinity'][:, nn] = np.interp(station.depth, ds.depth, ds[k],
                                                        right=np.nan, left=np.nan)

            if 'DOXYZZ01' in ds:
                station['oxygen'][:, nn] = np.interp(station.depth, ds.depth, ds['DOXYZZ01'],
                                                        right=np.nan, left=np.nan)
            if 'CNDST01' in ds:
                station['conductivity'][:, nn] = np.interp(station.depth, ds.depth, ds['CNDST01'],
                                                        right=np.nan, left=np.nan)

            if 'CNDST01' in ds:
                station['conductivity'][:, nn] = np.interp(station.depth, ds.depth, ds['CNDST01'],
                                                        right=np.nan, left=np.nan)
            if 'sea_water_pressure' in ds:
                station['pressure'][:, nn] = np.interp(station.depth, ds.depth, ds['sea_water_pressure'],
                                                        right=np.nan, left=np.nan)


    station.to_netcdf(f'_AllCastsLineP.nc')

# get lat/lon of stations:

dtypes = [('Stations', 'S5'), ('Lat', '<f8'), ('Latmin', '<f8'), ('Lon', '<f8'), ('Lonmin', '<f8'), ('depth', '<f8'), ('activity', 'S20')]

stationinfo = np.genfromtxt('./LinePStations.txt', dtype=dtypes, delimiter=' ', names=True, )
N = len(stationinfo)-3
station_info = xr.Dataset(coords={'stationind': np.arange(N)})
station_info['id'] = (('stationind'), ['     '] * N)
station_info['lon'] = (('stationind'), np.nan * np.zeros(N))
station_info['lat'] = (('stationind'), np.nan * np.zeros(N))
station_info['depth'] = (('stationind'), np.nan * np.zeros(N))
station_info['alongx'] = (('stationind'),  np.zeros(N))
for nn in range(N):
    station_info['id'][nn] = stationinfo[nn][0]
    station_info['lon'][nn] = - stationinfo[nn][3] - stationinfo[nn][4] / 60.
    station_info['lat'][nn] = stationinfo[nn][1] + stationinfo[nn][2] / 60.
    station_info['depth'][nn] = stationinfo[nn][5]

# lets defeine an alongline x...
dist, head = seawater.dist(station_info.lat, station_info.lon)
print(dist)
station_info['alongx'][1:] = np.cumsum(dist)
station_info['alongx'] = -(station_info['alongx'] - station_info['alongx'][0])
station_info.to_netcdf('LinePStations.nc')

# Mission ids:

missions = []
with xr.open_dataset('_AllCastsLineP.nc') as ds:
    for mi in ds.mission_id.values:
        if not mi in missions:
            missions += [mi]
print(missions)

# Make the initial grid:

with xr.open_dataset('_AllCastsLineP.nc') as ds:
    Nmission = len(missions)
    Nstations = 27
    grid = xr.Dataset(coords={'depth': ds.depth.values, 'mission_ind': np.arange(Nmission), 'station_ind': np.arange(Nstations)})
    for k in ds.keys():
        print(k)
        if ds[k].ndim == 2:
            grid[k] = (('depth', 'mission_ind', 'station_ind'), np.nan * np.zeros((len(ds.depth), Nmission, 27)))
        elif ds[k].ndim == 1:
            if ds[k].dtype == 'float64':
                grid[k] = (('mission_ind', 'station_ind'), np.nan * np.zeros((Nmission, 27)))
            elif ds[k].dtype == 'object':
                grid[k] = (('mission_ind', 'station_ind'),  [['            '] * 27] * Nmission)
            elif str(ds[k].dtype).startswith('<U'):
                grid[k] = (('mission_ind', 'station_ind'),  np.array([[''] * 27] * Nmission, dtype=ds[k].dtype))
            else:
                grid[k] = (('mission_ind', 'station_ind'), np.zeros( (Nmission, 27), dtype='datetime64[ns]'))

    print(grid)


    for castn in range(len(ds.cast)):
        if castn % 100 == 0:
            print(castn, len(ds.cast))
        cast = ds.isel(cast=castn)
        dx = np.sqrt((np.cos(50*np.pi/180)*(cast.longitude - station_info.lon))**2 +  (cast.latitude - station_info.lat)**2)
        if np.min(dx) > 0.12:
            station_ind = dx.argmin()

            #print(dx.min().values, f'too big, not at a station closest {station_info.id[station_ind].values}')
            continue
        station_ind = dx.argmin()
        #print(dx.min().values, f'from {station_info.id[station_ind].values}')
        # figure out mission index:
        nmission = missions.index(cast.mission_id.values)
        ngoodalready = np.isfinite(grid['temperature'][:, nmission, station_ind]).sum(axis=0)
        ngoodprofile = np.isfinite(cast['temperature']).sum(axis=0)
        if ngoodprofile > ngoodalready:
            for k in cast.keys():
                if cast[k].ndim == 1:
                    grid[k][:, nmission, station_ind] = cast[k]
                else:
                    grid[k][nmission, station_ind] = cast[k]


    grid['maxdepth'] = np.isfinite(grid.temperature).sum(dim='depth')
    grid.maxdepth.attrs = {'units': 'm', 'description':'maximum depth of data in profile'}
    grid['station_id'] = (('station_ind'), station_info.id.data)
    grid['station_alongx'] = (('station_ind'), station_info.alongx.data)
    grid.station_alongx.attrs = {'units':'km along LineP; P1=0, P26=-1423', 'description': 'nominal station location'}
    grid['alongx'] = (('mission_ind', 'station_ind'), np.interp(grid.longitude, station_info.lon, station_info.alongx))
    grid.alongx.attrs = {'units':'km along LineP; P1=0, P26=-1423', 'description': 'station location interpolated by longitude'}

    grid.to_netcdf('_InitialGridNew.nc')



with xr.open_dataset('_InitialGridNew.nc') as grid:

    # Get some mission stats
    grid['ncasts'] = (('mission_ind'), np.zeros(len(grid.mission_ind)))
    grid['furthest'] = (('mission_ind'), np.zeros(len(grid.mission_ind)))
    grid['mission'] = (('mission_ind'), len(grid.mission_ind) * ['            '])
    grid['mission_time'] = (('mission_ind'), np.zeros(len(grid.mission_ind), dtype='datetime64[ns]'))
    grid['station_name'] = (('station_ind'), len(grid.station_ind) * ['   '])
    for ind in grid.station_ind:
        grid['station_name'][ind] = station_info.id[ind]

    for ind in grid.mission_ind:
        good = np.where(grid.maxdepth[ind, :]>500)[0]
        grid['ncasts'][ind] = len(good)
        if len(good) > 0:
            grid['furthest'][ind] = max(good)

        goodt = np.where(grid.time[ind, :] != np.datetime64('1970-01-01T00:00:00'))[0]
        if len(goodt):
            grid['mission_time'][ind] = grid.time[ind, :][goodt].mean()

        # trim to just one d
        if True:
            for nn in grid.station_ind:
                if len(str(grid['mission_id'][ind, nn].values).strip())>0:
                    # print(str(grid['mission_id'][ind, nn].values))
                    grid['mission'][ind] = str(grid['mission_id'][ind, nn].values).strip()
                    continue
    good = np.where((grid.ncasts>4) & (grid.furthest>9))[0]
    grid = grid.isel(mission_ind=good)
    grid['mission'] = grid.mission.astype('S8')
    grid['potential_density'] = (('depth', 'mission_ind', 'station_ind'),
                                    seawater.pden(grid.salinity, grid.temperature, grid.pressure, 0))
    grid.potential_density.attrs = {'units':'kg m^-3', 'description':'potential density relative to 0 dbar'}
    grid['potential_temperature'] = (('depth', 'mission_ind', 'station_ind'),
                                        seawater.ptmp(grid.salinity, grid.temperature, grid.pressure, 0))
    grid.potential_temperature.attrs = {'units':'C', 'description':'potential temperature relative to 0 dbar'}
    grid.time.encoding = {'units': 'days since 1900-01-01', 'dtype':'float64', 'calendar':'proleptic_gregorian'}
    grid.mission_time.encoding = {'units': 'days since 1900-01-01', 'dtype':'float64', 'calendar':'proleptic_gregorian'}

    grid.attrs = {'Conventions': 'CF-1.11',
                'title': 'Line P cruise grid',
                'institution': 'Institute of Ocean Sciences, Sidney, BC, Canada',
                'creator': 'Jody Klymak',
                'creator_institution': 'University of Victoria, BC, Canada',
                'email': 'jklymak@uvic.ca',
                'source': 'Ship CTD data, collected since 1968 along LineP by IOS',
                'history': 'CTD data from waterproperties.ca, defined area "LineP".  Binned by ProcessStations.ipynb',
                'comment': 'All casts in "LineP" area were downloaded from waterproperties.ca. '
                            'Profile data was approximately 1-m, but not exactly (for some reason), so '
                            'depth was interpolated to exactly 1-m (leading to a bit of interpolation '
                            'smoothing).  Using the nominal stations locations, casts were assigned to '
                            'a station if they were within 0.12 degrees latitude of the station.  For each '
                            'mission, the deepest profile at each station was retained.  A mission was '
                            'retained if it has at least 4 casts, and the furthest cast was at least  '
                            'as far out as P10.',
                'references': 'https://github.com/jklymak/LinePGrid/',
                'keywords': 'CTD, Oceans, Ocean Pressure, '
                            'Water Pressure, Oceans, Ocean Temperature, Water Temperature, '
                            'Oceans, Salinity/Density, Conductivity, Oceans, '
                            'Salinity/Density, Density, Oceans, Salinity/Density, Salinity',
                'keywords_vocabulary': 'GCMD Science Keywords',
                'license': 'This data may be redistributed and used without restriction or warranty',
                'sea_name': 'Coastal Waters of Southeast Alaska and British Columbia',
                'standard_name_vocabulary': 'CF STandard Name Table v85'
                }
    grid.depth.attrs = {
        'standard_name': 'depth',
        'long_name': 'depth of CTD [m]',
        'units': 'm',
        'positive': 'down',
        'comment': 'Data was interpolated onto 1-m grid from data that '
                'was almost on a 1-m grid.  There will be some smoothing.'
    }
    grid.mission_ind.attrs = {
        'long_name': 'mission index',
        'comment': 'index into the mission list'
    }
    grid.station_ind.attrs = {
        'long_name': 'mission index',
        'comment': 'index in the station list.  Note that P1 is 0, P25 is 24 '
                'P35 is 25 and P26 is 26.'
    }
    grid.temperature.attrs = {
        'standard_name': 'sea_water_temperature',
        'units': 'Celsius',
        'long_name': 'Temperature [C]',
        'comment': 'From various CTDs so variable accuracy'
    }
    grid.pressure.attrs = {
        'standard_name': 'sea_water_pressure',
        'units': 'dbar',
        'long_name': 'Pressure [dbar]'
    }
    grid.salinity.attrs = {
        'standard_name': 'sea_water_practical_salinity',
        'units': 'psu',
        'long_name': 'Salinity [psu]',
        'comment': 'From various CTDs, probably from EOS80'
    }
    grid.oxygen.attrs = {
        'standard_name': 'mole_concentration_of_dissolved_molecular_oxygen_in_sea_water',
        'units': 'umol l-1',
        'long_name': 'O2 concentration [umol l^-1]',
        'comment': 'Many older CTDs did not have O2'
    }
    grid.conductivity.attrs = {
        'standard_name': 'sea_water_electrical_conductivity',
        'units': 'S m-1',
        'long_name': 'Conductivity [S m^-1]',
        'comment': 'Conductivty not reported for older CTDs'
    }
    grid.potential_density.attrs = {
        'standard_name': 'sea_water_potential_density',
        'long_name': 'potential density [kg m-3]',
        'comment': 'potential density relative to 0 dbar, using EOS80',
        'units': 'kg m-3'}

    grid.potential_temperature.attrs = {
        'standard_name': 'sea_water_potential_temperature',
        'long_name': 'potential temperature [C]',
        'units': 'Celsius',
        'comment': 'potential temperature relative to 0 dbar, using EOS80'}

    grid.time.attrs['standard_name'] = 'time'
    grid.time.attrs['long_name'] = 'CTD cast time'
    grid.mission_time.attrs['standard_name'] = 'time'
    grid.mission_time.attrs['long_name'] = 'Average mission time'

    grid.maxdepth.attrs['long_name'] = 'Maximum Cast depth'
    grid.maxdepth.attrs['positive'] = 'down'
    grid.maxdepth.attrs['comment'] = 'Maximum cast depth for this station for this mission'

    grid.alongx.attrs['long_name'] = 'distance along line P [km]'
    grid.alongx.attrs['units'] = 'km'
    grid.alongx.attrs['comment'] = ('km along LineP; P1=0, P26=-1423; negative '
                            'because lineP goes to the west')

    grid.station_alongx.attrs['long_name'] = 'distance along line P [km]'
    grid.station_alongx.attrs['units'] = 'km'
    grid.station_alongx.attrs['comment'] = ('Nominal station km along LineP; P1=0, P26=-1423; negative '
                            'because lineP goes to the west')

    grid.latitude.attrs = {'long_name': 'latitude [N]',
                        'standard_name': 'latitude',
                        'units':        'degrees_north'}

    grid.longitude.attrs = {'long_name': 'longitude [E]',
                        'standard_name': 'longitude',
                        'units':        'degrees_east',
                        'comment': 'west is less than zero'}

    grid.station_id.attrs = {'long_name': 'Station Name'}
    grid.mission.attrs = {'long_name': 'Mission Name',
                        'comment': 'IOS mission name. Note these '
                                    'are not necessarily in chronological '
                                    'order'}
    grid.mission.attrs = {'long_name': 'Mission Name',
                        'comment': 'IOS mission name. Note these '
                                    'are not necessarily in chronological '
                                    'order'}

    # get station lon and lat
    with xr.open_dataset('LinePStations.nc') as stations:
        grid['station_lon'] = ('station_ind', stations.lon.data)
        grid.station_lon.attrs = {'long_name': 'longitude [E]',
                        'standard_name': 'longitude',
                        'units':        'degrees_east',
                        'comment': 'Nominal station location. west is less than zero'}

        grid['station_lat'] = ('station_ind', stations.lat.data)
        grid.station_lat.attrs = {'long_name': 'latitude [N]',
                        'standard_name': 'latitude',
                        'units':        'degrees_north',
                        'comment': 'Nominal station location'}

    grid.to_netcdf('LinePGrid.nc')

