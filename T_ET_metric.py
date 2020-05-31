#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
;;#############################################################################
;;
;; T_ET_metric.py
;; Author: Jess Baker (j.c.baker@leeds.ac.uk)
;; LAPSE project, part of CSSP Brazil
;; University of Leeds, UK
;;
;;#############################################################################
;;
;; Description
;;    This script calculates the correlation between T and ET for all grid
;;    cells in input array(s). Same at terrestrial-couling index but for T and
;;    ET only.
;;
;;    Used to infer soil moisture - ET relationships in regions where SM not
;;    available. Where ET fluxes are moisture limited, as SM increases, ET 
;;    increases and thus T decreases as a result of elevated latent heat
;;    fluxes. Therefore, a negative T_ET correlation is indicative of a land
;;    surface control on the atmosphere. On the other hand, a positive T_ET
;;    correlation suggests that surface fluxes are radiation limited, and
;;    therefore the surface state is responsive to atmospheric forcing as
;;    opposed to the other way around. The T_ET metric is calculated using
;;    monthly anomalies from the climatological seasonal cycle (T’ and ET’),
;;    and determining the Pearson’s correlation coefficient between them:
;;                            T_ET = r(T', ET')
;;
;; Requirements
;;    Takes surface temperature and evapotranspiration as input variables. Data 
;;    should be formatted as Iris cubes, constrained to the same time period.
;;
;; References
;;    Seneviratne, S. I., Corti, T., Davin, E. L., Hirschi, M., Jaeger, E. B., 
;;    Lehner, I., Orlowsky, B. & Teuling, A. J. 2010. Investigating soil
;;    moisture–climate interactions in a changing climate: A review.
;;    Earth-Science Reviews, 99, 125-161.
;;
;;    Seneviratne, S. I., Lüthi, D., Litschi, M. & Schär, C. 2006.
;;    Land–atmosphere coupling and climate change in Europe. Nature, 443, 205.
;;
;;#############################################################################
"""

# Required Python packages
import numpy as np
import iris.coord_categorisation
import iris
import matplotlib.pyplot as plt
import copy
import os
from mpl_toolkits import basemap
from mpl_toolkits.basemap import maskoceans
from datetime import datetime
from scipy.stats import linregress as ols


def main(t_cube, et_cube, ols_out='r', wet_dry=False, constraint_yrs=None,
         pre_data_path=('/nfs/a68/gyjcab/datasets/lapse_data_harmonised/'
                        'Jan_2018/Final/1.0deg/'
                        'pr_trmm_3b43_mon_1.0deg_1998_2016.nc'),
         plotting=False, weighting=False, anom=True,
         plotting_args={'name': 'Temperature-Evapotranspiration Metric',
                        'lat_lims': [-60, 30],
                        'lon_lims': [-120, 180],
                        'levels': (-1, 1, 11)}):

    """

    This function uses temperature and evapotranspiration data to infer
    whether ET is controlled by surface moisture availability or energy
    limitation.

    Takes Iris cubes as input.

    Arguments
        t_cube = Iris cube of surface temperature (model or observations).
        et_cube = Iris cube of evapotranspiration (model or observations).
        ols_out = output from linear regression. Accepts 'slope' or 'r'.
        wet_dry = Boolean. Calculate metric using data from  6 wettest and 6
                  dryest months in each pixel (True) or using data from all
                  months (False).
        constraint_yrs = Length 2 array with start and end years of constraint.
        pre_data_path = If wet_dry is True, path for precipitation data used to
                        identify wet and dry months.
        plotting = Boolean. Plot output of metric. If False returns output of
                   metric as arrays only.
        weighting = Boolean. Weight output arrays by standard deviation of
                    denominator applied when calculating regression slopes.
                    Default is False.
        anom = Boolean. Calculate metric using anomalies from climatological
               seasonal cycle (True) or interannual monthly data
               (False).
        plotting_args = dictionary of plotting arguments, including name of
                        data being plotted (observations or name of model),
                        limits for output map, and colorbar levels.

    """
    
    if anom is True:
        # Calculate anomalies versus climatological seasonal cycle
        t_anom = monthly_anom_cube(t_cube)
        et_anom = monthly_anom_cube(et_cube)
    else:
        t_anom = t_cube
        et_anom = et_cube
        
    # Check if lats are ascending, if not then reverse
    t_anom = flip_lats(t_anom)
    et_anom = flip_lats(et_anom)
    
    # Reorder data from -180 to +180 degrees
    t_lon = t_anom.coord('longitude').points
    if any(i > 180 for i in t_lon) is True:
        t_anom = minus180_to_plus180(t_anom)

    et_lon = et_anom.coord('longitude').points
    if any(i > 180 for i in et_lon) is True:
        et_anom = minus180_to_plus180(et_anom)

    # Constrain data to required years
    if constraint_yrs is not None:
        constraint = iris.Constraint(time=lambda cell:
                                     constraint_yrs[0] <=
                                     cell.point.year <=
                                     constraint_yrs[1])
    else:
        constraint = None

    # Calculate for wet and dry months separately
    if wet_dry is True:

        # For each pixel identify wettest 6 months
        # Read in precipitation data
        try:
            data_path = (pre_data_path)
            pre_cube = iris.load_cube(data_path, constraint=constraint)
        except NameError:
            print('Need to specify filepath for precipitation data to '
                  'calculate wet/dry months')
            assert False

        # Regrid precipitation data to resolution of input array
        target_cube = t_anom
        scheme = iris.analysis.AreaWeighted(mdtol=0.5)
        pre_cube = pre_cube.regrid(target_cube, scheme)

        # Calculate seasonal cycle for each pixel
        iris.coord_categorisation.add_month(pre_cube, 'time', name='month')
        pre_mn = pre_cube.aggregated_by(['month'], iris.analysis.MEAN)

        # For all pixels get indices of wet months
        nyear = int(t_anom.shape[0]/12)
        wet_bool = np.zeros((nyear*12, pre_cube.shape[-2], pre_cube.shape[-1]))
        for ny in range(pre_mn.shape[-2]):
            for nx in range(pre_mn.shape[-1]):
                cycle = pre_mn.data[:, ny, nx]
                if np.nanmax(cycle) > 0:
                    wet_idx = sorted(range(12), key=lambda x: cycle[x])[-6:]
                    for yr in range(nyear):
                        for w in wet_idx:
                            wet_bool[w + 12*yr, ny, nx] = 1
                else:
                    wet_bool[:, ny, nx] = np.nan

        # Define dictionaries to hold output
        wet_arrays = {'t_et': None}
        wet_arrays = {'pval_array': None}
        dry_arrays = {'t_et': None}
        dry_arrays = {'pval_array': None}
        data_dict = {'wet': wet_arrays, 'dry': dry_arrays}

        # Calculate metric for wet and dry seasons
        for season in ['wet', 'dry']:
            print(season)
            t_et, pval_array = calculating_t_et(t_anom, et_anom, ols_out=ols_out,
                                  wet_bool=wet_bool, season=season,
                                  weighting=weighting)

            data_dict[season]['t_et'] = t_et
            data_dict[season]['pval_array'] = pval_array

            # Call plotting routine
            if plotting is True:

                # Define plotting variables
                name = plotting_args['name'] + ': ' + season + ' season'
                surf_name = t_anom.long_name
                flux_name = et_anom.long_name

                if ols_out == 'slope':
                    units = str(et_anom.units) + '/' + str(t_anom.units)

                elif ols_out == 'r':
                    units = ' '

                lat = t_anom.coord('latitude').points
                lon = t_anom.coord('longitude').points
                lat_lims = plotting_args['lat_lims']
                lon_lims = plotting_args['lon_lims']
                levels = plotting_args['levels']

                plot_t_et(name, surf_name, flux_name, t_et, units,
                         lat, lon, lat_lims, lon_lims, levs=levels)

        return(data_dict, wet_bool)

    # Calculate metric using data from all months
    else:
        t_et, pval_array = calculating_t_et(t_anom, et_anom, ols_out=ols_out,
                                          weighting=weighting)

        # Call plotting routine
        if plotting is True:

            # Define plotting variables
            name = plotting_args['name']
            surf_name = t_anom.long_name
            if surf_name is None:
                surf_name = t_anom.standard_name
            flux_name = et_anom.long_name
            if flux_name is None:
                flux_name = et_anom.standard_name

            if ols_out == 'slope':
                units = str(et_anom.units) + '/' + str(t_anom.units)
                if weighting is True:
                    units = str(et_anom.units)

            elif ols_out == 'r':
                units = ' '

            lat = t_anom.coord('latitude').points
            lon = t_anom.coord('longitude').points
            lat_lims = plotting_args['lat_lims']
            lon_lims = plotting_args['lon_lims']
            levels = plotting_args['levels']

            plot_t_et(name, surf_name, flux_name, t_et, units,
                     lat, lon, lat_lims, lon_lims, levs=levels)

        return(t_et, pval_array)


def calculating_t_et(t_anom, et_anom, ols_out='r',
                     wet_bool=None, season=None, weighting=False):

    # Define arrays to store data
    len_lat = t_anom.shape[-2]
    len_lon = t_anom.shape[-1]
    t_et = np.nan * np.empty((len_lat, len_lon))
    pval_array = np.nan * np.empty((len_lat, len_lon))

    for ny in range(len_lat):
        for nx in range(len_lon):

            # Extract data from one grid cell
            if wet_bool is not None:
                    if season == 'wet':
                        i, = np.where(wet_bool[:, ny, nx] == 1)
                        surf_temp = t_anom.data[i, ny, nx]
                        flux_temp = et_anom.data[i, ny, nx]

                    elif season == 'dry':
                        i, = np.where((wet_bool[:, ny, nx]) == 0)
                        surf_temp = t_anom.data[i, ny, nx]
                        flux_temp = et_anom.data[i, ny, nx]
            else:
                surf_temp = t_anom.data[:, ny, nx]
                flux_temp = et_anom.data[:, ny, nx]

            # 1. Find which months both surface and flux variables have data
            mask = ~np.isnan(surf_temp) & ~np.isnan(flux_temp)

            # Provided at least one month overlap proceed with calc
            if len(surf_temp[mask]) > 10:

                slope, intercept, r, p, std_err = ols(surf_temp[mask],
                                                      flux_temp[mask])

                # Save t_et and p value
                if ols_out == 'slope':
                    t_et[ny, nx] = slope
                elif ols_out == 'r':
                    t_et[ny, nx] = r
                    
                pval_array[ny, nx] = p

            # Weight by variability of denominator to emphasise 
            # places where actual impact is large
            if weighting is True:
                if (t_et[ny, nx] != -999.0):
                    t_et[ny, nx] = t_et[ny, nx] * np.std(surf_temp[mask])
    print(np.nanmin(t_et), np.nanmax(t_et))
    return(t_et, pval_array)


def plot_t_et(name, surf_var_name, flux_var_name, t_et, units,
             lat, lon, lat_lims, lon_lims, levs=(-10, 10, 11)):

    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)

    # Corners of subset map
    lat1 = lat_lims[0]
    lat2 = lat_lims[1]
    lon1 = lon_lims[0]
    lon2 = lon_lims[1]

    cmap = copy.copy(plt.cm.RdBu_r)
    m = basemap.Basemap(projection='mill',
                        llcrnrlat=lat1, urcrnrlat=lat2,
                        llcrnrlon=lon1, urcrnrlon=lon2,
                        lat_ts=20, resolution='c')
    lons1, lats1 = np.meshgrid(lon, lat)
    x, y = m(lons1, lats1)
    m.drawcoastlines()
    ds_new = maskoceans(lons1, lats1, t_et)
    print(levs)
    print(np.linspace(*levs))
    levels = np.linspace(*levs)
    cs = m.contourf(x, y, ds_new, levels=levels, cmap=cmap, extend='both')
    cb = plt.colorbar(cs, orientation='vertical', pad=0.05)
    m.contourf(x, y, ds_new, levels=[-1000, -998], colors='darkgrey')
    cb.set_label(units)

    title = ('Relationship between ' + surf_var_name +
             ' and ' + flux_var_name)
    ax.set_title(title)

    plt.suptitle(name, fontsize=14, y=1.03)
    path = str(os.getcwd()) + '/'
    print(path)
    today = datetime.today()
    date = today.strftime("_%d.%m.%Y")
    fname = 'T_ET_metric' + date + '.png'
    plt.savefig(path+fname, dpi=300, bbox_inches='tight')


def monthly_anom_cube(cube, fill=None):
    
    # Extract data array and identify nans
    ds = np.array(cube.data)
    if fill is not None:
        ds[np.where(ds == fill)] = np.nan
        
    # Find where original dataset is masked
    mask = np.where(ds >= 1e20)

    # Group data by month and calculate anomaly from seaonal climatology
    if len(ds.shape) == 3:
        
        # Check if analysis on seasonal cube
        try:
            nmonth = len(cube.coord('season').points[0])
        except:
            nmonth = 12
        ds = ds.reshape(-1, nmonth, cube.shape[-2], cube.shape[-1])
    anomalies = np.nan * np.zeros((ds.shape))
    for mn in range(nmonth):
        anomalies[:, mn, :, :] = ds[:, mn, :, :] - \
                                 np.nanmean(ds[:, mn, :, :], axis=0)

    cube2 = cube.copy()
    cube2.data = anomalies.reshape((-1, cube.shape[-2], cube.shape[-1]))
    
    cube2.data[mask] = np.nan
    
    # Remove null values
    cube2.data[cube2.data >= 1e20] = np.nan
    cube2.data[cube2.data <= -1e20] = np.nan
    
    return(cube2)


def minus180_to_plus180(var_cube):
    """
    Function to reorder cube data from -180 to +180.
    """
    # Reorganise data
    var = var_cube.data
    lat = var_cube.coord('latitude').points
    if len(lat.shape) > 1:
        lat = lat[:, 0]
    lon = var_cube.coord('longitude').points
    if len(lon.shape) > 1:
        lon = lon[0, :]
    half = int(var.shape[-1]/2)
    temp1 = var[:, :, 0:half]
    temp2 = var[:, :, half:]
    new_var = np.concatenate((temp2, temp1), axis=2)
    new_lon = np.arange(-180, 180, (abs(lon[1]-lon[0])))

    # Save re-ordered data as new cube
    new_cube = var_cube.copy()
    new_cube.data = new_var
    new_cube.coord('longitude').points = new_lon

    return(new_cube)


def flip_lats(data_cube):

        lats = data_cube.coord('latitude').points
        # Check if lats need flipping
        if lats[0] < lats[-1]:
            print('Lats already ascending')
            return(data_cube)
        else:
            new_cube = data_cube.copy()
            new_lats = lats[::-1]
            new_data = data_cube.data[:, ::-1, :]
            new_cube.data = new_data
            new_cube.coord('latitude').points = new_lats
            print('Lats flipped')
            return(new_cube)
