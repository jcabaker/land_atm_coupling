#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
;;#############################################################################
;;
;; terrestrial_coupling_index.py
;; Author: Jess Baker (j.c.baker@leeds.ac.uk)
;; LAPSE project, part of CSSP Brazil
;; University of Leeds, UK
;;
;;#############################################################################
;;
;; Description
;;    This script calculates the terrestrial coupling index for all grid cells
;;    in input array(s).
;;
;;    The terrestrial coupling index determines the coupling of soil moisture
;;    and surface fluxes. It is calculated as the slope of the sol moisture-
;;    surface flux relationship, weighted by the standard deviation in soil
;;    moisture to determine the degree to which soil moisture changes drive
;;    surface flux variability. The metric can be calculated for the wet and
;;    dry seasons separately, and plots can be generated automatically. 
;;
;; Requirements
;;    Takes soil moisture and evapotranspiration as input variables, but could
;;    be applied to other variables. Data should be formatted as Iris cubes,
;;    constrained to the same time period.
;;
;; References
;;    Dirmeyer, P. A. 2011. The terrestrial segment of soil moisture–climate
;;    coupling. Geophysical Research Letters, 38.
;;
;;    Guo, Z. et al., 2006. Glace: The Global Land–Atmosphere Coupling
;;    Experiment. Part II: Analysis. Journal of Hydrometeorology, 7, 611-625.
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


def main(sm_cube, et_cube, ols_out='slope', wet_dry=False, constraint_yrs=None,
         pre_data_path=('/Users/Jess/Google Drive/amip_analysis/observations/'
                        'pr_trmm_3b43_mon_1.0deg*.nc'),
         plotting=False, weighting=True, anom=True,
         plotting_args={'name': 'Terrestrial Coupling Index',
                        'lat_lims': [-60, 30],
                        'lon_lims': [-120, 180],
                        'levels': (-10, 10, 11)}, p_thresh=0.05):

    """

    This function uses soil moisture and evapotranspiration data to calculate
    the terrestrial coupling index.

    Takes Iris cubes as input.

    Arguments
        sm_cube = Iris cube of soil moisture (model or observations).
        et_cube = Iris cube of evapotranspiration (model or observations).
        ols_out = output from linear regression. Accepts 'slope' or 'r'.
        wet_dry = Boolean. Calculate metric using data from  6 wettest and 6
                  dryest months in each pixel (True) or using data from all
                  months (False).
        constraint_yrs = Length 2 array with start and end years of constraint.
        pre_data_path = String. If wet_dry is True, path for precipitation data used to
                        identify wet and dry months (must be NetCDF format).
        plotting = Boolean. Plot output of metric. If False returns output of
                   metric as arrays only.
        weighting = Boolean. Weight output arrays by standard deviation of
                    denominator (requirement of terrestrial coupling index).
                    Option to remove weighting may be preferred when
                    calculating correlation coefficents only.
        anom = Boolean. Calculate metric using anomalies from climatological
               seasonal cycle (True) or interannual monthly data
               (False).
        plotting_args = dictionary of plotting arguments, including name of
                        data being plotted (observations or name of model),
                        limits for output map, and colorbar levels.
        p_thresh = p threshold for calculating significance of correlations.

    """
    
    if anom is True:  # default
        # Calculate anomalies versus climatological seasonal cycle
        print('Calculating anomalies')
        sm_anom = monthly_anom_cube(sm_cube)
        et_anom = monthly_anom_cube(et_cube)
    else:
        sm_anom = sm_cube
        et_anom = et_cube
    
    # Check if lats are ascending, if not then reverse
    sm_anom = flip_lats(sm_anom)
    et_anom = flip_lats(et_anom)
    
    # Reorder data from -180 to +180 degrees
    sm_lon = sm_anom.coord('longitude').points
    if sm_lon.max() > 180:
        print('Reordering longitudes')
        sm_anom = minus180_to_plus180(sm_anom)
        
    et_lon = et_anom.coord('longitude').points
    if et_lon.max() > 180:
        print('Reordering longitudes')
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
        target_cube = sm_anom
        scheme = iris.analysis.AreaWeighted(mdtol=0.5)
        pre_cube = pre_cube.regrid(target_cube, scheme)

        # Calculate seasonal cycle for each pixel
        iris.coord_categorisation.add_month(pre_cube, 'time', name='month')
        pre_mn = pre_cube.aggregated_by(['month'], iris.analysis.MEAN)

        # For all pixels get indices of wet months
        nyear = int(sm_anom.shape[0]/12)
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
        wet_arrays = {'tci': None}
        wet_arrays = {'pval_array': None}
        dry_arrays = {'tci': None}
        dry_arrays = {'pval_array': None}
        data_dict = {'wet': wet_arrays, 'dry': dry_arrays}

        # Calculate metric for wet and dry seasons
        for season in ['wet', 'dry']:
            print(season)
            tci, pval_array = calculating_tci(sm_anom, et_anom,
                                              ols_out=ols_out,
                                              wet_bool=wet_bool,
                                              season=season,
                                              weighting=weighting,
                                              p_thresh=p_thresh)

            data_dict[season]['tci'] = tci
            data_dict[season]['pval_array'] = pval_array
            
            # Call plotting routine
            if plotting is True:

                # Define plotting variables
                name = plotting_args['name'] + ': ' + season + ' season'
                surf_name = sm_anom.long_name
                flux_name = et_anom.long_name

                if ols_out == 'slope':
                    units = str(et_anom.units) + '/' + str(sm_anom.units)

                elif ols_out == 'r':
                    units = ' '

                lat = sm_anom.coord('latitude').points
                lon = sm_anom.coord('longitude').points
                lat_lims = plotting_args['lat_lims']
                lon_lims = plotting_args['lon_lims']
                levels = plotting_args['levels']

                plot_tci(name, surf_name, flux_name, tci, units,
                         lat, lon, lat_lims, lon_lims, levs=levels)

        return(data_dict, wet_bool)

    # Calculate metric using data from all months
    else:
        tci, pval_array = calculating_tci(sm_anom, et_anom, ols_out=ols_out,
                                          weighting=weighting,
                                          p_thresh=p_thresh)

        # Call plotting routine
        if plotting is True:

            # Define plotting variables
            name = plotting_args['name']
            surf_name = sm_anom.long_name
            if surf_name is None:
                surf_name = sm_anom.standard_name
            flux_name = et_anom.long_name
            if flux_name is None:
                flux_name = et_anom.standard_name

            if ols_out == 'slope':
                units = str(et_anom.units) + '/' + str(sm_anom.units)
                if weighting is True:
                    units = str(et_anom.units)

            elif ols_out == 'r':
                units = ' '

            lat = sm_anom.coord('latitude').points
            lon = sm_anom.coord('longitude').points
            lat_lims = plotting_args['lat_lims']
            lon_lims = plotting_args['lon_lims']
            levels = plotting_args['levels']

            plot_tci(name, surf_name, flux_name, tci, units,
                     lat, lon, lat_lims, lon_lims, levs=levels)

        return(tci, pval_array)


def calculating_tci(sm_anom, et_anom, ols_out='slope',
                    wet_bool=None, season=None, weighting=True,
                    p_thresh=0.05):

    # Define arrays to store data
    len_lat = sm_anom.shape[-2]
    len_lon = sm_anom.shape[-1]
    tci = np.nan * np.empty((len_lat, len_lon))
    pval_array = np.nan * np.empty((len_lat, len_lon))

    for ny in range(len_lat):
        for nx in range(len_lon):

            # Extract data from one grid cell
            if wet_bool is not None:
                    if season == 'wet':
                        i, = np.where(wet_bool[:, ny, nx] == 1)
                        surf_temp = sm_anom.data[i, ny, nx]
                        flux_temp = et_anom.data[i, ny, nx]

                    elif season == 'dry':
                        i, = np.where((wet_bool[:, ny, nx]) == 0)
                        surf_temp = sm_anom.data[i, ny, nx]
                        flux_temp = et_anom.data[i, ny, nx]
            else:
                surf_temp = sm_anom.data[:, ny, nx]
                flux_temp = et_anom.data[:, ny, nx]

            # 1. Find which months both surface and flux variables have data
            mask = ~np.isnan(surf_temp) & ~np.isnan(flux_temp)
#            print(mask)
            # Provided at least 10 months overlap proceed with calc
            if len(surf_temp[mask]) > 10:

                slope, intercept, r, p, std_err = ols(surf_temp[mask],
                                                      flux_temp[mask])

                # Save tci and p value
                if ols_out == 'slope':
                    tci[ny, nx] = slope
                elif ols_out == 'r':
                    tci[ny, nx] = r
                    
                pval_array[ny, nx] = p

            # Weight by variability of denominator (see Dirmeyer et al., 2011)
            # this emphasises places where actual impact is large
            if weighting is True:
                if (tci[ny, nx] != -999.0):
                    tci[ny, nx] = tci[ny, nx] * np.std(surf_temp[mask])
    print(np.nanmin(tci), np.nanmax(tci))
    return(tci, pval_array)


def plot_tci(name, surf_var_name, flux_var_name, tci, units,
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
    ds_new = maskoceans(lons1, lats1, tci)
#    ds_new=tci
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
    fname = 'terrestrial_coupling_index' + date + '.png'
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
#    print(lat)
    if len(lat.shape) > 1:
        plt.figure()
        plt.imshow(lat)
        lat = lat[:,0]
    lon = var_cube.coord('longitude').points
#    print(lon)
    if len(lon.shape) > 1:
        plt.figure()
        plt.imshow(lon)
        lon = lon[0,:]
    l = int(var.shape[-1]/2)
    
    if len(var_cube.shape) > 2:
        temp1 = var[:, :, 0:l]
        temp2 = var[:, :, l:]
        new_var = np.concatenate((temp2, temp1), axis=2)
    if len(var_cube.shape) == 2:
        temp1 = var[:, 0:l]
        temp2 = var[:, l:] 
        new_var = np.concatenate((temp2, temp1), axis=1)
    
    a = lon[int(len(lon)/2):]
    b = lon[:int(len(lon)/2)]
    
    new_lon = np.concatenate((a-360, b))
    
    # Save re-ordered data as new cube
    try:
        new_cube = var_cube.copy()
        new_cube.data = new_var
        new_cube.coord('longitude').points = new_lon
    except ValueError:
        print('Making fresh cube!!!!!!!!!!!!!!!!!!!!')
        ### Make fresh cube
        if len(var_cube.shape) > 2:
            ### Establish lat and lon dimensions
            latitude = iris.coords.DimCoord(lat, standard_name='latitude',
                                            units='degrees')
            longitude = iris.coords.DimCoord(new_lon, standard_name='longitude',
                                             units='degrees')
            times = var_cube.coord('time').points
            time_unit = var_cube.coord('time').units
            time = iris.coords.DimCoord(times, standard_name='time', units=time_unit)
            
            # Call cube
            new_cube = iris.cube.Cube(new_var, 
                                      dim_coords_and_dims=
                                      [(time, 0), (latitude, 1), (longitude, 2)])
        
        if len(var_cube.shape) == 2:
            ### Establish lat and lon dimensions
            latitude = iris.coords.DimCoord(lat, standard_name='latitude',
                                            units='degrees')
            longitude = iris.coords.DimCoord(new_lon, standard_name='longitude',
                                             units='degrees')
            
            # Call cube
            new_cube = iris.cube.Cube(new_var, 
                                      dim_coords_and_dims=
                                      [(latitude, 0), (longitude, 1)])
             
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
