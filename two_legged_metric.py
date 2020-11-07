#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
;;#############################################################################
;;
;; two_legged_metric.py
;; Author: Jess Baker (j.c.baker@leeds.ac.uk)
;; LAPSE project, part of CSSP Brazil
;; University of Leeds, UK
;;
;;#############################################################################
;;
;; Description
;;    This script is the diagnostic and plotting script for the global
;;    two-legged metric. The two-legged coupling metric is used to trace energy
;;    or moisture feedback pathways from the surface to the atmosphere in a
;;    mechanistic way (Dirmeyer, 2011, Dirmeyer et al., 2014, Guo et al.,
;;    2006). The metric is based on having a physical understanding of the
;;    factors that control interactions between the land and the atmosphere,
;;    and can be used to identify areas where land-atmosphere coupling is
;;    particularly strong. The feedback pathway is broken down into two stages:
;;    the surface leg, which measures the strength of regression between a
;;    surface state variable (S) and a surface flux variable (F), and the
;;    atmospheric leg, which measures the regression relationship between the
;;    surface flux variable (F) and an atmospheric variable (A). Regression
;;    relationships are calculated using monthly anomalies from the local (grid
;;    cell level) climatological seasonal cycle for each variable (S', F' and
;;    A'). The product of these represents the total land-atmosphere feedback
;;    pathway:
;;                            dF'/dS'×dA'/dF'=dA'/dS'
;;
;;    Finally, dF'/dS', dA'/dF', and dA'/dS' are each multiplied by the
;;    standard deviation of the term in the denominator (σ(S'), σ(F') and
;;    σ(S')). This is to account for the fact that, in some areas,
;;    relationships may be strong while interannual variability is low, and
;;    thus the actual response of the atmosphere to the surface is minimal. 
;;    The metric includes an option to perform the analysis over the wet and
;;    dry seasons separately, or to use data from all months. Further options
;;    allow the user to choose the output variable (regression slope or
;;    correlation coefficient), and to automatically generate plots of the
;;    results.
;;
;; Requirements
;;    Takes three variables relating to the land surface, land-atmosphere flux
;;    and the atmosphere. Data should be formatted as Iris cubes, constrained
;;    to the same time period.
;;
;; References
;;    Dirmeyer, P. A. 2011. The terrestrial segment of soil moisture–climate
;;    coupling. Geophysical Research Letters, 38.
;;
;;    Dirmeyer, P. A. et al. 2014. Intensified land surface control on boundary
;;    layer growth in a changing climate. Geophysical Research Letters, 41,
;;    1290-1294.
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
from datetime import datetime
from mpl_toolkits import basemap
from mpl_toolkits.basemap import maskoceans
from scipy.stats import linregress as ols
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def main(surf_cube, flux_cube, atm_cube, anom=True, ols_out='slope',
         surf_scale=None, flux_scale=None, atm_scale=None,
         wet_dry=False, weighting=True,
         pre_data_path=('/nfs/a68/gyjcab/datasets/lapse_data_harmonised/'
                        'Jan_2018/Final/1.0deg/'
                        'pr_trmm_3b43_mon_1.0deg_1998_2016.nc'),
         constraint_yrs=None,
         plotting=False,
         plotting_args={'name': 'Observations',
                        'lat_lims': [-60, 30],
                        'lon_lims': [-120, 180],
                        'levels': [(-10, 10, 11),
                                   (-15, 15, 11),
                                   (-10, 10, 11)]}, p_thresh=0.05,
         corr_method='pearson'):

    """

    Program for calculating two legged metric with surface, flux and
    atmospheric variables.

    Takes Iris cubes as input.

    Arguments:
        surf_cube = Iris cube of surface state variable.
        flux_cube = Iris cube of flux variable.
        atm_cube = Iris cube of atmospheric state variable.
        anom = Boolean. Calculate metrics using anomalies from climatological
               seasonal cycle (True) or interannual monthly data (False).
        ols_out = output from linear regression. Accepts 'slope' or 'r'.
        surf_scale = scale factor for surface variable.
        flux_scale = scale factor for flux variable.
        atm_scale = scale factor for atmospheric variable.
        wet_dry = Boolean. Calculate metric using data from  6 wettest and 6
                  dryest months in each pixel (True) or using data from all
                  months (False).
        weighting = Boolean. Weight output arrays by standard deviation of
                    denominator (requirement of two-legged metric). Option to
                    remove weighting may be preferred when calculating
                    correlation coefficents.
        pre_data_path = If wet_dry is True, path for precipitation data used to
                        identify wet and dry months.
        constraint_yrs = Length 2 array with start and end years of constraint.
        plotting = Boolean. Plot output of metric. If False returns output of
                   metric as arrays only.
        plotting_args = dictionary of plotting arguments, including name of
                        data being plotted (observations or name of model),
                        figure size, limits for output map, and colorbar
                        levels.
        p_thresh = p threshold for calculating significance of correlations.
        corr_method = correlation method. Can be 'pearson' (assumes data are
                      normally distributed) or 'spearman' (no assumption 
                      about the distribution).

    """
    # Apply scaling factors
    if surf_scale is not None:
        surf_cube.data = surf_cube.data * surf_scale
    if flux_scale is not None:
        flux_cube.data = flux_cube.data * flux_scale
    if atm_scale is not None:
        atm_cube.data = atm_cube.data * atm_scale
    
    # Check if lats are ascending, if not then reverse
    surf_cube = flip_lats(surf_cube)
    flux_cube = flip_lats(flux_cube)
    atm_cube = flip_lats(atm_cube)
    
    # Reorder data from -180 to +180 degrees
    temp_lon = surf_cube.coord('longitude').points
    if temp_lon.max() > 180:
        surf_cube = minus180_to_plus180(surf_cube)
        flux_cube = minus180_to_plus180(flux_cube)
        atm_cube = minus180_to_plus180(atm_cube)

    # Calculate anomalies versus climatological seasonal cycle
    if anom is True:
        surf_cube = monthly_anom_cube(surf_cube)
        flux_cube = monthly_anom_cube(flux_cube)
        atm_cube = monthly_anom_cube(atm_cube)

    # Extract data from input cubes
    surf_var = surf_cube.data
    lat = surf_cube.coord('latitude').points
    lon = surf_cube.coord('longitude').points
    flux_var = flux_cube.data
    atm_var = atm_cube.data

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
        target_cube = surf_cube
        scheme = iris.analysis.AreaWeighted(mdtol=0.5)
        pre_cube = pre_cube.regrid(target_cube, scheme)

        # Calculate seasonal cycle for each pixel
        iris.coord_categorisation.add_month(pre_cube, 'time', name='month')
        pre_mn = pre_cube.aggregated_by(['month'], iris.analysis.MEAN)

        # For all pixels get indices of wet months
        nyear = int(surf_var.shape[0]/12)
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
        wet_arrays = {'surf_leg': None, 'atm_leg': None, 'product': None}
        dry_arrays = {'surf_leg': None, 'atm_leg': None, 'product': None}
        data_dict = {'wet': wet_arrays, 'dry': dry_arrays}

        # Calculate metric for wet and dry seasons
        for season in ['wet', 'dry']:
            print(season)
            print(np.nanmin(surf_var), np.nanmax(surf_var))
            two_legged_output = calculating_legs(surf_var,
                                                 flux_var,
                                                 atm_var,
                                                 ols_out=ols_out,
                                                 wet_bool=wet_bool,
                                                 season=season,
                                                 weighting=weighting,
                                                 p_thresh=p_thresh,
                                                 corr_method=corr_method)
            surf_leg = two_legged_output[0]
            surf_pvals = two_legged_output[1]
            atm_leg = two_legged_output[2]
            atm_pvals = two_legged_output[3]
            product = two_legged_output[4]
            product_pvals = two_legged_output[5]

            data_dict[season]['surf_leg'] = surf_leg
            data_dict[season]['surf_pvals'] = surf_pvals
            data_dict[season]['atm_leg'] = atm_leg
            data_dict[season]['atm_pvals'] = atm_pvals
            data_dict[season]['product'] = product
            data_dict[season]['product_pvals'] = product_pvals

            # Call plotting routine
            if plotting is True:

                # Define plotting variables
                name = plotting_args['name'] + ': ' + season + ' season'
                surf_name = surf_cube.long_name
                flux_name = flux_cube.long_name
                atm_name = atm_cube.long_name

                if ols_out == 'slope':
                    if surf_scale is None:
                        surf_scale = ''
                    else:
                        surf_scale = str(' ({:.0e}'.format(surf_scale)) + ' '

                    if flux_scale is None:
                        flux_scale = ''
                    else:
                        flux_scale = str(' ({:.0e}'.format(flux_scale)) + ' '

                    if atm_scale is None:
                        atm_scale = ''
                    else:
                        atm_scale = str(' ({:.0e}'.format(atm_scale)) + ' '

                    surf_leg_unit = (flux_scale + str(flux_cube.units) + '/' +
                                     surf_scale + str(surf_cube.units))
                    atm_leg_unit = (atm_scale + str(atm_cube.units) + '/' +
                                    flux_scale + str(flux_cube.units))
                    product_unit = (atm_scale + str(atm_cube.units) + '/' +
                                    surf_scale + str(surf_cube.units))

                elif ols_out == 'r':
                    surf_leg_unit = ' '
                    atm_leg_unit = ' '
                    product_unit = ' '
                lat_lims = plotting_args['lat_lims']
                lon_lims = plotting_args['lon_lims']
                levels = plotting_args['levels']

                plot_two_legged(name,
                                surf_name, surf_leg, surf_leg_unit,
                                flux_name, atm_leg, atm_leg_unit,
                                atm_name, product, product_unit,
                                lat, lon, lat_lims, lon_lims,
                                levs=levels)

        return(data_dict, wet_bool, lat, lon)

    # Calculate metric using data from all months
    else:
        two_legged_output = calculating_legs(surf_var,
                                             flux_var,
                                             atm_var,
                                             ols_out=ols_out,
                                             weighting=weighting,
                                             p_thresh=p_thresh,
                                             corr_method=corr_method)
        surf_leg = two_legged_output[0]
        surf_pvals = two_legged_output[1]
        atm_leg = two_legged_output[2]
        atm_pvals = two_legged_output[3]
        product = two_legged_output[4]
        product_pvals = two_legged_output[5]
            
        # Call plotting routine
        if plotting is True:

            # Define plotting variables
            name = plotting_args['name']
            surf_name = surf_cube.long_name
            if surf_name is None:
                surf_name = surf_cube.standard_name
            print(surf_name)
            flux_name = flux_cube.long_name
            if flux_name is None:
                flux_name = flux_cube.standard_name
            print(flux_name)
            atm_name = atm_cube.long_name
            if atm_name is None:
                atm_name = atm_cube.standard_name
            print(atm_name)

            if ols_out == 'slope':
                    if surf_scale is None:
                        surf_scale = ''
                    else:
                        surf_scale = str(' ({:.0e}'.format(surf_scale)) + ' '

                    if flux_scale is None:
                        flux_scale = ''
                    else:
                        flux_scale = str(' ({:.0e}'.format(flux_scale)) + ' '

                    if atm_scale is None:
                        atm_scale = ''
                    else:
                        atm_scale = str(' ({:.0e}'.format(atm_scale)) + ' '

                    surf_leg_unit = (flux_scale + str(flux_cube.units) + '/' +
                                     surf_scale + str(surf_cube.units))
                    atm_leg_unit = (atm_scale + str(atm_cube.units) + '/' +
                                    flux_scale + str(flux_cube.units))
                    product_unit = (atm_scale + str(atm_cube.units) + '/' +
                                    surf_scale + str(surf_cube.units))

            elif ols_out == 'r':
                surf_leg_unit = ' '
                atm_leg_unit = ' '
                product_unit = ' '
            lat_lims = plotting_args['lat_lims']
            lon_lims = plotting_args['lon_lims']
            levels = plotting_args['levels']

            plot_two_legged(name,
                            surf_name, surf_leg, surf_leg_unit,
                            flux_name, atm_leg, atm_leg_unit,
                            atm_name, product, product_unit,
                            lat, lon, lat_lims, lon_lims,
                            levs=levels)

    return(surf_leg, surf_pvals, atm_leg, atm_pvals,
           product, product_pvals, lat, lon)


def calculating_legs(surf_var, flux_var, atm_var, ols_out='slope',
                     wet_bool=None, season=None, weighting=True,
                     p_thresh=0.05, corr_method='pearson'):
    
    len_lat = surf_var.shape[-2]
    len_lon = surf_var.shape[-1]

    # Define arrays to store data
    surf_leg = np.empty((len_lat, len_lon))
    surf_leg[:] = np.nan
    surf_pvals = np.empty((len_lat, len_lon))
    surf_pvals[:] = np.nan
    atm_leg = np.empty((len_lat, len_lon))
    atm_leg[:] = np.nan
    atm_pvals = np.empty((len_lat, len_lon))
    atm_pvals[:] = np.nan
    product = np.empty((len_lat, len_lon))
    product[:] = np.nan
    product_pvals = np.empty((len_lat, len_lon))
    product_pvals[:] = np.nan

    for ny in range(surf_var.shape[-2]):
        for nx in range(surf_var.shape[-1]):

            # Extract data from one grid cell
            if wet_bool is not None:
                    if season == 'wet':
                        i, = np.where(wet_bool[:, ny, nx] == 1)
                        surf_temp = surf_var[i, ny, nx]
                        flux_temp = flux_var[i, ny, nx]
                        atm_temp = atm_var[i, ny, nx]
                    elif season == 'dry':
                        i, = np.where((wet_bool[:, ny, nx]) == 0)
                        surf_temp = surf_var[i, ny, nx]
                        flux_temp = flux_var[i, ny, nx]
                        atm_temp = atm_var[i, ny, nx]
            else:
                surf_temp = surf_var[:, ny, nx]
                flux_temp = flux_var[:, ny, nx]
                atm_temp = atm_var[:, ny, nx]

            # First calculate surface leg of metric
            # 1. Find which months both surface and flux variables have data
            mask1 = ~np.isnan(surf_temp) & ~np.isnan(flux_temp)

            # Provided at least 10 months overlap proceed with calc
            if len(surf_temp[mask1]) > 10:

                # If significant then save value otherwise -999
                if ols_out == 'slope':
                    slope, intercept, r, p, std_err = ols(surf_temp[mask1],
                                                          flux_temp[mask1])
                    surf_leg[ny, nx] = slope
                    surf_pvals[ny, nx] = p
                    
                elif ols_out == 'r':
                    
                    if corr_method == 'pearson':
                        r, p = pearsonr(surf_temp[mask1], flux_temp[mask1])
                        
                    if corr_method == 'spearman':
                        r, p = spearmanr(surf_temp[mask1], flux_temp[mask1])
                    
                    surf_leg[ny, nx] = r    
                    surf_pvals[ny, nx] = p

            # As above for atmospheric leg
            # Find which months both flux and atm variables have data
            mask2 = ~np.isnan(flux_temp) & ~np.isnan(atm_temp)
            if len(flux_temp[mask2]) > 10:
                
                if ols_out == 'slope':
                    slope, intercept, r, p, std_err = ols(flux_temp[mask2],
                                                          atm_temp[mask2])
                    atm_leg[ny, nx] = slope
                    atm_pvals[ny, nx] = p
                
                elif ols_out == 'r':
                    if corr_method == 'pearson':
                        r, p = pearsonr(flux_temp[mask2], atm_temp[mask2])
                        
                    if corr_method == 'spearman':
                        r, p = spearmanr(flux_temp[mask2], atm_temp[mask2])
                    
                    atm_leg[ny, nx] = r
                    atm_pvals[ny, nx] = p

            # Calculate response of atm var to surface var
            product[ny, nx] = surf_leg[ny, nx] * atm_leg[ny, nx]
            product_pvals[ny, nx] = max([surf_pvals[ny, nx],
                                         atm_pvals[ny, nx]])

            # Weight by variability of denominator (see Dirmeyer et al., 2014)
            # this emphasises places where actual impact is large
            if weighting is True:
#                if (surf_pvals[ny, nx] < p_thresh):
                surf_leg[ny, nx] = surf_leg[ny, nx] *\
                                   np.std(surf_temp[mask1])
#                if (atm_pvals[ny, nx] < p_thresh):
                atm_leg[ny, nx] = atm_leg[ny, nx] *\
                                  np.std(flux_temp[mask2])
#                if (product_pvals[ny, nx] < p_thresh):
                product[ny, nx] = product[ny, nx] *\
                                  np.std(surf_temp[mask1])
                                  
    print(np.nanmin(surf_leg), np.nanmax(surf_leg))
    print(np.nanmin(atm_leg), np.nanmax(atm_leg))
    print(np.nanmin(product), np.nanmax(product))
#    assert False
    return(surf_leg, surf_pvals, atm_leg, atm_pvals, product, product_pvals)


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


def plot_two_legged(name,
                    surf_var_name, surf_leg, surf_leg_unit,
                    flux_var_name, atm_leg, atm_leg_unit,
                    atm_var_name, product, product_unit,
                    lat, lon, lat_lims, lon_lims, figsize=(10,8),
                    levs=[(-10, 10, 11),
                          (-15, 15, 11),
                          (-10, 10, 11)]):

    # Make maps of surface and atm legs of metric, plus product
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(wspace=0., hspace=0.2)
    subplots = [311, 312, 313]

    # Corners of subset map
    lat1 = lat_lims[0]
    lat2 = lat_lims[1]
    lon1 = lon_lims[0]
    lon2 = lon_lims[1]
    datasets = [surf_leg, atm_leg, product]

    for i in range(3):
        iplot = subplots[i]
        ax = fig.add_subplot(iplot)
        data = datasets[i]
        levels = np.linspace(*levs[i % 3])
        cmap = copy.copy(plt.cm.RdBu_r)
        m = basemap.Basemap(projection='mill',
                            llcrnrlat=lat1, urcrnrlat=lat2,
                            llcrnrlon=lon1, urcrnrlon=lon2,
                            lat_ts=20, resolution='c')
        lons1, lats1 = np.meshgrid(lon, lat)
        x, y = m(lons1, lats1)
        m.drawcoastlines()
        ds_new = maskoceans(lons1, lats1, data)

        cs = m.contourf(x, y, ds_new, levels=levels, cmap=cmap, extend='both')
        cb = plt.colorbar(cs, orientation='vertical', pad=0.05, shrink=1)
        m.contourf(x, y, ds_new, levels=[-1000, -998], colors='darkgrey')

        if i == 0:
            unit = surf_leg_unit
            pad = 0.05
        if i == 1:
            unit = atm_leg_unit
            pad = 5.2
        if i == 2:
            unit = product_unit
            pad = 0.05

        cb.set_label(unit, labelpad=pad, fontsize=10)

        if i % 3 == 0:
            title = ('Surface leg: relationship between ' + surf_var_name +
                     ' and ' + flux_var_name)
            ax.set_title(title, fontsize=10)
        if i % 3 == 1:
            title = ('Atm leg: relationship between ' + flux_var_name +
                     ' and ' + atm_var_name)
            ax.set_title(title, fontsize=10)
        if i % 3 == 2:
            title = ('Total feedback path: relationship between ' +
                     surf_var_name + ' and ' + atm_var_name)
            ax.set_title(title, fontsize=10)
    plt.suptitle(name, fontsize=14, y=1.05)
    plt.tight_layout()
    path = str(os.getcwd()) + '/'
    print(path)
    today = datetime.today()
    date = today.strftime("_%d.%m.%Y")
    fname = 'two_legged_' + name + '_' + surf_var_name + '-' + \
            flux_var_name + '-' + atm_var_name + date + '.png'
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
