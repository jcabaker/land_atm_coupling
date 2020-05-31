#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
;;#############################################################################
;;
;; zengs_gamma.py
;; Author: Jess Baker (j.c.baker@leeds.ac.uk)
;; LAPSE project, part of CSSP Brazil
;; University of Leeds, UK
;;
;;#############################################################################
;;
;; Description
;;    This script calculates Zeng's Gamma for all grid cells in input array(s).
;;
;;    Zeng's gamma is the correlation between evaporation and precipitation
;;    scaled by the standard deviation of the evaporation and normalized by the
;;    standard deviation of precipitation to keep the index dimensionless.
;;
;;    Γ = r(P',E')*  σ(E')    
;;                   _____
;;                   σ(P')
;;
;; Requirements
;;    Takes precipitation and evapotranspiration as input variables. Data
;;    should be formatted as Iris cubes, constrained to the same time period.
;;
;; Reference
;;    Zeng et al (2010) Comparison of Land-Precipitation Coupling Strength
;;    Using Observations and Models
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
from scipy.stats import pearsonr
from mpl_toolkits.basemap import maskoceans
from datetime import datetime


def main(pre_cube, et_cube, plotting=False,
         plotting_args={'lat_lims': [-50, 20],
                        'lon_lims': [-100, -30]}):
    """
    
    This function uses precipitation and evapotranspiration data to calculate
    Zeng's gamma.
    
    Takes Iris cubes as input.
    
    Arguments
        pre_cube = Iris cube of precipitation (model or observations).
        et_cube = Iris cube of evapotranspiration (model or observations).
        plotting = Boolean. Plot output of metric. If False returns output of
                   metric as array only.
        plotting_args = limits for output map.
        
    """
    
    # Calculate anomalies versus climatological seasonal cycle
    pre_anom = monthly_anom_cube(pre_cube)
    et_anom = monthly_anom_cube(et_cube)
    
    # Check if lats are ascending, if not then reverse
    pre_anom = flip_lats(pre_anom)
    et_anom = flip_lats(et_anom)
    
    # Reorder data from -180 to +180 degrees
    pre_lon = pre_anom.coord('longitude').points
    if pre_lon.max() > 180:
        pre_anom = minus180_to_plus180(pre_anom)
        
    et_lon = et_anom.coord('longitude').points
    if et_lon.max() > 180:
        et_anom = minus180_to_plus180(et_anom)
        
    # Compute correlation
    rvals = np.nan * np.zeros((pre_anom.shape[-2], pre_anom.shape[-1]))
    pvals = np.nan * np.zeros((pre_anom.shape[-2], pre_anom.shape[-1]))
    
    for nlat in range(pre_anom.shape[-2]):
        for nlon in range(pre_anom.shape[-1]):
            x = pre_anom.data[:, nlat, nlon]
            y = et_anom.data[:, nlat, nlon]
            mask = ~np.isnan(x) & ~np.isnan(y)
            
            try:
                r, p = pearsonr(x[mask], y[mask])
                rvals[nlat, nlon] = r
                pvals[nlat, nlon] = p
                
            except ValueError:
                rvals[nlat, nlon] = np.nan
                pvals[nlat, nlon] = np.nan
                continue
                
            
    # Get standard deviation of evapotranspiration
    sd_et = et_anom.collapsed('time', iris.analysis.STD_DEV).data       
    
    # Get standard deviation of precipitation
    sd_pre = pre_anom.collapsed('time', iris.analysis.STD_DEV).data

    # Calculate gamma
    gamma = rvals *(sd_et/sd_pre)
    
    lat = pre_anom.coord('latitude').points
    lon = pre_anom.coord('longitude').points
    
    if plotting is True:
        lat_lims = plotting_args['lat_lims']
        lon_lims = plotting_args['lon_lims']
        plot_array(gamma, lat, lon, lat_lims, lon_lims)

    return(gamma, pvals, lat, lon)
    

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
    

def plot_array(array, lat, lon, lat_lims, lon_lims):
    # Corners of subset map
    lat1 = lat_lims[0]
    lat2 = lat_lims[1]
    lon1 = lon_lims[0]
    lon2 = lon_lims[1]
    
    cmap = copy.copy(plt.cm.RdBu_r)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    m = basemap.Basemap(projection='mill',
                        llcrnrlat=lat1, urcrnrlat=lat2,
                        llcrnrlon=lon1, urcrnrlon=lon2,
                        lat_ts=20, resolution='c')
    lons1, lats1 = np.meshgrid(lon, lat)
    x, y = m(lons1, lats1)
    m.drawcoastlines()
    ds_new = maskoceans(lons1, lats1, array)
    levels = np.linspace(-0.5, 0.5, 11)
    cs = m.contourf(x, y, ds_new, levels=levels, cmap=cmap, extend='both')
    plt.colorbar(cs, orientation='vertical', pad=0.05)
    
    title = ("Zeng's gamma")
    ax.set_title(title)
        
    path = str(os.getcwd()) + '/'
    print(path)
    today = datetime.today()
    date = today.strftime("_%d.%m.%Y")
    fname = 'zengs_gamma' + date + '.png'
    plt.savefig(path+fname, dpi=300, bbox_inches='tight')


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
