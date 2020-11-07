#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
;;#############################################################################
;;
;; betts_approach.py
;; Author: Jess Baker (j.c.baker@leeds.ac.uk)
;; LAPSE project, part of CSSP Brazil
;; University of Leeds, UK
;;
;;#############################################################################
;;
;; Description
;;    This script calculates and plots the Betts' relationships between
;;    variables relating to the land surface and land-atmosphere exchange.
;;    Users can define the domain as a simple list of coordinates defining a
;;    grid box, or by specifying a path to a shapefile, from which data will
;;    then be extracted.  Although the example in this study uses monthly data,  
;;    the metric can be applied to data with different temporal frequencies.
;;    Monthly anomalies from the climatological seasonal cycle may be
;;    calculated ad hoc if required. The data can either be plotted as a
;;    scatter plot, or as a contour plot with frequency distributions for each
;;    variable. 
;;
;; Requirements
;;    Takes one variable relating to the land surface state/surface flux and
;;    one variable relating to surface flux/atmospheric state. Data should be
;;    formatted as Iris cubes, constrained to the same time period. The
;;    analysis domain can be specified as a grid box or as a path to a
;;    shapefile.
;;
;;#############################################################################
"""

# Required Python packages
import numpy as np
import matplotlib.pyplot as plt
import shapefile
import os
import iris
import matplotlib
from matplotlib.path import Path
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from datetime import datetime
from mpl_toolkits.basemap import maskoceans
import seaborn as sns


def main(var1, var2, domain, scale1=None, scale2=None, contour=False, scatter=False,
         calculate_anomalies=False, show_mask=False, title=None, 
         xlim=None, ylim=None, xticks=None, yticks=None,
         fontsize=12, name1=None, name2=None, lonflip=False,
         outpath=None, annotate=None, annotation_font_size=12,
         corr_method='pearson', n_levels=8, markersize=5):

    """

    This function calculates and plots relationships between variables
    relating to the land surface and land-atmosphere exchange.

    Takes Iris cubes as input and assumes cubes are harmonised to the same
    temporal and spatial resolution.

    Arguments:
        var1 = Iris cube of surface state variable or flux variable.
        var2 = Iris cube of flux variable or atmospheric state variable.
        domain = string to shapefile path e.g. '/home/shapefiles/shapefile.shp'
                 OR
                 list of box coordinates i.e. [latmin, latmax, lonmin, lonmax]
        scale1 = optional scale for variable 1 to adjust units
        scale2 = optional scale for variable 2 to adjust units
        contour = Boolean. If True a seaborn contour plot is drawn.
        scatter = Boolean. If True a scatter plot is drawn.
        calculate_anomalies = Boolean. If True then monthly anomalies are
                              calculated and plotted.
        show_mask = Boolean. If True then extraction domain is plotted.
        title = optional string as title for plot.
        xlim = optional tuple to set x axis limits, e.g. (-50, 50).
        ylim = optional tuple to set y axis limits, e.g. (-50, 50).
        xticks = optional list to set the xticklabels.
        yticks = optional list to set the yticklabels.
        fontsize = set font size for plot.
        name1 = label for x axis.
        name2 = label for y axis.
        lonflip = Boolean. If True then inputs converted to minus 180 to plus
                  180.
        outpath = optional out directory for figure (string).
        annotate = optional string to annotate plot.
        annotation_font_size = set font size for annotation.
        corr_method = correlation method. Can be 'pearson' (assumes data are
                      normally distributed) or 'spearman' (no assumption 
                      about the distribution).
        n_levels = number of contour levels. Default = 8.
        markersize = optional marker size for scatterplots. Default=5.
    """

    # Calculate anomalies if required
    if len(var1.shape) > 2:
        if calculate_anomalies is True:
            try:
                print('Calculating monthly anomalies...')
                try:
                    var1 = monthly_anom_cube(var1, fill=var1.data.fill_value)
                    var2 = monthly_anom_cube(var2, fill=var2.data.fill_value)
                except AttributeError:
                    var1 = monthly_anom_cube(var1)
                    var2 = monthly_anom_cube(var2)
            except:
                print()
                print('Can not calculate anomalies from data provided. '
                      'Requires monthly data.')
    
    # Check if lats are ascending, if not then reverse
    var1 = flip_lats(var1)
    var2 = flip_lats(var2)
    
    # If needed flip longitudes
    if lonflip is True:
        if var1.coord('longitude').points.max() > 180:
            var1 = minus180_to_plus180(var1)
            
        if var2.coord('longitude').points.max() > 180:
            var2 = minus180_to_plus180(var2)
    
    # Mask oceans
    var1 = mask_ocean_points(var1)
    var2 = mask_ocean_points(var2)

    # Find mask according to domain type
    domain_type = type(domain)

    if domain_type == str:
        mask = get_shape_mask(var1.coord('latitude').points,
                              var1.coord('longitude').points,
                              domain)

    elif domain_type == list:
        mask = get_box_mask(var1.coord('latitude').points,
                            var1.coord('longitude').points,
                            domain[0:2], domain[2:4])

    else:
        print('Wrong domain type!')
        print('Domain type is ', domain_type, 'but should be str or list')

    if show_mask is True:
        plt.figure()
        plt.imshow(mask, origin='lower')
        plt.title('Domain mask')

    if len(var1.shape) > 2:
        # Clip data to mask, looping over all time coordinates
        nt = var1.shape[0]
        subset1 = np.zeros((nt))
        subset2 = np.zeros((nt))
        for n in range(nt):
            # Calculate spatial means for each time step
            subset1[n] = np.nanmean(var1.data[n, :, :][mask])
            subset2[n] = np.nanmean(var2.data[n, :, :][mask])

    elif len(var1.shape) == 2:
        # Calculate spatial means for each time step
        subset1 = np.nanmean(var1.data[:, :][mask])
        subset2 = np.nanmean(var2.data[:, :][mask])
    
    print(subset1.shape)
    print(subset2.shape)
    
    # Flatten arrays and find nans
    xdata = subset1.flatten()

    if scale1 is not None:
        xdata = xdata * scale1

    ydata = subset2.flatten()

    if scale2 is not None:
        ydata = ydata * scale2

    # convert from masked to regular array with Nans
    xdata = np.ma.filled(xdata, np.nan)
    ydata = np.ma.filled(ydata, np.nan)

    nan_mask = ~np.isnan(xdata) & ~np.isnan(ydata)
    xdata = xdata[nan_mask]
    ydata = ydata[nan_mask]

    # Plot the data
    s = fontsize  # set font size
    
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : s}
    plt.rc('font', **font)
    
    if scatter is False and contour is False:
        print('Set "scatter" or "contour" keyword to True')
        assert False

    if scatter is True and contour is False:
        fig = plt.figure(figsize=(8, 8))
        sns.set_style("white")
        sns.set_context("talk")
        fig = plt.figure(figsize=(8, 8))

        # Define the axes positions
        left1 = 0.05
        bottom1 = 0.05
        width1 = height1 = 0.7
        width2 = height2 = 0.2
        ax_main = plt.axes([left1, bottom1, width1, height1])
        ax_top = plt.axes([left1, bottom1+height1, width1, height2])
        ax_right = plt.axes([left1+width1, bottom1, width2, height1])
        
        # Set up colour map
        cmap_main = sns.cubehelix_palette(8, start=2.7, rot=0, dark=0.05,
                                          light=.95, as_cmap=True)
        N = 8
        cmap_hist = plt.cm.get_cmap('Blues_r', N)
        my_color_values = []
        for i in range(cmap_hist.N):
            rgb = cmap_hist(i)[:3] # will return rgba, we take only first 3 so we get rgb
            my_color_values.append(matplotlib.colors.rgb2hex(rgb))
        color1 = my_color_values[1]
        
        # Plot data
        # Main plot
        ax_main.scatter(xdata, ydata, alpha=0.5, color='k', s=markersize)
        if xticks is not None:
            ax_main.set_xticks(xticks)
        if yticks is not None:
            ax_main.set_yticks(yticks)
        plt.xticks(fontsize=s)
        plt.yticks(fontsize=s)
        for tick in ax_main.xaxis.get_major_ticks():
            tick.label.set_fontsize(s) 
        for tick in ax_main.yaxis.get_major_ticks():
            tick.label.set_fontsize(s) 

        if xlim is not None:
            ax_main.set_xlim(xlim)

        if ylim is not None:
            ax_main.set_ylim(ylim)

        if name1 is None:
            name1 = var1.long_name
            if name1 is None:
                name1 = var1.standard_name

        if scale1 is not None:
            ax_main.set_xlabel(name1 +
                               ' ({:.0e}'.format(scale1) +
                               ' ' + str(var1.units) + ')', fontsize=s)
        else:
            ax_main.set_xlabel(name1 +
                               ' (' + str(var1.units) + ')', fontsize=s)

        if name2 is None:
            name2 = var2.long_name
            if name2 is None:
                name2 = var2.standard_name

        if scale2 is not None:
            ax_main.set_ylabel(name2 +
                               ' ({:.0e}'.format(scale2) +
                               ' ' + str(var2.units) + ')', fontsize=s)
        else:
            ax_main.set_ylabel(name2 +
                               ' (' + str(var2.units) + ')', fontsize=s)

        # Top pdf plot
        sns.kdeplot(xdata, ax=ax_top, shade=True, color=color1, legend=False)
        ax_top.set_xticklabels([])
        ax_top.set_yticklabels([])
        ax_top.axis('off')

        # Right pdf plot
        sns.kdeplot(ydata, ax=ax_right, vertical=True, shade=True,
                    color=color1, legend=False)
        ax_right.set_xticklabels([])
        ax_right.axis('off')

        # Add correlation coefficient
        if corr_method == 'pearson':
            r, p = pearsonr(xdata, ydata)
                        
        if corr_method == 'spearman':
            r, p = spearmanr(xdata, ydata)
                        
        txt = "r = " + str('%.2f' % r) + ',  p = ' +\
              str('%.3f' % p)

        ax_main.annotate(txt, xy=(0.5, 0.95), xycoords='axes fraction',
                         xytext=(0.95, 0.95), fontsize=s,
                         horizontalalignment='right',
                         verticalalignment='top')
        if title is not None:
            ax_main.annotate(title, xy=(0.05, 0.05), xycoords='axes fraction',
                             xytext=(0.05, 0.05), fontsize=s,
                             fontweight='bold')
            
        # If required add annotation
        if annotate is not None:
            ax_main.annotate(annotate, xy=(0.9, 0.05), xycoords='axes fraction',
                        xytext=(0.9, 0.05), fontsize=annotation_font_size,
                        fontweight='bold')
            
        # Save figure
        today = datetime.today()
        date = today.strftime("_%d.%m.%Y")
        if outpath is None:
            path = str(os.getcwd()) + '/'
        else:
            path = outpath
        fname = 'betts_relationship_scatter_plot_' +\
                name1 + '_' + name2 + date + '.png'
        print(path+fname)    
        plt.savefig(path+fname, dpi=150, bbox_inches='tight')

    if contour is True and scatter is False:
        sns.set_style("white")
        sns.set_context("talk")
        fig = plt.figure(figsize=(8, 8))

        # Define the axes positions
        left1 = 0.05
        bottom1 = 0.05
        width1 = height1 = 0.7
        width2 = height2 = 0.15
        ax_main = plt.axes([left1, bottom1, width1, height1])
        
        if xticks is not None:
            ax_main.set_xticks(xticks)
        if yticks is not None:
            ax_main.set_yticks(yticks)
            
        # Have distribution axes outside main axis
        ax_top = plt.axes([left1, bottom1+height1, width1, height2])
        ax_right = plt.axes([left1+width1, bottom1, width2, height1])
        if title is not None:
            ax_main.annotate(title, xy=(0.05, 0.05), xycoords='axes fraction',
                             xytext=(0.05, 0.05), fontsize=s,
                             fontweight='bold')
            #ax_main.grid(color='gray', linestyle='dashed')
            ax_main.set_axisbelow(True)
        
        # OR have distribution axes within main axis
        
#        ax_top = plt.axes([left1, bottom1, width1, height2])
#        ax_right = plt.axes([left1, bottom1, width2, height1])
#        if title is not None:
#            ax_main.set_title(title, fontsize=12)
        
        # Set up colour map
        cmap_main = sns.cubehelix_palette(8, start=2.7, rot=0, dark=0.05,
                                          light=.95, as_cmap=True)
        N = 8
        cmap_hist = plt.cm.get_cmap('Blues_r', N)
        my_color_values = []
        for i in range(cmap_hist.N):
            rgb = cmap_hist(i)[:3] # will return rgba, we take only first 3 so we get rgb
            my_color_values.append(matplotlib.colors.rgb2hex(rgb))
        color1 = my_color_values[1]
        
        # Plot data
        # Main plot
        plot = sns.kdeplot(xdata, ydata, shade=True, ax=ax_main,
                           cmap=cmap_main,
                           shade_lowest=False, n_levels=n_levels)
        plt.xticks(fontsize=s)
        plt.yticks(fontsize=s)
        plot.tick_params(labelsize=s)

        if xlim is not None:
            ax_main.set_xlim(xlim)
            ax_top.set_xlim(xlim)
            
        if ylim is not None:
            ax_main.set_ylim(ylim)
            ax_right.set_ylim(ylim)
            
        if name1 is None:
            name1 = var1.long_name
            if name1 is None:
                name1 = var1.standard_name

        if scale1 is not None:
            ax_main.set_xlabel(name1 +
                               ' ({:.0e}'.format(scale1) +
                               ' ' + str(var1.units) + ')', fontsize=s)
        else:
            ax_main.set_xlabel(name1 +
                               ' (' + str(var1.units) + ')', fontsize=s)

        if name2 is None:
            name2 = var2.long_name
            if name2 is None:
                name2 = var2.standard_name

        if scale2 is not None:
            ax_main.set_ylabel(name2 +
                               ' ({:.0e}'.format(scale2) +
                               ' ' + str(var2.units) + ')', fontsize=s)
        else:
            ax_main.set_ylabel(name2 +
                               ' (' + str(var2.units) + ')', fontsize=s)

        # Top pdf plot
        
        # Without histogram
        sns.kdeplot(xdata, ax=ax_top, shade=True, color=color1, legend=False)
        
        # OR with histogram
#        sns.distplot(xdata, ax=ax_top, norm_hist=True, color=color1)
        
        ax_top.set_xticklabels([])
        ax_top.set_yticklabels([])
        ax_top.axis('off')

        # Right pdf plot
        
        # Without histogram
        sns.kdeplot(ydata, ax=ax_right, vertical=True, shade=True,
                    color=color1, legend=False)
        
        # OR with histogram
#        sns.distplot(ydata, ax=ax_right, color=color1, vertical=True,
#                     norm_hist=True)
        ax_right.set_xticklabels([])
        ax_right.axis('off')
        

        # Add correlation coefficient
        if corr_method == 'pearson':
            r, p = pearsonr(xdata, ydata)
                        
        if corr_method == 'spearman':
            r, p = spearmanr(xdata, ydata)
                        
        txt = "r = " + str('%.2f' % r) + ',  p = ' +\
              str('%.3f' % p)
        
        ax_main.annotate(txt, xy=(0.5, 0.95), xycoords='axes fraction',
                         xytext=(0.95, 0.95), fontsize=s,
                         horizontalalignment='right',
                         verticalalignment='top')
        
         # If required add annotation
        if annotate is not None:
            ax_main.annotate(annotate, xy=(0.95, 0.05), xycoords='axes fraction',
                        xytext=(0.9, 0.05), fontsize=annotation_font_size,
                        fontweight='bold')
            
        # Save figure
        today = datetime.today()
        date = today.strftime("_%H:%M.%d.%m.%Y")
        if outpath is None:
            fname = 'betts_relationship_contour_plot_' +\
                name1 + '_' + name2 + date + '.png'
            path = str(os.getcwd()) + '/' + fname
        else:
            path = outpath
        
        print(path)            
        plt.savefig(path, dpi=150, bbox_inches='tight')
        
    if contour is True and scatter is True:
        sns.set_style("white")
        sns.set_context("talk")
        fig = plt.figure(figsize=(8, 8))

        # Define the axes positions
        left1 = 0.05
        bottom1 = 0.05
        width1 = height1 = 0.7
        width2 = height2 = 0.2
        ax_main = plt.axes([left1, bottom1, width1, height1])
        ax_top = plt.axes([left1, bottom1+height1, width1, height2])
        ax_right = plt.axes([left1+width1, bottom1, width2, height1])

        # Set up colour map
        cmap_main = sns.cubehelix_palette(8, start=2.7, rot=0, dark=0.05,
                                          light=.95, as_cmap=True)
        N = 8
        cmap_hist = plt.cm.get_cmap('Blues_r', N)
        my_color_values = []
        for i in range(cmap_hist.N):
            rgb = cmap_hist(i)[:3] # will return rgba, we take only first 3 so we get rgb
            my_color_values.append(matplotlib.colors.rgb2hex(rgb))
        color1 = my_color_values[1]

        # Plot data
        # Main plot
        plot = sns.kdeplot(xdata, ydata, shade=True, ax=ax_main, cmap=cmap_main,
                           shade_lowest=False, n_levels=n_levels)
        ax_main.scatter(xdata, ydata, alpha=0.5, color='k', s=markersize)
        if xticks is not None:
            ax_main.set_xticks(xticks)
        if yticks is not None:
            ax_main.set_yticks(yticks)
        plt.xticks(fontsize=s)
        plt.yticks(fontsize=s)
        plot.tick_params(labelsize=s)

        if xlim is not None:
            ax_main.set_xlim(xlim)
            ax_top.set_xlim(xlim)
            
        if ylim is not None:
            ax_main.set_ylim(ylim)
            ax_right.set_ylim(ylim)

        if name1 is None:
            name1 = var1.long_name
            if name1 is None:
                name1 = var1.standard_name

        if scale1 is not None:
            ax_main.set_xlabel(name1 +
                               ' ({:.0e}'.format(scale1) +
                               ' ' + str(var1.units) + ')', fontsize=s)
        else:
            ax_main.set_xlabel(name1 +
                               ' (' + str(var1.units) + ')', fontsize=s)

        if name2 is None:
            name2 = var2.long_name
            if name2 is None:
                name2 = var2.standard_name

        if scale2 is not None:
            ax_main.set_ylabel(name2 +
                               ' ({:.0e}'.format(scale2) +
                               ' ' + str(var2.units) + ')', fontsize=s)
        else:
            ax_main.set_ylabel(name2 +
                               ' (' + str(var2.units) + ')', fontsize=s)

        # Top pdf plot

        # Without histogram
        sns.kdeplot(xdata, ax=ax_top, shade=True, color=color1, legend=False)
        
        # OR with histogram
#        sns.distplot(xdata, ax=ax_top, norm_hist=True, color=color1)
        
        ax_top.set_xticklabels([])
        ax_top.set_yticklabels([])
        ax_top.axis('off')

        # Right pdf plot
        
        # Without histogram
        sns.kdeplot(ydata, ax=ax_right, vertical=True, shade=True,
                    color=color1, legend=False)
        
        # OR with histogram
#        sns.distplot(ydata, ax=ax_right, color=color1, vertical=True,
#                     norm_hist=True)
        ax_right.set_xticklabels([])
        ax_right.axis('off')

        # Add correlation coefficient
        if corr_method == 'pearson':
            r, p = pearsonr(xdata, ydata)
                        
        if corr_method == 'spearman':
            r, p = spearmanr(xdata, ydata)
                        
        txt = "r = " + str('%.2f' % r) + ',  p = ' +\
              str('%.3f' % p)

        ax_main.annotate(txt, xy=(0.5, 0.9), xycoords='axes fraction',
                         xytext=(0.95, 0.95), fontsize=s,
                         horizontalalignment='right',
                         verticalalignment='top')
        if title is not None:
            ax_main.annotate(title, xy=(0.05, 0.05), xycoords='axes fraction',
                             xytext=(0.05, 0.05), fontsize=s,
                             fontweight='bold')
            
            
        # If required add annotation
        if annotate is not None:
            ax_main.annotate(annotate, xy=(0.9, 0.05), xycoords='axes fraction',
                        xytext=(0.9, 0.05), fontsize=annotation_font_size,
                        fontweight='bold')
        # Save figure
        today = datetime.today()
        date = today.strftime("_%d.%m.%Y")
        if outpath is None:
            fname = 'betts_relationship_contour_plot_' +\
                name1 + '_' + name2 + date + '.png'
            path = str(os.getcwd()) + '/' + fname
        else:
            path = outpath
        
        print(path)            
        plt.savefig(path, dpi=150, bbox_inches='tight')
    
    # Reset configuration settings that may have changed after using seaborne
    sns.reset_orig()
    return(fig, xdata, ydata)


def get_shape_mask(data_lat, data_lon, shp):

    # Load a shapefile
    sf = shapefile.Reader(shp)

    # Extract coordinates from shapefile
    for shape_rec in sf.shapeRecords():
        mask_lons = []
        mask_lats = []
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i+1]):
                mask_lons.append(pts[j][0])
                mask_lats.append(pts[j][1])

    # Determine mask and apply to data
    # Vertices of extraction domain
    coordlist = np.vstack((mask_lons, mask_lats)).T

    # Co-ordinates of every grid cell
    dat_x, dat_y = np.meshgrid(data_lon, data_lat)
    coord_map = np.vstack((dat_x.flatten(), dat_y.flatten())).T
    polypath = Path(coordlist)

    # Work out which coords are within the polygon
    mask = polypath.contains_points(coord_map).reshape(dat_x.shape)

    return(mask)


def get_box_mask(data_lat, data_lon, mask_lats, mask_lons):

    # Convert domain vertices to bounding sequence
    lats = [mask_lats[0], mask_lats[0], mask_lats[1],
            mask_lats[1], mask_lats[0]]
    lons = [mask_lons[0], mask_lons[1], mask_lons[1],
            mask_lons[0], mask_lons[0]]

    # Vertices of extraction domain
    coordlist = np.vstack((lons, lats)).T

    # Co-ordinates of every grid cell
    dat_x, dat_y = np.meshgrid(data_lon, data_lat)
    coord_map = np.vstack((dat_x.flatten(), dat_y.flatten())).T
    polypath = Path(coordlist)

    # Work out which coords are within the polygon
    mask = polypath.contains_points(coord_map).reshape(dat_x.shape)
    return(mask)


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
        plt.figure()
        plt.imshow(lat)
        lat = lat[:,0]
    lon = var_cube.coord('longitude').points
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


def get_lats(cube):
    try:
        lats = cube.coords('latitude').points
    except AttributeError:
        lats = cube.coords('latitude')[0][:].points
    return(lats)
    
    
def get_lons(cube):
    try:
        lons = cube.coords('longitude').points
    except AttributeError:
        lons = cube.coords('longitude')[0][:].points
    return(lons)
    

def mask_ocean_points(cube):
    print('Masking ocean points')
    
    # Get lons and lats
    lons = get_lons(cube)
    lats = get_lats(cube)
    lons1, lats1 = np.meshgrid(lons, lats)
    
    # Replace cube data with data that has ocean points masked
    if len(cube.shape) == 3:
        for n in range(cube.shape[0]):
            data = cube[n, :, :].data.copy()
            ds_new = maskoceans(lons1, lats1, data)
            cube.data[n, :, :] = ds_new
            
    elif len(cube.shape) == 2:
        data = cube[:, :].data.copy()
        ds_new = maskoceans(lons1, lats1, data)
        cube.data[:, :] = ds_new
    
    else:
        print('Check cube dimensions - should have two or three dimensions')
        
    return(cube)
