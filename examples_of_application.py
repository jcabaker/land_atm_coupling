"""
;;#############################################################################
;;
;; examples.py
;; Author: Jess Baker (University of Leeds, UK)
;; LAPSE project
;;
;;#############################################################################
;;
;; Description
;;    This script gives examples for applying metrics to obervations or
;;    model data.
;;
;;#############################################################################
"""

import iris
import importlib
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from metric_scripts import terrestrial_coupling_index
from metric_scripts import zengs_gamma
from metric_scripts import T_ET_metric
from metric_scripts import betts_approach
from metric_scripts import two_legged_metric


def get_climatology(cube, agg):
    iris.coord_categorisation.add_year(cube, 'time', name='year')
    cube2 = cube.aggregated_by(['year'], agg)
    cube_out = cube2.collapsed('time', iris.analysis.MEAN)
    return(cube_out)


# %%
# Read in some data within a constrained timeframe
startyr = 2003
endyr = 2014
nyear = endyr-startyr+1
constraint = iris.Constraint(time=lambda cell:
                             startyr <= cell.point.year <= endyr)
    
# Set data_path to location of files
data_path = str(os.getcwd()) + '/'

# precipitation
pre = iris.load_cube(data_path + 'pr_*.nc',
                     constraint=constraint)
pre.convert_units('kg m^-2 month^-1')
pre.data[pre.data == 1e20] = np.nan

# ET
et = iris.load_cube(data_path + 'evspsbl*.nc',
                    constraint=constraint)
et.convert_units('kg m^-2 month^-1')
et.data[et.data == 1e20] = np.nan

# Surface temperature
ts = iris.load_cube(data_path + 'ts*.nc',
                    constraint=constraint)
ts.data[ts.data == 1e20] = np.nan
    
# soil moisture
sm = iris.load_cube(data_path + 'sm*.nc',
                    constraint=constraint)
sm.data[sm.data == 1e20] = np.nan


# %%
###############################################################################
#                 Example for terrestrial coupling index gamma                #
###############################################################################
importlib.reload(terrestrial_coupling_index)
sm_cube = sm.copy()
et_cube = et.copy()
et_cube.data = et_cube.data*60*60*24*30.5  # convert units to mm month-1
tci = terrestrial_coupling_index.main(sm_cube, et_cube, plotting=True)
# %%
###############################################################################
#                         Example for Zeng's gamma                            #
###############################################################################
importlib.reload(zengs_gamma)
pre_cube = pre.copy()
et_cube = et.copy()
gamma = zengs_gamma.main(pre_cube, et_cube, plotting=True,
                         plotting_args={'lat_lims': [-60, 20],
                                        'lon_lims': [-120, 180]})
# %%
###############################################################################
#                         Example for T_ET metric                           #
###############################################################################
importlib.reload(T_ET_metric)
ts_cube = ts.copy()
et_cube = et.copy()
t_et = T_ET_metric.main(ts_cube, et_cube, plotting=True)

# %%
###############################################################################
#                         Example for Betts approach                            #
###############################################################################
importlib.reload(betts_approach)
domain = [-15, 5, -85, -45]  # box over Amazon
var1_cube = sm.copy()
var2_cube = et.copy()

# For this example we will calculate the climatological mean
var1_cube = get_climatology(var1_cube, iris.analysis.MEAN)
var2_cube = get_climatology(var2_cube, iris.analysis.MEAN)
fig, xdata, ydata = betts_approach.main(var1_cube, var2_cube, domain,
                                        contour=True, scatter=True,
                                        name1='SM', name2='ET')
# %%
###############################################################################
#                         Example for two-legged metric                       #
###############################################################################
importlib.reload(two_legged_metric)
surf = sm.copy()
flux = et.copy()
atm = pre.copy()

two_legged = two_legged_metric.main(surf, flux, atm,
                                    wet_dry=False, plotting=True, ols_out='slope',
                                    plotting_args={'name': 'Observations',
                                                   'lat_lims': [-50, 30],
                                                   'lon_lims': [-180, 180],
                                                   'levels': [(-10, 10, 11),
                                                              (-15, 15, 11),
                                                              (-10, 10, 11)]})
