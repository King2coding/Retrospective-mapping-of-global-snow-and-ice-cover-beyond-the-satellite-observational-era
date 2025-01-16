'''
Author: Dr K. K. Kumah

Date = Friday, 2024-11-22
This code is what is used now
***************************************************************************************

We derived static predictos for ML-EC approach
These static predictos are:
    Predominant surface type (0 = Water, 1 = Snow free land, 2 = Snow covered land, 3 = Ice) in a grid box
    The probability (value between 0-1) of a grid box been either of these surface type
This information was derived using data spanning 1992 to 2022
'''

# %%
# import packages
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util_functions import *
# import map_manipulation_functions as my_map_functions
# import evaluation_fucntions_algorithms as my_eval_func

import scipy.stats as stats

import xarray as xr
import rasterio
#%%
# define paths
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'
path_to_save_df = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/save_dfs_Oct'
path_to_put_static_predictors = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/static_predictors_for_ML-EC_based_on_1992_2022_data'
# r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/all_data_1992_2022_climatology_based_estimate'

#%%
# decalre global variables
autosnow_nme_part1 = 'gmasi_snowice_reproc_v003'
autosnow_nme_part2 = '0.1deg_wgs.tif'

data_var_summary = 'predominant surface type during the period 1992 to 2022 estimated based on Autosnow data'
wtr_prob_summary = 'probability of a grid box being water during the period 1992 to 2022 estimated based on Autosnow data'
snfr_prob_summary = 'probability of a grid box being snow free during the period 1992 to 2022 estimated based on Autosnow data '
snc_prob_summary = 'probability of a grid box being snow cover during the period 1992 to 2022 estimated based on Autosnow data '
ice_prob_summary = 'probability of a grid box being ice cover during the period 1992 to 2022 estimated based on Autosnow data '

# 0 = Water
# 1 = Snow free land
# 2 = Snow covered land
# 3 = Ice

#%%
print('************ begin estimation ********************')

count = 0

for d in range(1,367):    

    lst_autosnow_ars = []

    if len(str(d)) == 1:
        dofy = '00' + str(d)

    elif len(str(d)) == 2:

        dofy = '0' + str(d)

    else:
        dofy = str(d)         

    for y in range(1992,2023):

        yr = str(y)

        est_yr_doy = ''.join([yr,dofy])

        autosnow_file = '_'.join([autosnow_nme_part1,est_yr_doy,autosnow_nme_part2])

        autosnow_file_to_read = os.path.join(path_to_autosnow_data,autosnow_file)

        if os.path.isfile(autosnow_file_to_read):

            dat_auto_ = xr.open_dataarray(autosnow_file_to_read)        

            dat_auto = dat_auto_.data[0,:,:]

            dat_auto = np.where(dat_auto > 3, np.nan,dat_auto)

            lst_autosnow_ars.append(dat_auto)

    autosnow_3d = make_3d_array(lst_autosnow_ars)

    lons,lats = dat_auto_.x.values, dat_auto_.y.values

    estimated_autosnow_arr = np.apply_along_axis(lambda a: stats.mode(a)[0][0], 2, autosnow_3d)  
    estimated_autosnow_arr = estimated_autosnow_arr.astype(np.int64)  

    # find the number of times a pixel belong to a particular autosnow class
    water_class_count = np.apply_along_axis(lambda a: count_occurrences(a,0),2,autosnow_3d)
    water_class_prob = water_class_count/autosnow_3d.shape[2]

    snow_free_land_class_count = np.apply_along_axis(lambda a: count_occurrences(a,1),2,autosnow_3d)
    snow_free_land_class_prob = snow_free_land_class_count/autosnow_3d.shape[2]

    snow_covered_land_class_count = np.apply_along_axis(lambda a: count_occurrences(a,2),2,autosnow_3d)
    snow_covered_land_class_prob = snow_covered_land_class_count/autosnow_3d.shape[2]

    ice_class_count = np.apply_along_axis(lambda a: count_occurrences(a,3),2,autosnow_3d)
    ice_class_prob = ice_class_count/autosnow_3d.shape[2]
    
    # save the estimated autosnow and std for use with machine learning techn
    yr_doy_sve = '1992_2022_DOY_' + dofy
    
    autsnow_estimates_svenme = '_'.join(['predominant_surface_type_based_on',yr_doy_sve,autosnow_nme_part2.replace('.tif', '.nc')])
    autsnow_estimates_svenme = os.path.join(path_to_put_static_predictors,autsnow_estimates_svenme)
    spit_nc_file(estimated_autosnow_arr, autsnow_estimates_svenme, lons, lats, 'predominant surface type', data_var_summary)    
    
    water_prob_svenme = '_'.join(['probability_of_grid_being_water_based_on',yr_doy_sve,autosnow_nme_part2.replace('.tif', '.nc')])
    water_prob_svenme = os.path.join(path_to_put_static_predictors,water_prob_svenme)
    spit_nc_file(water_class_prob, water_prob_svenme, lons, lats, 'probability of water', wtr_prob_summary) 

    snfr_land_svenme = '_'.join(['probability_of_grid_being_snowfree_based_on',yr_doy_sve,autosnow_nme_part2.replace('.tif', '.nc')])
    snfr_land_svenme = os.path.join(path_to_put_static_predictors,snfr_land_svenme)
    spit_nc_file(snow_free_land_class_prob, snfr_land_svenme, lons, lats, 'probability of snowfree', snfr_prob_summary)     
    
    snow_covered_svenme = '_'.join(['probability_of_grid_being_snowcovered_based_on',yr_doy_sve,autosnow_nme_part2.replace('.tif', '.nc')])
    snow_covered_svenme = os.path.join(path_to_put_static_predictors,snow_covered_svenme)
    spit_nc_file(snow_covered_land_class_prob, snow_covered_svenme, lons, lats, 'probability of snowcvr', snc_prob_summary)     
    
    ice_svenme = '_'.join(['probability_of_grid_being_ice_cvr_based_on',yr_doy_sve,autosnow_nme_part2.replace('.tif', '.nc')])
    ice_svenme = os.path.join(path_to_put_static_predictors,ice_svenme)
    spit_nc_file(ice_class_prob, ice_svenme, lons, lats, 'probability of icecvr', ice_prob_summary) 
    
    count = count + 1

    if count % 50 == 0:
        print(str(count) + ' daily files processed so far') 

print('******************* done with estimation ****************************')
print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')

#%%
# some cheks
# autosnow_based_on_alldata_clim_1992_2022_DOY_365_0.1deg_wgs.tif
# water_class_prob_based_on_all_data_clim_1992_2022_DOY_365_0.1deg_wgs.tif
# snow_free_land_class_prob_based_on_all_data_clim_1992_2022_DOY_365_0.1deg_wgs.tif
# snow_covered_land_class_prob_based_on_all_data_clim_1992_2022_DOY_365_0.1deg_wgs.tif
# ice_class_prob_based_on_all_data_clim_1992_2022_DOY_365_0.1deg_wgs.tif


# check if old method if .tf and new methoid of compressed nc are equal
# autsnow_estimates_svenme
# snow_covered_svenme
# snfr_land_svenme
# water_prob_svenme
# ice_svenme

# check = xr.open_dataset(autsnow_estimates_svenme)

# chec_ = xr.open_dataset(r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/all_data_1992_2022_climatology_based_estimate/autosnow_based_on_alldata_clim_1992_2022_DOY_365_0.1deg_wgs.tif')

# are_numerically_equal = np.allclose(
#     check['predominant surface type'].data,  # First array
#     chec_['band_data'].data,                # Second array
#     equal_nan=True                          # Treat NaNs as equal
# )
# print("Are the arrays numerically equal?", are_numerically_equal)


# np.nansum(check['predominant surface type'].data - chec_['band_data'].data)


# ch_ = xr.open_dataset(r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/all_data_1992_2022_climatology_based_estimate/snow_free_land_class_prob_based_on_all_data_clim_1992_2022_DOY_365_0.1deg_wgs.tif')
# np.nansum(check['probability of water'].data - ch_['band_data'].data)

# def rasterio_based_save_array_to_disk(path,savename,metadata,arrayTosave):
#     '''
# 	functions saves an array to disk in a desired extension
# 	requires:
#     (i) path = the path to save the map
#     (ii) file name for saving the map (add .extension e.g. .tif)
#     (iii) map meta data
#     (iv) array to save 
#     '''
#     with rasterio.open(os.path.join(path,savename),'w',**metadata) as mp:
#         mp.write(arrayTosave,indexes=1)


# get the shape of the autosnow array
# file_sample = os.path.join(path_to_autosnow_data,'gmasi_snowice_reproc_v003_1987184_0.1deg_wgs.tif')
# with xr.open_dataarray(file_sample)as dt:

#     dat = dt.data[0]

#     y_shp,x_shp = dat.shape
# #--------------------------------------------------
# autosnow_meta = rasterio.open(file_sample).meta

# autosnow_meta_cpy = autosnow_meta.copy()
# autosnow_meta_cpy.update({'dtype': np.float32})

# rasterio_based_save_array_to_disk(path_to_put_static_predictors,
#                                   'wt_365_check.tif',
#                                   autosnow_meta_cpy,
#                                   water_class_prob)


# new1_hek = xr.open_dataarray(os.path.join(path_to_put_static_predictors,'wt_365_check.tif'))

# np.nansum(check['probability of water'].data - new1_hek.data)

# fg,ax=plt.subplots(dpi=500)
# ax.imshow(check['probability of water'].data,cmap='jet')
# ax.set_title('check')

# fg,ax=plt.subplots(dpi=500)
# ax.imshow(chec_.band_data[0],cmap='jet')
# ax.set_title('chec_')

# fg,ax=plt.subplots(dpi=500)
# ax.imshow(ch_.band_data[0],cmap='jet')
# ax.set_title('ch_')


# np.nansum(check['probability of snowfree'].data - new1_hek.data)

# np.nansum(new1_hek.data - chec_.band_data)

# np.nansum(new1_hek.data - ch_.band_data)

