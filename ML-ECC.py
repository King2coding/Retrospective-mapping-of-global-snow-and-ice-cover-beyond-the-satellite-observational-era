#%%
'''
This code applies spatial consistency and ice filtering techniques to ML-EC data aimed at improvimng the results
'''

#%%
# imporrt packages
import warnings
warnings.filterwarnings('ignore')

import os
import gc
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime
from util_functions import *

import xarray as xr

from scipy import stats
from scipy.ndimage import distance_transform_edt, generic_filter

#%%
# define paths
path_to_put_mlecc_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/ML-ECC-approach_based_estimates'
# path_to_mlec_data = r''

path_to_rutgers_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Rutgers_24km_NH_SCE'
path_to_era5_vars = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_daily_data_for_extending_autosnow'

path_to_estimated_autosnw = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_estimated_1988_1991'


#%%
# define global variables
mlec_nme_str = 'RF_estimated_autosnow_using_alldataclim'

all_mlec_data = sorted([os.path.join(path_to_estimated_autosnw,x) for x in os.listdir(path_to_estimated_autosnw) if x.startswith(mlec_nme_str)])


rutgers_data = 'G10035-rutgers-nh-24km-weekly-sce-v01r00-19800826-20220905_01_wgs.nc' # rutgers sce data

ds_rutgers = xr.open_dataset(os.path.join(path_to_rutgers_data,rutgers_data),decode_coords="all")

# crs = ds_rutgers.snow_cover_extent.rio.crs.to_string()

crfs_dest = '+proj=longlat +datum=WGS84 +no_defs +type=crs'

cc = CRS.from_string(crfs_dest)#from_authority(code=4326,auth_name='EPSG')

ds_rutgers.rio.write_crs(cc,inplace=True)

# land sea mask
land_sea_mask = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4'
lsm_arr = xr.open_dataset(land_sea_mask).landseamask.data
# transpose the data to get longitude on the x axis, and flip vertically
# so that latitude is displayed south to north as it should be
lsm_arr = np.flip(lsm_arr.transpose(), axis=0)
lsm = np.where(lsm_arr < 25, 1, 0)

mesh_xy = np.meshgrid(np.arange(-180, 180, 0.1), np.arange(90, -90, -0.1))

lons,lats = mesh_xy[0],mesh_xy[1]

antartica_msk = (lats <= -66.5) & (lsm == 1) # antartica lands mask

# Tropical and equatorial regions: 25°S to 25°N, assumed snow free
tropical_mask = (lats >= -25) & (lats <= 25) & (lsm == 1)

mlecc_summary = 'This data constitutes a corrected version of ML-EC'
#%%
count = 0

for x in all_mlec_data:

    datetime_obj = os.path.basename(x).split('_')[5]

    # create a filename for saving your output
    mlecc_svenme = '_'.join(['ML-ECC',datetime_obj,'0.1deg_wgs.nc'])
    mlecc_svenme = os.path.join(path_to_put_mlecc_data,mlecc_svenme)

    # check if file exists or not, else continue
    if os.path.isfile(mlecc_svenme):
        continue

    ml_ec_estimated = xr.open_dataarray(x)

    lons, lats, = ml_ec_estimated.x.values, ml_ec_estimated.y.values
    ml_ec_estimated_arr = ml_ec_estimated.data[0,:,:]
    ml_ec_estimated_arr = np.where(ml_ec_estimated_arr > 3,np.nan,ml_ec_estimated_arr)   

    yr, doY = datetime_obj[:4], datetime_obj[-3:]

    date_time = pd.to_datetime(f'{yr}-{doY}', format='%Y-%j')   
    
    # Post process the RF-based autosnow using ERA5 and autosnow climatology data
    # this implementation uses Rutgers data, in which the week is on Monday
    # and it includes data from Tuesday through to Monday
    input_date_str = date_time.strftime('%Y-%m-%d')
    relevant_monday = find_relevant_monday(input_date_str) # which is the week in which our current date falls in

    # Rutgers snow cover for the week having our date
    snc_relev_mond = ds_rutgers.snow_cover_extent.sel(time=pd.to_datetime(relevant_monday))
    snc_relev_mond = snc_relev_mond.isel(lat=slice(None,None,-1)).values
    snc_relev_mond[901:,:] = np.nan

    # finding adjacent Mondays in previous years
    input_date_str_m = relevant_monday.strftime('%Y-%m-%d')
    past_years = sorted(list(range(1980, relevant_monday.year)),reverse=True) 
    monday_dates = find_monday_of_same_week_past_years(input_date_str_m, past_years)
    # these represent adjacent weeks in previous years
    monday_dates_ = pd.to_datetime(monday_dates,format='%Y-%m-%dT%H:%M:%S') # pd.DatetimeIndex(
    #[pd.to_datetime(x).date(format='%Y-%m-%d') for x in monday_dates] # .strftime('%Y-%m-%d')

    time_series = pd.Series(ds_rutgers.time.to_index())

    # Initialize an empty list to hold indices of matching dates
    matching_indices = []

    # Iterate over monday_dates to find matches in the dataset's time index
    for dte in monday_dates_:
        matching = time_series[time_series == dte]
        if not matching.empty:
            # Collect indices of matching dates
            matching_indices.extend(matching.index.tolist())

    # Now, use matching_indices to select or filter data in ds_rutgers
    data_var = ds_rutgers.sel(time=ds_rutgers.time[matching_indices]) #if time is a coordinate   

    # flip data along the vertical axis, i.e up down
    data_var = data_var.isel(lat=slice(None, None, -1))

    # get the snow cover data representing adjacent weeks in previous years 
    # 0 = no snow, 1 = snow covered pixel over land only
    snc_var = data_var.snow_cover_extent.values
    snc_var[:,901:,:] = np.nan # the data is only valid in the Northern hemisphere
    # get the most common classification of each pixel based on adjacent weeks in previous years data
    sce_mode_ = np.apply_along_axis(lambda a: stats.mode(a,nan_policy='omit')[0][0], 0, snc_var)
    sce_mode = sce_mode_.copy()
    # if most coomon is 1, relable to 2 (snow covered land in autosnow label)
    sce_mode = np.where((sce_mode == 1) & (lsm == 1), 2 ,sce_mode) # 
    # if most common is 0 over land, relable to 1 (snow free land in autosnow label)
    sce_mode = np.where(((sce_mode == 0) & (lsm == 1)), 1 ,sce_mode)
    sce_mode = sce_mode.astype(float)
    sce_mode[901:, :] = np.nan  # restrict the data to NH

    # Here, we calculate the prbability of a pixel having no snow or snow based on the past years data        
    # Create a boolean mask where the snow or no snow condition is met 
    mask_snw = (snc_var == 1)
    mask_nosnw= (snc_var == 0)
    # Sum the occurrences of value_to_check across the first dimension (num_arrays), ignoring np.nan
    count_snw = np.nansum(mask_snw, axis=0)
    count_snw = count_snw.astype(float)  # Convert to float
    count_snw[901:, :] = np.nan  # restrict the data to NH

    count_nosnw = np.nansum(mask_nosnw, axis=0)
    count_nosnw = count_nosnw.astype(float)  # Convert to float
    count_nosnw[901:, :] = np.nan  # Now you can assign np.nan

    # Calculate the probability for each pixel, considering only the non-nan values
    snw_probab = count_snw / snc_var.shape[0]
    nosnw_probab = count_nosnw / snc_var.shape[0]

    past_years_week_ave_temp_snw_land_lst = []
    past_years_week_ave_temp_nosnw_land_lst = []
    # the tempearture analysis
    for t in data_var.time.values:
        input_t = pd.to_datetime(t).strftime('%Y-%m-%d')

        week_dates = get_aggregated_dates(input_t)

        week_ave_temp_lst = []
        # find and read all surface temperature on these dates  _1983_
        for tt in week_dates:
            year_tt = pd.to_datetime(tt).year

            day_in_week = pd.to_datetime(tt)

            flenme_temp = '_'.join(['skin_temperature',str(year_tt),'daily_mean.nc'])

            surface_temps = xr.open_dataset(os.path.join(path_to_era5_vars,flenme_temp))

            surface_temp = surface_temps.skt.sel(time=day_in_week).values

            week_ave_temp_lst.append(surface_temp)

        week_stck = np.dstack(week_ave_temp_lst)
        week_ave_temp = np.nanmean(week_stck,axis=2)

        week_snw_cv = data_var.snow_cover_extent.sel(time = t).values

        week_snw_cv_ave_temp_land = np.where((week_snw_cv == 1) & (lsm == 1),week_ave_temp,np.nan)

        week_nosnw_cv_ave_temp_land = np.where((week_snw_cv == 0) & (lsm == 1),week_ave_temp,np.nan)

        past_years_week_ave_temp_snw_land_lst.append(week_snw_cv_ave_temp_land)
        past_years_week_ave_temp_nosnw_land_lst.append(week_nosnw_cv_ave_temp_land)

    past_years_week_ave_temp_snw_land = np.nanmean(past_years_week_ave_temp_snw_land_lst)

    past_years_week_ave_temp_nosnw_land = np.nanmean(past_years_week_ave_temp_nosnw_land_lst)

    # find the abs of the days temp with the diff temp conditions
    day_flenme_temp = '_'.join(['skin_temperature',str(date_time.year),'daily_mean.nc'])

    day_surface_temps = xr.open_dataset(os.path.join(path_to_era5_vars,day_flenme_temp))
    day_surface_temp = day_surface_temps.skt.sel(time=pd.to_datetime(date_time)).values

    abs_diff_snw = np.abs(day_surface_temp - past_years_week_ave_temp_snw_land)

    abs_diff_nosnw = np.abs(day_surface_temp - past_years_week_ave_temp_nosnw_land)

    # the post processing
    RF_clim_est = ml_ec_estimated_arr.copy()

    est_snw = ml_ec_estimated_arr.copy()

    est_snw[901:, :] = np.nan  

    est_snw_bol = (est_snw == 2)

    est_snw_bol = est_snw_bol.astype(float)

    est_snw_bol[901:, :] = np.nan    

    snw_pp__ = np.full(est_snw.shape, np.nan)        

    snw_pp__ = np.where(est_snw_bol,4,snw_pp__)
    snw_pp__[901:,:] = np.nan

    # Apply the conditions
    condition1 = (est_snw == 2) & (snw_probab >= 0.2) & (abs_diff_snw <= np.nanmedian(abs_diff_snw))
    condition2 = (est_snw == 2) & (snw_probab < 0.2) & (abs_diff_snw > np.nanmedian(abs_diff_snw))        

    snw_pp__[condition1] = 2
    snw_pp__[condition2] = 1

    snw_pp__ = snw_pp__.astype(float)
    snw_pp__[901:,:]  = np.nan

    RF_clim_est_pp = np.where(snw_pp__ <= 2,snw_pp__,RF_clim_est) 

    # for all the 4 areas representing undeff areas, we fill with previous days observations
    # if previous day doesnt exist we maintian current observation

    current_day = datetime.strptime(input_date_str, '%Y-%m-%d').date()

    prev_day = current_day - timedelta(days=1)

    next_day = current_day + timedelta(days=1)

    # RF estimated using Climatology
    if len(str(day_of_year(prev_day.strftime('%Y-%m-%d')))) == 1:
        prev_doy = str(prev_day.year) + '00' +str(day_of_year(prev_day.strftime('%Y-%m-%d')))

    elif len(str(day_of_year(prev_day.strftime('%Y-%m-%d')))) == 2:
        prev_doy = str(prev_day.year) + '0' +str(day_of_year(prev_day.strftime('%Y-%m-%d')))
    else:
        prev_doy = str(prev_day.year) + str(day_of_year(prev_day.strftime('%Y-%m-%d')))

    #-------------------------------------------------

    if len(str(day_of_year(next_day.strftime('%Y-%m-%d')))) == 1:
        nxt_doy = str(next_day.year) + '00' +str(day_of_year(next_day.strftime('%Y-%m-%d')))

    elif len(str(day_of_year(next_day.strftime('%Y-%m-%d')))) == 2:
        nxt_doy = str(next_day.year) + '0' +str(day_of_year(next_day.strftime('%Y-%m-%d')))
    else:
        nxt_doy = str(next_day.year) + str(day_of_year(next_day.strftime('%Y-%m-%d')))

    #-------------------------------------------------

    prev_day_climrfmp_read_nme = os.path.join(path_to_estimated_autosnw,
                                            '_'.join(['RF_estimated_autosnow_using_alldataclim',
                                                prev_doy,'0.1deg_wgs']) + '.tif')

    nxt_day_climrfmp_read_nme = os.path.join(path_to_estimated_autosnw,
                                            '_'.join(['RF_estimated_autosnow_using_alldataclim',
                                                nxt_doy,'0.1deg_wgs']) + '.tif')

    #-------------------------------------------------

    if os.path.isfile(prev_day_climrfmp_read_nme):

        prev_climrf_autsnw_estimated = xr.open_dataarray(prev_day_climrfmp_read_nme)
        prev_climrf_autsnw_estimated_arr = prev_climrf_autsnw_estimated.data[0,:,:]
        prev_climrf_autsnw_estimated_arr = np.where(prev_climrf_autsnw_estimated_arr > 3,np.nan,
                                                    prev_climrf_autsnw_estimated_arr)
        
        snow_corrected_rf_clim = np.where(snw_pp__ > 2, prev_climrf_autsnw_estimated_arr, RF_clim_est_pp)
    else:
        nxt_climrf_autsnw_estimated = xr.open_dataarray(nxt_day_climrfmp_read_nme)
        nxt_climrf_autsnw_estimated_arr = nxt_climrf_autsnw_estimated.data[0,:,:]
        nxt_climrf_autsnw_estimated_arr = np.where(nxt_climrf_autsnw_estimated_arr > 3,np.nan,
                                                    nxt_climrf_autsnw_estimated_arr)
        
        snow_corrected_rf_clim = np.where(snw_pp__ > 2, nxt_climrf_autsnw_estimated_arr, RF_clim_est_pp)        

    #---------------------------------------------

    #  # Step 1: Apply distance transform to identify proximity to coastline
    # # Invert land_sea_mask for distance calculation (distance_transform_edt considers non-zero as features)
    # distances = distance_transform_edt(np.logical_not(lsm))

    # # Define classification based on distance to coast (specific thresholds would be derived from the article)
    # # For example purposes, distances <= 2 are considered close to the coast, >2 and <=5 are intermediate, and >5 are far
    # # These thresholds should be adjusted according to the article's specifications or empirical analysis
    # coastal_proximity = np.digitize(distances, bins=[2, 5])

    # # Step 2: Correct surface data based on proximity and land-sea mask
    # # Pixels close to the coast or on land may need special handling to correct for land spillover or misclassification
    corrected_surface_data = np.copy(snow_corrected_rf_clim)        

    corrected_surface_data = generic_filter(corrected_surface_data, correct_pixel, size=(3, 3), mode='nearest')

    #-------------------------- regional specfic corrections -------------------------------
    corrected_surface_data_ = np.where(antartica_msk,2,corrected_surface_data)

    ml_ecc_estimated_arr = np.where(tropical_mask,1,corrected_surface_data_)   

    # save the finally correcetd map
    spit_nc_file(ml_ecc_estimated_arr, mlecc_svenme, lons, lats, 'ML-ECC', mlecc_summary)     

    count = count + 1

    if count % 50 == 0:
        print(str(count) + ' daily files processed so far') 

print('******************* done with estimation ****************************')
print('*********************************************************************')
print('*********************************************************************')
print('*********************************************************************')
