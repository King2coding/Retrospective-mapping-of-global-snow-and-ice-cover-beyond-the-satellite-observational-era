'''
Author: Dr K. K. Kumah

Date = Monday, 2024-12-01
This code is what is used now
***************************************************************************************
This code estimates/predict surface type using two method:
    ML-E :- in which we trained random forest using ERA5-based surface variabl;es as predictors/features
    and GMASI-Autosnow as target

    ML-EC :- similar toi ERA5 except we complement the features with Climatological information
    derived from the GMASI-Autosnow

'''
#%%
# import apckages
import warnings
warnings.filterwarnings('ignore')
import datetime
from datetime import datetime, date, timedelta
import os
import pickle
import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt

from util_functions import *

import xarray as xr

#%%
#  define path to data
path_to_autosnow_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/autosnow_in_geotif'

path_to_era5_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/era5_daily_data_for_extending_autosnow'

path_to_models = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/trained_models'

path_to_autosno_climatological_data = r'/ra1/pubdat/AVHRR_CloudSat_proj/ERA5_multi_variables/alldata_climatology_based_autosnow_estimate'

path_to_put_ml_e_estimates = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/ML-E-approach_based_estimates'
path_to_put_ml_ec_estimates = r'/ra1/pubdat/AVHRR_CloudSat_proj/Autosnow_archive_1987_june2023/extending_autosnow_estimated/ML-EC-approach_based_estimates'

#%%
# define global variables

all_autosnow_files = [os.path.join(path_to_autosnow_data,a) for a in os.listdir(path_to_autosnow_data) if (int(a.split('_')[4][:4]) >=1988) and (int(a.split('_')[4][:4]) <=1991)]

lake_cover = r'/ra1/pubdat/AVHRR_CloudSat_proj/global_variables/lake_cover_2007_0.1deg.nc'

land_sea_mask = r'/ra1/pubdat/AVHRR_CloudSat_proj/IMERG/ancillary_imerg_data/GPM_IMERG_LandSeaMask.2.nc4'
#------------------------------------------------
lc_arr = xr.open_dataset(lake_cover).cl.data[0]

lsm_arr = xr.open_dataset(land_sea_mask).landseamask.data
# transpose the data to get longitude on the x axis, and flip vertically
# so that latitude is displayed south to north as it should be
lsm_arr = np.flip(lsm_arr.transpose(), axis=0)
#------------------------------------------------


mesh_xy = np.meshgrid(np.arange(-180, 180, 0.1), np.arange(90, -90, -0.1))
#------------------------------------------------

# read the model
ml_e_rf_mdl_fle = os.path.join(path_to_models,'RF_classifier_model_trained_with_ERA5_20231207.pkl')
with open(ml_e_rf_mdl_fle, 'rb') as rf_mdl_file:  
    ml_e_rf_model = pickle.load(rf_mdl_file)

ml_ec_rf_mdl_fle = os.path.join(path_to_models,'RF_classifier_model_trained_with_ERA5_and_autosnowclim_data_20231201.pkl')
with open(ml_ec_rf_mdl_fle, 'rb') as rf_mdl_file:  
    ml_ec_rf_model = pickle.load(rf_mdl_file)

classes = ['Water','Snow free land', 'Snow covered land','Ice']

comn_nme_prt1 = 'based_on_alldata_clim_1999_2022_DOY'
comn_nme_prt2 = '0.1deg_wgs'
comn_nme_prt3 = 'prob_based_on_all_data_clim_1999_2022_DOY'

ml_e_summary = (
    "This data was estimated using the Random Forest machine learning algorithm "
    "trained with ERA5-based surface variables as predictor/feature variables "
    "and GMASI-Autosnow as the target variable"
)

ml_ec_summary = ("This data was estimated using the Random Forest machine learning algorithm "
    "trained using ERA5-based surface variables and a climatology of GMASI-Autosnow (1992-2022)" 
    "as predictor/feature variables and GMASI-Autosnow as the target variable"
)
cde_run_dte = str(date.today().strftime('%Y%m%d'))

gc.collect()

#%%
print('begin the estimating autosnow using trained RF machine learning algorithm!')
print('****************************')
print('****************************')
print('****************************')

# Function to process a single autosnow file
def process_autosnow_file(l):
    try:
        # Extract year and DOY
        yr_DOY = os.path.basename(l).split('_')[4]
        doy = os.path.basename(l).split('_')[4][-3:]
        yr = int(yr_DOY[:4])
        doY = int(yr_DOY[-3:])
        dt = datetime(yr, 1, 1) + timedelta(doY - 1)

        # Open autosnow file
        dat_auto = xr.open_dataarray(l)
        lon, lat = dat_auto.x.values, dat_auto.y.values
        dat_auto_array = np.where(dat_auto.data[0, :, :] > 3, np.nan, dat_auto.data[0, :, :])
        y_shp, x_shp = dat_auto_array.shape

        # Read ERA5 input files
        dwpnt_2m_arr = read_era5_file(path_to_era5_data, '2m_dewpoint_temperature', str(yr), dt)
        temp_2m_arr = read_era5_file(path_to_era5_data, '2m_temperature', str(yr), dt)
        sea_ice_arr = read_era5_file(path_to_era5_data, 'sea_ice_cover', str(yr), dt)
        sst_arr = read_era5_file(path_to_era5_data, 'sea_surface_temperature', str(yr), dt)
        skt_arr = read_era5_file(path_to_era5_data, 'skin_temperature', str(yr), dt)
        albedo_arr = read_era5_file(path_to_era5_data, 'forecast_albedo', str(yr), dt)

        # Read climatology-based data
        clim_autsnw_data_array = read_climatology_file(path_to_autosno_climatological_data, 'autosnow', comn_nme_prt1, doy, comn_nme_prt2)
        wtr_cls_prob_data_array = read_climatology_file(path_to_autosno_climatological_data, 'water_class', comn_nme_prt3, doy, comn_nme_prt2)
        snw_free_lnd_cls_prob_array = read_climatology_file(path_to_autosno_climatological_data, 'snow_free_land_class', comn_nme_prt3, doy, comn_nme_prt2)
        snw_cvrd_lnd_cls_prob_data_array = read_climatology_file(path_to_autosno_climatological_data, 'snow_covered_land_class', comn_nme_prt3, doy, comn_nme_prt2)
        ice_cls_prob_data_array = read_climatology_file(path_to_autosno_climatological_data, 'ice_class', comn_nme_prt3, doy, comn_nme_prt2)

        # Prepare auxiliary data
        lc_arr[np.isnan(lc_arr)] = -99999
        lsm_arr[np.isnan(lsm_arr)] = -99999
        lat_2d_arr = mesh_xy[1]
        lon_2d_arr = mesh_xy[0]
        doy_arr = np.full_like(dwpnt_2m_arr, doY)

        # Prepare input lists
        ml_e_arr_lst = [lon_2d_arr, lat_2d_arr, doy_arr, dwpnt_2m_arr, temp_2m_arr, sst_arr, skt_arr, sea_ice_arr, albedo_arr, lc_arr, lsm_arr]
        ml_ec_arr_lst = [lon_2d_arr, lat_2d_arr, doy_arr, dwpnt_2m_arr, temp_2m_arr, sst_arr, skt_arr, sea_ice_arr, albedo_arr, clim_autsnw_data_array,
                         wtr_cls_prob_data_array, snw_free_lnd_cls_prob_array, snw_cvrd_lnd_cls_prob_data_array, ice_cls_prob_data_array, lc_arr, lsm_arr]

        # Process ML-E
        ml_e_arr_rshp = prepare_rf_input(ml_e_arr_lst, x_shp, y_shp)
        ml_e_rf_predicted = ml_e_rf_model.predict(ml_e_arr_rshp)
        ml_e_rf_predicted_arr = ml_e_rf_predicted.reshape(y_shp, x_shp)
        ml_e_name = '_'.join(['ML-E', yr_DOY, '0.1deg_wgs']) + '.nc'
        ml_e_file = os.path.join(path_to_put_ml_e_estimates, ml_e_name)
        spit_nc_file(ml_e_rf_predicted_arr, ml_e_file, lon, lat, 'ML-E', ml_e_summary)

        # Process ML-EC
        ml_ec_arr_rshp = prepare_rf_input(ml_ec_arr_lst, x_shp, y_shp)
        ml_ec_rf_predicted = ml_ec_rf_model.predict(ml_ec_arr_rshp)
        ml_ec_rf_predicted_arr = ml_ec_rf_predicted.reshape(y_shp, x_shp)
        ml_ec_name = '_'.join(['ML-EC', yr_DOY, '0.1deg_wgs']) + '.nc'
        ml_ec_file = os.path.join(path_to_put_ml_ec_estimates, ml_ec_name)
        spit_nc_file(ml_ec_rf_predicted_arr, ml_ec_file, lon, lat, 'ML-EC', ml_ec_summary)

        return f"Processed file {l}"

    except Exception as e:
        return f"Error processing file {l}: {e}"


# Main parallel processing function

def parallel_process_files(file_paths, max_workers=15):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    """
    Parallel process a list of file paths using ProcessPoolExecutor.
    
    Parameters:
    - file_paths (list): List of file paths to process.
    - max_workers (int): Number of worker processes to spawn.

    Returns:
    - results (list): List of results from processing each file.
    """
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_autosnow_file, file_path): file_path for file_path in file_paths}
        
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)  # Collect results for aggregation
            except Exception as exc:
                print(f'{file_path} generated an exception: {exc}')
    
    return results

if __name__ == "__main__":
    print("Starting parallel processing...")

    # Use ProcessPoolExecutor for CPU-bound tasks
    files_to_process = sorted(all_autosnow_files)  # Adjust the range for testing
    max_workers = 15#os.cpu_count()  # Use all available CPU cores

    results = parallel_process_files(files_to_process, max_workers=max_workers)

    # Log results
    for result in results:
        print(result)

    print("Parallel processing complete!")
#%%
# # The truth test
# ml_e_file = os.path.join(path_to_put_ml_e_estimates,'ML-E_1988001_0.1deg_wgs.nc')
# ml_e = xr.open_dataarray(ml_e_file)
# ml_e.plot()

# ml_ec_file = os.path.join(path_to_put_ml_ec_estimates,'ML-EC_1988001_0.1deg_wgs.nc')

# ml_ec = xr.open_dataarray(ml_ec_file)
# ml_ec.plot()

#%%
#%%
# The retrieval
# l = '2009006'
# print('begin the estimating autosnow using trained RF machine learning algorithm!')
# print('****************************')
# print('****************************')
# print('****************************')

# count = 0
# for l in sorted(all_autosnow_files)[:5]:
    

#     yr_DOY = os.path.basename(l).split('_')[4]

#     yr  = int(os.path.basename(l).split('_')[4][:4])#int(l[:4])

#     doy_ = os.path.basename(l).split('_')[4][-3:]

#     doY = int(os.path.basename(l).split('_')[4][-3:]) #int(l[-3:])

#     doy = os.path.basename(l).split('_')[4][-3:]

#     dt = datetime(yr, 1, 1) + timedelta(doY - 1)

#     # dt = datetime.datetime(yr, 1, 1) + datetime.timedelta(doY - 1) 

#     dt_sve = dt.strftime('%Y%m%d') 
#     #------------------------------------------------------

#     # open autosnow files
#     dat_auto = xr.open_dataarray(l)   

#     lon, lat = dat_auto.x.values, dat_auto.y.values    

#     dat_auto_array = dat_auto.data[0,:,:]

#     dat_auto_array = np.where(dat_auto_array > 3,np.nan,dat_auto_array)

#     y_shp,x_shp = dat_auto_array.shape[0],dat_auto_array.shape[1]   

#     # Define and read ERA5 input files
#     dwpnt_2m_arr =  read_era5_file(path_to_era5_data, '2m_dewpoint_temperature', str(yr), dt)    

#     temp_2m_arr = read_era5_file(path_to_era5_data, '2m_temperature', str(yr), dt)   
    
#     sea_ice_arr = read_era5_file(path_to_era5_data, 'sea_ice_cover', str(yr), dt) 

#     sst_arr =   read_era5_file(path_to_era5_data, 'sea_surface_temperature', str(yr), dt)

#     skt_arr =   read_era5_file(path_to_era5_data, 'skin_temperature', str(yr), dt)    

#     albedo_arr = read_era5_file(path_to_era5_data, 'forecast_albedo', str(yr), dt) 
    
#     #------------------------------------------------------
#     clim_autsnw_data_array = read_climatology_file(path_to_autosno_climatological_data, 
#                                                    'autosnow', comn_nme_prt1, doy, comn_nme_prt2)
    
#     wtr_cls_prob_data_array = read_climatology_file(path_to_autosno_climatological_data, 
#                                                     'water_class', comn_nme_prt3, doy, 
#                                                     comn_nme_prt2) 

#     snw_free_lnd_cls_prob_array = read_climatology_file(path_to_autosno_climatological_data, 
#                                                       'snow_free_land_class', 
#                                                       comn_nme_prt3, doy, comn_nme_prt2)   

#     snw_cvrd_lnd_cls_prob_data_array = read_climatology_file(path_to_autosno_climatological_data, 
#                                                             'snow_covered_land_class', 
#                                                             comn_nme_prt3, doy, comn_nme_prt2)

#     ice_cls_prob_data_array = read_climatology_file(path_to_autosno_climatological_data, 
#                                                    'ice_class', comn_nme_prt3, doy, 
#                                                    comn_nme_prt2)
#     #------------------------------------------------------

#     lc_arr[np.isnan(lc_arr)] = -99999

#     lsm_arr[np.isnan(lsm_arr)] = -99999

#     lat_2d_arr = mesh_xy[1]

#     lon_2d_arr = mesh_xy[0]

#     doy_arr = np.empty_like(dwpnt_2m_arr)

#     doy_arr[:,:] = doY
#     #------------------------------------------------------

#     # make and append msg data to list
#     ml_e_arr_lst = [lon_2d_arr, lat_2d_arr, doy_arr, 
#                     dwpnt_2m_arr, temp_2m_arr, sst_arr, 
#                     skt_arr, sea_ice_arr, albedo_arr, 
#                     lc_arr, lsm_arr]
    
#     ml_ec_arr_lst = [lon_2d_arr, lat_2d_arr, doy_arr, 
#                      dwpnt_2m_arr, temp_2m_arr, sst_arr, 
#                      skt_arr, sea_ice_arr, albedo_arr, 
#                      clim_autsnw_data_array, 
#                      wtr_cls_prob_data_array, snw_free_lnd_cls_prob_array, 
#                      snw_cvrd_lnd_cls_prob_data_array, ice_cls_prob_data_array,
#                      lc_arr, lsm_arr, ]   
    
#     #------------------------------------------------------

#     ml_e_arr_rshp = prepare_rf_input(ml_e_arr_lst,x_shp, y_shp)    
    
#     ml_e_rf_predicted = ml_e_rf_model.predict(ml_e_arr_rshp)
#     ml_e_rf_predicted_arr = ml_e_rf_predicted.reshape(y_shp, x_shp)            

#     ml_e_name = '_'.join(['ML-E',yr_DOY,'0.1deg_wgs']) + '.nc'
#     ml_e_file = os.path.join(path_to_put_ml_e_estimates,ml_e_name)
#     spit_nc_file(ml_e_rf_predicted_arr, ml_e_file, lon, lat, 'ML-E', ml_e_summary,)

#     #------------------------------------------------------
#     ml_ec_arr_rshp = prepare_rf_input(ml_ec_arr_lst,x_shp, y_shp)

#     ml_ec_rf_predicted = ml_ec_rf_model.predict(ml_ec_arr_rshp)
#     ml_ec_rf_predicted_arr = ml_ec_rf_predicted.reshape(y_shp, x_shp)

#     ml_ec_name = '_'.join(['ML-EC',yr_DOY,'0.1deg_wgs']) + '.nc'
#     ml_ec_file = os.path.join(path_to_put_ml_ec_estimates,ml_ec_name)
#     spit_nc_file(ml_ec_rf_predicted_arr, ml_ec_file, lon, lat, 'ML-EC', ml_ec_summary)        

#     count += 1

#     if count % 50 == 0:
#         print(str(count) + ' files are read so far')

#         gc.collect()


# print('done!') 
