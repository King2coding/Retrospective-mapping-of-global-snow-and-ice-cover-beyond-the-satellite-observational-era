'''
These are functions that were used in the Autosnow extending work
'''
#%%
# import packages
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
# import datetime
from datetime import datetime, timedelta, date

import os
import gc

import geopandas as gpd
import fiona
from osgeo import osr
from rasterio import features
import rioxarray as rio
import rasterio
import xarray as xr
import netCDF4
from pyproj import CRS

#%%
# floating variables
cc_epsg = CRS.from_authority(code=4326,auth_name='EPSG')

#%%
# define the functions
def spatial_resampling(data, shape, resampling_technique):
    """
    Spatially resamples the input data to a target shape using the specified resampling technique.

    Parameters:
        data (xarray.DataArray): The input spatial data to be resampled.
        shape (tuple): The target shape (rows, cols) for the resampled data.
        resampling_technique (rio.enums.Resampling): Resampling method (e.g., nearest, bilinear, cubic).
        crs (str or pyproj.CRS): The coordinate reference system (CRS) to write into the data.

    Returns:
        xarray.DataArray: The resampled data with the specified shape and CRS.
    """    

    # Write the CRS to the input data
    data = data.rio.write_crs(cc_epsg.to_string(), inplace=False)

    # Resample the data to the specified shape using the given resampling technique
    resampled_data = data.rio.reproject(
        dst_crs=data.rio.crs,
        shape=shape,  # Target shape (e.g., matching the resolution of the Autosnow data)
        resampling=resampling_technique
    )
    
    return resampled_data 
#-----------------------------------------------------------------------------------------

# function for saving files to compressed netcdf files
def spit_nc_file(data, filename, lon, lat,var_name, summary):
    import datetime

    """
    Writes a 2D data array to a NetCDF file compatible with xarray.open_dataarray.

    Parameters:
        data (2D array): Data to save to the NetCDF file.
        filename (str): Name of the output NetCDF file.
        lon (1D array): Longitude values.
        lat (1D array): Latitude values.
        var_name (str): Name of the variable to store in the NetCDF file.
        summary (str): Description of the data for metadata.
    """
    # Validate input data dimensions
    if data.shape != (len(lat), len(lon)):
        raise ValueError("Data dimensions must match the size of latitude and longitude arrays.")

    # Create NetCDF file
    with netCDF4.Dataset(filename, 'w', format='NETCDF4') as nco:
        # Define dimensions
        nco.createDimension('lon', len(lon))
        nco.createDimension('lat', len(lat))

        # Create longitude variable
        lono = nco.createVariable('lon', 'f4', ('lon',))
        lono.units = 'degrees_east'
        lono.long_name = 'longitude'
        lono[:] = lon  # Assign longitude values

        # Create latitude variable
        lato = nco.createVariable('lat', 'f4', ('lat',))
        lato.units = 'degrees_north'
        lato.long_name = 'latitude'
        lato[:] = lat  # Assign latitude values

        # Create data variable
        data_var = nco.createVariable(var_name, 'f4', ('lat', 'lon'), zlib=True)
        data_var.units = 'unknown'  # You can customize this based on your data
        data_var.long_name = var_name
        data_var[:, :] = data  # Assign the 2D data array

        # Add global attributes
        nco.description = summary
        nco.history = f"Created on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        nco.institution = "The University of Arizona - Hydrology and Atmospheric Sciences"
        nco.source = "Generated using custom Python script."

    # Close the NetCDF file    
    gc.collect()
#-----------------------------------------------------------------------------------------
def make_3d_array(lst_arr):
    """
    Convert a list of 2D arrays into a 3D array.

    Parameters:
        lst_arr (list): A list of 2D numpy arrays of the same shape.

    Returns:
        numpy.ndarray: A 3D array with shape (n, m, len(lst_arr)), 
                       where n, m are the dimensions of each 2D array.
    """
    # Stack the list of 2D arrays along a new third axis
    arr_3d = np.stack(lst_arr, axis=-1)
    return arr_3d
#-----------------------------------------------------------------------------------------
def count_occurrences(lst, target):
    '''
    requires:
    lst = list of numbers
    target = target value to find its occurrences
    '''
    count = 0
    for num in lst:
        if num == target:
            count += 1
    return count

#-----------------------------------------------------------------------------------------
def count_class_pixels(arr):
    """
    Count the total number of pixels in the array with values for each class (0, 1, 2, 3).

    Parameters:
    - arr: A 2D numpy array with class values 0, 1, 2, 3.

    Returns:
    - A dictionary with keys as class values (0, 1, 2, 3) and values as counts of pixels in each class.
    """
    # Flatten the 2D array to a 1D array
    flat_arr = arr.flatten()
    
    # Count occurrences of each class value
    counts = np.bincount(flat_arr)
    
    # Ensure counts for all classes [0, 3], extending the counts array if necessary
    num_classes = 4  # Assuming classes 0, 1, 2, 3
    if counts.size < num_classes:
        # Extend counts array to include zeros for missing classes
        counts = np.pad(counts, (0, num_classes - counts.size), 'constant')
    
    # Create a dictionary {class_value: count}
    class_counts = {class_value: counts[class_value] for class_value in range(num_classes)}
    
    return class_counts

#-----------------------------------------------------------------------------------------
def populate_df_count(df, count_dict, dt, prdt):
    df.loc[dt, prdt +'_wtr_px_cnt'] = count_dict[0]
    df.loc[dt, prdt +'_snfr_px_cnt'] = count_dict[1]
    df.loc[dt, prdt +'_snc_px_cnt'] = count_dict[2]
    df.loc[dt, prdt +'_ice_px_cnt'] = count_dict[3]
#-----------------------------------------------------------------------------------------
def year_day_to_datetime(year, day_of_year):
    from datetime import datetime

    """
    Converts a year and day of year (DOY) into a datetime object, 
    handling leap year adjustments correctly.

    Parameters:
        year (int): The year (e.g., 2023).
        day_of_year (int): The day of the year (1-365 or 1-366 for leap years).

    Returns:
        tuple: A datetime object for the given year and day_of_year, and the adjusted day_of_year.
    """
    # Check if the year is a leap year
    is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    # Adjust the day of the year for leap years
    day_of_year = day_of_year + 1 if is_leap_year and day_of_year >= 60 else day_of_year

    # Create a datetime object
    date_object = datetime.strptime(f"{year}-{day_of_year}", "%Y-%j")

    return date_object, day_of_year

#-----------------------------------------------------------------------------------------
def rasterio_based_save_array_to_disk(path,savename,metadata,arrayTosave):
    '''
	functions saves an array to disk in a desired extension
	requires:
    (i) path = the path to save the map
    (ii) file name for saving the map (add .extension e.g. .tif)
    (iii) map meta data
    (iv) array to save 
    '''
    with rasterio.open(os.path.join(path,savename),'w',**metadata) as mp:
        mp.write(arrayTosave,indexes=1)

#--------------------------------------------------------------------------------------------------------

# function to polygionize an array maps
def array_to_vector(array, out_vector_file, mask_values,crs_item,trns):  

    ext_lists = ['.shx', '.shp', '.prj', '.dbf', '.cpg']  

    array = array.astype(np.int16)
    srs = osr.SpatialReference()   
    srs.SetFromUserInput(crs_item) 
    wgs84 = srs.ExportToProj4()    

    # Create a binary mask based on mask_values
    mask = np.isin(array, mask_values)

    output_records = ( 
                     {'properties': {'raster_val': int(v)}, 'geometry': s}
                     for i, (s, v) in enumerate(features.shapes(array, mask = mask, transform = trns)))
    
    # using fiona, write the dientified records to shapefile     
    with fiona.open(
                out_vector_file, 'w', 
                driver = 'ESRI Shapefile',
                crs = wgs84,
                schema = {'properties': [('raster_val', 'int')],
                        'geometry': 'Polygon'}) as dst:
            dst.writerecords(output_records)
            
    polygon_read = gpd.read_file(out_vector_file) # read the polygon files (clouds converted to polygons)

    file_path, filename = os.path.split(out_vector_file)
    filename, _ = os.path.splitext(filename)

    for ext in ext_lists:
        os.remove(os.path.join(file_path,filename + ext))
    return polygon_read

#---------------------------------------------------------------

# function to find season given month
def find_season(month, hemisphere):
    if hemisphere == 'Southern':
        season_month_south = {
            12:'Summer', 1:'Summer', 2:'Summer',
            3:'Autumn', 4:'Autumn', 5:'Autumn',
            6:'Winter', 7:'Winter', 8:'Winter',
            9:'Spring', 10:'Spring', 11:'Spring'}
        return season_month_south.get(month)
        
    elif hemisphere == 'Northern':
        season_month_north = {
            12:'Winter', 1:'Winter', 2:'Winter',
            3:'Spring', 4:'Spring', 5:'Spring',
            6:'Summer', 7:'Summer', 8:'Summer',
            9:'Autumn', 10:'Autumn', 11:'Autumn'}
        return season_month_north.get(month)
    else:
        print('Invalid selection. Please select a hemisphere and try again')

#--------------------------------------------------------------------------------------------------------

def find_relevant_monday(input_date_str):
    # Convert input string to datetime object
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d').date()
    
    # Calculate the day of the week (0=Monday, 6=Sunday)
    day_of_week = input_date.weekday()
    
    # If the date is already a Monday, return it directly
    if day_of_week == 0:
        return input_date
    else:
        # Find the next Monday
        days_until_next_monday = 7 - day_of_week
        next_monday = input_date + timedelta(days=days_until_next_monday)
        return next_monday

#--------------------------------------------------------------------------------------------------------
def find_monday_of_same_week_past_years(input_date_str, past_years):
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
    week_number = input_date.isocalendar()[1]
    
    mondays_of_same_week = []
    for year in past_years:
        # Find the first Monday of the year
        year_start = datetime(year, 1, 1)
        first_monday = year_start + timedelta(days=(7-year_start.weekday()) % 7)
        # Calculate the Monday of the same week number
        monday_of_same_week = first_monday + timedelta(weeks=week_number-1)
        mondays_of_same_week.append(monday_of_same_week.date())
    
    return mondays_of_same_week

#--------------------------------------------------------------------------------------------------------

def get_aggregated_dates(input_date_str):
    # Parse the input date
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
    
    # Check if the input date is a Monday
    if input_date.weekday() != 0:
        raise ValueError("The input date must be a Monday.")
    
    # Calculate the start date as the Tuesday before the last week
    start_date = input_date - timedelta(days=6)  # Previous Tuesday
    
    # Generate the list of dates for the aggregation period
    aggregated_dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    
    return aggregated_dates

#--------------------------------------------------------------------------------------------------------
def day_of_year(date_str):
    # Parse the input string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # Get the day of the year
    day_of_year = date_obj.timetuple().tm_yday
    return day_of_year

#--------------------------------------------------------------------------------------------------------

def correct_pixel(values):
    # Center pixel is values[4] in a 3x3 kernel
    # center_pixel = values[4]
    # if center_pixel == 3 and np.any(values[[0, 1, 2, 3, 5, 6, 7, 8]] == 1):  # Check for land in the immediate vicinity
    #     return 2  # Reclassify as snow-covered land
    # return center_pixel
    center_pixel = values[4]
    surrounding_land_count = np.sum(np.array(values) == 1)
    surrounding_water_count = np.sum(np.array(values) == 0)
    
    # Only reclassify ice (3) to snow-covered land (2) if there's significant land presence
    if center_pixel == 3 and surrounding_land_count > 4:
        return 2  # Reclassify as snow-covered land
    elif center_pixel == 3 and surrounding_water_count >= 5:
        return 3  # Keep as ice to preserve small water bodies covered with ice
    return center_pixel

#--------------------------------------------------------------------------------------------------------

# compute hit/mis counts
def calculate_hits_miss(predicted, observed, integer_list, kind):
    """
    Calculate hits for each integer in the list based on predicted and observed values.

    Parameters:
    - predicted: 1D array representing predicted values.
    - observed: 1D array representing observed values.
    - integer_list: List of integers (0, 1, 2, 3, etc.).
    - kind: hits or misses is computed

    Returns:
    - hits: A 1D array containing the count of hits for each integer.
    """
    if kind == 'hit':
        hits = np.sum([np.sum((observed == i) & (predicted == i)) for i in integer_list])
        return hits
    elif kind == 'miss':
        miss = np.sum([np.sum((observed == i) & (predicted != i)) for i in integer_list])
        return miss
#--------------------------------------------------------------------------------------------------------

def prepare_rf_input(array_list, x_shape, y_shape):
    """
    Prepare input data for a Random Forest model.
    
    Parameters:
    - array_list (list of np.array): List of 2D arrays to be stacked and processed.
    - x_shape (int): Width (number of columns) of the target image.
    - y_shape (int): Height (number of rows) of the target image.
    
    Returns:
    - reshaped_array (np.array): Reshaped 2D array ready for Random Forest prediction.
    """
    # Check if the number of arrays matches the desired depth of the 3D array
    arr_depth = len(array_list)
    # Stack input arrays into a 3D array along the last axis
    arr = np.stack(array_list, axis=-1)
    
    # Reshape the 3D array into a 2D array (pixels x features)
    reshaped_array = arr.reshape((y_shape * x_shape), arr_depth)
    
    return reshaped_array
#--------------------------------------------------------------------------------------------------------

# Define and read ERA5 input files
def read_era5_file(path_to_era5_data, variable, year, dt):
    
    file_name = '_'.join([variable, year, 'daily_mean.nc'])
    file_path = os.path.join(path_to_era5_data, file_name)

    if variable == '2m_dewpoint_temperature':
        arr = xr.open_dataset(file_path).sel(time = dt).d2m.values
    elif variable == '2m_temperature':
        arr = xr.open_dataset(file_path).sel(time = dt).t2m.values
    elif variable == 'sea_ice_cover':
        arr = xr.open_dataset(file_path).sel(time=dt).siconc.values
    elif variable == 'sea_surface_temperature':
        arr = xr.open_dataset(file_path).sel(time=dt).sst.values
    elif variable == 'skin_temperature':
        arr = xr.open_dataset(file_path).sel(time=dt).skt.values
    elif variable == 'forecast_albedo':
        arr = xr.open_dataset(file_path).sel(time=dt).fal.values
    
    return np.where(np.isnan(arr), -99999, arr)
#--------------------------------------------------------------------------------------------------------

# Read climatology-based data
def read_climatology_file(path_to_autosno_climatological_data, name_parts, ext):
    file_name = '_'.join(name_parts) + ext
    file_path = os.path.join(path_to_autosno_climatological_data, file_name)
    if ext == '.tif':
        arr = xr.open_dataarray(file_path).data[0, :, :]
    else:
        arr = xr.open_dataarray(file_path).data
    return np.where(np.isnan(arr), -99999, arr)
#--------------------------------------------------------------------------------------------------------

# Read estimated files
def read_processed_files(path_to_data, name_parts, ext):
    filename = '_'.join(name_parts) + ext
    filename = os.path.join(path_to_data, filename)
    if ext == '.tif':
        arr = xr.open_dataarray(filename).data[0, :, :]
    else:
        arr = xr.open_dataarray(filename).data
    return np.where(arr > 3,np.nan,arr)

#--------------------------------------------------------------------------------------------------------

def get_percent_hitmiss(df, obs, pred, prdt, dt, integer_list):
    # get counts
    input_arr = np.column_stack([obs.flatten(), pred.flatten()])
    hit_count = calculate_hits_miss(input_arr[:,1],input_arr[:,0],integer_list,'hit')
    miss_count = calculate_hits_miss(input_arr[:,1],input_arr[:,0],integer_list,'miss') 
    total_count = hit_count + miss_count

    # calculate hit/miss in %        
    df.loc[dt, prdt + '-GMASI-hit'] = round((hit_count/total_count)*100,2)
    df.loc[dt, prdt + '-GMASI-miss'] = round((miss_count/total_count)*100,2)

#--------------------------------------------------------------------------------------------------------

def df_hit_miss_per_class(hitmis_df, obs, pred, dte, prdt):

    input_arr = np.column_stack([obs.flatten(), pred.flatten()])

    wtr_hit = np.sum((input_arr[:,0] == 0) & (input_arr[:,1] == 0))
    wtr_miss = np.sum((input_arr[:,0] == 0) & (input_arr[:,1] != 0))
    wtr_cnt = wtr_hit + wtr_miss
    hitmis_df.loc[dte, prdt +'-wter-hit']  = round((wtr_hit/wtr_cnt)*100,2) 
    hitmis_df.loc[dte, prdt +'-wter-miss']  = round((wtr_miss/wtr_cnt)*100,2) 

    snfr_hit = np.sum((input_arr[:,0] == 1) & (input_arr[:,1] == 1))
    snfr_miss = np.sum((input_arr[:,0] == 1) & (input_arr[:,1] != 1))
    snfr_cnt = snfr_hit + snfr_miss
    hitmis_df.loc[dte, prdt +'-snflnd-hit']  = round((snfr_hit/snfr_cnt)*100,2) 
    hitmis_df.loc[dte, prdt +'-snflnd-miss']  = round((snfr_miss/snfr_cnt)*100,2) 

    snc_hit = np.sum((input_arr[:,0] == 2) & (input_arr[:,1] == 2))
    snc_miss = np.sum((input_arr[:,0] == 2) & (input_arr[:,1] != 2))
    snc_cnt = snc_hit + snc_miss
    hitmis_df.loc[dte, prdt +'-snclnd-hit']  = round((snc_hit/snc_cnt)*100,2) 
    hitmis_df.loc[dte, prdt +'-snclnd-miss']  = round((snc_miss/snc_cnt)*100,2) 

    ice_hit = np.sum((input_arr[:,0] == 3) & (input_arr[:,1] == 3))
    ice_miss = np.sum((input_arr[:,0] == 3) & (input_arr[:,1] != 3))
    ice_cnt = ice_hit + ice_miss
    hitmis_df.loc[dte, prdt +'-ice-hit']  = round((ice_hit/ice_cnt)*100,2) 
    hitmis_df.loc[dte, prdt +'-ice-miss']  = round((ice_miss/ice_cnt)*100,2) 

#--------------------------------------------------------------------------------------------------------

# fucntion that compute the area of polygons in shape file
def get_total_area(geo_df,clss_val):

    geo_df = geo_df.loc[geo_df['raster_val'] == clss_val]

    # convert to Lambert Cylindrical Equal Area crs; ideal for area computations
    geo_df = geo_df.to_crs({'proj':'cea'}) 

    # Calculate the total area in square kilometers
    total_area_km2 = geo_df['geometry'].area.sum() / 1e6 

    return total_area_km2
#--------------------------------------------------------------------------------------------------------

def calcualte_total_area(df, arr, intermediate_path, dt1, dt2, prdt, integer_list, crs_obj, geo_trns):
    intermediate_file = '_'.join([prdt, dt1]) + '.shp'
    intermediate_file_name = os.path.join(intermediate_path, intermediate_file)

    arr_ = arr.astype(np.int16).copy() 

    arr_to_gdf = array_to_vector(arr_, intermediate_file_name, integer_list, crs_obj.to_string(),geo_trns,)       

    df.loc[dt2, prdt + '_wtr_total_area'] = get_total_area(arr_to_gdf,0)
    df.loc[dt2, prdt + '_snfr_total_area'] = get_total_area(arr_to_gdf,1)
    df.loc[dt2, prdt + '_snc_total_area'] = get_total_area(arr_to_gdf,2)
    df.loc[dt2, prdt + '_ice_total_area'] = get_total_area(arr_to_gdf,3)

#--------------------------------------------------------------------------------------------------------

# Function to extract year day-of-year (YYYYDDD) based on file type
def extract_yeardoy(file_name):
    base_name = os.path.basename(file_name)
    if base_name.startswith("gmasi_snowice_reproc_v003"):
        # For gmasi files, YYYYDDD is in the 4th part
        return int(base_name.split('_')[4])
    elif base_name.startswith("UofA_gmasi_snowice_extended_v001"):
        # For extended files, YYYYDDD is in the 5th part
        return int(base_name.split('_')[5])
    else:
        raise ValueError(f"Unknown file naming format: {file_name}")
    
#--------------------------------------------------------------------------------------------------------

def compute_area_from_raster(df, dt, arr, resolution_degrees, prdt):
    """
    Compute the area of a specific raster value directly from the raster data.
    
    Parameters:
    - arr (numpy array): The input raster array.
    - raster_value (int): The raster value to calculate the area for (e.g., 1 for snow cover).
    - resolution_degrees (float): Resolution of the raster in degrees (e.g., 0.1 for a 0.1-degree grid).
    - latitudes (1D numpy array): Latitude values corresponding to the rows of the array.

    Returns:
    - total_area_km2 (float): Total area in square kilometers.
    """
    # Earth's radius in kilometers
    R = 6371.0

    # Convert resolution from degrees to radians
    resolution_radians = np.deg2rad(resolution_degrees)

    # Calculate the area of a single grid cell for each latitude
    cell_areas = (R ** 2) * resolution_radians ** 2 * np.cos(np.deg2rad(arr.lat))

    # Count the number of cells matching the raster value for each latitude
    matching_snow_cells = (arr == 2).sum(axis=1)  # Sum cells row by row
    matching_ice_cells = (arr == 3).sum(axis=1)  # Sum cells row by row

    # Compute the total area
    total_snow_area_km2 = np.sum(matching_snow_cells * cell_areas)
    total_ice_area_km2 = np.sum(matching_ice_cells * cell_areas)

    # df.loc[dt, prdt + '_wtr_total_area'] = get_total_area(arr_to_gdf,0)
    # df.loc[dt, prdt + '_snfr_total_area'] = get_total_area(arr_to_gdf,1)
    df.loc[dt, prdt + '_snc_total_area'] = total_snow_area_km2
    df.loc[dt, prdt + '_ice_total_area'] = total_ice_area_km2
