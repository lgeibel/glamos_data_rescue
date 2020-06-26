from __future__ import absolute_import
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(r"C:\Users\lea\Documents\GitHub\glamos_data_rescue")

from pandas import DataFrame, read_csv
import os
# Fix Random Basemaps Bug (adjust filepath when running
# somewhere else or in different environment)
os.environ['PROJ_LIB'] = r'C:\Users\lea\anaconda3\envs\glamos_ve1\Lib\site-packages\mpl_toolkits\basemap\data'
import matplotlib.pyplot as plt
import geopandas as gpd

import file_handling
from plotting import plot_map


fname = "aletsch_annual_ordered.xlsx"
working_dir = r"C:\Users\lea\Documents\data\working_dir\intermediate_steps"
if __name__ == '__main__':
# First run with all annual files:
# define which type of measurements:
    type_list = ["annual", "intermediate", "winter"]
    for m_type in type_list:
        fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version4", m_type)
        fpath_0 = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version0_noduplicates", m_type)

        # Read version0 into list of dataframes as well:
       # DF_list_0 = [file_handling.import_v0(sheet, fpath_0, print_preview = False) \
       #            for sheet in [f for f in os.listdir(fpath_0) \
       #                          if not f.startswith('.')]]
       # gdf_list_0 = file_handling.pd_to_gpd(DF_list_0)

        # Read Excel files into list of pandas dataframes
        #DF_list = [file_handling.import_excel(sheet, fpath, print_preview=False) \
        #           for sheet in [f for f in os.listdir(fpath) \
        #                         if not f.startswith('.')]]
        ## Convert to GeoDataFrame and add crs as LV03
        #gdf_list = file_handling.pd_to_gpd(DF_list)

        # Compare Version 0 and current version:
       # file_handling.compare_v0(gdf_list, gdf_list_0)

        # read .shp files from local working directory:
        gdf_list = [gpd.read_file(os.path.join(working_dir,filename))\
                    for filename in [f for f in os.listdir(working_dir) \
                                     if f.endswith(m_type+".shp")]]

        #Start data check:
        for i, gdf in enumerate(gdf_list):
            ## 1. Check date/time and date_ID
            #gdf = file_handling.check_date(gdf,m_type)
            ## Print to .shp file as intermediate file:
            #gdf.to_file(os.path.join(working_dir, str(gdf_list[i].Stake[0])[:3]+"_"+ m_type + ".shp"))

            # Read file again (help for code development)
            ### THIS IS NOT FINISHED YET! Check annual and winter
            #gdf = file_handling.check_geometry(gdf, m_type)
            #gdf = file_handling.adjust_location_ID(gdf, m_type)
            gdf = file_handling.fill_elevation(gdf)
            print("Doing some fancy corrections here")


        plot_map(gdf_list)