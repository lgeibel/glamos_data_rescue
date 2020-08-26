from __future__ import absolute_import
import sys
import warnings
import numpy as np
import fortranformat as ff
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
from plotting import plot_map, plot_timeline, plot_histogram, plot_massbalance


fname = "aletsch_annual_ordered.xlsx"
working_dir = r"C:\Users\lea\Documents\data\working_dir\intermediate_steps"
if __name__ == '__main__':
# First run with all annual files:
# define which type of measurements:
    type_list = ["annual", "winter"]
    for m_type in type_list:
        fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version4", m_type)
        fpath6 = os.path.join(r"S:\glazio\projects\8003-VAW_GCOS_data_rescue\version6", m_type)

        fpath_0 = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version0_noduplicates", m_type)

        # Read version0 into list of dataframes as well:
       # DF_list_0 = [file_handling.import_v0(sheet, fpath_0, print_preview = False) \
       #            for sheet in [f for f in os.listdir(fpath_0) \
       #                          if not f.startswith('.')]]
       # gdf_list_0 = file_handling.pd_to_gpd(DF_list_0)

        #Read Excel files into list of pandas dataframes
        DF_list = [file_handling.import_excel(sheet, fpath6, 0, print_preview=False) \
                    for sheet in ["aletsch_annual.xlsx",
                                  "clariden_annual.xlsx",
                                  "untgrindelwald_annual.xlsx",
                                  "gries_annual.xlsx",
                                  "silvretta_annual.xlsx"
                                #   Limmern?
                                # Allalin (P1), Hohlaub (P2), Gietro (P1, P3, P5)
                                # , Rhone
                                  ]]
        # Convert to GeoDataFrame and add crs as LV03
        gdf_list = file_handling.pd_to_gpd(DF_list)
        plot_massbalance(gdf_list)







    if m_type == "winter":
        # read .shp files from local working directory:
        gdf_list_winter = [gpd.read_file(os.path.join(working_dir,filename))\
                     for filename in [f for f in os.listdir(working_dir) \
                                      if f.endswith(m_type+".shp")]]
        gdf_list_summer = [gpd.read_file(os.path.join(working_dir, filename)) \
                            for filename in [f for f in os.listdir(working_dir) \
                                             if f.endswith("annual.shp")]]
        gdf_list_intermediate = [gpd.read_file(os.path.join(working_dir, filename)) \
                            for filename in [f for f in os.listdir(working_dir) \
                                             if f.endswith("intermediate.shp")]]

        print("Let's do some fancy ass plots:")
        plot_map(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)
        #plot_timeline(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)
        #plot_histogram(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)

        print("All finished!")