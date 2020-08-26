from __future__ import absolute_import
import sys
import warnings
import pandas as pd
import geopandas as gpd
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
from analysis import density_analysis, plot_densities, interpolate_densities


fname = "aletsch_annual_ordered.xlsx"
working_dir = r"C:\Users\lea\Documents\data\working_dir\intermediate_steps"
if __name__ == '__main__':
# First run with all annual files:
# define which type of measurements:
#     type_list = ["annual", "intermediate", "winter"]
#     rdf_list =[0,1,2]
#     for m_type in type_list:
#         fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", m_type)
#
#         fpath_0 = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version0_noduplicates", m_type)
#
#         #Read Excel files into list of pandas dataframes
#         DF_list = [file_handling.import_excel(sheet, fpath, 0, print_preview=False) \
#                    for sheet in [f for f in os.listdir(fpath)]]
#                                  #if f.startswith('sanktanna')]]
#         #DF_list = [file_handling.import_excel(sheet, fpath, print_preview=False) \
#         #           for sheet in [f for f in os.listdir(fpath) \
#         #                         if f.startswith('plattalva')]]
#         # Convert to GeoDataFrame and add crs as LV03
#         gdf_list = file_handling.pd_to_gpd(DF_list)
#
#         print("Lets do this Density Thing:")
#         rdf = density_analysis(gdf_list, m_type)
#         if m_type == "annual":
#             rdf_list[0] = rdf
#         if m_type == "intermediate":
#             rdf_list[1] = rdf
#         if m_type == "winter":
#             rdf_list[2] = rdf
#     rdf_all = gpd.GeoDataFrame(pd.concat(rdf_list, ignore_index=True) )
#
#     rdf_all.to_file(os.path.join(working_dir, 'measures_densities.txt', 'measures_densities.shp'))

    rdf_all = gpd.read_file(os.path.join(working_dir, 'measures_densities.txt', 'measures_densities.shp'))
    model, predictions = plot_densities(rdf_all)

    print("So what now?")
   # Read ALL data, find (for each glacier
    type_list = ["annual", "intermediate", "winter"]
    rdf_list =[0,1,2]
    for m_type in type_list:
        fpath = os.path.join(r"S:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", m_type)

        #Read Excel files into list of pandas dataframes
        DF_list = [file_handling.import_excel(sheet, fpath, 0, print_preview=False) \
                   for sheet in [f for f in os.listdir(fpath)
                   #if f.startswith('silvretta')
                                 ]]

        # Convert to GeoDataFrame and add crs as LV03
        gdf_list = file_handling.pd_to_gpd(DF_list)

        # Do some fancy interpolation with the model for each glacier here:
        gdf_list = interpolate_densities(model, gdf_list, m_type)

        # write  gdf_list into a new version 5:
        # # Write output to excel file:
        fpath6 = os.path.join(r"S:\glazio\projects\8003-VAW_GCOS_data_rescue\version6", m_type)

        for gdf in gdf_list:
            gdf_print = gdf[['Stake', 'date0', \
                             'time0', 'date1', 'time1', \
                             'period', 'date-ID', 'x-pos', \
                             'y-pos', 'z-pos', 'position-ID', \
                             'raw_balance', 'density', \
                             'dens-ID', 'Mass Balance', \
                             'quality-I', 'type-ID', 'observer/source']]
            gdf_print.to_excel(os.path.join(fpath6,
                                            str(gdf.Glaciername[0]) + "_" + m_type + ".xlsx"), index=False,
                               na_rep='NaN')
    print("All finished!")