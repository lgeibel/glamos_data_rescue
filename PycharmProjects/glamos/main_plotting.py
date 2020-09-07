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
from plotting import plot_map, plot_timeline, plot_histogram, plot_massbalance_mean, plot_massbalance_individual


fname = "aletsch_annual_ordered.xlsx"
working_dir = r"C:\Users\lea\Documents\data\working_dir\intermediate_steps"
if __name__ == '__main__':
# First run with all annual files:
# define which type of measurements:
    type_list = ["intermediate", "winter"]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 15))
    fig1, axes1 = plt.subplots(nrows=4, ncols=2,sharex=True, figsize=(24, 15))
    for m_type in type_list:
        fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version4", m_type)
        fpath6 = os.path.join(r"S:\glazio\projects\8003-VAW_GCOS_data_rescue\version6", m_type)

        fpath_0 = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version0_noduplicates", m_type)

        #Read Excel files into list of pandas dataframes
        if m_type == "annual":
            DF_list = [file_handling.import_excel(sheet, fpath6, 0, print_preview=False) \
                    for sheet in ["aletsch_" +m_type+".xlsx",
                                  "allalin_" + m_type + ".xlsx",
                                  "basodino_" + m_type + ".xlsx",
                                  "clariden_" +m_type+".xlsx",
                                  #"untgrindelwald_annual.xlsx",
                                  "gietro_" + m_type + ".xlsx",
                                  "gries_" +m_type+".xlsx",
                                  "silvretta_" +m_type+".xlsx"
                                # Allalin (P1), Hohlaub (P2), Gietro (P1, P3, P5)
                                # , Rhone
                                  ]]

        elif m_type == "winter":
            DF_list = [file_handling.import_excel(sheet, fpath6, 0, print_preview=False) \
                    for sheet in ["aletsch_" +m_type+".xlsx",
                                  "allalin_" + m_type + ".xlsx",
                                  "basodino_" + m_type + ".xlsx",
                                  "clariden_" +m_type+".xlsx",
                                  #"untgrindelwald_annual.xlsx",
                                  "gries_" +m_type+".xlsx",
                                  "silvretta_" +m_type+".xlsx",
                                # Allalin (P1), Hohlaub (P2), Gietro (P1, P3, P5)
                                # , Rhone
                                  ]]
        else:
            DF_list = [file_handling.import_excel(sheet, fpath6, 0, print_preview=False) \
                       for sheet in [f for f in os.listdir(fpath6)]]

        # Convert to GeoDataFrame and add crs as LV03
        gdf_list = file_handling.pd_to_gpd(DF_list)
        file_handling.test_deviations(gdf_list)
        #plot_massbalance_mean(gdf_list, m_type, axes)
        #plot_massbalance_individual(gdf_list, m_type, fig1, axes1)

    #plt.show()


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

        print("Let's do some fancy plots:")
        plot_map(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)
        #plot_timeline(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)
        #plot_histogram(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)

        print("All finished!")