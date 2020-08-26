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
from plotting import plot_map, plot_timeline, plot_histogram


fname = "aletsch_annual_ordered.xlsx"
working_dir = r"C:\Users\lea\Documents\data\working_dir\intermediate_steps"
if __name__ == '__main__':
# First run with all annual files:
# define which type of measurements:
    type_list = ["annual", "intermediate"]
    for m_type in type_list:
        fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version4", m_type)
        fpath5 = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", m_type)

        fpath_0 = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version0_noduplicates", m_type)

        # Read version0 into list of dataframes as well:
       # DF_list_0 = [file_handling.import_v0(sheet, fpath_0, print_preview = False) \
       #            for sheet in [f for f in os.listdir(fpath_0) \
       #                          if not f.startswith('.')]]
       # gdf_list_0 = file_handling.pd_to_gpd(DF_list_0)

        #Read Excel files into list of pandas dataframes
        DF_list = [file_handling.import_excel(sheet, fpath, 3, print_preview=False) \
                   for sheet in [f for f in os.listdir(fpath)]]
                                 #if f.startswith('sanktanna')]]
        #DF_list = [file_handling.import_excel(sheet, fpath, print_preview=False) \
        #           for sheet in [f for f in os.listdir(fpath) \
        #                         if f.startswith('plattalva')]]
        # Convert to GeoDataFrame and add crs as LV03
        gdf_list = file_handling.pd_to_gpd(DF_list)


        DF_list_5 = [file_handling.import_excel(sheet, fpath5, 0, print_preview=False) \
               for sheet in [f for f in os.listdir(fpath5)]]
                          #       if f.startswith('sanktanna')]]

        # Convert to GeoDataFrame and add crs as LV03
        gdf_list_5 = file_handling.pd_to_gpd(DF_list_5)

        # Compare Version 0 and current version:
       # file_handling.compare_v0(gdf_list, gdf_list_0)

        # read .shp files from local working directory:
        #gdf_list = [gpd.read_file(os.path.join(working_dir,filename))\
        #            for filename in [f for f in os.listdir(working_dir) \
        #                             if f.endswith(m_type+".shp")]]
#        gdf_list = [gpd.read_file(os.path.join(working_dir,filename))\
#                    for filename in [f for f in os.listdir(working_dir) \
#                                     if f.startswith("plt_annual.shp")]]

        #Start data check:
        for i, gdf in enumerate(gdf_list):
 #            print(gdf.Stake[0])
            # rename keys (whats wrong with them?)
            try:
                gdf.columns = ['Stake', 'date0', \
                           'time0', 'date1', 'time1', \
                           'period', 'date-ID', 'x-pos', \
                           'y-pos', 'z-pos', 'position-ID', \
                           'raw_balance', 'density', \
                           'dens-ID', 'Mass Balance', \
                           'quality-I', 'type-ID', 'observer/source', 'Glaciername', 'gk-no', 'inv-no', 'lk25',
                           'geometry']
 #                gdf.fillna(value=gpd.np.nan, inplace=True)
 #
 #
            except ValueError:
                print("What is wrong with me? I dont have all columns. Try again with original file")
                # try again with original file:
                DF1_list = [file_handling.import_excel(sheet, fpath, print_preview=False) \
                                      for sheet in [f for f in os.listdir(fpath) \
                                                    if f.startswith(gdf.Glaciernam[0])]]
                gdf1_list = file_handling.pd_to_gpd(DF1_list)
                gdf = gdf1_list[0]
 #
 #            print(gdf["z-pos"])
 #            #print(gdf[(gdf["z-pos"] < 1300)|np.isnan(gdf['z-pos'])])
 #            if len(gdf[(gdf["z-pos"] < 1300)|np.isnan(gdf['z-pos'])]) > 0:
 #                print("we need to rerun this: ")
 #            ## 1. Check date/time and date_ID
 #             #   gdf = file_handling.check_date(gdf,m_type)
 #
 #             # Read file again (help for code development)
 #             #gdf = file_handling.check_geometry(gdf, m_type)
            gdf = file_handling.adjust_location_ID(gdf, m_type)
            gdf_5 = gdf_list_5[i]
            gdf = file_handling.fill_elevation(gdf, m_type, gdf_5)
 #             ## Print to .shp file as intermediate file:
 #                gdf.to_file(os.path.join(working_dir, str(gdf_list[i].Stake[0])[:3] + "_" + m_type + ".shp"))
 # #           print("This glacier is done")
               # print(gdf[(gdf["z-pos"] < 1300)|np.isnan(gdf['z-pos'])])
               # print(gdf["z-pos"])

            # Now safe all gdfs to version 5:
            # remove all non-relevant columns:
            print("Save to version 5:")
            gdf_print = gdf[['Stake', 'date0', \
                           'time0', 'date1', 'time1', \
                           'period', 'date-ID', 'x-pos', \
                           'y-pos', 'z-pos', 'position-ID', \
                           'raw_balance', 'density', \
                           'dens-ID', 'Mass Balance', \
                           'quality-I', 'type-ID', 'observer/source']]
            try:
                gdf_print.to_excel(os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", m_type,
                                str(gdf_list[i].Glaciername[0]) + "_" + m_type + ".xlsx"), index=False,
                                na_rep='NaN')
            except AttributeError:
                # Something wrong with unterergrindelwald...
                gdf_print.to_excel(os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", m_type,
                                                str(gdf_list[i].Glaciernam[0]) + "_" + m_type + ".xlsx"), index=False,
                                   na_rep='NaN')
            except FileNotFoundError:
                # Something wrong with Sanktanna?
                print("Stop me")
            print(gdf_print["z-pos"])

            #plot_map(gdf_list)

            #print(m_type , "done")



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
        # fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", "annual")
        # DF_list_annual = [file_handling.import_excel(sheet, fpath, print_preview=True) \
        #            for sheet in [f for f in os.listdir(fpath)[13:] \
        #                           if not f.startswith('.')]]
        # gdf_list_summer = file_handling.pd_to_gpd(DF_list_annual)
        # fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", "intermediate")
        # DF_list_intermediate = [file_handling.import_excel(sheet, fpath, print_preview=False) \
        #            for sheet in [f for f in os.listdir(fpath)[13:] \
        #                           if not f.startswith('.')]]
        # gdf_list_intermediate = file_handling.pd_to_gpd(DF_list_intermediate)
        # fpath = os.path.join(r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\version5", "winter")
        # DF_list_winter = [file_handling.import_excel(sheet, fpath, print_preview=False) \
        #            for sheet in [f for f in os.listdir(fpath)[13:] \
        #                           if not f.startswith('.')]]
        # gdf_list_winter = file_handling.pd_to_gpd(DF_list_winter)


        #plot_map(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)
        #plot_timeline(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)
        #plot_histogram(gdf_list_summer, gdf_list_intermediate, gdf_list_winter)

        # # Write output to excel file:
        # sum_name = [win_gla["Glaciernam"][0] for win_gla in gdf_list_summer]
        # win_name = [win_gla["Glaciernam"][0] for win_gla in gdf_list_winter]
        # for name in sum_name:
        #     if name in win_name:
        #         gdf_winter = [num for num in gdf_list_winter if num["Glaciernam"][0] == name][0]
        #         gdf_summer = [num for num in gdf_list_summer if num["Glaciernam"][0] == name][0]
        #         gdf_winter = file_handling.rename_winter_probes(gdf_summer, gdf_winter)
        #         #print(gdf_winter)
        #         gdf_winter.to_file(os.path.join(working_dir, str(gdf_winter.Stake[0])[:3] + "_winter.shp"))
        print("All finished!")