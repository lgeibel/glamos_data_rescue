import os
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
import contextily as ctx
import more_itertools as mit
from shapely.geometry import Point, LineString, Polygon
from itertools import chain
import mapclassify as mc
import geoplot.crs as gcrs
import numpy as np
import geoplot
from math import log
import matplotlib.ticker as ticker
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def plot_map(gdf_list, gdf_list_intermediate, gdf_list_winter):
    '''Plots Map of Switzerland with all data points on it:

    Keyword arguments:
    df -- list of pandas geodata frame with all  mb measurements of each glacier


    Returns: none

    '''
    f, ax = plt.subplots(1)
    # setup background of Map
    world = gpd.read_file(r"C:\Users\lea\Documents\data\plots\CHE_adm0.shp")
    water = gpd.read_file(r"C:\Users\lea\Documents\data\plots\waterways.shp")
    sgi = gpd.read_file(r"C:\Users\lea\Documents\data\plots\SGI_2010.shp")
    df = pd.read_csv(r"C:\Users\lea\Documents\data\plots\ch_digi.dat", delim_whitespace=True,header=1, thousands = ".")

    gdf_all = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y))
    gdf_all.crs = {'init' :'epsg:21781'}

    gdf_river_points = gdf_all[gdf_all["ID"]>=14]

    gdf_river = gdf_all[gdf_all["ID"]>=14]
    gdf_river = gdf_river.groupby('ID')['geometry'].apply(lambda x: LineString(x.tolist())).reset_index()
    # Declare the result as a new a GeoDataFrame
    gdf_river = gpd.GeoDataFrame(gdf_river, geometry='geometry')

    gdf_all = gdf_all[gdf_all["ID"]<14]
    gdf_all['geometry'] = gdf_all['geometry'].apply(lambda x: x.coords[0])
    gdf_all = gdf_all.groupby('ID')['geometry'].apply(lambda x: Polygon(x.tolist())).reset_index()
    # Declare the result as a new a GeoDataFrame
    gdf_all = gpd.GeoDataFrame(gdf_all, geometry='geometry')

    borders = gdf_all[gdf_all["ID"] == 0]
    lakes = gdf_all[(gdf_all["ID"]>0) & (gdf_all["ID"]<14)]
    rivers = gdf_river

    name_list = ['Albina', 'Aletsch', 'Allalin', 'BlauSchnee','Cantun',\
                 'NaN', 'Clariden', 'NaN',\
                 'Dents Du Midi', 'Diablerets', 'Piz d\'Err', 'Forno', \
                 'Gietro', 'Gorner', 'Gries','NaN', 'NaN', 'NaN',\
                 'Jöri', 'Limmern', \
                 'Misaun', 'Pers', 'NaN', \
                 'Plattalva', 'Rhone', 'Rosatsch', \
                 'Silvretta', 'NaN', 'NaN']
    name_list = ['Alb', 'Ale', 'Al', 'Bla','Can',\
                 'NaN', 'Cla', 'NaN',\
                 'DdM', 'Dia', 'Err', 'For', \
                 'Gie', 'Gor', 'Gri','NaN', 'NaN', 'NaN',\
                 'Jör', 'Lim', \
                 'Mis', 'Per', 'NaN', \
                 'Plat', 'Rhone', 'Ros', \
                 'Sil', 'NaN', 'NaN']

    world = world.to_crs(epsg=21781)
    water = water.to_crs(epsg=21781)
    sgi = sgi.to_crs(epsg=21781)
    pt_size = []
    df_pt = gdf_list[0].loc[0:1]
    for i, gdf in enumerate(gdf_list):
        #print(i, len(name_list), len(df_list))
        year_list = [int(str(gdf["date1"][i])[:4]) for i in gdf.index]
        gdf["year"] = year_list
        # drop duplicates:
        unique = gdf["year"].drop_duplicates()
        #get label for 5 longest time series:
    # Plot each glacier individually
        pt_size.append(len(unique))
        df_pt = df_pt.append(gdf_list[i].loc[0])
        print(i, gdf_list[i].loc[0], df_pt)
    df_pt = df_pt.iloc[2:][:]
    # Change point size list to log scale:
    pt_size_large = [y*4 for y in pt_size]
    df_pt["pt_size"] = pt_size
    df_pt["pt_size_l"] = pt_size_large

   # df_pt[df_pt.nlargest(5, 'pt_size')]

    df_pt.plot(ax=ax, column="pt_size", \
               markersize = df_pt["pt_size_l"],\
               cmap='viridis', zorder = 5, alpha = 0.8, \
               legend="True", legend_kwds={'label': "Years of Observation"}, edgecolor = "black")
    largest = df_pt.nlargest(5, 'pt_size') # get 5 longest time series
    label_point(largest["x-pos"], largest["y-pos"], largest["Glaciernam"], ax)

    # Add basemap:
    #world.geometry.boundary.plot(ax=ax, color=None, edgecolor='grey', zorder=1)
    #water.plot(ax=ax, zorder=2, color='cornflowerblue' )
    #gdf_new.plot(ax=ax, color='grey', zorder=0, linewidth = 5)
    borders.boundary.plot(ax=ax, zorder=0, linewidth = 1)
    gdf_river_points.plot(ax=ax, zorder=1, color= 'lightsteelblue', markersize = 0.05 )
    lakes.plot(ax=ax, zorder=2, color='lightsteelblue', linewidth = 5)

    sgi.plot(ax=ax, zorder=3, color = 'lightslategrey')



    plt.title("Location of Observed Glaciers")

    plt.show()
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y']+10000, str(point['val'].title()), \
                fontsize = 12, zorder = i + 5, weight= "bold", color = "black")

def plot_timeline(gdf_list_annual, gdf_list_intermediate, gdf_list_winter):
    """Plot timeline for all glaciers for all measurement types"""
    gl_net_info = pd.read_excel(\
         r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\gl_net_info1.xlsx",
                     header = 1,
         names=['gk-no', 'short', 'lc', 'mb', 'vel', \
                  'name', 'real name', 'inv-no', \
                  'wgms-no', 'kanton', 'gemeinde', 'lk25',
                  '1', '1', '1', '1', '1', '1', \
                  '1', '1', '1', '1', '1', '1', \
                  '1', '1', '1', '1', '1', \
                  '1', '1', '1' \
                ])
     # get glacier ID (inv-no) and gk-no, kartenblatt lk25 for all glaciers as list



    fig, (ax1,ax2) = plt.subplots(1,2, figsize= (8, 20))

    # Create list of all glacier names:
    name_list = [gdf_list_annual[i]["Glaciernam"][0] for i in range(len(gdf_list_annual))]
    for gdf in gdf_list_intermediate:
        # add glaciers with only intermediate values
        if gdf["Glaciernam"][0] not in name_list:
            name_list.append(gdf["Glaciernam"][0])
    for gdf in gdf_list_winter:
        # add glaciers with only seasonal values
        if gdf["Glaciernam"][0] not in name_list:
            name_list.append(gdf["Glaciernam"][0])
    # Get real name of glaciers from gl_net_info.xlsx file:
    real_name_list = []
    for name in name_list:
        real_name = [gl_net_info["real name"][i] \
            for i in range(len(gl_net_info["real name"]))]
        glacier_loc = [gl_net_info["short"][i][1:] \
                   for i in range(len(gl_net_info["inv-no"]))].index(name)
        real_ind = real_name[glacier_loc][1:]
        real_name_list.append(real_ind)
    print(real_name_list)

    # For each entry, create tuples with first entry x min and x width
    for ind, gdf in enumerate(gdf_list_annual):
        gdf = gdf.sort_values(by=['date0', 'date1'])
        # make new column with year of date1:
        year_list = [int(str(gdf["date1"][i])[:4]) for i in gdf.index]
        gdf["year"] = year_list
        # drop duplicates:
        unique = gdf["year"].drop_duplicates()
        # First index that is not continuous:
        tuple_list_annual = list(find_ranges(unique))
        y_loc = name_list.index(gdf["Glaciernam"][0])
        # if glacier is in first half of name_list, plot to ax1, otherwise to ax2
        if gdf["Glaciernam"][0] in name_list[:round((len(name_list)/2))]:
            if ind == 1:
                ax1.broken_barh(tuple_list_annual, (y_loc*20,5) , facecolors='tab:orange', label = "Annual")
            else:
                ax1.broken_barh(tuple_list_annual, (y_loc*20,5) , facecolors='tab:orange')
        else:
            ax2.broken_barh(tuple_list_annual, (y_loc*20, 5), facecolors='tab:orange')



    for ind, gdf in enumerate(gdf_list_intermediate):
        gdf = gdf.sort_values(by=['date0', 'date1'])
        # make new column with year of date1:
        year_list = [int(str(gdf["date1"][i])[:4]) for i in gdf.index]
        gdf["year"] = year_list
        # drop duplicates:
        unique = gdf["year"].drop_duplicates()
        # List of tuple of continuous years plus length of series::
        tuple_list_annual = list(find_ranges(unique))
        #
        y_loc = name_list.index(gdf["Glaciernam"][0])
        if gdf["Glaciernam"][0] in name_list[:(round(len(name_list) / 2))]:
            if ind == 1:
                ax1.broken_barh(tuple_list_annual, (y_loc*20+5,4) , facecolors='tab:blue', label = "Intermediate")
            else:
                ax1.broken_barh(tuple_list_annual, (y_loc*20+5,4) , facecolors='tab:blue')
        else:
            ax2.broken_barh(tuple_list_annual, (y_loc * 20+5, 4), facecolors='tab:blue')

    for ind, gdf in enumerate(gdf_list_winter):
        gdf = gdf.sort_values(by=['date0', 'date1'])
        # make new column with year of date1:
        year_list = [int(str(gdf["date1"][i])[:4]) for i in gdf.index]
        gdf["year"] = year_list
        # drop duplicates:
        unique = gdf["year"].drop_duplicates()
        # First index that is not continuous:
        tuple_list_annual = list(find_ranges(unique))
        y_loc = name_list.index(gdf["Glaciernam"][0])
        if gdf["Glaciernam"][0] in name_list[:(round(len(name_list) / 2))]:
            if ind == 1:
                ax1.broken_barh(tuple_list_annual, (y_loc*20+9,4) , facecolors='tab:green', label = "Winter")
            else:
                ax1.broken_barh(tuple_list_annual, (y_loc*20+9,4) , facecolors='tab:green')
        else:
            ax2.broken_barh(tuple_list_annual, (y_loc * 20+9, 4), facecolors='tab:green')

    ax1.set_ylim(0, 650)
    ax1.set_xlim(1885, 2019)
    ax1.set_xlabel('Year', fontsize=5)
    ax1.invert_yaxis()
    #ax1.yaxis.tick_right()
    ax1.set_yticks(range(0, len(name_list)*10, 20))
    ax1.set_xticks(range(1885, 2019, 10))
    #ax.set_yticklabels(real_name_list)
    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1.yaxis.set_minor_locator(ticker.FixedLocator(range(10, len(name_list)*10+10, 20)))
    ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(real_name_list[:(round(len(name_list) / 2))]))
    ax1.tick_params(axis='y', which='major', labelsize=0)
    ax1.tick_params(axis='y', which='minor', labelsize=7)


    ax1.tick_params(axis="x", labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.grid(True)
    ax1.legend(loc='lower left', fontsize = 7)

    ax2.set_ylim(620, 1100)
    ax2.set_xlim(1885, 2019)
    ax2.set_xlabel('Year', fontsize=5)
    ax2.invert_yaxis()
    #ax2.yaxis.tick_right()
    ax2.set_yticks(range(len(name_list*10)+10, len(name_list)*20, 20))
    ax2.set_xticks(range(1885, 2019, 10))
    #ax.set_yticklabels(real_name_list)
    ax2.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(ticker.FixedLocator(range(len(name_list)*10+20, len(name_list)*20+20, 20)))
    ax2.yaxis.set_minor_formatter(ticker.FixedFormatter(real_name_list[(round(len(name_list) / 2)):]))
    ax2.tick_params(axis='y', which='major', labelsize=0)
    ax2.tick_params(axis='y', which='minor', labelsize=7)

    ax2.tick_params(axis="x", labelsize=7)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.grid(True)

    plt.title('Available Measurement Types and Periods for each Glacier')


    plt.tight_layout()
    plt.savefig("foo.eps", papertype='a4', orientation='portrait', format='eps')
    plt.savefig("foo.pdf", bbox_inches='tight', format='pdf')

    plt.show()

    print("Plot Timeline:")

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0], 1
        else:
            #print difference between first and last consecutive number:
            yield group[0], (group[-1]-group[0])

def plot_histogram(gdf_list_annual, gdf_list_intermediate, gdf_list_winter):
    """Plot histogram showing the amount of measured glaciers per year"""
    print("Plot Histogram")
    fig, ax = plt.subplots()
    df = pd.read_csv("temp.csv")
    avg_a = df["glacier_bins_a"]
    avg_w = df["glacier_bins_b"]
    window_size = 5

    i = 0
    moving_averages_a = []
    moving_averages_w = []

    while i < len(avg_a):
        this_window_a = avg_a[i: i + window_size]
        this_window_w = avg_w[i: i + window_size]

        window_average_a = sum(this_window_a) / window_size
        window_average_w = sum(this_window_w) / window_size

        moving_averages_a.append(window_average_a)
        moving_averages_w.append(window_average_w)

        i += window_size

    avg_year= df["year"][1::5].to_list()
    df_avg = pd.DataFrame(list(zip(moving_averages_a, moving_averages_w,avg_year )),
                      columns=['Annual', 'Winter', 'Year'])


    ax = df_avg.plot.bar(x="Year", y=["Annual", "Winter"], color = ["tab:orange", "tab:blue"],legend= False, width=0.7, grid=True)
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels=('Annual Measurements','Winter Measurements'), loc='best', fontsize = 18)
    ax.set_xlabel('Year', fontsize = 20)
    ax.set_ylabel('Number of monitored glaciers (#)', fontsize = 20)
 #   ax.set_xticklabels([t if not (i-1) % 5 else "" for i, t in enumerate(ax.get_xticklabels())])
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    plt.show()


    glacier_bins_a = []
    measurements_bins_a = []
    glacier_annual = 0
    measurements_annual = 0
    glacier_bins_w = []
    measurements_bins_w = []
    glacier_winter = 0
    measurements_winter = 0
    for year in range(1884,2020):
        print(year)
        # Find entries with "year" in gdf_list_annual
        for gdf in gdf_list_annual:
            year_list = [int(str(gdf["date1"][i])[:4]) for i in gdf.index]
            if year in year_list:
                glacier_annual = glacier_annual + 1
                measurements_annual = measurements_annual + year_list.count(year)
        try:
            print(glacier_annual, measurements_annual)
            measurements_bins_a.append(measurements_annual)
            glacier_bins_a.append(glacier_annual)
            print("Annual:" , glacier_bins_a, measurements_bins_a)
            glacier_annual = 0
            measurements_annual = 0
        except AttributeError:
            print("What is the problem?")
            glacier_annual = 0
            measurements_annual = 0
        for gdf in gdf_list_winter:
            year_list = [int(str(gdf["date1"][i])[:4]) for i in gdf.index]
            if year in year_list:
                glacier_winter = glacier_winter + 1
                measurements_winter = measurements_winter + year_list.count(year)
                print(glacier_winter, measurements_winter)
        try:
            print(glacier_winter, measurements_winter)
            measurements_bins_w.append(measurements_winter)
            glacier_bins_w.append(glacier_winter)
            print("Winter: ", glacier_bins_w, measurements_bins_w)
            glacier_winter = 0
            measurements_winter = 0
        except AttributeError:
            glacier_winter = 0
            measurements_winter = 0

    print("Now plot this:", measurements_bins_a, measurements_bins_w, glacier_bins_a, glacier_bins_w)
    plt.hist(measurements_annual, bins= glacier_bins_a)

def plot_massbalance(gdf_list):
    """ Plot the annual mass balance time series of a few selected, particularly long series
    normalize them with median of time series (maybe chose a fixed time period for that?)
    to be able to compare them"""
    fig, axes = plt.subplots(nrows=1, ncols=len(gdf_list), figsize=(24, 15))
    ###########  ANNUAL BALANCE ###########
    for i, glacier in enumerate(gdf_list):
        print("Glacier : ",glacier.Glaciername[0])
        # sort by time
        glacier = glacier.sort_values("date1")
        # Remove summer measurements:
        s = axes[i].scatter(glacier.date1 / 10000, \
                            glacier["z-pos"],
                            label = glacier.Glaciername.iloc[0],
                            s = 20, zorder = 1)
        axes[i].legend(loc="upper left")

        # plt.show()

        print(" Glacier_s : ", glacier.Glaciername[0])
        if glacier.Glaciername[0] =="silvretta":
            print("stooop")
            axes[0]=glacier.plot()

            for x, y, label in zip(glacier.geometry.x, glacier.geometry.y, glacier.Stake):
                axes[0].annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")

        # For each glacier, pick first stake/first stakes that we want to look at:
        glacier_s = glacier
        plot_i = []
        glacier["Year"] = [int(str(glacier.date1.iloc[i])[:4]) for i, y in enumerate(glacier.date1)]

        if len(glacier_s) > 0:
            if glacier["Glaciername"].iloc[0] == "aletsch":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "ale-PO"]
                init_i = init_stakes[init_stakes.date1 == min([i for i in init_stakes.date1])]
                print(init_i)


                # find each closest measurement for all following years
                for year in range(int(str(init_i.date1.iloc[0])[:4]), 2019, 1):
                    print(year)
                    this_year = glacier[glacier.Year == year]
                    gpd2 = this_year.geometry.unary_union
                    gpd2_pts_list = this_year.geometry.tolist()
                    gpd2_pts = MultiPoint(gpd2_pts_list)
                    closest = nearest(init_i, gpd2_pts, gpd2)
                    # Find closest point to init_i


#                 plot_i = glacier_s[glacier_s["z-pos"] > 3000].copy()
#             elif glacier_s["Glaciernam"].iloc[0] == "clariden":
#                 plot_i = glacier_s[(glacier_s["z-pos"] > 2800) & (glacier_s["z-pos"] < 2930)].copy()
#             elif glacier_s["Glaciernam"].iloc[0] == "forno":
#                 plot_i = glacier_s[glacier_s["z-pos"] > 3000].copy()
#             elif glacier_s["Glaciernam"].iloc[0] == "plainemorte":
#                 plot_i = glacier_s.copy()
#             elif glacier_s["Glaciernam"].iloc[0] == "rhone":
#                 plot_i = glacier_s[(glacier_s["z-pos"] > 2500) & (glacier_s["z-pos"]<3000)].copy()
#             elif glacier_s["Glaciernam"].iloc[0] == "silvretta":
#                 plot_i = glacier_s[(glacier_s["z-pos"] > 2500)].copy()
#
#
#         if len(plot_i) >0:
#             print("before: ", plot_i.Glaciernam)
#             # Remove average value for this site for comparison:
#             #for i in range(0, len(plot_i)):
#             #   plot_i.iloc[i, 12] = plot_i.iloc[i, 12] - plot_i.iloc[:, 12].mean()
#             print("after : ", plot_i.Glaciernam)
#             s = axes[0].plot(plot_i.date1[plot_i.date1>19530000]/10000, \
#                                      plot_i[plot_i.date1>19530000].density - plot_i[plot_i.date1>19530000].density.mean() , \
#                                      linestyle = "dashed", zorder = 2, \
#                                      label = plot_i.Glaciernam.iloc[0])
#             s = axes[0].scatter(plot_i.date1[plot_i.date1>19530000]/10000,\
#                                       plot_i[plot_i.date1>19530000].density-plot_i[plot_i.date1>19530000].density.mean(), \
#                                         s=20, zorder = 1)
#
#
#     axes[0].yaxis.grid(color='gray', linestyle='dashed')
#     axes[0].xaxis.grid(color='gray', linestyle='dashed')
#     axes[0].set(xlabel="Year", ylabel="Density in kg/m3")
#     axes[0].set_xlim(1954, 2020)
#     axes[0].legend(loc = "upper left")
#     axes[0].set_title('Annual Density Measurements', size = 16)
#     axes[1].yaxis.grid(color='gray', linestyle='dashed')
#     axes[1].xaxis.grid(color='gray', linestyle='dashed')
#     axes[1].set(xlabel="Year", ylabel="Density in kg/m3")
#     axes[1].legend(loc = "upper left")
#     axes[1].set_title('Seasonal Winter Density Measurements', size = 16)
#     axes[1].set_xlim(1950, 2020)
#
#     axes[0].tick_params(axis='both', which='major', labelsize=16)
#     axes[1].tick_params(axis='both', which='major', labelsize=16)

def nearest(point, gpd2_pts, gpd2, geom_col='geometry', src_col='Place'):
    # find the nearest point
    nearest_point = nearest_points(point, gpd2_pts)[1]
    # return the corresponding value of the src_col of the nearest point
    value = gpd2[gpd2[geom_col] == nearest_point][src_col].get_values()[0]
    return value

if __name__ == '__main__':
    print("Use file_handling as module")