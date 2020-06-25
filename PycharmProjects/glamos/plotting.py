import os
import pandas as pd
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import contextily as ctx


def plot_map(gdf_list):
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

    for i in range(len(gdf_list)):
        #print(i, len(name_list), len(df_list))

       # gdf_list[i] = gdf_list[i].to_crs(epsg=3857)
    # Plot each glacier individually
        gdf_list[i].plot(ax=ax, markersize = 30, edgecolor="k" , zorder=(i+3))

#        if len(gdf_list[i].Stake)>1: #Why are some gdfs empty?
#            ax.annotate(gdf_list[i].Stake[1][:3], \
#                    xy=(gdf_list[i].geometry.x[1], gdf_list[i].geometry.y[1]),\
#            fontsize=15, xytext=(10, 10), textcoords="offset points")

    #ctx.add_basemap(ax, zoom=5, url=ctx.providers.Stamen.TonerLite)
    # Add basemap:
    world.geometry.boundary.plot(ax=ax, color=None, edgecolor='grey', zorder=1)
    water.plot(ax=ax, zorder=2, color='cornflowerblue' )
    sgi.plot(ax=ax, zorder=3, color = 'lightsteelblue')

    plt.show()

if __name__ == '__main__':
    print("Use file_handling as module")