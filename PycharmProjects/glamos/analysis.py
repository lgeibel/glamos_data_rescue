import pandas as pd
import geopandas as gpd
import os
import datetime
from file_handling import check_date
from geopandas.tools import sjoin
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, metrics
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence as olsi
import seaborn as sns



from raster2xyz.raster2xyz import Raster2xyz
import csv
import matplotlib.pyplot as plt
import itertools
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from scipy import spatial

def density_analysis(gdf_list, m_type):
    """
    Read out all density measurements (with density ID 2)
    available and plot them against doy and and elevation?
    :param gdf_list:
    :return:
    """
    # Put all entries with Density ID 2 into a new gdf_list
    print("test me")

    for gdf in gdf_list:
        if (len(gdf[gdf["dens-ID"] == 2]) > 0) and 'rdf' not in locals():
            rdf = gdf[gdf["dens-ID"] == 2]
            print(rdf)
        elif 'rdf' in locals():
            rdf = rdf.append(gdf[gdf["dens-ID"] == 2])
            print(rdf)

    return rdf

def plot_densities(rdf_all):
    """ Scatter plot all densities for which ID 2 was given
    (that were actually measured). Perform Multivariable Regression
    with Elevation and DOY to obtain a linear model
    :param: rdf_all: List of all glaciers with entries that were actually measured
    :return: model: statsmodel linear model from 2-D regression"""
    # get doy:
    doy_1 = []
    for i in range(len(rdf_all.date0)):
        doy_1.append(datetime.datetime.strptime( \
            str(int(rdf_all.date1[i])), '%Y%m%d').timetuple().tm_yday)
    rdf_all["doy"] = doy_1
    df1 = pd.DataFrame(rdf_all.drop(columns='geometry'))

    # exclude None Values (now corrected in Version 5)
    res = [i for i in range(len(df1.density.to_list())) if df1.density.to_list()[i] == None]
    bad_df = df1.index.isin(res)
    df1 = df1[~bad_df]
    df1['density'] = df1['density'].astype(int)

    df1 = df1[df1["Mass Balan"] > 0] # only use positive balances
    df1 = df1[df1.density != 400]
    df1 = df1[df1.density != 450]
    df1 = df1[df1.density != 500]
    df1 =  df1[df1.density != 550]
    df1 = df1[df1. density != 600] # Ignore standard values
    df1 = df1[df1.density < 900]
    df1 = df1[df1["z-pos"] < 3800] # exclude
            # Morteratsch value (by now the ID is changed so no need to worry about it anymore
    df1 = df1[df1["z-pos"]>1000] # Ignore the faulty Plaine Morte file
    df1 = df1[~df1["raw_balanc"].isnull()]
    df1['raw_balanc'] = df1['raw_balanc'].astype(int)

    # Interpolation for measurements after spring
    #  (DOY >= 160): Use Doy and Elevation
    X = df1[['doy',
            'z-pos']]  # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
    Y = df1['density']

     # Use statsmodel as its much cooler
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X)
    print_model = model.summary()
    # # Calculate Studentized Residuals:
    # studentized_residuals = olsi(model).resid_studentized
    # # calculate Cook distance/leverage
    # cook_dist = dict(olsi(model).cooks_distance[0])
    # cook_df = pd.DataFrame.from_dict(cook_dist, orient="index")
    # df1_corr = df1_summer[abs(studentized_residuals) < 2.7]
    # df1_corr = df1_corr[cook_df < 0.022]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
    plt.sca(axes[0])  # set the current axes instance to the top left
    plt.rcParams['axes.labelsize'] = 18
    s = axes[0].scatter(df1.doy, df1.density, c=df1["z-pos"], s=20)
    cb = plt.colorbar(s)
    cb.set_label('Elevation in m')
    cb.ax.tick_params(labelsize=16)
    axes[0].set_axisbelow(True)
    axes[0].yaxis.grid(color='gray', linestyle='dashed')
    axes[0].xaxis.grid(color='gray', linestyle='dashed')
    axes[0].set(xlabel="Day of Year", ylabel="Density in kg/m3")
    axes[0].tick_params(axis='both', which='major', labelsize=16)

    plt.sca(axes[1])  # set the current axes instance to the top left
    s = axes[1].scatter(df1["z-pos"], df1.density, c=df1.doy, s=20)
    cb = plt.colorbar(s)
    cb.ax.tick_params(labelsize=16)
    cb.set_label('Day of Year')
    axes[1].yaxis.grid(color='gray', linestyle='dashed')
    axes[1].xaxis.grid(color='gray', linestyle='dashed')
    axes[1].set(xlabel="Elevation in m", ylabel="Density in kg/m3")
    axes[1].tick_params(axis='both', which='major', labelsize=16)

    plt.sca(axes[2])  # set the current axes instance to the top left
    s = axes[2].scatter(df1.raw_balanc, df1.density, c=df1.doy, s=20)
    cb = plt.colorbar(s)
    cb.set_label('Day of Year')
    cb.ax.tick_params(labelsize=16)
    axes[2].yaxis.grid(color='gray', linestyle='dashed')
    axes[2].xaxis.grid(color='gray', linestyle='dashed')
    axes[2].set(xlabel="Snow Depth in m", ylabel="Density in kg/m3")
    axes[2].tick_params(axis='both', which='major', labelsize=16)
    plt.subplots_adjust(wspace=0.3, hspace=1)


    # Look at time series:
    # Values from same glacier and similar eleveation:
    glacier_wise = [v for k, v in df1.groupby('Glaciernam') if len(v) > 7]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 15))
    ###########  ANNUAL BALANCE ###########
    for glacier in glacier_wise:
        #print("Glacier : ",glacier)
        # sort by time
        glacier = glacier.sort_values("date1")
        # Remove summer measurements:
        glacier_s = glacier[glacier ["date0"] != 0]
        #print(" Glacier_s : ", glacier_s.Glaciernam)
        plot_i = []
        if len(glacier_s) > 0:
            if glacier_s["Glaciernam"].iloc[0] == "albigna":
                plot_i = glacier_s[glacier_s["Stake"] == "alb-T"].copy()
            elif glacier_s["Glaciernam"].iloc[0] == "aletsch":
                plot_i = glacier_s[glacier_s["z-pos"] > 3000].copy()
            elif glacier_s["Glaciernam"].iloc[0] == "clariden":
                plot_i = glacier_s[(glacier_s["z-pos"] > 2800) & (glacier_s["z-pos"] < 2930)].copy()
            elif glacier_s["Glaciernam"].iloc[0] == "forno":
                plot_i = glacier_s[glacier_s["z-pos"] > 3000].copy()
            elif glacier_s["Glaciernam"].iloc[0] == "plainemorte":
                plot_i = glacier_s.copy()
            elif glacier_s["Glaciernam"].iloc[0] == "rhone":
                plot_i = glacier_s[(glacier_s["z-pos"] > 2500) & (glacier_s["z-pos"]<3000)].copy()
            elif glacier_s["Glaciernam"].iloc[0] == "silvretta":
                plot_i = glacier_s[(glacier_s["z-pos"] > 2500)].copy()


        if len(plot_i) >0:
            print("before: ", plot_i.Glaciernam)
            # Remove average value for this site for comparison:
            #for i in range(0, len(plot_i)):
            #   plot_i.iloc[i, 12] = plot_i.iloc[i, 12] - plot_i.iloc[:, 12].mean()
            print("after : ", plot_i.Glaciernam)
            s = axes[0].plot(plot_i.date1[plot_i.date1>19530000]/10000, \
                                     plot_i[plot_i.date1>19530000].density - plot_i[plot_i.date1>19530000].density.mean() , \
                                     linestyle = "dashed", zorder = 2, \
                                     label = plot_i.Glaciernam.iloc[0])
            s = axes[0].scatter(plot_i.date1[plot_i.date1>19530000]/10000,\
                                      plot_i[plot_i.date1>19530000].density-plot_i[plot_i.date1>19530000].density.mean(), \
                                        s=20, zorder = 1)


    ###########  WINTER BALANCE #############
    for glacier in glacier_wise:
        # sort by elevation
        glacier = glacier.sort_values("date1").copy()
        # Remove summer measurements:
        glacier = glacier[(glacier["period"] < 200) | (glacier["date0"] == 0)].copy()
        plot_i_s=[]
        if len(glacier) > 0:
            if glacier["Glaciernam"].iloc[0] == "albigna":
                plot_i_s = glacier[glacier["Stake"] == "alb-T"].copy()
            elif glacier["Glaciernam"].iloc[0] == "aletsch":
                plot_i_s = glacier[glacier["z-pos"] > 3000].copy()
            elif glacier["Glaciernam"].iloc[0] == "clariden":
                plot_i_s = glacier[(glacier["z-pos"] > 2600)& (glacier["z-pos"]<2900)].copy()
            elif glacier["Glaciernam"].iloc[0] == "forno":
                plot_i_s = glacier[glacier["z-pos"] > 3300].copy()
            elif glacier["Glaciernam"].iloc[0] == "plainemorte":
                plot_i_s = glacier.copy()
            elif glacier["Glaciernam"].iloc[0] == "rhone":
                plot_i_s = glacier[(glacier["z-pos"] > 2000) & (glacier["z-pos"]<3000)].copy()
            elif glacier["Glaciernam"].iloc[0] == "silvretta":
                plot_i_s = glacier[(glacier["z-pos"] > 2500) & (glacier["z-pos"]<2600)].copy()


        if len(plot_i_s) >0:
            # Remove average value for this site for comparison:
            #for i in range(0, len(plot_i_s)):
            #    plot_i_s.iloc[i, 12] = plot_i_s.iloc[i, 12] - plot_i_s.iloc[:, 12].mean()
            s = axes[1].plot(plot_i_s.date1/10000, \
                                 plot_i_s.density - plot_i_s.density.mean(), \
                                 linestyle = "dashed", \
                                 zorder = 2,
                                 label = plot_i_s.Glaciernam.iloc[0])
            s = axes[1].scatter(plot_i_s.date1/10000, plot_i_s.density - plot_i_s.density.mean(),\
                                    s=20, zorder = 1)


    axes[0].yaxis.grid(color='gray', linestyle='dashed')
    axes[0].xaxis.grid(color='gray', linestyle='dashed')
    axes[0].set(xlabel="Year", ylabel="Density in kg/m3")
    axes[0].set_xlim(1954, 2020)
    axes[0].legend(loc = "upper left")
    axes[0].set_title('Annual Density Measurements', size = 16)
    axes[1].yaxis.grid(color='gray', linestyle='dashed')
    axes[1].xaxis.grid(color='gray', linestyle='dashed')
    axes[1].set(xlabel="Year", ylabel="Density in kg/m3")
    axes[1].legend(loc = "upper left")
    axes[1].set_title('Seasonal Winter Density Measurements', size = 16)
    axes[1].set_xlim(1950, 2020)

    axes[0].tick_params(axis='both', which='major', labelsize=16)
    axes[1].tick_params(axis='both', which='major', labelsize=16)

    return model, predictions


    print("What else?")


def interpolate_densities(model, gdf_list, m_type):
    """ Replace all NaN densities with Value from Interpolation
    Return: gdf_list"""
    # Add DOY as column:
    for gdf in gdf_list:
        gdf = check_date(gdf, m_type)
        if gdf.Glaciername[0] == "diablerets":
            print("Check me")
            print(gdf.period)
        doy_1 = []
        for i in range(len(gdf.date0)):
            doy_1.append(datetime.datetime.strptime( \
                str(int(gdf.date1[i])), '%Y%m%d').timetuple().tm_yday)
        gdf["doy"] = doy_1
        print("Start interpolation:", gdf.Glaciername[0])
        # Find the following entries: (to be interpolated tbi)
        # Density = NaN and ID not 5
        # Density = 450, 500, 550, 600
        tbi = gdf[(gdf["density"].isnull() & (gdf["dens-ID"] != 5)) \
                  | (gdf.density == "500") \
                  | (gdf.density == "550")].copy()
        if len(tbi)>0:
            print("Actually Start interpolation:", gdf.Glaciername[0])
            # Interpolate Density
            X = tbi[['doy', 'z-pos']]
            # Get Predictions for all points without known Density
            predictions = model.predict(X)
            # Replace Density column in temporary DF
            tbi['density'] = round(predictions)
            # replace ID
            tbi.loc[:, 'dens-ID'] = 6
            # Replace w.e.:
            tbi.loc[:, "Mass Balance"] = round(tbi.loc[:, "density"] * tbi.loc[:, "raw_balance"] / 100)

            for ind in tbi.index:
                # Re-Write gdf
                gdf.loc[ind, "density"] = tbi.loc[ind, "density"]
                gdf.loc[ind, "dens-ID"] = tbi.loc[ind, "dens-ID"]
                gdf.loc[ind, "Mass Balance"] = tbi.loc[ind, "Mass Balance"]

    return gdf_list



if __name__ == '__main__':
    print("Use file_handling as module")