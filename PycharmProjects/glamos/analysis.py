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
from plotting import get_timeseries



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

def plot_density_mean(rdf_all):

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
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 15))


    all_years_all_glaciers = []
    for m_type in ["annual", "winter"]:
        if m_type == "annual":
            rdf_summer = rdf_all[rdf_all.doy > 180]
            grouped = rdf_summer.groupby('Glaciernam')
            gdf_list = [group for _, group in grouped]
        else:
            rdf_winter = rdf_all[rdf_all.doy < 180]
            grouped = rdf_winter.groupby('Glaciernam')
            gdf_list = [group for _, group in grouped]


 #       elif glacier_s["Glaciernam"].iloc[0] == "forno":
 #           plot_i = glacier_s[glacier_s["z-pos"] > 3000].copy()
 #       elif glacier_s["Glaciernam"].iloc[0] == "plainemorte":
 #           plot_i = glacier_s.copy()
 #       elif glacier_s["Glaciernam"].iloc[0] == "rhone":
 #           plot_i = glacier_s[(glacier_s["z-pos"] > 2500) & (glacier_s["z-pos"] < 3000)].copy()




        ###########  ANNUAL BALANCE ###########
        for i, glacier in enumerate(gdf_list):
            print("Glacier : ",glacier.Glaciernam.iloc[0])
            # sort by time
            glacier = glacier.sort_values("date1")
            # For each glacier, pick first stake/first stakes that we want to look at:
            glacier_s = glacier
            plot_i = []
            glacier["Year"] = [int(str(glacier.date1.iloc[i])[:4]) for i, y in enumerate(glacier.date1)]

            if len(glacier_s) > 0:
                if glacier["Glaciernam"].iloc[0] == "aletsch":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "ale-PO"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                    # for more stakes per glacier, just add other Stake name here and rerun "get_timeseries"
                if glacier["Glaciernam"].iloc[0] == "allalin":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "all-AVII"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                    # for more stakes per glacier, just add other Stake name here and rerun "get_timeseries"
                elif glacier["Glaciernam"].iloc[0] == "albigna":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "alb-T"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                    init_stakes = glacier[glacier.Stake == "alb-O"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "basodino":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "bas-8"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "clariden":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "cla-L"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                    init_stakes = glacier[glacier.Stake == "cla-U"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "findelen":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "fin-1010"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "forno":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "for-D"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "gries":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "grs-111"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "gietro":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "gie-001"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "limmern":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "lim-3"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "silvretta":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "sil-BU"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "rhone":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "rho-0701"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                    init_stakes = glacier[glacier.Stake == "rho-0602"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "pizol":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "pzl-PS"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "plainemorte":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "plm-s2"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                elif glacier["Glaciernam"].iloc[0] == "stanna":
                    # define inital stake that starts timeseries
                    init_stakes = glacier[glacier.Stake == "sta-s30"]
                    all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)


        print("Checking things and plotting now : ")
        for one_glacier in all_years_all_glaciers:
            if m_type == "annual":
                ind = 0
            elif m_type == "winter":
                ind = 1
            print(m_type, ind)
            one_glacier["density"] = pd.to_numeric(one_glacier["density"])
            # What happens if one_glacier == "None"?
            s = axes[ind].plot(one_glacier.date1 / 10000, \
                             one_glacier["density"] \
                             - one_glacier[one_glacier.Year<2020]["density"].mean(), \
                             linestyle="solid", alpha=0.2, \
                             zorder=2,
                             label=one_glacier.Glaciernam.iloc[0])
            s = axes[ind].scatter(one_glacier.date1 / 10000, one_glacier["density"]\
                                - one_glacier[one_glacier.Year<2020]["density"].mean(), \
                                s=20, alpha=0.3,zorder=1)
        # Create Mean value for each year for all glaciers:
        avg = []
        for ye in range(1915, 2020, 1):
            print(ye)
            lst = ([glacier[glacier.Year == ye].loc[:, "density"].values for glacier in all_years_all_glaciers])
            i = 0
            sum = 0
            for l in lst:
                if l.size:
                    if not np.isnan(l).any():
                        print(l)
                        sum = sum + l[0]
                        i = i + 1
                        print(sum, i)
                    else:
                        print(l)
                        print(sum, i)
            try:
                app = sum / i
            except ZeroDivisionError:
                app = np.nan
            avg.append(app)
        print(avg)
        avg_li = []
        # Flatten List
        for a in avg:
            print(a)
            try:
                avg_li.append(a[0])
            except TypeError:
                avg_li.append(np.nan)
            except  IndexError:
                avg_li.append(a)

        # Remove mean from all values:
        avg_me = avg_li - np.nanmean(avg_li)
        # Compute moving average:
        mov_avg = pd.Series(avg_me).rolling(5).mean()
        mov_avg = mov_avg.fillna(method='ffill')
        if m_type == "winter":
            print("stop")
        axes[ind].plot(range(1915, 2020, 1), mov_avg, color = "firebrick", linewidth = 2)
        plt.sca(axes[ind])  # set the current axes instance to the top left
        plt.rcParams['axes.labelsize'] = 18

        axes[ind].yaxis.grid(color='gray', linestyle='dashed')
        axes[ind].xaxis.grid(color='gray', linestyle='dashed')
        axes[ind].set(xlabel="Year", ylabel="Density in kg/m3")
        for label in (axes[ind].get_xticklabels() + axes[ind].get_yticklabels()):
            label.set_fontsize(16)
            #axes[0].set_xlim(1954, 2020)
        axes[0].legend(loc="upper left")
        axes[1].legend(loc="upper left")
        axes[0].set_title('Mean of Annual Density Measurements', size=16)
        axes[ind].tick_params(axis='both', which='major', labelsize=16)
        axes[1].set_title('Mean of Winter Density Measuremens', size=16)






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