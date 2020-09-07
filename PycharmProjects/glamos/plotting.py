import os
import pandas as pd
import geopandas as gpd
from scipy import spatial
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib
if os.name == 'nt':
    matplotlib.rc('font', family='Arial')
else:  # might need tweaking, must support black triangle for N arrow
    matplotlib.rc('font', family='DejaVu Sans')
import more_itertools as mit
from shapely.geometry import Point, LineString, Polygon
import numpy as np
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

   # df_pt[df_pt.nlargest(8, 'pt_size')]

    df_pt.plot(ax=ax, column="pt_size", \
               markersize = df_pt["pt_size_l"],\
               cmap='viridis', zorder = 5, alpha = 0.8, \
               legend="True", legend_kwds={'label': "Years of Observation"}, edgecolor = "black")
    # Add Labels:
    largest = df_pt.nlargest(8, 'pt_size') # get 8 longest time series
    label_point(largest["x-pos"], largest["y-pos"], largest["Glaciernam"], ax)

    # Add basemap:
    borders.boundary.plot(ax=ax, zorder=0, linewidth = 1)
    gdf_river_points.plot(ax=ax, zorder=1, color= 'lightsteelblue', markersize = 0.05 )
    lakes.plot(ax=ax, zorder=2, color='lightsteelblue', linewidth = 5)

    sgi.plot(ax=ax, zorder=3, color = 'lightslategrey')
    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax.transData,
                               100000, '100 km', 'lower center',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=1,
                               fontproperties=fontprops)

    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.add_artist(scalebar)
    plt.tight_layout()

    plt.title("Locations of Observed Glaciers")
    plt.show()
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y']+10000, str(point['val'].title()), \
                fontsize = 12, zorder = i + 5, weight= "bold", color = "black")

def plot_timeline(gdf_list_annual, gdf_list_intermediate, gdf_list_winter):
    """Plot timeline for all glaciers for all measurement types"""
    gl_net_info = pd.read_excel(\
         r"S:\glazio\projects\8003-VAW_GCOS_data_rescue\gl_net_info1.xlsx",
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
                ax1.broken_barh(tuple_list_annual, (y_loc*20,5) ,
                                alpha=0.8,
                                edgecolors='tab:green',
                                facecolors='tab:green', label = "Annual")
            else:
                ax1.broken_barh(tuple_list_annual, (y_loc*20,5) ,
                                alpha=0.8,
                                edgecolors='tab:green',
                                facecolors='tab:green')
        else:
            if ind == 32:
                ax2.broken_barh(tuple_list_annual, (y_loc * 20, 5),
                                alpha=0.8,
                                edgecolors='tab:green',
                                facecolors='tab:green', label="Annual")
            else:
                ax2.broken_barh(tuple_list_annual, (y_loc * 20, 5),
                                alpha=0.8,
                                edgecolors='tab:green',
                                facecolors='tab:green')



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
                ax1.broken_barh(tuple_list_annual, (y_loc*20+5,4) ,
                                facecolors='tab:orange',
                                alpha=0.8,
                                edgecolors='tab:orange',
                                label = "Intermediate")
            else:
                ax1.broken_barh(tuple_list_annual, (y_loc*20+5,4) ,
                                alpha=0.8,
                                edgecolors='tab:orange',
                                facecolors='tab:orange')
        else:
            if ind == 0:
                ax2.broken_barh(tuple_list_annual, (y_loc * 20 + 5, 4),
                                facecolors='tab:orange',
                                alpha=0.8,
                                edgecolors='tab:orange',
                                label="Intermediate")
            else:
                ax2.broken_barh(tuple_list_annual, (y_loc * 20 + 5, 4),
                                alpha=0.8,
                                edgecolors='tab:orange',
                                facecolors='tab:orange')

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
                ax1.broken_barh(tuple_list_annual, (y_loc*20+9,4) ,\
                                facecolors='tab:blue', alpha =0.5,
                                edgecolors = 'tab:blue',
                                label = "Winter")
            else:
                ax1.broken_barh(tuple_list_annual, (y_loc*20+9,4) ,
                                facecolors='tab:blue', alpha=0.5,
                                edgecolors = 'tab:blue')
        else:
            if ind == 28:
                ax2.broken_barh(tuple_list_annual, (y_loc * 20 + 9, 4), \
                                facecolors='tab:blue', alpha=0.5,
                                edgecolors='tab:blue',
                                label="Winter")
            else:
                ax2.broken_barh(tuple_list_annual, (y_loc * 20 + 9, 4),
                                facecolors='tab:blue', alpha=0.5,
                                edgecolors='tab:blue')

    ax1.set_ylim(0, 650)
    ax1.set_xlim(1885, 2019)
    ax1.set_xlabel('Year', fontsize=9)
    ax1.invert_yaxis()
    #ax1.yaxis.tick_right()
    ax1.set_yticks(range(0, len(name_list)*10, 20))
    ax1.set_xticks(range(1885, 2019, 10))
    #ax.set_yticklabels(real_name_list)
    ax1.yaxis.set_major_formatter(ticker.NullFormatter())
    ax1.yaxis.set_minor_locator(ticker.FixedLocator(range(10, len(name_list)*10+10, 20)))
    #ax1.yaxis.set_minor_formatter(ticker.FixedFormatter(real_name_list[:(round(len(name_list) / 2))]), horizontalalignment="left")
    ax1.tick_params(axis='y', which='major', labelsize=0)
    ax1.tick_params(axis='y', which='minor', labelsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.grid(True)
    ax2.legend(loc='lower right', fontsize = 9)
    ax1.set_yticklabels(real_name_list[:(round(len(name_list) / 2))],
                        horizontalalignment="left", minor = True)
    ax1.tick_params(axis='y', which = 'minor', direction='in', pad=-5)

    ax2.set_ylim(620, 1100)
    ax2.set_xlim(1885, 2019)
    ax2.set_xlabel('Year', fontsize=9)
    ax2.invert_yaxis()
    ax2.set_yticks(range(len(name_list*10)+10, len(name_list)*20, 20))
    ax2.set_xticks(range(1885, 2019, 10))
    ax2.yaxis.set_major_formatter(ticker.NullFormatter())
    ax2.yaxis.set_minor_locator(ticker.FixedLocator(range(len(name_list)*10+20, len(name_list)*20+20, 20)))
    #ax2.yaxis.set_minor_formatter(ticker.FixedFormatter(real_name_list[(round(len(name_list) / 2)):]))
    ax2.tick_params(axis='y', which='major', labelsize=0)
    ax2.tick_params(axis='y', which='minor', labelsize=9)
    ax2.set_yticklabels(real_name_list[(round(len(name_list) / 2)):],
                        horizontalalignment="left", minor = True)

    ax2.tick_params(axis='y', which = 'minor', direction='in', pad=-5, labelsize = 9)
    ax1.tick_params(axis="x", labelsize=9)
    ax2.grid(True)
    #fig.suptitle('Available Measurement Types and Periods for each Glacier')
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.88)
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

def plot_massbalance_mean(gdf_list, m_type, axes):
    """ Plot the annual mass balance time series of a few selected, particularly long series
    normalize them with median of time series (maybe chose a fixed time period for that?)
    to be able to compare them"""
    all_years_all_glaciers = []
    ###########  ANNUAL BALANCE ###########
    for i, glacier in enumerate(gdf_list):
        print("Glacier : ",glacier.Glaciername[0])
        # sort by time
        glacier = glacier.sort_values("date1")
        # s = axes[i].scatter(glacier.date1 / 10000, \
        #                     glacier["z-pos"],
        #                     label = glacier.Glaciername.iloc[0],
        #                     s = 20, zorder = 1)
        # axes[i].legend(loc="upper left")
        # print(" Glacier_s : ", glacier.Glaciername[0])
        # if glacier.Glaciername[0] =="gries":
        #     print("stooop")
        #     axes[0]=glacier.plot()
        #     for x, y, label in zip(glacier.geometry.x, glacier.geometry.y, glacier.Stake):
        #        axes[0].annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
        # For each glacier, pick first stake/first stakes that we want to look at:
        glacier_s = glacier
        plot_i = []
        glacier["Year"] = [int(str(glacier.date1.iloc[i])[:4]) for i, y in enumerate(glacier.date1)]

        if len(glacier_s) > 0:
            if glacier["Glaciername"].iloc[0] == "aletsch":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "ale-PO"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                # for more stakes per glacier, just add other Stake name here and rerun "get_timeseries"
            elif glacier["Glaciername"].iloc[0] == "allalin":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "all-AI"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "basodino":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "bas-2"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "clariden":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "cla-L"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                init_stakes = glacier[glacier.Stake == "cla-U"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "untgrindelwald":
                # define inital stake that starts timeseries
                init_stakes = glacier
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "gietro":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "gie-005"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "gries":
                # define inital stake that starts timeseries
                init_stakes = glacier[(glacier["z-pos"] > 3030)& (glacier["z-pos"]<3050)]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "silvretta":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "sil-BU"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)




    print("Checking things and plotting now : ")
    for one_glacier in all_years_all_glaciers:
        if m_type == "annual":
            ind = 0
        elif m_type == "winter":
            ind = 1
        print(m_type, ind)
        s = axes[ind].plot(one_glacier.date1 / 10000, \
                         one_glacier["Mass Balance"] \
                         - one_glacier[one_glacier.Year<2020]["Mass Balance"].mean(), \
                         linestyle="solid", alpha=0.2, \
                         zorder=2,
                         label=one_glacier.Glaciername.iloc[0])
        s = axes[ind].scatter(one_glacier.date1 / 10000, one_glacier["Mass Balance"] \
                            - one_glacier[one_glacier.Year<2020]["Mass Balance"].mean(), \
                            s=20, alpha=0.3,zorder=1)
    # Create Mean value for each year for all glaciers:
    avg = []
    for ye in range(1915, 2020, 1):
        print(ye)
        lst = ([glacier[glacier.Year == ye].loc[:, "Mass Balance"].values for glacier in all_years_all_glaciers])
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
    axes[ind].set(xlabel="Year", ylabel="Mass Balance in mm.w.e.")
    for label in (axes[ind].get_xticklabels() + axes[ind].get_yticklabels()):
        label.set_fontsize(16)
        #axes[0].set_xlim(1954, 2020)
    axes[0].legend(["Aletsch P3 3350m",
                    "Allalin P1 2850 m",
                    "Basodino P2 2680",
                      "Clariden Lower 2700 m",
                      "Clariden Upper 2900 m",
                    "Gietro P5 3070 m ",
                      "Gries P5 3030 m",
                      "Silvretta BU 2750",
                      "Mean Value"],loc="upper left")
    axes[1].legend(["Aletsch P3 3350m",
                    "Allalin P1 2850 m",
                    "Basodino P2 2680",
                      "Clariden Lower 2700 m",
                      "Clariden Upper 2900 m",
                      "Gries P5 3030 m",
                      "Silvretta BU 2750",
                      "Mean Value"],loc="upper left")
    axes[0].set_title('Annual Point Mass Balance', size=16)
    axes[ind].tick_params(axis='both', which='major', labelsize=16)
    axes[1].set_title('Winter Point Mass Balance', size=16)


def plot_massbalance_individual(gdf_list, m_type, fig, axes):
    """ Plot the annual mass balance time series of a few selected, particularly long series
    normalize them with median of time series (maybe chose a fixed time period for that?)
    to be able to compare them"""
    all_years_all_glaciers = []
    name_list =["Aletsch P3 3350m",
                    "Allalin P1 2850 m",
                    "Basodino P2 2680",
                      "Clariden Lower 2700 m",
                      "Clariden Upper 2900 m",
                    "Gietro P5 3070 m ",
                      "Gries P5 3030 m",
                      "Silvretta BU 2750"]
    ###########  ANNUAL BALANCE ###########
    for i, glacier in enumerate(gdf_list):
        print("Glacier : ",glacier.Glaciername[0])
        # sort by time
        glacier = glacier.sort_values("date1")
        # For each glacier, pick first stake/first stakes that we want to look at:
        glacier_s = glacier
        plot_i = []
        glacier["Year"] = [int(str(glacier.date1.iloc[i])[:4]) for i, y in enumerate(glacier.date1)]

        if len(glacier_s) > 0:
            if glacier["Glaciername"].iloc[0] == "aletsch":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "ale-PO"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                indcol = 0
                indr = 0
            elif glacier["Glaciername"].iloc[0] == "basodino":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "bas-2"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                # for more stakes per glacier, just add other Stake name here and rerun "get_timeseries"
            elif glacier["Glaciername"].iloc[0] == "clariden":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "cla-L"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
                init_stakes = glacier[glacier.Stake == "cla-U"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "untgrindelwald":
                # define inital stake that starts timeseries
                init_stakes = glacier
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "gries":
                # define inital stake that starts timeseries
                init_stakes = glacier[(glacier["z-pos"] > 3030)& (glacier["z-pos"]<3050)]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "silvretta":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "sil-BU"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "allalin":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "all-AI"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
            elif glacier["Glaciername"].iloc[0] == "gietro":
                # define inital stake that starts timeseries
                init_stakes = glacier[glacier.Stake == "gie-005"]
                all_years_all_glaciers = get_timeseries(init_stakes, glacier, all_years_all_glaciers)
    print("Checking things and plotting now : ")
    for ll,one_glacier in enumerate(all_years_all_glaciers):
        if ll == 0:
            indcol = 0
            indr = 0
        elif ll == 1:
            indcol = 1
            indr = 0
        elif ll == 2:
            indcol = 0
            indr = 1
        elif ll == 3:
            indcol = 1
            indr = 1
        elif ll == 4:
            indcol = 0
            indr = 2
        elif ll == 5:
            indcol = 1
            indr = 2
        elif ll == 6:
            indcol = 0
            indr = 3
        elif ll == 7:
            indcol = 1
            indr = 3

        print(m_type, indcol, indr)
        s = axes[indr, indcol].plot(one_glacier.date1 / 10000, \
                         one_glacier["Mass Balance"] \
                         , \
                         linestyle="solid", alpha=0.5, \
                         zorder=2,
                         label=one_glacier.Glaciername.iloc[0])
        s = axes[indr,indcol].scatter(one_glacier.date1 / 10000, one_glacier["Mass Balance"] \
                            , \
                            s=20, alpha=0.6,zorder=1)
        axes[indr, indcol].set_title(name_list[ll])
        axes[indr, indcol].yaxis.grid(color='gray', linestyle='dashed')
        axes[indr, indcol].xaxis.grid(color='gray', linestyle='dashed')
        #axes[indr, indcol].set(xlabel="Year", ylabel="mm.w.e.")
        #axes[indr, indcol].tick_params(axis='both', which='major', labelsize=16)

    # Create Mean value for each year for all glaciers:
    # avg = []
    # for ye in range(1915, 2020, 1):
    #     print(ye)
    #     lst = ([glacier[glacier.Year == ye].loc[:, "Mass Balance"].values for glacier in all_years_all_glaciers])
    #     i = 0
    #     sum = 0
    #     for l in lst:
    #         if l.size:
    #             if not np.isnan(l).any():
    #                 print(l)
    #                 sum = sum + l[0]
    #                 i = i + 1
    #                 print(sum, i)
    #             else:
    #                 print(l)
    #                 print(sum, i)
    #     try:
    #         app = sum / i
    #     except ZeroDivisionError:
    #         app = np.nan
    #     avg.append(app)
    # print(avg)
    # avg_li = []
    # # Flatten List
    # for a in avg:
    #     print(a)
    #     try:
    #         avg_li.append(a[0])
    #     except TypeError:
    #         avg_li.append(np.nan)
    #     except  IndexError:
    #         avg_li.append(a)
    #
    # # Remove mean from all values:
    # avg_me = avg_li - np.nanmean(avg_li)
    # # Compute moving average:
    # mov_avg = pd.Series(avg_me).rolling(5).mean()
    # mov_avg = mov_avg.fillna(method='ffill')
    # if m_type == "winter":
    #     print("stop")
    # #axes[3,1].plot(range(1915, 2020, 1), mov_avg, linestyle="solid", alpha=0.5, \
    # #                    zorder=2)
    # #axes[3,1].scatter(range(1915, 2020, 1), mov_avg,
    # #                s=20, alpha=0.9,zorder=1)
    # #axes[3,1].set_title(name_list[-1])
    #
    # axes[3,1].yaxis.grid(color='gray', linestyle='dashed')
    # axes[3,1].xaxis.grid(color='gray', linestyle='dashed')
    # axes[3,1].set(xlabel="Year", ylabel="Mass Balance in mm.w.e.")
    # axes[3,1].tick_params(axis='both', which='major', labelsize=16)
    plt.sca(axes[indr,indcol])  # set the current axes instance to the top left
    plt.rcParams['axes.labelsize'] = 18
    plt.tight_layout()

    fig.text(0.5, 0.001, 'Year', ha='center')
    fig.text(0.001, 0.5, 'Mass Balance m m.w.e', va='center', rotation='vertical')

    axes[0,0].legend(['Annual','Winter'])

def get_timeseries(init_stakes, glacier, all_years_all_glaciers):
    """ Create TimeSeries of points for each year starting vom Location x,y at year t in init_i
    For each year find the closest measurements that is less than 250 Meters away. If nothing is found,
    append nothing. Write this result for all years to GeoPadas Dataframe "all_years" and appends this
    Dataframe to the list "all_years_all_glaciers" """
    try:
        init_i = init_stakes[init_stakes.date1 == min([i for i in init_stakes.date1])]
    except ValueError:
        return all_years_all_glaciers
    all_years = init_i
    for year in range(int(str(init_i.date1.iloc[0])[:4])+1, 2019, 1):
        this_year = glacier[glacier.Year == year]
        if len(this_year) > 0:
            # find index of point in dem that is closest to our point pt for which we need
            # add the z-component:
            A = np.column_stack((this_year["x-pos"].to_list(), this_year["y-pos"].to_list()))
            pt = [init_i["x-pos"].iloc[0], init_i["y-pos"].iloc[0]]
            # distance and index of clostest point:
            distance, ind = spatial.KDTree(A).query(pt)
            # Find closest point to init_i
            if distance < 600:
                next_year = this_year.iloc[ind]
                print(next_year)
                all_years = all_years.append(next_year)
    everything = all_years_all_glaciers.append(all_years) # PD Dataframe with measurements for each year
    return all_years_all_glaciers


if __name__ == '__main__':
    print("Use file_handling as module")