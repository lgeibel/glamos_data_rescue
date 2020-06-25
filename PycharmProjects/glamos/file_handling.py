import pandas as pd
import geopandas as gpd
import os
import datetime
from geopandas.tools import sjoin
import csv
import matplotlib.pyplot as plt
import itertools

def import_v0(filename, filepath, print_preview=True):
    '''Imports .dat file from version0 into df dataframe

    Keyword arguments:
    filename -- name of .dat file
    filepath -- path to folder containing file
    print_preview -- if True (default), preview of file is printed

    Returns: df

    '''
    file = os.path.join(filepath, filename)
    print(filename)
    datContent = [i.strip().split() for i in open(file).readlines()]

    # write it as a new CSV file
    with open("test.dat", "w") as f:
        writer = csv.writer(f)
        writer.writerows(datContent)

    df = pd.read_csv("test.dat", header=4, \
                                names = ['Stake', 'date0', \
                                         'time0', 'date1', 'time1', \
                                         'period', 'date-ID', 'x-pos', \
                                         'y-pos', 'z-pos', 'position-ID', \
                                         'raw_balance', 'density', \
                                         'dens-ID', 'Mass Balance', \
                                         'quality-I', 'type-ID', 'observer/source'])
    if print_preview:
        print(df.head(5))

    # Add column with glacier name
    glaciername = filename.split("_", 1)[0]
    glacier_list = []
    for i in range(len(df.Stake)):
        glacier_list.append(glaciername)
    df["Glaciername"] = glacier_list
    return df

def import_excel(filename, filepath, print_preview=True):
    '''Imports .xls file into df dataframe

    Keyword arguments:
    filename -- name of .xls file
    filepath -- path to folder containing file
    print_preview -- if True (default), preview of file is printed

    Returns: df

    '''
    file = os.path.join(filepath, filename)
    print(filename)
    df = pd.read_excel(file, header=3, \
                       names=['Stake', 'date0', \
                              'time0', 'date1', 'time1', \
                              'period', 'date-ID', 'x-pos', \
                              'y-pos', 'z-pos', 'position-ID', \
                              'raw_balance', 'density', \
                              'dens-ID', 'Mass Balance', \
                              'quality-I', 'type-ID', 'observer/source']
                       )


    # Add column with glacier name
    glaciername = filename.split("_", 1)[0]
    glacier_list = []
    for i in range(len(df.Stake)):
        glacier_list.append(glaciername)
    df["Glaciername"] = glacier_list

    # Add column with Glacier ID - Read this from .xls file:
    gl_net_info = pd.read_excel(\
         r"Z:\glazio\projects\8003-VAW_GCOS_data_rescue\gl_net_info.xlsx",
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
    ids = [gl_net_info["inv-no"][i].replace("/", "-") \
            for i in range(len(gl_net_info["inv-no"]))]
    gk_no= [gl_net_info["gk-no"][i] \
            for i in range(len(gl_net_info["gk-no"]))]
    lk25 = [gl_net_info["lk25"][i] \
            for i in range(len(gl_net_info["lk25"]))]
    # get index of this glacier
    try:
         glacier_loc = [gl_net_info["short"][i][1:] \
                for i in range(len(gl_net_info["inv-no"]))].index(glaciername)
         gk_loc =  gk_no[glacier_loc]
         ids_loc = ids[glacier_loc]
         lk_loc = lk25[glacier_loc]

         # Append to dataframe as extra columns
         gk_list  =[]
         ids_list = []
         lk25_list = []
         for i in range(len(df.Stake)):
             gk_list.append(gk_loc)
             ids_list.append(ids_loc)
             lk25_list.append(lk_loc)

         df["gk-no"] = gk_list
         df["inv-no"] = ids_list
         df["lk25"] = lk25_list
    except ValueError:
         print(glaciername, " IS NOT IN LIST")

    if print_preview:
        print(df.head(5))
    return df
def pd_to_gpd(df_list):
    ''' Converts list of pandas dataframes to list of
    geodataframes and set crs

    :param df_list: list of dataframes
    :return: gdf_list: list of geodataframes
    '''
    gdf_list = [gpd.GeoDataFrame() for _ in range(len(df_list))]
    for i in range(len(df_list)):
        # Replace commas with dots and convert to floats:
        if type(df_list[i]['x-pos']) == str:
            df_list[i]['x-pos'] = [x.replace(',', '.') for x in df_list[i]['x-pos']]
            df_list[i]['x-pos'] = df_list[i]['x-pos'].astype(float)
            df_list[i]['y-pos'] = [x.replace(',', '.') for x in df_list[i]['y-pos']]
            df_list[i]['y-pos'] = df_list[i]['y-pos'].astype(float)
       # print(df_list[i]['Stake'],df_list[i]['date0'],df_list[i]['x-pos'], df_list[i]['y-pos'])

        gdf_list[i] = gpd.GeoDataFrame(df_list[i],\
                geometry=gpd.points_from_xy(df_list[i]['x-pos'], df_list[i]['y-pos']))
        gdf_list[i].crs = {'init': 'epsg:21781'}
    return gdf_list

def compare_v0(gdf_list, gdf_list_0):
    '''Compares entries in current version with
    version 0'''
    print(gdf_list, gdf_list_0)
    print("Glaciers in Version0: ", len(gdf_list_0), " Glaciers in Version4", len(gdf_list))
    entries=0
    for glacier in gdf_list_0:
        entries = entries + len(glacier)

    print("Entries in Version 0 ", entries)
    entries = 0
    for glacier in gdf_list:
        entries = entries + len(glacier)

    print("Entries in Version 4 ", entries)
def check_date(gdf, m_type):
    '''Checks for consistency of dates, period
    and checks if IDs are correct/corrects them

        Keyword arguments:
        gdf -- geodataframes
        m_type: str: type of measurements ("annual", "intermediate", "winter")
        '''

    print(gdf.Stake[0])
    # Detect Generic Dates (0915, 1001, ) --> set ID correctly
    # find average date of all measured values for this glacier
    # to use as generic date
    # for years without known dates
    doy_1 = []
    problematic = []
    for i in range(len(gdf.date0)):
        if m_type == " annual":
            if str(gdf.date1[i])[4:] != "1001" and \
                    str(gdf.date1[i])[4:] != "0000" and \
                    int(gdf.date1[i]) > 2100:
                # Calculate Average Starting day for measurements -
                # get doy for the measurement
                try:
                    doy_1.append(datetime.datetime.strptime( \
                    str(int(gdf.date1[i])), '%Y%m%d').timetuple().tm_yday)
                except ValueError:
                    problematic.append(str(gdf.Stake[i])+ " :  "+str(gdf.date1[i]))
            if len(problematic)>0:
                print("There is a problem with those entries: ", problematic)

            # calc average doy
            avg_doy1 = sum(doy_1) / len(doy_1)
            # convert doy to day of year as string (mmdd):
            avg_day1 = (datetime.datetime(1000, 1, 1) \
                    + datetime.timedelta(avg_doy1- 1))\
                       .strftime('%Y%m%d')[4:]


        # Compare Period, correct if necessary
    for i in range(len(gdf.date0)):
        #Convert to Datetime Object
        if gdf.date0[i]==0:
            #Depth Probe with unkown reference horizon/date:
            # Set IDs accordingly:
            if m_type == "winter" or m_type =="intermediate":
                gdf.loc[[i], ['date-ID']] = 3
                gdf.loc[[i], ['type-ID']] = 2
                gdf.loc[[i], ['period']] = 0
        else:
            try:
                if str(gdf.date0[i])[4:] == "1001" or \
                        str(gdf.date0[i])[4:] == "0000" or \
                        1840 < int(gdf.date0[i]) < 2100:
                    # start date unkown
                    if str(gdf.date1[i])[4:] == "1001"\
                            or str(gdf.date1[i])[4:] == "0000" or \
                            1840 < int(gdf.date1[i]) < 2100:
                        # End date and start date is estimated\unkown:
                        # set ID 9:
                        # set generic dates 1001
                        if m_type == "annual":
                            gdf.loc[[i], ['date-ID']] = 9
                            print("Replacing this one: : ", \
                                  i, gdf.date0[i], gdf.date1[i], \
                                  gdf.time0[i], gdf.time1[i] , " with ", int(str(gdf.date0[i])[:4]+avg_day1), \
                                  gdf.Stake[i])
                            gdf.loc[[i], ['date0']] =int(str(gdf.date0[i])[:4]+avg_day1)
                            gdf.loc[[i], ['date1']] = int(str(gdf.date1[i])[:4]+avg_day1)

                    else:
                        # Start day unknown, end date known
                        # set ID 3
                        # set generic date 1001
                        if m_type == "annual":
                            gdf.loc[[i], ['date-ID']] = 3
                            print("Replacing this one: : ", \
                                  i, gdf.date0[i], gdf.date1[i], \
                                  gdf.time0[i], gdf.time1[i] , " with ", int(str(gdf.date0[i])[:4]+avg_day1), \
                                  gdf.Stake[i])
                            gdf.loc[[i], ['date0']] = int(str(gdf.date0[i])[:4]+avg_day1)

                else:
                    # start date known
                    if str(gdf.date1[i])[4:] == "1001" or \
                        str(gdf.date1[i])[4:] == "0000" or \
                            1840 < int(gdf.date1[i])< 2100:
                        # End date unknown and start date kown:
                        # set ID 2:
                        if m_type == "annual":
                            gdf.loc[[i], ['date-ID']] = 2
                            print("Replacing this one: : ", \
                                  i, gdf.date0[i], gdf.date1[i], \
                                  gdf.time0[i], gdf.time1[i] , " with ", int(str(gdf.date1[i])[:4]+avg_day1), \
                                  gdf.Stake[i])
                            gdf.loc[[i], ['date1']] = int(str(gdf.date1[i])[:4]+avg_day1)


                date_time_0 = datetime.datetime.strptime(
                     str(int(gdf.date0[i]))+str(int(gdf.time0[i])), '%Y%m%d%H%M')
                date_time_1 = datetime.datetime.strptime(
                    str(int(gdf.date1[i]))+str(int(gdf.time1[i])), '%Y%m%d%H%M')

            except ValueError:
                # Error when date / time is in wrong format/ has typos
                print("Value Error: There is a problem with the date/time format : ",\
                      i, gdf.date0[i], gdf.date1[i], \
                      gdf.time0[i], gdf.time1[i], \
                      gdf.Stake[i])

            # Calculate Period:
            period_new = date_time_1.date()-date_time_0.date()
            if period_new.days != gdf.period[i]:
                # Replace Periods if they are not matching
                print("Period not matching: Replacing ", \
                      i, gdf.Stake[i], gdf.date0[i],\
                      gdf.date1[i], gdf.Stake[i], \
                      gdf.period[i], " - with ", period_new.days)
                gdf.loc[[i],['period']] = period_new.days
    return gdf

def check_geometry(gdf, m_type):
    '''Checks for consistency of locations
    and checks if IDs are correct/corrects them.


        Keyword arguments:
        gdf -- geodataframes
        m_type: str: type of measurements ("annual", "intermediate", "winter")
        '''
    print(gdf.Stake[0])
    if (gdf['y-pos'].min() < 80000) or \
            (gdf['y-pos'].max() > 240000) or \
            (gdf['x-pos'].min() < 550000) or \
            (gdf['y-pos'].max() > 850000):
        print("Detected extreme value outside of Switzerland")
        return
    else:
        return

    # setup background of Map
    sgi = gpd.read_file(r"C:\Users\lea\Documents\data\plots\SGI_2010.shp")
    sgi_old = gpd.read_file(r"C:\Users\lea\Documents\data\plots\Glacierarea_1935_split.shp")
    world = gpd.read_file(r"C:\Users\lea\Documents\data\plots\CHE_adm0.shp")
    world = world.to_crs(epsg=21781)

    sgi = sgi.to_crs(epsg=21781)
    sgi_old = sgi_old.to_crs(epsg=21781)


    # flag all entries that are NOT located on any glaciers to detect outliers:
    pointInPolys = sjoin(gdf, sgi_old, how='left')
    # Join Multipolygon of sgi with points
    # Output is weird grouped index format with all points
    # that lie in sgi

    grouped = pointInPolys.groupby('index_right')

    if len(grouped.groups.keys()) == 1:
        # all points on the same glacier
        # check if it is complete:
        ind_glacier = list(grouped.groups.keys())[0]
        inside_list = list(grouped.groups[ind_glacier])
        if inside_list[len(inside_list)-1] == len(inside_list)-1:
            # All points are on the glacier :)
            print("All points on the glacier")
        else:
            # Compare range and inside_list and print differences:
            difference = list(set(range(len(inside_list)-1)) - set(inside_list))
            f, ax = plt.subplots(1)
            gdf.plot(ax=ax, markersize=30, edgecolor="k", zorder=(30))
            gdf.iloc[difference]['geometry'].plot( \
                ax=ax, markersize=30, edgecolor="r", zorder=(31))
            for dif_ind in difference:
                print("This point is not on the glacier: ", \
                      gdf.Stake[dif_ind], gdf.date0[dif_ind], gdf.date1[dif_ind], \
                        gdf.geometry.x[dif_ind], gdf.geometry.y[dif_ind])
                ax.annotate(gdf.Stake[dif_ind] + str(gdf.date0[dif_ind]) +\
                            str(gdf.date1[dif_ind]) + str(gdf.geometry.x[dif_ind]), \
                                       xy=(gdf.geometry.x[dif_ind], gdf.geometry.y[dif_ind]),\
                        fontsize=15, xytext=(10, 10), textcoords="offset points")
            sgi_old.plot(ax=ax, zorder=3, color='lightsteelblue')

            plt.show()
    else:
            print(" Points on several different glaciers -", \
              " not sure how to continue ", gdf.Stake[0], \
              gdf['y-pos'].min(), gdf['y-pos'].max(),\
              gdf['x-pos'].min(), gdf['x-pos'].max())



def adjust_location_ID(gdf, m_type):
    '''Checks if IDs are correct/corrects them.


        Keyword arguments:
        gdf -- geodataframes
        m_type: str: type of measurements ("annual", "intermediate", "winter")
        '''

    # If same location occurs for other date, leave ID of the first occurence
    # the same and set
    # the ID of the later occurence(s) to 4
    print(gdf.Stake[0])
    for ind in range(len(gdf["x-pos"])):
        if gdf["position-I"][ind] != 4:
            indices = [i for i, x in enumerate(gdf["x-pos"]) if x == gdf["x-pos"][ind]]

            if len(indices) > 1:
                for indices_i in indices[1:]:
                    gdf.loc[[indices_i], ["position-I"]] = 4
    print(gdf["position-I"])

def fill_elevation(gdf):
        '''Fills elevation with NaN/-9999
        with values read from DEMs. Make a rough quality check
        for all other values


            Keyword arguments:
            gdf -- geodataframes
            '''
        print("Lets start this DEM stuff :)  ")
        # Check which date range is available for this glacier:
        # Check if there are any elevations that are NaN or outside of 1000-4800 m - remember index
        # Check if glacier wide DEMs are available - find them, list which dates, read those that are between/close to the dates we need
        # if not, read either DHM25 (for measurements before 1980)
        # or read SwissALti 3D - reproject to LV03
        # find location of NaN in raster and read elevation --> write into gdf
        #


def rename_winter_probes(gdf):
    '''Search winter probes in a radius of 20 meter
    around summer stakes for same year and rename them accordingly'''



if __name__ == '__main__':
    print("Use file_handling as module")