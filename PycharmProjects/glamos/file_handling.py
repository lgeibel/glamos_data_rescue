import pandas as pd
import geopandas as gpd
import os
import datetime
from geopandas.tools import sjoin
from raster2xyz.raster2xyz import Raster2xyz
import csv
import matplotlib.pyplot as plt
from scipy import stats
import itertools
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from scipy import spatial

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

def import_excel(filename, filepath, headerl, print_preview=True):
    '''Imports .xls file into df dataframe

    Keyword arguments:
    filename -- name of .xls file
    filepath -- path to folder containing file
    headerl -- header lines
    print_preview -- if True (default), preview of file is printed

    Returns: df

    '''
    file = os.path.join(filepath, filename)
    print(filename)
    df = pd.read_excel(file, header=headerl, \
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
         r"S:\glazio\projects\8003-VAW_GCOS_data_rescue\gl_net_info.xlsx",
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
        print(df_list[i]["Stake"])
        # Replace commas with dots and convert to floats:
        if type(df_list[i]['x-pos']) == str:
            df_list[i]['x-pos'] = [x.replace(',', '.') for x in df_list[i]['x-pos']]
            df_list[i]['x-pos'] = df_list[i]['x-pos'].astype(float)
            df_list[i]['y-pos'] = [x.replace(',', '.') for x in df_list[i]['y-pos']]
            df_list[i]['y-pos'] = df_list[i]['y-pos'].astype(float)
       # print(df_list[i]['Stake'],df_list[i]['date0'],df_list[i]['x-pos'], df_list[i]['y-pos'])

        try:
            gdf_list[i] = gpd.GeoDataFrame(df_list[i],\
                geometry=gpd.points_from_xy(df_list[i]['x-pos'], df_list[i]['y-pos']))
        except TypeError:
            print("What is wrong?")
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
        if m_type == "annual":
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
                            try:
                                print("Replacing this one: : ", \
                                  i, gdf.date0[i], gdf.date1[i], \
                                  gdf.time0[i], gdf.time1[i] , " with ", int(str(gdf.date0[i])[:4]+avg_day1), \
                                  gdf.Stake[i])
                            except UnboundLocalError:
                                print("What is wrong?")
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
    print("Adjusting location IDs for", gdf.Stake[0])
    duplicateRowsDF = gdf[gdf.duplicated(['x-pos', 'y-pos'])]
    # Exclude Entries that have same date1 (otherwise GPR does funny stuff):
    duplicateRowsDF[~duplicateRowsDF.duplicated(["date1"])]
    for ind in duplicateRowsDF.index:
        for ind1 in duplicateRowsDF.index:
            if duplicateRowsDF["x-pos"][ind] == duplicateRowsDF["x-pos"][ind1]:
                if duplicateRowsDF["y-pos"][ind] == duplicateRowsDF["y-pos"][ind1]:
                    if duplicateRowsDF["date1"][ind] != duplicateRowsDF["date1"][ind1]:
                        # exclude values from the same year (only occurs for GPR measurements)
                        gdf.loc[[ind], ["position-ID"]] = 4
    print(gdf[gdf["position-ID"] == 4])
    return gdf

def fill_elevation(gdf, m_type, gdf5):
    '''Fills elevation with NaN/-9999
        with values read from DEMs. Make a rough quality check
        for all other values - values with deviation of more than 150 from SwissAlti3D
        will be replaced


            Keyword arguments:
            gdf -- geodataframes
    '''
    print("Lets start this DEM stuff :)  ")


    # Check which date range is available for this glacier:
    DEM_dir = r"C:\Users\lea\Documents\data\plots\DEMs"
    date_range = (str(gdf.date0.min), str(gdf.date0.max()))
    ### Overall test of all points: - if deviation is more than 200 m form SwissALti3D,
    # replace value:
    # Index of all unique x-pos values for this glacier:
    dhm_ind = gdf["x-pos"].drop_duplicates().index.to_list()

    DHM_dir = os.path.join(DEM_dir, "swissAlti3D")
    ind_dhm = [f for i, f in enumerate(os.listdir(DHM_dir))
               if (gdf["inv-no"][0][1:] in os.listdir(DHM_dir)[i])]
#    for file in ind_dhm:
#        gdf = get_SwissAlti3D(DEM_dir, file, gdf, dhm_ind, check_distance=True)

    # Check if there are any elevations that are NaN or outside of 1000-4800 m
    outside_date = [str(i)[:4] for i in gdf[np.isnan(gdf['z-pos']) | \
                                            (gdf["z-pos"] > 4500) | \
                                            (gdf["z-pos"] < 1500)]["date1"].to_list()]
    outside_index = gdf[np.isnan(gdf['z-pos']) | (gdf["z-pos"] > 4500) | (gdf["z-pos"] < 1500)][
        "z-pos"].index.to_list()
    # Only check values where version 5 is different than version 4:
    outside_date = [str(i)[:4] for i in gdf[gdf["z-pos"] != gdf5["z-pos"]]["date1"].to_list()]
    outside_index = gdf[gdf["z-pos"] != gdf5["z-pos"]]["z-pos"].index.to_list()

    # Check if glacier wide DEMs are available - find them, list which dates,
    # read those that are between/close to the dates we need
    ind_dem = [f for f in os.listdir(DEM_dir) if f.startswith(gdf.Glaciername[0])]
    if len(outside_date) != 0:
        print("There are values that are to be corrected for ", gdf.Stake[0],outside_index)
    # only for those entries where we actually need to correct something...
        ind_dem_dates = [date.split("_")[1][:4] for date in ind_dem]
        if (len(ind_dem) !=0) and (min([int(i) for i in ind_dem_dates])< 1990):
        # only do this if we have local DEM - otherwise SWissALti or DHM25
        # check which dates we have DEMs for: if oldest dem is younger than 1990,
        # use DHM 25 and SwissAlti instead
            print("Use local DEMs for: ", gdf.Stake[0])
            aux = []
            for valor in ind_dem_dates:
                aux.append(abs(min([int(i) for i in outside_date]) - int(valor)))
            # index of oldest DEM that we have MB measurements for
            oldest_dem_ind = aux.index(min(aux))

            # for each entry that needs to be checked, find the index of the dem (dem_ind) that is closest:
            dem_ind = []
            for date in outside_date:
                aux = []
                for valor in ind_dem_dates:
                    aux.append(abs(int(date) - int(valor)))
                dem_ind.append(aux.index(min(aux)))

            dem_ind_unique = list(set(dem_ind))
            # Read this DEM:
            for dem_ind_ind in dem_ind_unique:
                dem = pd.read_csv(os.path.join(DEM_dir, ind_dem[dem_ind_ind]), delim_whitespace=True, engine="python")
                #dem = pd.read_csv(os.path.join(DEM_dir, ind_dem[dem_ind_ind]), sep='  ', engine="python")
                dem = dem.fillna(0)
                # check dimension of dem. If not 3 columns, something went wrong...

                # now we have dem. now get the entries n entries out of outside_index that have the
                # a dem_ind that is the same as dem_ind_ind:
                indices = [i for i, x in enumerate(dem_ind) if x == dem_ind_ind]
                # now we have all the indices of the points for which this dem is used to fill the z-pos:
                fill_ind = [outside_index[i] for i in indices]
                A = dem.iloc[:,:2].to_numpy()
                for f in fill_ind:
                    # find index of point in dem that is closest to our point pt for which we need
                    # add the z-component:
                    pt = [gdf["x-pos"][f], gdf["y-pos"][f]]
                    # distance and index of clostest point:
                    distance, ind = spatial.KDTree(A).query(pt)
                    # get z-pos of closest point:
                    z_val = dem.iloc[ind,2]
                    # write to gdf:
                    print("Replacing: ", f, z_val, gdf["z-pos"][f])

                    gdf.loc[[f], ['z-pos']] = z_val


        elif (min([int(i) for i in outside_date]) < 1990):
                print("Use DHM25 for :", gdf.Stake[0])
                # get index of points that need DHM25:
                dhm_ind = [int(i) for ind, i in enumerate(outside_index) if int(outside_date[ind])< 1990]
                DHM_dir = os.path.join(DEM_dir, "DHM25", "asc")
                ind_dhm = [f for f in os.listdir(DHM_dir)
                       if f.startswith("mm" + str(gdf.lk25[0]))]
                for file in ind_dhm:
                    # Read file, convert to xyz
                    with rasterio.open(os.path.join(DEM_dir, "DHM25", "asc", file)) as src:
                        image = src.read()
                        # transform image
                        bands, rows, cols = np.shape(image)
                        image1 = image.reshape(rows * cols, bands)
                        print(np.shape(image1))
                        # bounding box of image
                        l, b, r, t = src.bounds
                        # resolution of image
                        res = src.res
                        res = src.res
                        # meshgrid of X and Y
                        x = np.arange(l, r, res[0])
                        y = np.arange(t, b, -res[0])
                        X, Y = np.meshgrid(x, y)
                        print(np.shape(X))
                        # flatten X and Y
                        newX = np.array(X.flatten('C'))
                        newY = np.array(Y.flatten('C'))
                        print(np.shape(newX))
                        # join XY and Z information
                        dem = np.column_stack((newX, newY, image1))
                        A = dem[:, :2]
                        for f in dhm_ind:
                            # find index of point in dem that is closest to our point pt for which we need
                            # add the z-component:
                            pt = [gdf["x-pos"][f], gdf["y-pos"][f]]
                            print(pt)
                            # distance and index of clostest point:
                            distance, ind = spatial.KDTree(A).query(pt)
                            # get z-pos of closest point:
                            z_val = dem[ind, 2]
                            print("Replacing: ", z_val, gdf["z-pos"][f])
                            # write to gdf:
                            gdf.loc[[f], ['z-pos']] = z_val
        else:
                print("Use Swiss Alti 3D for :", gdf.Stake[0])
                # get index of points that need Swiss Alti 3D:
                dhm_ind = [int(i) for ind, i in enumerate(outside_index)
                           if int(outside_date[ind])>= 1990]
                DHM_dir = os.path.join(DEM_dir, "swissAlti3D")
                ind_dhm = [f for i, f in enumerate(os.listdir(DHM_dir))
                     if (gdf["inv-no"][0] in os.listdir(DHM_dir)[i])]
                if len(ind_dhm)==0:
                    ind_dhm = [f for i, f in enumerate(os.listdir(DHM_dir))
                               if (gdf["inv-no"][0][1:] in os.listdir(DHM_dir)[i])]
                for file in ind_dhm:
                    gdf = get_SwissAlti3D(DEM_dir, file, gdf, dhm_ind, check_distance= False)
    return gdf

def get_SwissAlti3D(DEM_dir, file, gdf, dhm_ind, check_distance):
    """Read SwissAlti*d for glacier, reproject, convert to xyz, find closest value,
    reads out elevation
    Params:
    DEM_dir: dircetory containing SwissAlti3D glacier wide tiles
    file: filename of SA3D tile that contains glacier (based on Gl-ID)
    gdf: geopandas dataframe containing all point measurememts for this m_type and glacier
    dhm_ind: list of index of all points in gdf for which we need to check elevation
    check_distance: if true, check if current z-val is more than 150 m away from SWA3D,
                    if not, just replace all NaN z-values
    """
    dst_crs = 'EPSG:21781'
    # Read file, reproject to LV03
    with rasterio.open(os.path.join(DEM_dir, "swissAlti3D", file)) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open('RGB.tif', 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

    # Read file, convert to xyz
    # with rasterio.open('RGB.tif') as src:
    #     image = src.read()
    #     # transform image
    #     bands, rows, cols = np.shape(image)
    #     image1 = image.reshape(rows * cols, bands)
    #     print(image1.shape)
    #     # bounding box of image
    #     l, b, r, t = src.bounds
    #     # resolution of image
    #     res = src.res
    #     res = src.res
    #     if (res[0]< 10):
    #         res = list(res)
    #         res[0]=10
    #     # meshgrid of X and Y
    #     x = np.arange(l, r, res[0])
    #     y = np.arange(t, b, -res[0])
    #     X, Y = np.meshgrid(x, y)
    #     # flatten X and Y
    #     newX = np.array(X.flatten('C'))
    #     newY = np.array(Y.flatten('C'))
    #     print(newX.shape, newY.shape)
    #     # join XY and Z information
    #     try:
    #         dem = np.column_stack((newX, newY, image1))
    #     except ValueError:
    #         print("What is wrong here?")
    rtxyz = Raster2xyz()
    rtxyz.translate("RGB.tif", "out_xyz.csv")
    dem = pd.read_csv("out_xyz.csv", delimiter=",", engine="python")
    dem = dem[dem["z"] > 0]
    dem = dem.fillna(0).to_numpy()

    A = dem[:, :2]
    for f in dhm_ind:
        # find index of point in dem that is closest to our point pt for which we need to
        # add the z-component:
        pt = [gdf["x-pos"][f], gdf["y-pos"][f]]
        print(f)
        # distance and index of clostest point:
        distance, ind = spatial.KDTree(A).query(pt)
        # get z-pos of closest point:
        z_val = dem[ind, 2]
        if check_distance:
            # Check if z_val and original z_val are more than 150 m apart - if so, replace, if not, leave
            dis = abs(z_val - gdf["z-pos"][f])
            if dis>150:
                if dis>10000000000000:
                    print("Discrepancy bigger only because of NaN val ")
                else:
                    print("Discrepancy bigger than 150 meter, " , z_val, gdf["z-pos"][f])
                    gdf.loc[[f], ['z-pos']] = z_val
                    print("Z-Val and index to be replaced: ", z_val, gdf["x-pos"][f], f)
                    # Find index of all gdf entries with same position (x-pos) and replace
                    # z-pos there as well:
                    for ix in gdf[gdf["x-pos"] == gdf["x-pos"][f] and gdf["y-pos"] == gdf["y-pos"][f]].index.tolist():
                        if len(gdf[gdf["x-pos"] == gdf["x-pos"][f]].index.tolist()) > 1:
                            gdf.loc[[ix], ['z-pos']] = z_val
                            print("Also adjusted z-val for index ", z_val, gdf["x-pos"][ix], ix)

        else:
            # write to gdf:
            print("Replacing ", z_val, gdf["z-pos"][f])
            gdf.loc[[f], ['z-pos']] = z_val
    return gdf

def rename_winter_probes(gdf_summer, gdf_winter):
    '''Search winter probes in a radius of 30 meter
    around summer stakes for same year and rename them accordingly
    gdf_summer: geodataframe of annual measurements of one glacier
    gdf_winter: geodataframe of winter measurements of the same glacier
    '''
    # For each entry in gdf_summer see if there is entry in gdf_winter
    # for same year (date0 of summer = year/first4 digits of date1 winter -1)
    # if so, look which gdf_winter entry is 30 m around --> change name of winter entry to summer entry
    for i, summ in enumerate(gdf_summer["date0"]):
        # for each summer entry, check if there are winter entries for the same year:
        # list of entries of winter entries for same year:
        ind_list = [i for i,ent in enumerate(gdf_winter["date1"]) if (int(str(ent)[:4])-1) \
                    == int(str(summ)[:4])]
        if len(ind_list)>0:
            # Find/check for points in 30 meter radius
            x_pos = [gdf_winter["x-pos"][ind] for ind in ind_list]
            y_pos = [gdf_winter["y-pos"][ind] for ind in ind_list]
            A = np.column_stack((x_pos, y_pos))
            pt = [gdf_summer["x-pos"][i], gdf_summer["y-pos"][i]]
            # distance and index of closest point to this summer point::
            distance, ind = spatial.KDTree(A).query(pt)
            if distance < 100:
                # Rename winter point with summer name:
                print("Renaming winter point:", gdf_summer["Stake"][i], gdf_winter.loc[ind_list[ind], ['Stake']])
                gdf_winter.loc[ind_list[ind], ['Stake']] = gdf_summer["Stake"][i]

    return gdf_winter

def test_deviations(gdf_list):
    """Test deviations and uncertainties introduced e.g. bei spatial uncertainties, uncertainties in time"""
    interest = gdf_list[0].iloc[0:1]
    for gdf in gdf_list:
        print(interest)
        interest = interest.append(gdf[(gdf.period > 1)])
        interest = interest[(interest.period) < 30]

    print(interest)



if __name__ == '__main__':
    print("Use file_handling as module")