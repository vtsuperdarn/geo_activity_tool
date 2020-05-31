import datetime   
import os

class ExtractAL(object):
    """
    A utils class to download and extract AL
    from real time plot!
    """
    def __init__(self, local_file="data/"):
        """
        initialize the vars
        """
        import datetime
        import pathlib

        self.base_url = "http://wdc.kugi.kyoto-u.ac.jp/ae_realtime/"
        self.curr_utc = datetime.datetime.utcnow()
        if pathlib.Path(local_file).exists():
            self.local_file = local_file
        else:
            self.local_file = pathlib.Path.cwd().parent.joinpath(local_file)
            self.local_file = self.local_file.as_posix() + "/"
        self.curr_file_str = "rt_" + self.curr_utc.strftime("%Y%m%d-%H%M") + ".png"
        self.prev_date = self.curr_utc - datetime.timedelta(days=1)
        self.prev_file_str = "rt_" + self.prev_date.strftime("%Y%m%d-%H%M") + ".png"
        self.hist_al_data_file = self.local_file + "hist_rt_al_data.csv"
        # some constants for extracting al data
        self.al_zero_loc = -57.5
        self.al_scale = 17.

    def dwnld_png_file(self, check_recent=True,\
             update_time_limit=60., delete_old_files=True):
        """
        Download the data from the online url
        We don't want to download data from the server
        every single time! if check_recent is set to true, we'll
        check if recently downlaoded data is within a certain time
        limit ('update_time_limit') and download the data only if
        the time limit is within 20 minutes.
        """
        import requests
        import pathlib
        import glob
        import os

        download_data = True
        if check_recent:
            recent_date = datetime.datetime(2000,1,1)
            recent_file = [None]
            if pathlib.Path(self.local_file).is_dir():
                file_list = glob.glob(self.local_file + "rt_*.png")
                # extract download date from file!
                # Note there may be multiple files due to some mistake!                
                for _f in file_list:
                    _str1 = _f.split("_")
                    _str2 = _str1[1].split("-")
                    _date_str = _str2[0]
                    _time_str = _str2[1].split(".")[0]
                    _dtstr = _date_str + "-" + _time_str
                    _dt = datetime.datetime.strptime(_dtstr, "%Y%m%d-%H%M")
                    if _dt > recent_date:
                        recent_date = _dt
                        recent_file = [_f]
            file_dwnld_time_diff = (self.curr_utc - recent_date).total_seconds()/60.
#             print(recent_date, self.curr_utc, file_dwnld_time_diff, update_time_limit)
            if file_dwnld_time_diff < update_time_limit:
                print("AL image is recent! not downloading from server!")
                download_data = False
        if download_data:
            # get the data
            print("downloading most recent AL image!")
            file_name = self.local_file + self.curr_file_str
            prev_file_name = self.local_file + self.prev_file_str
            # construct image url!
            img_url = self.base_url + self.curr_utc.strftime("%Y%m") + "/"\
                        "rtae_" + self.curr_utc.strftime("%Y%m%d") + ".png"
            # construct prev day image url!
            prev_img_url = self.base_url + self.prev_date.strftime("%Y%m") + "/"\
                        "rtae_" + self.prev_date.strftime("%Y%m%d") + ".png"
            recent_file = []
            if self.download_image( img_url, file_name ):
                recent_file.append(file_name)
            if self.download_image( prev_img_url, prev_file_name ):
                recent_file.append(prev_file_name)
            if delete_old_files:
                del_files = list(\
                             set(glob.glob(self.local_file + "rt_*.png"))\
                              - set([file_name, prev_file_name])\
                              )
                print(del_files)
                for _df in del_files:
                    if os.path.isfile(_df):
                        _csv_fname = _df.split(".")[0] + ".csv"
                        print("deleting files-->", _df)
                        os.remove(_df)
                        if os.path.exists(_csv_fname):
                            print("deleting files-->", _csv_fname)
                            os.remove(_csv_fname)
                    else:
                        print("file not found-->", _df)
        return recent_file

    def download_image(self, url, file_name):
        import urllib.request
        print("url-->", url)
        try:
            urllib.request.urlretrieve(url,file_name)
            return True
        except:
            print("url not found!")
            return False

    def get_al_data(self, store_data=True):
        """
        Get AL data from the image if a new image is downloaded
        else, extract data from the image!
        """
        import pandas
        import os
        # we'll first check if the data we the data is already
        # downloaded within the last one hour! we'll skip the
        # download and extract process if it was already present!
        hist_al_df = None
        if os.path.isfile(self.hist_al_data_file):
            hist_al_df = pandas.read_csv(\
                            self.hist_al_data_file,\
                            parse_dates=[2],\
                            index_col=0
                            )
        png_filenames = self.dwnld_png_file()
        df_list = []
        for _pf in png_filenames:
            # check if we have a corresponding csv file!
            if os.path.isfile( _pf.split(".")[0] + ".csv" ):
                if os.path.isfile(self.hist_al_data_file):
                    return hist_al_df
                else:
                    return self.read_stored_data(\
                                    _pf.split(".")[0] + ".csv"\
                                    )
            else:
                print("extracting data from image file : " + _pf)
                df_list.append( self.extract_al_data(_pf) )
        data_df = pandas.concat(df_list)
        data_df.sort_values(by=['date'], inplace=True)
#         if store_data:
        new_file_name = png_filenames[0].split(".")[0] + ".csv"
        data_df.to_csv( new_file_name )
        print("saved file--->", new_file_name)
        if hist_al_df is not None:
            hist_al_df = pandas.concat([hist_al_df,data_df])
            hist_al_df.drop_duplicates(\
                        subset="date",\
                        keep="last",\
                        inplace=True\
                        )
            hist_al_df.sort_values(by=['date'], inplace=True)
            hist_al_df.reset_index(drop=True,inplace=True)
            hist_al_df.to_csv( self.hist_al_data_file )
            return hist_al_df
        else:
            data_df.to_csv( self.hist_al_data_file )
            return data_df


    def extract_al_data(self, png_file):
        """
        Extract AL data from the image!
        """
        import pandas
        from PIL import Image
        import numpy
        from math import ceil

        # get the date from the png file
        dt_str1 = png_file.split(".")[0]
        dt_str2 = dt_str1.split("_")[1]
        dt_str = dt_str2.split("-")[0]

        # now get to image processing
        im = Image.open(png_file, "r")
        width, height = im.size 
        # Setting the points for cropped image 
        left = 75#5
        top = 27#height / 2
        right = width-45
        bottom = height / 2 - 25
          
        # Cropped image of above dimension 
        # (It will not change orginal image) 
        crpd_image = im.crop((left, top, right, bottom)) 
        iar = numpy.asarray(crpd_image)
        # we need a few col strings for extracting data
        white_col = "#ffffff"
        grey_col = "#bbbbbb"
        # while we already cropped the image, we'll get a little more
        # precise and locate exact top and bottom edges of AU/AL plot
        first_hor_tick_loc = iar.shape[0]
        first_hor_tick_found = False
        last_hor_tick_loc = iar.shape[0]

        for _horind in range(iar.shape[0]):
            _horarr = iar[_horind,:,:]
            col_arr = numpy.apply_along_axis(self.rgb2hex, 1, _horarr)
            # check for grey cols
            _graycols = numpy.where( col_arr == grey_col )
            if _graycols[0].shape[0] > 400:
                if not first_hor_tick_found:
                    first_hor_tick_found = True
                    first_hor_tick_loc = _horind
                else:
                    last_hor_tick_loc = _horind
        iar = iar[first_hor_tick_loc:last_hor_tick_loc,:,:]
        # Now we'll process by vertical pixels(which mark time)
        # need to see where the ticks start
        # this will be the first time we encounted
        # a lot of grey colors
        first_tick_found = False
        first_tick_loc = 0
        last_tick_loc = iar.shape[1]

        al_dict = {}
        # we need to keep track of hours and minutes
        # we'll do it by counting the grey lines in the
        # plot, which seperate each hour!
        time_ind_count = 0

        hour_count = 0
        minute_count = 0
        # intitalize some temp arrays!
        tmp_minute_arr = []
        tmp_hour_arr = []
        tmp_ind_arr = []
        tmp_col_arr = []

        for _utind in range(iar.shape[1]):
            _utarr = iar[:,_utind,:]
            col_arr = numpy.apply_along_axis(self.rgb2hex, 1, _utarr)
            # check for grey cols
            _graycols = numpy.where( col_arr == grey_col )
            # these are the line markers
            if _graycols[0].shape[0] > 100:
                if not first_tick_found:
                    first_tick_found = True
                    first_tick_loc = _utind
                else:
                    # store data into a dict and start a new hour
                    # normalize minutes
                    if len(tmp_minute_arr) > 0:
                        if max(tmp_minute_arr) > 20:
                            norm_minutes = [ceil(x*59./max(tmp_minute_arr)) for x in tmp_minute_arr]
                        else:
                            norm_minutes = [ceil(x*59./21) for x in tmp_minute_arr]
                        for _hr, _mn, _col, _val in zip(tmp_hour_arr,norm_minutes,tmp_col_arr,tmp_ind_arr):
                            if _hr < 10:
                                _hrmnstr = "0" + str(_hr) + ":" + str(int(_mn))
                            else:
                                _hrmnstr = str(_hr) + ":" + str(int(_mn))
                            _curr_dt = datetime.datetime.strptime(\
                                                dt_str + "-" + _hrmnstr,\
                                                "%Y%m%d-%H:%M"
                                                )
                            al_dict[time_ind_count] = {}
                            al_dict[time_ind_count]["col"] = _col
                            al_dict[time_ind_count]["date"] = _curr_dt
                            al_dict[time_ind_count]["al"] = _val
                            time_ind_count += 1
                    # reinitialize vals
                    minute_count = 0
                    tmp_minute_arr = []
                    tmp_hour_arr = []
                    tmp_ind_arr = []
                    tmp_col_arr = []
                    hour_count += 1
                    last_tick_loc = _utind
                continue
            # start the estimates after the first tick is found
            if not first_tick_found:
                continue
            _nonwhite = numpy.where( \
                                    (col_arr != white_col) &\
                                    (col_arr != grey_col)
                                   )
            
            if _nonwhite[0].shape[0] == 0:
                continue

            _hours = int(time_ind_count/24.)
            _mints = ((time_ind_count/24.)*60) % 60
            _curr_dt = datetime.datetime.strptime(\
                                        dt_str + "-" +\
                                        str(_hours) + "-" +\
                                        str(int(_mints)),\
                                        "%Y%m%d-%H-%M"
                                        )
            tmp_minute_arr.append( minute_count )
            tmp_hour_arr.append( hour_count )
            tmp_ind_arr.append( -1*_nonwhite[0].max() )
            tmp_col_arr.append( col_arr[_nonwhite[0].max()] )

            minute_count += 1
        data_df = pandas.DataFrame.from_dict(al_dict, orient="index")
        al_scale = last_hor_tick_loc - first_hor_tick_loc
        data_df["al"] = (data_df["al"] - data_df["al"].max()) * (3000./al_scale)
        data_df["al"] = data_df["al"].round(1)
        return data_df

    def rgb2hex(self,rgb_arr):
        return "#{:02x}{:02x}{:02x}".format(rgb_arr[0],rgb_arr[1],rgb_arr[2])

    def read_stored_data(self, file_name):
        """
        Read data from the stored file!
        """
        # Note there could be multiple files!
        # so we need to read the latest one!
        import pandas
        print("reading from file-->", file_name)
        data_df = pandas.read_csv(file_name, index_col=0, parse_dates=[2])
        return data_df





if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from matplotlib.dates import date2num, DateFormatter, MinuteLocator
    data_obj = ExtractAL()
    data_df = data_obj.get_al_data()
    print(data_df.head())
    # f = plt.figure(figsize=(12, 8))
    # f, ax = plt.subplots(1)
    # plt.style.use("fivethirtyeight")
    # ax.plot( data_df["date"], data_df["al"] )
    # ax.set_ylim( [-3000, 1000] ) #
    # _cd = data_df["date"].min()
    # _ed = _cd + datetime.timedelta(days=1)
    # xlim = [
    #         datetime.datetime(_cd.year,_cd.month, _cd.day) ,\
    #         datetime.datetime(_ed.year,_ed.month, _ed.day)
    #          ]
    # ax.set_xlim(xlim)
    # ax.get_xaxis().set_major_formatter(DateFormatter('%H:%M'))
    # ax.grid()
    # plt.savefig("data/test_al.pdf")