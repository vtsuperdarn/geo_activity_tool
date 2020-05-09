import datetime   
import os

class ExtractAL(object):
    """
    A utils class to download and extract AL
    from real time plot!
    """
    def __init__(self, download_date_list, local_dir="../data/"):
        """
        initialize the vars.
        """
        import datetime
        import pathlib

        self.base_url = "http://wdc.kugi.kyoto-u.ac.jp/ae_realtime/"
        self.curr_utc = datetime.datetime.utcnow()
        if pathlib.Path(local_dir).exists():
            self.local_dir = local_dir
        else:
            self.local_dir = pathlib.Path.cwd().parent.joinpath(local_dir)
            self.local_dir = self.local_dir.as_posix() + "/"
        self.download_date_list = download_date_list
        # some constants for extracting al data
        self.al_zero_loc = -57.5
        self.al_scale = 17.

    def dwnld_png_file(self, download_date):
        """
        Download the data from the online url.
        """
        import requests
        import pathlib
        import glob
        import os
        # get the data
        print("downloading the AL image from Kyoto!")
        # construct image url!
        img_url = self.base_url + download_date.strftime("%Y%m") + "/"\
                    "rtae_" + download_date.strftime("%Y%m%d") + ".png"
        file_name = self.local_dir + "rtae_" + download_date.strftime("%Y%m%d") + ".png"
        if self.download_image( img_url, file_name ):
            print("Successful!")
            return file_name
        else:
            print("Failed!")
            return None

    def download_image(self, url, file_name):
        import urllib.request
        print("url-->", url)
        try:
            urllib.request.urlretrieve(url,file_name)
            return True
        except:
            print("url not found!")
            return False

    def get_al_data(self):
        """
        Extract data from the image!
        """
        import pandas
        import os
        # we'll first check if the data we the data is already
        # downloaded within the last one hour! we'll skip the
        # download and extract process if it was already present!
        df_list = []
        for _date in self.download_date_list:
            png_filename = self.dwnld_png_file(_date)
            if png_filename is None:
                continue
            else:
                df_list.append( self.extract_al_data(png_filename) )
        if len(df_list) > 0:
            data_df = pandas.concat(df_list)
            data_df.sort_values(by=['date'], inplace=True)
            # interpolate the data
            return data_df
        return None


    def extract_al_data(self, png_file):
        """
        Extract AL data from the image!
        """
        import pandas
        from PIL import Image
        import numpy
        from math import ceil

        # get the date from the png file
        dt_str1 = png_file.split(".png")[0]
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
        tmp_au_arr = []
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
                        for _hr, _mn, _col, _val, _au in zip(tmp_hour_arr,norm_minutes,tmp_col_arr,tmp_ind_arr,tmp_au_arr):
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
                            al_dict[time_ind_count]["au"] = _au
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
            tmp_au_arr.append( _nonwhite[0].min() )
            tmp_col_arr.append( col_arr[_nonwhite[0].max()] )

            minute_count += 1
        data_df = pandas.DataFrame.from_dict(al_dict, orient="index")
        al_scale = last_hor_tick_loc - first_hor_tick_loc
        max_al_val = -1*data_df["al"].max()
        data_df["al"] = (data_df["al"] - data_df["al"].max()) * (3000./al_scale)
        data_df["au"] = ( max_al_val - data_df["au"] ) * (3000./al_scale)
        data_df["al"] = data_df["al"].astype(numpy.int16)
        data_df["au"] = data_df["au"].astype(numpy.int16)
        return data_df
    
    def rgb2hex(self,rgb_arr):
        return "#{:02x}{:02x}{:02x}".format(rgb_arr[0],rgb_arr[1],rgb_arr[2])


if __name__ == "__main__":
    import datetime
    download_date_list = [ datetime.datetime(2019,1,1),\
                          datetime.datetime(2019,1,2) ]
    data_obj = ExtractAL(download_date_list)
    data_df = data_obj.get_al_data()
    print(data_df.tail())
