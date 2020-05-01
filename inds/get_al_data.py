import datetime   
import os
from kyoto_utils import create_url_form, iaga_format_to_df
import pandas
import numpy

class ExtractAL(object):
    """
    A utils class to download and extract AL
    from real time plot!

    Written By - Bharat Kunduri (04/2020)
    """
    def __init__(self, date_range, local_data_store="data/",\
                cheat_mode=True):
        """
        initialize the vars
        date_range = [ start_date, end_date ]
                e.x : date_range = [ 
                                datetime.datetime(2015,1,1),
                                datetime.datetime(2015,3,1),
                               ]
        setting "cheat_mode" to True will enable us to download the
        images and then extract AL/AU from the images 
        (when data is not available). Obviously, people
        may not like it if you did this! But it looks like THEMIS
        folks are doing it or plan to do it! 
        """
        import datetime
        import pathlib

        self.aul_base_url = "http://wdc.kugi.kyoto-u.ac.jp/cgi-bin/aeasy-cgi?"
        self.ndays_dt_range = (date_range[1] - date_range[0]).days
        if self.ndays_dt_range <= 0:
            print("End date should be later than start date! Please try again")
            return
        # we can only download 366 days at a time!
        # to download more, we need to use multiple requests!
        max_download_day_limit = 366
        if self.ndays_dt_range <= max_download_day_limit:
            self.date_range_list = [ date_range ]
        else:
            self.date_range_list = []
            start_date = date_range[0]
            for _nranges in range(self.ndays_dt_range/max_download_day_limit + 1):
                end_date = start_date + datetime.timedelta(days=max_download_day_limit)
                if end_date > date_range[1]:
                    end_date = date_range[1]
                self.date_range_list.append(
                                    [start_date,end_date]
                                )
                start_date  = end_date
        # some constants for cheat mode data
        self.al_zero_loc = -57.5
        self.al_scale = 17.

    def get_al_data(self, convert_aul_to_int=True):
        """
        download AE data
        """

        missed_date_range = []
        aul_df_list = []
        for _dt in self.date_range_list:
            resp = create_url_form(_dt, self.aul_base_url)
            raw_data = resp.readlines()
            _df = iaga_format_to_df(raw_data)
            if _df is not None:
                # we have several columns as float64's by default!
                # convert them into int16's
                if convert_aul_to_int:
                    _df["ae"] = _df['ae'].astype(numpy.int16)
                    _df["al"] = _df['al'].astype(numpy.int16)
                    _df["ao"] = _df['ao'].astype(numpy.int16)
                    _df["au"] = _df['au'].astype(numpy.int16)
                aul_df_list.append(_df)
        # merge all the data into a larger DF
        if len(aul_df_list) > 0:
            # check if we missed any dates!
            # if yes we'll try the cheat mode method
            # to download the data
            aul_df = pandas.concat(aul_df_list)
            last_download_date = aul_df["date"].max()
            diff_hours = ( self.date_range_list[-1][-1] - last_download_date ).total_seconds()/3600.
            # NOTE if the difference is greater than 24 hours!
            if diff_hours <= 24:
                return aul_df, []
            else:
                return aul_df, [ last_download_date, self.date_range_list[-1][-1] ]

        else:
            print("No data found")
            return None, [ self.date_range_list[0][0], self.date_range_list[-1][-1] ]


if __name__ == "__main__":
    data_obj = ExtractAL(
                    date_range = [ 
                                datetime.datetime(2015,1,1),
                                datetime.datetime(2015,1,3),
                               ]
                        )
    data_df, missing_dates = data_obj.get_al_data()
    print(data_df.head())
    print("----------------")
    print(missing_dates)
