import datetime   
import os
from kyoto_utils import create_url_form, iaga_format_to_df
import pandas
import numpy
import sys
sys.path.append("../")
import db_utils

class DownloadAsym(object):
    """
    A utils class to download and extract AL
    from real time plot!

    Written By - Bharat Kunduri (05/2020)
    """
    def __init__(self, date_range):
        """
        initialize the vars
        date_range = [ start_date, end_date ]
        e.x : date_range = [ 
                        datetime.datetime(2015,1,1),
                        datetime.datetime(2015,3,1),
                       ]
        """
        import datetime

        self.asy_base_url = "http://wdc.kugi.kyoto-u.ac.jp/cgi-bin/aeasy-cgi?"
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
            for _nranges in range(int(self.ndays_dt_range/max_download_day_limit) + 1):
                end_date = start_date + datetime.timedelta(days=max_download_day_limit)
                if end_date > date_range[1]:
                    end_date = date_range[1]
                self.date_range_list.append(
                                    [start_date,end_date]
                                )
                start_date  = end_date
        
    def fetch_store_aur_data(self, db_name="gme_data",\
                        table_name="sym_inds",\
                        local_data_store="../data/sqlite3/"):
        """
        Download AUR inds data and store in a db
        """
        from db_utils import DbUtils
        # fetch the data
        data_df, missing_dates = self.get_asym_data()
        # set up the database connections!
        db_obj = DbUtils(db_name=db_name,\
                     local_data_store=local_data_store)
        if data_df is not None:
            print("Downloaded data!")
            db_obj.sym_inds_to_db(data_df, table_name=table_name)
            print("Updated DB!")
        

    def get_asym_data(self, convert_asy_to_int=True):
        """
        download Asym/Sym data
        """

        missed_date_range = []
        asym_df_list = []
        for _dt in self.date_range_list:
            resp = create_url_form(_dt, self.asy_base_url, output="ASY")
            raw_data = resp.readlines()
            _df = iaga_format_to_df(raw_data)
            bad_date_list = []
            if _df is not None:
                # remove unwanted values!
                bad_val_dates = _df[_df["sym-h"] <= -10000.]["date"].dt.date.unique()
                if bad_val_dates.shape[0] > 0:
                    bad_date_list = list(bad_val_dates)
                # we have several columns as float64's by default!
                # convert them into int16's
                if convert_asy_to_int:
                    _df["asy-d"] = _df['asy-d'].astype(numpy.int16)
                    _df["asy-h"] = _df['asy-h'].astype(numpy.int16)
                    _df["sym-d"] = _df['sym-d'].astype(numpy.int16)
                    _df["sym-h"] = _df['sym-h'].astype(numpy.int16)
                asym_df_list.append(_df)
        # merge all the data into a larger DF
        if len(asym_df_list) > 0:
            # check if we missed any dates!
            # if yes we'll try the cheat mode method
            # to download the data
            asym_df = pandas.concat(asym_df_list)
            last_download_date = asym_df["date"].max()
            diff_hours = ( self.date_range_list[-1][-1] - last_download_date ).total_seconds()/3600.
            # NOTE if the difference is greater than 24 hours!
            if diff_hours <= 24:
                return asym_df, [] + bad_date_list
            else:
                num_days = (self.date_range_list[-1][-1] - last_download_date).days
                date_list = [\
                            last_download_date +\
                             datetime.timedelta(days=x) for x in range(num_days)\
                            ]
                return asym_df, date_list + bad_date_list

        else:
            print("No data found")
            num_days = (self.date_range_list[-1][-1] - self.date_range_list[0][0]).days
            date_list = [\
                        self.date_range_list[0][0] +\
                         datetime.timedelta(days=x) for x in range(num_days)\
                        ]
            return None, date_list + bad_date_list


if __name__ == "__main__":
    data_obj = DownloadAsym(
                    date_range = [ 
                                datetime.datetime(2017,1,1),
                                datetime.datetime(2019,1,1),
                               ]
                        )
#     data_df, missing_dates = data_obj.get_asym_data()
#     if data_df is not None:
#         print(data_df.tail())
#     print("----------------")
#     print(missing_dates)
    data_obj.fetch_store_aur_data()

