import datetime   
import os
from kyoto_utils import set_kp_url, kp_to_df
import pandas
import numpy
import sys
sys.path.append("../")
import db_utils

class DownloadKp(object):
    """
    A utils class to download and extract AL
    from real time plot!

    Written By - Bharat Kunduri (04/2020)
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
        self.date_range = date_range
        
    def fetch_store_kp_data(self, db_name="gme_data",\
                        table_name="kp",\
                        local_data_store="../data/sqlite3/"):
        """
        Download AUR inds data and store in a db
        """
        from db_utils import DbUtils
        # fetch the data
        data_df, missing_dates = self.get_kp_data()
        # set up the database connections!
        db_obj = DbUtils(db_name=db_name,\
                     local_data_store=local_data_store)
        if data_df is not None:
            print("Working with data wdc provides!")
            db_obj.kp_to_db(data_df, table_name=table_name)
            print("Updated DB!")
        

    def get_kp_data(self):
        """
        download Kp data
        """
        resp = set_kp_url(self.date_range)
        raw_data = resp.readlines()
        kp_df = kp_to_df(raw_data)
        if kp_df is not None:
            last_download_date = kp_df["date"].max()
            missing_dates = []
            if ( (last_download_date.year != self.date_range[1].year) &\
                (last_download_date.month != self.date_range[1].month) ):
                num_days = (self.date_range_list[-1][-1] - last_download_date).days
                missing_dates = [\
                            last_download_date +\
                             datetime.timedelta(days=x) for x in range(num_days)\
                            ]
            return kp_df, missing_dates
        else:
            num_days = (self.date_range_list[-1][-1] - self.date_range_list[0][0]).days
            date_list = [\
                        self.date_range_list[0][0] +\
                         datetime.timedelta(days=x) for x in range(num_days)\
                        ]
            return None, date_list + bad_date_list

if __name__ == "__main__":
    data_obj = DownloadKp(
                    date_range = [ 
                                datetime.datetime(2016,1,1),
                                datetime.datetime(2017,1,1),
                               ]
                        )
#     data_df, missing_dates = data_obj.get_kp_data()
#     if data_df is not None:
#         print(data_df.head())
    data_obj.fetch_store_kp_data()

