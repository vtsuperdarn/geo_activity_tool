import datetime   
import os

class DwnldRTSW(object):
    """
    A utils class to download and read real time
    propagated SW IMF data
    """
    def __init__(self, local_file="data/"):
        """
        initialize the vars
        """
        import pathlib
        self.file_url = "https://services.swpc.noaa.gov/products/geospace/propagated-solar-wind.json"
        if pathlib.Path(local_file).exists():
            self.local_file = local_file
        else:
            self.local_file = pathlib.Path.cwd().parent.joinpath(local_file)
            self.local_file = self.local_file.as_posix() + "/"
        self.file_str = "rt_sw_imf_"

    def dwnld_file(self, check_recent=True, update_time_limit=20.):
        """
        Download the data from the online url
        We don't want to download data from the NOAA server
        every single time! if check_recent is set to true, we'll
        check if recently downlaoded data is within a certain time
        limit ('update_time_limit') and download the data only if
        the time limit is within 20 minutes.
        """
        import urllib.request
        import json
        import pandas
        import pathlib
        download_data = True
        if check_recent:
            if pathlib.Path(self.local_file).is_dir():
                data_df = self.read_stored_data()
                if data_df is not None:
                    recent_time = data_df['time_tag'].max()
                    time_diff = (datetime.datetime.utcnow() - recent_time \
                                        ).total_seconds()/60.
                    if time_diff < update_time_limit:
                        download_data = False
                        print("Data is recent! not downloading from server!")
        data = None
        if download_data:
            # get the data
            with urllib.request.urlopen(self.file_url) as url:
                if url.getcode() == 200:
                    print("Data succesfully downloaded!!!")
                    data = json.loads(url.read().decode())
                else:
                    data = None
        return data
        

    def read_url_data(self, url_data,\
             store_data=True, delete_old_files=True):
        """
        read the latest data
        """
        import pandas
        if url_data is None:
            print("download failed! check again")
            return
        col_list = url_data[0]
        sw_imf_data = url_data[1:]
        # convert to a df
        data_df = pandas.DataFrame.from_records(sw_imf_data,\
                             columns=col_list)
        # convert to datetime
        data_df["time_tag"] = pandas.to_datetime( data_df["time_tag"] )
        data_df["propagated_time_tag"] = pandas.to_datetime(\
                                         data_df["propagated_time_tag"] )
        # these are propagated to bow shock (I think!). We'll add a 10 minute
        # propagation delay to magnetopause!
        data_df["time_tag"] = data_df["time_tag"] +\
                                     datetime.timedelta(minutes=10)
        data_df["propagated_time_tag"] = data_df["propagated_time_tag"] +\
                                             datetime.timedelta(minutes=10)
        if store_data:
            file_date = datetime.datetime.utcnow().strftime("%Y%m%d-%H:%M")
            if delete_old_files:
                for _f in os.listdir(self.local_file):
                    if os.path.isfile(os.path.join(self.local_file,_f))\
                         and self.file_str in _f:
                         os.remove(os.path.join(self.local_file,_f))
            new_file_name = self.local_file + self.file_str +\
                                 file_date + ".csv"
            data_df.to_csv( new_file_name )
            print("saved file--->", new_file_name)
        return data_df

    def read_stored_data(self):
        """
        Read data from the stored file!
        """
        # Note there could be multiple files!
        # so we need to read the latest one!
        import pandas
        read_file = None
        latest_file_date = None
        for _f in os.listdir(self.local_file):
            _file_loc = os.path.join(self.local_file,_f)
            if os.path.isfile(_file_loc) and self.file_str in _f:
                if read_file is None:
                    read_file = _file_loc
                    _file_date = datetime.datetime.strptime(\
                                    _f.split("_")[-1].split(".")[0],\
                                    "%Y%m%d-%H:%M")
                    latest_file_date = _file_date
                else:
                    _file_date = datetime.datetime.strptime(\
                                    _f.split("_")[-1].split(".")[0],\
                                    "%Y%m%d-%H:%M")
                    if latest_file_date is None:
                        latest_file_date = _file_date
                        read_file = _file_loc
                    else:
                        if latest_file_date < _file_date:
                            latest_file_date = _file_date
                            read_file = _file_loc
        print("reading from file-->", read_file)
        if read_file is None:
            return None
        data_df = pandas.read_csv(read_file, index_col=0, parse_dates=[1])
        data_df.sort_values( by=["propagated_time_tag"], inplace=True )
        data_df["propagated_time_tag"] = pandas.to_datetime(\
                                         data_df["propagated_time_tag"] )
        return data_df





if __name__ == "__main__":
    data_obj = DwnldRTSW()
    url_data = data_obj.dwnld_file()
    if url_data is not None:
        data_df = data_obj.read_url_data(url_data)
    data_df = data_obj.read_stored_data()