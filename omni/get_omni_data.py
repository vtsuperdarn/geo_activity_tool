import datetime
import os
import pandas
import numpy
import sys
sys.path.append("../")
import db_utils

class DownloadOmni(object):
    """
    A utils class to download and save Omni data set
    
    Written By - Shibaji Chakraborty (05/2020)
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
        return

    def fetch_omni_data(self, db_name="omni_data",\
            table_name="omni",\
            local_data_store="../data/sqlite3/"):
        """
        Download OMNI data and store the data
        """
        from db_utils import DbUtils
        dbo = DbUtils(db_name=db_name, local_data_store=local_data_store)
        _df = self._download_omni()
        if _df is not None:
            dbo.omni_to_db(_df, table_name=table_name)
            print("Updated DB!")
        dbo._close_dbconn()
        return

    def _to_pandas(self, fname):
        """
        Convert ASCII data file to pandas object
        Parameters
        ----------
        fname : Name of the temp ascii file
        """
        header = ["date","id_imf","id_sw","nimf","nsw","point_ratio","time_shift(sec)","rms_time_shift","rms_pf_normal",
                "time_btw_obs","bfa","bx","by_gse","bz_gse","by_gsm","bz_gsm","b_rms","bfa_rms","v","vx_gse","vy_gse",
                "vz_gse","n","t","p_dyn","e","beta","mach_a","x_gse","y_gse","z_gse","bsn_xgse","bsn_ygse","bsn_zgse",
                "ae","al","au","sym-d","sym-h","asy-d","asy-h","pc-n","mach_m"]
        with open(fname, "r") as f: lines = f.readlines()
        linevalues = []
        for i, line in enumerate(lines):
            values = list(filter(None, line.replace("\n", "").split(" ")))
            timestamp = datetime.datetime(int(values[0]),1,1,int(values[2]),int(values[3])) + datetime.timedelta(days=int(values[1])-1)
            linevalues.append([timestamp,int(values[4]),int(values[5]),int(values[6]),int(values[7]),int(values[8]),int(values[9]),
                int(values[10]),float(values[11]),int(values[12]),float(values[13]),float(values[14]),float(values[15]),float(values[16]),
                float(values[17]),float(values[18]),float(values[19]),float(values[20]),float(values[21]),float(values[22]),float(values[23]),
                float(values[24]),float(values[25]),float(values[26]),float(values[27]),float(values[28]),float(values[29]),float(values[30]),
                float(values[31]),float(values[32]),float(values[33]),float(values[34]),float(values[35]),float(values[36]),
                float(values[37]),float(values[38]),float(values[39]),float(values[40]),float(values[41]),float(values[42]),float(values[43]),
                float(values[44]),float(values[45])])
        _o = pandas.DataFrame(linevalues, columns=header)
        _o = _o[["date", "bx", "by_gse", "bz_gse", "b_rms", "v", "vx_gse", "vy_gse", "vz_gse", "n", "t"]]
        _o = _o.rename(columns={"vx_gse": "vx", "vy_gse": "vy", "vz_gse": "vz", "by_gse": "by", "bz_gse": "bz", "b_rms": "b"})
        _o = _o.astype({"bx": "float16", "by": "float16", "bz": "float16", "b": "float16", "v": "float16",
            "vx": "float16", "vy": "float16", "vz": "float16", "n": "float16", "t": "float32"})
        return _o

    def _download_omni(self):
        """
        Download all the files for this date range
        """
        years = (self.date_range[1].year - self.date_range[0].year) + 1
        cmd = "wget -O /tmp/omni_min{year}.asc https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/omni_min{year}.asc"
        local = "/tmp/omni_min{year}.asc"
        _df = pandas.DataFrame()
        for y in range(years):
            yr = y + self.date_range[0].year
            uri = cmd.format(year = yr)
            tmp = local.format(year = yr)
            print(uri)
            os.system(uri)
            _df = pandas.concat([_df, self._to_pandas(tmp)])
            os.system("rm /tmp/omni_min{year}.asc".format(year = yr))
        _df = _df.set_index("date")
        return _df


def fetch_omni_by_dates(sdate, edate, db_name="omni_data",\
        table_name="omni", local_data_store="../data/sqlite3/"):
    """
    Get the stored OMNI data from omni database
    Parameters
    ----------
    sdate : start datetime
    edate : end datetime
    db_name : name of the database
    table_name : table name
    local_data_store : folder location of the files
    """
    from db_utils import DbUtils
    dbo = DbUtils(db_name=db_name, local_data_store=local_data_store)
    sql = """SELECT * from {tb} WHERE strftime('%s', date) BETWEEN strftime('%s', '{sdate}') AND strftime('%s', '{edate}')\
            """.format(tb=table_name,sdate=sdate,edate=edate)
    print("Running sql query >> ",sql)
    df = dbo.fetch_table_by_sql(sql)
    nan_directory = {"bx":10000.0, "by":10000.0, "bz":10000.0, "b":10000.0, "v":99999.9,
            "vx":99999.9, "vy":99999.9, "vz":99999.9, "n":1000.0, "t":9999999.}
    for _kv in nan_directory.keys():
        df = df.replace(nan_directory[_kv], numpy.inf)
    print(df.head())
    return df

if __name__ == "__main__":
    domni = DownloadOmni(
            date_range = [
                datetime.datetime(2017,1,1),
                datetime.datetime(2018,1,1),
                ]
            )
    domni.fetch_omni_data()
    fetch_omni_by_dates(datetime.datetime(2017,1,31), datetime.datetime(2017,2,2))
    #os.system("rm ../data/sqlite3/*")
