import datetime
import os
import requests 
from bs4 import BeautifulSoup
import pandas
import numpy
#from netCDF4 import Dataset
import xarray
import sys
sys.path.append("../")
import db_utils

class DownloadDscovr(object):
    """
    A utils class to download and save DSCOVR satellite data set
    
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

    def fetch_dscovr_data(self, db_name="gme_data",\
            table_name="dscovr",\
            local_data_store="../data/sqlite3/"):
        """
        Download dscovr data and store the data
        """
        from db_utils import DbUtils
        dbo = DbUtils(db_name=db_name, local_data_store=local_data_store)
        _df = self._download_dscovr()
        if _df is not None:
            dbo.dscovr_to_db(_df, table_name=table_name)
            print("Updated DB!")
        dbo._close_dbconn()
        return

    def _download_page_content(self):
        """
        Download html web page by web scraping
        """
        furls = []
        murls = []
        uri = "https://www.ngdc.noaa.gov/dscovr/data/{year}/{month}/"
        days = int((self.date_range[1] - self.date_range[0]) / datetime.timedelta(days=1)) + 1
        month = -1
        for d in range(days):
            e = self.date_range[0] + datetime.timedelta(days=d)
            if month != e.month:
                url = uri.format(year=e.year, month="%02d"%e.month)
                print(url)
                resp = requests.get(url)
                if resp.status_code==200:
                    soup = BeautifulSoup(resp.text,"html.parser")
                    rows = [str(x.string) for x in soup.find_all("a")]
                    for r in rows:
                        if (r is not None) and ("oe_f1m" in r):
                            furls.append(url + r)
                        elif (r is not None) and ("oe_m1m" in r):
                            murls.append(url + r)
                month = e.month
        return furls, murls

    def _download_dscovr(self):
        """
        Download all the files for this date range
        """
        days = (self.date_range[1].day - self.date_range[0].day) + 1
        cmd = "wget -O /tmp/dscovr.nc.gz {url}"
        local_gz = "/tmp/dscovr.nc.gz"
        local = "/tmp/dscovr.nc"
        _df = pandas.DataFrame()
        furls, murls = self._download_page_content()

        params = {"time": "date", "proton_vx_gse": "vx", "proton_vy_gse": "vy", "proton_vz_gse": "vz", "proton_density": "n", 
                "proton_temperature": "t"}
        _f = pandas.DataFrame()
        for url in furls:
            e = datetime.datetime.strptime(url.split("_")[3].replace("s",""), "%Y%m%d%H%M%S")
            if (e >= self.date_range[0]) and (e <= self.date_range[1]):
                uri = cmd.format(url=url)
                os.system(uri)
                os.system("gunzip -c {input} > {output}".format(input=local_gz,output=local))
                _f = pandas.concat([_f, (xarray.open_dataset(local).to_dataframe().reset_index()[params.keys()]).rename(columns=params)])
                os.system("rm {local}".format(local=local))
                os.system("rm {local}".format(local=local_gz))
        _f = _f.sort_values(by=["date"])

        params = {"time": "date", "by_gse": "by_gse", "bz_gse": "bz_gse", "bx_gsm": "bx", "by_gsm": "by", "bz_gsm": "bz"}
        _m = pandas.DataFrame()
        for url in murls:
            e = datetime.datetime.strptime(url.split("_")[3].replace("s",""), "%Y%m%d%H%M%S")
            if (e >= self.date_range[0]) and (e <= self.date_range[1]):
                uri = cmd.format(url=url)
                os.system(uri)
                os.system("gunzip -c {input} > {output}".format(input=local_gz,output=local))
                _m = pandas.concat([_m, (xarray.open_dataset(local).to_dataframe().reset_index()[params.keys()]).rename(columns=params)])
                os.system("rm {local}".format(local=local))
                os.system("rm {local}".format(local=local_gz))
        _m = _m.sort_values(by=["date"])

        _df = pandas.merge(_f, _m, on="date").drop_duplicates(subset="date", keep="first").set_index("date")
        _df = _df.astype({"bx": "float16", "by": "float16", "bz": "float16", "by_gse": "float16", "bz_gse": "float16",
                        "vx": "float16", "vy": "float16", "vz": "float16", "n": "float16", "t": "float32"})
        print(_df.head())
        return _df

def fetch_dscovr_by_dates(sdate, edate, db_name="gme_data",\
        table_name="dscovr", local_data_store="../data/sqlite3/", imf_coord="gsm"):
    """
    Get the stored DSCOVR data from omni database
    Parameters
    ----------
    sdate : start datetime
    edate : end datetime
    db_name : name of the database
    table_name : table name
    local_data_store : folder location of the files
    imf_coord : Coordinate of the IMF data "gsm", "gse"
    """
    from db_utils import DbUtils
    dbo = DbUtils(db_name=db_name, local_data_store=local_data_store)
    sql = """SELECT * from {tb} WHERE strftime('%s', date) BETWEEN strftime('%s', '{sdate}') AND strftime('%s', '{edate}')""".format(tb=table_name,sdate=sdate,edate=edate)
    print("Running sql query >> ",sql)
    df = dbo.fetch_table_by_sql(sql)
    nan_directory = {"bx":10000.0, "by":10000.0, "bz":10000.0, "by_gse":10000.0, "bz_gse":10000.0,
            "b":10000.0, "v":99999.9, "vx":99999.9, "vy":99999.9, "vz":99999.9,
            "n":1000.0, "t":9999999.}
    for _kv in nan_directory.keys():
        df = df.replace(nan_directory[_kv], numpy.inf)
    if imf_coord == "gsm": df = df.drop(columns=["by_gse","bz_gse"])
    elif imf_coord == "gse":
        df = df.drop(columns=["by","bz"])
        df = df.rename(columns={"by_gse":"by", "bz_gse":"bz"})
    print(df.head())
    return df

if __name__ == "__main__":
    dscovr = DownloadDscovr(
            date_range = [
                datetime.datetime(2016,1,1),
                datetime.datetime(2020,5,21),
                ]
            )
    dscovr.fetch_dscovr_data()
    #fetch_dscovr_by_dates(datetime.datetime(2017,12,1), datetime.datetime(2017,12,4))
    #os.system("rm ../data/sqlite3/*")
