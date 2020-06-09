import os
import bz2
import pandas
import numpy
import sys
sys.path.append("../")
import db_utils
import datetime 
import pydarn


class DownloadSDMap(object):
    """
    A utils class to fetch SuperDARN map data 
    using pydarn!
    Written By - Xueling Shi (05/2020)
    """
    def __init__(self, date_range,ftype='map2',hemi='north'):
        """
        initialize the vars
        date_range = [ start_date, end_date ]
        e.x : date_range = [ 
                        datetime.date(2017,4,16),
                        datetime.date(2017,4,17),
                       ]
        """
        import datetime
        self.date_range = date_range
        self.ftype = ftype
        self.hemi = hemi
        
    def store_map_data(self, db_name="gme_data",
                        table_name="sd_map",
                        local_data_store="../data/sqlite3/"):
        """
        Download AUR inds data and store in a db
        """
        from db_utils import DbUtils
        # fetch the data
        data_df = self.fetch_map_data()
        # set up the database connections!
        db_obj = DbUtils(db_name=db_name,
                     local_data_store=local_data_store)
        if data_df is not None:
            print("Working with map data from VT server!")
            db_obj.map_to_db(data_df, table_name=table_name)
            print("Updated DB!")
        

    def fetch_map_data(self):
        """
        fetch SuperDARN map2 data from the northern hemisphere 
        from VT server
        """
        from datetime import date, timedelta 

        if self.hemi == 'north':
            #stid for mid latitude radars in the northern hemisphere
            sel_rad_ids_midlat=[208,209,206,207,204,205,33,32,40,41]
            #stid for high latitude and polar radars
            sel_rad_ids_highlat=[64,65,16,7,6,5,3,1,8,9,10,66,90] 
        else:
            #stid for mid latitude radars  in the southern hemisphere
            sel_rad_ids_midlat=[14,18,21,24]
            #stid for high latitude and polar radars
            sel_rad_ids_highlat=[96,4,15,20,11,22,13,12,19]             
            
        map2_files = []

        dt_arr = []
        Nvec_mid = []
        Nvec_high = []
        cpcps = []
        HMB_mlat_min = []
           
        date1 = date(self.date_range[1].year,self.date_range[1].month,
                     self.date_range[1].day)
        date0 = date(self.date_range[0].year,self.date_range[0].month,
                     self.date_range[0].day)
        num_days = (date1-date0).days
        dates = [self.date_range[0]+timedelta(days=ii) for ii in range(num_days)]
        for date in dates: 
            local_map_dir="/sd-data/"+str(date.year)+"/"+self.ftype+"/"+self.hemi+"/"
            #print(local_map_dir)
            fnamefmt=date.strftime("%Y%m%d")+"."+self.hemi+"."+self.ftype+".bz2"
            print(fnamefmt)
            map2_files.append(local_map_dir+fnamefmt)
            
        if len(map2_files) == 0:
            print('No '+self.ftype+' file in the '+self.hemi+'ern hemisphere was found!')
            return
        else:    
            print("Reading in map files")
            for file in map2_files:
                # read in compressed file
                with bz2.open(file) as fp: 
                    map2_stream = fp.read()
                # pyDARN functions to read a map file stream
                reader = pydarn.SDarnRead(map2_stream, True)                
                try:
                    map_data = reader.read_map()
                except pydarn.superdarn_exceptions.SuperDARNFieldMissingError as err:
                    print(err)
                    print(file+' data skipped due to missing field exception!')
                    continue
                #map_data = reader.read_map()
                stids = []
                nvecs = []
                
                for i in map_data:
                    cpcps.append(i['pot.drop'])
                    HMB_mlat_min.append(i['latmin'])
                    dt_arr.append(datetime.datetime(i['start.year'],i['start.month'],
                                                    i['start.day'],i['start.hour'],
                                                    i['start.minute']))
                    stids.append(i['stid'])
                    nvecs.append(i['nvec'])
                    
                for ii,id_list in enumerate(stids):
                    tmp_Nvec_mid = 0
                    tmp_Nvec_high = 0
                    for jj,id_ele in enumerate(id_list):
                        if id_ele in sel_rad_ids_midlat:
                            tmp_Nvec_mid += nvecs[ii][jj]
                        if id_ele in sel_rad_ids_highlat:
                            tmp_Nvec_high += nvecs[ii][jj]
                    Nvec_mid.append(tmp_Nvec_mid)
                    Nvec_high.append(tmp_Nvec_high)   
        print("Reading complete...")
        d = {'date':dt_arr,'highlat_nvec':Nvec_high,'midlat_nvec':Nvec_mid,
             'hm_bnd':HMB_mlat_min,'cpcp':cpcps}
        map_df = pandas.DataFrame(d)
        map_df = map_df.astype({"highlat_nvec": "int16", "midlat_nvec": "int16",
                                  "hm_bnd": "float16", "cpcp": "float32"})
        return map_df


if __name__ == "__main__":
    data_obj = DownloadSDMap(
                    date_range = [ 
                                datetime.datetime(2017,1,1),
                                datetime.datetime(2019,1,1),
                               ]
                        )
    #data_df = data_obj.fetch_map_data()
    #if data_df is not None:
    #    print(data_df.head())
    #    print(data_df)
    data_obj.store_map_data()
