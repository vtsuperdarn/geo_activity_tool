import numpy
import pandas
import datetime
import sqlite3
import sys
sys.path.append("../")

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
    #print(df.head())
    return df

def fetch_gme_by_dates(sdate, edate, db_name="gme_data",\
        table_name="sd_map", col_names='*', local_data_store="../data/sqlite3/"):
    """
    Get the stored data from gme_data database
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
    sql = """SELECT {cols} from {tb} WHERE strftime('%s', date) BETWEEN strftime('%s', '{sdate}') AND strftime('%s', '{edate}')\
            """.format(cols=col_names,tb=table_name,sdate=sdate,edate=edate)
    print("Running sql query >> ",sql)
    df = dbo.fetch_table_by_sql(sql)
    #print(df.head())
    return df


def fetch_gme_by_dates_shi(sdate, edate, db_name="gme_data",\
        table_name="sd_map", col_names='*', local_data_store="../data/sqlite3/"):
    """
    Get the stored data from gme_data database
    Parameters
    ----------
    sdate : start datetime
    edate : end datetime
    db_name : name of the database
    table_name : table name
    local_data_store : folder location of the files
    """
    conn = sqlite3.connect(local_data_store + db_name, detect_types = sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    sqlite_select_query_symh = """SELECT ? from ? where date between ? and ?"""
    cursor.execute(sqlite_select_query_symh,(col_names, table_name, start_time, end_time))
    records = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return records
        
        
def fetch_data_plotting(sdate,edate,plot_style="classic",\
                        flags=[False,True,True,True,True],\
                        local_data_store="../data/sqlite3/"):
    """
    Get the stored data from the database and process it for plotting
    Parameters
    ----------
    sdate : start datetime
    edate : end datetime
    plot_style: plotting style
    flags : flag to fetch data from [DSCOVER,OMNI,STORM,SUBSTORM,SUPERDARN]
    local_data_store : folder location of the files
    """
    import numpy
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    ax_per_dataset = [2,2,1,1,4] 
    nrows = numpy.sum(ax_per_dataset,where=flags)
    
    plt.style.use(plot_style)
    plt.rcParams['xtick.labelsize']=14
    plt.rcParams['ytick.labelsize']=14
    myFmt = mdates.DateFormatter('%H:%M')
    
    fig, axes  = plt.subplots(nrows=nrows, ncols=1,sharex=True,figsize=(10,14))    
    xlim=[sdate,edate]
    
    ind = 0
    if flags[0]:
        ind += ax_per_dataset[0]
        
    if flags[1]:
        #fetch and process omni data for plotting
        omni_df = fetch_omni_by_dates(sdate,edate,local_data_store=local_data_store)
        dt_omni = omni_df.date
        By = omni_df.by
        Bz = omni_df.bx
        Vt = omni_df.v
        Np = omni_df.n
        
        axes[ind].plot(dt_omni,By,'k', label='By') 
        axes[ind].plot(dt_omni,Bz,'r', label='Bz') 
        axes[ind].axhline(y=0,color='k',linestyle=':')
        axes[ind].set_ylabel('OMNI IMF [nT]',fontsize=12) 
        axes[ind].set(xlim=xlim)
        axes[ind].legend(fontsize=12)        
        ind += 1
        
        axes[ind].plot(dt_omni,Vt,'k', label='Vt')
        axes[ind].set(xlim=xlim)
        axes[ind].set_ylabel('OMNI Vt [km/s]',fontsize=12)
        axt1 = axes[ind].twinx()
        axt1.plot(dt_omni,Np,'b') 
        axt1.set_ylabel('Np [cm^-3]',color='b',fontsize=12) 
        axt1.tick_params(axis='y', labelcolor='b')
        ind += 1
    
        
    if flags[2]:
        #fetch and process symh data for plotting
        symh_df = fetch_gme_by_dates(sdate,edate,local_data_store=local_data_store,\
                                     table_name="sym_inds", col_names='date, symh')
        sort_symh_df = symh_df.sort_values(by='date')
        dt_symh = sort_symh_df.date
        symh = sort_symh_df.symh
        
        #fetch and process kp data for plotting
        kp_df = fetch_gme_by_dates(sdate,edate,local_data_store=local_data_store,\
                                   table_name="kp", col_names='date, kp')         
        dt_kp = kp_df.date
        kp = kp_df.kp
        
        axes[ind].plot(dt_symh,symh,'k') 
        axes[ind].axhline(y=0,color='k',linestyle=':')
        axes[ind].set_ylabel('Symh ind [nT]',fontsize=12) 
        axes[ind].set(xlim=xlim)
        axt2 = axes[ind].twinx()
        axt2.plot(dt_kp,kp,'bo') 
        axt2.set(ylim=[-1,9])
        axt2.set_yticks(numpy.arange(0, 9,3))
        axt2.set_ylabel('Kp-Index',color='b',fontsize=12) 
        axt2.tick_params(axis='y', labelcolor='b')
        ind += 1
    

    if flags[3]:    
        #fetch and process ae data for plotting
        ae_df = fetch_gme_by_dates(sdate,edate,local_data_store=local_data_store,\
                                   table_name="aur_inds", col_names='date, ae')
        dt_ae = ae_df.date
        ae = ae_df.ae
        
        axes[ind].plot(dt_ae,ae,'k')
        axes[ind].set(xlim=xlim)
        axes[ind].set_ylabel('AE ind [nT]',fontsize=12)
        ind += 1

    if flags[4]:  
        #fetch and process sdmap data for plotting
        sdmap_df = fetch_gme_by_dates(sdate,edate,local_data_store=local_data_store,\
                                      table_name="sd_map", col_names='*')
        dt_arr = sdmap_df.date
        Nvec_mid = sdmap_df.midlat_nvec
        Nvec_high = sdmap_df.highlat_nvec
        HMB_mlat_min = sdmap_df.hm_bnd
        cpcps = sdmap_df.cpcp
        Nvec_total=[sum(i) for i in zip(Nvec_mid, Nvec_high)] 
        
        axes[ind].plot(dt_arr,Nvec_high,'orange', label='High-lat') 
        axes[ind].plot(dt_arr,Nvec_total,'k', label='Total') 
        axes[ind].set(xlim=xlim)#
        axes[ind].set_ylabel('Num.Vecs',fontsize=12)
        axes[ind].legend(fontsize=12)
        ind += 1

        axes[ind].plot(dt_arr,Nvec_mid,'k', label='Mid-lat')
        axes[ind].set(xlim=xlim)
        axes[ind].set_ylabel('Num.Vecs',fontsize=12)#Midlat-NumVecs
        axes[ind].legend(fontsize=12)#loc='upper right'
        ind += 1
        
        axes[ind].plot(dt_arr,HMB_mlat_min,'k', label='HM-BND')#
        axes[ind].axhline(y=60,color='k',linestyle=':')
        axes[ind].set(xlim=xlim)
        axes[ind].set_ylabel('MLAT [degree]',fontsize=12)
        axes[ind].legend(fontsize=12)
        ind += 1
        
        axes[ind].plot(dt_arr,numpy.array(cpcps)/1000.,'k') 
        axes[ind].set(xlim=xlim)#,ylim=ylim
        axes[ind].set_ylabel('CPCP [keV]',fontsize=12)
        axes[ind].xaxis.set_major_formatter(myFmt)        
    return fig

#sdate=datetime.datetime(2018,1,1)
#edate=datetime.datetime(2018,1,2)
#fig = fetch_data_plotting(sdate,edate,plot_style="classic",\
#                          flags=[False,True,True,True,True],\
#                          local_data_store="/Users/xueling/data/sqlite3/")