import numpy
import pandas
import datetime
import sqlite3


class DbUtils(object):
    """
    A utils class for working with SQLITE3 db

    Written By - Bharat Kunduri (05/2020)
    MOdified By - Shibaji Chakraborty (05/2020)
    MOdified By - Xueling Shi (05/2020)
    """
    def __init__(self, db_name="gme_data",\
                 local_data_store="data/sqlite3/"):
        """
        Initialize some vars for settings

        Parameters
        ----------
        db_name : str, default to None
            Name of the sqlite db to which data will be written
        local_data_store : str
            Path to db_name file
        """
        self.db_name = db_name
        self.local_data_store = local_data_store
        self.conn = self._create_dbconn()

    def _create_dbconn(self):
        """
        make a db connection.
        """

        # make a db connection
        conn = sqlite3.connect(self.local_data_store + self.db_name, detect_types = sqlite3.PARSE_DECLTYPES)
        return conn

    def _commit_tx(self):
        """
        Commit last transaction
        """
        self.conn.commit()
        return

    def _close_dbconn(self):
        """
        Close database connection
        """
        self.conn.close()
        return

    def map_to_db(self, map_df, table_name="sd_map", chunksize=3600, if_exists="replace"):
        """
        Write the dataframe with SuperDARN map data into the db.
        Parameters
        ----------
        map_df : pandas dataframe with SuperDAN map data
        `"""
        sql_create_map_table = """
                                    CREATE TABLE IF NOT EXISTS {tb} (
                                    date TIMESTAMP PRIMARY KEY,
                                    highlat_nvec integer,
                                    midlat_nvec integer,
                                    hm_bnd float,
                                    cpcp float
                                    );
                                """
        command = sql_create_map_table.format(tb=table_name)
        if self.conn is not None:
            self.conn.cursor().execute(command)
        else:
            print("Error! cannot create the db connection!")
        map_df.to_sql(table_name, con=self.conn, if_exists=if_exists, chunksize=chunksize)
        self._commit_tx()
        return
    
    def omni_to_db(self, omni_df, table_name="omni", chunksize=3600, if_exists="replace"):
        """
        Write the dataframe with OMNI data into the db.
        Parameters
        ----------
        omni_df : pandas dataframe with solar wind data
        `"""
        sql_create_omni_table = """
                                    CREATE TABLE IF NOT EXISTS {tb} (
                                    date TIMESTAMP PRIMARY KEY,
                                    bx float,
                                    by float,
                                    bz float,
                                    b float,
                                    vx float,
                                    vy float,
                                    vz float,
                                    v float,
                                    n float,
                                    t float
                                    );
                                """
        command = sql_create_omni_table.format(tb=table_name)
        if self.conn is not None:
            self.conn.cursor().execute(command)
        else:
            print("Error! cannot create the db connection!")
        omni_df.to_sql(table_name, con=self.conn, if_exists=if_exists, chunksize=chunksize)
        self._commit_tx()
        return
    
    def sym_inds_to_db(self, asym_df, table_name="sym_inds"):
        """
        Write the dataframe with aur data into the db.
        Parameters
        ----------
        asym_df : pandas dataframe with sym/asym inds data
        """

        # create table if it doesn't exist
        sql_create_projects_table = """ 
                                        CREATE TABLE IF NOT EXISTS {tb} (
                                        date TIMESTAMP PRIMARY KEY,
                                        asyd integer NOT NULL,
                                        asyh integer,
                                        symd integer,
                                        symh integer
                                    ); 
                                    """
        command = sql_create_projects_table.format(tb=table_name)
        if self.conn is not None:
            self.conn.cursor().execute(command)
        else:
            print("Error! cannot create the db connection!")
        # populate the table
        cols = "date, asyd, asyh, symd, symh"
        for _nrow, _row in asym_df.iterrows():
            command = "INSERT OR REPLACE INTO {tb}({cols}) VALUES (?, ?, ?, ?, ?)".\
                      format(tb=table_name, cols=cols)
            self.conn.cursor().execute(\
                        command,\
                         (_row["date"].to_pydatetime(), _row["asy-d"], _row["asy-h"], _row["sym-d"], _row["sym-h"])\
                         )
        self.conn.commit()
        # close db connection
        self.conn.close()
        
    def kp_to_db(self, kp_df, table_name="kp"):
        """
        Write the dataframe with kp data into the db.
        Parameters
        ----------
        kp_df : pandas dataframe with aur data
        """

        # create table if it doesn't exist
        sql_create_projects_table = """ 
                                        CREATE TABLE IF NOT EXISTS {tb} (
                                        date TIMESTAMP PRIMARY KEY,
                                        kp REAL NOT NULL,
                                        bin_start_date TIMESTAMP,
                                        bin_end_date TIMESTAMP
                                    ); 
                                    """
        command = sql_create_projects_table.format(tb=table_name)
        if self.conn is not None:
            self.conn.cursor().execute(command)
        else:
            print("Error! cannot create the db connection!")
        # populate the table
        cols = "date, kp, bin_start_date, bin_end_date"
        for _nrow, _row in kp_df.iterrows():
            command = "INSERT OR REPLACE INTO {tb}({cols}) VALUES (?, ?, ?, ?)".\
                      format(tb=table_name, cols=cols)
            self.conn.cursor().execute(\
                        command,\
                         (_row["date"].to_pydatetime(), _row["kp"], _row["bin_start_date"].to_pydatetime(), _row["bin_end_date"].to_pydatetime())\
                         )
        self.conn.commit()
        # close db connection
        self.conn.close()
        
    def aur_inds_to_db(self, au_df, table_name="aur_inds"):
        """
        Write the dataframe with aur data into the db.
        Parameters
        ----------
        au_df : pandas dataframe with aur data
        """

        # create table if it doesn't exist
        sql_create_projects_table = """ 
                                        CREATE TABLE IF NOT EXISTS {tb} (
                                        date TIMESTAMP PRIMARY KEY,
                                        au integer NOT NULL,
                                        al integer,
                                        ae integer,
                                        ao integer,
                                        cheat_flag integer
                                    ); 
                                    """
        command = sql_create_projects_table.format(tb=table_name)
        if self.conn is not None:
            self.conn.cursor().execute(command)
        else:
            print("Error! cannot create the db connection!")
        # populate the table
        cols = "date, au, al, ae, ao, cheat_flag"
        for _nrow, _row in au_df.iterrows():
            command = "INSERT OR REPLACE INTO {tb}({cols}) VALUES (?, ?, ?, ?, ?, ?)".\
                      format(tb=table_name, cols=cols)
            self.conn.cursor().execute(\
                        command,\
                         (_row["date"].to_pydatetime(), _row["au"], _row["al"], _row["ae"], _row["ao"], _row["cheat_flag"])\
                         )
        self.conn.commit()
        # close db connection
        self.conn.close()


    def fetch_table_by_sql(self, sql, coerce_float=True, parse_dates=["date"]):
        """
        Fetch data from database table using query
        """
        df = pandas.read_sql(sql, self.conn, coerce_float=coerce_float, parse_dates=parse_dates)
        self._close_dbconn()
        return df
