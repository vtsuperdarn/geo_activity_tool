import matplotlib
matplotlib.use("Agg")

import pandas as pd
import datetime as dt
import numpy as np
import sqlite3
import operator

import os
import sys
import glob

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from matplotlib import ticker
from matplotlib.dates import date2num, DateFormatter
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates

import argparse
import dateutil

sym_storm_cutoff = -50
crngtn_rtn = 27.2753
crngtn_2012_first = dt.datetime( 2012, 1, 9 )
crngtn_2012_first = crngtn_2012_first + dt.timedelta( days = 0.9416 )
succ_pred_days = 3
ylim = [-200, 100]


def get_data(start, end, v):
    if v: print(" Date limits-", start, end)
    sym_db_name = "data/sqlite3/gme_data"
    sym_table_name = "sym_inds"
    sym_train_params = ["symh"]
    conn = sqlite3.connect(sym_db_name, detect_types = sqlite3.PARSE_DECLTYPES)
    command = "SELECT * FROM {tb} WHERE date BETWEEN '{stm}' AND '{etm}'"
    command = command.format(tb=sym_table_name,
            stm=start,
            etm=end)
    sym_df = pd.read_sql(command, conn)
    sym_df = sym_df.rename(columns={"date":"datetime"})
    sym_df = sym_df[ sym_train_params + [ "datetime" ] ]
    sym_df = sym_df[sym_df["symh"] < 999.].reset_index(drop=True)
    sym_df = sym_df.replace(np.inf, np.nan)
    sym_df = sym_df.set_index("datetime")
    sym_df["date"] = sym_df.index.date
    sym_df.reset_index(inplace=True)
    if v:
        print(" Dataset:")
        print(sym_df.head())
    return sym_df

def agg_symh_data(df, v):
    sym_agg = df.groupby(["date"]).agg([np.nanmax, np.nanmin, np.nanmedian]).reset_index()
    sym_agg.columns = sym_agg.columns.droplevel()
    sym_agg = sym_agg.rename(columns={"nanmax":"symh_max", "nanmin":"symh_min", "nanmedian":"symh_median"})
    sym_agg.rename(columns={"": "date"}, inplace=True)
    sym_agg.head()
    if v:
        print(" Aggrigated Dataset:")
        print(sym_agg.head())
    return sym_agg

def storm_symh_data(df, v):
    sym_storms_df = df[ df["symh_min"] <= sym_storm_cutoff ].reset_index(drop=True)
    sym_storms_df["pred_1cyc"] = sym_storms_df["date"] + dt.timedelta( days = crngtn_rtn )
    sym_storms_df["pred_2cyc"] = sym_storms_df["date"] + dt.timedelta( days = crngtn_rtn*2 )
    sym_storms_df.head()
    if v:
        print(" Storm Aggrigated Dataset:")
        print(sym_storms_df.head())
    return sym_storms_df

def get_crnt_dates(start, end):
    crngtn_cycle_num = []
    crngtn_dates = []
    curr_crngtn_date = crngtn_2012_first
    curr_crngtn_cycle = 2119
    while curr_crngtn_date <= end:
        if curr_crngtn_date > start - dt.timedelta(days=40):
            crngtn_dates.append(curr_crngtn_date +  dt.timedelta( days = crngtn_rtn ) )
            crngtn_cycle_num.append( curr_crngtn_cycle + 1 )
        curr_crngtn_date += dt.timedelta( days = crngtn_rtn )
        curr_crngtn_cycle += 1
    return crngtn_cycle_num, crngtn_dates

def plot_data(sym_agg, sym_storms_df, sym_storm_hour, crngtn_dates, crngtn_cycle_num, fname):
    plt.style.use("fivethirtyeight")
    f = plt.figure(figsize=(11, 8))
    ax = f.add_subplot(1,1,1)
    colors = sns.color_palette()
    med_plt = ax.scatter(x=sym_agg["date"].values,
            y=sym_agg["symh_median"].values, marker=".",
            color="k", s=14,zorder=8, label="Median SymH (24 hours)")
    rng_plt = ax.fill_between(sym_agg["date"].values, 
            y1=sym_agg["symh_min"],
            y2=sym_agg["symh_max"],
            alpha=0.75,label="SymH Range (24 hours)",\
                    facecolor=colors[0], zorder=7)
    pr2 = ax.plot_date( sym_storm_hour["pred_2cyc"].values,
            sym_storm_hour["symh_min"].values, "s",
            label="Projected event (2 cycles) ",
            markersize = 6, color=colors[1], markerfacecolor="none" )
    pr1 = ax.plot_date( sym_storm_hour["pred_1cyc"].values,
            sym_storm_hour["symh_min"].values, ".",
            label="Projected event (1 cycle) ",
            markersize = 12, color=colors[1], markerfacecolor="none")
    ac1 = ax.plot_date( sym_storm_hour["date"].values,
            sym_storm_hour["symh_min"].values, "rx" ,
            label = "Storm Event (< -50 nT) ",
            markersize = 12, color=colors[1] )
    for _row in sym_storms_df.iterrows():
        if _row[1]["pred_1cyc"] <= crngtn_dates[-1].date():
            ax.plot_date( [_row[1]["date"], _row[1]["pred_1cyc"]], [_row[1]["symh_min"],  _row[1]["symh_min"]],
                    ":", color = "gray", linewidth = 1. )
        if _row[1]["pred_2cyc"] <= crngtn_dates[-1].date():
            ax.plot_date( [_row[1]["pred_1cyc"], _row[1]["pred_2cyc"]], [_row[1]["symh_min"], _row[1]["symh_min"]],
                    ":", color = "gray", linewidth = 1. )
    count_add_pos = 0
    old_cr_xpos = 0
    for _ncr_dt, _cr_dt in enumerate(crngtn_dates) :
        if _ncr_dt < len(crngtn_dates)-1:
            _sel_df = sym_storm_hour[(sym_storm_hour["datetime"] >= _cr_dt) &
                    (sym_storm_hour["datetime"] <= crngtn_dates[_ncr_dt+1])]
        if _sel_df.shape[0] > 0 :
            for _row in _sel_df.iterrows():
                str_curr = _row[1]["datetime"].strftime("%m/%d-%H") + "UT"
                xpos_dates = _cr_dt + dt.timedelta( days = 1 )
                new_cr_pos = _ncr_dt
                if ( new_cr_pos == old_cr_xpos and _row[1]["datetime"] != sym_storm_hour["datetime"].min() ) :
                    count_add_pos = count_add_pos + 5
                else:
                    old_cr_xpos = new_cr_pos
                    count_add_pos = 0
                if _row[1]["symh"] == _sel_df["symh"].min():
                    text_col = "red"
                else:
                    text_col = "black"
                ax.annotate( str_curr, (mdates.date2num(xpos_dates),
                    -195 + count_add_pos), size = 8, color =text_col )
    for d, e in zip( crngtn_cycle_num, crngtn_dates ) :
        ax.annotate( "cyc-"+str(d), xy =( mdates.date2num(e), ylim[1] - 8 ), size = 8 )                
    lab_1cyc_realized = False
    lab_2cyc_realized = False
    for _row in sym_storm_hour.iterrows():
        _time_del = sym_storm_hour["date"] - _row[1]["pred_1cyc"]
        _time_del_2cyc = sym_storm_hour["date"] - _row[1]["pred_2cyc"]
        _sel_dates = sym_storm_hour["date"]
        _time_del = _time_del[(_time_del.dt.days.abs() <= succ_pred_days)]
        _time_del_2cyc = _time_del_2cyc[_time_del_2cyc.dt.days.abs() <= succ_pred_days]
        if _time_del.shape[0] > 0:
            if not lab_1cyc_realized:
                label = "Realized Event (1 cycle)"
                lab_1cyc_realized = True
            else:
                label = None
            ac1_rlzd = ax.plot_date( [_row[1]["pred_1cyc"]], [_row[1]["symh_min"]], "." , label = label,
                    markersize = 12, color=colors[1] )
        else:
            if not lab_1cyc_realized:
                ac1_rlzd = ax.plot_date( [], [], "s" , label = "Realized Event (1 cycle)", markersize = 6, color=colors[1] )
                lab_1cyc_realized = True
        if _time_del_2cyc.shape[0] > 0:
            if not lab_2cyc_realized:
                label = "Realized Event (2 cycles)"
                lab_2cyc_realized = True
            else:
                label = None
            ac2_rlzd = ax.plot_date( [_row[1]["pred_2cyc"]], [_row[1]["symh_min"]], "s" , label = label, markersize = 6, color=colors[1] )
        else:
            if not lab_2cyc_realized:
                ac2_rlzd = ax.plot_date( [], [], "s", label = "Realized Event (2 cycles)", markersize = 6, color=colors[1] )
                lab_2cyc_realized = True

    sorted_1cyc_df = sym_storm_hour.sort_values(["pred_1cyc"])#.nlargest(2)
    sorted_2cyc_df = sym_storm_hour.sort_values(["pred_2cyc"])
    
    max_1cyc_dates = [sorted_1cyc_df.iloc[-1]["pred_1cyc"], sorted_1cyc_df.iloc[-2]["pred_1cyc"]]
    max_2cyc_dates = [sorted_2cyc_df.iloc[-1]["pred_2cyc"], sorted_2cyc_df.iloc[-2]["pred_2cyc"]]
    for _1cyc, _2cyc in zip(max_1cyc_dates, max_2cyc_dates):
        _sym_1cyc = sym_storm_hour[ sym_storm_hour["pred_1cyc"] == _1cyc ]["symh_min"]
        _sym_2cyc = sym_storm_hour[ sym_storm_hour["pred_2cyc"] == _2cyc ]["symh_min"]
        if _sym_1cyc.shape[0] > 0:
            ax.text(x = _1cyc, y = _sym_1cyc.iloc[0], s = _1cyc.strftime("%m/%d"),
                    fontsize = 8, weight = "bold", alpha = .75)
        if _sym_2cyc.shape[0] > 0:
            ax.text(x = _2cyc, y = _sym_2cyc.iloc[0], s = _2cyc.strftime("%m/%d"),
                    fontsize = 8, weight = "bold", alpha = .75)    
    ax.set_ylabel("SymH Index [nT]", fontsize = 14)
    ax.set_xlabel("Carrington cycle start dates", fontsize = 14)
    ax.set_title("24-hr SymH vs Carrington cycle", fontsize = 15 )
    ax.set_xticks( crngtn_dates )
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=70 )
    ax.set_ylim(ylim)
    handle_ordered = [ med_plt, rng_plt, ac1[0], pr1[0], pr2[0], ac1_rlzd[0], ac2_rlzd[0] ]
    label_ordered = [ "Median SymH (24 hours)", "SymH Range (24 hours)", "Storm Event (< -50 nT)", "Projected event (1 cycle)",
            "Projected event (2 cycles)", "Realized Event (1 cycle)", "Realized Event (2 cycles)" ]
    fontProp = matplotlib.font_manager.FontProperties( size = 8 )
    ax.legend(handle_ordered, label_ordered, loc = 1, prop = fontProp, bbox_to_anchor=(0.98,0.97))
    f.savefig(fname, bbox_inches='tight')
    return

def execute(start, end, v):
    sym_df = get_data(start, end, v)
    sym_agg = agg_symh_data(sym_df, v)
    sym_storms_df = storm_symh_data(sym_agg, v)
    crngtn_cycle_num, crngtn_dates = get_crnt_dates(start, end)
    print(" Cycle num:", crngtn_cycle_num)
    print(" Cycle date:", crngtn_dates)
    sym_storm_hour = pd.merge( sym_df.drop_duplicates(["date", "symh"]), sym_storms_df,
            left_on=["date", "symh"], right_on=["date", "symh_min"] )
    sym_storm_hour = sym_storm_hour[ (sym_storm_hour["pred_1cyc"] <= crngtn_dates[-1].date()) &
            (sym_storm_hour["pred_2cyc"] <= crngtn_dates[-1].date()) ]
    print(" Aggrigate storm hours:")
    print(sym_storm_hour.head())
    fname = "data/crngtn_" + str(start.year) + ".png"
    plot_data(sym_agg, sym_storms_df, sym_storm_hour, crngtn_dates, crngtn_cycle_num, fname)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", default=dt.datetime(2016,1,1), help="Start date",   
            type=dateutil.parser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2017,1,1), help="End date",
            type=dateutil.parser.isoparse)
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default False)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for processing ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    execute(args.start, args.end, args.verbose)
