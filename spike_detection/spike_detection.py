#!/usr/bin/env python

"""spike_detection.py: Spike detection in ASYM-H, SYM-H, AE, AL and AU"""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = ""
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import matplotlib
matplotlib.use("Agg")
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.dates as mdates

import os
import datetime as dt
import argparse
import dateutil

def parse(l):
    """
    Parse data into CSV file
    """
    y = {}
    y["datetime"] = dt.datetime.strptime(l[0]+" "+l[1], "%Y-%m-%d %H:%M:%S.%f")
    y["doy"], y["asy-d"], y["asy-h"], y["sym-d"], y["sym-h"] = l[2], l[3], l[4], l[5], l[6]
    return y

def fetch_symh():
    ## TODO
    uri = "http://wdc.kugi.kyoto-u.ac.jp/cgi-bin/aeasy-cgi?Tens=202&Year=0&Month=01&Day_Tens=0&Days=0&Hour=00&min=00&Dur_Day_Tens=300&Dur_Day=1&Dur_Hour=00&Dur_Min=00&Image+Type=GIF&COLOR=COLOR&AE+Sensitivity=0&ASY%2FSYM++Sensitivity=0&Output=ASY&Out+format=IAGA2002&Email=shibaji7%40vt.edu"
    print("wget -O {fnm} \"{uri}\"".format(fnm="symh.txt", uri=uri))
    os.system("wget -O {fnm} \"{uri}\"".format(fnm="symh.txt", uri=uri))
    return

def parse_data():
    parse_data = False
    if parse_data: fetch_symh()
    with open("symh.txt") as f:
        h = None
        d = []
        lines = f.readlines()
        for i, l in enumerate(lines):
            if i >= 14:
                l = list(filter(None, l.replace("\n", "").split(" ")))
                if i == 14: head = l[:-1]
                else: d.append(parse(l))
        u = pd.DataFrame.from_dict(d)
        u.to_csv("sym-h.csv", index=False)
        print(u.head())
    return

def neo(x):
    """
    NEO(x): diff(x,2)-x.diff(x,1)
    Implemented from "Holleman2011.Chapter.SpikeDetection_Characterization"
    """
    y = np.gradient(x,edge_order=1)**2 - (np.gradient(x,edge_order=2)*x)
    return y

def smooth(x,window_len=51,window="hanning"):
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y

def cascading_neo_calculator(x, th=5, order=3, lag=101):
    y = smooth(x, lag)
    N, _I = [], []
    N.append(np.array(x))
    N.append(y)
    for i in range(order):
        N.append(neo(N[-1]))
    n = N[-1]
    _I = x[np.abs(n) >= th]
    if len(_I) > 0: _I = True
    else: _I = False
    return N, _I

def plot_data(N, x=None, threshold=5, ylab="ASYM-H", order=3):
    fig, axes = plt.subplots(figsize=(8,6), nrows=2, ncols=1)
    ax = axes[0]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M-%d"))
    ax.plot(x, N[0], lw=0.8, ls = "--", color="r")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(r"$y\sim$"+ylab)
    ax.set_title("Date: "+x[0].strftime("%Y-%m-%d"))
    ax = axes[1]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M-%d"))
    ax.plot(x, N[-1], lw=0.8, ls = "--")
    ax.axhline(threshold, color="green", lw=0.4, ls = "-")
    ax.axhline(-threshold, color="green", lw=0.4, ls = "-")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(r"$NEO^%d(y)$"%order)
    fig.autofmt_xdate()
    fig.savefig("plots/"+x[0].strftime("%Y-%m-%d")+".png", bbox_inches="tight")
    plt.close()
    return

def run(args):
    df = pd.read_csv("sym-h.csv", parse_dates=["datetime"])
    start, end = args.start, args.end
    order, lag, threshold = args.order, args.win, args.thresh
    days = (end - start).days + 1
    date_list = []
    for d in range(days):   
        s, e = start + dt.timedelta(days=d), start + dt.timedelta(days=d+1)
        du = df[(df.datetime >= s) & (df.datetime < e)]
        try:
            N, _I = cascading_neo_calculator(np.array(du["asy-h"]), th=threshold, order=order, lag=lag)
            if _I: date_list.append(s)
            plot_data(N, [x.to_pydatetime() for x in du.datetime], order=order, threshold=threshold)
        except: print("Error in filtering...", s)
        #break
    u = pd.DataFrame()
    u["dates"] = date_list
    u.to_csv("dates.csv", index=False)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--param", default="asym-h", help="Parameter code")
    parser.add_argument("-s", "--start", default=dt.datetime(2020,1,1), help="Start date",
            type=dateutil.parser.isoparse)
    parser.add_argument("-e", "--end", default=dt.datetime(2020,6,30), help="End date",
            type=dateutil.parser.isoparse)
    parser.add_argument("-t", "--thresh", type=float, default=5, help="Threshold value for filterig")
    parser.add_argument("-o", "--order", type=int, default=2, help="Order of the Filter")
    parser.add_argument("-w", "--win", type=int, default=101, help="Window of the Filter")
    parser.add_argument("-v", "--verbose", action="store_false", help="Increase output verbosity (default True)")
    parser.add_argument("-c", "--parse", action="store_true", help="Parse data (default False)")
    args = parser.parse_args()
    if args.verbose:
        print("\n Parameter list for simulation ")
        for k in vars(args).keys():
            print("     ", k, "->", vars(args)[k])
    if args.parse: parse_data()
    run(args)
    print("")
