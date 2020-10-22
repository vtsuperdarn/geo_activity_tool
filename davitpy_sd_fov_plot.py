import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import datetime as dt
from davitpy.utils import plotUtils
from davitpy.utils.plotUtils import textHighlighted
from davitpy.pydarn.plotting.mapOverlay import overlayRadar, overlayFov


dateTime = dt.datetime(2015, 1, 1)
fig = plt.figure(figsize=(10,5), dpi=300)
ax = fig.add_subplot(121)
m = plotUtils.mapObj(projection='npstere', lon_0=270, boundinglat=30,
        coords="geo", dateTime=dateTime, ax=ax, fill_alpha=.7, )
codes = ['bks', 'fhe','fhw','cve','cvw',"wal", "ade", "adw", "hok", "hkw"]
overlayFov(m, codes=codes, dateTime=dateTime, fovColor="darkorange", fovAlpha=0.4, maxGate=75, lineWidth=0.5)
codes = ['kap','sas','pgr',"gbr", "ksr","kod","sto","pyk","han"]
overlayFov(m, codes=codes, dateTime=dateTime, fovColor="aqua", fovAlpha=0.4, maxGate=75, lineWidth=0.5)
codes = ["inv","cly","rkn","lyr"]
overlayFov(m, codes=codes, dateTime=dateTime, fovColor="lime", fovAlpha=0.4, maxGate=75, lineWidth=0.5)

codes = ['bks', 'fhe','fhw','cve','cvw',"wal", "ade", "adw", "hok", "hkw"]
overlayRadar(m, fontSize=30, codes=codes, dateTime=dateTime)
codes = ['kap','sas','pgr',"gbr", "ksr","kod","sto","pyk","han"]
overlayRadar(m, fontSize=30, codes=codes, dateTime=dateTime)
codes = ["inv","cly","rkn","lyr"]
overlayRadar(m, fontSize=30, codes=codes, dateTime=dateTime)
ax.set_title("Northern Hemisphere", fontdict={"size":20,})

ax = fig.add_subplot(122)
m = plotUtils.mapObj(projection='spstere', lon_0=270, boundinglat=-35,
                coords="geo", dateTime=dateTime, ax=ax, fill_alpha=.7, )
codes = ["bpk", "tig", "unw", "fir"]
overlayRadar(m, fontSize=30, codes=codes, dateTime=dateTime)
overlayFov(m, codes=codes, dateTime=dateTime, fovColor="darkorange", fovAlpha=0.4, maxGate=75, lineWidth=0.5)
codes = ["ker","zho","sye","sys","sps","san","dce"]
overlayRadar(m, fontSize=30, codes=codes, dateTime=dateTime)
overlayFov(m, codes=codes, dateTime=dateTime, fovColor="aqua", fovAlpha=0.4, maxGate=75, lineWidth=0.5)
codes = ["mcm"]
overlayRadar(m, fontSize=30, codes=codes, dateTime=dateTime)
overlayFov(m, codes=codes, dateTime=dateTime, fovColor="lime", fovAlpha=0.4, maxGate=75, lineWidth=0.5)
ax.set_title("Southern Hemisphere", fontdict={"size":20,})

fig.savefig("data/overview.png", bbox_inches="tight")
