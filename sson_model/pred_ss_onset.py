import keras
import datetime   
import json
import pandas
import numpy
import glob
from copy import deepcopy
import os
import sys
sys.path.append("../")
import dwnld_sw_imf_rt
import gauge_plot
import custom_cmap 
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.cm import ScalarMappable

class PredSSON(object):
    """
    A class to predict AMPERE FACs 
    based on the data input from 
    RT SW/IMF params
    """
    def __init__(self, model_name="model_paper",\
                 epoch=200, omn_pred_hist=120,\
                local_data_folder="sson_model/data/"):
        """
        Load the SW/IMF data and the mean input parameters
        used during training!
        We may also need to clean up the data!!
        """
        import pathlib
        # folder to store the data/predictions
        # this will be used so that we don't repeat
        # calculations
        import os
        local_data_folder = pathlib.Path.cwd().joinpath(local_data_folder)
        if pathlib.Path(local_data_folder).exists():
            self.local_data_folder = local_data_folder.as_posix() + "/"
        else:
            self.local_data_folder = pathlib.Path.cwd().parent.joinpath(local_data_folder)
            self.local_data_folder = self.local_data_folder.as_posix() + "/"
        # get/set some filenames
        self.latest_pred_data_file = self.local_data_folder + "latest_preds.npz"
        self.hist_pred_data_file = self.local_data_folder + "hist_preds.npz"
        # Load the RT SW/IMF data
        data_obj = dwnld_sw_imf_rt.DwnldRTSW()
        url_data = data_obj.dwnld_file()
        if url_data is not None:
            data_obj.read_url_data(url_data)
        self.sw_imf_df = data_obj.read_stored_data()
        self.original_sw_imf_data = deepcopy(self.sw_imf_df)
        # we need a 1-minute interval data!
        self.sw_imf_df.set_index('propagated_time_tag', inplace=True)
        self.sw_imf_df = self.sw_imf_df.resample('1min').median()
        # linearly interpolate data
        self.sw_imf_df.interpolate(method='linear', axis=0, inplace=True)
        # Now we need to normalize the input based on train data
        # Load mean and std values of input features from a json file
        inp_par_file_name = "input_mean_std.json"
        file_path = pathlib.Path(inp_par_file_name)
        if file_path.exists():
            inp_mean_std_file_path = file_path.as_posix()
        else:
            inp_mean_std_file_path = pathlib.Path.cwd().joinpath("amp_model",\
                                         inp_par_file_name).as_posix()
        with open(inp_mean_std_file_path) as jf:
            params_mean_std_dct = json.load(jf)
        # Load Vx
        self.sw_imf_df["Vx"] = -1.*self.sw_imf_df["speed"]
        # Normalize Vx
        self.sw_imf_df["Vx"] = (self.sw_imf_df["Vx"]-params_mean_std_dct["Vx_mean"])/\
                                    params_mean_std_dct["Vx_std"]
        # Load and Normalize Np
        self.sw_imf_df["Np"] = (self.sw_imf_df["density"]-params_mean_std_dct["Np_mean"])/\
                                        params_mean_std_dct["Np_std"]
        # Load and Normalize Bz
        self.sw_imf_df["Bz"] = (self.sw_imf_df["bz"]-params_mean_std_dct["Bz_mean"])/\
                                    params_mean_std_dct["Bz_std"]
        # Load and Normalize By
        self.sw_imf_df["By"] = (self.sw_imf_df["by"]-params_mean_std_dct["By_mean"])/\
                                    params_mean_std_dct["By_std"]
        # Load and Normalize Bx
        self.sw_imf_df["Bx"] = (self.sw_imf_df["bx"]-params_mean_std_dct["Bx_mean"])/\
                                    params_mean_std_dct["Bx_std"]
        
        self.omn_pred_hist = omn_pred_hist
        self.model_name = model_name
        self.epoch = epoch
        
    def int_ax_round(self, x, base=5):
        return base * round(x/base) + base

    def round_dt(self, dt, resolution):
        """
        round time to nearest resolution (in minutes)
        """
        new_minute = (dt.minute // resolution + 1) * resolution
        return dt + datetime.timedelta(minutes=new_minute - dt.minute)

    def load_model(self, used_model, epoch):

        """ 
        load the model
        """
        import glob
        import pathlib
        if not pathlib.Path(used_model).is_dir():
            used_model = pathlib.Path.cwd().joinpath("sson_model",used_model).as_posix()
        if epoch < 10:
            model_name = glob.glob(os.path.join(used_model, "weights.epoch_0" + str(epoch) + "*hdf5"))[0]
        else:
            model_name = glob.glob(os.path.join(used_model, "weights.epoch_" + str(epoch) + "*hdf5"))[0]    
        model = keras.models.load_model(model_name)
        return model
    
    def plot_daywise_preds(self, plot_date, al_df=None,del_time=60,\
             omn_train_params=["Bx", "By", "Bz", "Vx", "Np"],\
             time_extent=24, plot_type="bar", gen_pred_diff_time=30.):
        """
        Plot historical predictions as a bar graph!
        """
        import matplotlib.dates as mdates
        import sys
        sys.path.append("../")
        import extract_rt_al
        if al_df is None:
        # get AL data
            al_obj = extract_rt_al.ExtractAL()
            al_df = al_obj.get_al_data()
        al_df.sort_values(by=['date'], inplace=True)
        plot_title = plot_date.strftime("%B %d, %Y")
        # set the plots based on the extent chosen
        plot_min_date = datetime.datetime(plot_date.year, plot_date.month, plot_date.day, 0, 0)    
        plot_max_date = datetime.datetime(plot_date.year, plot_date.month, plot_date.day, 0, 0) + datetime.timedelta(days=1)
        plot_xlim = [ 
                        plot_min_date,\
                        plot_max_date,\
                         ]
        # now read stored data (if we have it)
        generate_predictions = True
        if os.path.isfile(self.hist_pred_data_file):
            stored_data = numpy.load(self.hist_pred_data_file, allow_pickle=True)
            stored_date_arr = stored_data['date_arr']
            stored_sson_prob_arr = stored_data['sson_prob_arr']
            if ( plot_min_date >= stored_date_arr.min() ) and\
                    ( plot_max_date <= stored_date_arr.max() ):
                generate_predictions = False
                date_arr = stored_date_arr
                sson_prob_arr = stored_sson_prob_arr
            else:
                if (plot_date - stored_date_arr.max()).total_seconds()/60. < gen_pred_diff_time:
                    generate_predictions = False
                    date_arr = stored_date_arr
                    sson_prob_arr = stored_sson_prob_arr
        if generate_predictions:
            date_arr, sson_prob_arr = self.calculate_historical_preds(\
                                        del_time=del_time,\
                                        omn_train_params=omn_train_params
                                        )

        hist_pred_date_lim = [ min(date_arr), max(date_arr) ]
        # check if we have data
        if ( plot_date < plot_min_date ) or ( plot_date > plot_max_date ):
            return None
        date_formatter = DateFormatter('%H:%M')
        cmap = custom_cmap.create_custom_cmap()#plt.cm.get_cmap('Reds')
        norm = Normalize(vmin=0, vmax=1)
        colors = cmap(norm(sson_prob_arr))
        # finally!!!
        plt.style.use("fivethirtyeight")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax2 = ax.twinx()
        if plot_type == "filled_curve":
            hist_plot = ax.fill_between(date_arr, 0, sson_prob_arr,cmap=cmap)
        else:
            # The unit for bar width on a date x axis is days
            bar_width = del_time/(60.*24.)
            plt_date_arr = [x.to_pydatetime() for x in date_arr] 
            hist_plot = ax.bar(plt_date_arr,sson_prob_arr, width=bar_width,\
                     alpha=0.7, align="edge",\
                     color=colors)
            # colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = plt.colorbar(sm)
            cbar.ax.tick_params(labelsize=8) 
        cbar.set_label('Ponset', fontsize=10)
        # plot a horizontal line at 0.5
#         ax.axhline(y=0., color='#018571', linestyle='-',\
#                         linewidth=1.5)
#         ax.axhline(y=0.25, color='#80cdc1', linestyle='-',\
#                         linewidth=1.5)
        ax.axhline(y=0.5, color='k', linestyle='-',\
                        linewidth=1.)##fdb863
#         ax.axhline(y=0.75, color='#d7191c', linestyle='-',\
#                         linewidth=1.5)
        ax2.plot_date(al_df["date"], al_df["al"],\
                  '--',color='#008fd5', linewidth=1.)
        ax2.set_ylim([-2000,100])
        ax2.set_yticks([-2000, -1000, -500, 0])
        ax2.tick_params(axis='y', colors='#008fd5')
        # Hide grid lines
        ax2.grid(linestyle=':', linewidth='1.', color='#008fd5')
        # axes settings
        ax.set_ylim([0,1])
        ax.set_xlim(plot_xlim)
        ax.get_xaxis().set_major_formatter(date_formatter)
        ax.set_ylabel("Prob. of Onset", fontsize=12)
        ax.set_xlabel('UT TIME', fontsize=12)
        # set minor ticks
        ax.xaxis.set_major_locator(mdates.HourLocator(interval = 3))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval = 1))
        ax.grid(b=True, which='minor', linestyle=':')
#         plt.xticks(rotation=60, fontsize=10)
        plt.setp( ax.yaxis.get_majorticklabels(), rotation=70, fontsize=10 )
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=70, fontsize=10 )
        plt.setp( ax2.yaxis.get_majorticklabels(), rotation=45, fontsize=6 )
        plt.title(plot_title, fontsize=10)
#         plt.yticks(fontsize=10)
        plt.tight_layout()
        return fig

    def plot_historical_preds(self, del_time=60,\
             omn_train_params=["Bx", "By", "Bz", "Vx", "Np"],\
             time_extent=24, plot_type="bar"):
        """
        Plot historical predictions as a bar graph!
        """
        import sys
        sys.path.append("../")
        import extract_rt_al
#         # get AL data
        al_obj = extract_rt_al.ExtractAL()
        al_df = al_obj.get_al_data()
        al_df.sort_values(by=['date'], inplace=True)
        # now get to substorms
        date_arr, sson_prob_arr = self.calculate_historical_preds(\
                                    del_time=del_time,\
                                    omn_train_params=omn_train_params
                                    )
        # set the plots based on the extent chosen
        plot_xlim = [ 
                        max(date_arr) - datetime.timedelta(minutes=time_extent*60),\
                        max(date_arr) + datetime.timedelta(minutes=180)
                         ]
        if time_extent <= 24:
            date_formatter = DateFormatter('%H:%M')
        else:
            date_formatter = DateFormatter('%m/%d-%H')
        cmap = custom_cmap.create_custom_cmap()#plt.cm.get_cmap('Reds')
        norm = Normalize(vmin=0, vmax=1)
        colors = cmap(norm(sson_prob_arr))
        # finally!!!
        plt.style.use("fivethirtyeight")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax2 = ax.twinx()
        if plot_type == "filled_curve":
            hist_plot = ax.fill_between(date_arr, 0, sson_prob_arr,cmap=cmap)
        else:
            # The unit for bar width on a date x axis is days
            bar_width = del_time/(60.*24.)
            hist_plot = ax.bar(date_arr,sson_prob_arr, width=bar_width,\
                     alpha=0.7, align="edge",\
                     color=colors)
            # colorbar
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            cbar = plt.colorbar(sm)
            cbar.ax.tick_params(labelsize=8) 
        cbar.set_label('Ponset', fontsize=10)
        # plot a horizontal line at 0.5
#         ax.axhline(y=0., color='#018571', linestyle='-',\
#                         linewidth=1.5)
#         ax.axhline(y=0.25, color='#80cdc1', linestyle='-',\
#                         linewidth=1.5)
        ax.axhline(y=0.5, color='k', linestyle='-',\
                        linewidth=1.)##fdb863
#         ax.axhline(y=0.75, color='#d7191c', linestyle='-',\
#                         linewidth=1.5)
        ax2.plot_date(al_df["date"], al_df["al"],\
                  '--',color='#008fd5', linewidth=1.)
        ax2.set_ylim([-2000,100])
        ax2.set_yticks([-2000, -1000, -500, 0])
        ax2.tick_params(axis='y', colors='#008fd5')
        # Hide grid lines
        ax2.grid(linestyle=':', linewidth='1.', color='#008fd5')
        # axes settings
        ax.set_ylim([0,1])
        ax.set_xlim(plot_xlim)
        ax.get_xaxis().set_major_formatter(date_formatter)
        ax.set_ylabel("Prob. of Onset", fontsize=12)
        ax.set_xlabel('UT TIME', fontsize=12)
#         plt.xticks(rotation=60, fontsize=10)
        plt.setp( ax.yaxis.get_majorticklabels(), rotation=70, fontsize=10 )
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=70, fontsize=10 )
        plt.setp( ax2.yaxis.get_majorticklabels(), rotation=45, fontsize=6 )
#         plt.yticks(fontsize=10)
        plt.tight_layout()
        return fig

    def calculate_historical_preds(self, del_time=30,\
             omn_train_params=["Bx", "By", "Bz", "Vx", "Np"]):
        """
        Generate predictions every 30 minutes based on solarwind
        and IMF data available.
        """
        # get nearest time rounded to 30 minutes
        min_time = self.round_dt(\
                                self.sw_imf_df.index.min() +\
                                 datetime.timedelta(minutes=self.omn_pred_hist),\
                                del_time\
                                 )
        max_time = self.sw_imf_df.index.max()
        date_arr = []
        sson_prob_arr = []
        model = self.load_model(self.model_name, self.epoch)
        while min_time < max_time:
            _sson_prob_enc = self.generate_sson_pred(\
                                 min_time, model=model,\
                                 omn_train_params=omn_train_params
                                                )
            sson_prob_arr.append( _sson_prob_enc )
            date_arr.append( min_time )
            min_time += datetime.timedelta(minutes=del_time)
        # save the predictions
        numpy.savez(\
                        self.hist_pred_data_file,\
                        date_arr=date_arr,\
                        sson_prob_arr=sson_prob_arr\
                       )
        return date_arr, sson_prob_arr

    def generate_latest_sson_pred(self,\
             omn_train_params=["Bx", "By", "Bz", "Vx", "Np"],\
            gen_pred_diff_time=20):
        """
        Get the latest prediction!
        """
        omn_end_time = self.sw_imf_df.index.max()
        omn_begin_time = (omn_end_time - datetime.timedelta(\
                    minutes=self.omn_pred_hist) ).strftime(\
                    "%Y-%m-%d %H:%M:%S")
        generate_new_pred = True
        if os.path.isfile(self.latest_pred_data_file):
            stored_data = numpy.load(self.latest_pred_data_file, allow_pickle=True)
            stored_sson_pred_enc = stored_data['sson_pred_enc'][0]
            stored_omn_begin_time = stored_data['omn_begin_time'][0]
            stored_omn_end_time = stored_data['omn_end_time'][0]
            # now check if the data is not stale!
            if (omn_end_time - stored_omn_end_time).total_seconds()/60. > gen_pred_diff_time:
                generate_new_pred = True
            else:
                generate_new_pred = False
                sson_pred_enc = stored_sson_pred_enc
                omn_begin_time = stored_omn_begin_time
                omn_end_time = stored_omn_end_time
        # now we'll generate the data since the stored
        # data doesn't work!
        if generate_new_pred:
            sson_pred_enc = self.generate_sson_pred(\
                                     omn_end_time,\
                                     omn_train_params=omn_train_params
                                                )
            numpy.savez(\
                        self.latest_pred_data_file,\
                        sson_pred_enc=[sson_pred_enc],\
                        omn_begin_time=[omn_begin_time],\
                        omn_end_time=[omn_end_time]\
                       )
        return sson_pred_enc, omn_begin_time, omn_end_time


    def generate_sson_pred(self, pred_time, model=None,\
             omn_train_params=["Bx", "By", "Bz", "Vx", "Np"]):
        """
        Get the predictions!
        """
        # Load the model
        if model is None:
            model = self.load_model(self.model_name, self.epoch)
        # make the preds
        omn_begin_time = (pred_time - datetime.timedelta(\
                    minutes=self.omn_pred_hist) ).strftime(\
                    "%Y-%m-%d %H:%M:%S")
        inp_omn_vals = self.sw_imf_df.loc[\
                         omn_begin_time : pred_time \
                         ][omn_train_params].values
        inp_omn_vals = inp_omn_vals.reshape(1,inp_omn_vals.shape[0],\
                             inp_omn_vals.shape[1])
        sson_pred_enc = model.predict(inp_omn_vals, batch_size=1)
        sson_pred_enc = sson_pred_enc[0].round(2)
        return sson_pred_enc[1]

    def generate_gauge_plot(self,sson_prob):
        # we'll have the following substorm label encoding
        # 0-0.25 : Highly unlikely
        # 0.25-0.5 : Unlikely
        # 0.5-0.75 : Possibility of substorms
        # 0.75-1. : High likelihood of substorms
        ss_labels = [ "Very unlikely",\
                     "Unlikely",\
                    "Likely",\
                     "Very likely" ]
        # get arrow location
        if sson_prob <= 0.25:
            arrow_loc = 1
        elif (sson_prob > 0.25) & (sson_prob <= 0.5):
            arrow_loc = 2
        elif (sson_prob > 0.5) & (sson_prob <= 0.75):
            arrow_loc = 3
        else:
            arrow_loc = 4

        title_str = "Possibility of substorm onset"
        gauge_fig = gauge_plot.gauge(labels=ss_labels, \
                    colors=['#018571', '#80cdc1', '#fdb863', '#d7191c'],\
                    arrow=arrow_loc, title=title_str) 
        return gauge_fig


    def generate_bin_plot(self, gen_pred_diff_time=20,\
                omn_train_params=["Bx", "By", "Bz", "Vx", "Np"]):
        """
        Generate the plot.
        """
        # get the plot details
        # get the time range of the plot
        sson_prob, omn_begin_time, omn_end_time = self.generate_latest_sson_pred(\
                                        omn_train_params=omn_train_params,\
                                        gen_pred_diff_time=20)
        plotTimeRange = [ omn_begin_time,\
                     omn_end_time + datetime.timedelta(minutes=60)]
        # manipulate original sw imf data for plotting
        # rename some parameters
        self.original_sw_imf_data["Vx"] = -1.*self.original_sw_imf_data["speed"]
        self.original_sw_imf_data.rename( columns={"density" : "Np", "bx" : "Bx", "by":"By", "bz":"Bz"},\
                                         inplace=True )
        # set the time as selected
        self.original_sw_imf_data.set_index('propagated_time_tag', inplace=True)
        # select the period of interest
        plot_df = self.original_sw_imf_data.loc[\
             str(omn_begin_time) : str(omn_end_time) \
             ][omn_train_params]
        plot_df["datetime"] = plot_df.index
        # linearly interpolate data
        plot_df.interpolate(method='linear', axis=0, inplace=True)
        # set plot styling
        plt.style.use("fivethirtyeight")
        # get the number of panels
        nPanels = len(omn_train_params)
        fig, axes = plt.subplots(nrows=nPanels, ncols=1,\
                                 figsize=(8,8), sharex=True)
        # axis formatting
        dtLabFmt = DateFormatter('%H:%M')
        axCnt = 0
        # have a dict to set up a axis limit scale
        ax_lim_dict = {
                        "Bz" : [-1,5], 
                        "By" : [-1,5],
                        "Bx" : [-1,5],
                        "Vx" : [0,100],
                        "Np" : [0,5]
                      }
        # plot omni IMF
        for _op in [ "Bz", "By", "Bx", "Vx", "Np" ]:
            axes[axCnt].plot( plot_df["datetime"].values,\
                          plot_df[_op].values, linewidth=2 )
            axes[axCnt].set_ylabel(_op, fontsize=14)
            axes[axCnt].xaxis.set_major_formatter(dtLabFmt)
            # set axis limits
            _axlim_base = ax_lim_dict[_op][1]
            _axlim_low_limit = ax_lim_dict[_op][0]
            # get the max min limit
            _max_val = max( [ numpy.abs(plot_df[_op].min()),\
                             numpy.abs(plot_df[_op].max()) ] ) 
            _max_rnd = self.int_ax_round( _max_val, base=_axlim_base )
            if _op == "Vx":
                axes[axCnt].set_yticks(\
                    numpy.arange(-1*_max_rnd, _max_rnd, 250) )
                axes[axCnt].set_ylim( [ -1*_max_rnd, _axlim_low_limit ] )
            elif _op == "Np":
                axes[axCnt].set_yticks(\
                    numpy.arange(_axlim_low_limit*_max_rnd, _max_rnd, 5) )
                axes[axCnt].set_ylim( [ _axlim_low_limit*_max_rnd, _max_rnd ] )
            else:
                axes[axCnt].set_yticks(\
                    numpy.arange(_axlim_low_limit*_max_rnd, _max_rnd, 5) )
                axes[axCnt].set_ylim( [ _axlim_low_limit*_max_rnd, _max_rnd ] )
                axes[axCnt].axhline(y=0, color='k', linestyle='--',\
                        linewidth=1.)
            axCnt += 1
        # shade the region based on the type (TP/TN/FP/FN)
        for _nax, _ax in enumerate(axes):
            # plot vertical lines to mark the prediction bins
            binStart = omn_end_time 
            _ax.axvline(x=binStart, color='k', linestyle='--',\
                        linewidth=0.5)
            binEnd = omn_end_time + datetime.timedelta(minutes=60)
            text_sson = "Ponset = " + str(sson_prob)
            if sson_prob <= 0.25:
                pred_col = '#018571'
            elif (sson_prob > 0.25) & (sson_prob <= 0.5):
                pred_col = '#80cdc1'
            elif (sson_prob > 0.5) & (sson_prob <= 0.75):
                pred_col = '#fdb863'
            else:
                pred_col = '#d7191c' 
            _ax.axvspan(binStart, binEnd, alpha=0.4, color=pred_col)
            if _nax == 0 :
                text_xloc = omn_end_time + datetime.timedelta(\
                        minutes=10) 
                text_yloc = _ax.get_ylim()[1] - (\
                     abs(_ax.get_ylim()[1]) - _ax.get_ylim()[0] )/2
                _ax.text(text_xloc, text_yloc, text_sson)
        plt.tight_layout()
        return sson_prob, fig, omn_begin_time, omn_end_time

if __name__ == "__main__":
    pred_obj = PredSSON()
    fig = pred_obj.generate_gauge_plot(0.4)#plot_historical_preds()#generate_bin_plot()
    fig.savefig("../data/test_al.png")