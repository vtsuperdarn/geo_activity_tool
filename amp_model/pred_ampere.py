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
from custom_loss_functions import mse, rmse, mae, cce, mae_med_jr, rmse_med_jr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates

class PredAMP(object):
    """
    A class to predict AMPERE FACs 
    based on the data input from 
    RT SW/IMF params
    """
    def __init__(self, model_name="b0fee2e6_20190921-12:56:55",\
            epoch=15, results_mapper_file="mapper.txt",\
            inp_au=240, inp_al=-240, inp_symh=-10,\
            inp_asymh=15, inp_f107=115, use_manual_bz=False,\
            use_manual_by=False, use_manual_bx=False,\
            use_manual_vx=False, use_manual_np=False,\
            use_manual_month=False, manual_bz_val=None,\
            manual_by_val=None, manual_bx_val=None,\
            manual_vx_val=None, manual_np_val=None,\
            manual_mnth_val=None):
        """
        Load the SW/IMF data and the mean input parameters
        used during training!
        We may also need to clean up the data!!
        """
        import pathlib
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
        # Note When training hte model we had Vx as negative! but in real time data
        # it is used as a positive value! so change the sign
        # we need a switch to tell us if we've used manual inputs!
        # However, ignore the season for manual input part
        # since we can see how the plots vary with season
        # for same IMF/solarwind data!
        self.used_manual_inputs = False
        if use_manual_vx:
            self.sw_imf_df["Vx"] = manual_vx_val
            self.used_manual_inputs = True
        else:    
            self.sw_imf_df["Vx"] = -1.*self.sw_imf_df["speed"]
        # Normalize
        self.sw_imf_df["Vx"] = (self.sw_imf_df["Vx"]-params_mean_std_dct["Vx_mean"])/\
                                    params_mean_std_dct["Vx_std"]
        if use_manual_np:
            self.sw_imf_df["Np"] = (manual_np_val-params_mean_std_dct["Np_mean"])/\
                                    params_mean_std_dct["Np_std"]
            self.used_manual_inputs = True
        else:
            self.sw_imf_df["Np"] = (self.sw_imf_df["density"]-params_mean_std_dct["Np_mean"])/\
                                        params_mean_std_dct["Np_std"]
        if use_manual_bz:
            self.sw_imf_df["Bz"] = (manual_bz_val-params_mean_std_dct["Bz_mean"])/\
                                        params_mean_std_dct["Bz_std"]
            self.used_manual_inputs = True
        else:
            self.sw_imf_df["Bz"] = (self.sw_imf_df["bz"]-params_mean_std_dct["Bz_mean"])/\
                                        params_mean_std_dct["Bz_std"]
        if use_manual_by:
            self.sw_imf_df["By"] = (manual_by_val-params_mean_std_dct["Bz_mean"])/\
                                        params_mean_std_dct["Bz_std"]
            self.used_manual_inputs = True
        else:
            self.sw_imf_df["By"] = (self.sw_imf_df["by"]-params_mean_std_dct["By_mean"])/\
                                        params_mean_std_dct["By_std"]
        if use_manual_bx:
            self.sw_imf_df["Bx"] = (manual_bx_val-params_mean_std_dct["Bz_mean"])/\
                                        params_mean_std_dct["Bz_std"]
            self.used_manual_inputs = True
        else:
            self.sw_imf_df["Bx"] = (self.sw_imf_df["bx"]-params_mean_std_dct["Bx_mean"])/\
                                        params_mean_std_dct["Bx_std"]
        if use_manual_month:
            self.sw_imf_df["month_sine"] = numpy.sin(2*numpy.pi/12 *\
                                         manual_mnth_val)
            self.sw_imf_df["month_cosine"] = numpy.cos(2*numpy.pi/12 *\
                                         manual_mnth_val)
        else:
            self.sw_imf_df["month_sine"] = numpy.sin(2*numpy.pi/12 *\
                                         self.sw_imf_df.index.month)
            self.sw_imf_df["month_cosine"] = numpy.cos(2*numpy.pi/12 *\
                                         self.sw_imf_df.index.month)
        self.sw_imf_df["au"] = (inp_au-params_mean_std_dct["au_mean"])/\
                                    params_mean_std_dct["au_std"]
        self.sw_imf_df["al"] = (inp_al-params_mean_std_dct["al_mean"])/\
                                    params_mean_std_dct["al_std"]
        self.sw_imf_df["symh"] = (inp_symh-params_mean_std_dct["symh_mean"])/\
                                    params_mean_std_dct["symh_std"]
        self.sw_imf_df["asymh"] = (inp_asymh-params_mean_std_dct["asymh_mean"])/\
                                    params_mean_std_dct["asymh_std"]
        self.sw_imf_df["F107"] = (inp_f107-params_mean_std_dct["F107_mean"])/\
                                    params_mean_std_dct["F107_std"]                                    
        # print("solar wind imf data!!!")
        # print(self.sw_imf_df.columns)
        # print("solar wind imf data!!!")
        # Load the params
        res_map_file_path = pathlib.Path(results_mapper_file)
        if res_map_file_path.exists():
            res_map_file_path = res_map_file_path.as_posix()
        else:
            res_map_file_path = pathlib.Path.cwd().joinpath("amp_model",\
                                         results_mapper_file).as_posix()
        with open(res_map_file_path) as jf:
            self.params_dict = json.load(jf)
            self.params_dict = self.params_dict["../data/trained_models/" +\
                                     model_name]
        # load the reference numpy array to reconstruct mlat/mlt locs of jr
        self.ref_amp_df_nth = self._load_sample_csv_file()[0]
        self.ref_amp_df_sth = self._load_sample_csv_file()[1]
        # Load the model
        self.model = self.load_model(model_name, epoch)

    def _load_sample_csv_file(self,\
         nth_csv_file="ref_amp_frame.csv",\
         sth_csv_file="ref_amp_frame_south.csv"):
        """
        Here we'll load a sample raw frame from a csv file
         This is for converting back the numpy files to a
         dataframe without loosing mlat/mlt information. This way
         we are using the exact reverse process through which we 
         created the numpy files! and avoid using xarray
        """
        import pathlib
        nth_file_path = pathlib.Path(nth_csv_file)
        sth_file_path = pathlib.Path(sth_csv_file)
        if nth_file_path.exists():
            nth_file_path = nth_file_path.as_posix()
            sth_file_path = sth_file_path.as_posix()
        else:
            nth_file_path = pathlib.Path.cwd().joinpath("amp_model",\
                                         nth_csv_file).as_posix()
            sth_file_path = pathlib.Path.cwd().joinpath("amp_model",\
                                         sth_csv_file).as_posix()
        return (pandas.read_csv(nth_file_path, index_col=0),\
                 pandas.read_csv(sth_file_path, index_col=0))


    def load_model(self, used_model, epoch):

        """ 
        load the model
        """
        import pathlib
        loss_dct = {"mse":mse,
                    "rmse":rmse,
                    "mae": mae,
                    "cce":cce,
                    "mae_med_jr":mae_med_jr,
                    "rmse_med_jr":rmse_med_jr}
        model_dir_name = used_model
        if not pathlib.Path(model_dir_name).is_dir():
            model_dir_name = pathlib.Path.cwd().joinpath("amp_model",\
                                         used_model).as_posix()
        if epoch < 10:
            model_name = glob.glob(os.path.join(model_dir_name, "weights.epoch_0" +\
                         str(epoch) + "*hdf5"))[0]
        else:
            model_name = glob.glob(os.path.join(model_dir_name, "weights.epoch_" +\
                         str(epoch) + "*hdf5"))[0]
        model = keras.models.load_model(model_name, custom_objects=loss_dct)
        return model


    def get_unique_pred_times(self, max_time_limit=3, pred_time_interval=10):
        """
        get the predictions over the previous 3 (max_time_limit)
        hours or less if data is not availeble,
        starting from the latest available time stamp!
        """
        uniq_date_list = []
        newest_date = self.sw_imf_df.index.max()
        # lets round to nearest 10th minute
        newest_date = newest_date - datetime.timedelta(minutes=newest_date.minute % 10,
                             seconds=newest_date.second,
                             microseconds=newest_date.microsecond)
        oldest_date = self.sw_imf_df.index.min()
        if newest_date - datetime.timedelta(hours=max_time_limit) >= \
                (oldest_date + datetime.timedelta(minutes=\
                    self.params_dict["general_params"]["omn_history"])):
            oldest_pred_date = newest_date - datetime.timedelta(hours=max_time_limit)
        else:
            oldest_pred_date = oldest_date +\
                            datetime.timedelta(minutes=\
                            self.params_dict["general_params"]["omn_history"])
        # generate a pred date list
        _curr_date = newest_date
        while _curr_date >= oldest_pred_date:
            uniq_date_list.append( _curr_date )
            _curr_date -= datetime.timedelta(minutes=pred_time_interval)
        return uniq_date_list

    def generate_amp_pred(self, input_times, output_map_shape = [50, 24]):
        """
        Given an input time generate prediction
        """
        if not isinstance(input_times, (list, numpy.array)):
            input_times = [input_times]
        amp_df_list = []
        for _inp_time in input_times:
            # print(_inp_time)
            input_params = deepcopy(self.params_dict["general_params"]["omn_train_params"])
            # add the other params!
            if self.params_dict["model_params"]["sml_as_input"]:
                input_params = input_params + ["au", "al"]
            else:
                print("The model is trained with sml_as_inpu=False"+\
                      ", therefore 'SMU and SML' features are ignored.") 
            if self.params_dict["model_params"]["symh_as_input"]:
                input_params = input_params + ["symh", "asymh"]
            else:
                print("The model is trained with symh_as_inpu=False"+\
                      ", therefore 'SYMH and ASYMH' features are ignored.") 
            if self.params_dict["model_params"]["f107_as_input"]:
                input_params = input_params + ["F107"]
            else:
                print("The model is trained with f107_as_inpu=False"+\
                      ", therefore 'f107' feature is ignored.") 
            # add month params
            input_params += [ "month_sine", "month_cosine" ]
            min_time = _inp_time-datetime.timedelta(\
                            minutes=self.params_dict[\
                                "general_params"]["omn_history"] - 1)
            model_inputs = self.sw_imf_df.loc[ min_time : _inp_time ][\
                                input_params].values
            model_inputs = model_inputs.reshape(1,model_inputs.shape[0],\
                             model_inputs.shape[1])
            # Make predictions
            amp_pred = self.model.predict(model_inputs, batch_size=1)
            amp_pred = amp_pred[0].reshape(output_map_shape[0], output_map_shape[1])
            self.ref_amp_df_nth[self.ref_amp_df_nth.columns] = amp_pred
            # load data into a df
            _amp_df = self.ref_amp_df_nth.unstack().reset_index(\
                            name='pred_jr')
            # if we are reading from the csv file we need to change
            # the column name for mlt, for some reason I'm not able 
            # to store it when creating the ref file
            _amp_df.rename({'level_0': 'mlt'}, axis=1, inplace=True)
            _amp_df["mlt"] = _amp_df["mlt"].astype("float")
            _amp_df["date"] = _inp_time
            amp_df_list.append(_amp_df)
        if len(amp_df_list) > 0:
            amp_df = pandas.concat(amp_df_list)
            return amp_df
        else:
            return None

    def generate_plots(self, uniq_dates, amp_data,\
                     nplots=6, n_plot_cols=2):
        """
        Generate plots for the given times!
        """
        # first note down which files are present in the folder
        # after that generate new one's and delete older one's 
        # finally!
        # 
        # plot AMPERE predictions for manual inputs
        if self.used_manual_inputs:
            fig, ax = plt.subplots(nrows=1,\
                                ncols=1, figsize=(11, 10), 
                               subplot_kw={"projection":"polar"})
            _plt = self.poly_plot(fig, ax, uniq_dates[0],\
                                 amp_data, plot_cbar=True,\
                                 plt_title="Manual Inputs",\
                                 cax = fig.add_axes([0.9, 0.30, 0.02, 0.4]))
        else: 
            # Plot the realtime AMPERE predictions
            fig, axes = plt.subplots(nrows=int(nplots/n_plot_cols),\
                                ncols=n_plot_cols, figsize=(9, 11), 
                               subplot_kw={"projection":"polar"})
            plt.subplots_adjust(hspace=0.25, wspace=0.1)
            axes = axes.flatten()
            for _ind in range(nplots):
                _ax = axes[_ind]
                _date = uniq_dates[_ind]
                if _ind < nplots-1:
                    _plt = self.poly_plot(fig, _ax, _date,\
                                 amp_data, plot_cbar=False)
                else:
                    _plt = self.poly_plot(fig, _ax, _date,\
                                 amp_data, plot_cbar=True)
        return fig        

    def format_lat_ticks(self, tick_val, tick_pos):
        # format lat ticks on the plot
        return int(90.-tick_val)

    def format_long_ticks(self, tick_val, tick_pos):
        # format mlt ticks on the plot
        return int(numpy.rad2deg(tick_val)/15.)

    def poly_plot(self, fig, ax, plot_date, amp_df,\
                    filter_jr_magn=0.05, cmap=plt.get_cmap('RdBu_r'),\
                    plot_cbar=True, vmin=-0.6,vmax=0.6,\
                    use_538=True, out_format="png", alpha=0.85,\
                    save_fig=False, plt_title=None, cax=None):
        """
        create a pcolormesh plot.
        """
        if use_538:
            plt.style.use("fivethirtyeight")
            sns.set_style("whitegrid")
        plt_df = amp_df[(amp_df["date"] == plot_date)]
        # check if we have data
        if plt_df.shape[0] == 0:
            print("No data found for this period! skipping")
            return None
        # we plot in colatitude and mlt is in radians, so we work with them
        plt_df["colat"] = 90 - plt_df["mlat"]
        # add an additional mlt (24) whose values are equal to 0 mlt
        # for contour plotting
        tmp_data = plt_df[ plt_df["mlt"] == 0.]
        tmp_data["mlt"] = 24.
        plt_df = pandas.concat([plt_df, tmp_data])
        plt_df["adj_mlt"] = numpy.deg2rad(plt_df["mlt"]*15)
        # we'll need to pivot the DF to covnert to plotting
        plt_df = plt_df[ ["colat", "adj_mlt",\
                        "pred_jr"] ].pivot( "colat", "adj_mlt" )
        colat_vals = plt_df.index.values
        adj_mlt_vals = plt_df.columns.levels[1].values
        colat_cntr, adj_mlt_cntr  = numpy.meshgrid( colat_vals, adj_mlt_vals )

        jr_vals = numpy.ma.masked_where((numpy.absolute(\
                    plt_df["pred_jr"].values)<=filter_jr_magn) | (numpy.isnan(\
                    plt_df["pred_jr"].values)),plt_df["pred_jr"].values)

        amp_plot = ax.pcolor(adj_mlt_cntr, colat_cntr, jr_vals.T,\
                               vmin=vmin,vmax=vmax, cmap=cmap,alpha=alpha)
        # set the yticks
        ax.yaxis.set_ticks(numpy.arange(10, 40, 10))
        ax.yaxis.set_major_formatter(FuncFormatter(self.format_lat_ticks))
        ax.set_ylim(0.,40.)
        if plt_title is None:
            ax.set_title( plot_date.strftime("%Y%m%d %H:%M"), fontsize=14 )
        else:
            ax.set_title( plt_title, fontsize=14 )
        # set the xticks for the plot
        ax.set_theta_offset(-1*numpy.pi/2)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_long_ticks))
        ax.grid(linestyle='--', linewidth='1', color='k')
        # sometimes the border takes up wierd values! rectify it!
        [i.set_linewidth(1.) for i in ax.spines.values()]
        [i.set_linestyle('--') for i in ax.spines.values()]
        [i.set_edgecolor('k') for i in ax.spines.values()]
        if plot_cbar:
            # Plot a colorbar
            fig.subplots_adjust(right=0.9)
            if cax is None:
                cax = fig.add_axes([0.88, 0.20, 0.02, 0.6])
            else:
                cax=cax
            cbar = fig.colorbar(amp_plot, cax=cax, orientation='vertical')
            cbar.set_label(r"J $ [\mu A/m^{2}]$")

        # save the figure
        if save_fig:
            fig_name = self.results_dir + self.amp_plot_name_ptrn +\
                         plot_date.strftime("%Y%m%d.%H%M") +\
                          "." + out_format
            fig.savefig( fig_name, bbox_inches='tight' )
            return fig_name, plot_date


if __name__ == "__main__":
    # from pathlib import Path
    # print(Path.cwd())
    pred_obj = PredAMP()
    uniq_times = pred_obj.get_unique_pred_times()
    pred_amp_df = pred_obj.generate_amp_pred(uniq_times)