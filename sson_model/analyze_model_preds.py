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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.cm import ScalarMappable

class AnlyzSSON(object):
    """
    A class to predict AMPERE FACs 
    based on the data input from 
    RT SW/IMF params
    """
    def __init__(self, model_name="model_anlyz", epoch=200, omn_pred_hist=120):
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
        self.original_sw_imf_data.set_index('propagated_time_tag', inplace=True)
        self.original_sw_imf_data = self.original_sw_imf_data.resample('1min').median()
        # manipulate original sw imf data for plotting
        # rename some parameters
        self.original_sw_imf_data["Vx"] = -1.*self.original_sw_imf_data["speed"]
        self.original_sw_imf_data.rename(\
                                    columns={\
                                    "density" : "Np", "bx" : "Bx", "by":"By", "bz":"Bz"\
                                    }, inplace=True\
                                     )
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
        # Load the model
        self.model = self.load_model(model_name, epoch)
        self.omn_pred_hist = omn_pred_hist
        
    def int_ax_round(self, x, base=5):
        return base * round(x/base) + base

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
        print("model_name--->", model_name)
        model = keras.models.load_model(model_name)
        return model

    def innvestigate_pred(self,\
                omn_train_params=["Bx", "By", "Bz", "Vx", "Np"]):
        """
        Use the innvestigate lib to analyze the network!
        """
        import innvestigate
        import innvestigate.utils as iutils

        omn_end_time = self.sw_imf_df.index.max()
        omn_begin_time = (omn_end_time - datetime.timedelta(\
                    minutes=self.omn_pred_hist) ).strftime(\
                    "%Y-%m-%d %H:%M:%S")
        inp_omn_vals = self.sw_imf_df.loc[\
                         omn_begin_time : omn_end_time \
                         ][omn_train_params].values
        inp_omn_vals = inp_omn_vals.reshape(1,inp_omn_vals.shape[0],\
                             inp_omn_vals.shape[1])
        # innvestigate now
        model_wo_softmax = iutils.keras.graph.model_wo_softmax(self.model)
        analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_softmax)
        anlyz_res = analyzer.analyze(inp_omn_vals)
        # Aggregate along color channels and normalize to [-1, 1]
#         a = a.sum(axis=numpy.argmax(numpy.asarray(a.shape) == 3))
        anlyz_res = numpy.squeeze(anlyz_res,axis=0)
        anlyz_res /= numpy.max(numpy.abs(anlyz_res))
        # Now also get the imf/sw values!
        swimf_data = self.original_sw_imf_data.loc[\
                         omn_begin_time : omn_end_time \
                         ][omn_train_params]
        return anlyz_res, swimf_data


    def plot_innves_results(self,\
                omn_train_params=["Bx", "By", "Bz", "Vx", "Np"]):
        """
        plot the results alongside actual input values!
        """
        import seaborn as sns
        from matplotlib.colors import ListedColormap, Normalize
        from matplotlib.cm import ScalarMappable

        anlyz_res, swimf_data = self.innvestigate_pred(\
                                omn_train_params=omn_train_params)
        swimf_data["datetime"] = swimf_data.index
        # set plot styling
        plt.style.use("fivethirtyeight")
        # get the number of panels
        nPanels = len(omn_train_params)
        fig, axes = plt.subplots(nrows=nPanels, ncols=1,\
                                 figsize=(8,8), sharex=True)
        # axis formatting
        dtLabFmt = DateFormatter('%H:%M')
        ax_cnt = 0
        # have a dict to set up a axis limit scale
        ax_lim_dict = {
                        "Bz" : [-1,5], 
                        "By" : [-1,5],
                        "Bx" : [-1,5],
                        "Vx" : [0,100],
                        "Np" : [0,5]
                      }
        omn_param_loc_dict = {
                        "Bz" : 2,
                        "By" : 1,
                        "Bx" : 0,
                        "Vx" : 3,
                        "Np" : 4
                      }
        # set parameters for shading the region by importance
        sea_map = ListedColormap(sns.color_palette("RdPu"))
        col_norm = Normalize( vmin=0, vmax=1 )
        # plot omni IMF
        for _op in [ "Bz", "By", "Bx", "Vx", "Np" ]:
            # plot actual Bz By values
            axes[ax_cnt].plot( swimf_data["datetime"].values,\
                          swimf_data[_op].values, linewidth=2 )
            axes[ax_cnt].set_ylabel(_op, fontsize=14)
            axes[ax_cnt].xaxis.set_major_formatter(dtLabFmt)
            # set axis limits
            _axlim_base = ax_lim_dict[_op][1]
            _axlim_low_limit = ax_lim_dict[_op][0]
            # get the max min limit
            _max_val = max( [ numpy.abs(swimf_data[_op].min()),\
                             numpy.abs(swimf_data[_op].max()) ] ) 
            _max_rnd = self.int_ax_round( _max_val, base=_axlim_base )
            if _op == "Vx":
                axes[ax_cnt].set_yticks(\
                    numpy.arange(-1*_max_rnd, _max_rnd, 250) )
                axes[ax_cnt].set_ylim( [ -1*_max_rnd, _axlim_low_limit ] )
            elif _op == "Np":
                axes[ax_cnt].set_yticks(\
                    numpy.arange(_axlim_low_limit*_max_rnd, _max_rnd, 5) )
                axes[ax_cnt].set_ylim( [ _axlim_low_limit*_max_rnd, _max_rnd ] )
            else:
                axes[ax_cnt].set_yticks(\
                    numpy.arange(_axlim_low_limit*_max_rnd, _max_rnd, 5) )
                axes[ax_cnt].set_ylim( [ _axlim_low_limit*_max_rnd, _max_rnd ] )
                axes[ax_cnt].axhline(y=0, color='k', linestyle='--',\
                        linewidth=1.)
            # shade the innovate preds
            _inn_pars = anlyz_res[:,omn_param_loc_dict[_op]]
            for _nb, _bins in enumerate(swimf_data["datetime"].values):
                if _nb == swimf_data["datetime"].values.shape[0] -1:
                    break
                bin_start = _bins
                bin_end = swimf_data["datetime"].values[_nb+1]
                curr_col = sea_map( col_norm(_inn_pars[_nb]) )
                shade_rgn_plot = axes[ax_cnt].axvspan(bin_start, bin_end, alpha=0.4, color=curr_col )
            ax_cnt += 1
        cax = fig.add_axes([0.27, 0.2, 0.5, 0.03])
        cbar = fig.colorbar(ScalarMappable(norm=col_norm, cmap=sea_map),
             orientation='horizontal', cax=cax)
        plt.tight_layout()
        fig.savefig("../data/a_test.png")
        # print(anlyz_res[:,0].shape, swimf_data["datetime"].values.shape)

    

if __name__ == "__main__":
    pred_obj = AnlyzSSON()
    pred_obj.plot_innves_results()