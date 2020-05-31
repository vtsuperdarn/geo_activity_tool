import datetime
import pandas
import numpy
import glob
import os
import sys
import shutil
sys.path.append("../utils/")
sys.path.append("utils/")
import sw_imf_utils
import pred_ampere
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates


class PlotPreds(object):
    """
    A class to plot predicted ampere FAC
    values along with IMF parameters input!
    """
    def __init__(self, model_name="b0fee2e6_20190921-12:56:55",\
             epoch=15, results_mapper_file="pred/mapper.txt",\
             results_dir="plots/"):
        """
        setup parameters and get the predicted ampere values
        as well as the latest IMF values!
        """
        # load IMF data
        # Load the RT SW/IMF data
        data_obj = sw_imf_utils.PropSW()
        self.sw_imf_df = data_obj.read_stored_data()
        # we need a 1-minute interval data!
        self.sw_imf_df.set_index('propagated_time_tag', inplace=True)
        # Get estimated AMPERE values
        pred_obj = pred_ampere.PredAMP(model_name=model_name, epoch=epoch,\
                        results_mapper_file=results_mapper_file)
        self.uniq_times = pred_obj.get_unique_pred_times()
        self.pred_amp_df = pred_obj.generate_amp_pred(self.uniq_times)
        self.results_dir = results_dir
        self.amp_plot_name_ptrn = "amp_pred_"
        self.imf_plot_name_ptrn = "imf_sw_"
        
    def correct_path(self, given_path):
        """
        Some times there is a problem with the 
        relative/absolute paths. Depending on where
        the code is called from. Correct it!
        """
        import pathlib
        file_path = pathlib.Path(given_path)
        if is_file:
            if file_path.exists():
                file_path = file_path.as_posix()
            else:
                file_path = pathlib.Path.cwd().joinpath("amp_model",\
                                             given_path).as_posix()
        return file_path
    
    def generate_plots(self, clear_existing=True):
        """
        Generate plots for the given times!
        """
        # first note down which files are present in the folder
        # after that generate new one's and delete older one's 
        # finally!

        if clear_existing:
            old_amp_files = glob.glob(self.results_dir + self.amp_plot_name_ptrn + "*")
            old_imf_plots = glob.glob(self.results_dir + self.imf_plot_name_ptrn + "*")
        # generate the AMPERE plots!
        # but be careful not to delete new ones
        new_amp_plots = []
        date_amp_plots = []
        for _time in self.uniq_times:
            fig = plt.figure(figsize=(12, 8))
            ax = plt.subplot(1, 1, 1, projection='polar')
            _figname, _plt_date = self.poly_plot(fig, ax, _time)
            new_amp_plots.append( _figname )
            date_amp_plots.append( _plt_date )
        new_amp_plots = [ x for _, x in sorted(zip(date_amp_plots, new_amp_plots)) ][::-1]
        for _np, _plot in enumerate(new_amp_plots):
            print(_plot, _np)
            shutil.copyfile(_plot, "static/rt_imgs/amp_plot" + str(_np) + ".png")
        # generate IMF plots
        new_imf_plot = self.imf_sw_plot()
        # copy imf file
        shutil.copyfile(new_imf_plot, "static/rt_imgs/sw_imf_plot.png")
        print("generated new plots")
        # delete the older one's
        for _old in old_amp_files:
            if _old not in new_amp_plots:
                os.remove(_old)
        for _old_imf in old_imf_plots:
            if new_imf_plot != _old_imf:
                os.remove(_old_imf)
        print("deleted old plots")

    def format_lat_ticks(self, tick_val, tick_pos):
        # format lat ticks on the plot
        return int(90.-tick_val)

    def format_long_ticks(self, tick_val, tick_pos):
        # format mlt ticks on the plot
        return int(numpy.rad2deg(tick_val)/15.)

    def poly_plot(self, fig, ax, plot_date,\
                    filter_jr_magn=0.05, cmap=plt.get_cmap('RdBu_r'),\
                    plot_cbar=True, vmin=-0.5,vmax=0.5,\
                    use_538=True, out_format="png", alpha=0.85,save_fig=True):
        """
        create a pcolormesh plot.
        """
        if use_538:
            plt.style.use("fivethirtyeight")
            sns.set_style("whitegrid")
        plt_df = self.pred_amp_df[(self.pred_amp_df["date"] == plot_date)]
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
        ax.set_title( plot_date.strftime("%Y%m%d %H:%M") )
        # set the xticks for the plot
        ax.set_theta_offset(-1*numpy.pi/2)
        ax.xaxis.set_major_formatter(FuncFormatter(self.format_long_ticks))
        ax.grid(linestyle='--', linewidth='1', color='k')#linestyle='--', linewidth='1', color='k'
        # sometimes the border takes up wierd values! rectify it!
        [i.set_linewidth(1.) for i in ax.spines.values()]
        [i.set_linestyle('--') for i in ax.spines.values()]
        [i.set_edgecolor('k') for i in ax.spines.values()]
        if plot_cbar:
            cbar = plt.colorbar(amp_plot, orientation='vertical')
            cbar.set_label(r"J $ [\mu A/m^{2}]$")

        # save the figure
        if save_fig:
            fig_name = self.results_dir + self.amp_plot_name_ptrn +\
                         plot_date.strftime("%Y%m%d.%H%M") +\
                          "." + out_format
            fig.savefig( fig_name, bbox_inches='tight' )
        return fig_name, plot_date

    def imf_sw_plot(self,plot_time_interval=2,\
         plot_params=["Bz", "By", "Bx", "Vx", "Np"],\
         use_538=True, out_format="png",save_fig=True):
        """
        Generate IMF plot!
        plot time interval is in hours!
        """
        # filter the data
        if use_538:
            plt.style.use("fivethirtyeight")
            sns.set_style("whitegrid")
        max_time = max(self.uniq_times)
        min_time = max_time - datetime.timedelta(hours=plot_time_interval)
        plot_min_marker = max_time - datetime.timedelta(hours=1)
        omn_df = self.sw_imf_df.loc[ min_time : max_time ]
        # rename the col names
        omn_df.rename( columns={
                "bz":"Bz",
                "by":"By",
                "bx":"Bx",
                "speed":"Vx",
                "density":"Np",
            }, inplace=True )
        fig = plt.figure(figsize=(12, 8))
        fig, ax_arr = plt.subplots(len(plot_params), sharex=True)
        # set up some params
        linewidth=2
        # loop through and plot!
        print(omn_df.head())
        for _n,_par in enumerate(plot_params):
            ax_arr[_n].plot(omn_df.index.values,\
                     omn_df[_par].values, linewidth=linewidth)
            ax_arr[_n].set_ylabel(_par, fontsize=12)
            ax_arr[_n].tick_params(axis='both', which='major', labelsize=10)
            ax_arr[_n].set_xlim([min_time, max_time])
            ax_arr[_n].axvspan(plot_min_marker, max_time,\
                         color='#fc4f30', linestyle=':',alpha=.25)
        ax_arr[-1].set_xlabel("Propaged Time (UT)", fontsize=12)
        xlab_date_format = mdates.DateFormatter('%d/%H:%M')
        ax_arr[-1].xaxis.set_major_formatter(xlab_date_format)
        # save the figure
        if save_fig:
            fig_name = self.results_dir + self.imf_plot_name_ptrn +\
                         max_time.strftime("%Y%m%d.%H%M") + "_" +\
                          max_time.strftime("%Y%m%d.%H%M") +\
                          "." + out_format
            print(fig_name)
            fig.savefig( fig_name, bbox_inches='tight' )

        return fig_name

# if __name__ == "__main__":
#     plot_obj = PlotPreds()
#     plot_obj.generate_plots()