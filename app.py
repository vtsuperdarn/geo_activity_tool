import streamlit as st
import session_state
import pandas
import datetime
import sys
sys.path.append("./sson_model")
import dwnld_sw_imf_rt

# Streamlit encourages well-structured code, like starting execution in a main() function.
def main():
    # setup a title
    st.title('Select a tool')
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    model_option = st.selectbox(
                             '',
                             (
                              'Daily geoactivity tool',\
                              'GPS TEC tool',\
                              'AMPERE FACs forecast',\
                              'Substorm onset forecast'
                              )
                             )
    # we'll need a session state to switch between dates
    # basically the prev day and next day buttons!
    # session state details for the geo_activity_tool page
    # session state details for the geo_activity_tool page
    geo_all_param_list = [ "DSCOVER", "OMNI", "STORM",\
                  "SUBSTORM", "SUPERDARN" ]
    nhours_plot_default = 0
    ndays_plot_default = 1
    inp_start_date = datetime.date(2018, 1, 2)
    inp_start_time = datetime.time(0, 0)
    # session state details for the geo_activity_tool page
    # session state details for the geo_activity_tool page
    # session state details for the sson_model page
    # session state details for the sson_model page
    data_obj = dwnld_sw_imf_rt.DwnldRTSW()
    url_data = data_obj.dwnld_file()
    if url_data is not None:
        data_obj.read_url_data(url_data)
    # repeat the operations we do with sson_model calc
    sw_imf_df = data_obj.read_stored_data()
    sw_imf_df.set_index('propagated_time_tag', inplace=True)
    sw_imf_df = sw_imf_df.resample('1min').median()
    # linearly interpolate data
    sw_imf_df.interpolate(method='linear', axis=0, inplace=True)
    omn_end_time = sw_imf_df.index.max()
    # session state details for the sson_model page
    # session state details for the sson_model page
    # common session state details for the all pages
    state = session_state.get(\
                            plot_start_date=inp_start_date,\
                            plot_start_time=inp_start_time,\
                            plot_param_list=geo_all_param_list,\
                            plot_nhours_plot=nhours_plot_default,\
                            plot_ndays_plot=ndays_plot_default,\
                            date_sson_hist_plot=omn_end_time
                            )
    
    if model_option == 'Daily geoactivity tool':
        geo_activity_page(
                      state,\
                      local_data_store="./geo_tool/data/sqlite3/",\
                      plot_style="classic",\
                      inp_start_date=inp_start_date,\
                      inp_start_time=inp_start_time,\
                      all_param_list=geo_all_param_list,\
                      nhours_plot_default=nhours_plot_default,\
                      ndays_plot_default=ndays_plot_default
                    )
    elif model_option == 'GPS TEC tool':
        gps_tec_page()
    elif model_option == 'AMPERE FACs forecast':
        fac_model_page()
    else:
        ss_onset_page(state)

def gps_tec_page():
    """
    GPS TEC page
    """
    import sys
    sys.path.append("./gps_tec/")
    from TECapp import tec_webtool 
    tec_webtool()
    
def fac_model_page():
    """
    FAC model page
    """
    import sys
    sys.path.append("./amp_model/")
    from amp_tool_app import real_time_amp_preds
    real_time_amp_preds()
    
def ss_onset_page(state):
    """
    FAC model page
    """
    import sys
    sys.path.append("./sson_model/")
    from sson_tool_app import fill_ssonset_preds
    fill_ssonset_preds(state)
    
        
def geo_activity_page(state,local_data_store="./geo_tool/data/sqlite3/",\
                      plot_style="classic",\
                      inp_start_date=datetime.date(2018, 1, 2),\
                      inp_start_time=datetime.time(0, 0),\
                      all_param_list=[ "DSCOVER", "OMNI", "STORM",\
                                      "SUBSTORM", "SUPERDARN" ],\
                      nhours_plot_default=0,\
                      ndays_plot_default=1):
    """
    Geo activity tool page
    """
    import sys
    sys.path.append("./geo_tool/")
    from geo_tool_app import geo_activity_page
    geo_activity_page(state, local_data_store=local_data_store,\
                      plot_style=plot_style,\
                     inp_start_date=inp_start_date,\
                     inp_start_time=inp_start_time,\
                     all_param_list=all_param_list,\
                     nhours_plot_default=nhours_plot_default,\
                     ndays_plot_default=ndays_plot_default)
    
if __name__ == "__main__":
    main()
