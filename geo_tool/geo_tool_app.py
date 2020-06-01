import datetime

def geo_activity_page(
                      state,
                      local_data_store="./data/sqlite3/",\
                      plot_style="classic",\
                      inp_start_date=datetime.date(2018, 1, 2),\
                      inp_start_time=datetime.time(0, 0),\
                      all_param_list = [ "DSCOVER", "OMNI", "STORM",\
                                          "SUBSTORM", "SUPERDARN" ],\
                      nhours_plot_default=0,\
                      ndays_plot_default=1
                     ):
    """
    Geo activity tool page
    """
    import sys
    import streamlit as st
    import sys
    sys.path.append("../")
    import session_state
    import pandas
    
    # set the session state and get the parameters
    st.sidebar.markdown("### Plotting parameters")
    # setup the input params!
    geo_plot = st.empty()
    params_multi_sel = st.sidebar.empty()
    start_day_inpdt = st.sidebar.empty()
    start_time_inpdt = st.sidebar.empty()
    ndays_slider = st.sidebar.empty()
    nhours_slider = st.sidebar.empty()
    # select the parameters
    state.plot_param_list = params_multi_sel.multiselect(\
                            'Select Parameters',\
                            all_param_list,\
                            default=all_param_list\
                            )
    state.plot_start_date = start_day_inpdt.date_input(
                 "Start Date",
                 state.plot_start_date
            )
    state.plot_start_time = start_time_inpdt.time_input(
                 "Start Time",
                 state.plot_start_time
            )
    state.plot_nhours_plot = ndays_slider.slider("Number of hours to plot",\
                                      0, 23, state.plot_nhours_plot)
    state.plot_ndays_plot = nhours_slider.slider("Number of days to plot",\
                                      0, 15, state.plot_ndays_plot)
    # ss onset predictions
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_spin.image("misc/brewing2.gif")
    geo_fig = create_geo_plot_fig(\
                                 all_param_list,\
                                 state.plot_param_list,\
                                 state.plot_start_date,\
                                 state.plot_start_time,\
                                 state.plot_nhours_plot,\
                                 state.plot_ndays_plot,\
                                 plot_style=plot_style,\
                                 local_data_store=local_data_store
                                 )

    tmp_plchldr_spin.empty()
    geo_plot.pyplot(geo_fig)
    # Previous day button
    if st.sidebar.button("Prev day"):
        state.plot_start_date = state.plot_start_date - datetime.timedelta(days=1)
        # reset the page!
        geo_fig = create_geo_plot_fig(all_param_list, state.plot_param_list,\
                                 state.plot_start_date, state.plot_start_time,\
                                 state.plot_nhours_plot,\
                                 state.plot_ndays_plot, plot_style=plot_style,\
                                 local_data_store=local_data_store)
        tmp_plchldr_spin.empty()
        geo_plot.pyplot(geo_fig)
        state.plot_start_date = start_day_inpdt.date_input(
                 "Start Date",
                 state.plot_start_date
            )
    # Next day button
    if st.sidebar.button("Next day"):
        state.plot_start_date = state.plot_start_date + datetime.timedelta(days=1)
        # reset the page!
        geo_fig = create_geo_plot_fig(all_param_list, state.plot_param_list,\
                                 state.plot_start_date, state.plot_start_time,\
                                 state.plot_nhours_plot,\
                                 state.plot_ndays_plot, plot_style=plot_style,\
                                 local_data_store=local_data_store)
        tmp_plchldr_spin.empty()
        geo_plot.pyplot(geo_fig)
        state.plot_start_date = start_day_inpdt.date_input(
                 "Start Date",
                 state.plot_start_date
            )
        
        
def create_geo_plot_fig(all_param_list, params_selected,\
                     start_day, start_time, nhours_plot,\
                      ndays_plot, plot_style="classic",\
                     local_data_store="./data/sqlite3/"):
    """
    Create the figure!
    """
    from plotting import fetch_data_plotting
    # populate the flags for the plotting parameter
    flags = [ True for _i in all_param_list ]
    for _npar,_par in enumerate(all_param_list):
        if _par not in params_selected:
            flags[_npar] = False
    sdate = datetime.datetime.combine(start_day, 
                              start_time)
    edate = sdate + datetime.timedelta(days=ndays_plot,\
                                       hours=nhours_plot)
    return fetch_data_plotting(\
                         sdate, edate, plot_style=plot_style,\
                         flags=flags,\
                         local_data_store=local_data_store)
    
    
    
