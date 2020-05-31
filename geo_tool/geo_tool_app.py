def geo_activity_page(local_data_store="./geo_tool/data/sqlite3/",\
                      plot_style="classic"):
    """
    Geo activity tool page
    """
    import sys
    import streamlit as st
    import session_state
    import pandas
    import datetime
    from plotting import fetch_data_plotting
    
    st.sidebar.markdown("### Plotting parameters")
    # select the parameters
    all_param_list = [ "DSCOVER", "OMNI", "STORM",\
                  "SUBSTORM", "SUPERDARN" ]
    params_selected = st.sidebar.multiselect(\
                            'Select Parameters',\
                            all_param_list,\
                            default=all_param_list\
                            )
    start_day = st.sidebar.date_input(
                 "Start Date",
                 datetime.date(2018, 1, 2)
            )
    start_time = st.sidebar.time_input(
                 "Start Time",
                 datetime.time(0, 0)
            )
    sdate = datetime.datetime.combine(start_day, 
                              start_time)
    ndays_hours = st.sidebar.slider("Number of hours to plot", 0, 23, 0)
    ndays_plot = st.sidebar.slider("Number of days to plot", 0, 15, 1)
    edate = sdate + datetime.timedelta(days=ndays_plot,\
                                       hours=ndays_hours)
    # ss onset predictions
    tmp_plchldr_txt = st.empty()
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_spin.image("misc/brewing2.gif")
    
    # populate the flags for the plotting parameter
    flags = [ True for _i in all_param_list ]
    for _npar,_par in enumerate(all_param_list):
        if _par not in params_selected:
            flags[_npar] = False
    geo_plot = st.empty()
#     geo_date = st.markdown(\
#                   sdate.strftime("%Y%m%d-%H%M") +\
#                    " to " + edate.strftime("%Y%m%d-%H%M")\
#                   )
    geo_fig = fetch_data_plotting(\
                         sdate, edate, plot_style=plot_style,\
                         flags=flags,\
                         local_data_store=local_data_store)
    tmp_plchldr_spin.empty()
    geo_plot.pyplot(geo_fig)
#     if st.sidebar.button("Prev day"):
#         state = session_state.get(\
#                             plot_start_date=sdate,\
#                             plot_end_date=edate\
#                             )
#         state.plot_start_date = state.plot_start_date - datetime.timedelta(days=1)
#         state.plot_end_date = state.plot_end_date - datetime.timedelta(days=1)
#         geo_date.markdown(\
#                   state.plot_start_date.strftime("%Y%m%d-%H%M") +\
#                    " - " + state.plot_end_date.strftime("%Y%m%d-%H%M")\
#                   )
#         geo_fig = fetch_data_plotting(\
#                         state.plot_start_date,\
#                         state.plot_end_date,\
#                         plot_style=plot_style,\
#                         flags=flags,\
#                         local_data_store=local_data_store)
#         tmp_plchldr_spin.empty()
#         geo_plot.pyplot(geo_fig)
    
    
