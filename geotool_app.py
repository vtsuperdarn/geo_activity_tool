import streamlit as st
import session_state
import pandas
import datetime

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

    if model_option == 'Daily geoactivity tool':
        geo_activity_page()
    elif model_option == 'GPS TEC tool':
        gps_tec_page()
    elif model_option == 'AMPERE FACs forecast':
        fac_model_page()
    else:
        ss_onset_page()

def gps_tec_page():
    """
    GPS TEC page
    """
    tmp_plchldr_txt = st.empty()
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_txt.subheader("Under development. Come back later...")
    tmp_plchldr_spin.image("misc/brewing2.gif")
    
def fac_model_page():
    """
    FAC model page
    """
    tmp_plchldr_txt = st.empty()
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_txt.subheader("Under development. Come back later...")
    tmp_plchldr_spin.image("misc/brewing2.gif")
    
def ss_onset_page():
    """
    FAC model page
    """
    tmp_plchldr_txt = st.empty()
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_txt.subheader("Under development. Come back later...")
    tmp_plchldr_spin.image("misc/brewing2.gif")    
    
        
def geo_activity_page(local_data_store="data/sqlite3/"):
    """
    Geo activity tool page
    """
    import sys
    sys.path.append("../")
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
    sdate = st.sidebar.date_input(
                 "Start Date",
                 datetime.date(2018, 1, 1)
            )
    edate = st.sidebar.date_input(
                 "End Date",
                 datetime.date(2018, 1, 1)
            )
    # ss onset predictions
    tmp_plchldr_txt = st.empty()
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_spin.image("misc/brewing2.gif")
    
    # populate the flags for the plotting parameter
    flags = [ True for _i in all_param_list ]
    for _npar,_par in enumerate(all_param_list):
        if _par not in params_selected:
            flags[_npar] = False
    
    geo_fig = fetch_data_plotting(\
                         sdate, edate, plot_style="classic",\
                         flags=flags,\
                         local_data_store=local_data_store)
    tmp_plchldr_spin.pyplot(geo_fig)
#     st.pyplot(geo_fig)
    
#     sdate_state = session_state.get(plot_start_date = sdate, plot_end_date = edate)
    
    
if __name__ == "__main__":
    main()