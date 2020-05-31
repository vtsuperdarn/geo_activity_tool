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
    
        
def geo_activity_page(local_data_store="./geo_tool/data/sqlite3/",\
                      plot_style="classic"):
    """
    Geo activity tool page
    """
    import sys
    sys.path.append("./geo_tool/")
    from geo_tool_app import geo_activity_page
    geo_activity_page(local_data_store=local_data_store,\
                      plot_style=plot_style)
    
if __name__ == "__main__":
    main()