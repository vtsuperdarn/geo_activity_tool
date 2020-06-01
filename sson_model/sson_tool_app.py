import altair as alt
import streamlit as st
import sys
sys.path.append("../")
import session_state
import pandas
import datetime
import pred_ss_onset
import extract_rt_al

def fill_ssonset_preds(state):
    # work with some FAQs in the sidebar
#     st.sidebar.markdown("# FAQs")
    st.sidebar.markdown("### What is a substorm?")
#     if ques_option == faq_list[0]:
    st.sidebar.markdown("A substorm, sometimes referred to as an auroral or magnetospheric substorm, is a disturbance in the Earth's magnetosphere that releases energy stored in the magnetotail into the high latitude ionosphere. Typically, a substorm has a life span ranging between 1 to 3 hours.")
    st.sidebar.markdown("### What happens during a substorm?")
#     elif ques_option == faq_list[1]:
    st.sidebar.markdown("During substorms, quiet auroral arcs suddenly explode into spectacular auroral displays. From a scientific perspective forecasting the onset of substorms is important as it can provide insights into several important features in the near-Earth space environment. Forecasting substorms is also important from a purely space weather perspective since their onset can drive geomagnetically induced currents.")
    st.sidebar.markdown("### How do we characterize a substorm in this paper?")
#     elif ques_option == faq_list[2]:
    st.sidebar.markdown("Another important feature of a substorm is a very strong ionospheric current called the auroral electrojet which can be measured using ground based magnetometers. SuperMAG is a chain of magnetometers and data from the magnetometers is summarized using the SML index. Here, we characterize a substorm based on the [SML index](http://supermag.jhuapl.edu/indices)")
    st.sidebar.markdown("### How do we predict substorm onset?")
#     else:
    st.sidebar.markdown("We trained a [ResNet convolutional neural network] (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019SW002251) using the [SuperMAG substorm database] (http://supermag.jhuapl.edu/substorms/). Our model can correctly identify substorms 75% of the time.")
    # ss onset predictions
    tmp_plchldr_txt = st.empty()
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_txt.subheader("calculating substorm probability...")
    tmp_plchldr_spin.image("./misc/cnn.gif")
    curr_al_df = get_al_data(round_time(datetime.datetime.now(), period=60))
    prob_onset, imf_fig, gauge_fig,\
            curr_ss_string, omn_begin_time,\
            omn_end_time, ss_string_24hr = get_sson_preds(curr_al_df)
    pred_hor_strt = omn_end_time + datetime.timedelta(minutes=1)
    pred_hor_end = omn_end_time + datetime.timedelta(minutes=60)
    pred_hor = pred_hor_strt.strftime("%H:%M") +\
                "-" + pred_hor_end.strftime("%H:%M") + " UT"
    prob_str = "Probability of substorm onset in the next hour " +\
                    " (" + pred_hor + ")" + " : " + "***" + str(prob_onset) + "***" 
    tmp_plchldr_txt.empty()
    tmp_plchldr_spin.empty()
    curr_ss_activity = "Substorm activity in the last two hours : " + curr_ss_string
    pred_dt_str = "Substorm prediction time: " + omn_end_time.strftime("%B %d, %Y %H:%M") + " UT"
    time_to_ss_str = "Time to the last substorm activity: " + ss_string_24hr
    st.markdown(pred_dt_str)
    st.markdown(curr_ss_activity)
    st.markdown(time_to_ss_str)
    st.markdown(prob_str)
    st.pyplot(gauge_fig)
    # additional plots
    ss_plot_option = st.selectbox(
                             'Plot',
                             (
                              'Prediction history',\
                              'IMF Values used for latest prediction'
                              )
                             )
    if ss_plot_option == 'IMF Values used for latest prediction':
        st.pyplot(imf_fig)
    else:
        hist_fig = get_hist_sson_preds(state.date_sson_hist_plot, curr_al_df)#
        if hist_fig is not None:
            hist_plot = st.empty()
            hist_plot.pyplot(hist_fig)
        else:
            plot_date_err = state.date_sson_hist_plot + datetime.timedelta(days=1)
            st.markdown("No data at this date, try a date > " +\
                        plot_date_err.strftime("%B %d, %Y") )
        if st.button("Prev day"):
            state.date_sson_hist_plot = state.date_sson_hist_plot - datetime.timedelta(days=1)
            hist_fig = get_hist_sson_preds(state.date_sson_hist_plot, curr_al_df)
            if hist_fig is not None:
                hist_plot.pyplot(hist_fig)
            else:
                plot_date_err = state.date_sson_hist_plot + datetime.timedelta(days=1)
                st.markdown("No data at this date, try a date > " +\
                            plot_date_err.strftime("%B %d, %Y") )
        if st.button("Next day"):
            state.date_sson_hist_plot = state.date_sson_hist_plot + datetime.timedelta(days=1)
            hist_fig = get_hist_sson_preds(state.date_sson_hist_plot, curr_al_df)
            if hist_fig is not None:
                hist_plot.pyplot(hist_fig)
            else:
                plot_date_err = state.date_sson_hist_plot - datetime.timedelta(days=1)
                st.markdown("No data at this date, try a date < " +\
                            plot_date_err.strftime("%B %d, %Y") )
        st.markdown("[Link to real time Auroral indices](http://wdc.kugi.kyoto-u.ac.jp/ae_realtime/today/today.html)")
        
def get_al_data(call_time):
    """
    Download the AL index data
    """
    al_obj = extract_rt_al.ExtractAL()
    al_df = al_obj.get_al_data()
    return al_df


def get_sson_preds(al_df):
    # now plot historical estimates
    pred_sson_obj = create_sson_obj(round_time(datetime.datetime.now()))
    prob_onset, fig, omn_begin_time, omn_end_time = pred_sson_obj.generate_bin_plot(\
                                                                        gen_pred_diff_time=20)
    gauge_fig = pred_sson_obj.generate_gauge_plot(prob_onset)
     # get real time al index for estimates of 
    # current activity!
    al_df_2hour = al_df[\
             (al_df["date"] >= str(omn_begin_time)) &\
             (al_df["date"] <= str(omn_end_time))\
             ]
    # estimate current activity
    if al_df_2hour.shape[0] == 0:
        curr_ss_string = "Unknown"
    else:
        if (al_df_2hour["al"].min() > -500) and ((al_df_2hour["al"].min() < -200)):
            curr_ss_string = "May be"
        elif (al_df_2hour["al"].min() <= -500):
            curr_ss_string = "Yes"
        else:
            curr_ss_string = "No"
    # we also want to estimate time to last ss activity
    # it will be either 0-24 hours or > 24 hours
    al_df_24hour = al_df[\
             (al_df["date"] >= omn_end_time - datetime.timedelta(hours=24)) &\
             (al_df["date"] <= omn_end_time)\
             ]
    # estimate current activity
    if al_df_24hour.shape[0] == 0:
        ss_string_24hr = "Unknown"
    elif (al_df_24hour["al"].min() <= -200):
        ss_string_24hr = "0-24 hours"
    else:
        ss_string_24hr = ">24 hours"
    
    return prob_onset, fig, gauge_fig, curr_ss_string, omn_begin_time, omn_end_time, ss_string_24hr

# @st.cache(hash_funcs={pred_ss_onset.PredSSON: id})
def create_sson_obj(call_time,model_name="model_paper",\
         epoch=200, omn_pred_hist=120):
        return pred_ss_onset.PredSSON(model_name=model_name,\
                        epoch=epoch, omn_pred_hist=omn_pred_hist)

def get_hist_sson_preds(plot_date,al_df):
    
    pred_sson_obj = create_sson_obj(round_time(datetime.datetime.now()))
#     hist_fig = pred_sson_obj.plot_historical_preds(time_extent=time_extent)
    hist_fig = pred_sson_obj.plot_daywise_preds(plot_date,al_df=al_df, gen_pred_diff_time=30.)
    return hist_fig

def round_time(dt_time, period=20):
    '''
    Given a period in minute rounds the datetime to the nearest lowest divisor for that minute:
    For Example: 
    Given dt_time='2020-03-27 02:32:19.684443' and period = 5 the function returns '2020-03-27 02:30:00.0'
    '''
    import math
    # 1. Find nearest floor of minute to the current time minute
    new_minute_part = math.floor(dt_time.minute/period)*period
    # 2. Replace minute part in curr_time with nearest floored down minute 
    dt_time = dt_time.replace(minute=new_minute_part,second=0, microsecond=0)
    return dt_time
