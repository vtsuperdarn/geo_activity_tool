import altair as alt
import streamlit as st
import pandas
import datetime
import sys
sys.path.append("./amp_model")
import pred_ampere

def real_time_amp_preds():
    # ss onset predictions
    tmp_plchldr_txt = st.empty()
    tmp_plchldr_spin = st.empty()
    tmp_plchldr_txt.subheader("calculating currents...")
    tmp_plchldr_spin.image("./misc/cnn.gif")
    amp_data_df, amp_plot, sw_imf_df = get_amp_preds(\
                        model_name="3f8579fa_20190921-13:37:19")
    tmp_plchldr_txt.empty()
    tmp_plchldr_spin.empty()
    # AMPERE plots!
    st.header("Field aligned current forecasts")
    st.pyplot(amp_plot)
    # plot IMF/SW data
    st.header("Realtime IMF and solar wind data")
    rt_imf_sw_chart = imf_altair_chart(sw_imf_df)
    st.altair_chart(rt_imf_sw_chart)

    
def imf_altair_chart(sw_imf_df, date_format='%H:%M'):
    # we'll vconcat the charts
    chart_list = []
    # modulate the data
    imf_only = sw_imf_df[ ["bx", "by", "bz", "propagated_time_tag"] ]
    imf_only = imf_only.melt('propagated_time_tag',\
                 var_name='category', value_name='y')
    # create an altair chart for realtime IMF/SW data
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['propagated_time_tag'], empty='none')

    bz_chart = alt.Chart(sw_imf_df, width=600).mark_line(interpolate='basis').encode(
        x=alt.X('propagated_time_tag', axis=alt.Axis(format=date_format,title='PROPAGATED UT TIME')),
        y=alt.Y('bz', axis=alt.Axis(title='IMF Bz [nT]'))
    )
    chart_list.append(bz_chart)
    by_chart = alt.Chart(sw_imf_df, width=600).mark_line(interpolate='basis').encode(
        x=alt.X('propagated_time_tag', axis=alt.Axis(format=date_format,title='PROPAGATED UT TIME')),
        y=alt.Y('by', axis=alt.Axis(title='IMF By [nT]'))
    )
    chart_list.append(by_chart)
    bx_chart = alt.Chart(sw_imf_df, width=600).mark_line(interpolate='basis').encode(
        x=alt.X('propagated_time_tag', axis=alt.Axis(format=date_format,title='PROPAGATED UT TIME')),
        y=alt.Y('bx', axis=alt.Axis(title='IMF Bx [nT]'))
    )
    chart_list.append(bx_chart)
    # Put the five layers into a chart and bind the data
    # solar wind speed chart
    vx_chart = alt.Chart(sw_imf_df, width=600).mark_line(interpolate='basis').encode(
        x=alt.X('propagated_time_tag', axis=alt.Axis(format=date_format,title='PROPAGATED UT TIME')),
        y=alt.Y('speed', axis=alt.Axis(title='Solar wind speed [km/s]'))
    )
    chart_list.append(vx_chart)
    # solar wind number density chart
    np_chart = alt.Chart(sw_imf_df, width=600).mark_line(interpolate='basis').encode(
        x=alt.X('propagated_time_tag', axis=alt.Axis(format=date_format,title='PROPAGATED UT TIME')),
        y=alt.Y('density', axis=alt.Axis(title='Density [/cm^3]'))
    )
    chart_list.append(np_chart)
    return alt.vconcat(*chart_list)    

def get_amp_preds(model_name="b0fee2e6_20190921-12:56:55",\
                    au_val=240, al_val=-240,\
                    symh_val=-10, asymh_val=15,\
                    f107_val=115, use_manual_bz=False,\
                    use_manual_by=False, use_manual_bx=False,\
                    use_manual_vx=False, use_manual_np=False,\
                    use_manual_month=False, manual_bz_val=None,\
                    manual_by_val=None, manual_bx_val=None,\
                    manual_vx_val=None, manual_np_val=None,\
                    manual_mnth_val=None):
    import datetime
    pred_obj = pred_ampere.PredAMP(model_name=model_name, inp_au=au_val, inp_al=al_val,\
                    inp_symh=symh_val, inp_asymh=asymh_val,\
                    inp_f107=f107_val, use_manual_bz=use_manual_bz,\
                    use_manual_by=use_manual_by, use_manual_bx=use_manual_bx,\
                    use_manual_vx=use_manual_vx, use_manual_np=use_manual_np,\
                    use_manual_month=use_manual_month, manual_bz_val=manual_bz_val,\
                    manual_by_val=manual_by_val, manual_bx_val=manual_bx_val,\
                    manual_vx_val=manual_vx_val, manual_np_val=manual_np_val,\
                    manual_mnth_val=manual_mnth_val)
    # we need to manipulate the solarwind IMF data
    # sort the data first and choose the latest 6 hours!
    sw_imf_df = pred_obj.original_sw_imf_data
    sw_imf_df.sort_values(by=['propagated_time_tag'],\
                     ascending=False, inplace=True)
    max_date = sw_imf_df['propagated_time_tag'].max()
    min_date = max_date-datetime.timedelta(hours=6)
    sw_imf_df = sw_imf_df[ (sw_imf_df["propagated_time_tag"] >= min_date) ].reset_index(drop=True)
    # other paramterss to be sent!
    uniq_times = pred_obj.get_unique_pred_times()
    pred_amp_df = pred_obj.generate_amp_pred(uniq_times)
    amp_plot = pred_obj.generate_plots(uniq_times, pred_amp_df)
    return pred_amp_df, amp_plot, sw_imf_df