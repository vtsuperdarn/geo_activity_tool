import datetime
import pandas

def create_url_form(date_range,base_url,\
                 email='test@gmail.com', output="AE"):
    """
    create url based on a date range
    """
    import urllib.parse, urllib.request

    # url details
    url_data = {} 
    del_dt = date_range[1] - date_range[0]
    # create the form
    url_data['Tens'] = int(date_range[0].year/10)
    url_data['Year'] = int(date_range[0].year - 10*url_data['Tens'])
    url_data['Month']= date_range[0].month
    url_data['Day_Tens'] = int(date_range[0].day/10)
    url_data['Days']     = int(date_range[0].day - 10*url_data['Day_Tens'])
    url_data['Hour'] = date_range[0].hour
    url_data['min']  = date_range[0].minute
    url_data['Dur_Day_Tens'] = int(del_dt.days/10)
    url_data['Dur_Day']      = int(del_dt.days - 10*url_data['Dur_Day_Tens'])
    url_data['Dur_Hour']     = int(del_dt.seconds/3600.)
    url_data['Dur_Min']      = int(del_dt.seconds/60.-url_data['Dur_Hour']*60.)

    url_data['Output']     = output
    url_data['Out+format'] = 'IAGA2002'
    url_data['Email']      =  urllib.parse.quote(email)

    data='Tens={0[Tens]:03d}&Year={0[Year]:d}&'.format(url_data)
    data+='Month={0[Month]:02d}&Day_Tens={0[Day_Tens]:d}&'.format(url_data)
    data+='Days={0[Days]:d}&Hour={0[Hour]:02d}&min={0[min]:02d}'.format(url_data)
    data+='&Dur_Day_Tens={0[Dur_Day_Tens]:02d}&'.format(url_data)
    data+='Dur_Day={0[Dur_Day]:d}&Dur_Hour={0[Dur_Hour]:02d}&'.format(url_data)
    data+='Dur_Min={0[Dur_Min]:02d}&Image+Type=GIF&COLOR='.format(url_data)
    data+='COLOR&AE+Sensitivity=0&ASY%2FSYM++Sensitivity=0&'
    data+='Output=AE&Out+format='.format(url_data)
    data+='IAGA2002&Email={0[Email]}'.format(url_data)
    return urllib.request.urlopen(base_url+'?'+data) 

def iaga_format_to_df(input_data, num_head=12):
    """
    READ WDC Kyoto's IADA format and convert to
    a data (assuming there is out_data_dict availble!)
    """
    from dateutil.parser import parse
    # Begin by parsing header; ensuring the correct file format.
    fmt=(input_data.pop(0)).split()
    if len(fmt) == 0:
        print("no data found!")
        return None
    if (fmt[0].decode("utf-8") !='Format') or (fmt[1].decode("utf-8") !='IAGA-2002'):
        print('wrong inputs.')
        return None
    # header
    source=(input_data.pop(0)).split()[1]
    input_data.pop(0)
    code=(input_data.pop(0)).split()[2]
    for i in range(8):
        input_data.pop(0)
    while True:
        line=input_data.pop(0)
        if line.decode("utf-8")[:2]!=' #': break
        num_head+=1
    # col labs
    line_splits=line.decode("utf-8").lower().split()[3:-1] 
    out_data_dict={'date':[]}
    for name in line_splits:
        out_data_dict[name]=[]
    # Read all out_data_dict.
    for _line_bytes in input_data:
        _line = _line_bytes.decode("utf-8")
        if _line[-2]=='|':continue 
        p=_line.split()
        out_data_dict['date'].append(parse(' '.join(p[0:2])))
        for i,name in enumerate(line_splits):
            out_data_dict[name].append(float(p[i+3]))
    # convert to DF!
    data_df = pandas.DataFrame.from_dict(out_data_dict)
    return data_df
