import datetime
import pandas

def create_url_form(date_range,base_url,\
                 email='test@gmail.com', output="AE"):
    """
    create url based on a date range
    
    Written By - Bharat Kunduri (04/2020)
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
    data+='Output={0[Output]}&Out+format='.format(url_data)
    data+='IAGA2002&Email={0[Email]}'.format(url_data)
    return urllib.request.urlopen(base_url+'?'+data) 

def set_kp_url(date_range, email='test@gmail.com'):
    """
    create url for Kp based on a date range
    
    Written By - Bharat Kunduri (05/2020)
    """
    import urllib.parse, urllib.request

    # Kp is downloaded in months! So get the relevant dates
    # get the year/month from dates
    if type(date_range) == list:
        start_date = date_range[0]
        end_date = date_range[1]
    elif type(date_range) == datetime.datetime:
        start_date = date_range
        end_date = date_range
    else:
        print("wrong inputs, need a list of datetime or one datetime obj!")
        return None
    year_start = start_date.year
    year_end = end_date.year
    month_start = start_date.month
    month_end = end_date.month
    # Setup the url data
    url_data = {} 
    if year_start >= 2000:
        url_data['SCent'] = 20
    else:
        url_data['SCent'] = 19
    url_data['STens'] = int(('%d'%year_start)[2])
    url_data['SYear'] = int(('%d'%year_start)[3])
    url_data['SMonth'] = '%02i' % month_start
    # End Time:
    if year_end >= 2000:
        url_data['ECent'] = 20
    else:
        url_data['ECent'] = 19
    url_data['ETens'] = int(('%d'%year_end)[2])
    url_data['EYear'] = int(('%d'%year_end)[3])
    url_data['EMonth'] = '%02i' % month_end

    # wrap up the url
    email = urllib.parse.quote(email)
    data=urllib.parse.urlencode(url_data)
    base_url='http://wdc.kugi.kyoto-u.ac.jp/cgi-bin/kp-cgi'#
    data='%s=%2i&%s=%i&%s=%i&%s=%s&%s=%2i&%s=%i&%s=%i&%s=%s&%s=%s' % \
        ('SCent', url_data['SCent'], \
         'STens', url_data['STens'], \
         'SYear', url_data['SYear'], \
         'SMonth',url_data['SMonth'],\
         'ECent', url_data['ECent'], \
         'ETens', url_data['ETens'], \
         'EYear', url_data['EYear'], \
         'EMonth',url_data['EMonth'],\
         'Email' ,email)
    return urllib.request.urlopen(base_url, data.encode())

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

def kp_to_df(input_data):
    """
    raw Kp data to DF
    """
    import numpy

    # In Python3 data is returned in bytes!
    if type(input_data[0])==bytes:
        for i in range(len(input_data)):
            input_data[i] = input_data[i].decode()
    
    # Discard the header
    if input_data[0][1:5]=='HTML':
        input_data.pop(0);input_data.pop(0);input_data.pop(-1)
    if input_data[0][0:4]=='YYYY':
        input_data.pop(0)

    data_len = len(input_data)
    date = numpy.zeros(8*data_len, dtype=object)
    bin_start_date = numpy.zeros(8*data_len, dtype=object)
    bin_end_date = numpy.zeros(8*data_len, dtype=object)
    kp   = numpy.zeros(8*data_len)
    hr1  = [0,3,6,9,12,15,18,21,0]
    hrs  = [1,4,7,10,13,16,19,22]
    frac_to_dec_dict = {' ':0.0, '-':-1./3., '+':1./3.}
    # Parse input_data.
    for i,line in enumerate(input_data):
        yy = int(line[0:4])
        mm = int(line[4:6])
        dd = int(line[6:8])
        for j in range(8):
            kp[8*i+j]  = float(line[9+2*j])+frac_to_dec_dict[line[10+2*j]]
            date[8*i+j]= datetime.datetime(yy,mm,dd,hrs[j],30,0)
            bin_start_date[8*i+j]= datetime.datetime(yy,mm,dd,hr1[j],0,0)
            # its a 3 hour index!
            bin_end_date[8*i+j]= bin_start_date[8*i+j]+datetime.timedelta(hours=3)
    # convert to a dataframe
    if date.shape[0] > 0:
        return pandas.DataFrame(\
                            {
                             'date': date,\
                             'kp': kp,
                             'bin_start_date': bin_start_date,
                             'bin_end_date': bin_end_date,
                            }\
                        )
    return None
