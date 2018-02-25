import pandas as pd
from WindPy import *
import matplotlib.pyplot as plt
import datetime
import numpy as np


def Wind2Df(wind_data):
    df = pd.DataFrame(wind_data.Data).T
    df.columns = wind_data.Fields
    df.index = wind_data.Times
    return df

def is_due_date(date):
    if 15 <= date.day <= 21:
        if date.weekday() == 4:
            return True
    return False

def gen_due_date(year, month):
    date0 = datetime.date(year,month, 1)
    for i in range(31):
        date0 = date0 + datetime.timedelta(1)
        if is_due_date(date0):
            return date0
    return None

def get_due_date(date):
    due_date_this_month = gen_due_date(date.year, date.month)
    if date.month != 12:
        due_date_next_month = gen_due_date(date.year, date.month + 1)
    else:
        due_date_next_month = gen_due_date(date.year - 1, 1)
    if date > due_date_this_month:
        return due_date_next_month
    else:
        return due_date_this_month

w.start()

df_if00 = Wind2Df(w.wsi("IF00.CFE", "close, volume", "2016-02-01 09:30:00", "2017-08-16 13:48:43", "periodstart=09:30:00;periodend=15:00:00"))
df_if01 = Wind2Df(w.wsi("IF01.CFE", "close, volume", "2016-02-01 09:30:00", "2017-08-16 13:48:43", "periodstart=09:30:00;periodend=15:00:00"))
df_if00.columns = ['close0', 'volume0']
df_if01.columns = ['close1', 'volume1']

df_all = pd.merge(df_if00, df_if01, left_index=True, right_index=True)

df_all['diff'] = df_all.close0 - df_all.close1
df_all['date'] = map(lambda x: x.date(), df_all.index)
df_all['due_time'] = map(lambda x: get_due_date(x.date()), df_all.index)
df_all['t'] = map(lambda x,y: (y - x.date() ).days, df_all.index, df_all.due_time)


df_all1 = df_all.copy()
df_all1.index = range(df_all1.shape[0])
df_all1['0day_diff'] = map(lambda y,x: y if x == 0 else np.nan, df_all1['diff'], df_all1['t'])
df_all1['1day_diff'] = map(lambda y,x: y if x == 1 else np.nan, df_all1['diff'], df_all1['t'])
df_all1['2day_diff'] = map(lambda y,x: y if x == 2 else np.nan, df_all1['diff'], df_all1['t'])

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
df_all1[['volume0', 'volume1']].plot(ax = ax1)
df_all1[['diff','0day_diff','1day_diff','2day_diff' ]].plot(ax = ax2)
plt.show()




