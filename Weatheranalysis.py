from pyspark.sql import SparkSession
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
import pyspark.sql.functions as sqlf



# Create a sql context
sc = SparkContext()
sqlc = SQLContext(sc)

# Analysis on the latest 2 years
years = range(2017, 2019)

# Yearly stats
for year in years:
    # get data as raw text
    txtfile = sc.textFile('./%s.csv' % year)
    # split attribute values using commas
    data = txtfile.map(lambda x: x.split(','))
    # create table
    table = data.map(lambda r: Row(station=r[0], date=r[1], ele=r[2], val=int(r[3]), m_flag=r[4], q_flag=r[5], s_flag=r[6], obs_time=r[7]))
    # create dataframe a table 
    df = sqlc.createDataFrame(table)

    # Handle abnomalities and missing data
    clean_df = df.filter(df['q_flag'] == '')

    print("\nYear %s Stats:\n" % year)
    # 1. Average min
    res = clean_df.filter(clean_df['ele'] == 'TMIN').groupby().avg('val').first()
    print('Avg. Min Temp = %.2f degrees Celsius' % (res['avg(val)'] / 10.0))

    # 1. Average max
    res = clean_df.filter(clean_df['ele'] == 'TMAX').groupby().avg('val').first()
    print('Avg. Max Temp = %.2f degrees Celsius' % (res['avg(val)'] / 10.0))

    # 2. Max TMAX
    res = clean_df.filter(clean_df['ele'] == 'TMAX').groupby().max('val').first()
    print('Max TMAX value = %.2f degrees Celsius' % (res['max(val)'] / 10.0))

    # 2. Min TMIN
    res = clean_df.filter(clean_df['ele'] == 'TMIN').groupby().min('val').first()
    print('Min TMIN value = %.2f degrees Celsius' % (res['min(val)'] / 10.0))

    # 3. Five distinct hottest weather stations
    res = clean_df.filter(clean_df['ele'] == 'TMAX').sort(sqlf.desc('val')).groupBy(clean_df['station']).agg(sqlf.max('val')).sort(sqlf.desc('max(val)')).limit(5).collect()
    print("Top 5 distinct hottest stations")
    for i in res:
        print('Station:%s\tTemperature:%.2f degrees Celsius' % (i.station, float(i['max(val)']) / 10.0))


    # 3. Five hottest weather stations only by temperature
    res = clean_df.filter(clean_df['ele'] == 'TMAX').sort(sqlf.desc('val')).limit(5).collect()
    print("Top 5 hottest weather stations only by temperature")
    for i in res:
        print('Station:%s\tTemperature:%.2f degrees Celsius' % (i.station, float(i['val']) / 10.0))

    # 3. Five distinct coldest weather stations
    res = clean_df.filter(clean_df['ele'] == 'TMIN').sort(sqlf.asc('val')).groupBy(clean_df['station']).agg(sqlf.min('val')).sort(sqlf.asc('min(val)')).limit(5).collect()
    print("Top 5 distinct coldest stations")
    for i in res:
        print('Station:%s\tTemperature:%.2f degrees Celsius' % (i.station, float(i['min(val)']) /10.0))

    # 3. Five coldest weather stations only by temperature
    res = clean_df.filter(clean_df['ele'] == 'TMIN').sort(sqlf.asc('val')).limit(5).collect()
    print("Top 5 coldest stations only by temperature")
    for i in res:
        print('Station:%s\tTemperature:%.2f degrees Celsius' % (i.station, float(i['val']) / 10.0))

# Aggregate statistics
# 4. Hottest and coldest weather stations on entire data
txtfile1 = sc.textFile('./20??.csv')
data = txtfile1.map(lambda x: x.split(','))
table = data.map(lambda r: Row(station=r[0], date=r[1], ele=r[2], val=int(r[3]), m_flag=r[4], q_flag=r[5], s_flag=r[6], obs_time=r[7]))
df = sqlc.createDataFrame(table)
clean_df = df.filter(df['q_flag'] == '')
# The code filters the dataframe by removing all rows where the q_flag column is null.

# hottest day and weather station
res = clean_df.filter(clean_df['ele'] == 'TMAX').sort(sqlf.desc('val')).first()
print("Hottest station: %s on %s with temperature:%.2f degrees Celsius" % (res.station, res.date, float(res['val']) / 10.0))

# coldest day and weather station
res = clean_df.filter(clean_df['ele'] == 'TMIN').sort(sqlf.asc('val')).first()
print("Coldest Station: %s on %s with temperature:%.2f degrees Celsius" % (res.station, res.date, float(res['val']) / 10.0))

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


spark = SparkSession.builder.appName('ml-bank').getOrCreate()
df = spark.read.csv('DelhiWeather.csv', header = True, inferSchema = True)

df = df.select('datetime_utc', '_dewptm', '_pressurem', '_tempm')


df = df.toPandas()
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])


df = df.set_index('datetime_utc')

df = df.replace('', np.nan, regex=True)

df = df.fillna(method='ffill')     #Fills the last valid observation to the next missing value
y = df['_tempm'].resample('MS').mean().ffill()

# Parameter Config for monthly predictions
mod = sm.tsa.statespace.SARIMAX(y,order=(1, 0, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()


pred_uc = results.get_forecast(steps=30)
pred_ci = pred_uc.conf_int()

ax = y['2014':].plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Temperature')

plt.legend()
plt.show()
exit