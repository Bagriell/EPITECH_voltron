"""
USAGE:
    * Create virtual environment:
        python3 -m venv voltron-sensor-ai-env
    * Load virtual env(windows style):
        voltron-sensor-ai-env\Scripts\activate
    * install requirements:
        pip install -r requirements.txt
    * add your csv data 'data.csv' next to build_model script
    * python build_model.py

    => output "serialized_forecasting_model.json" model.
"""


import pandas as pd
from datetime import datetime, timedelta
from prophet import Prophet
from prophet.serialize import model_to_json
import sys

# read data
try:
    df = pd.read_csv('data.csv')
except FileNotFoundError:
    sys.exit("Please place a 'data.csv' file in the script folder/")


# format and clean data
df.dropna(inplace=True)
start = datetime(1970, 1, 1)  # Unix epoch start time
df['time'] =  df['ts'].apply(lambda x: start + timedelta(seconds=x))
df.replace(['b8:27:eb:bf:9d:51', '00:0f:00:70:91:0a', '1c:bf:ce:15:ec:4d'], ['Device1','Device2','Device3'], inplace=True)


# groupby devices
df_device2 = df[df.device == 'Device2']
df_prophet = pd.DataFrame({"ds": df_device2['time'], "y": df_device2['temp']})

# create model
model = Prophet()
model.fit(df_prophet)


with open('serialized_forecasting_model.json', 'w') as fout:
    fout.write(model_to_json(model))
