#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

plt.style.use('seaborn-whitegrid')

#load the data

data = pd.read_csv("data/Dogecoin.csv")



print("Shape of Dataset is: ",data.shape,"\n")

print(data.head())

print(data.describe())

#Drop Missing Values

data = data.dropna()

#Data Visualization

plt.figure(figsize=(10, 4))

plt.title("DogeCoin Price INR")

plt.xlabel("Date")

plt.ylabel("Close")

plt.plot(data["Close"])

plt.show()

#Build the Model

from autots import AutoTS

model = AutoTS(forecast_length=10, frequency='infer',
               ensemble='simple', drop_data_older_than_periods=200)
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
 
prediction = model.predict()
forecast = prediction.forecast
print("DogeCoin Price Prediction")
print(forecast)