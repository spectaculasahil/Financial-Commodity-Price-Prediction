import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

script_dir = os.path.dirname(os.path.abspath(__file__)) # get dir of script
csv_file = os.path.join(script_dir, 'data', 'Nat_Gas.csv') # path to csv

data = pd.read_csv(csv_file) # read csv

data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%y') # convert dates

data['Time'] = range(len(data)) # add time col

X = data[['Time']] # feature: time
y = data['Prices'] # target: prices

model = LinearRegression() # init model
model.fit(X, y) # fit model

def estimate_price(date_str):
    start_date = data['Dates'].min() # earliest date
    new_date = pd.to_datetime(date_str, format='%m/%d/%y') # convert date
    days_diff = (new_date - start_date).days # diff in days
    time_value = days_diff / 30.0 # approx months
    pred = model.predict([[time_value]]) # predict price
    return round(pred[0], 2) # return rounded

plt.figure(figsize=(8, 4)) # fig size
plt.scatter(data['Dates'], data['Prices'], label='Historical Prices') # scatter plot
line_values = model.predict(X) # predicted line values
plt.plot(data['Dates'], line_values, label='Fitted Line', color='red') # plot fitted line
plt.title('Natural Gas Prices Over Time') # title
plt.xlabel('Date') # x label
plt.ylabel('Price') # y label
plt.legend() # legend
plt.show() # show plot

print("Estimated price on May 15, 2023:", estimate_price('05/15/23')) # print pred
print("Estimated price on Jan 31, 2025:", estimate_price('01/31/25')) # print pred