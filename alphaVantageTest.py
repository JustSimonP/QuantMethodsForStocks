from pandas_datareader import data
import pandas as pandas
from datetime import datetime
import matplotlib.pyplot as plt
import os

start_date= '2014-01-01'
end_date = '2020-01-07'
panel_data = data.DataReader("AAPL", "av-daily", start=start_date,
                    end=end_date,
                    api_key='3OWVKI5MGERYCS8A')

panel_data.head(9)

# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the panel_data.
close = panel_data['close']

# Getting all weekdays between 01/01/2000 and 12/31/2016
#Getting all the weekdays is achieved by passing the freq=’B’
all_weekdays = pandas.date_range(start=start_date, end=end_date, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

# Reindexing will insert missing values (NaN) for the dates that were not present
# in the original set. To cope with this, we can fill the missing by replacing them
# with the latest available price for each instrument.
close = close.fillna(method='ffill')
print(all_weekdays)

# Get the MSFT timeseries. This now returns a Pandas Series object indexed by date.
msft = close.loc[0,'AAPL']

# Calculate the 20 and 100 days moving averages of the closing prices
short_rolling_msft = msft.rolling(window=20).mean()
long_rolling_msft = msft.rolling(window=100).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(msft.index, msft, label='MSFT')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='20 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label='100 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()