#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import zipfile
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# # Data  collection done from extracting data from the binance site.
# 
# ###  Data URL

# In[2]:


url=["https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-11.zip",
     
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-10.zip",
     
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-09.zip",

"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-08.zip",
     
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-07.zip",
     
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-06.zip",
     
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-05.zip",
     
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-04.zip",
   
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-03.zip",
     
"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-02.zip",

"https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-01.zip"]
     


# # OCTOBER data

# In[3]:


octdata = pd.read_csv(r"C:\Users\seasi\Downloads\XRPUSDT-1d-2023-10.csv")  


# In[4]:


octdata


# # January data

# In[5]:


url_january = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-01.zip"
response = requests.get(url_january)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-01.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-01.zip', 'r') as zip_ref:
        zip_ref.extractall('jandata')
    Januarydata = pd.read_csv('jandata/XRPUSDT-1d-2023-01.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')



# # February data

# In[6]:


url_february = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-02.zip"
response = requests.get(url_february)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-02.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-02.zip', 'r') as zip_ref:
        zip_ref.extractall('febdata')
    februarydata = pd.read_csv('febdata/XRPUSDT-1d-2023-02.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')
    


# # March data

# In[7]:


url_march = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-03.zip"
response = requests.get(url_march)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-03.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-03.zip', 'r') as zip_ref:
        zip_ref.extractall('marchdata')
    marchdata = pd.read_csv('marchdata/XRPUSDT-1d-2023-03.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')
    


# # April data

# In[8]:


url_april = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-04.zip"
response = requests.get(url_april)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-04.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-04.zip', 'r') as zip_ref:
        zip_ref.extractall('aprildata')
    aprildata = pd.read_csv('aprildata/XRPUSDT-1d-2023-04.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # May data

# In[9]:


url_may = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-05.zip"
response = requests.get(url_may)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-05.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-05.zip', 'r') as zip_ref:
        zip_ref.extractall('maydata')
    maydata = pd.read_csv('maydata/XRPUSDT-1d-2023-05.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # june data

# In[10]:


url_june = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-06.zip"
response = requests.get(url_june)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-06.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-06.zip', 'r') as zip_ref:
        zip_ref.extractall('junedata')
    junedata = pd.read_csv('junedata/XRPUSDT-1d-2023-06.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # july data

# In[11]:


url_july = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-07.zip"
response = requests.get(url_july)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-07.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-07.zip', 'r') as zip_ref:
        zip_ref.extractall('julydata')
    julydata = pd.read_csv('julydata/XRPUSDT-1d-2023-07.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # August data

# In[12]:


url_august = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-08.zip"
response = requests.get(url_august)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-08.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-08.zip', 'r') as zip_ref:
        zip_ref.extractall('augustdata')
    augustdata = pd.read_csv('augustdata/XRPUSDT-1d-2023-08.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # september data

# In[13]:


url_september = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-09.zip"
response = requests.get(url_september)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-09.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-09.zip', 'r') as zip_ref:
        zip_ref.extractall('septemberdata')
    septemberdata = pd.read_csv('septemberdata/XRPUSDT-1d-2023-09.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # october data

# In[14]:


url_october = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-10.zip"
response = requests.get(url_october)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-10.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-10.zip', 'r') as zip_ref:
        zip_ref.extractall('octoberdata')
    octoberdata = pd.read_csv('octoberdata/XRPUSDT-1d-2023-10.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # November data

# In[15]:


url_november = "https://data.binance.vision/data/spot/monthly/klines/XRPUSDT/1d/XRPUSDT-1d-2023-11.zip"
response = requests.get(url_november)

if response.status_code == 200:
    with open('XRPUSDT-1d-2023-11.zip', 'wb') as file:
        file.write(response.content)
    with zipfile.ZipFile('XRPUSDT-1d-2023-11.zip', 'r') as zip_ref:
        zip_ref.extractall('novemberdata')
    novemberdata = pd.read_csv('novemberdata/XRPUSDT-1d-2023-11.csv', header=None)
else:
    print(f'Failed to retrieve file: {response.status_code}')


# # inserting header

# In[16]:


column_names=["Open_time","Open","High","Low","Close","Volume","Close_time","Quote_asset_volume","Number_of_trades","Takers_buy_base_asset_vol","Taker_buy_quote_asset_vol","ignore"]


# In[17]:


All_data = pd.concat([Januarydata,februarydata,marchdata,aprildata,maydata,junedata,julydata,augustdata,septemberdata,octoberdata,novemberdata])
All_data.columns = column_names
All_data.head(3)


# In[18]:


All_data.info()


# In[19]:


total_missing_count = All_data.isna().sum().sum()
print(f"Total missing values: {total_missing_count}")


# In[20]:


duplicate_count = All_data.duplicated().sum()
print(f"Number of duplicated rows: {duplicate_count}")


# # Timestamp to datetime

# In[21]:


#function to convert timestamp to datetime
def convert_timestamp(timestamp_ms):
    timestamp_s = timestamp_ms / 1000
    date = datetime.datetime.fromtimestamp(timestamp_s)
    return date

#apply the function to the timestamp column
All_data['Open_time'] = All_data['Open_time'].apply(convert_timestamp)
All_data['Close_time'] = All_data['Close_time'].apply(convert_timestamp)
All_data


# # unit normalization

# In[ ]:





# In[22]:


# Convert "Quote_asset_volume" to numeric, ignoring errors for non-numeric values
All_data["Quote_asset_volume"] = pd.to_numeric(All_data["Quote_asset_volume"], errors='coerce')

# Apply formatting only to numeric values
All_data["Quote_asset_volume"] = All_data["Quote_asset_volume"].apply(lambda x: '{:.2f}'.format(x) if pd.notna(x) else x)

# Assuming "Taker_buy_quote_asset_vol" is numeric, do the same for that column
All_data["Taker_buy_quote_asset_vol"] = pd.to_numeric(All_data["Taker_buy_quote_asset_vol"], errors='coerce')
All_data["Taker_buy_quote_asset_vol"] = All_data["Taker_buy_quote_asset_vol"].apply(lambda x: '{:.2f}'.format(x) if pd.notna(x) else x)

All_data.head(3)


# # Descriptive analysis
# 
# ## Statistics Price and data

# In[23]:


price_stats = All_data['Close'].describe()
volume_stats = All_data['Volume'].describe()

stats=pd.DataFrame({"Close_price":price_stats, "Volume":volume_stats})
stats


# In[24]:


# stats from 'Close' and 'Volume' columns  in All_data DataFrame
price_stats = All_data['Close'].describe()
volume_stats = All_data['Volume'].describe()

# Create a DataFrame with descriptive statistics
stats = pd.DataFrame({
    "Close_price": price_stats,
    "Volume": volume_stats
})

# Display the resulting DataFrame
stats


# # Statistics daily returns

# In[25]:


#Calculate Daily Returns
daily_returns = All_data['Close'].pct_change().dropna()

#Volatility (Standard deviation of daily Returns)
volatility = daily_returns.std()

#Skewness
skewness = daily_returns.skew()

#Kurtosis
kurtosis = daily_returns.kurtosis()

# sharpe Ratio
# Assuming a risk-free rate of 5% (0.05) 
risk_free_rate = 0.05
average_daily_return = daily_returns.mean()
sharpe_ratio = (average_daily_return - risk_free_rate)/volatility

# Displaying the results
results = pd.DataFrame({
    "Volatility": [volatility],
    "Skewness":[skewness],
    "Kurtosis":[kurtosis],
    "Sharpe Ratio": [sharpe_ratio]
})
results


# # Calculating RSI

# In[31]:


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi  # Return calculated RSI values

# Assuming 'Close' column exists in your DataFrame
All_data['RSI'] = calculate_rsi(All_data['Close'])

# Display the first few rows of the DataFrame
print(All_data.head())

# Display the 'RSI' column
print(All_data['RSI'])






# In[ ]:





# In[32]:


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # Corrected from delta > 0 to delta < 0
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


All_data['RSI'] = calculate_rsi(All_data['Close'])

# Display the RSI
print(All_data["RSI"])


# # Exploratory analysis
# 
# ### Price Trend

# In[33]:


plt.figure(figsize=(12,6))
plt.plot(All_data['Close_time'],All_data['Close'])
plt.title('price trend Over Time')
plt.xlabel("Date")
plt.ylabel('Closing Price')
plt.show()


# # Volume Trend

# In[34]:


plt.figure(figsize=(12,6))
plt.plot(All_data['Close_time'],All_data['Volume'])
plt.title('Volume Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()


# # Volume by Days of the month

# In[35]:


All_data['Day'] = All_data['Close_time'].dt.day
average_volume_by_day = All_data.groupby('Day')['Volume'].mean()
# Calculate the overall average volume for comparison
overall_average_volume = All_data['Volume'].mean()
plt.figure(figsize=(12,6))
average_volume_by_day.plot(kind='bar')
plt.axhline(y=overall_average_volume,color='g',linestyle='--',linewidth=2)
# Adding text to indicate the average line
plt.text(0, overall_average_volume, f'Average: {overall_average_volume:.2f}',color='g',va='bottom')
plt.title('Average Volume by Day of Month')
plt.xlabel('Day of Month')
plt.ylabel('Average Volume')
plt.show()


# # Price fluntuations by months

# In[36]:


All_data['Month']=All_data['Close_time'].dt.month
average_price_by_month = All_data.groupby('Month')['Close'].mean()
plt.figure(figsize=(12,6))
average_price_by_month.plot(kind='bar')
plt.title('Average Price by Month')
plt.xlabel('Month')
plt.ylabel('Average closing Price')
plt.show()


# In[ ]:





# # Correlation between Volume and Quote Asset Volume

# In[37]:


# Convert Volume and Quote Asset Volume to numeric types (float)
All_data['Volume'] = pd.to_numeric(All_data['Volume'], errors='coerce')
All_data['Quote Asset Volume'] = pd.to_numeric(All_data['Quote_asset_volume'],  errors='coerce')

# Drop any rows with NaN values that resulted from conversion
All_data.dropna(subset=['Volume', 'Quote Asset Volume'], inplace=True)


# In[38]:


# what is the correlation and p-value
correlation,p_value = pearsonr(All_data['Volume'],All_data['Volume'])

print("Correlation:", correlation)
print("P-value:",p_value)


# # Trend in Number of Trades

# In[39]:


plt.figure(figsize=(12,6))
plt.plot(All_data['Close_time'],All_data['Number_of_trades'])
plt.title('Trend in Number of Trades Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Trades')
plt.show()


# In[42]:


plt.figure(figsize=(12, 6))
plt.plot(All_data['Close_time'], All_data['Number_of_trades'])
plt.show()


# In[ ]:




