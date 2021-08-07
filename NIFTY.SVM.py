#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional 
import math
from sklearn.metrics import mean_squared_error
# from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import datetime
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore') #Supressing warnings
pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
pd.set_option("display.max_colwidth",5000)


# # EDA

# In[2]:


#First, we get the data
dataset = pd.read_csv("DataFrame.csv", index_col='Date', parse_dates=['Date'])
dataset.head()


# In[3]:


dataset=dataset.drop(["Type"],1)
dataset["Time"]=dataset["Time"].str.replace(":",".").astype("float")

dataset.head()


# In[4]:


dataset.describe(percentiles=(.25,.5,.75,.95,.99))


# In[5]:


# Show the info 
dataset.info()


# In[6]:


#checking for null values if any in the datset 

round(dataset.isnull().sum()/len(dataset.index)*100,2)


# In[7]:


#Finding Correlation between columns

sn.pairplot(dataset, x_vars=['open','high','low'],y_vars=["close"],aspect=0.9,size=4,diag_kind=None)
plt.show()


# In[8]:


figure, axis = plt.subplots(2, 2,figsize=(20,12))
    
# For Opening Price
axis[0, 0].plot(dataset['open'],color='b')
axis[0, 0].set_title("Opening Price")
  
# For Closing Price
axis[0, 1].plot(dataset['high'],color='y')
axis[0, 1].set_title("high")
  
# For High
axis[1, 0].plot(dataset['close'],color='r')
axis[1, 0].set_title("close")
  
# For Low
axis[1, 1].plot(dataset['low'],color='g')
axis[1, 1].set_title("Low")
  
# Combine all the operations and display
plt.show()


# In[9]:


#Visually show the the stock price

plt.figure(figsize=(20.2,4.5))
plt.plot(dataset["close"],label="Close")
plt.title("close price History")
plt.xlabel("Date",)
plt.xticks(rotation=45)
plt.ylabel("Price in RS ")
plt.show()


# ## SVM

# In[10]:


#Reading Data 
df=pd.read_csv("DataFrame.csv")
print(df.shape)


# In[11]:


#checking rows and columns

print(df.shape)
df.tail()


# In[12]:


df=df.drop(["Type"],1)


# In[13]:


#DAte column Feature enginearing 

df["Datetime"]=df["Date"].astype("str")+df["Time"]
df["Datetime"]=df["Datetime"].str.replace(":","")
df=df.drop(["Date","Time"],1)


m=[]
t=[]
d=[]

for i in df["Datetime"]:
    m.append(i[4:6])
    t.append(i[8:])
    d.append(i[6:8])
    
df.drop("Datetime",1,inplace=True)
df["month"]=m
df["time"]=t
df["day"]=d
df["month"]=pd.DataFrame(df["month"])
df["time"]=pd.DataFrame(df["time"])
df["day"]=pd.DataFrame(df["day"])
df.head()



# In[14]:


#extarcting last row for comparing as actual price 

Actual_Price=df.tail(1)

Actual_Price


# In[15]:


#Prepare the data for training the SVR models

df=df.head(len(df)-1)


# In[16]:


print(df.head())


# In[17]:


df["day"]


# In[18]:


#Train & Test Data 

k=df[["day","month","time","open"]]
close_price=df["close"]

#Backup process 
b=k.copy()

# #Scaling 

# sc=MinMaxScaler()
# print(k)
# Vars=["day","month","time","open"]
# k[Vars]=sc.fit_transform(k[Vars])
# k.head()


# In[19]:


#Model Bulinding 

k=np.array(k)
rbf_svr=SVR(kernel="poly")
rbf_svr.fit(k,close_price)


# In[20]:


y_pred = rbf_svr.predict(k)


# In[21]:


y_pred


# In[22]:


# Model Testing


#Taking input from user 
day=input()  # enter day "31" default 
month=input() #enter month "03" default
time=input()  #time  15:17 onwards 
Open=float(input()) # First day price where you started investing (open price)

#Storing values 
data1 =day
data2 =month
data3 =time
data4 =Open


#Coverting to Pandas Dataframe 
data=pd.DataFrame({"day":data1,"month":data2,"time":data3,"open":data4},index=[0])


# #Scaling/Training model 
# data[Vars]=sc.transform(data[Vars])
y_pred = rbf_svr.predict(data)

#Predicitng future per minute price for 31 March 2021 only 

y_pred 


# In[23]:


#Converting to pickle File 

with open('Nifty-SVM-all', 'wb') as picklefile:
    pickle.dump(rbf_svr,picklefile)


# In[24]:


with open('Nifty-SVM-all', 'rb') as training_model:
    model1 = pickle.load(training_model)


# In[25]:


plt.figure(figsize=(16,8))
plt.plot(b.index,close_price,color="Red",label="Actual Price")
plt.plot(b.index,rbf_svr.predict(k),color="blue",label="Prediction Price")
plt.legend()
plt.show;


# In[26]:


#Accuracy

Accuracy=rbf_svr.score(k,close_price)


# In[27]:


Accuracy


# In[28]:


predicted=rbf_svr.predict(k)


# ## **Finding % Increase and Classifying the price to labels "up" and "down"** 

# In[29]:


results=b
results["actual_close"]=close_price
results["Predicted_close"]=predicted
results["Predicted_close"]=round(results["Predicted_close"],2)
results["%_Change"]=results.Predicted_close.pct_change(axis=0)
results["Class"]=np.where(results["Predicted_close"].shift(-1)>results["Predicted_close"],"up","Down")
results.head()


# ##  Augumenting Features like Buy/Sell indicators (Moving Avg's , MACD, Signal Line)

# In[30]:


#calculating the short term exponential moving average (EMA) 
EMA_12=results.Predicted_close.ewm(span=12,adjust=False).mean()                      #for 12 days                 
#calculating the Long term exponential moving average (EMA)
EMA_26=results.Predicted_close.ewm(span=26,adjust=False).mean()                        #for 26 days  
#Calculate the MACD line 
MACD=EMA_12-EMA_26
#Calculate the signal line 
signal=MACD.ewm(span=9,adjust=False).mean()


# Adding above indicators to the datset


results["EMA_12"]=EMA_12
results["EMA_26"]=EMA_26
results["MACD"]=MACD
results["Signal_line"]=signal


results.reset_index(drop=True,inplace=True)
results.head(15)


# In[31]:


#plot te chart b/t MACD and  Signal line 

plt.figure(figsize=(14.5,5))
plt.plot(results.index,MACD,label="MACD",color="red")
plt.plot(results.index,signal,label="Signal line",color="blue")
plt.legend(loc="upper left")
plt.xticks(rotation=45)
plt.show()


# In[32]:


# Creating a Function to signal whan to buy and sell an asset

def buy_sell(signal):
    Buy=[]
    Sell=[]
    status=[]
    flag=-1

    for i in range(0,len(signal)):
        if signal["MACD"][i]> signal["Signal_line"][i]:
            Sell.append(np.nan)
            if flag!=1:
                Buy.append(signal["Predicted_close"][i])
                flag=1
            else:
                Buy.append(np.nan)
        elif signal["MACD"][i]< signal["Signal_line"][i]:
            Buy.append(np.nan)
            if flag!=0:
                Sell.append(signal["Predicted_close"][i])
                flag=0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    return(Buy,Sell)


# In[33]:


#Creating Buy &  Sell Col 

a=buy_sell(results)

results["Buy"]=a[0]
results["Sell"]=a[1]


# In[34]:


#Visually Show the stock buy and Sell Signal 
plt.figure(figsize=(15.5,8))
plt.scatter(results.index,results["Buy"],color="green",label="Buy",marker="^",alpha=1)
plt.scatter(results.index,results["Sell"],color="red",label="Sell",marker="v",alpha=1)
plt.plot(results["Predicted_close"],label="Close Price",alpha=0.35,color="orange")

plt.title("Buy & Sell Signals for predicted Price")
plt.xlabel("month")
plt.xticks(rotation=45)
plt.ylabel("Predicted price in Rupees")
plt.tick_params(axis='x', which='minor', labelsize=80)
plt.legend(loc="upper right");


# In[35]:


results.head()


# In[ ]:




