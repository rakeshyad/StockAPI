
# Intraday -Stock-Trading DST ML Project


![image](https://user-images.githubusercontent.com/78646864/122196843-38116a80-ceb5-11eb-85ec-58d7652135c4.png)







# AIM:

The principle focus of our project is to perform data analysis and train a model using the most popular Machine Learning algorithm – SVM in order to analyze the day-to-day price movements in stock market.


# ABSTRACT:

Intraday trading, also called day trading, is the buying and selling of stocks and other financial instruments within the same day. In other words, intraday trading means all positions are squared-off before the market closes and there is no change in ownership of shares as a result of the trades.
Until recently, people perceived day trading to be the domain of financial firms and professional traders. But this has changed today, thanks to the popularity of electronic trading and margin trading.
Today, it is extremely easy to start day trading. If you want to start, read on to understand the basics of intraday trading:
 
# PROJECT OBJECTIVE:

Building a model based on supervised learning which have ability to Precisely predict the price movement of stocks is the key to profitability in trading. In light of the increasing availability of financial data, prediction of price movement in the financial market with machine learning has become a topic of interests for both investors and researchers. Our group aims to predict whether the price in the next minute will go up or down, using the time series data of stock price, technical-analysis indicators, and trading volume


# OVERVIEW:

Data Segmentation and Data Cleaning
Exploratory Data Analysis using python’s data visualization
Training the model based on the historical data available
Deployment of model using Heroku Platform and Flask framework


# DATASET:

There are two datasets:

1.	MSFT-Stock-Trading

2.	NIFTY-Stock-Trading

So, our data set comprises of 7 columns, out of which 6 are features and 1 is target.


# Features


Open — It is the opening price of a shares for that day

High — It is the highest price the shares have touched in throughout day

Low — It is the lowest price the shares have fallen to in throughout day

Date — The date of the observation, mostly the index of our data

Volume — The number of shares sold in throughout day


# Target:

Close — It is the closing price of the shares for that day
 
# DATA SEGMENTATION AND DATA CLEANING:
In this project, we have prepared a processed dataset by and collected the clear-cut data available online.
Using panda’s data frame, we have calculated the mean of every column.
By using the fill-na we have filled all the cells with empty values.



# EXPLORATORY DATA ANALYSIS:

Correlations (NIFTY)

Correlations (MSFT)


Training the model 

Feature Engineering


# Model Building

SVM

Model Accuracy

Conclusion:

# MODEL DEPLOYEMENT

Preparing Pickle Files for Both Model


Framework used “Flask”


Rendering Inputs From “User” using request ( )



# Scripts/Language Used for deployment

Python: HTML: CSS


# App Details

**App	Name:** "stockpriceapi"


**App Link:** https://stockpriceapi.herokuapp.com/

