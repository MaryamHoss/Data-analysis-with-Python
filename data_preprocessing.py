
#Lesson 1: data wrangling

import pandas as pd
import numpy as np
import matplotlib as plt

url='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data' #where we download the data from 
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]  #The data doesn't come with headers, these are the headers provided on the website
df=pd.read_csv(url,names=headers) #load the data with these headers

df.head(5)  #print the 5 first rows of the data

df.replace('?',np.nan,inplace=True)  # Python automatically replaces the values that are not provided with ?, to be able
                                    # to work with this data, we need to replace these ? with NaN
df.head(5)

missing_data = df.isnull()
missing_data.head(5)

for c in missing_data.columns:
    print(c)
    print(missing_data[c].value_counts())
    print("")    

#deal with missing data
# replace by average    
av=df["normalized-losses"].astype("float").mean()
df["normalized-losses"].replace(np.nan, av, inplace=True)

#replace by the most common
df['num-of-doors'].value_counts().idxmax()
df['num-of-doors'].replace(np.nan,df['num-of-doors'].value_counts().idxmax(),inplace=True)


#drop the rows
df.dropna(axis=0,subset=["price"],inplace=True)

df.reset_index(drop=True,inplace=True)

# check the types to see if they are correct

df.dtypes

df[["bore","stroke"]]=df[["bore","stroke"]].astype("float")

df["normalized-losses"]=df["normalized-losses"].astype("float")

#Data Standardization

df["height"]=df["height"]/df["height"].max()

#Data binning
average_horsePower=df["horsepower"].astype(float).mean()
df.replace(np.nan,average_horsePower,inplace=True)
df["horsepower"]=df["horsepower"].astype(int, copy=True)

%matplotlib inline
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

#to get three bins
bins=np.linspace(df["horsepower"].min(),df["horsepower"].max(),4)

labels=["low","medium","high"]
df['horsepower-binned']=pd.cut(df["horsepower"],bins,labels=labels,include_lowest=True)
df.head(5)

df['horsepower-binned'].value_counts()

plt.pyplot.bar(labels,df['horsepower-binned'].value_counts())
plt.pyplot.hist(df["horsepower"], bins = 3)


#dummy variables

df.columns
dummy_variables=pd.get_dummies(df['fuel-type'])
dummy_variables.head(20)
pd.concat([dummy_variables,df],axis=1)
df.drop(["fuel-type"],axis=1,inplace=True)
