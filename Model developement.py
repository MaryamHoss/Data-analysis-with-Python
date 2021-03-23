import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.linear_model import LinearRegression
import seaborn as sns
%matplotlib inline 

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

#Linear Regression

lm=LinearRegression()

X=df[['highway-mpg']] #with 2[] it gives a df, with one bracket it gives a series

Y=df[['price']] 
lm.fit(X,Y)
yhat=lm.predict(X)
lm.coef_
lm.intercept_
#giving back:
    #price = 38423.31 - 821.73 x highway-mpg
    
    
#Multiple linear regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm.fit(Z,df[['price']])
Yhat=lm.predict(Z)

lm.coef_
lm.intercept_
#giving back:
    #yhat=-15806.62462633+53.49574423*horsepower+4.70770099*curb-weight+81.53026382*engine-size+36.05748882*'highway-mpg

#Model Evaluation using Visualization

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

#stats.pearsonr(df[["peak-rpm"]],df[["price"]])
import scipy
from scipy import stats
df[["peak-rpm","highway-mpg","price"]].corr()

#lets look at the residual plots for the price and highway-mpg
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x="highway-mpg", y="price", data=df)

#this plots shows that the residuals are not randomely scattered, showing that the linearmodel is not the best model
#to describe the relationship between price and highway-mpg

#lets take a look at the multiple regression model:
Yhat=lm.predict(Z)

#here we cant use residual plots or regression lines, because we have multiple variables. We can however look at the distribution plots

ax1=sns.distplot(df[["price"]],hist=False,label="Actual Value")
sns.distplot(yhat,hist=False,label="predicted Value")



#polynomial regression
x=df['horsepower'] #for polyfit, we dont need a dataframe, we need a 1D vector, that is why we dont do [[]]
y=df['price']
f=np.polyfit(x,y,11)
p=np.poly1d(f)
print(p)

#now lets define a function that plots this polynomial line and the data
def poly_plot(model, independent_variable,dependent_variable,x_name ):
    
    x_new=np.linspace(np.int(np.min(independent_variable)),np.int(np.max(independent_variable)),50)
    y_new=model(x_new)
    plt.plot(independent_variable,dependent_variable ,'r+',x_new, y_new,'b-')
    plt.title('Polynomial Fit for Price')
    plt.xlabel(x_name)
    plt.ylabel('Price of cars')
    
    plt.show()
    plt.close()
    
    
    
    
poly_plot(p,x,y,'Horse power')



#now lets see how we can have a polynomial fit, but for multiple regression

from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures()
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Z_poly=pr.fit_transform(Z)



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Input=[('Scale',StandardScaler()),('poly',PolynomialFeatures()),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
x=df[['horsepower']]
pipe.fit(Z,y)
out_pipe=pipe.predict(Z)
out_pipe[0:4]



#Measures for In-Sample Evaluation
#R-squared
lm=LinearRegression()
lm.fit(x,y)
lm.score(x,y) #=0.65 65percent of the data variatioon can be explained by the horsepower3

print('The R-squared is {}'.format(lm.score(x,y)))

y_hat=lm.predict(x)

from sklearn.metrics import mean_squared_error

print('mse is {}'.format(mean_squared_error(y,y_hat)))


#lets  do the same for a multiple linear regression model
lm=LinearRegression()
lm.fit(Z, y)
lm.score(Z,y)
y_hat=lm.predict(Z)
mean_squared_error(y,y_hat)


#for a polynomial, we need to use sickit learn packages
from sklearn.metrics import r2_score
r2_score(y,p(x))
mean_squared_error(y, p(x))
