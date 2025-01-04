import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv(r"C:\Users\Shaik Mahimood\Downloads\Salary_Data.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,:-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
m_slope=regressor.coef_
print(m_slope)
c_inter=regressor.intercept_
print(c_inter)
y_15=m_slope*15+c_inter
y_20=m_slope*20+c_inter
dataset.mean()
dataset['Salary'].mean
dataset.median()
dataset['Salary'].median()
dataset.mode()
dataset['Salary'].mode()
dataset.describe()
dataset.var()
dataset.std()

from scipy.stats import variation
variation(dataset.values)

#Corelation
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

#SKEWNESS
dataset.corr()
dataset.skew()
dataset['Salary'].skew()
dataset.sem()

#Z-score
import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

#Degree of freedom
a=dataset.shape[0]
b=dataset.shape[1]
degree_of_freedom=a-b
print(degree_of_freedom)

#SSR
y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

#   R-SQUER
r_square=1-SSR/SST
r_square
from sklearn.metrics import mean_squared_error
import pickle
filename='regressor.pkl'
with open(filename,'wb')as file:
    pickle.dump(regressor,file)
    print("model has been pickled and saved as regressor.pkl")
    
    import os
print(os.getcwd())
bias=regressor.score(x_train,y_train)
print(bias)
variance=regressor.score(x_test,y_test)
print(variance)