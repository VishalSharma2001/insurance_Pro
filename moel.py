import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
df=pd.read_csv("data.csv")
#print(df)
x=df.drop(['charges'],axis=1)
print(x.shape)
y=df["charges"]
x.head()
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.10,random_state=0)
Reg=RandomForestRegressor( n_estimators=200,
                             min_samples_split= 10,
                             min_samples_leaf= 5,
                             max_features= 'auto',
                             max_depth= 5,
                             criterion= 'mse',
                             bootstrap= True)
Reg.fit(train_x,train_y)
#Reg=LinearRegression()
#Reg.fit(train_x,train_y)
pred_y=Reg.predict(test_x)
print(Reg.score(train_x,train_y)*100)
print(Reg.score(test_x,test_y)*100)

pickle.dump(Reg,open("Model.pkl","wb"))