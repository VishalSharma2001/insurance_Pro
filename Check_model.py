import pickle
from sklearn.ensemble import RandomForestRegressor

model=pickle.load(open('Model.pkl','rb'))
print(model.predict([[19,0,27.9,0,1,2]]))