import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys
import pickle

class HousePrice:
    def __init__(self):
        try:
            df = pd.read_csv("data.csv")
            df = df.drop(['date', 'street'], axis=1)
            l=df['city'].unique()
            d={}
            for i in range(len(l)):
                d[l[i]]=i
            m=df['statezip'].unique()
            n={}
            for j in range(len(m)):
                n[m[j]]=j
            df['city']=df['city'].map(d)
            df['statezip']=df['statezip'].map(n)
            df['country']=df['country'].map({'USA':0}).astype(int)
            self.X = df.iloc[:, 1:]
            self.y = df.iloc[:, 0]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")
    def train(self):
        try:
            self.reg=LinearRegression()
            self.reg.fit(self.X_train,self.y_train)
            train_values= self.reg.predict(self.X_train)
            return train_values
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")
    def test(self):
        try:
            test_values= self.reg.predict(self.X_test)
            return test_values
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")
    def accuracy(self,y,predictions):
        try:
            n=0
            d=0
            y=np.array(y).flatten()
            predictions=predictions.flatten()
            for i in range(len(y)):
                n=n+(y[i]-predictions[i])**2
                d=d+(y[i]-predictions.mean())**2
            acc=1-(n/d)
            print(f"Accuracy:{acc}")
            loss=np.sqrt(n/len(y))
            print(f"Loss:{loss}")
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")

    def saving_the_file(self):
        try:
            with open("model.pkl", 'wb') as f:
                pickle.dump(self.reg, f)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            print(f"Error in line no : {er_line.tb_lineno} : due to : {er_msg} : reason : {er_type}")
if __name__ == "__main__":
    obj=HousePrice()
    train_values=obj.train()
    test_values=obj.test()
    obj.accuracy(obj.y_train,train_values)
    obj.accuracy(obj.y_test,test_values)
    obj.saving_the_file()