import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

def create_model(data):
    print(data)
    X=data.drop(['Dataset'],axis=1)
    y=data['Dataset']
    scaler=StandardScaler()#scale data
    X=scaler.fit_transform(X)
    #split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # train the model
    model = LogisticRegression()#Using model
    model.fit(X_train, y_train)

    #test model
    y_pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test,y_pred))

    return model,scaler

def readcsv():
    data=pd.read_csv("liver_data.csv")
    
    data.dropna(inplace=True)
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    
    return data


def main():
    data =readcsv()
    model,scaler=create_model(data)
    #using pickle to save training pip install pickle
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)#writing into file as binary
    with open ('scaler.pkl','wb')as f:
        pickle.dump(scaler,f)

if __name__=='__main__':
    main()        




