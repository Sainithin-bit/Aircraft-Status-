import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("train.csv")
xtrain = data.iloc[:, [1, 2, 3, 5, 6, 7, 8, 10]]
ytrain = data.iloc[:, [0]]
clf = LogisticRegression()
clf.fit(xtrain, ytrain)
data1 = pd.read_csv("test.csv")
xtest = data1.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9]]
ytest = clf.predict(xtest)
ytest=pd.DataFrame(ytest,columns=["Result"]).to_csv("test.csv",index=None)



