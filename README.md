# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.

2. Analyse the data. 

3. Use modelselection and Countvectorizer to preditct the values. 

4. Find the accuracy and display the result. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: YOGESH RAO S D
RegisterNumber:  212222110055
*/
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()

data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### result
![image](https://github.com/amal-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/148410730/3d85ed0d-8cbe-4489-9c3a-c229e89ec80b)
### data.head()
![image](https://github.com/amal-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/148410730/a099c02f-4f26-498c-aabd-16c69e53c82f)
### data.info()
![image](https://github.com/amal-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/148410730/df76ba5b-680e-4a34-9ee1-8b4cd746b096)
### data.isnull.sum()
![image](https://github.com/amal-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/148410730/a48408d8-2f7c-468f-ae8b-d5c64a6f1643)
### y_pred
![image](https://github.com/amal-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/148410730/13d7d225-1be2-4043-9788-4b2dc14c50d2)
### accuracy
![image](https://github.com/amal-2006/Implementation-of-SVM-For-Spam-Mail-Detection/assets/148410730/eefcd515-b00f-46df-9ce3-8aa983bac501)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
