# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output. 


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RITHISH P
RegisterNumber:  212223230173
*/

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

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

y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
 ![image](https://github.com/RITHISHlearn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446645/00a76074-0d6e-4c3b-8863-36ee9e888fd9)

![image](https://github.com/RITHISHlearn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446645/3093400d-b6f0-486c-bdf4-0821eebdc908)

![image](https://github.com/RITHISHlearn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446645/c159df5d-e207-44f6-89d7-7dad12dfbd22)

![image](https://github.com/RITHISHlearn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446645/dd75a8b3-5d7a-4e8e-ba34-16cedd980495)

![image](https://github.com/RITHISHlearn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446645/9ad8efa6-ce51-4159-b51c-53b51438a4e8)

![image](https://github.com/RITHISHlearn/Implementation-of-SVM-For-Spam-Mail-Detection/assets/145446645/69df67fd-0481-49ce-aad3-bb0d230d1195)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
