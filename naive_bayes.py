import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB
from sklearn.metrics import  confusion_matrix
from sklearn.tree import export_graphviz
import graphviz
import seaborn as sns
df.head()
df.describe()
df.info()
df.tail()
import matplotlib.pyplot as plt
x = "BloodPressure"
y = "Age"
plt.scatter(x,y)
plt.show()
import seaborn as sns
plt.figure(figsize = (20,15))
plotnumber = 1
for col in df.columns:
  if plotnumber <=8:
    ax = plt.subplot(5,5,plotnumber)
    sns.boxplot(df[col])
    plt.xlabel(col,fontsize = 15)
  plotnumber +=1
plt.tight_layout()
plt.show()
x = df.drop('Outcome', axis =1)
y = df['Outcome']
import missingno as msno
msno.bar(df)
plt.show()
x_train, x_test, y_train, y_test=train_test_split (x,y,test_size = 0.5, random_state=5)
a = StandardScaler()
x_train = a.fit_transform(x_train)
x_test = a.transform(x_test)
clf = GaussianNB()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
cm
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
