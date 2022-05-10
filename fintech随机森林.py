from sklearn.ensemble import RandomForestRegressor
import sklearn
import pandas as pd
from sklearn.metrics import roc_auc_score
import xlrd
import random
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('datafintech.csv',encoding = 'gb2312')
def tihuan(cc):
    if cc == 'N':
        return 0
    if cc == 'Y':
        return 1
    if cc == '#':
        return 5
    if cc == 'X':
        return 0
    else:
        return cc
var_list = data.columns.values
for i in var_list:
    data[i] = data[i].apply(tihuan)

text = data.loc[0:15000,:]
train = data.loc[15000:,:]

file = 'fintechiv1.xlsx'
dataa = xlrd.open_workbook(filename=file)
table = dataa.sheets()[0]
a = table.col_values(0)[1:100]

from sklearn.model_selection import train_test_split
zz = []
m = 0
vv = []
mm=99
x = random.uniform(0,mm)
y = random.uniform(0, mm)
while x == y:
    y = random.uniform(0, mm)
z = random.uniform(0, mm)
while x==z or y ==z:
    z = random.uniform(0, mm)
q = random.uniform(0, mm)
while x==q or y ==q or z==q:
    q = random.uniform(0, mm)
w = random.uniform(0, mm)
while x==w or y ==w or z == w or q == w:
    w = random.uniform(0, mm)

cc=[a[int(x)],a[int(y)],a[int(z)],a[int(q)],a[int(w)]]

X = data[cc].values
y = data['是否优质'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
model3 = RandomForestRegressor().fit(X_train, y_train)
predictions3 = model3.predict(X_test)
nn = roc_auc_score(y_test, predictions3)

list1 = []
list2 = []
for i in range(100):
    print(i)
    w = random.uniform(0,mm)
    print(a[int(w)])
    g = random.uniform(0,4)
    bb = cc
    bb.pop(int(g))
    bb.append(a[int(w)])
    try:
        model3 = RandomForestRegressor().fit(X_train, y_train)
        predictions3 = model3.predict(X_test)
        nn = roc_auc_score(y_test, predictions3)
        zz.append(nn)
        if nn>=max(zz):
            print(bb)
            list1=bb
            print('最优auc为' + str(nn))
        else:
            print('最优auc仍为' + str(max(zz)))
    except:
        print('最优auc依然为' + str(max(zz)))
    list2.append(max(zz))
print(list1)
plt.plot(list2)
plt.ylabel('auc')
plt.show()
X = data[list1].values
y = data['是否优质'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
model = RandomForestRegressor().fit(X_train, y_train)
predictions = model.predict(X_test)
plt.scatter(y_test, predictions,c='b')
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
z = np.polyfit(y_test, predictions, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test))
plt.show()