from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
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

cc = ['六个月内网银异名他行转入月均交易笔数', '最近三个月内账户借方月均交易次数', '客户持有的全部产品数量（24种产品）', '柜面六个月月均交易金额', '三个月内柜面异名他行转入月均交易金额增加值']
X = data[cc].values
y = data['是否优质'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
model3 = RandomForestRegressor().fit(X_train, y_train)
predictions3 = model3.predict(X)
list1=[]
for i in range(len(predictions3)):
    if predictions3[i]>=0.5:
        list1.append(i)
print(list1)
import copy
import numpy as np

data2 = pd.read_csv('fintechsss.csv')
print(data2)
data3=copy.deepcopy(data2)
label_need=data2.keys()
[m,n]=data3.shape
ymin=0.002
ymax=1
for j in range(0,n):
    d_max=max(data2[label_need[j]])
    d_min=min(data2[label_need[j]])
    data3[label_need[j]]=(ymax-ymin)*(data2[label_need[j]]-d_min)/\
                         (d_max-d_min)+ymin


p=copy.deepcopy(data3)
for j in range(0,n):
    p[label_need[j]]=data3[label_need[j]]/sum(data3[label_need[j]])
E=copy.deepcopy(data3.iloc[0])
for j in range(0,n):
    E[j]=-1/np.log(m)*sum(p[label_need[j]]*np.log(p[label_need[j]]))

E = 1-E
print(E)
E.to_excel('D:/aa打工人/fintechsq.xlsx')
w=(E)/E.sum()
s=np.dot(data3,w.values)
Score=100*s/s.max()
list2=[]
for i in range(len(Score)):
    if Score[i]>=2.0240488310022986:
        list2.append(i)
print(list2)
mm = 0
for i in list1:
    if i in list2:
        mm+=1
print(mm)