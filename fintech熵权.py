import copy
import pandas as pd
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
Score.sort()
print(Score[-6200])