import threading
import pandas as pd
import time
import math
import numpy as np
import multiprocessing


df1 = pd.read_csv('datafintech.csv',encoding = 'gb2312')
Y_flag = '是否优质'
x = 0
for i in df1['是否优质'].values:
    if i == 1:
        x+=1
print(x)
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
var_list = df1.columns.values
for i in var_list:
    df1[i] = df1[i].apply(tihuan)
print(df1)

def compute_woe_iv(df,Y_flag,start,end):
    var_list = df.columns.values
    totalG_B = df.groupby([Y_flag])[Y_flag].count()  # 计算正负样本多少个
    G = totalG_B[1]
    B = totalG_B[0]
    z={}
    for k in range(start, end):
        print(k)
        var1 = df.groupby([var_list[k]])[Y_flag].count()  # 计算col每个分组中的组的个数
        var_class = var1.shape[0]
        woe = np.zeros((var_class, 8))
        woe_pre = pd.DataFrame(data={'x1': [], 'ifgood': [], 'values': []})
        total = df.groupby([var_list[k], Y_flag])[Y_flag].count()  # 计算该变量下每个分组响应个数
        total1 = pd.DataFrame({'total': total})
        mu = []
        for u, group in df.groupby([var_list[k], Y_flag])[Y_flag]:
            mu.append(list(u))
        for lab1 in total.index.levels[0]:
            for lab2 in total.index.levels[1]:
                if [lab1, lab2] not in mu:
                    temporary = pd.DataFrame(data={'x1': [lab1], 'ifgood': [lab2], 'values': [0]})
                else:
                    temporary = pd.DataFrame(
                        data={'x1': [lab1], 'ifgood': [lab2], 'values': [total1.xs((lab1, lab2)).values[0]]})
                woe_pre = pd.concat([woe_pre, temporary])
            # print(woe_pre)
        woe_pre.set_index(['x1', 'ifgood'], inplace=True)

        # 计算 WOE
        for i in range(0, var_class):  # var_class
            woe[i, 1] = woe_pre.values[2 * i + 1] / G  # pyi
            woe[i, 2] = woe_pre.values[2 * i] / B  # pni
            abb = lambda i: (math.log(woe[i, 1] / woe[i, 2])) if woe[i, 1] != 0 else 0  # 防止 ln 函数值域报错
            woe[i, 3] = abb(i)
            woe[np.isinf(woe)] = 0  # 将无穷大替换为0，参与计算 woe 计算

            woe[i, 4] = (woe[i, 1] - woe[i, 2]) * woe[i, 3]  # iv_part
        x = woe[:,4].sum()
        z[var_list[k]]=x
    cf = pd.DataFrame(data=z.items(),columns=['变量', 'iv'])
    return cf

def test1():
    cf = compute_woe_iv(df1,Y_flag,601,670)
    cf.to_excel('fintechiv5.xlsx')

def test2():
    cf = compute_woe_iv(df1,Y_flag,671,740)
    cf.to_excel('fintechiv6.xlsx')

def test3():
    cf = compute_woe_iv(df1,Y_flag,741,820)
    cf.to_excel('fintechiv7.xlsx')


def test4():
    cf = compute_woe_iv(df1,Y_flag,821,900)
    cf.to_excel('fintechiv8.xlsx')

def test5():
    cf = compute_woe_iv(df1,Y_flag,901,984)
    cf.to_excel('fintechiv9.xlsx')

def main():
    """进程"""
    p1 = multiprocessing.Process(target=test1)
    p2 = multiprocessing.Process(target=test2)
    p3 = multiprocessing.Process(target=test3)
    p4 = multiprocessing.Process(target=test4)
    p5 = multiprocessing.Process(target=test5)
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()


if __name__ == '__main__':
    main()
