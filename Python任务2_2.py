import pandas as pd            #导入pandas模块
import datetime as dt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

task1 = pd.read_csv('task1.csv', dtype={'kh': 'str', 'csrq': 'str', 'djsj': 'str', 'gzmc': 'str'})       #导入task1数据表
task1['dtime'] = pd.to_datetime(task1['dtime'], errors= 'coerce')

task1['khfz'] = pd.isna(task1['kh'])           #取会员信息表

task2_2_1 = task1.drop_duplicates(subset=['khfz', 'kh', 'djh', 'dtime'])
task2_2_1 = task2_2_1.groupby(by = ['khfz'])['djh'].count()          #nunique()
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.bar(x = np.arange(2), height = task2_2_1, width = 0.45)
plt.xticks(np.arange(2), ('会员', '非会员'))
plt.xlabel('客户身份')
plt.ylabel('订单数量')
plt.title('不同客户的订单数')
plt.show()

task2_2_2 = task1.groupby(by = ['khfz'])['je'].sum()
plt.bar(x = np.arange(2), height = task2_2_2, width = 0.45)
plt.xticks(np.arange(2), ('会员', '非会员'))
plt.xlabel('客户身份')
plt.ylabel('消费金额')
plt.title('不同客户的消费金额')
plt.show()

task1hy = task1[task1['khfz'] == False]
task1hy['dtimeyear'] = task1hy['dtime'].dt.year
task2_2_3 = task1hy.groupby(by = ['dtimeyear'])['je'].sum()
plt.bar(x = np.arange(4), height = task2_2_3, width = 0.45)
plt.xticks(np.arange(4), ('2015', '2016', '2017', '2018'))
plt.xlabel('年份')
plt.ylabel('消费金额')
plt.title('不同年份的消费金额')
plt.show()

task1hy['dtimemonth'] = task1hy['dtime'].dt.month
task2_2_4 = task1hy.groupby(by = ['dtimeyear', 'dtimemonth'])['je'].sum()
plt.bar(x = np.arange(33), height = task2_2_4, width = 0.45)
plt.xticks(np.arange(33), ('15/1', '2', '3', '4', '5',
                           '6', '7', '8', '12', '16/1',
                           '3', '4', '5', '6', '7',
                           '8', '9', '10', '11', '12',
                           '17/1', '2', '3', '4', '5',
                           '6', '7', '8', '9', '10',
                           '11''12''18/1'))
plt.xlabel('月份')
plt.ylabel('消费金额')
plt.title('不同月份的消费金额')
plt.show()