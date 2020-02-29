import pandas as pd            #导入pandas模块
import datetime as dt
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

task1 = pd.read_csv('taskfl_3.csv', dtype={'kh': 'str', 'csrq': 'str', 'djsj': 'str', 'gzmc': 'str'})         #导入task1数据表
task1['csrq'] = pd.to_datetime(task1['csrq'], errors= 'coerce')

# task1 = pd.read_csv('task1.csv', dtype={'kh': 'str', 'csrq': 'str', 'djsj': 'str', 'gzmc': 'str'})       #导入task1数据表
#
# task1['khfz'] = pd.isna(task1['kh'])                #提取出有出生日期的会员信息表
# task1['csrqfz'] = pd.isna(task1['csrq'])
# task2hy = task1[task1['khfz'] == False]
# task2hy2_1_3 = task2hy[task2hy['csrqfz'] == False]
#
# task2hy2_1_3['csrq'] = task2hy2_1_3['csrq'].apply(lambda x:str(x))            #对出生日期字符化，分列，改符号，取月日处理
# task2hy2_1_3['csrq'] = task2hy2_1_3['csrq'].map(lambda x:x.split(' ')[0])
# task2hy2_1_3['csrq'] = task2hy2_1_3['csrq'].apply(lambda x:x.replace('/', '-'))
# task2hy2_1_3['csrqxx'] = task2hy2_1_3['csrq'].apply(lambda x:time.strptime(x, '%Y-%m-%d'))
# task2hy2_1_3['csrqxx'] = task2hy2_1_3['csrqxx'].apply(lambda x: x[1:3])
# task2hy2_1_3['csrqY'] = task2hy2_1_3['csrqxx'].apply(lambda x: x[1])
#
# today = str(dt.date.today())                 #获取今天的月日
# Md = time.strptime(today, '%Y-%m-%d')[1:3]
#
# task2hy2_1_3['nl'] = dt.datetime.today().year - task2hy2_1_3['csrqY'] - (Md < task2hy2_1_3['csrqxx'])    #计算会员年龄

task1['nl'] = dt.datetime.today().year - task1['csrq'].dt.year - (dt.datetime.today().month < task1['csrq'].dt.month)     #计算会员年龄的简化版

task1['nld'] = pd.cut(task1['nl'], bins = [0, 30, 59, 105], labels = ['青年', '中年', '老年'])  #对数据进行分段处理并赋予相应的标签
task1_1 = task1.drop_duplicates(subset = ['kh'])               #取每个会员的一条消费记录做表task1_1

task2_nld_pie = task1_1.groupby(by = ['nld'])['nld'].count()   #计算会员年龄段

task2bar_hg = task1.groupby(by = ['nld'])['je'].sum()          #计算不同年龄段的消费金额
task2bar_id = np.arange(3)

task2_xb_pie = task1_1.groupby(by = ['xb'])['xb'].count()     #计算会员不同性别数量
task2_je_pie = task1.groupby(by = ['xb'])['je'].sum()

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.pie(task2_nld_pie, autopct='%.2f %%', labels = ['青年', '中年', '老年'],
        explode=[0, 0.1, 0], colors=["gold", "tomato", "orange"])
plt.title('会员年龄占比')
plt.show()

plt.bar(x = task2bar_id, height = task2bar_hg, width = 0.45)
plt.xticks(task2bar_id, ('青年', '中年', '老年'))
plt.xlabel('不同年龄段')
plt.ylabel('消费总金额')
plt.title('不同年龄的消费金额')
plt.show()

plt.pie(task2_xb_pie, autopct='%.2f %%', labels = ['女性', '男性'],
        explode=[0.1, 0], colors=["tomato", "orange"])
plt.title('会员性别比例')
plt.show()
plt.pie(task2_je_pie, autopct='%.2f %%', labels = ['女性', '男性'],
        explode=[0.1, 0], colors=["tomato", "orange"])
plt.title('会员不同性别消费比例')
plt.show()
