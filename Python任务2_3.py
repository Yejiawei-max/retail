import pandas as pd            #导入pandas模块
import datetime as dt
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

task1 = pd.read_csv('task1.csv', dtype={'kh': 'str', 'csrq': 'str', 'djsj': 'str', 'gzmc': 'str'})          #导入task1数据表并根据卡号列去重

task1['khfz'] = pd.isna(task1['kh'])
task1['dtimefz'] = pd.isna(task1['dtime'])
task2hy = task1[task1['khfz'] == False]                          #取会员信息表
task2hy = task2hy.drop_duplicates(subset = ['kh', 'dtime'])      #剔除同一订单的消费记录
task2hy2_3_1 = task2hy[task2hy['dtimefz'] == False]              #取会员消费记录

task2hy2_3_1['dtime1'] = task2hy2_3_1['dtime'].map(lambda x:x.split('.')[0])           #提取会员消费记录在一天当中的时间点
task2hy2_3_1['timestamp'] = task2hy2_3_1['dtime1'].apply(lambda x:time.mktime(time.strptime(x,'%Y-%m-%d %H:%M:%S')))
task2hy2_3_1['daytime'] = task2hy2_3_1['timestamp']%86400

task2hy2_3_1['dayfz'] = pd.cut(task2hy2_3_1['daytime'], bins = [0, 9360, 23040, 40320, 57600, 74880, 86400],
                               labels = ['2早上', '3中午', '4下午', '5晚上', '1凌晨', '2早上_1'])     #按照时间点分段
task2hy2_3_1['dayfz'] = task2hy2_3_1['dayfz'].map(lambda x:x.split('_')[0])

task2hy2_3_1bar = task2hy2_3_1.groupby(by = ['dayfz'])['kh'].count()
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.bar(x = np.arange(5), height = task2hy2_3_1bar, width = 0.45)
plt.xticks(np.arange(5), ('凌晨', '早上', '中午', '下午', '晚上'))
plt.xlabel('不同时间段')
plt.ylabel('人数')
plt.title('不同时间段的会员人数')
plt.show()

task2hy2_3_2 = task2hy
task2hy2_3_2['dtime'] = pd.to_datetime(task2hy2_3_2['dtime'], errors= 'coerce')
task2hy2_3_2['year'] = task2hy2_3_2['dtime'].dt.year
task2hy2_3_2['month'] = task2hy2_3_2['dtime'].dt.month
task2hy2_3_2['seasonfz'] = pd.cut(task2hy2_3_2['month'], bins = [0, 2, 5, 8, 11, 12],
                               labels = ['4冬季', '1春季', '2夏季', '3秋季', '4冬季_1',])
task2hy2_3_2['seasonfz'] = task2hy2_3_2['seasonfz'].map(lambda x:x.split('_')[0])

task2hy2_3_2bar = task2hy2_3_2.groupby(by = ['year', 'seasonfz'])['kh'].count()
plt.bar(x = np.arange(12), height = task2hy2_3_2bar, width = 0.45)
plt.xticks(np.arange(12), ('2015春季', '夏季', '冬季',
                           '2016春季', '夏季', '秋季', '冬季',
                           '2017春季', '夏季', '秋季', '冬季',
                           '2018冬季'))
plt.xlabel('不同季节')
plt.ylabel('人数')
plt.title('不同季节的会员人数')
plt.show()