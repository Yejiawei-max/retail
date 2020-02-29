import pandas as pd            #导入各种需要的模块
import datetime as dt
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas._libs.tslibs.nattype import NaT
import pkuseg
from collections import Counter
import random

task1 = pd.read_csv('task1.csv', dtype={'kh': 'str', 'csrq': 'str', 'djsj': 'str', 'gzmc': 'str'})   #导入task1数据表

task1['khfz'] = pd.isna(task1['kh'])          #创建会员分组列和出生日期分组列
task1['csrqfz'] = pd.isna(task1['csrq'])
task1['dtimefz'] = pd.isna(task1['dtime'])
task3hy = task1[task1['khfz'] == False]       #筛选出会员数据

task3hy['csrq'] = pd.to_datetime(task3hy['csrq'], errors= 'coerce')         #转化出生日期列的格式并计算会员年龄
task3hy['nl'] = dt.datetime.today().year - task3hy['csrq'].dt.year - (dt.datetime.today().month < task3hy['csrq'].dt.month)

task3hy['xb'] = task3hy['xb'].astype(np.str)                                #转会员性别为字符
task3hy['xb'] = task3hy['xb'].apply(lambda x:x.replace('0.0', '女性'))
task3hy['xb'] = task3hy['xb'].apply(lambda x:x.replace('1.0', '男性'))

task3hy['djsj'] = pd.to_datetime(task3hy['djsj'], errors= 'coerce')
task3hy['rhsc'] = (dt.datetime.today() - task3hy['djsj']).dt.days           #尝试用新方法计算会员入会时长

task3hy3_1 = task3hy.groupby(by=['kh'])['jf'].sum().reset_index()           #计算积分

task3hy = task3hy[task3hy['dtimefz'] == False]

task3hy3_2 = task3hy          #取会员的消费记录并计算每个会员当天的消费金额
task3hy3_2['daytime'] = task3hy3_2['dtime'].map(lambda x : x.split(' ')[0])
task3hy3_2_1 = task3hy3_2[['kh', 'daytime', 'je']].groupby(by = ['kh', 'daytime'])
task3hy3_2fin = task3hy3_2_1.agg({'je' : np.sum}).reset_index()           #构建会员消费标签
task3hy3_2fin['xf'] = pd.cut(task3hy3_2fin['je'], bins = [-100000000, 1000, 3000, 100000000], labels = ['低消费', '中消费', '高消费'])
task3hy3_2fin = task3hy3_2fin.groupby(['kh'])['xf'].apply(lambda x : Counter(x).most_common(1)).to_frame().reset_index()

task3hy3_3 = task3hy      #取会员的消费记录

task3hy3_3_1 = task3hy3_3            #构建会员消费时间段标签
task3hy3_3_1['dtime1'] = task3hy3_3_1['dtime'].map(lambda x:x.split('.')[0])
task3hy3_3_1['timestamp'] = task3hy3_3_1['dtime1'].apply(lambda x:time.mktime(time.strptime(x,'%Y-%m-%d %H:%M:%S')))
task3hy3_3_1['daytime'] = task3hy3_3_1['timestamp']%86400

task3hy3_3_1['dayfz'] = pd.cut(task3hy3_3_1['daytime'], bins = [0, 9360, 23040, 40320, 57600, 74880, 86400],
                               labels = ['早上', '中午', '下午', '晚上', '凌晨', '早上_1'])
task3hy3_3_1['dayfz'] = task3hy3_3_1['dayfz'].map(lambda x:x.split('_')[0])
task3hy3_3_1 = task3hy3_3_1.groupby(['kh'])['dayfz'].apply(lambda x : Counter(x).most_common(2)).to_frame().reset_index()

task3hy3_3_2 = task3hy3_3            #构建会员消费季节标签
task3hy3_3_2['dtime'] = pd.to_datetime(task3hy3_3_2['dtime'], errors= 'coerce')
task3hy3_3_2['year'] = task3hy3_3_2['dtime'].dt.year
task3hy3_3_2['month'] = task3hy3_3_2['dtime'].dt.month
task3hy3_3_2['seasonfz'] = pd.cut(task3hy3_3_2['month'], bins = [0, 2, 5, 8, 11, 12],
                               labels = ['冬季', '春季', '夏季', '秋季', '冬季_1',])
task3hy3_3_2['seasonfz'] = task3hy3_3_2['seasonfz'].map(lambda x:x.split('_')[0])
task3hy3_3_2 = task3hy3_3_2.groupby(['kh'])['seasonfz'].apply(lambda x : Counter(x).most_common(1)).to_frame().reset_index()

task3hy3_3_3 = task3hy3_3            #构建会员偏好标签
stop_list = ['/', '\\', '(', ')', '（', '）', '.', 'ml']
seg = pkuseg.pkuseg()     # 以默认配置加载模型
task3hy3_3_3['cut'] = task3hy3_3_3['spmc'].apply(lambda x: [i for i in seg.cut(x)])     # 对商品名称进行分词
task3hy3_3_3['cut'] = task3hy3_3_3['cut'].apply(lambda x: [i for i in x if i not in stop_list])

task3hy3_3_3fin = task3hy3_3_3[['kh', 'cut']]


def peralwords(x):
    words = []
    for content in x:
        words.extend(content)
    return words


task3hy3_3_3fin = task3hy3_3_3fin.groupby(['kh'])['cut'].\
    apply(lambda x : Counter(peralwords(x)).most_common(3)).to_frame().reset_index()     #统计会员商品词频

index = task3hy[task3hy.index == random.randint(1, task3hy.shape[0])]['kh']
kh = index.values[0]
xb = task3hy[task3hy['kh'] == kh]['xb'].values[0]
nl = task3hy[task3hy['kh'] == kh]['nl'].values[0]
rhsc = task3hy[task3hy['kh'] == kh]['rhsc'].values[0]
jf = task3hy3_1[task3hy3_1['kh'] == kh]['jf'].values[0]
xf = task3hy3_2fin[task3hy3_2fin['kh'] == kh]['xf'].values[0]
sjd = task3hy3_3_1[task3hy3_3_1['kh'] == kh]['dayfz'].values[0]
jjd = task3hy3_3_2[task3hy3_3_2['kh'] == kh]['seasonfz'].values[0]
ph = task3hy3_3_3fin[task3hy3_3_3fin['kh'] == kh]['cut'].values[0]

print('卡号：', kh, '\n''性别：', xb, '\n''年龄：', nl, '岁', '\n''入会时长：', rhsc, '天','\n''积分：', jf, '点',
      '\n''消费能力及频率：', xf, '\n''最喜欢消费的时间段及频率：', sjd, '\n''最喜欢消费的季节及频率：', jjd,
      '\n''偏好及偏好出现频率：', ph)