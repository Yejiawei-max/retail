import pandas as pd            #导入pandas模块
import numpy as np
import matplotlib.pyplot as plt
from pandas._libs.tslibs.nattype import NaT


c1 = pd.read_excel('cumcm2018c1.xlsx')           #导入c1数据表并根据卡号列去重
c1_1 = c1.drop_duplicates(subset = ['kh'])

print('去重后的卡号数量：', c1_1.shape, '\n''原表的卡号数量：', c1.shape)     #检查会员卡号是否有缺失值和去重情况
print(pd.isna(c1_1).sum())
print(c1_1.describe())

c2 = pd.read_csv('cumcm2018c2.csv', dtype={'kh': 'str', 'gzmc': 'str'})       #导入c2数据表


print(c2.head())          #检查c2数据表的基本情况
print(c2.shape)

print(c2['dtime'].dtype)                        #检查消费产生的时间是否有异常值
c2['dtime'] = pd.to_datetime(c2['dtime'])
print(c2['dtime'].dtype)
print(c2['dtime'].min(), c2['dtime'].max())

diff = set(c1_1['kh']) ^ set(c2['kh'])          #删除c2表里面其它分店的会员卡号
diff.remove(np.nan)
c2_1 = c2[-c2.kh.isin(diff)]

c2_1['je'] = c2_1.je.apply(lambda x: 0 if x < 0 else x)    #把金额为负数的改为0


task1 = pd.merge(c1_1, c2_1, on='kh', how='outer')  # 外连接数据表，保留表中所有信息（任务1完成）
print('任务1最后表格情况：', pd.isna(task1).sum())  # gzmc列的缺失值看情况处理
#导出数据
task1.to_csv('task1.csv')