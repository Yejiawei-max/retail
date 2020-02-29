import pandas as pd            #导入各种需要的模块
import datetime as dt
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib


task1 = pd.read_csv('task1.csv', dtype={'kh': 'str', 'csrq': 'str', 'djsj': 'str', 'gzmc': 'str'})   #导入task1数据表

task1['khfz'] = pd.isna(task1['kh'])          #创建会员分组列和出生日期分组列
task1['csrqfz'] = pd.isna(task1['csrq'])
task1['dtimefz'] = pd.isna(task1['dtime'])
task4hy = task1[task1['khfz'] == False]       #筛选出会员数据

task4hy4_1 = task4hy[task4hy['dtimefz'] == False]          #取会员的消费记录并计算每个会员当天的消费金额
task4hy4_1['daytime'] = task4hy4_1['dtime'].map(lambda x : x.split(' ')[0])
task4hy4_1 = task4hy4_1[['kh', 'daytime', 'je', 'djh']].groupby(by = ['kh', 'daytime'])
task4hy4_1 = task4hy4_1.agg({'je' : np.sum, 'djh' : 'nunique'}).reset_index()            #构建会员最近的购物时间、消费金额和订单数量标签
task4hy4_1['daytime'] = pd.to_datetime(task4hy4_1['daytime'], errors= 'coerce')
task4hy4_1fin = task4hy4_1.groupby(by = ['kh'])
task4hy4_1fin = task4hy4_1fin.agg({'daytime' : max , 'je' : np.sum, 'djh' : np.sum}).reset_index()
task4hy4_1fin['daytime'] = (pd.to_datetime(dt.date.today()) - task4hy4_1fin['daytime']).dt.days

plt.boxplot(task4hy4_1fin['je'])        #观察消费标签有无异常值
plt.show()

task4hy4_1fin = task4hy4_1fin.drop(index=(task4hy4_1fin[task4hy4_1fin['je'] > 400000].index))    #剔除部分干扰项

sc = StandardScaler()
task4hy4_1fin[['daytime', 'je', 'djh']] = sc.fit_transform(task4hy4_1fin[['daytime', 'je', 'djh']])

model = KMeans(n_clusters = 4).fit(task4hy4_1fin[['daytime', 'je', 'djh']])      #构建模型

print(silhouette_score(task4hy4_1fin[['daytime', 'je', 'djh']], model.labels_))        #模型评估

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
#标签
labels = ['近期购买行为', '消费金额', '购买频率']
#每个类别中心点数据
plot_data = model.cluster_centers_
#指定颜色
color = ['b', 'g', 'r', 'c']
# 设置角度
angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
# 闭合
angles = np.concatenate((angles, [angles[0]]))
plot_data = np.concatenate((plot_data, plot_data[:,[0]]), axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, polar=True) # polar参数为True即极坐标系
for i in range(len(plot_data)):
    ax.plot(angles, plot_data[i], 'o-', color = color[i], label = '类别'+str(i), linewidth=2)

# ax.set_rgrids(np.arange(0.01, 3.5, 0.5), np.arange(-1, 2.5, 0.5), fontproperties="SimHei") # 手动配置r网格刻度
ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="SimHei")
ax.set_title('聚类类别特征分析')
plt.legend(loc = 4) # 设置图例位置
plt.show()