# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from time import time
import re
import math

from scipy.stats import chi2_contingency #卡方检验
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score,roc_curve,roc_auc_score


from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.ensemble import BaggingClassifier #Bagging
from sklearn.ensemble import RandomForestClassifier #随机森林
import xgboost
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline 

sns.set(font_scale = 0.5)
sns.set_style('whitegrid', {'font.sans-serif': ['simhei','FangSong']})
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False



'''加载数据'''
ac01 = pd.read_csv(r'F:\数据\企业保险\ac01.csv')
ac02 = pd.read_csv(r'F:\数据\企业保险\ac02.csv')
ab01 = pd.read_csv(r'F:\数据\企业保险\ab01.csv')
ab02 = pd.read_csv(r'F:\数据\企业保险\ab02.csv')
ab07 = pd.read_csv(r'E:\数据\企业保险\ab07.csv')
ic10 = pd.read_csv(r'E:\数据\企业保险\ic10.csv')
ac61 = pd.read_csv(r'E:\数据\企业保险\ac61.csv')
ic66 = pd.read_csv(r'E:\数据\企业保险\ic66.csv')

'''ab01:单位基本信息'''
Counter(ab01['aab019']) #单位类型
Counter(ab01['aab020']) #经济类型
Counter(ab01['aab022']) #所属行业
Counter(ab01['aaa149']) #行业风险类别
Counter(ab01['aab301']) #行政区划
Counter(ab01['aab065']) #特殊单位类别

#单位有效情况
aaa120 = ab01['aaa120']
#列联表
crosstab = pd.crosstab(ab01['aab019'],ab01['aaa120'] ) #列联表

#单位类型
vc1 = ab01['aab019'].value_counts()
index = ['企业','个体工商户','其他','自定义机构','全额拨款事业单位','自收自支事业单位',
                            '机关','民办非企业单位','差额拨款事业单位','社会团体']
xlabel = ['0','25000','50000','75000','100000','125000',
          '150000','175000','200000']
b = sns.barplot(y=index,x=vc1[:10],palette="Blues_d")
b.set_xticklabels(xlabel,size=12)
b.set_yticklabels(index,size=14) #设置标签字体
plt.show()

#经济类型
vc2 = ab01['aab020'].value_counts()
index = ['其他有限责任公司','个体经营','国有全资','私营有限责任公司',
         '其他','私有独资','集体全资','其他私有','股份有限公司','其他内资']
xlabel = ['0','20000','40000','60000','80000','100000']
b = sns.barplot(y=index,x=np.array(vc2)[:10],palette="mako")
b.set_xticklabels(xlabel,size=14)
b.set_yticklabels(index,size=14) #设置标签字体
plt.show()

#所属行业
aab022 = ab01['aab022'].fillna('Unknown') #缺失值填充为unknown
map022 = aab022.map(lambda c:c[0]) #提取行业代码首字母（大类）
map022 = map022.replace({'0':'U','1':'U'}) #批量替换：首字母为数字的填充为unknown
vc3 = map022.value_counts()[1:11]
index = ['批发和零售业','建筑业','居民服务、修理和其他服务业','制造业'
         ,'租赁和商务服务业','农、林、牧、渔业','交通运输、仓储和邮政业',
         '科学研究和技术服务业','公共管理和社会组织','水利、环境和公共设施管理业']
xlabel = ['0','5000','10000','15000','20000'
          ,'25000','30000','35000','40000']
b = sns.barplot(y=index,x=np.array(vc3)[:10],palette="flare")
b.set_xticklabels(xlabel,size=14)
b.set_yticklabels(index,size=14) #设置标签字体
plt.show()

#行业风险类别
aaa149 = ab01['aaa149'].fillna(0)
vc4 = aaa149.value_counts(sort=False).sort_index() #按index排序
index = ['未知','一类','二类','三类','四类'
         ,'五类','六类','七类',"八类"]
b = sns.barplot(y=index,x=vc4,palette="magma")
b.set_xticklabels(xlabel,size=14)
b.set_yticklabels(index,size=14) #设置标签字体
plt.show()

#行政区划
aab301 = ab01['aab301']
for i in range(len(aab301)):
    if 140101 <= aab301[i] < 140200: #太原
        aab301[i] = 1
    if 140200 <= aab301[i] < 140300: #大同
        aab301[i] = 2
    if 140300 <= aab301[i] < 140400: #阳泉
        aab301[i] = 3
    if 140400 <= aab301[i] < 140500: #长治
        aab301[i] = 4
    if 140500 < aab301[i] <= 140600: #晋城
        aab301[i] = 5
    if 140600 < aab301[i] <= 140700: #朔州
        aab301[i] = 6
    if 140700 < aab301[i] <= 140800: #晋中
        aab301[i] = 7
    if 140800 < aab301[i] <= 140900: #运城
        aab301[i] = 8
    if 140900 < aab301[i] <= 141000: #沂州
        aab301[i] = 9
    if 141000 < aab301[i] <= 141100: #临汾
        aab301[i] = 10
    if 141100 < aab301[i] <= 141200: #吕梁
        aab301[i] = 11
vc5 = aab301.value_counts()
index = ['太原','晋中','大同','长治','运城','晋城','吕梁','临汾',
         '沂州','阳泉','朔州']
b = sns.barplot(y=index,x=vc5[:11],
                palette="viridis")
b.set_xticklabels(xlabel,size=14)
b.set_yticklabels(index,size=13) #设置标签字体   
plt.show()

#特殊单位类别
ab01['aab065'].value_counts()

#单位成立日期


'''ab02:单位参保情况'''
#险种类型 饼状图
aae140 = ab02['aae140'].value_counts()
aae140.plot(kind='pie',
            autopct="(%1.1f%%)" , #设置饼块内标签
            labels=['城镇职工基本养老保险','失业保险','工伤保险'],fontsize=12) 
plt.ylabel([])
plt.title("险种类型",fontsize=16)

#单位参保状态
aab316 = ab02['aab316']
vc2_1 = aab316.value_counts()
aab316 = ab02[ab02['aae140'] ==110]['aab316'] #养老险参保状态
vc2_2 = aab316.value_counts()

#单位缴费状态
aab051 = ab02[ab02['aae140'] ==110]['aab051'] #养老险缴费状态
vc2_3 = aab051.value_counts()

#单位参保日期分布
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 #标题与图间距
plt.figure(figsize=(10,10), dpi=800) 
plt.hist(ab02[ab02['aab316'] ==1]['aab050']/10000,
         40,alpha=0.9,color = 'r',label='正常参保',range=(1980,2022))
plt.hist(ab02[ab02['aab316'] ==4]['aab050']/10000,
         40,alpha=0.7,color = 'k',label='中断参保',range=(1980,2022))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=16)
plt.xlabel('参保日期',fontsize=20)
plt.ylabel('单位数',fontsize=20)
plt.title('单位参保日期分布',fontsize=24)

#征缴方式
aab033 = ab02[ab02['aae140'] ==110]['aab033']



'''ab07：单位征缴明细'''
#应缴类型：aaa115 ab07['aaa115'].value_counts() 数据字典对不上
#参保身份：aac066 101    342646
#缴费资金来源：aae737 10    342646

#应缴人数
aab119 = ab07['aab119']
sum(aab119) #743106

#缴费金额
aae020 = ab07['aae020']  #单位应缴金额
print(sum(aae020)) #单位应缴总金额:235155423.21026865 
aae080 = ab07['aae080'] #单位实缴金额
print(sum(aae080)) #234259713.39026633
aae022 = ab07['aae022'] #个人应缴金额
print(sum(aae022)) #102432803.32000375

aae082 = ab07['aae082'] #个人实缴金额 有缺失值，填充
print(sum(aae082))
aae026 = ab07['aae026'] #其他来源应缴
print(sum(aae026))
aae086 = ab07['aae086'] #其他来源实缴
print(sum(aae086))

#不同单位类型对个人的应缴金额
from matplotlib import rcParams
rcParams['axes.titlepad'] = 60 #标题与图间距
plt.figure(figsize=(40,20), dpi=700) 
index = ['企业','机关','全额拨款事业单位','差额拨款事业单位',
         '自收自支事业单位','社会团体','个体工商户','律师事务所',
         '民办非企业单位','其他']
ylabel = ['0','500','1000','1500','2000','2500']
sns.set_style('whitegrid', {'font.sans-serif': ['simhei','FangSong']})
b = sns.violinplot(x='aab019',y=aae020/aab119,palette="Set2",width=3,
               data=ab07,scale="area")
sns.despine(left=True)
b.set_xticklabels(index,size=60,rotation = 60) 
b.set_yticklabels(ylabel,size=60,rotation = 60)
plt.xlabel('单位类型',fontsize=60)
plt.ylabel('单位应缴金额（每人/月/元）',fontsize=60)
plt.title('不同单位类型对个人的应缴金额',fontsize=80)
plt.show()


ind_020 = np.array(aae020/aab119) #每个人 单位应缴
ind_022 = np.array(aae022/aab119) #个人
ind_026 = np.array(aae026/aab119) #其他来源

#三种来源应缴金额比例对比
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 #标题与图间距
plt.title('三种来源应缴金额对比')
#柱状图堆叠
plt.bar(range(0,len(aae020)),ind_020,color='green', label='单位应缴')
plt.bar(range(0,len(aae020)),ind_022,bottom=ind_020,
        color='blue', label='个人应缴')
plt.bar(range(0,len(aae020)),ind_026,bottom=ind_020+ind_022,
        color='blue', label='其他来源应缴')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='upper right',fontsize=16) # 显示图例
plt.xlabel('samples')
plt.ylabel('应缴金额')


#应缴金额和实缴金额的对比
from matplotlib import rcParams
rcParams['axes.titlepad'] = 60 #标题与图间距
plt.figure(figsize=(40,20), dpi=700) 
index = ['单位应缴金额','单位实缴金额']



'''ac01：城镇职工基本情况'''
print(len(ac01[ac01['aac004']==1]),len(ac01[ac01['aac004']==2])) #男女

'人员结构直方图，2个子图'
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 #标题与图间距
plt.title('城镇职工年龄结构',fontsize=20)
#男性
plt.figure(figsize=(10,10), dpi=800) 
ax1 = plt.subplot(211) #子图
plt.hist(ac01[ac01['aac004'] == 1]['aac006']/10000,
         100,alpha=0.9,color = 'r',label='男性',range=(1910,2010))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=16)
plt.xlabel('出生日期',fontsize=16)
plt.ylabel('人数',fontsize=16)
#调整间隔
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
    wspace=None, hspace=0.3)
#女性
ax2 = plt.subplot(212)
plt.hist(ac01[ac01['aac004'] == 2]['aac006']/10000,
         100,alpha=0.9,label='女性',
         range=(1910,2009),histtype='barstacked')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=16)
plt.xlabel('出生日期',fontsize=16)
plt.ylabel('人数',fontsize=16)
plt.show()

#民族
aac005 = ac01['aac005']
vc4_1 = aac005.value_counts()[:10]

#退休
aac084 = ac01['aac084']
#生存状态
aac060 = ac01['aac060']
#列联表
crosstab = pd.crosstab(aac084, aac060) #列联表
chi2_contingency(crosstab, correction=False, lambda_=None)



'''ac02：人员参保关系'''



'''城镇职工：单位'''
ab0102 = pd.read_csv(r'F:\数据\企业保险\ab0102.csv')
ab0102.info()
aaa120 = ab0102['有效情况']
aab019 = ab0102['单位类型']
aab020 = ab0102['经济类型']
aab022 = ab0102['所属行业']
aaa149 = ab0102['行业风险类别']
aab301 = ab0102['行政区划']
aab065 = ab0102['特殊单位类别']
aae140 = ab0102['险种类型']
aab136 = ab0102['参保状态']
aab051 = ab0102['缴费状态']

aaa120.value_counts() #有效情况：1    1938848  0      61152
aab019.value_counts() #单位类型：10 企业 1100736 81 个体工商户（有雇工的） 532352 
                      # 99 其他 244608 56 差额拨款事业单位 61152 62 个体工商户（无雇工的） 61152
aab020.value_counts() #经济类型：175 个体经营 776960 159 其他有限责任(公司) 672672 
                      # 900 其他 428064 179 其他私有 122304
#所属行业
aab022 = aab022.fillna('Unknown') 
aab022 = aab022.replace({852:'U',1580:'U',962:'U',1375:'U',
                         1889:'U'}) #批量替换：首字母为数字的填充为unknown
map022 = aab022.map(lambda c:c[0]) #提取行业代码首字母（大类）
map022.value_counts() #行业大类
#行业风险类别
aaa149.value_counts() #1.0    1388480 2.0     428064 5.0     122304 6.0      61152
aaa149.isnull().sum()
aaa149 = aaa149.fillna(1.0) #行业风险类别填充
#特殊单位类别
aab065.value_counts() #9.0    61152
#险种类型
aae140.value_counts() #110    1621124 210     194803 410     184073
#参保状态
aab136.value_counts() #1    1988243 4      11757
#缴费状态
aab051.value_counts() #1    1973088 2      15023 3      11889



'''哑变量处理'''
aab019df = pd.get_dummies(aab019,prefix = '单位类型') #prefix：前缀
aab019df.head()

aab020df = pd.get_dummies(aab020,prefix = '经济类型') #prefix：前缀
aab020df.tail()

map022df = pd.get_dummies(map022,prefix = '行业大类') 

aab301df = pd.get_dummies(aab301,prefix = '行政区划') 

aae140df = pd.get_dummies(aae140,prefix = '险种类型') 

outfile = pd.concat([aaa120,aab019df,aab020df,map022df,
                     aaa149,aab301df,aae140df,aab051],
                    axis = 1) #组合成参保分析dataframe

'''相关分析（哑变量前）'''
outfile = pd.concat([aaa120,aab019,aab020,map022,
                     aaa149,aab301,aae140,aab051,aab136],
                    axis = 1)
corrDf = outfile.corr()
sns.set(font_scale = 1)
sns.set_style('whitegrid', {'font.sans-serif': ['simhei','FangSong']})
sns.heatmap(abs(outfile.corr()),cmap='YlGnBu') #sns.heatmap：相关矩阵热力图
corrDf['有效情况'].sort_values(ascending=False) 

'''相关分析（哑变量后）'''
corrDf = outfile.corr()
#绘图
plt.figure(figsize=(60, 30)) 
sns.set(font_scale = 1.8)
sns.set_style('whitegrid', {'font.sans-serif': ['simhei','FangSong']})
sns.heatmap(outfile.corr(),cmap='coolwarm') #sns.heatmap：相关矩阵热力图
corrDf['参保状态'].sort_values(ascending=False)**0.35

'''区域性相关分析'''
#行政区划与行业大类
outfile2 = pd.concat([aab301df,
                     map022df],
                    axis = 1)
#行政区划与单位类型
outfile2 = pd.concat([aab301df,
                     aab019df],
                    axis = 1)

corrDf = outfile2.corr()
plt.figure(figsize=(20, 10)) 
sns.set(font_scale = 1.2)
sns.set_style('whitegrid', {'font.sans-serif': ['simhei','FangSong']})
sns.heatmap(outfile2.corr(),cmap='coolwarm') #sns.heatmap：相关矩阵热力图





'''模型训练'''
Source_X = outfile[[
    '行政区划_3','单位类型_81','行政区划_1',
                    '经济类型_175','行业大类_O','单位类型_99',
                    '行政区划_4','经济类型_900','行业大类_G',
                    '行政区划_5',
                    '行业风险类别','单位类型_10'
                    ]]
Source_Y = outfile['有效情况']

train_x, test_x, train_y, test_y = train_test_split(Source_X,Source_Y,
                                                    train_size=0.75)
print(train_x.shape,test_x.shape)

model = []
'逻辑回归'
t0 = time()
logreg = LogisticRegression(solver='liblinear') #有报错，改了求解器
logreg.fit(train_x, train_y)
model.append(logreg)

#网格搜索
param_grid = {'C': [1e-2, 1e-1,1e0,1e1, 1e2]}
clf = GridSearchCV(logreg, param_grid)
clf = clf.fit(train_x, train_y)
best_clf = clf.best_estimator_
model.append(best_clf)
t1 = time()
print(t1-t0)

'Bagging,随机森林'
t0 = time()
#训练集有放回采样，在多个训练子集上用相同的算法
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(),n_estimators = 500,
    max_samples=100,bootstrap=True,n_jobs=-1,oob_score=True)
#n_estimators=500:500个相同的决策器 默认为10
#max_samples=100，表示在数据集上有放回采样 100 个训练实例。
#n_jobs=-1:使用所有空闲核
#oob_score=True，表示包外评估，设定37%左右的实例是未被采样的，用这些实例来对模型进行检验
bag_clf.fit(train_x, train_y)
model.append(bag_clf)
t1 = time()
print(t1-t0)

#随机森林
t0 = time()
rnd_clf = RandomForestClassifier(n_estimators=500,max_leaf_nodes=16, 
                                 max_depth=15,max_features='sqrt',
                                 n_jobs=-1) 
#500棵树，深度最多为15，每棵树最多16个叶结点（预剪枝）
#max_features:选取的特征子集中的特征个数，可取sqrt,auto,log2
rnd_clf.fit(train_x, train_y)
model.append(rnd_clf)
t1 = time()
print(t1-t0)

importance = rnd_clf.feature_importances_
importance = np.sort(importance)[::-1]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.tick_params(labelsize=15)
#绘制柱状图
label = [
         '行政区划_晋中','行业风险类别','行业大类_建筑类',
         '经济类型_其他有限责任公司',
         '单位类型_企业','行政区划_晋城','行政区划_阳泉',
                         '经济类型_国有全资','行政区划_太原','经济类型_个体私营',
                         '行业大类_建筑类','单位类型_有雇工的个体工商户']
ax.barh(label,importance)
plt.grid(b=None)
plt.xticks(rotation=270)
plt.show()

'XGBoost'
t0 = time()
xgb = XGBClassifier(max_depth=2, #max_depth: 树的深度，默认值是6，值过大容易过拟合
                   learning_rate=0.05, 
                   silent=True,
                   reg_lambda = 1.1, #L2正则化
                   objective='binary:logistic')

#Grid Search+CV 将交叉验证和网格搜索封装在一起
param_test = {'n_estimators': range(10, 100, 1)} #需要优化的参数：基学习器的个数（1-100，函数默认值是100）
xgb_clf = GridSearchCV(estimator = xgb, 
                       param_grid = param_test, 
                       verbose=True, #输出训练过程
                       scoring= 'roc_auc',           
                       cv=5,                   
                       n_jobs=-1,
                       )# 5折交叉验证
#报错：feature_names may not contain [, ] or <
#Debug:import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
train_x.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train_x.columns.values]

xgb_clf.fit(train_x, train_y,eval_metric=["error", "logloss"])
t1 = time()
print(t1-t0) #7秒
model.append(xgb_clf)



'模型评估'
len(model) #一共几个模型
predtest = []
predtest_y1 = best_clf.predict(test_x)
predtest.append(predtest_y1)
predtest_y2 = logreg.predict(test_x)
predtest.append(predtest_y2)
predtest_y3 = bag_clf.predict(test_x)
predtest.append(predtest_y3)
predtest_y4 = rnd_clf.predict(test_x)
predtest.append(predtest_y4)
predtest_y5 = xgb_clf.predict(test_x)
predtest.append(predtest_y5)

'ACC F1 混淆矩阵 AUC K折ACC'
Acc = []
F1 = []
AUC = []

for i in range(len(predtest)):
    Acc.append(accuracy_score(predtest[i], test_y)) #精度
    F1.append(f1_score(predtest[i], test_y,average='micro')) #f1
    #AUC.append(roc_auc_score(predtest[i], test_y))
print(Acc,F1,AUC)

'决策函数，ROC曲线绘制'
#逻辑回归，随机森林，xgboost比较
#.decision_function/.predict_proba:返回一个Numpy数组
#其中每个元素表示【分类器对x_test的预测样本是位于超平面的右侧还是左侧】，以及离超平面有多远。
log_score = logreg.decision_function(test_x)
rnd_score = rnd_clf.predict_proba(test_x) 
bag_score = bag_clf.predict_proba(test_x)
xgb_score = xgb_clf.predict_proba(test_x)

fpr1, tpr1, thresholds = roc_curve(test_y,log_score)
fpr2, tpr2, thresholds = roc_curve(test_y,rnd_score[:,1])
fpr3, tpr3, thresholds = roc_curve(test_y,bag_score[:,1])
fpr4, tpr4, thresholds = roc_curve(test_y,xgb_score[:,1])

plt.plot(fpr1,tpr1,label="Logreg")
'''plt.plot(fpr2,tpr2,label="Random Forest")
plt.plot(fpr3,tpr3,label="Bagging")
plt.plot(fpr4,tpr4,label="Xgboost")'''
plt.legend(loc="lower right",fontsize=16)
plt.grid(b=None)
plt.plot()
plt.xlabel('FPR',fontsize=14)
plt.ylabel('TPR',fontsize=14)
plt.tick_params(labelsize=13)
my_x_ticks = np.arange(0, 1, 0.1)
my_y_ticks = np.arange(0, 1, 0.1)
plt.xticks(my_x_ticks,fontsize=10)
plt.yticks(my_y_ticks,fontsize=10)

'''outfile.grid(False)
plt.show()'''














'''保存字段到csv'''
aab301.to_csv(r'E:\2022Work\人社\aab301.csv')



























