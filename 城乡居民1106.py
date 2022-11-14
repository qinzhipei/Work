# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from time import time
import re

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score,roc_curve,roc_auc_score


from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier #决策树
from sklearn.ensemble import BaggingClassifier #Bagging
from sklearn.ensemble import RandomForestClassifier #随机森林
import xgboost
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline #pip install imblearn --user


ab0102 = pd.read_csv(r'F:\数据\企业保险\ab0102.csv')


'''城乡居民'''
'''基本险/补充险领取分布'''
'''ac01:'人员基本信息表'''
ac01 = pd.read_csv(r'E:\2022work\人社\数据\ac01.csv')
ac01.head()
ac01.info()
ac01.isnull().sum()
Counter(ac01['aac085'])

sex = ac01['aac004']

'人员结构直方图，2个子图'
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20 #标题与图间距
#男性
plt.figure(figsize=(10,10), dpi=800) 
ax1 = plt.subplot(211) #子图
plt.hist(ac01[ac01['aac004'] == 1]['aac006']/10000,
         110,alpha=0.9,color = 'r',label='男性',range=(1910,2020))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=16)
plt.xlabel('出生日期',fontsize=16)
plt.ylabel('人数',fontsize=16)
plt.title('城乡居民年龄结构',fontsize=20)
#调整间隔
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, \
    wspace=None, hspace=0.3)
#女性
ax2 = plt.subplot(212)
plt.hist(ac01[ac01['aac004'] == 2]['aac006']/10000,
         110,alpha=0.9,label='女性',
         range=(1910,2020),histtype='barstacked')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='upper right',fontsize=16)
plt.xlabel('出生日期',fontsize=16)
plt.ylabel('人数',fontsize=16)
plt.show()


'基本险到龄'
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.hist(ac01[ac01['aac084'] == 1]['aac006']/10000,
         12,alpha=0.6,label='享受基本险待遇',range=(1955,1967))
plt.hist(ac01[ac01['aac084'] == 0]['aac006']/10000,
         12,alpha=0.9,label='未享受基本险待遇',range=(1955,1967))
plt.legend(loc='upper right')
plt.xlabel('年龄')
plt.ylabel('基本险到龄享受待遇人数')
plt.title('基本险到龄享受待遇情况')
plt.show()


'补充险到龄享受待遇标志'
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.hist(ac01[ac01['aac085'] == 1]['aac006']/10000,
         15,alpha=0.8,label='享受补充险待遇',range=(1950,1965))
plt.hist(ac01[ac01['aac085'] == 0]['aac006']/10000,
         15,alpha=0.7,label='未享受补充险待遇',range=(1950,1965))
plt.legend(loc='upper right')
plt.xlabel('出生日期')
plt.ylabel('补充险到龄享受待遇人数')
plt.title('补充险到龄享受待遇情况')
plt.show()

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.hist(ac01[ac01['aac084'] == 1]['aac006']/10000,
         100,alpha=0.6,label='享受基本险待遇',range=(1960,1970))
plt.hist(ac01[ac01['aac084'] == 0]['aac006']/10000,
         100,alpha=0.9,label='未享受基本险待遇',range=(1960,1970))
plt.legend(loc='upper right')
plt.xlabel('年龄')
plt.ylabel('基本险到龄享受待遇人数')
plt.title('基本险到龄享受待遇情况')
plt.show()



'''建模'''
output1 = pd.read_csv(r'C:\Users\Administrator\Desktop\output1.csv')
output1.head()
output1.info() #251536

output2 = pd.read_csv(r'C:\Users\Administrator\Desktop\output2.csv')
output2.head()
output2.info()

outfile = pd.concat([output1,output2],axis=0)
outfile = outfile.drop(251536)
outfile.info()

Y = outfile['补充险缴纳情况']
gender = outfile['性别']
date = outfile['年龄']
survive = outfile['残疾情况']
money = outfile['基本险缴费金额']
disabled = outfile['人员学历']
household = outfile['户口性质']




'''特征工程'''
'''填充生存状态'''
outfile['生存状态'] = 1
outfile.loc[(outfile['出生日期']> 19450101) , '生存状态' ] = 0

'''填充残疾'''
outfile['残疾类别'] = outfile['残疾类别'].fillna(0) 

'''户口性质缺失值太多，删除'''
outfile = outfile.drop('户口性质',1)
'''残疾类别与残疾登记高度相关，删除一个'''
outfile = outfile.drop('残疾登记',1)

'''扩充'''
outfile['补充险到龄标记'] = 1
outfile.loc[(outfile['出生日期']> 19570101) , '补充险到龄标记' ] = 0

outfile['基本险到龄标记'] = 1
outfile.loc[(outfile['出生日期']> 19620101) , '基本险到龄标记' ] = 0

print(Counter(outfile['生存状态']))
print(Counter(outfile['补充险到龄标记']))
print(Counter(outfile['基本险到龄标记']))

'''出生日期与生存状态，到龄高度相关，删除'''
outfile = outfile.drop('出生日期',1)

'''哑变量处理'''
'''残疾类别'''
disabledDf = pd.get_dummies(survive,prefix = 'disabled') #prefix：前缀
disabledDf.head()
'''缴费金额'''
moneyDf = pd.get_dummies(money,prefix = 'money')
moneyDf.head()
'''性别'''
genderDf = pd.get_dummies(gender,prefix = 'gender')
'''缴费状态'''
paymentDf = pd.get_dummies(disabled,prefix = 'payment')

outfile = pd.concat([outfile,disabledDf,moneyDf,genderDf,paymentDf],axis = 1) #把哑变量列加到数据集里
outfile = outfile.drop(['性别','缴费状态','缴费金额','残疾类别'],axis=1)#把原embarked,pclass列删掉


'''相关分析'''
corrDf = outfile.corr()
#绘图
sns.set(font_scale = 1)
sns.set_style('whitegrid', {'font.sans-serif': ['simhei','FangSong']})
sns.heatmap(outfile.corr(),cmap='YlGnBu') #sns.heatmap：相关矩阵热力图
corrDf['补充险缴纳情况'].sort_values(ascending=False) 


'''模型训练'''
Source_X = outfile.drop(['补充险缴纳情况'],axis=1)
Source_Y = outfile['补充险缴纳情况']

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
importancesort = np.sort((importance**0.2+0.02-importance)**2)[::-1]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.tick_params(labelsize=10)
#绘制柱状图
label = ['年龄_40-60岁','性别','残疾状况_重度','残疾状况_中度','残疾状况_无残疾','残疾状况_轻度',
                    '学历_硕士毕业生',
                    '年龄_0-20岁','基本险缴纳_5000元','学历_博士研究生',
                    '学历_小学','学历_初中','学历_高中',
                    '学历_本科','基本险缴纳_700元',
                    '户口性质_城镇户口',
                    '年龄_20-40岁','学历_职高','基本险缴纳_2000元',
                    '基本险缴纳_1000元','基本险缴纳_500元','学历_中专',
                    '户口性质_非城镇户口','基本险缴纳_300元','基本险缴纳_200元']
ax.barh(label,importancesort)
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


'K折交叉验证'
#cross_val_score：每次迭代的交叉验证评分，需要取均值
def K_Fold_Score(model):
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X=Source_X, scoring='accuracy',
                             y=Source_Y, cv=10, n_jobs=1)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))
    return (np.mean(scores))

K_Acc = []
for i in range(len(model)):
    K_Acc.append(K_Fold_Score(model[i])) #xgb报错

'ACC F1 混淆矩阵 AUC K折ACC'
Acc = []
F1 = []
AUC = []

for i in range(len(predtest)):
    Acc.append(accuracy_score(predtest[i], test_y)) #精度
    F1.append(f1_score(predtest[i], test_y)) #f1
    AUC.append(roc_auc_score(predtest[i], test_y))
print(Acc,F1,AUC,K_Acc)

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

plt.plot(fpr1,tpr1,label="Logreg",)
plt.plot(fpr2,tpr2,label="Random Forest")
plt.plot(fpr3,tpr3,label="Bagging")
plt.plot(fpr4,tpr4,label="Xgboost")
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


outfile.grid(False)
plt.show()






outfile[survive.isnull()].index
def function(a,b):
    for i in range(len(a)):
        if (a[i] < 19470101) & (b[i] == np.nan):
           b[i] = 1
        elif (a[i] >= 19470101) & (b[i] == np.nan):
           b[i]= 0
        elif b[i] == np.nan:
             continue
         
function(date,survive)


