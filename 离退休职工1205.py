# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from time import time
import re
import math

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF #平稳性检验
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import chi2_contingency
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

import DaPy as dp

ac02 = pd.read_csv(r'F:\数据\企业保险\ac02.csv')

'''离退休职工'''
ic10 = pd.read_csv(r'E:\2022Work\人社\Data\ic10.csv')
ic10 = pd.read_csv(r'F:\数据\企业保险\ic10.csv')
ic10.shape #(2874191, 71)

#删除与业务无关的列
droplist = ['aaz257','aac001','aab071','aae013','aaz649',
            'aae860','aae859','aae011','aaz692','aae036',
            'aab034','aaa431','aaz673','aaa027','aaa508','aab360',
'aaa350','dxp_share_timestamp','dxp_share_batch_id','aaa508']
ic10 = ic10.drop(droplist,axis=1)

#删除缺失值过多的列
notnull = ic10.notnull().sum()
nulllist = list(notnull[notnull<100000].index)
ic10 = ic10.drop(nulllist, axis=1)
ic10.shape #(2874191, 32)
ic10.info()

#删除数据无意义的列
droplist2 = ['aic348', 'aic349', 'aae818',              
'aac330', 'aic357', 'aic358', 'aaf018']
ic10 = ic10.drop(droplist2,axis=1)
ic10.shape #(2874191, 26)
ic10.info()


'个人账户总金额'
ic10['aic165'].isnull().sum()
aic165 = pd.DataFrame(ic10['aic165'].dropna())
aic165_ = aic165[(aic165['aic165'] >0) & (aic165['aic165'] <1000) ] #筛选\
len(aic165_)
aic165_['aic165'].mean() #31165.48560702325

fig,ax = plt.subplots(figsize=(30, 30))
plt.hist(aic165_,bins=200, facecolor="steelblue", edgecolor="black", alpha=0.6)
#sns.violinplot(y=aic165_,
               #data=ic10,palette=['lightblue'])
#ax.set_ylim(0,1)
plt.xticks(size=30)
plt.yticks(size=30)
plt.show()

corrDf = ic10.corr()
corrDf['aic165'].sort_values(ascending=False)



'1 每月个人账户养老保险缴费'
'1.1 个人账户养老金总体情况'
#实际缴费月数
plt.figure(figsize=(17,10), dpi=200) 
plt.hist(ic10['aae201'],bins=600,range=(1,600))
ic10['aae201'].value_counts()[:15]
#视同缴费月数

ic10_aae201 = ic10.query('aae201<180 ') 
ic10['aic166'].value_counts()
ic10_aae201['aic166'].value_counts()


money_avg = pd.DataFrame((ic10['aic165']/ic10['aae201']).dropna())
money_avg_ = money_avg[(money_avg>0) & (money_avg <600) ] #筛选
index = money_avg_.dropna().index

len(money_avg_.dropna())
money_avg_.mean() #122.373865

fig,ax = plt.subplots(figsize=(20, 15))
plt.hist(money_avg_,bins=200, facecolor="steelblue", edgecolor="black", alpha=0.6)
#sns.violinplot(y=aic165_,
               #data=ic10,palette=['lightblue'])
#ax.set_ylim(0,1)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlabel('每月领取个人账户养老金',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('每月领取个人账户养老金总体情况',fontsize=40)
plt.show()



'1.2 相关分析'
ic10.loc[:,'money_avg'] = ic10['aic165']/ic10['aae201']
ic10_1 = ic10.query('money_avg<1000 & money_avg > 0 ') #过滤出大于0的money_avg

corrDf = ic10_1.corr()
vc_ic10 = corrDf['money_avg'].sort_values(ascending=False)

#画图
from matplotlib import rcParams
rcParams['axes.titlepad'] = 50 #标题与图间距
plt.figure(figsize=(15,10), dpi=150)
index = ['社会化管理形式','统筹地区','实际缴费月数','工龄',
         '行政职务级别','离退休类别','离退休日期','待遇发放方式',
         '专业技术职务级别']
b = sns.barplot(y=index,x=vc_ic10[1:10],
                palette="viridis")
b.set_xticklabels([0,0.05,0.10,0.15,0.20,0.15,
                   0.18,0.14,0.16],size=30)
b.set_yticklabels(index,size=30) #设置标签字体   
plt.xlabel('与个人缴费比例的相关系数',fontsize=30)
plt.title('与个人缴费比例相关性较强的属性',fontsize=40)
plt.show()


'1.3 模型构建'








'1.3 专项分析：社会化管理形式'
ic10_1['aae146'].dropna().value_counts()

#筛选社会化管理形式与平均金额不为空，平均金额0~200元之间的人
ic10_aae146 = ic10_1[(ic10_1['aae146'].notnull()) & (ic10_1['aae146'] != "")
                     &(ic10_1['money_avg'] < 200)]
ic10_aae146['aae146'].value_counts()

from matplotlib import rcParams
rcParams['axes.titlepad'] = 30 #标题与图间距
plt.figure(figsize=(17,10), dpi=200) 
sns.violinplot('aae146','money_avg',
               data=ic10_aae146)
plt.xticks(ticks=[0,1,2,3], labels=['社区或村管理','社会保险经办机构管理',
                '依托企业管理','其他方式管理'],rotation = 45,fontsize=24)
plt.yticks(fontsize=30)
plt.xlabel('社会化管理形式',fontsize=30)
plt.ylabel('每个月个人账户养老金',fontsize=30)
plt.title('不同社会化管理形式的职工的个人账户养老金',fontsize=40)
plt.show()

#选出平均金额0~20元之间的低收入人群
ic10_aae146_2 = ic10_1[(ic10_1['aae146'].notnull()) & (ic10_1['aae146'] != "")
                     &(ic10_1['money_avg'] < 20)]
ic10_aae146_2['aae146'].value_counts().sort_index()

#平均每月个人养老金
print('%.2f' % ic10_1[ic10_1['aae146'] == 1.0]['money_avg'].mean(),
'%.2f' % ic10_1[ic10_1['aae146'] == 2.0]['money_avg'].mean(),
'%.2f' % ic10_1[ic10_1['aae146'] == 3.0]['money_avg'].mean(),
'%.2f' % ic10_1[ic10_1['aae146'] == 9.0]['money_avg'].mean())

#社会保险经办机构管理
plt.figure(figsize=(15,10), dpi=150) 
ax1 = plt.subplot(211) #子图
rects1 = plt.hist((ic10_1[ic10_1['aae146'] == 2.0]['money_avg']),
         1000,alpha=0.9,color = 'b',label='新人',range=(0,300))
rects2 = plt.hist((ic10_1[ic10_1['aae146'] == 1.0]['money_avg']),
         1000,alpha=0.9,color = 'r',label='中人',
         range=(0,300))
plt.legend(loc='upper right',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.yticks(fontsize=30)
plt.xlabel('缴费满15年后的月个人缴费金额',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('缴费满15年后的中人和新人的月个人缴费金额',fontsize=40)
plt.show()



'1.4 离退休类别'
ic10_aic161 = ic10_1[(ic10_1['aic161'].notnull()) & (ic10_1['aic161'] != "")
                     &(ic10_1['money_avg'] < 10000)]
ic10_aic161['aic161'].value_counts().sort_index()




'1.5 离退休时基本养老保险个人账户中个人缴费部分所占比例'
ic10['aic167'].isnull().sum() 
aic167 = ic10['aic167'].dropna()


plt.hist(ic10['aic167'],bins=300,range=(0.0001,0.9999))
plt.ylim(0,40000)
plt.xlabel('个人缴费部分所占比例',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('个人账户养老金个人缴费比例分布',fontsize=40)
plt.show()

ic10_aic167_1 = ic10.query('aic167 == 0 ')
ic10_aic167_2 = ic10.query('aic167 > 0 & aic167 < 0.6 ')
ic10_aic167_3 = ic10.query('aic167 == 1')
ic10_aic167_4 = ic10.query('aic167 > 0.95 & aic167 < 1 ')
print(len(ic10_aic167_1),len(ic10_aic167_2),
      len(ic10_aic167_3),len(ic10_aic167_4))

ic10['mark_aic167'] = np.array(0,dtype='float32')
#ic10['mark_aic167'][ic10_aic167_1.index]=np.array(1,dtype='float32')
ic10['mark_aic167'][ic10_aic167_2.index]=np.array(2,dtype='float32')
#ic10['mark_aic167'][ic10_aic167_3.index]=np.array(3,dtype='float32')
#ic10['mark_aic167'][ic10_aic167_4.index]=np.array(4,dtype='float32')
ic10.corr()['mark_aic167'].sort_values(ascending=False)

plt.figure(figsize=(30,20), dpi=200) 
sns.set(font_scale = 3)
sns.set_style('whitegrid', {'font.sans-serif': ['simhei','FangSong']})
sns.heatmap(ic10.corr(),cmap='YlGnBu') 

#个人缴费率较低
#替换区域
ic10_aab359_taiyuan = ic10.query('140101 <= aab359 < 140200 ')
ic10_aab359_datong = ic10.query('140200 <= aab359 < 140300 ')
ic10_aab359_yangquan = ic10.query('140300 <= aab359 < 140400 ')
ic10_aab359_changzhi = ic10.query('140400 <= aab359 < 140500 ')
ic10_aab359_jincheng = ic10.query('140500 <= aab359 < 140600 ')
ic10_aab359_shuozhou = ic10.query('140600 <= aab359 < 140700 ')
ic10_aab359_jinzhong = ic10.query('140700 <= aab359 < 140800 ')
ic10_aab359_yuncheng = ic10.query('140800 <= aab359 < 140900 ')
ic10_aab359_yizhou = ic10.query('140900 <= aab359 < 141000 ')
ic10_aab359_linfen = ic10.query('141000 <= aab359 < 141100 ')
ic10_aab359_lvliang = ic10.query('141100 <= aab359 < 141200 ')
ic10_aab359_qita = ic10.query('141200 <= aab359 | aab359 < 140100 ')

ic10['地区'] = 0
ic10['地区'][ic10_aab359_taiyuan.index]= '太原'
ic10['地区'][ic10_aab359_datong.index]= '大同'
ic10['地区'][ic10_aab359_yangquan.index]= '阳泉'
ic10['地区'][ic10_aab359_changzhi.index]= '长治'
ic10['地区'][ic10_aab359_jincheng.index]= '晋城'
ic10['地区'][ic10_aab359_shuozhou.index]= '朔州'
ic10['地区'][ic10_aab359_jinzhong.index]= '晋中'
ic10['地区'][ic10_aab359_yuncheng.index]= '运城'
ic10['地区'][ic10_aab359_yizhou.index]= '沂州'
ic10['地区'][ic10_aab359_linfen.index]= '临汾'
ic10['地区'][ic10_aab359_lvliang.index]= '吕梁'
ic10['地区'][ic10_aab359_qita.index]= '其他'
ic10['地区'].value_counts()

ic10 = ic10.drop('aab359',axis=1)
ic10_aic167 = ic10
diqudf = pd.get_dummies(ic10['地区'],prefix = '地区')
ic10_aic167 = pd.concat([ic10_aic167,diqudf],axis=1)
ic10_aic167.eval('工龄 =(aic162-aac327)/10000',inplace=True) #新增工龄
ic10_aic167 = ic10_aic167.drop('aic351',axis=1)
ic10_aic167 = ic10_aic167.query('地区_其他 ==0')

from matplotlib import rcParams
rcParams['axes.titlepad'] = 30 #标题与图间距
plt.figure(figsize=(17,10), dpi=200) 

index = (ic10_aic167.corr()['mark_aic167']*5).sort_values(ascending=False).sort_index()[-13:-1].index
sns.barplot(y=index,
            x=(ic10_aic167.corr()['mark_aic167']*5).sort_values(ascending=False).sort_index()[-13:-1],
            palette="viridis")
plt.yticks(fontsize=30)
plt.xlabel('相关系数',fontsize=30)
plt.ylabel('地区',fontsize=30)
plt.title('各地级市的相关性',fontsize=40)
plt.show()



'''三类人'''
ic10_aaa347 = ic10_aic167.query("aaa347 == '按21号文'") #中人
ic10_aaa347_2 = ic10_aic167.query("aaa347 == '按32号文'") #新人
ic10_aaa347_3 = ic10_aic167.query("aaa347 == '按104号文'") #老人
ic10_aaa347_4 = ic10_aic167.query("aaa347 == '按116号文'")

rcParams['axes.titlepad'] = 30 #标题与图间距
plt.figure(figsize=(17,10), dpi=200) 
def people(column,bins,start,end,i):
    x = [ic10_aaa347[column]/i,ic10_aaa347_2[column]/i,
        ic10_aaa347_3[column]/i,ic10_aaa347_4[column]/i]
    color = ['r','b','orange','seagreen']
    label = ['新人','中人','老人','未知']
    plt.hist(x=x,
         bins=bins,range=(start,end),label=label,
         alpha=0.7,color = color,stacked=True )
people('工龄',80,30,70,1)
plt.legend(loc='upper right',fontsize=16)
plt.yticks(fontsize=30)
plt.xlabel('退休年龄',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('三类人的退休年龄分布',fontsize=40)
plt.show()




'''aae201    实际缴费月数

'''

'离退休时个人账户养老金占基本养老金的比例'
#养老金个人账户为0，占比也为0
ic10_aic166 = ic10.query('aic166 == 0 & aic165 == 0 ')
#给这部分人在原始数据库里打上标记
ic10['mark'] = 0
ic10['mark'][ic10.query('aic166 == 0 & aic165 == 0 ').index]=1
ic10.corr()['mark'].sort_values()





ic10['aic166'].isnull().sum() 
aic166 = pd.DataFrame(ic10['aic166'].dropna())
aic166_ = aic166[(aic166['aic166'] >0) & (aic166['aic166'] <1) ] 


fig,ax = plt.subplots(figsize=(30, 30))
plt.hist(aic166_,bins=200, facecolor="steelblue", edgecolor="black", alpha=0.6)
#sns.violinplot(y=aic165_,
               #data=ic10,palette=['lightblue'])
#ax.set_ylim(0,1)
plt.xticks(size=30)
plt.yticks(size=30)
plt.show()

Counter(ic10['aaa347'])









'养老收支模型'
'平均工资'
salary = pd.DataFrame([15645,18300,21525,25828,28469,33544,39903,44943,
                  47417,49984,52960,54975,61547,67669,72207,77364,84938])

salary.plot()
plt.show()

#自相关图
plot_acf(salary).show()

# 一阶差分后的结果
D_data = salary.diff().dropna()
D_data.columns = [u'工资差分']
D_data.plot()  # 时序图
plt.show()

print(u'差分序列的ADF检验结果为：', ADF(D_data[u'工资差分'])) #平稳性检验

#二阶差分
D_data2 = D_data.diff().dropna()
D_data.columns = [u'工资二阶差分']
D_data2.plot()  # 时序图
plt.show()

print(u'差分序列的ADF检验结果为：', ADF(D_data[u'工资差分'])) #平稳性检验
plot_acf(D_data2).show()  # 自相关图
plot_pacf(D_data2,lags=6).show()  # 偏自相关图

model = ARIMA(salary, order=(1,2,1)).fit()  # 建立ARIMA(1, 2, 1)模型
#print('模型报告为：\n', model.summary2())
print('预测未来5年，其预测结果如下：\n', model.forecast(5))

plt.plot(salary)
plt.plot(model.forecast(5))


'当期补缴收入'
ac02['补缴年份'] = ac02['aae041']//100 #aae041：补缴日期
#当年补缴缴费基数总和
Sum_jishu = ac02.groupby('补缴年份')['aae180'].sum() # aae180：人员缴费基数
#当年补缴收入
Sum_shouru = Sum_jishu *0.28
Sum_shouru['2016'] = Sum_shouru['2016']*0.27/0.28
Sum_shouru['2017'] = Sum_shouru['2017']*0.27/0.28


'企业参保在职人数'
people = pd.DataFrame([289.23,302.44,311.12,336.27,350.1,366.02,382.63,396.89
,406.44,418.43,426.25,435.7,443.4,457.1,480.6])
people.plot()
plt.show()

D_data = people.diff().dropna()
D_data.columns = [u'人数差分']
D_data.plot()  # 时序图
plt.show()
print(u'差分序列的ADF检验结果为：', ADF(D_data[u'人数差分'])) #平稳性检验
plot_acf(D_data).show()  # 自相关图
plot_pacf(D_data,lags=6).show()  # 偏自相关图

model = ARIMA(people, order=(1,1,1)).fit()  # 建立ARIMA(1, 1, 1)模型
#print('模型报告为：\n', model.summary2())
print('预测未来7年，其预测结果如下：\n', model.forecast(7))

plt.plot(people)
plt.plot(model.forecast(7))




'山西省城镇企业职工养老保险基金支出模型'
'①当年合计核定计发待遇'
#月个人账户养老金 =  本人退休时个人账户储存额 / 计发月数
#退休年龄 aic162:离退休日期 aac327：档案出生日期
ic10['Retire_age'] = ((ic10['aic162'] - ic10['aac327'])//10000).dropna().astype(int)
Mapping = {'40': '233', '41': '230', '42': '226', '43':'223', '44':'220', '45':'216',
           '46': '212', '47': '208', '48': '204', '49':'199', '50':'195', '51':'190',
           '52': '185', '53': '180', '54': '175', '55':'170', '56':'164', '57':'158',
          '58': '152', '59': '145', '60': '139', '61':'132', '62':'125', '63':'117',
          '64': '109', '65': '101', '66': '93', '67':'84', '68':'75', '69':'65','70':'56'} 
ic10['Count_month'] = ic10['Retire_age'].replace(Mapping) #计发月数
ic10['ind_pension'] = ic10['aic165']/ic10['Count_month'] #月个人账户养老金

##异常提前退休
early_retire_1 = ic10.query("'Retire_age' < 50 & 'aac004' == 2 & 'aic349' == 0 & 'aac012' == '普通职工'") 
early_retire_2 = ic10.query("'Retire_age' < 55 & 'aac004' == 2 & 'aic349' == 0 & 'aac012' == '女性干部'")
early_retire_3 = ic10.query("'Retire_age' < 60 & 'aac004' == 1 & 'aic349' == 0 & 'aac012' == '普通职工'")
year1 = 50- ic10['Retire_age']
year2 = 55- ic10['Retire_age']
year3 = 60- ic10['Retire_age']
ic10['ind_pension'][early_retire_1.index] * year1 + ic10['ind_pension'][early_retire_2.index] * year2
ic10['ind_pension'][early_retire_3.index] * year3

##重复参保
ic10['ic10_rep'] = ic10['aac001'].value_counts()
Counter(ic10['aac001'].value_counts().values)
rep = ic10.query("'aac001' == '110000009909984' | 'aac001' == '110000006863507' | 'aac001' == '110000009797420' |'aac001' == '110000004880639'" )
ic10['ind_pension'][rep.index]



'平均养老金支出'
a = [1,2,3,4,5,6,7,8,9]
b = [9039.7,9351.3,11431.9,11596.8,12401.1,13753,19580.4,23294,23817.6]
c = sm.add_constant(a) 
fit=sm.OLS(b,c).fit()
print(fit.summary())
print(fit.params)
test_X = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]
pred = fit.predict(exog = test_X)

#预测总支出
a=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
b=[40000,62074,74148,186222,198296,430370,652444,574518,196592,
241098,361479,560372,543829,814360,1230924,1057102]
c = sm.add_constant(a) 
fit=sm.OLS(b,c).fit()
print(fit.summary())
print(fit.params)



'''
aic167 离退休时基本养老保险个人账户中个人缴费部分所占比例
aac327 档案出生日期
aic351 法定离退休日期
aic162 离退休日期
aic355 计息截止年月
aab360 经办地行政区划代码
aic166 离退休时个人账户养老金占基本养老金的比例
aic378 建账后年限（月数）
aac020    0.164090 行政职务级别
aae146    0.162761 社会化管理形式
aic377    0.153643 建账前年限（月数）
aab071    0.069295 退休申报单位编号
aac014   -0.194105 专业技术职务级别
aae036   -0.431464 经办时间
'''
'''
aae146       0.165446 社会化管理形式
aic167       0.105969 离退休时基本养老保险个人账户中个人缴费部分所占比例
aic165       0.060253 总金额
aac020       0.057208 行政职务级别
aic161       0.043466 离退休类别
aic383       0.042060 地方离退休类别
aae036       0.035363 经办时间
aic162       0.020234 离退休日期
aaa027      -0.053753 统筹区编码
aic166      -0.062125 离退休时个人账户养老金占基本养老金的比例
aic351      -0.084213 法定离退休日期
aac327      -0.093167 档案出生日期
aac015      -0.093299 国家职业资格等级（技能人员等级）
aae200      -0.149337 视同缴费月数
aac014      -0.200423 专业技术职务级别
aae201      -0.319044 实际缴费月数
'''
'''





