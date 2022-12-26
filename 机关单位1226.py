# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


ab02 = pd.read_csv(r'E:\2022Work\人社\Data\ab02.csv') #机关单位基本信息表
ac01 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ac01.csv')
ac02 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ac02.csv')
ac29 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ac29.csv') #人员在职转退休事件
ac43 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ac43.csv',nrows=1000000) #人员征缴明细
ac50 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ac50.csv') #个人账户
ic10 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ic10.csv')
id19 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ic19.csv') #年金定期待遇
id72 = pd.read_csv(r'E:\2022Work\人社\Data\gio_ic72.csv') #年金个人实账账户表


'退休年龄'
'ac01,ac29'
merge = pd.merge(ac01,ac29,on='aac001')
#男性普通职工 438023
outfile_male1 = merge[merge[['aac222']].isnull().T.any()]
outfile_male1 = outfile_male1.query("aac004==1")
#男性高级干部
outfile_male2 = merge[merge['aac222'].notna()]
outfile_male2 = outfile_male2.query("aac004==1 & aac222==1")
#女性职工 399502
outfile_female1 = merge[merge[['aac222']].isnull().T.any()]
outfile_female1 = outfile_female1.query("aac004==2")
#女性高级干部 17783
outfile_female2 = merge[merge['aac222'].notna()]
outfile_female2 = outfile_female2.query("aac004==2 & aac222==1")

#退休年龄：退休日期-出生日期
outfile_male1['male_RA'] = ((outfile_male1['aic162']-outfile_male1['aac006'])/10000).round(0).astype(int)
outfile_male2['male_RA'] = ((outfile_male2['aic162']-outfile_male2['aac006'])/10000).round(0).astype(int)
outfile_female1['female_RA'] = ((outfile_female1['aic162']-outfile_female1['aac006'])/10000).round(0).astype(int)
outfile_female2['female_RA'] = ((outfile_female2['aic162']-outfile_female2['aac006'])/10000).round(0).astype(int)

rcParams['axes.titlepad'] = 30 
plt.rcParams.update({'font.size':24})

fig,ax = plt.subplots(figsize=(20, 15))
#ax.set(xlim=[45,65])
ax.hist(outfile_male1['male_RA'],bins=2000,width=0.8, facecolor="steelblue", 
        edgecolor="black", alpha=0.6, label='男性普通职工及初级干部')
ax.hist(outfile_male2['male_RA'],bins=2000,width=0.8, facecolor="black", 
        edgecolor="black", alpha=0.9, label='男性高级干部')
ax.hist(outfile_female1['female_RA'],bins=2000,width=0.8, facecolor="orange", edgecolor="black", 
         alpha=0.6, label='女性普通职工及初级干部')
ax.hist(outfile_female2['female_RA'],bins=2000,width=0.8, facecolor="red", edgecolor="black", 
         alpha=0.6, label='女性高级干部')
leg = ax.legend()
ax.legend(loc='upper left', frameon=False)
#设置标签列数
ax.legend(frameon=False, loc='lower center', ncol=2,fontsize=32)
#fancybox:圆角边框  framealpha:外边框透明度 shadow：增加阴影 borderpad：
ax.legend(fancybox=True, framealpha=0.9, shadow=True, borderpad=2)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlim(45,65)
plt.ylim(0,110000)
plt.xlabel('退休年龄（岁）',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('机关单位离退休人员退休年龄总体分布',fontsize=40)
plt.show()

#各个退休年龄的占比
round(outfile_male1['male_RA'].value_counts()/len(outfile_male1['male_RA'])*100,2).sort_index().loc[:64].sum()
round(outfile_male2['male_RA'].value_counts()/len(outfile_male1['male_RA'])*100,2).sort_index().loc[:64].sum()
round(outfile_female1['female_RA'].value_counts()/len(outfile_female1['female_RA'])*100,2).sort_index().loc[:49].sum()
round(outfile_female2['female_RA'].value_counts()/len(outfile_female2['female_RA'])*100,2).sort_index().loc[:54].sum()




'''视同缴费年限\合计缴费年限\视同缴费指数'''
acic10_male_1 = pd.merge(outfile_male1, ic10, on='aac001')
acic10_male_2 = pd.merge(outfile_male2, ic10, on='aac001')
acic10_female_1 = pd.merge(outfile_female1, ic10, on='aac001')
acic10_female_2 = pd.merge(outfile_female2, ic10, on='aac001')
outfile_male1 = acic10_male_1.query("aae200>0")
outfile_male1['实际缴费'] = outfile_male1['aae200'] +outfile_male1['aae201']
outfile_male2 = acic10_male_2.query("aae200>0")
outfile_male2['实际缴费'] = outfile_male2['aae200'] +outfile_male2['aae201']
outfile_female1 = acic10_female_1.query("aae200>0")
outfile_female1['实际缴费'] = outfile_female1['aae200'] +outfile_female1['aae201']
outfile_female2 = acic10_female_2.query("aae200>0")
outfile_female2['实际缴费'] = outfile_female2['aae200'] +outfile_female2['aae201']

outfile_male1.groupby('male_RA')['aae200'].median()/12
outfile_male1.groupby('male_RA')['实际缴费'].median()/12
outfile_male1.groupby('male_RA')['aae294'].median()
outfile_male2.groupby('male_RA')['aae200'].median()/12
outfile_male2.groupby('male_RA')['实际缴费'].median()/12
outfile_male2.groupby('male_RA')['aae294'].median()
outfile_female1.groupby('male_RA')['aae200'].median()/12
outfile_female1.groupby('male_RA')['实际缴费'].median()/12
outfile_female2.groupby('male_RA')['aae200'].median()/12
outfile_female2.groupby('male_RA')['实际缴费'].median()/12


rcParams['axes.titlepad'] = 30 
plt.rcParams.update({'font.size':20}) #图例大小

fig,ax = plt.subplots(figsize=(20, 10), dpi=150)
ax.plot(outfile_male1.groupby('male_RA')['aae200'].median()/12,'-ok', color='green',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='男性普通职工及初级干部')
ax.plot(outfile_male2.groupby('male_RA')['aae200'].median()/12,'-ok', color='orange',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='男性高级干部')
ax.plot(outfile_female1.groupby('female_RA')['aae200'].median()/12,'-ok', color='steelblue',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='女性普通职工及初级干部')
ax.plot(outfile_female2.groupby('female_RA')['aae200'].median()/12,'-ok', color='red',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='女性高级干部')
#设置标签列数
ax.legend(frameon=False, ncol=2)
#fancybox:圆角边框  framealpha:外边框透明度 shadow：增加阴影 borderpad：
ax.legend(fancybox=True, framealpha=0.8 ,loc='lower left',shadow=True, borderpad=2)
leg = ax.legend()
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlim(45,63)
plt.ylim(20,50)
plt.xlabel('退休年龄（岁）',fontsize=30)
plt.ylabel('视同缴费年限（年）',fontsize=30)
plt.title('视同缴费年限随退休年龄的变化',fontsize=40)
plt.show()





'''月工资、平均缴费基数比例'''
ac0143 = pd.merge(ac01,ac43,on='aac001')
#男性普通职工 438023
outfile_male1 = ac0143[ac0143[['aac222']].isnull().T.any()]
outfile_male1 = outfile_male1.query("aac004==1")
#男性高级干部
outfile_male2 = ac0143[ac0143['aac222'].notna()]
outfile_male2 = outfile_male2.query("aac004==1 & aac222==1")
#女性职工 399502
outfile_female1 = ac0143[ac0143[['aac222']].isnull().T.any()]
outfile_female1 = outfile_female1.query("aac004==2")
#女性高级干部 17783
outfile_female2 = ac0143[ac0143['aac222'].notna()]
outfile_female2 = outfile_female2.query("aac004==2 & aac222==1")


'工资'
rcParams['axes.titlepad'] = 30 
plt.rcParams.update({'font.size':24}) #字体大小

fig,ax = plt.subplots(figsize=(20, 15))
#ax.set(xlim=[45,65])
ax.hist(outfile_male1['aac040'],bins=5000,width=15, facecolor="steelblue", 
        alpha=0.6, label='男性普通职工及初级干部')
ax.hist(outfile_male2['aac040'],bins=5000,width=15, facecolor="black",
         alpha=0.6, label='男性高级干部')
ax.hist(outfile_female1['aac040'],bins=5000,width=15, facecolor="orange",
         alpha=0.6, label='女性普通职工及初级干部')
ax.hist(outfile_female2['aac040'],bins=5000,width=15, facecolor="red",
         alpha=0.6, label='女性高级干部')

#平均月工资
print(outfile_male1['aac040'].mean(),outfile_male2['aac040'].mean(),
      outfile_female1['aac040'].mean(),outfile_female2['aac040'].mean())


leg = ax.legend()
ax.legend(loc='upper left', frameon=False)
ax.legend(frameon=False, loc='lower center', ncol=2,fontsize=32)
ax.legend(fancybox=True, framealpha=0.9, shadow=True, borderpad=2)
ax.text(0.5, 0.6, "男性普通职工及初级干部平均月工资：4885.5元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.5, 0.53, "男性高级干部平均月工资：5018.8元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.5, 0.46, "女性普通职工及初级干部平均月工资：4532.1元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.5, 0.39, "女性高级干部平均月工资：4596.3元", 
        transform=ax.transAxes,fontdict=None)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlim(0,12000)
plt.ylim(0,4000)
plt.xlabel('平均月工资（元）',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('机关单位离退休人员平均月工资总体分布',fontsize=40)
plt.show()


'不同退休年龄'
ac014329 = pd.merge(ac0143,ac29,on='aac001')
outfile_total = ac014329.query("2000<aac040<15000")
wage_total = outfile_total['aac040'].mean() #平均工资

outfile_male1 = ac014329[ac014329[['aac222']].isnull().T.any()]
outfile_male1 = outfile_male1.query("aac004==1")
outfile_male2 = ac014329[ac014329['aac222'].notna()]
outfile_male2 = outfile_male2.query("aac004==1 & aac222==1")
outfile_female1 = ac014329[ac014329[['aac222']].isnull().T.any()]
outfile_female1 = outfile_female1.query("aac004==2")
outfile_female2 = ac014329[ac014329['aac222'].notna()]
outfile_female2 = outfile_female2.query("aac004==2 & aac222==1")

outfile_male1['male_RA'] = ((outfile_male1['aic162']-outfile_male1['aac006'])/10000).round(0).astype(int)
outfile_male2['male_RA'] = ((outfile_male2['aic162']-outfile_male2['aac006'])/10000).round(0).astype(int)
outfile_female1['female_RA'] = ((outfile_female1['aic162']-outfile_female1['aac006'])/10000).round(0).astype(int)
outfile_female2['female_RA'] = ((outfile_female2['aic162']-outfile_female2['aac006'])/10000).round(0).astype(int)

#修正离群值
result = outfile_female2.groupby('female_RA')['aac040'].mean() 
result.loc[57] = (result.loc[56] + result.loc[58])/2
result.loc[59] = (result.loc[58] + result.loc[60])/2
result.loc[61] = (result.loc[60] + 0.97*result.loc[63])/2
result2 = outfile_female1.groupby('female_RA')['aac040'].mean() 
result2.loc[53] = (result.loc[52] + result.loc[54])/2

rcParams['axes.titlepad'] = 30 
plt.rcParams.update({'font.size':24})

fig,ax = plt.subplots(figsize=(20, 10), dpi=150)
ax.plot(outfile_male1.groupby('male_RA')['aac040'].mean(),'-ok', color='orange',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='男性普通职工及初级干部')
ax.plot(outfile_male2.groupby('male_RA')['aac040'].mean(),'-ok', color='black',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='男性高级干部')
ax.plot(result2,'-ok', color='steelblue',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='女性普通职工及初级干部')
ax.plot(result,'-ok', color='red',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='女性高级干部')
#设置标签列数
ax.legend(frameon=False, ncol=2)
#fancybox:圆角边框  framealpha:外边框透明度 shadow：增加阴影 borderpad：
ax.legend(fancybox=True, framealpha=0.8 ,loc='lower left',shadow=True, borderpad=2)
leg = ax.legend()
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlim(47,62)
plt.ylim(2000,10000)
plt.xlabel('退休年龄（岁）',fontsize=30)
plt.ylabel('退休前上年度平均月工资（元）',fontsize=30)
plt.title('退休前上年度平均月工资随退休年龄的变化',fontsize=40)
plt.show()








'''个人账户养老金月标准'''
'总金额分布'
ac0150 = pd.merge(ac01,ac50,on='aac001')
#ac0150 = pd.merge(ac0150,ic10,on='aac001') #选出养老离退休人员，分析账户金额 279782条
outfile_male1 = ac0150[ac0150[['aac222']].isnull().T.any()]
outfile_male1 = outfile_male1.query("aac004==1")
outfile_male2 = ac0150[ac0150['aac222'].notna()]
outfile_male2 = outfile_male2.query("aac004==1 & aac222==1")
outfile_female1 = ac0150[ac0150[['aac222']].isnull().T.any()]
outfile_female1 = outfile_female1.query("aac004==2") #109148
outfile_female2 = ac0150[ac0150['aac222'].notna()]
outfile_female2 = outfile_female2.query("aac004==2 & aac222==1") #5075

print(outfile_male1['aae240'].mean(),outfile_male2['aae240'].mean(), 
      outfile_female1['aae240'].mean(), outfile_female2['aae240'].mean())


rcParams['axes.titlepad'] = 30 
plt.rcParams.update({'font.size':24}) #字体大小

fig,ax = plt.subplots(figsize=(20, 15))
#ax.set(xlim=[45,65])
ax.hist(outfile_male1['aae240'],bins=20000,width=18, facecolor="steelblue", 
        alpha=0.6, label='男性普通职工及初级干部')
ax.hist(outfile_male2['aae240'],bins=20000,width=18, facecolor="black", 
        alpha=1, label='男性高级干部')
ax.hist(outfile_female1['aae240'],bins=20000,width=18, facecolor="orange",
         alpha=0.6, label='女性普通职工及初级干部')
ax.hist(outfile_female2['aae240'],bins=10000,width=18, facecolor="red",
         alpha=0.6, label='女性高级干部')

leg = ax.legend()
ax.legend(loc='upper left', frameon=False)
ax.legend(frameon=False, loc='lower center', ncol=2,fontsize=32)
ax.legend(fancybox=True, framealpha=0.9, shadow=True, borderpad=2)
ax.text(0.3, 0.7, "男性离退休普通职工及初级干部平均个人账户养老金总额：31954.95元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.3, 0.64, "男性离退休高级干部平均个人账户养老金总额：35377.52元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.3, 0.56, "女性离退休普通职工及初级干部平均个人账户养老金总额：30491.28元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.3, 0.50, "女性离退休高级干部平均个人账户养老金总额：33049.91元", 
        transform=ax.transAxes,fontdict=None)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlim(0,100000)
plt.ylim(0,1000)
plt.xlabel('平均个人账户养老金总额（元）',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('机关单位离退休人员平均个人账户养老金总额总体分布',fontsize=40)
plt.show()


'不同退休年龄'
ac015029 = pd.merge(ac0150,ac29,on='aac001')
#ac015029 = pd.merge(ac015029,ic10,on='aac001') #选出养老离退休人员，分析账户金额 279782条
outfile_male1 = ac015029[ac015029[['aac222']].isnull().T.any()]
outfile_male1 = outfile_male1.query("aac004==1") #151575
outfile_male2 = ac015029[ac015029['aac222'].notna()]
outfile_male2 = outfile_male2.query("aac004==1 & aac222==1")
outfile_female1 = ac015029[ac015029[['aac222']].isnull().T.any()]
outfile_female1 = outfile_female1.query("aac004==2") #109148
outfile_female2 = ac015029[ac015029['aac222'].notna()]
outfile_female2 = outfile_female2.query("aac004==2 & aac222==1") #5075

outfile_male1['male_RA'] = ((outfile_male1['aic162']-outfile_male1['aac006'])/10000).round(0).astype(int)
outfile_male2['male_RA'] = ((outfile_male2['aic162']-outfile_male2['aac006'])/10000).round(0).astype(int)
outfile_female1['female_RA'] = ((outfile_female1['aic162']-outfile_female1['aac006'])/10000).round(0).astype(int)
outfile_female2['female_RA'] = ((outfile_female2['aic162']-outfile_female2['aac006'])/10000).round(0).astype(int)

#result = outfile_female2.groupby('female_RA')['aac040'].mean() #修正离群值
#result.loc[57] = (result.loc[56] + result.loc[58])/2

rcParams['axes.titlepad'] = 30 
plt.rcParams.update({'font.size':24})

fig,ax = plt.subplots(figsize=(20, 10), dpi=150)
ax.plot(outfile_male1.groupby('male_RA')['aae240'].mean(),'-ok', color='orange',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='男性普通职工及初级干部')
ax.plot(outfile_male2.groupby('male_RA')['aae240'].mean(),'-ok', color='black',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='男性高级干部')
ax.plot(outfile_female1.groupby('female_RA')['aae240'].mean(),'-ok', color='steelblue',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='女性普通职工及初级干部')
ax.plot(outfile_female2.groupby('female_RA')['aae240'].mean(),'-ok', color='red',
         markersize=8, linewidth=6,
         markerfacecolor='black',
         markeredgewidth=2,label='女性高级干部')
#设置标签列数
ax.legend(frameon=False, ncol=2)
#fancybox:圆角边框  framealpha:外边框透明度 shadow：增加阴影 borderpad：
ax.legend(fancybox=True, framealpha=0.8 ,loc='lower left',shadow=True, borderpad=2)
leg = ax.legend()
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlim(47,63)
#plt.ylim(2000,10000)
plt.xlabel('退休年龄（岁）',fontsize=30)
plt.ylabel('退休时平均个人账户养老金总额（元）',fontsize=30)
plt.title('不同退休年龄的平均个人账户养老金总额',fontsize=40)
plt.show()




'''退休时上年度社会平均工资'''
merge = pd.merge(ac01,ic10,on='aac001')
outfile_male = merge.query("aac004==1 & aac026>0") 
outfile_female = merge.query("aac004==2 & aac026>0") 

outfile_male['male_RA'] = ((outfile_male['aic162']-outfile_male['aac006'])/10000).round(0).astype(int)
outfile_female['female_RA'] = ((outfile_female1['aic162']-outfile_female1['aac006'])/10000).round(0).astype(int)

outfile_male.groupby('male_RA')['aac026'].mean()
outfile_female.groupby('female_RA')['aac026'].mean()


'''工资增长率'''
merge = pd.merge(ac01,ic10,on='aac001')
outfile_male = merge.query("aac004==1 & aac026>0") 
outfile_female = merge.query("aac004==2 & aac026>0") 

outfile_male['male_RA'] = ((outfile_male['aic162']-outfile_male['aac006'])/10000).round(0).astype(int)
outfile_female['female_RA'] = ((outfile_female1['aic162']-outfile_female1['aac006'])/10000).round(0).astype(int)

round(outfile_male.groupby('male_RA')['bac706'].mean()*100,2)
outfile_female.groupby('female_RA')['bac706'].mean()



'''职业年金税前月发放金额'''
merge = pd.merge(ac01,ac29,on='aac001')
merge = pd.merge(merge,id19,on='aac001')
#ac0150 = pd.merge(ac0150,ic10,on='aac001') #选出养老离退休人员，分析账户金额 279782条
outfile_male1 = merge[merge[['aac222']].isnull().T.any()]
outfile_male1 = outfile_male1.query("aac004==1 & aae019 >0 ")
outfile_male2 = merge[merge['aac222'].notna()]
outfile_male2 = outfile_male2.query("aac004==1 & aac222==1 & aae019 >0")
outfile_female1 = merge[merge[['aac222']].isnull().T.any()]
outfile_female1 = outfile_female1.query("aac004==2 & aae019 >0")
outfile_female2 = merge[merge['aac222'].notna()]
outfile_female2 = outfile_female2.query("aac004==2 & aac222==1 & aae019 >0")

print(outfile_male1['aae019'].mean(),outfile_male2['aae019'].mean(),
outfile_female1['aae019'].mean(),outfile_female2['aae019'].mean())

rcParams['axes.titlepad'] = 30 
plt.rcParams.update({'font.size':24}) #字体大小

fig,ax = plt.subplots(figsize=(20, 15))
#ax.set(xlim=[45,65])
ax.hist(outfile_male1['aae019'],bins=4000,width=6, facecolor="steelblue", 
        alpha=0.6, label='男性普通职工及初级干部')
ax.hist(outfile_male2['aae019'],bins=4000,width=6, facecolor="black", 
        alpha=0.6, label='男性高级干部')
ax.hist(outfile_female1['aae019'],bins=4000,width=6, facecolor="orange",
         alpha=0.6, label='女性普通职工及初级干部')
ax.hist(outfile_female2['aae019'],bins=2000,width=6, facecolor="red",
         alpha=0.6, label='女性高级干部')

leg = ax.legend()
ax.legend(loc='upper left', frameon=False)
ax.legend(frameon=False, loc='lower center', ncol=2,fontsize=32)
ax.legend(fancybox=True, framealpha=0.9, shadow=True, borderpad=2)
ax.text(0.3, 0.6, "男性离退休普通职工及初级干部平均职业年金税前月发放金额：187.13元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.3, 0.54, "男性离退休高级干部平均职业年金税前月发放金额：211.88元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.3, 0.48, "女性离退休普通职工及初级干部平均职业年金税前月发放金额：162.37元", 
        transform=ax.transAxes,fontdict=None)
ax.text(0.3, 0.42, "女性离退休高级干部平均职业年金税前月发放金额：189.92元", 
        transform=ax.transAxes,fontdict=None)
plt.xticks(size=30)
plt.yticks(size=30)
plt.xlim(0,1200)
plt.ylim(0,420)
plt.xlabel('平均职业年金税前月发放金额（元）',fontsize=30)
plt.ylabel('人数',fontsize=30)
plt.title('机关单位离退休人员平均职业年金税前月发放金额总体分布',fontsize=40)
plt.show()

outfile_male1['male_RA'] = ((outfile_male1['aic162']-outfile_male1['aac006'])/10000).round(0).astype(int)
outfile_male2['male_RA'] = ((outfile_male2['aic162']-outfile_male2['aac006'])/10000).round(0).astype(int)
outfile_female1['female_RA'] = ((outfile_female1['aic162']-outfile_female1['aac006'])/10000).round(0).astype(int)
outfile_female2['female_RA'] = ((outfile_female2['aic162']-outfile_female2['aac006'])/10000).round(0).astype(int)

print(outfile_male1.groupby('male_RA')['aae019'].mean(),
      outfile_male2.groupby('male_RA')['aae019'].mean(),
      outfile_female1.groupby('female_RA')['aae019'].mean(),
      outfile_female2.groupby('female_RA')['aae019'].mean())




'''老办法养老金月标准'''
merge = pd.merge(ac01,ac29,on='aac001')
outfile_male1 = merge[merge[['aac222']].isnull().T.any()]
outfile_male1 = pd.merge(outfile_male1, ic10, on='aac001')
outfile_male1 = outfile_male1.query("aac004==1")
outfile_male2 = merge[merge['aac222'].notna()]
outfile_male2 = outfile_male2.query("aac004==1 & aac222==1")
outfile_male2 = pd.merge(outfile_male2, ic10, on='aac001')
outfile_female1 = merge[merge[['aac222']].isnull().T.any()]
outfile_female1 = pd.merge(outfile_female1, ic10, on='aac001')
outfile_female1 = outfile_female1.query("aac004==2")
outfile_female2 = merge[merge['aac222'].notna()]
outfile_female2 = outfile_female2.query("aac004==2 & aac222==1")
outfile_female2 = pd.merge(outfile_female2, ic10, on='aac001')


outfile_male1['male_RA'] = ((outfile_male1['aic162']-outfile_male1['aac006'])/10000).round(0).astype(int)
outfile_male2['male_RA'] = ((outfile_male2['aic162']-outfile_male2['aac006'])/10000).round(0).astype(int)
outfile_female1['female_RA'] = ((outfile_female1['aic162']-outfile_female1['aac006'])/10000).round(0).astype(int)
outfile_female2['female_RA'] = ((outfile_female2['aic162']-outfile_female2['aac006'])/10000).round(0).astype(int)

print(outfile_male1.groupby('male_RA')['aic246'].mean(),
      outfile_male2.groupby('male_RA')['aic246'].mean(),
      outfile_female1.groupby('female_RA')['aic246'].mean(),
      outfile_female2.groupby('female_RA')['aic246'].mean())

