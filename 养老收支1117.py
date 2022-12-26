# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller as ADF #平稳性检验
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import chi2_contingency
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

#平均工资
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


#人数
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

#平均养老金支出
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
