from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#载入数据
data_path = './/data//m5.csv' #数据
label_path = './/data//label.csv' #标签（反射率）

data = np.loadtxt(open(data_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)
label = np.loadtxt(open(label_path, 'rb'), dtype=np.float64, delimiter=',', skiprows=0)

# 绘制原始后图片
plt.figure(500)
x_col = np.linspace(0,len(data[0,:]),len(data[0,:]))  #数组逆序
y_col = np.transpose(data)
plt.plot(x_col, y_col)
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The spectrum of the corn dataset",fontweight= "semibold",fontsize='x-large')
plt.savefig('.//Result//MSC.png')
plt.show()

#随机划分数据集
x_data = np.array(data)
y_data = np.array(label[:,2])

test_ratio = 0.2
X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=test_ratio,shuffle=True,random_state=2)

#PCA降维到10个维度,测试该数据最好
pca=PCA(n_components=10)  #只保留2个特征
pca.fit(X_train)
X_train_reduction = pca.transform(X_train)
X_test_reduction = pca.transform(X_test)

# PCA降维后图片绘制
plt.figure(100)
plt.scatter(X_train_reduction[:, 0], X_train_reduction[:, 1],marker='o')
plt.xlabel("Wavenumber(nm)")
plt.ylabel("Absorbance")
plt.title("The  PCA for corn dataset",fontweight= "semibold",fontsize='large')
plt.savefig('.//Result//PCA.png')
plt.show()

#pls预测
pls2 = PLSRegression(n_components=3)
pls2.fit(X_train_reduction, y_train)

train_pred = pls2.predict(X_train_reduction)
pred = pls2.predict(X_test_reduction)

#计算R2
train_R2 = r2_score(train_pred,y_train)
R2 = r2_score(y_test,pred) #Y_true, Pred
print('训练R2:{}'.format(train_R2))
print('测试R2:{}'.format(R2))
#计算MSE
print('********************')
x_MSE = mean_squared_error(train_pred,y_train)
t_MSE = mean_squared_error(y_test,pred)
print('训练MSE:{}'.format(x_MSE))
print('测试MSE:{}'.format(t_MSE))

#计算RMSE
print('********************')
print('测试RMSE:{}'.format(sqrt(x_MSE)))
print('训练RMSE:{}'.format(sqrt(t_MSE)))

#绘制拟合图片
plt.figure(figsize=(6,4))
x_col = np.linspace(0,16,16)  #数组逆序
# y = [0,10,20,30,40,50,60,70,80]
# x_col = X_test
y_test = np.transpose(y_test)
ax = plt.gca()
ax.set_xlim(0,16)
ax.set_ylim(6,11)
# plt.yticks(y)
plt.scatter(x_col, y_test,label='Ture', color='blue')
plt.plot(x_col, pred,label='predict', marker='D',color='red')
plt.legend(loc='best')
plt.xlabel("测试集的样本")
plt.ylabel("样本的值")
plt.title("The Result of corn dataset",fontweight= "semibold",fontsize='large')
plt.savefig('.//Result//Reslut.png')
plt.show()

















