# Nirs-Pls-Corn
This code is a near-infrared spectrum modeling method based on PCA and pls

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">
<font size =4 color=bule >近红外光谱分析技术属于交叉领域，需要化学、计算机科学、生物科学等多领域的合作。为此，在（北邮邮电大学杨辉华老师团队）指导下，近期准备开源传统的PLS，SVM，ANN，RF等经典算和SG，MSC，一阶导，二阶导等预处理以及GA等波长选择算法以及CNN、AE等最新深度学习算法，以帮助其他专业的更容易建立具有良好预测能力和鲁棒性的近红外光谱模型。代码仅供学术使用，如有问题，联系方式：QQ：1427950662，微信：Fu_siry
<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

## 1.读取数据并显示光谱曲线

```python
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
```
![显示的光谱曲线](https://img-blog.csdnimg.cn/2d09358a92d4476fb073c3705d57561d.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)
## 2.划分训练集和测试集
```python
#随机划分数据集
x_data = np.array(data)
y_data = np.array(label[:,2])

test_ratio = 0.2
X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=test_ratio,shuffle=True,random_state=2)

```
## 3.PCA降维并显示

```python
#载入数据
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
```
<font size=4 color=#99AA >PCA降维后的数据分布：
![PCA降维后的数据分布](https://img-blog.csdnimg.cn/a1c70d07f29b4aa2b949eb32fc649e82.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)


## 4.建立校正模型（数据拟合）

```python
#pls预测
pls2 = PLSRegression(n_components=3)
pls2.fit(X_train_reduction, y_train)

train_pred = pls2.predict(X_train_reduction)
pred = pls2.predict(X_test_reduction)
```
## 5.模型评估（使用R2、RMSE、MSE指标）

```python
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
```
<font size=4 color=#99AA >模型评估结果：
![模型评估结果](https://img-blog.csdnimg.cn/a4c61a9972ba4381bca3dc487800d1ca.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_20,color_FFFFFF,t_70,g_se,x_16)
## 6.绘制拟合差异曲线图

```python
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
```
<font size=4 color=#99AA >结果如图：
![拟合差异曲线](https://img-blog.csdnimg.cn/278eaaa484e142e68f0d76e9fb401a50.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBARWNob19Db2Rl,size_19,color_FFFFFF,t_70,g_se,x_16)
