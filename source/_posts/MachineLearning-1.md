---
title: 机器学习1
date: 2023-08-12 21:50:40
tags: 
- 机器学习
- Python
categories:
- 机器学习
description: 记录有关在pytorch学习中线性模型的知识
---

# 线性模型

---

## 机器学习(machine learning)几个步骤 
1. 准备数据集(dataset)
2. 选择模型(model)
3. 训练(training)
4. 应用(inferring)

数据集分为两部分：训练集、测试集。

训练集又可以细分为训练集、开发集

损失函数是针对一个样本的,平均平方误差(Mean Square Error,mse)是针对于整个训练集。

训练神经网络本质上就是使用几个或者一些参数将一个模型变换为更加复杂的模型

损失函数的选择很重要，因为它是一种对训练样本中要修正的错误进行优先处理的方法，可以强调或者忽略某些误差

## 过拟合

用训练集去训练模型，在尽可能的使损失最小后，将模型在测试集验证时发现，模型产生的损失比预期的要高得多，即过拟合

<div align=center>
<img src="MachineLearning-1-3.jpg" height = '360'>
引自《Deep Learning with PyTorch》
</div>

解决过拟合的方法
1. 在损失函数中添加惩罚项，以降低模型的成本，使其表现得更加平稳、变换更缓慢
2. 在输入样本中添加噪声，人为地在训练数据样本之间创建新的数据点，并使模型也拟合这些点
3. ···

那么现在可以将训练神经网络（选择合适参数）的过程分为两步：增大参数直到拟合；缩小参数以避免出现过拟合

## 练习

### Question

Suppose that students would get y points in final exam, if they spent x hours in study

| x   | y   |
| --- | --- |
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |
| 4   | ?   |

### Answer

```python

import numpy as np
import matplotlib.pyplot as plt
	
# 数据集
x_data = {1.0, 2.0, 3.0}
y_data = {2.0, 4.0, 6.0}

# 定义模型
def forward(x):
    return x * w

# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

#权重及其对应损失值
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print('w=',w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()
```

<div align=center>
<img src="MachineLearning-1-1.png" height = '360' title="y=x*w" alt="y=x*w">
y=x*w
</div>

---

### Question
Suppose that students would get y points in final exam, if they spent x hours in study.Try to use the model y=x*w+b, and draw the cost graph.

| x   | y   |
| --- | --- |
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |
| 4   | ?   |

### Answer

```python
import numpy as np
import matplotlib.pyplot as plt

# 数据集
x_data = {1.0, 2.0, 3.0}
y_data = {2.0, 4.0, 6.0}


# 定义模型
def forward(x):
    return x * w + b


# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 权重及其对应损失值
w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.0, 0.1):
        print('w=',  w, 'b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum/3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/3)

print(mse_list)
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  # 创建三维坐标系
ax3d.plot_trisurf(w_list, b_list, mse_list)
plt.show()
```

<div align=center>
<img src="MachineLearning-1-2.png" height = '360' title="y=x*w+b" alt="y=x*w+b">
y=x*w+b
</div>

---

matplotlib中的函数还不怎么会用，后面抽个时间看一看

> https://blog.csdn.net/hustlei/article/details/122408179

这个博客里面思维导图可以看一看捏
