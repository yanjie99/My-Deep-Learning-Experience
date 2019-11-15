# 实例：房价预测
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

plt.ion()
plt.figure(figsize=(8,6))
plt.rcParams['font.family'] = ['SimHei']
# 制造假数据
x = Variable(torch.linspace(0,100).type(torch.FloatTensor))
rand = Variable(torch.randn(100))*10    # 均值0，方差10
y = x + rand
# 分成训练数据和测试数据
x_train = x[:-10]
y_train = y[:-10]
x_test = x[-10:]
y_test = y[-10:]
# 对训练数据进行可视化
# plt.figure(figsize=(8,6))
# plt.plot(x_train.data.numpy(),y_train.data.numpy(),'o')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# y = ax + b
a = Variable(torch.rand(1),requires_grad = True)
b = Variable(torch.rand(1),requires_grad = True)
LR = 0.0001
for i in range(1000):
    predictions = a.expand_as(x_train)*x_train + b.expand_as(x_train)
    loss = torch.mean((y_train - predictions)**2)
    # print('loss:',loss)
    loss.backward()
    a.data.add_(-LR*a.grad.data)
    b.data.add_(-LR*b.grad.data)
    a.grad.data.zero_()
    b.grad.data.zero_()
    if i % 20 == 0:
        plt.cla()
        x_data = x_train.data.numpy()
        xplot, = plt.plot(x_data,y_train.data.numpy(),'o')
        yplot, = plt.plot(x_data,a.data.numpy()*x_data+b.data.numpy(),lw=5,c='green',)
        plt.xlabel('X')
        plt.ylabel('Y')
        str1 = str(a.data.numpy()[0])+'x+'+str(b.data.numpy()[0])
        plt.text(65,0,r'$Loss=%.3f$'%loss.data.numpy(),fontdict={'size':'12'})
        plt.legend([xplot,yplot],['Data',str1])
        plt.pause(0.001)
plt.ioff()
plt.show()

# 测试数据
pred = a.expand_as(x_test)*x_test+b.expand_as(x_test)
print(pred)
print('------------')
print(y_test)

x_data = x_train.data.numpy()
x_pred = x_test.data.numpy()
plt.figure(num=2,figsize=(10,7))
plt.plot(x_data,y_train.data.numpy(),'o')
plt.plot(x_pred,y_test.data.numpy(),'s',c='r')
x_data = np.r_[x_data,x_test.data.numpy()]
plt.plot(x_data,a.data.numpy()*x_data+b.data.numpy())
plt.plot(x_pred,a.data.numpy()*x_pred+b.data.numpy(),'o')
plt.xlabel('X')
plt.ylabel('Y')
str2 = str(a.data.numpy()[0]) + 'x+'+str(b.data.numpy()[0])
plt.legend([xplot,yplot],['Data',str2])
plt.show()
