# To add a new cell, type '# #'
# To add a new markdown cell, type '# # [markdown]'
# # [markdown]
# # 3.6 softmax回归的从零开始实现

# #
import torch
import torchvision
import numpy as np
import sys
sys.path.append("D:/Deeplearning/Dive-into-DL-PyTorch-master/code") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

print(torch.__version__)
print(torchvision.__version__)

# # [markdown]
# ## 3.6.1 获取和读取数据

# #
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# # [markdown]
# ## 3.6.2 初始化模型参数

# #
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)


# #
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True) 


# #
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))
print(X.sum(dim=1, keepdim=True))

# # [markdown]
# ## 3.6.3 实现softmax运算

# #
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# #
X = torch.rand((2, 5))
x1 = torch.tensor([[100, 101, 102],[-100, -101, -102],[1000, 1010, 1020], [10.0, 10.1, 10.2], [-2, -1, 0]], dtype=torch.float64) 
X_prob = softmax(x1)
print(X_prob, X_prob.sum(dim=1))

# # [markdown]
# ## 3.6.4 定义模型

# #
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# # [markdown]
# ## 3.6.5 定义损失函数

# #
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))


# #
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# # [markdown]
# ## 3.6.6 计算分类准确率

# #
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# #
print(accuracy(y_hat, y))


# #
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# #
print(evaluate_accuracy(test_iter, net))

# # [markdown]
# ## 3.6.7 训练模型

# #
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

# # [markdown]
# ## 3.6.8 预测

# #
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])

