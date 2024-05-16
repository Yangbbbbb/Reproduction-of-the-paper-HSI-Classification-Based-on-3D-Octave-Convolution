import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from model import HSINet
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration = 100
batch_size = 16
learn_rate = 1e-4
weight_decay = 0.0005  # Your weight decay value
l2_reg = torch.tensor(0.0).to(device)

model = HSINet().to(device)

# 加载数据集
data_train = np.load("./data/data_train.npy")
data_test = np.load("./data/data_test.npy")
labels_train = np.load("./data/labels_train.npy")
labels_test = np.load("./data/labels_test.npy")

# 将 numpy 数组转换为 PyTorch 张量
tensor_x_train = torch.Tensor(data_train)
tensor_y_train = torch.Tensor(labels_train)
tensor_x_test = torch.Tensor(data_test)
tensor_y_test = torch.Tensor(labels_test)

# 创建数据集
train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0005)

fig_loss=[]
fig_accuracy=[]
fig_loss_test=[]
fig_accuracy_test=[]
# 训练模型
num_epochs = iteration
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU上
        optimizer.zero_grad()
        logit_spatial, logit_spectral, logit, predict = model(inputs)
        # print(predict.shape())
        loss1 = criterion(logit_spatial, labels)
        loss2 = criterion(logit_spectral, labels)
        loss3 = criterion(logit, labels)
        loss = loss1 + loss2 + loss3

        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(predict, 1)
        _, true_label = torch.max(labels, 1)
        # print(predicted.size(), labels.size())  # 打印预测值和标签的形状
        total += labels.size(0)
        correct += (predicted == true_label).sum().item()  # 计算预测正确的样本数
    #

    # 计算并打印每个epoch的平均损失
    epoch_loss = running_loss / len(train_dataset)
    accuracy = 100 * correct / total
    fig_loss.append(epoch_loss)
    fig_accuracy.append(accuracy)
    # print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {accuracy:.2f}%")

    # 计算测试集的精度
    test_correct = 0
    test_total = 0
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 不进行梯度计算
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_logit_spatial, test_logit_spectral, test_logit, test_predict = model(test_inputs)
            _, test_predicted = torch.max(test_predict, 1)
            _, test_true_label = torch.max(test_labels, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_true_label).sum().item()

    test_accuracy = 100 * test_correct / test_total
    fig_accuracy_test.append(test_accuracy)

    # 输出loss和ACC
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {accuracy:.2f}%  Test Accuracy: {test_accuracy:.2f}%")
    # print(f"Test Accuracy: {test_accuracy:.2f}%")
    if epoch>0 and (epoch+1)%10==0:
        torch.save(model.state_dict(), f'./trainedModel/model_epoch_{epoch+1}.pt')
    # torch.save(model.state_dict(), f'./trainedModel/model_epoch_{epoch + 1}.pt')



fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
lns1 = ax1.plot(np.arange(iteration), fig_loss, label="Loss")

# 按一定间隔显示实现方法
# ax2.plot(200 * np.arange(len(fig_accuracy)), fig_accuracy, 'r')
lns2 = ax2.plot(np.arange(iteration), fig_accuracy, 'r', label="Accuracy")
ax1.set_xlabel('iteration')
ax1.set_ylabel('training loss')
ax2.set_ylabel('training accuracy')
# 合并图例
lns = lns1 + lns2
labels = ["Loss", "Accuracy"]
# labels = [l.get_label() for l in lns]
plt.legend(lns, labels, loc=7)
plt.show()
#
fig2, ax3 = plt.subplots()
lns3 = ax3.plot(np.arange(iteration), fig_accuracy_test, label="test_acc")
ax3.set_xlabel('iteration')
ax3.set_ylabel('test_acc')
lns4 = lns3
labels_5 = ["acc"]
# labels = [l.get_label() for l in lns]
plt.legend(lns4, labels_5, loc=7)
plt.show()
