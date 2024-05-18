import numpy as np
import torch
from model import HSINet
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration = 100
batch_size = 8
learn_rate = 1e-4
weight_decay = 0.0005  # Your weight decay value
l2_reg = torch.tensor(0.0).to(device)

model = HSINet().to(device)
# 加载保存的模型状态字典
model.load_state_dict(torch.load(f'./trainedModel/model_epoch_{100}.pt'))

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
    test_correct = 0
    test_total = 0
    class_correct = dict()
    class_total = dict()
    model.eval()  # 设置模型为评估模式

    with torch.no_grad():  # 不进行梯度计算
        for test_inputs, test_labels in test_loader:
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_logit_spatial, test_logit_spectral, test_logit, test_predict = model(test_inputs)
            _, test_predicted = torch.max(test_predict, 1)
            _, test_true_label = torch.max(test_labels, 1)
            test_total += test_labels.size(0)
            test_correct += (test_predicted == test_true_label).sum().item()

            for label, prediction in zip(test_true_label, test_predicted):
                if label.item() not in class_correct:
                    class_correct[label.item()] = 0
                    class_total[label.item()] = 0
                if label == prediction:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1

    test_accuracy = 100 * test_correct / test_total
    class_accuracies = {cls: 100 * class_correct[cls] / class_total[cls]
                        for cls in class_correct}
    average_accuracy = np.mean(list(class_accuracies.values()))
    fig_accuracy_test.append(test_accuracy)
    fig_accuracy.append(average_accuracy)

    # 输出ACC
    print(f"Test Overall Accuracy: {test_accuracy:.2f}%")
    print(f"Test Average Accuracy: {average_accuracy:.2f}%")
