import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
try:
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图片大小
        transforms.ToTensor(),  # 转换成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 标签映射
    class_mapping = {
        'non': 0,
        'verymild': 1,
        'mild': 2,
        'moderate': 3
    }

    # 加载数据集
    data_path = 'data'
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataset.samples = [(p, class_mapping[os.path.split(os.path.dirname(p))[-1]]) for p, _ in dataset.samples]

    # 分割数据集 1:9
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义神经网络模型
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(64 * 64 * 64, 128)  # 隐藏层
            self.fc2 = nn.Linear(128, 4)  # 输出层，4个类别

        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 64 * 64)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # 训练模型
    def train_model():
        model.train()
        for epoch in range(10):  # 进行10个周期的训练
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)  # 将数据转移到GPU上
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    train_model()

    # 测试模型并保存结果
    model.eval()
    correct=0
    total=0
    test_results = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据转移到GPU上
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_results.extend([(p.item(), l.item()) for p, l in zip(predicted, labels)])

    accuracy = correct / total
    print('Test Accuracy: {:.2f}%'.format(100 * accuracy))

    # 保存测试结果到CSV
    df_test_results = pd.DataFrame(test_results, columns=['Predicted', 'Actual'])
    df_test_results.to_csv('test_results.csv', index=False)

    # 保存模型
    torch.save(model.state_dict(), 'simple_cnn.pth')

except Exception as e:
    print(f"An error occurred: {e}")
