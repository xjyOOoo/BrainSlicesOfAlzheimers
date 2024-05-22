import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import os
import pandas as pd
import torch.nn as nn


os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图片大小
    transforms.ToTensor(),  # 转换成Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 定义模型
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

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('simple_cnn.pth', map_location=device))
model.eval()

# 标签映射
class_mapping = {
    0: 'non',
    1: 'verymild',
    2: 'mild',
    3: 'moderate'
}

# 预测函数
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return class_mapping[predicted.item()]

# 读取并预测新图片
new_data_path = 'data/mild'
results = []

for img_name in os.listdir(new_data_path):
    img_path = os.path.join(new_data_path, img_name)
    if os.path.isfile(img_path):
        prediction = predict_image(img_path)
        results.append((img_name, prediction))

# 保存预测结果到CSV
df_results = pd.DataFrame(results, columns=['Image', 'Prediction'])
df_results.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
