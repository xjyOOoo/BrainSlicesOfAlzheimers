import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet模型的输入尺寸
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

def load_dataset(data_path):
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataset.samples = [(p, class_mapping[os.path.split(os.path.dirname(p))[-1]]) for p, _ in dataset.samples]
    return dataset

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
