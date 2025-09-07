import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from Net import Net
import common
from CaptchaData import CaptchaData

def calculat_acc(output, target):
    output = output.view(-1, len(common.captcha_array))
    target = target.view(-1, len(common.captcha_array))
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    target = torch.argmax(target, dim=1)
    c = (output == target).sum().item()
    return c / output.size(0) * 100

def train(epochs):
    # 初始化
    net = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_data = CaptchaData('datasets/train/', transform=transform)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    test_data = CaptchaData('datasets/test/', transform=transform)
    test_loader = DataLoader(test_data, batch_size=128)
    
    # 优化器和损失函数
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(epochs):
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                acc = calculat_acc(outputs, labels)
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
        
        # 测试
        net.eval()
        with torch.no_grad():
            total_acc = 0
            for images, labels in test_loader:
                outputs = net(images.to(device))
                total_acc += calculat_acc(outputs, labels.to(device))
            print(f'Test Acc: {total_acc/len(test_loader):.2f}%')
        
        # 保存模型
        torch.save(net.state_dict(), f'model_epoch{epoch+1}.pth')

if __name__ == '__main__':
    train(epochs=50)
