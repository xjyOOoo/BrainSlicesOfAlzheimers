import torch
import logging
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from utils import save_model


# 训练模型
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=10,
                save_best_model=True):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        logging.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        scheduler.step(val_loss)  # 更新学习率

        if save_best_model and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, 'best_model.pth')
            logging.info(f'Saved best model with Val Loss: {best_val_loss:.4f}')


# 评估模型损失
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
    return loss / len(data_loader)


# 评估模型
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    logging.info('Test Accuracy: {:.2f}%'.format(100 * accuracy))
    logging.info('Confusion Matrix:')
    logging.info(conf_matrix)
    logging.info('Precision: {:.2f}'.format(precision))
    logging.info('Recall: {:.2f}'.format(recall))

    return accuracy, conf_matrix, precision, recall


