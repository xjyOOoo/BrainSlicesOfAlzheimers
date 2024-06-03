import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import load_dataset, split_dataset, create_data_loaders
from model import ResNetModel
from train import train_model, evaluate_model
from utils import setup_logging, save_test_results, save_model
import logging
import torch.nn as nn
import os
os.chdir(os.path.dirname(__file__))
if __name__ == "__main__":
    try:
        setup_logging()

        data_path = 'data'
        dataset = load_dataset(data_path)
        train_dataset, val_dataset, test_dataset = split_dataset(dataset)
        train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)

        train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device)
        accuracy, conf_matrix, precision, recall = evaluate_model(model, test_loader, device)
        save_test_results(conf_matrix, precision, recall, accuracy)
        save_model(model, 'final_model.pth')

    except Exception as e:
        logging.error(f"An error occurred: {e}")


