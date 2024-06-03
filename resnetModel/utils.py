import logging
import torch
import pandas as pd

# 设置日志
def setup_logging(log_file='training.log'):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

# 保存测试结果
def save_test_results(conf_matrix, precision, recall, accuracy, filename='test_results.csv'):
    df_conf_matrix = pd.DataFrame(conf_matrix)
    with open(filename, 'w') as f:
        f.write('Confusion Matrix:\n')
        df_conf_matrix.to_csv(f, index=False)
        f.write(f'\nPrecision: {precision:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'Accuracy: {accuracy:.2f}\n')

# 保存模型
def save_model(model, filename='model.pth'):
    torch.save(model.state_dict(), filename)


