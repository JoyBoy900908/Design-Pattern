from model_training.model_trainer import ModelTrainer
from model_training.evaluaters import KFoldValidation
from model_training.loggers import TrainingLogger
from model_training.data_initializer import DataInitializer
import torch.optim as optim
import torch.nn as nn
import torch
import logging
from datetime import datetime

from GNN_models.FRGCN_model import FRGCNModel
from GNN_models.Binary_FRGCN_model import BinaryFRGCNModel

K_FOLD = 4
SAVE_PATH = 'trained_models/'

stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s'))

file_handler = logging.FileHandler(f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        stdout_handler,
        file_handler
    ]
)

# 初始化資料，將所有模式都用 java_project_dependency_graph 的方式讀進來
data_initializer = DataInitializer()


'''
你可以在這邊設定要訓練的模型、訓練參數、要儲存的路徑、要使用的資料集

{
    'model': 模型的 class,
    'training_parameters': {
        'learning_rate': 學習率,
        'criterion': 損失函數,
        'device': 設備,
        'batch_size': 批次大小,
        'epoches': 訓練的 epoch 數,
    },
    'save_path': 模型儲存的路徑,
    'data_type': 資料的形式，如果是要給 binary model 的話那要輸入 'strategy'、'singleton'、'adapter'、'observer'、'factory'、'decorator' 之一，如果是要給 all in one model 的話就輸入 'all in one',
}
'''

'''
'model': BinaryFRGCNModel, 
        'training_parameters': {
            'learning_rate': 0.0025,
            'criterion': nn.CrossEntropyLoss(),
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'batch_size': 64,
            'epoches': 80,
        },
        'save_path': SAVE_PATH + '20240319-1',
        'data_type': 'bridge',
        'k_fold': 4
'''

'''
'model': FRGCNModel, 
        'training_parameters': {
            'learning_rate': 0.0025,
            'criterion': nn.CrossEntropyLoss(),
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'batch_size': 32,
            'epoches': 50,
        },
        'save_path': SAVE_PATH + '20240504',
        'data_type': 'all in one',
        'k_fold': 4
'''
        
schedules = [
    {
        'model': FRGCNModel, 
        'training_parameters': {
            'learning_rate': 0.0025,
            'criterion': nn.CrossEntropyLoss(),
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'batch_size': 32,
            'epoches': 100,
        },
        'save_path': SAVE_PATH + '20240504-1',
        'data_type': 'all in one',
        'k_fold': 4
    }
]

if __name__ == '__main__':
    trainer = ModelTrainer()
    logger = TrainingLogger()
    trainer.register_observer(logger)

    for schedule in schedules:
        data = data_initializer.get_binary_data(schedule['data_type']) if schedule['data_type'] != 'all in one' else data_initializer.get_data()
        k_fold = KFoldValidation(schedule['k_fold'], schedule['model'], data)
        for i in range(schedule['k_fold']):
            ith_fold = k_fold.get_ith_fold(i)
            model = ith_fold.model
            train_data = ith_fold.train_data
            test_data = ith_fold.test_data
            trainer.set_training_parameters(model, train_data, test_data,
                                            optimizer=optim.Adam(model.parameters(), lr=schedule['training_parameters']['learning_rate']),
                                            criterion=schedule['training_parameters']['criterion'],
                                            device=schedule['training_parameters']['device'],
                                            batch_size=schedule['training_parameters']['batch_size'],
                                            epoches=schedule['training_parameters']['epoches'],
                                            save_path=schedule['save_path']
                                            )
            trainer.register_observer(ith_fold)
            trainer.start(f'k_fold_{i}.pt')
            trainer.unregister_observer(ith_fold)
    

