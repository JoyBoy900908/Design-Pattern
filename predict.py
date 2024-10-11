import torch
from GNN_models.FRGCN_model import FRGCNModel
from java_project_dependency_graph import JavaProjectDependencyGraph
from binary_pattern_model import BinaryPatternRecognitionModel
from torch.nn.functional import softmax
import os
import random
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Predicter:

    '''
        Predicter is a encapsulated class that hide all the machine learning stuff from user.
    '''

    def __init__(self):
        # self.strategy_model = BinaryPatternRecognitionModel(INPUT_DIM, HIDDEN_DIM)
        # self.singleton_model = BinaryPatternRecognitionModel(INPUT_DIM, HIDDEN_DIM)
        # self.adapter_model = BinaryPatternRecognitionModel(INPUT_DIM, HIDDEN_DIM)
        # self.observer_model = BinaryPatternRecognitionModel(INPUT_DIM, HIDDEN_DIM)
        # self.factory_model = BinaryPatternRecognitionModel(INPUT_DIM, HIDDEN_DIM)
        # self.decorator_model = BinaryPatternRecognitionModel(INPUT_DIM, HIDDEN_DIM)
        self.multi_model = FRGCNModel()

        # self.strategy_model.load_state_dict(torch.load('0/0-fold-model.pt'))
        # self.singleton_model.load_state_dict(torch.load('1/0-fold-model.pt'))
        # self.adapter_model.load_state_dict(torch.load('2/3-fold-model.pt'))
        # self.observer_model.load_state_dict(torch.load('3/0-fold-model.pt'))
        # self.factory_model.load_state_dict(torch.load('4/0-fold-model.pt'))
        # self.decorator_model.load_state_dict(torch.load('5/0-fold-model.pt'))
        # self.multi_model.load_state_dict(torch.load('0-fold-model.pt'))

        # self.strategy_model.eval()
        # self.singleton_model.eval()
        # self.adapter_model.eval()
        # self.observer_model.eval()
        # self.factory_model.eval()
        # self.decorator_model.eval()
        self.multi_model.eval()

    def multi_model_predict(self, data):
        return self.multi_model(data, data.batch).argmax(dim=1)[0]

    def predict(self, data):
        '''
        return 0 for no pattern
        return 1 for strategy
        return 2 for singleton
        return 3 for adapter
        return 4 for observer
        return 5 for factory
        return 6 for decorator
        '''
        result = [
            self.strategy_model(data, data.batch).argmax(dim=1),
            self.singleton_model(data, data.batch).argmax(dim=1),
            self.adapter_model(data, data.batch).argmax(dim=1),
            self.observer_model(data, data.batch).argmax(dim=1),
            self.factory_model(data, data.batch).argmax(dim=1),
            self.decorator_model(data, data.batch).argmax(dim=1),
        ]
        # print(softmax(self.singleton_model(data, data.batch), dim=1))
        s = sum(result)
        if s == 0:
            return 0
            multi_result = self.multi_model(data, data.batch)
            multi_result = softmax(multi_result, dim=1)
            # print(multi_result)
            return_result = 0
            for i, prob in enumerate(multi_result[0]):
                if prob > 0.5:
                    return_result = i + 1
            return return_result
        
        if s >= 2:
            indices = [i for i, x in enumerate(result) if x == 1]
            multi_result = self.multi_model(data, data.batch)
            # pick the one with highest probability
            multi_result = softmax(multi_result, dim=1)
            max_index = 0
            max_prob = 0
            for i in indices:
                if multi_result[0][i] > max_prob:
                    max_prob = multi_result[0][i]
                    max_index = i
            return max_index + 1
        else:
            # return the index of the first 1
            return result.index(1) + 1

def k_fold_validation(i, k, data_list):
    n = len(data_list)
    test_size = n // k
    return data_list[i * test_size : (i + 1) * test_size]

def run_predict_statisic():
    predicter = Predicter()
    random.seed(520)
    pattern_folders = ['trainset/no_design_pattern', 'trainset/strategy-clean', 'trainset/singleton-CLEAN', 'trainset/adapter-CLEAN', 'trainset/observer-CLEAN', 'trainset/factory-CLEAN', 'trainset/decorator-CLEAN']
    data_list = []
    no_pattern_data_list = []

    for i, pattern_folder in enumerate(pattern_folders):
        projects = os.listdir(pattern_folder)
        for project in projects:
            graph = JavaProjectDependencyGraph(pattern_folder + '/' + project)
            if i == 0:
                no_pattern_data_list.append((graph.get_torch_geometric_data(), i))
            else:
                data_list.append((graph.get_torch_geometric_data(), i))

    data_list.extend(no_pattern_data_list)
    random.shuffle(data_list)
    k = 4
    cofusion_matrices = []
    precisions = []
    recalls = []
    f1s = []
    for i in range(k):
        validation_data = k_fold_validation(i, k, data_list)
        y_true = [x[1] for x in validation_data]
        y_pred = []
        # predicter.strategy_model.load_state_dict(torch.load(f'0/{i}-fold-model.pt'))
        # predicter.singleton_model.load_state_dict(torch.load(f'1/{i}-fold-model.pt'))
        # predicter.adapter_model.load_state_dict(torch.load(f'2/{i}-fold-model.pt'))
        # predicter.observer_model.load_state_dict(torch.load(f'3/{i}-fold-model.pt'))
        # predicter.factory_model.load_state_dict(torch.load(f'4/{i}-fold-model.pt'))
        # predicter.decorator_model.load_state_dict(torch.load(f'5/{i}-fold-model.pt'))
        predicter.multi_model.load_state_dict(torch.load(f'{i}-fold-model.pt'))

        predicter.strategy_model.eval()
        predicter.singleton_model.eval()
        predicter.adapter_model.eval()
        predicter.observer_model.eval()
        predicter.factory_model.eval()
        predicter.decorator_model.eval()
        predicter.multi_model.eval()

        for data in validation_data:
            y_pred.append(predicter.multi_model_predict(data[0]))

        print(f'Fold {i} accuracy: {accuracy_score(y_true, y_pred)}')
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        cofusion_matrices.append(confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6]))

    plt.figure(figsize=(10, 10))
    for i in range(k):
        plt.subplot(2, 2, i + 1)
        sns.heatmap(cofusion_matrices[i], annot=True, fmt='d', cmap='Blues', xticklabels=['None', 'strategy', 'singleton', 'adapter', 'observer', 'factory', 'decorator'], yticklabels=['None', 'strategy', 'singleton', 'adapter', 'observer', 'factory', 'decorator'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Fold {i}')
    plt.savefig('integrated_confusion_matrix.png')

    # plot every fold's precision, recall, f1 using table
    
    for i in range(k):
        dataframe = pd.DataFrame({
            'Label': ['None', 'strategy', 'singleton', 'adapter', 'observer', 'factory', 'decorator'],
            'Precision': precisions[i],
            'Recall': recalls[i],
            'F1': f1s[i]
        })

        dataframe.to_csv(f'fold-{i}-precision-recall-f1.csv', index=False, sep=',')





def run_custom_project_predict():
    predicter = Predicter()
    #predicter.strategy_model.load_state_dict(torch.load(f'0/2-fold-model.pt'))
    #predicter.singleton_model.load_state_dict(torch.load(f'1/2-fold-model.pt'))
    #predicter.adapter_model.load_state_dict(torch.load(f'2/2-fold-model.pt'))
    #predicter.observer_model.load_state_dict(torch.load(f'3/2-fold-model.pt'))
    #predicter.factory_model.load_state_dict(torch.load(f'4/2-fold-model.pt'))
    #predicter.decorator_model.load_state_dict(torch.load(f'5/2-fold-model.pt'))
    predicter.multi_model.load_state_dict(torch.load(f'trained_models/20240504-1/k_fold_2.pt'))

    #predicter.strategy_model.eval()
    #predicter.singleton_model.eval()
    #predicter.adapter_model.eval()
    #predicter.observer_model.eval()
    #predicter.factory_model.eval()
    #predicter.decorator_model.eval()
    predicter.multi_model.eval()
    project_path = './custom_project/'
    graph = JavaProjectDependencyGraph(project_path)
    data = graph.get_torch_geometric_data()
    # print(predicter.predict(data))
    result = predicter.multi_model_predict(data)

    if result == 0:
        print('No pattern')
        return "No pattern"
    elif result == 1:
        print('Strategy')
        return "Strategy"
    elif result == 2:
        print('Singleton')
        return "Singleton"
    elif result == 3:
        print('Adapter')
        return "Adapter"
    elif result == 4:
        print('Observer')
        return "Observer"
    elif result == 5:
        print('Abstract Factory')
        return "Abstract Factory"
    elif result == 6:
        print('Decorator')
        return "Decorator"
    elif result == 7:
        print('Builder')
        return "Builder"
    elif result == 8:
        print('Bridge')
        return "Bridge"
    elif result == 9:
        print('Composite')
        return "Composite"
    elif result == 10:
        print('Mediator')
        return "Mediator"
    elif result == 11:
        print('Memento')
        return "Memento"
    elif result == 12:
        print('Prototype')
        return "Prototype"
    elif result == 13:
        print('State')
        return "State"
    elif result == 14:
        print('Template')
        return "Template"
    else:
        print(result)

if __name__ == "__main__":
    while True:
        input('按下 Enter 後開始預測')
        run_custom_project_predict()