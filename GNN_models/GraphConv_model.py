import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import os
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, List
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from java_project_dependency_graph import JavaProjectDependencyGraph
import random
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

# hyperparameters
# TRAIN_RATE = 0.8
BATCH_SIZE = 16
IS_SHUFFLE = True
OPTIMIZER_LR = 0.0025
EPOCHES = 40 # fastrgcn: 40, graphconv: 80, gcnconv: 300, gatconv: 150, ginconv: 150
K_FOLD = 4
input_dim = 6
hidden_dim = 80
output_dim = 7
classes = ['none', 'adapter', 'singleton', 'strategy', 'observer', 'factory', 'decorator']

class GraphConvPatternRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphConvPatternRecognitionModel, self).__init__()
        self.conv1 = gnn.GraphConv(input_dim, hidden_dim)
        self.conv2 = gnn.GraphConv(hidden_dim, hidden_dim)

        #self.conv1 = gnn.GATConv(input_dim, hidden_dim, heads=5, dropout=0.25)
        #self.conv2 = gnn.GATConv(hidden_dim * 5, hidden_dim, heads=5, dropout=0.25)

        #self.conv1 = gnn.GINConv(nn.Sequential(
        #    nn.Linear(input_dim, hidden_dim),
        #))
        #self.conv2 = gnn.GINConv(nn.Sequential(
        #    nn.Linear(hidden_dim, hidden_dim),
        #))

        #self.linear = nn.Linear(hidden_dim * 5, output_dim)
        
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, batch):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        #x = self.conv1(x, edge_index)
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #x = self.conv2(x, edge_index)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)

        # readout layer
        #x = gnn.global_mean_pool(x, batch)
        x = gnn.global_add_pool(x, batch)

        # apply a final classifier
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.linear(x)
        return x

