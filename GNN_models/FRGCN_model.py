import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F

# hyperparameters
BATCH_SIZE = 16
IS_SHUFFLE = True
OPTIMIZER_LR = 0.0025
EPOCHES = 40 # fastrgcn: 40, graphconv: 80, gcnconv: 300, gatconv: 150, ginconv: 150
K_FOLD = 4
input_dim = 6#6
hidden_dim = 80
output_dim = 15 # 這裡改成 2 的話就是 binary classification，改成 7 的話就是 multi-class classification
classes = ['none', 'adapter', 'singleton', 'strategy', 'observer', 'factory', 'decorator']

class FRGCNModel(nn.Module):
    def __init__(self):
        super(FRGCNModel, self).__init__()
        self.conv1 = gnn.FastRGCNConv(input_dim, hidden_dim, num_relations=4)
        self.conv2 = gnn.FastRGCNConv(hidden_dim, hidden_dim, num_relations=4)
        
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, batch):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)

        x = gnn.global_add_pool(x, batch)

        x = F.dropout(x, p=0.25, training=self.training)
        x = self.linear(x)
        return x