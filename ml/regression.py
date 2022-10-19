import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv
from data import SyntheticDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.nn import AvgPooling, GNNExplainer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import neptune.new as neptune
from neptune.new.types import File



class Regress(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(Regress, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim,weight = True, allow_zero_in_degree = True)
        self.conv2 = GraphConv(hidden_dim, hidden_dim,weight = True, allow_zero_in_degree = True)
        #self.linear = nn.Linear(in_dim, hidden_dim)
        self.MLP_layer = MLPReadout(hidden_dim, 1)   # 1 out dim since regression problem 

    def forward(self, graph, feat, eweight=None):
        feat = F.relu(self.conv1(graph, feat))
        feat = F.relu(self.conv2(graph, feat))
        
        with graph.local_scope():
            #feat = self.linear(feat)
            graph.ndata['h'] = feat
            hg = dgl.mean_nodes(graph, 'h')
            return self.MLP_layer(hg)

class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim, L=3): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

# prepare dataset 
dataset = SyntheticDataset()
num_examples = len(dataset)
num_train = int(num_examples * 0.8)
train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))
train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=32, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=32, drop_last=False)

""" parameters = {
    "epoch": 100,
    "learning_rate": 0.001,
    "batch_size": 32,
    "hidden_dim": 128,
} """

# train loop
device = torch.device('cpu')
model = Regress(19, 128).to(device)
print(model)
num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print("Number of Trainable Parameters:",num_params)
opt = torch.optim.Adam(model.parameters(), lr=0.001)

run = neptune.init(
    project="kempcao/gnnscheduling",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjcwZjM5NS0wNTE5LTQxZmUtOTljNS1mZWExOTkwMGVlMjMifQ==",
)  # your credentials

for epoch in range(10000):
    for batched_graph, labels in train_dataloader:
        feats = batched_graph.ndata['type']
        #feat1 = batched_graph.ndata['type']
        #feat2 = batched_graph.ndata['prop'].unsqueeze(1)
        #feats = torch.cat((feat1,feat2),dim=1)
        pred = model(batched_graph, feats)
        pred = torch.squeeze(pred,1)
        loss = F.l1_loss(pred, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # creating a logging object so that you can track it on Neptune dashboard
        run['metrics/train_loss'].log(loss)

    if epoch % 50 == 0:
        print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

# test 
loss = 0
for batched_graph, labels in test_dataloader:
    feats = batched_graph.ndata['type']
    #feat1 = batched_graph.ndata['type']
    #feat2 = batched_graph.ndata['prop'].unsqueeze(1)
    #feats = torch.cat((feat1,feat2),dim=1)
    pred = model(batched_graph, feats)
    pred = torch.squeeze(pred,1)
    loss = F.l1_loss(pred, labels)

print('Test accuracy:', loss.item())

 # Explain the prediction for graph 0
explainer = GNNExplainer(model, num_hops=2)
g, target = dataset[0]
nodes = g.nodes()
features = g.ndata['type']
#features = torch.cat((g.ndata['type'],g.ndata['prop'].unsqueeze(1)),dim=1)
output = model(g,features)
print("test regression value:", output)
print("real value:", target)

feat_mask, edge_mask = explainer.explain_graph(g, features) 


# visualize
space = {'OUTSIDE': 0, 'BASEMENT':1, 'STAIRS':2, 'CORRIDOR': 3, 'TECHNICAL ROOM':4, 'BEDROOM':5, 
    'BATHROOM':6, 'DINING/LIVING ROOM':7, 'KITCHEN':8, 'HALL':9, 'WC':10, 'WALK-IN CLOSET':11, 
    'GALLERY':12, 'SHOWER/WC':13, 'PANTRY':14, 'STORAGE':15, 'STUDY':16,'COAT RACK':17, 'DIELE  ':18}
space = {y: x for x, y in space.items()} 
print(space)
G = dgl.to_networkx(g)
widths = [5 * i for i in edge_mask.tolist()]
nodelist = G.nodes()
labellist = [space[i] for i in np.argmax(g.ndata['type'],axis=1).tolist()]
edgelist = G.edges()
plt.figure(figsize=(12,8))

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,
                       nodelist=nodelist,
                       node_size=1800,
                       node_color='black')
nx.draw_networkx_edges(G,pos,
                       edgelist = edgelist,
                       width=widths,
                       edge_color='blue')
nx.draw_networkx_labels(G, pos=pos,
                        labels=dict(zip(nodelist,labellist)),
                        font_color='white',
                        font_size=5)
plt.box(False)
plt.show()


