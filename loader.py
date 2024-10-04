import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import sys
from torch_geometric.nn import GCNConv

instance_name = sys.argv[1]
output_file = sys.argv[2]

class Deep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(142, 142)
        self.act1 = torch.nn.ReLU()
        # self.layer1_1 = nn.Linear(284, 142)
        # self.act1_1 = nn.ReLU()
        self.layer2 = torch.nn.Linear(142, 71)
        self.act2 = torch.nn.ReLU()
        self.layer3 = torch.nn.Linear(71, 71)
        self.act3 = torch.nn.ReLU()
        self.output = torch.nn.Linear(71, 1)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        x = self.act1(self.layer1(x))
        # x = self.act1_1(self.layer1_1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

individual_instance = {}
with open(instance_name, 'r') as text_file:
    cnt = 0
    instance = ""
    for line in text_file:
        if cnt < 9:
            if cnt == 0:
                instance = line.split()[0]
                individual_instance[instance] = []
            cnt += 1
            continue
        split_line = line.split()
        individual_instance[instance].append([int(i) for i in split_line])
    text_file.close()

instance_id = instance_name[:-4]
instance = individual_instance[instance_id]
x = torch.tensor(instance, dtype=torch.float)
pos = []
tw_sets_dict = {}
for i in instance:
    pos.append([i[1], i[2]])
pos = torch.tensor(pos, dtype=torch.double)
edge_index = knn_graph(pos, 15)
raw_instance = Data(x=x, edge_index=edge_index, pos=pos)

model_ed = torch.load('encoder_decoder.pth')
model_ls = torch.load('classifier.pth')

z_raw = model_ed.encode(raw_instance.x, raw_instance.edge_index)
with open(output_file, "w") as output_file:
    for i in range(1,101):
        line = ""
        for j in range(1,101):
            if i != j:
                if j > 1:
                    line += ","
                node_i = z_raw[i].tolist() + raw_instance.x[i].tolist()
                node_j = z_raw[j].tolist() + raw_instance.x[j].tolist()
                prediction = (1 if model_ls(torch.tensor(node_i+node_j, dtype=torch.float)) > 0.2 else 0)
                line += str(prediction)
            else:
                if j > 1:
                    line += ",1"
                else:
                    line+="1"
        line += ";"
        output_file.write(line)