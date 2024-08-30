import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

instance_name = "JLA000011"
folder = "./export/"
ng_outs_file = "ng_hard_instances.csv"
ng_outs_folder = "ng_sets/"
output_file = instance_name + "_ngset.txt"

individual_instance = {}
with open(folder+instance_name+".txt", 'r') as text_file:
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

ng_dict_aux = {}
cnt = -1
with open(ng_outs_folder+ng_outs_file, 'r') as text_file:
    for line in text_file:
        if cnt < 2:
            cnt += 1
            continue
        raw_line = line.strip()
        split_line_list = raw_line.split(sep=";")
        instance = split_line_list[3]
        if instance not in ng_dict_aux:
            ng_dict_aux[instance] = [[0 for i in range(101)]]
        ng_dict_aux[instance].append([0] + [int(i) for i in split_line_list[5:-1]])
        if len(split_line_list[5:-1]) != 100:
            print("case found for instance "+instance)
    text_file.close()

instance = individual_instance[instance_name]
y = torch.tensor(ng_dict_aux[instance_name], dtype=torch.float)
x = torch.tensor(instance, dtype=torch.float)
pos = []
tw_sets_dict = {}
for i in instance:
    pos.append([i[1], i[2]])
pos = torch.tensor(pos, dtype=torch.double)
edge_index = knn_graph(pos, 15)
raw_instance = Data(x=x, y=y, edge_index=edge_index, pos=pos)

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
                target = raw_instance.y[i][j]
                prediction = (1 if model_ls(torch.tensor(node_i+node_j, dtype=torch.float)) > 0.5 else 0)
                line += str(prediction)
            else:
                if j > 1:
                    line += ",1"
                else:
                    line+="1"
        line += "\n"
        output_file.write(line)