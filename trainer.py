import os
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from random import shuffle, sample
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import tqdm
import copy
from sklearn.model_selection import StratifiedKFold, train_test_split

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(142, 142)
        self.act1 = nn.ReLU()
        # self.layer1_1 = nn.Linear(284, 142)
        # self.act1_1 = nn.ReLU()
        self.layer2 = nn.Linear(142, 71)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(71, 71)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(71, 1)
        self.sigmoid = nn.Sigmoid()
 
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

class CreateModels:
    def __init__(self, instances_folder, ng_sets_file):
        self.instances_folder = instances_folder
        self.ng_sets_file = ng_sets_file

    def train(self, loader, epoch):
        self.model_ed.train()
        total_loss = 0

        if epoch == 8:
            for g in self.optimizer.param_groups:
                g['lr'] = 0.001

        for batch in loader:
            self.optimizer.zero_grad()
            z = self.model_ed.encode(batch.x, batch.edge_index)

            # We perform a new round of negative sampling for every training epoch:
            neg_edge_index = negative_sampling(
                edge_index=batch.edge_index, num_nodes=batch.num_nodes,
                num_neg_samples=batch.edge_label_index.size(1), method='sparse')

            # Concat positive and negative edge indices.
            edge_label_index = torch.cat(
                [batch.edge_label_index, neg_edge_index],
                dim=-1,
            )
            # Label for positive edges: 1, for negative edges: 0.
            edge_label = torch.cat([
                batch.edge_label,
                batch.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            # Note: The model is trained in a supervised manner using the given
            # `edge_label_index` and `edge_label` targets.
            out = self.model_ed.decode(z, edge_label_index).view(-1)
            loss = self.criterion(out, edge_label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)
    
    @torch.no_grad()
    def test(self, loader):
        self.model_ed.eval()
        all_out = []
        all_labels = []

        for batch in loader:
            z = self.model_ed.encode(batch.x, batch.edge_index)
            out = self.model_ed.decode(z, batch.edge_label_index).view(-1).sigmoid()
            all_out.append(out.cpu().numpy())
            all_labels.append(batch.edge_label.cpu().numpy())

        all_out = np.concatenate(all_out)
        all_labels = np.concatenate(all_labels)
        return roc_auc_score(all_labels, all_out)

    def read_files(self):
        data_files_list = ["./{self.instances_folder}/"+f for f in os.listdir("./{self.instances_folder}") ]
        instance_dict = {}
        for dir_str in data_files_list:
            with open(dir_str, 'r') as text_file:
                cnt = 0
                instance = ""
                for line in text_file:
                    if cnt < 9:
                        if cnt == 0:
                            instance = line.split()[0]
                            instance_dict[instance] = []
                        cnt += 1
                        continue
                    split_line = line.split()
                    instance_dict[instance].append([int(i) for i in split_line])
                text_file.close()

        ng_dict = {}
        cnt = -1
        with open("{self.ng_sets_file}", 'r') as text_file:
            for line in text_file:
                if cnt < 2:
                    cnt += 1
                    continue
                raw_line = line.strip()
                split_line_list = raw_line.split(sep=";")
                instance = split_line_list[3]
                if instance not in ng_dict:
                    ng_dict[instance] = [[0 for i in range(101)]]
                ng_dict[instance].append([0] + [int(i) for i in split_line_list[5:-1]])
                if len(split_line_list[5:-1]) != 100:
                    print("case found for instance "+instance)
            text_file.close()
        return instance_dict, ng_dict

    def format_input(self):
        instance_dict, ng_dict = self.read_files()
        data_list = []
        for instance_name in ng_dict:
            instance = instance_dict[instance_name]
            y = torch.tensor(ng_dict[instance_name], dtype=torch.float)
            x = torch.tensor(instance, dtype=torch.float)
            pos = []
            for i in instance:
                pos.append([i[1], i[2]])
            pos = torch.tensor(pos, dtype=torch.double)
            edge_index = knn_graph(pos, 15)
            data_list.append(Data(x=x, y=y, edge_index=edge_index, pos=pos))
        self.data_list = data_list

    def add_edge_labels(self, graph):
        transform = T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        add_negative_train_samples=False)
        return transform(graph)

    def split_data(self):
        self.format_input()
        labeled_graphs = [self.add_edge_labels(graph) for graph in self.data_list]
        shuffle(labeled_graphs)

        train_size = [g[0] for g in labeled_graphs]
        val_size = [g[1] for g in labeled_graphs]
        test_size = [g[2] for g in labeled_graphs]

        train_loader = DataLoader(train_size, batch_size=150, shuffle=True)
        val_loader = DataLoader(val_size, batch_size=150, shuffle=False)
        test_loader = DataLoader(test_size, batch_size=150, shuffle=False)
        num_features = self.data_list[0].num_features
        return train_loader, val_loader, test_loader, num_features

    def do_train_encoding_decoding(self):
        train_loader, val_loader, test_loader, num_features = self.split_data()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_ed = Net(num_features, 128, 64).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model_ed.parameters(), lr=0.01)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        # Train/Test Loop
        best_val_auc = final_test_auc = 0
        for epoch in range(1, 24):
            loss = self.train(train_loader, epoch)
            val_auc = self.test(val_loader)
            test_auc = self.test(test_loader)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                final_test_auc = test_auc
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, '
                f'Test: {test_auc:.4f}')

        print(f'Final Test: {final_test_auc:.4f}')

    def prepare_classification_data(self, sampling_size):
        self.do_train_encoding_decoding()
        indices = [i for i in range(len(self.data_list))]

        encodings_dict = {}
        targets_dict = {"positives": [], "negatives": []}

        for graph_idx in sample(indices, sampling_size):
            graph = self.data_list[graph_idx]
            ng_matrix = graph.y.tolist()
            graph_x = graph.x
            encoding_matrix = self.model_ed.encode(graph_x, graph.edge_index).tolist()
            graph_x = graph_x.tolist()
            for i in range(1,101):
                encode_i = encoding_matrix[i]
                node_i = graph_x[i]
                new_entry = encode_i + node_i
                new_entry = ["%.5f" % e for e in new_entry]
                new_entry = [float(e) for e in new_entry]
                encodings_dict[graph_idx, i] = new_entry
                for j in range(1,101):
                    if i != j:
                        target = int(ng_matrix[i][j])
                        if target == 1:
                            targets_dict["positives"].append((graph_idx, i, j))
                        else:
                            targets_dict["negatives"].append((graph_idx, i, j))
        return encodings_dict, targets_dict
    
    def sample_classification(self, graph_samples, row_samples=20000):
        encodings_dict, targets_dict = self.prepare_classification_data(graph_samples)
        pos_candidates = sample(targets_dict["positives"], row_samples)
        neg_candidates = sample(targets_dict["negatives"], row_samples)
        pos_list = [encodings_dict[k[0], k[1]] + encodings_dict[k[0], k[2]] + [1] for k in pos_candidates]
        neg_list = [encodings_dict[k[0], k[1]] + encodings_dict[k[0], k[2]] + [0] for k in neg_candidates]
        return pos_list, neg_list

    def classification_data(self):
        pos_list, neg_list = self.sample_classification(len(self.data_list))
        pre_tensor = pos_list + neg_list
        main_tensor = torch.tensor(pre_tensor, dtype=torch.float32)

        main_tensor = main_tensor[torch.randperm(main_tensor.size()[0])]
        labels = main_tensor[:,-1:]
        embeddings = main_tensor[:,:-1]
        return labels, embeddings

    def model_train(self, model, X_train, y_train, X_val, y_val):
        # loss function and optimizer
        loss_fn = nn.BCELoss()  # binary cross entropy
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # optimizer = optim.SGD(model.parameters(), lr=0.001)
    
        n_epochs = 60   # number of epochs to run
        batch_size = 500  # size of each batch
        batch_start = torch.arange(0, len(X_train), batch_size)
    
        # Hold the best model
        best_acc = - np.inf   # init to negative infinity
        best_weights = None
    
        for epoch in range(n_epochs):
            if epoch == 30:
                for g in optimizer.param_groups:
                    g['lr'] = 0.0001
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    X_batch = X_train[start:start+batch_size]
                    y_batch = y_train[start:start+batch_size]
                    # forward pass
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    acc = (y_pred.round() == y_batch).float().mean()
                    bar.set_postfix(
                        loss=float(loss),
                        acc=float(acc)
                    )
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            acc = float(acc)
            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        return best_acc
    
    def classification_model_train(self):
        labels, embeddings = self.classification_data()
        # train-test split: Hold out the test set for final model evaluation
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, train_size=0.7, shuffle=True)
        
        # define 5-fold cross validation test harness
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        cv_scores_deep = []
        for train, test in kfold.split(X_train, y_train):
            # create model, train, and get accuracy
            model_ls = Deep()
            acc = self.model_train(model_ls, X_train[train], y_train[train], X_train[test], y_train[test])
            print("Accuracy (deep): %.2f" % acc)
            cv_scores_deep.append(acc)
        
        # evaluate the model
        deep_acc = np.mean(cv_scores_deep)
        deep_std = np.std(cv_scores_deep)
        print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))

        self.model_ls = Deep()
        acc = self.model_train(model_ls, X_train, y_train, X_test, y_test)
        print(f"Final model accuracy: {acc*100:.2f}%")
    
    def store_models(self):
        torch.save(self.model_ed, 'encoder_decoder.pth')
        torch.save(self.model_ls, 'classifier.pth')



instances_folder = "export"
ng_sets_file = "ng_outs.csv"
trainer = CreateModels(instances_folder, ng_sets_file)
trainer.classification_model_train()
trainer.store_models()