# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import random
import argparse
import os
import warnings
import models
import util
import copy

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
from utils import process
from utils import aug
from modules.gcn import *
from net.merit import MERIT
from sklearn.metrics import *
from data import *
import xgboost as xgb
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

###


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='sage', choices=['sage', 'gcn', 'GAT'])
parser.add_argument('--nhid', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--setting', type=str, default='smote',
                    choices=['no', 'upsampling', 'smote', 'reweight', 'embed_up', 'recon', 'newG_cls', 'recon_newG'])

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data', type=str, default='Subject A')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--eval_every', type=int, default=1)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.2)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--sample_size', type=int, default=600)
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--sparse', type=str_to_bool, default=True)

parser.add_argument('--gnn_dim', type=int, default=256)
parser.add_argument('--proj_dim', type=int, default=256)
parser.add_argument('--proj_hid', type=int, default=2048)
parser.add_argument('--pred_dim', type=int, default=256)
parser.add_argument('--pred_hid', type=int, default=2048)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--beta', type=float, default=0.4)
parser.add_argument('--alpha', type=float, default=0.9)
parser.add_argument('--drop_edge', type=float, default=0.1)
parser.add_argument('--drop_feat1', type=float, default=0.4)
parser.add_argument('--drop_feat2', type=float, default=0.4)

args = parser.parse_args()
torch.set_num_threads(4)

class MLP(nn.Module):
    def __init__(self, n_feat, n_hid, nclass):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            nn.Linear(n_hid, nclass),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def evaluation(adj, diff, feat, gnn, idx_train, idx_test, sparse):
    model = GCN(input_size, gnn_output_size)  # 1-layer
    model.load_state_dict(gnn.state_dict())
    with torch.no_grad():
        embeds1 = model(feat, adj, sparse)
        embeds2 = model(feat, diff, sparse)
        train_embs = embeds1[0, idx_train] + embeds2[0, idx_train]
        test_embs = embeds1[0, idx_test] + embeds2[0, idx_test]

        train_labels = torch.argmax(labels[0, idx_train], dim=1)
        test_labels = torch.argmax(labels[0, idx_test], dim=1)

    model = xgb.XGBClassifier(max_depth=7, learning_rate=0.05, n_estimators=180)
    model.fit(train_embs, train_labels)
    pred_test_labels = model.predict(test_embs)

    return accuracy_score(test_labels, pred_test_labels), \
           f1_score(test_labels, pred_test_labels), \
           recall_score(test_labels, pred_test_labels), \



if __name__ == '__main__':

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    n_runs = args.runs
    eval_every_epoch = args.eval_every

    # dataset = args.data
    # input_size = args.input_dim

    gnn_output_size = args.gnn_dim
    projection_size = args.proj_dim
    projection_hidden_size = args.proj_hid
    prediction_size = args.pred_dim
    prediction_hidden_size = args.pred_hid
    momentum = args.momentum
    beta = args.beta
    alpha = args.alpha

    drop_edge_rate_1 = args.drop_edge
    drop_feature_rate_1 = args.drop_feat1
    drop_feature_rate_2 = args.drop_feat2

    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    sample_size = args.sample_size
    batch_size = args.batch_size
    patience = args.patience

    sparse = args.sparse

    # Loading dataset
    adj, features, labels, idx_train, idx_val, idx_test = data()
    # adj, features, labels, idx_train, idx_test = data()



    if args.setting == 'upsampling':
        adj, features, labels, idx_train = util.src_upsample(adj, features, labels, idx_train, portion=1,
                                                              im_class_num=1)
    if args.setting == 'smote':
        adj, features, labels, idx_train = util.src_smote(adj, features, labels, idx_train, portion=1,
                                                               im_class_num=1)

    ## From  'GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks' https://github.com/TianxiangZhao/

    adj = adj.detach().to_dense()
    adj = sp.csr_matrix(adj)
    labels = encode_onehot(labels.numpy())


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if args.model == 'sage':
        encoder = models.Sage_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.Sage_Classifier(nembed=args.nhid,
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    elif args.model == 'gcn':
        encoder = models.GCN_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GCN_Classifier(nembed=args.nhid,
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    elif args.model == 'GAT':
        encoder = models.GAT_En(nfeat=features.shape[1],
                nhid=args.nhid,
                nembed=args.nhid,
                dropout=args.dropout)
        classifier = models.GAT_Classifier(nembed=args.nhid,
                nhid=args.nhid,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)

    decoder = models.Decoder(nembed=args.nhid,
                             dropout=args.dropout)

    embed = encoder(features, adj)

    ori_num = labels.shape[0]
    embed, labels_new, idx_train_new, adj_up = util.recon_upsample(embed, labels, idx_train, adj=adj.detach(),
                                                                    portion=1, im_class_num=1)
    generated_G = decoder(embed)
    adj_new = copy.deepcopy(generated_G.detach())
    threshold = 0.5
    adj_new[adj_new < threshold] = 0.0
    adj_new[adj_new >= threshold] = 1.0


    adj_new = torch.mul(adj_up, adj_new)

    adj_new[:ori_num, :][:, :ori_num] = adj.detach()
    adj_new = adj_new.detach()
    adj = sp.csr_matrix(adj_new)
    labels_new = labels_new.numpy()
    labels = encode_onehot(labels_new)
    features = embed.detach().numpy().astype(np.uint8)
    features = torch.FloatTensor(features)

    input_size = features.shape[1]
    dataset = args.data
    if os.path.exists('data/diff_{}_{}.npy'.format(dataset, alpha)):
        diff = np.load('data/diff_{}_{}.npy'.format(dataset, alpha), allow_pickle=True)
    else:
        diff = aug.gdc(adj, alpha=alpha, eps=0.0001)
        np.save('data/diff_{}_{}'.format(dataset, alpha), diff)

    # features, _ = process.preprocess_features(features)

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = labels.shape[1]

    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    norm_diff = sp.csr_matrix(diff)
    if sparse:
        eval_adj = process.sparse_mx_to_torch_sparse_tensor(norm_adj)
        eval_diff = process.sparse_mx_to_torch_sparse_tensor(norm_diff)
    else:
        eval_adj = (norm_adj + sp.eye(norm_adj.shape[0])).todense()
        eval_diff = (norm_diff + sp.eye(norm_diff.shape[0])).todense()
        eval_adj = torch.FloatTensor(eval_adj[np.newaxis])
        eval_diff = torch.FloatTensor(eval_diff[np.newaxis])

    result_over_runs = []

    # Initiate models
    model = GCN(input_size, gnn_output_size)
    ## From Multi-Scale Contrastive Siamese Networks for Self-Supervised Graph Representation Learning. https://github.com/GRAND-Lab/MERIT
    merit = MERIT(gnn=model,
                  feat_size=input_size,
                  projection_size=projection_size,
                  projection_hidden_size=projection_hidden_size,
                  prediction_size=prediction_size,
                  prediction_hidden_size=prediction_hidden_size,
                  moving_average_decay=momentum, beta=beta).to(device)

    opt = torch.optim.Adam(merit.parameters(), lr=lr, weight_decay=weight_decay)

    results = []

    # Training
    best = 0
    patience_count = 0
    for epoch in range(epochs):
        for _ in range(batch_size):
            idx = np.random.randint(0, adj.shape[-1] - sample_size + 1)
            ba = adj[idx: idx + sample_size, idx: idx + sample_size]
            bd = diff[idx: idx + sample_size, idx: idx + sample_size]
            bd = sp.csr_matrix(np.matrix(bd))
            features = features.squeeze(0)
            bf = features[idx: idx + sample_size]

            aug_adj1 = aug.aug_random_edge(ba, drop_percent=drop_edge_rate_1)
            aug_adj2 = bd
            aug_features1 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_1)
            aug_features2 = aug.aug_feature_dropout(bf, drop_percent=drop_feature_rate_2)

            aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
            aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

            if sparse:
                adj_1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1).to(device)
                adj_2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2).to(device)
            else:
                aug_adj1 = (aug_adj1 + sp.eye(aug_adj1.shape[0])).todense()
                aug_adj2 = (aug_adj2 + sp.eye(aug_adj2.shape[0])).todense()
                adj_1 = torch.FloatTensor(aug_adj1[np.newaxis]).to(device)
                adj_2 = torch.FloatTensor(aug_adj2[np.newaxis]).to(device)

            aug_features1 = aug_features1.to(device)
            aug_features2 = aug_features2.to(device)

            opt.zero_grad()
            loss = merit(adj_1, adj_2, aug_features1, aug_features2, sparse)
            loss.backward()
            opt.step()
            merit.update_ma()

        if epoch % eval_every_epoch == 0:
            acc, auc, F1, recall, precision, r1, r2 = evaluation(eval_adj, eval_diff, features, model, idx_train, idx_test, sparse)
            if F1 > best:
                best = F1
                patience_count = 0
            else:
                patience_count += 1
            results.append(F1)
            print('\t epoch {:03d} | loss {:.5f} | test acc {:.5f}| test F1 {:.5f}| test recall {:.5f}'
                  .format(epoch+1, loss.item(), acc,  F1, recall))
            if patience_count >= patience:
                print('Early Stopping.')
                break

    result_over_runs.append(max(results))
    print('\t best auc {:.5f}'.format(max(results)))