import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse

import os

from grb.model.torch import GCN as GCNgrb
from grb.utils.normalize import GCNAdjNorm, SAGEAdjNorm
import scipy.sparse as sp

import time

import sys
import json

import random
from hang_model import GNN_graphcon
from torch_geometric.utils import to_scipy_sparse_matrix,dense_to_sparse
from tqdm import tqdm, trange
def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_model(model,data):
    model.eval()
    accs = []
    with torch.no_grad():
        logits = model(data.features,data.adj)
        # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        #     pred = logits[mask].max(1)[1]
        #     acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        #     accs.append(acc)
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.labels[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return  accs



def test_defend(args,data):
    features = data.features
    labels = data.labels
    adj = data.adj

    opt = vars(args)
    opt['num_classes'] = labels.max().item() + 1
    model = GNN_graphcon(opt, features.shape[1], args.device)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lf = torch.nn.CrossEntropyLoss()
    best_time = val_acc = test_acc = train_acc = best_epoch = 0
    counter = 0
    # add tqdm
    epoch_bar = trange(args.epochs,ncols=100)
    for i in epoch_bar:
        model.train()
        optimizer.zero_grad()
        out = model(features, adj)
        loss = lf(out[data.train_mask], data.labels.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()
        # set tqdm description

        # print("Epoch: {:03d}, Train loss: {:.4f}".format(i, loss.item()))
        tmp_train_acc, tmp_val_acc, tmp_test_acc = test_model(model, data)
        if tmp_val_acc > val_acc:
            val_acc = tmp_val_acc
            test_acc = tmp_test_acc
            train_acc = tmp_train_acc
            best_epoch = i
            counter = 0
        else:
            counter += 1
        # print("Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(i, tmp_train_acc, tmp_val_acc, tmp_test_acc))
        epoch_bar.set_description("Epoch: {:03d}, loss: {:.4f},Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(i, loss.item(),tmp_train_acc, tmp_val_acc, tmp_test_acc))
        if counter == args.patience:
            print("Early Stopping")
            break

    print("Best Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}".format(best_epoch, train_acc, val_acc, test_acc))
    return train_acc, val_acc, test_acc





def main(args):
    args.device = torch.device('cuda', args.gpu)
    set_seed(args.seed)
    # Load dataset
    data = Dataset(root='/tmp/', name=args.dataset, setting='prognn')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # idx_unlabeled = np.union1d(idx_val, idx_test)
    print("idx_train: ", len(idx_train))
    print("idx_val: ", len(idx_val))
    print("idx_test: ", len(idx_test))
    # perturbations = int(args.ptb_rate * (adj.sum() // 2))
    adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    attack_name = args.dataset + "_meta_adj_"+ str(args.ptb_rate) +".npz"

    if args.defence in ['gcn']:
        data.adj = adj.to(args.device)
    else:
        adj_csr = dense_to_sparse(adj)
        data.adj = adj_csr

    # adj_csr = csr_matrix(adj)
    data.features = features.to(args.device)
    data.labels = labels.to(args.device)
    # transfer idx_train to train_mask in torch
    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    train_mask[idx_train] = 1
    data.train_mask = train_mask
    data.val_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    data.val_mask[idx_val] = 1
    data.test_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    data.test_mask[idx_test] = 1

    # test original data
    print("Test original data")
    _,_, test_acc_clean = test_defend(args,data)
    modified_adj = sp.load_npz("./metattack/" + attack_name)
    modified_adj = modified_adj.todense()
    # transfer to torch
    modified_adj = torch.from_numpy(modified_adj).float().to(args.device)
    if args.defence in ['gcn']:
        data.adj = modified_adj.to(args.device)
    else:
        adj_mod_csr = dense_to_sparse(modified_adj)
        data.adj = adj_mod_csr
    print("Test modified data")
    _,_, test_acc_adv = test_defend(args,data)

    return test_acc_clean, test_acc_adv











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of hidden layers.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset', type=str, default='pubmed', choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.2,  help='pertubation rate')
    parser.add_argument('--model', type=str, default='Meta-Self',
            choices=['Meta-Self', 'A-Meta-Self', 'Meta-Train', 'A-Meta-Train'], help='model variant')

    parser.add_argument('--defence', type=str, default='hamgcnv5', help='model variant')
    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
    parser.add_argument('--patience', type=int, default=100, help='patience.')
    parser.add_argument('--runtime', type=int, default=3, help='runtime.')
    parser.add_argument('--time_ode', type=int, default=3, help='runtime.')

    ###### args for pde model ###################################

    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension.')
    parser.add_argument('--proj_dim', type=int, default=256, help='proj_dim dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.0, help='Input dropout rate.')
    # parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    # parser.add_argument('--lr', type=float, default=0.005, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                        help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=3.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                        help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, default='euler',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1.0,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100000, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                        help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                        help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                        help="Maximum number steps for the dopri5Early test integrator. "
                             "used if getting OOM errors at test time")

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                        help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=16,
                        help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                        help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                        help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                        help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    parser.add_argument('--data_norm', type=str, default='gcn',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--alpha_ode', type=float, default=0.5, help='alpha_ode')

    args = parser.parse_args()

    args_ori = args

    # generate
    timestr = time.strftime("%H%M%S")
    if not os.path.exists("./log_meta"):
        os.makedirs("./log_meta")
    filename_log = "./log_meta/" + args.dataset + "_" + args.function + "_" + "all_ptb_" + timestr + ".txt"
    # save command lines
    with open(filename_log, "a") as f:
        f.write(" ".join(sys.argv) + "\n")

    for ptb_rate in [0.05,0.1,0.15,0.2,0.25]:
        args.ptb_rate = ptb_rate
        seed_init = args.seed
        test_clean = []
        test_adv = []

        for j in range(args.runtime):
            args.seed = seed_init + j
            test_acc_clean, test_acc_adv = main(args)
            test_clean.append(test_acc_clean)
            test_adv.append(test_acc_adv)

        print("Test clean: ", np.mean(test_clean), np.std(test_clean))
        print("Test adv: ", np.mean(test_adv), np.std(test_adv))
        # create file to save results

        with open(filename_log, "a") as f:
            f.write("ptb_rate: " + str(args.ptb_rate) + "\n")
            f.write("Test clean: " + str(np.mean(test_clean)) + " " + str(np.std(test_clean)) + "\n")
            f.write("Test adv: " + str(np.mean(test_adv)) + " " + str(np.std(test_adv)) + "\n")
            f.write("\n")
    # save args
    # opt = vars(args_ori)
    # with open(filename_log, "a") as f:
    #     json.dump(opt, f, indent=2)
    #     f.write("\n")

