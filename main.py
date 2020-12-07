# Python3 Code
# Three Water's Coding Style
# edited and tested by Akida-Sho Wong
#
# log
#
# @Oct 3 11:19:18am begins
import os
from tqdm import tqdm

import random
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

# modules
from model import ToyCifarNet
from dataSplit import get_data_loaders
from utils import AverageMeter
# from sampling import uniform_sampling
from LossRatio import get_auxiliary_data_loader, compute_ratio_per_client_update
from scipy.stats import entropy

# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

NUM_CLASSES = 10
# Parameters that can be tuned during different simulations
num_clients = 100
num_selected = 20
num_rounds = 3150	# num of communication rounds

epochs = 5			# num of epochs in local client training (An epoch means you go through the entire dataset in that client)
batch_size = 10 # batch size already set in train dataloaderin dataSplit.py
num_batch = 10 # change here choose 100 - 300 default:50

# hyperparameters of deep models
lr = 0.1 # learning rate
decay_factor = 0.996

losses_train = []
losses_test = []
acc_train = []
acc_test = []
client_idx_lst = []
T_pull_lst = []



alpha = 0.8
# cucb parameters
reward_est = np.zeros(num_clients)
reward_bias = np.zeros(num_clients)
T_pull = np.zeros(num_clients)
reward_global = np.zeros(num_rounds)
reward_client = np.zeros(num_clients)

# V_pt statistic
V_pt_avg = np.zeros((num_clients, NUM_CLASSES))

def traverse(N,K,RR):
    '''sampling each client at least ones'''
    R = int(np.ceil(N/K))
    R_set = np.arange(R)

    selected_set = np.zeros(K)

    if RR in R_set:
        idx = np.where(R_set == RR)[0][0]
        selected_set = np.arange(idx*K, (idx+1)*K)
    for i in range(K):
        if selected_set[i] >= N:
            selected_set[i] = selected_set[i] - N
    return selected_set

def cross_entropy(y):
    x = np.ones(y.shape)
    return entropy(x) + entropy(x, y + 1e-15)

def get_client_set(reward, K, N, V_pt_avg):
    '''get client set for cucb'''
    # create dict
    V_pt_dict = {}
    for i in range(N):
        V_pt_dict[i] = V_pt_avg[i,:]
    # choose the max reward client as base
    r_max_index = np.argmax(reward)
    # remove the min index
    V_pt_dict.pop(r_max_index)
    # combination index set S
    S = np.array(r_max_index)
    # combination distribution set
    comb_set = np.array(V_pt_avg[r_max_index])

    while S.size < K:
        ce_reward_set = {}
        for key,value in V_pt_dict.items():
            # calculate the avg class distribution
            comb_dist = np.vstack([comb_set, V_pt_dict[key]])
            comb_dist_avg = np.sum(comb_dist, axis=0) / comb_dist.shape[0]
            # calculate cos loss of combined distribution
            ce_loss = cross_entropy(comb_dist_avg)
            ce_reward_set[key] = 1 / ce_loss

        # get the cos ratio loss index
        reward_max_idx = max(V_pt_dict.keys(),key=(lambda x:ce_reward_set[x]))

        # remove the selected client

        S = np.append(S, reward_max_idx)
        comb_set = np.vstack([comb_set, V_pt_dict[reward_max_idx]])
        V_pt_dict.pop(reward_max_idx)

    return S

# FedAvg
def main():
	# data loader
	## client_train_loader: a list with #clients of **local** data loaders
	## test_loader: test.py only on **global** testset (local doesn't own seperate test.py data)

    client_train_loader, test_loader, data_size_per_client = get_data_loaders(num_clients, batch_size, True)
    data_size_weights = data_size_per_client / data_size_per_client.sum()
    
    aux_loader = get_auxiliary_data_loader(testset_extract = True, data_size = 32)
    # model configurations
    # ASSUME that global model and client models are with exactly same structures
    # ASSUME that num_select is constant during FedAvg
    global_model = ToyCifarNet().to(device)
    client_models = [ToyCifarNet(init_weights=False).to(device) for _ in range(num_selected)]

    ## client models initialized by global model
    for model in client_models:
        model.load_state_dict(global_model.state_dict())


	# client optimizers setting
	# ASSUME that client optimizer settings are all the same, and controlled by global server
	### XXXXXXXX ###
    ## [TODO] Here need to finetune to match the standard training settings common
    ##... in CifarNet training under datacenter SGD and FedAvg settings
    ##... consider later
    ### XXXXXXXX ###
    opt_lst = [optim.SGD(model.parameters(), lr=lr, weight_decay=5e-4) for model in client_models]
    
    # for each communication round
    for r in range(num_rounds):

        print(lr * (decay_factor ** r))
        
            
        for opt in opt_lst:
            opt.param_groups[0]['lr'] = lr * (decay_factor ** r)

        RR = int(np.ceil(num_clients / num_selected))
        if r < RR:
            client_idx = traverse(num_clients, num_selected, r)

        else:
            reward_bias = reward_client + alpha * np.sqrt(3 * np.log(r) / (2 * T_pull))
            client_idx = get_client_set(reward_bias, num_selected, num_clients, V_pt_avg)

        # new: update pull time of each client
        print("client_idx: ", client_idx)
        for s in client_idx:
            T_pull[s] += 1

        loss = 0
        
        for i in range(num_selected):
            loss += client_update(client_models[i], opt_lst[i], client_train_loader[client_idx[i]], epochs)
            
        ra_dict = compute_ratio_per_client_update(client_models, client_idx, aux_loader)
            
            
        for i in range(num_selected):
            
            # new: get V_pt i, then calculate reward statistic (mean)
            reward_single, V_pt = 1/cross_entropy(ra_dict[client_idx[i]]), ra_dict[client_idx[i]]

            reward_client[client_idx[i]] = (reward_client[client_idx[i]]*(T_pull[client_idx[i]]-1) + reward_single) / T_pull[client_idx[i]]
            V_pt_avg[client_idx[i]] = (V_pt_avg[client_idx[i]]*(T_pull[client_idx[i]] - 1) + V_pt) / T_pull[client_idx[i]]
            
            
        

        # get reward of global model
        reward_global[r] = 1/cross_entropy(compute_ratio_per_client_update([global_model], client_idx, aux_loader)[client_idx[0]])

        # loss needed to average across selected clients
        losses_train.append(loss / num_selected)

        # Step 4: Updated local models send back for server aggregate
        server_aggregate(global_model, client_models, data_size_weights, client_idx)

        # Step 5: return loss and acc on testset
        test_loss, acc = test(global_model, test_loader)
        losses_test.append(test_loss)
        acc_test.append(acc)

        print('%d-th round' % r)
        print('average train loss %0.3g | test.py loss %0.3g | test.py acc: %0.3f' % (loss / num_selected, test_loss, acc))
        
        
        T_pull_lst.append(np.copy(T_pull))
        client_idx_lst.append(client_idx)
        name = "log_lr"+str(lr)+"_decay"+str(decay_factor)+"_alpha"+str(alpha)+"_C"+str(num_clients)+"_S"+str(num_selected)+"_Nbatch"+str(num_batch)+".mat"

        sio.savemat(name, {'V_pt_avg': V_pt_avg, 'reward_client': reward_client, 'reward_global': reward_global, 'acc_test': acc_test, 'T_pull': T_pull_lst, 'client_idx': client_idx_lst})




# actually standard training in a local client/device
def client_update(client_model, optimizer, train_loader, epoch):
    loss_avg = AverageMeter()

    client_model.train()

    for e in range(epoch):
        # batch size from data loader is set to 32
        for batch_idx, (data, target) in enumerate(train_loader):

            if batch_idx == num_batch:
                break

            # transfer a mini-batch to GPU
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = client_model(data)
            # here use F.nll_loss() is wrong
            loss = F.cross_entropy(output, target)

            loss_avg.update(loss.item(), data.size(0))

            loss.backward()

            optimizer.step()



    return loss_avg.avg # average loss in this client over entire trainset over multiple epochs

def server_aggregate(global_model, client_models, data_size_weights, client_idx):
    global_dict = global_model.state_dict()

    non_sampled_client_idx = [i for i in list(range(num_clients)) if i not in list(client_idx)]

    for k in global_dict.keys():

        global_dict[k] = torch.stack([data_size_weights[client_idx[i]] * client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).sum(0) + sum([data_size_weights[idx] for idx in non_sampled_client_idx]) * global_model.state_dict()[k].float()


    global_model.load_state_dict(global_dict)



    # step of 'send aggregated global model to client' is here
    for model in client_models:
        model.load_state_dict(global_model.state_dict())

def test(global_model, test_loader):
    loss_avg = AverageMeter()
    acc_avg = AverageMeter()

    global_model.eval()

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = global_model(data)

            loss = F.cross_entropy(output, target)
            loss_avg.update(loss.item(), data.size(0))


            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            acc_avg.update(pred.eq(target.view_as(pred)).sum().item(), data.size(0))

    return loss_avg.avg, acc_avg.avg



if __name__ == '__main__':
	main()
