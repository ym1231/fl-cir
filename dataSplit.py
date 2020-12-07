import os
from tqdm import tqdm

import random
import numpy as np


import torch

import torchvision
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import Compose

torch.backends.cudnn.benchmark = True

# parameters to be tuned
classes_pc = 2
num_clients = 20
batch_size = 10
real_wd = False # False: non_iid dataset, True: Real-world dataset

random.seed(43)
np.random.seed(43)

def get_cifar10():
    '''Return CIFAR10 train/test data and labels as numpy arrays'''
    data_train = datasets.CIFAR10('./data', train=True, download=True)
    data_test = datasets.CIFAR10('./data', train=False, download=True)

    x_train, y_train = data_train.data.transpose((0, 3, 1, 2)), np.array(data_train.targets)
    x_test, y_test = data_test.data.transpose((0, 3, 1, 2)), np.array(data_test.targets)

    return x_train, y_train, x_test, y_test

# [DONE] test finished
# print the following stats info:
# Data:
#  - Train Set: ((50000, 3, 32, 32),(50000,)), Range: [0.000, 255.000], Labels: 0,..,9
#  - Test Set: ((10000, 3, 32, 32),(10000,)), Range: [0.000, 255.000], Labels: 0,..,9
def print_image_data_stats(data_train, labels_train, data_test, labels_test):
    print("\nData: ")
    print(" - Train Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_train.shape, labels_train.shape, np.min(data_train), np.max(data_train),
        np.min(labels_train), np.max(labels_train)))
    print(" - Test Set: ({},{}), Range: [{:.3f}, {:.3f}], Labels: {},..,{}".format(
        data_test.shape, labels_test.shape, np.min(data_train), np.max(data_train),
        np.min(labels_test), np.max(labels_test)))

# [DONE] test finished
# use case: to_ret = clients(50000, 20)
# training data of 50000, num of clients of 20
###
# example 1: clients_rand(50000, 10)
# output: [6637, 8495, 4867, 3893, 2035, 4867, 1592, 8849, 4336, 4429]
#
# example 2:clients_rand(50000, 20)
# output: [2516,2936,3691,3859,3397,1635,629,3984,4194,3565,1342,1510,1761,
# 2432,2265,1384,3104, 3271, 419, 2106]

def clients_rand(train_len, num_clients, weights = [1 / num_clients for i in range(num_clients)], rand_partition = True):
    '''
    train_len: size of the train data
    nclients: number of clients
    (Not implemented) weight: user-defined parameter, distrute the number of examples to clients
    rand_partition: modified original random partition version by @threeWater, just set True now.

    Returns: to_ret

    This function creates a random distribution
    for the clients, i.e. number of images each client
    possess.

    '''

    # this part when random = True, contributed by @threeWater
    # there's little issue in original version:
    # 		always output a small number no matter which num_clients chose
    # 		maybe caused by numerial errors
    # [DONE] fixed in this version
    if rand_partition: # True by default
    	client_tmp = []
    	sum_ = 0
    	#### creating random values for each client ####
    	for i in range(num_clients - 1):
        	tmp = random.randint(10, 100) # fucking hardcode
        	sum_ += tmp
        	client_tmp.append(tmp)

    	client_tmp = np.array(client_tmp)
    	#### using those random values as weights ####
    	# error in this expression
    	# clients_dist = ((client_tmp / sum_) * train_len).astype(int)
    	# modified a little to
    	clients_dist = ((client_tmp / (sum_ + 50)) * train_len).astype(int) # "sum_+50" hardcoded
    	num = train_len - clients_dist.sum()
    	to_ret = list(clients_dist)
    	to_ret.append(num)
    	return to_ret


# [DONE] test finished
# test case
# example: split_image_data_realwd(data_train, labels_train, 20)
#
# print info:
### --------- ###
# Data split:
#  - Client 0: [500 312   0   0 500 454 384 384 500   0]
#  - Client 1: [500 312   0   0   0 454 384 384   0   0]
#  - Client 2: [  0 312   0   0 500 454 384 384   0   0]
#  - Client 3: [  0 312   0   0   0   0   0 384   0   0]
#  - Client 4: [  0 312   0 555   0   0 384 384 500   0]
#  - Client 5: [  0 312 555 555 500 454 384 385 500   0]
#  - Client 6: [  0 312   0   0   0   0   0   0   0   0]
#  - Client 7: [  0 312   0   0   0   0 385   0   0   0]
#  - Client 8: [500 313 555 555   0 454 385 385 500   0]
#  - Client 9: [500 313 555   0 500   0 385 385   0   0]
#  - Client 10: [500 313 555   0   0   0   0   0   0   0]
#  - Client 11: [500 313 556 555 500 455 385 385 500   0]
#  - Client 12: [   0  313    0  556  500  455    0    0  500 1000]
#  - Client 13: [ 500    0  556  556  500  455    0    0  500 1000]
#  - Client 14: [  0   0   0   0   0   0 385   0   0   0]
#  - Client 15: [   0    0    0  556  500  455  385  385  500 1000]
#  - Client 16: [500 313 556 556 500   0 385 385 500   0]
#  - Client 17: [ 500  313  556    0    0  455  385  385    0 1000]
#  - Client 18: [  0   0   0   0   0   0   0 385   0   0]
#  - Client 19: [ 500  313  556  556  500  455    0    0  500 1000]
### --- ###
# Remark of printed info
# check:
# row sum == num of data distributed to ith client
# col sum == 5000 (= 50000 / 10)
# returned array with (20, 2), 20 clients, 2 represents data position and label position
# array[5][0].shape return (#examplesOfClient5, 3, 32, 32) - data tensors
# array[5][1].shape return (#examplesOfClient5, ) - label ranging from {0...9}

def split_image_data_realwd(data, labels, n_clients=100, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds any number of classes which is trying to simulate real world dataset
    Input:
      data : [n_data x shape]
      labels : [n_data (x 1)] from 0 to n_labels(10)
      n_clients : number of clients
      verbose : True/False => True for printing some info, False otherwise
    Output:
      clients_split : splitted client data into desired format
    '''

    def break_into(n, m):
        '''
        return m random integers with sum equal to n
        '''
        to_ret = [1 for i in range(m)]
        for i in range(n - m):
            ind = random.randint(0, m - 1)
            to_ret[ind] += 1
        return to_ret

    #### constants ####
    n_classes = len(set(labels)) # n_classes = 10
    classes = list(range(n_classes)) # [0...9]
    np.random.shuffle(classes) # shuffled [0...9]
    # list with 'shape' (10, 5000)
    # 10 classes, each wtih 5000 examples, total 50000, indexes returned
    label_indcs = [list(np.where(labels == class_)[0]) for class_ in classes]

    n_labels = np.max(labels) + 1 # 10, hehe

    #### classes for each client ####
    tmp = [np.random.randint(1, 10) for i in range(n_clients)]
    total_partition = sum(tmp)

    #### create partition among classes to fulfill criteria for clients ####
    class_partition = break_into(total_partition, len(classes))

    #### applying greedy approach first come and first serve ####
    class_partition = sorted(class_partition, reverse=True) # sort in decreasing order
    class_partition_split = {}

    #### based on class partition, partitioning the label indexes ###
    # label_indc: lst with (10, 5000)
    # class_partition_split: dict, with ( i in {0..9} ) -> lst (#partition, 50000/#partion) of indexes
    for ind, class_ in enumerate(classes):
        class_partition_split[class_] = [list(i) for i in np.array_split(label_indcs[ind], class_partition[ind])]

    clients_split = []
    count = 0
    for i in range(n_clients):
        n = tmp[i] # random int from 1 to 10
        j = 0
        indcs = []

        while n > 0:
            # print(n, j, classes)
            class_ = classes[j]
            if len(class_partition_split[class_]) > 0:
                if class_ in {1, 5, 7} and len(class_partition_split[class_]) > 3:
                    count += len(class_partition_split[class_][-1])
                    class_partition_split[class_].pop()
                    n -= 1
                else:
                    indcs.extend(class_partition_split[class_][-1])
                    count += len(class_partition_split[class_][-1])
                    class_partition_split[class_].pop()
                    n -= 1
            j += 1

        ##### sorting classes based on the number of examples it has #####
        classes = sorted(classes, key=lambda x: len(class_partition_split[x]), reverse=True)
        if n > 0:
            raise ValueError(" Unable to fulfill the criteria ")
        clients_split.append([data[indcs], labels[indcs]])

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))


    if verbose:
        print_split(clients_split)

    clients_split = np.array(clients_split)

    return clients_split


# [DONE] test finished
# test case
# example: split_image_data(data_train, labels_train, 20, shuffle = True)
# Data split:
#  - Client 0: [ 98  98  98  98  98  98 104  98  98  98]
#  - Client 1: [187 187 187 187 187 187 187 187 187 190]
#  - Client 2: [481 478 478 478 478 478 478 478 478 478]
#  - Client 3: [216 216 216 225 216 216 216 216 216 216]
#  - Client 4: [374 374 374 381 374 374 374 374 374 374]
#  - Client 5: [172 172 172 172 172 172 172 172 172 177]
#  - Client 6: [364 364 372 364 364 364 364 364 364 364]
#  - Client 7: [226 226 226 234 226 226 226 226 226 226]
#  - Client 8: [414 414 414 414 414 416 414 414 414 414]
#  - Client 9: [142 142 142 142 142 142 142 142 142 151]
#  - Client 10: [69 69 69 69 69 69 69 69 69 69]
#  - Client 11: [276 276 276 276 276 277 276 276 276 276]
#  - Client 12: [128 128 128 128 128 128 128 128 128 130]
#  - Client 13: [73 73 73 73 73 82 73 73 73 73]
#  - Client 14: [448 448 448 448 455 448 448 448 448 448]
#  - Client 15: [295 295 295 303 295 295 295 295 295 295]
#  - Client 16: [355 355 355 355 355 355 355 355 355 355]
#  - Client 17: [118 121 118 118 118 118 118 118 118 118]
#  - Client 18: [310 310 310 310 310 310 310 316 310 310]
#  - Client 19: [254 254 249 225 250 245 251 251 257 238]
def split_image_data(data, labels, n_clients=100, classes_per_client=10, shuffle=True, verbose=True):
    '''
    Splits (data, labels) among 'n_clients s.t. every client can holds 'classes_per_client' number of classes
    Input:
      data : [n_data x shape]
      labels : [n_data (x 1)] from 0 to n_labels
      n_clients : number of clients
      classes_per_client : number of classes per client
      shuffle : True/False => True for shuffling the dataset, False otherwise
      verbose : True/False => True for printing some info, False otherwise
    Output:
      clients_split : client data into desired format
    '''
    #### constants ####
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    ### client distribution ####
    data_per_client = clients_rand(len(data), n_clients)
    data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)

    clients_split = np.array(clients_split)

    return clients_split

def shuffle_list(data):
  '''
  data: a 2d array with shape (#clients, 2)
  This function returns the shuffled data
  '''
  for i in range(len(data)): # #clients
    tmp_len= len(data[i][0]) # #data in client i
    index = [i for i in range(tmp_len)]
    random.shuffle(index) # you wanna do what?
    data[i][0], data[i][1] = shuffle_list_data(data[i][0],data[i][1])
  return data


def shuffle_list_data(x, y): # shuffle the data in specific client
  '''
  x, y: with shape (#data, 3, 32, 32) & (#data, )
  This function is a helper function, shuffles an
  array while maintaining the mapping between x and y
  '''
  inds = list(range(len(x))) # #data
  random.shuffle(inds)
  return x[inds], y[inds]



class CustomImageDataset(Dataset):
  '''
  A custom Dataset class for images
  inputs : numpy array [n_data x shape]
  labels : numpy array [n_data (x 1)]
  '''
  def __init__(self, inputs, labels, transforms=None):
      assert inputs.shape[0] == labels.shape[0]
      self.inputs = torch.Tensor(inputs)
      self.labels = torch.Tensor(labels).long()
      self.transforms = transforms

  def __getitem__(self, index):
      img, label = self.inputs[index], self.labels[index]

      if self.transforms is not None:
        img = self.transforms(img)

      return (img, label)

  def __len__(self):
      return self.inputs.shape[0]


def get_default_data_transforms(train=True, verbose=True):
  transforms_train = {
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),#(0.24703223, 0.24348513, 0.26158784)
  }
  transforms_eval = {
  'cifar10' : transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
  }
  if verbose:
    print("\nData preprocessing: ")
    for transformation in transforms_train['cifar10'].transforms:
      print(' -', transformation)
    print()

  return (transforms_train['cifar10'], transforms_eval['cifar10'])



def get_data_loaders(nclients, batch_size, real_wd, classes_pc=10, verbose=True):

    x_train, y_train, x_test, y_test = get_cifar10()

    if verbose:
        print_image_data_stats(x_train, y_train, x_test, y_test)

    transforms_train, transforms_eval = get_default_data_transforms(verbose=False)

    if real_wd:
        split = split_image_data_realwd(x_train, y_train, n_clients=nclients, verbose=verbose)
    else:
        split = split_image_data(x_train, y_train, n_clients=nclients,
                                 classes_per_client=classes_pc, verbose=verbose)

    data_size_per_client = np.array([split[i][0].shape[0] for i in range(nclients)])

    split_tmp = shuffle_list(split) # Q: why you shuffle here?

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transforms_train),
                                                  batch_size=batch_size, shuffle=True) for x, y in split_tmp] # Q: so you shuffle again?

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transforms_eval), batch_size=100,
                                              shuffle=False)

    return client_loaders, test_loader, data_size_per_client



