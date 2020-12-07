import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim


from dataSplit import get_cifar10, get_default_data_transforms, CustomImageDataset

# GPU settings
torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# [DONE] test finished
# return x_aux: (320, 3, 32, 32)
# y_aux:
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
#        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
#        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
###
###
# extract 10 * 32 data from testset, for 10 classes with each class of 32 examples
# here I don't seperate axuiliary dataset from testset due to its small scale
def get_auxiliary_data(testset_extract = True, data_size = 32):
	''' return the numpy array of auxiliary data, x: (320, 3, 32, 32), y: (320, ) (class labels from 0 to 9 sorted)'''
	if testset_extract:
		_, _, x, y = get_cifar10()
	else:
		x, y, _, _ = get_cifar10()

	x_aux, y_aux = [], []

	for i in range(10):
		for (sample, label) in zip(x, y):
			if (label == i) and (len(x_aux) + 1 <= data_size * (i + 1)):
				x_aux.append(sample)
				y_aux.append(label)

	x_aux = np.array(x_aux)
	y_aux = np.array(y_aux)

	return x_aux, y_aux


# [DONE] test finished
# the output of the following statements:
# for (input, target) in aux_loader:
#     print(input.shape, target)
# is here:
# torch.Size([32, 3, 32, 32]) tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#         0, 0, 0, 0, 0, 0, 0, 0])
# torch.Size([32, 3, 32, 32]) tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#         1, 1, 1, 1, 1, 1, 1, 1])
# torch.Size([32, 3, 32, 32]) tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#         2, 2, 2, 2, 2, 2, 2, 2])
# torch.Size([32, 3, 32, 32]) tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#         3, 3, 3, 3, 3, 3, 3, 3])
# torch.Size([32, 3, 32, 32]) tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#         4, 4, 4, 4, 4, 4, 4, 4])
# torch.Size([32, 3, 32, 32]) tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#         5, 5, 5, 5, 5, 5, 5, 5])
# torch.Size([32, 3, 32, 32]) tensor([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#         6, 6, 6, 6, 6, 6, 6, 6])
# torch.Size([32, 3, 32, 32]) tensor([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
#         7, 7, 7, 7, 7, 7, 7, 7])
# torch.Size([32, 3, 32, 32]) tensor([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#         8, 8, 8, 8, 8, 8, 8, 8])
# torch.Size([32, 3, 32, 32]) tensor([9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
#         9, 9, 9, 9, 9, 9, 9, 9])
def get_auxiliary_data_loader(testset_extract = True, data_size = 32):
	x_aux, y_aux = get_auxiliary_data(testset_extract, data_size)

	# No matter you choose trainset or testset to generate auxiliary data
	# use the transforms on testdata
	_, transforms = get_default_data_transforms(verbose=False)

	# batch_size = data_size, don't shuffle
	auxiliary_data_loader = torch.utils.data.DataLoader(
		CustomImageDataset(x_aux, y_aux, transforms),
		batch_size=data_size, shuffle=False
		)

	return auxiliary_data_loader


def compute_grad_aux(global_model, aux_loader):
	'''our own version of ratio'''
	optimizer = optim.SGD(global_model.parameters(), lr = 0.1) # lr doesn't matter here

	grad_square_sum_lst = [0] * 10


	for class_i_data_idx, (input, target) in enumerate(aux_loader):

		# 6. zero gradient, otherwise accumulated
		optimizer.zero_grad()

		# 1. prepare (X, Y) which belongs to same class
		# for ith class's data, with shape of [32, 3, 32, 32] batch size pf 32, with image 3*32*32
		# for ith class'labels, don't need one-hot coding here
		input, target = input.to(device), target.to(device)

		# 2. forward pass get Y_out
		output = global_model(input) # feed ith class data, network output with shape [32, 10]
                               		 # the output is logits (before softmax), usually output logits as default

		# 3. calculate cross-entropy between Y and Y_out
		loss = F.cross_entropy(output, target)

		# 4. backward pass to get gradients wrt weight using a batch of 32 of data
		loss.backward()

		# 5. record gradients wrt weights from the last layer
    	# print(cifarnet.fc2.weight.grad.shape) here is [500, 10]
    	# here we only need fetch ith gradient with shape [500]
    	# In short, send data from ith class, fetch ith gradient tensor from [500, 10]
		# grad_lst.append(global_model.fc2.weight.grad[class_i_data_idx].cpu().numpy())
		# new
		# the above descripting all wrong
		for name, param in global_model.named_parameters():
			# print(name, param.grad.shape)
			# print((param.grad ** 2).sum().item())
			grad_square_sum_lst[class_i_data_idx] += ((param.grad ** 2)).mean().item()


	return grad_square_sum_lst


# threshold changed the meaning to percentage
# threshold doesn't work here
# grad_square_sum_lst: [10, ]
def compute_ratio(grad_square_sum_lst, temp = 1):
    ''' original version in the paper '''

    grad_sum = np.array(grad_square_sum_lst)
    # print(grad_sum)

    grad_sum = grad_sum.min() / grad_sum
    # print(grad_sum)

    # def softmax(grad_sum, temp = 1):
    #     grad_sum = grad_sum - grad_sum.mean()
    #     return np.exp(grad_sum / temp) / np.exp(grad_sum / temp).sum()

    # grad_sum_normalize = softmax(grad_sum, temp)
    grad_sum_normalize = grad_sum / grad_sum.sum()
    # grad_sum_normalize = grad_sum
    

    return grad_sum_normalize

def compute_ratio_per_client_update(client_models, client_idx, aux_loader):
    ra_dict = {}
    for i, client_model_update in enumerate(client_models):
        grad_square_sum_lst = compute_grad_aux(client_model_update, aux_loader)
        grad_sum_normalize = compute_ratio(grad_square_sum_lst)
        ra_dict[client_idx[i]] = grad_sum_normalize

    return ra_dict