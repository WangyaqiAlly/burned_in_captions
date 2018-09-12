#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import autograd
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader

from sampling import mnist_iid, mnist_noniid, cifar_iid
from options import args_parser
from Update import LocalUpdate
from FedNets import MLP, CNNMnist, CNNCifar
from averaging import average_weights


def test(net_g, data_loader, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data), autograd.Variable(target)
        log_probs = net_g(data)
        test_loss += F.nll_loss(log_probs, target, size_average=False).data[0] # sum up batch loss
        y_pred = log_probs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    ndata = len(data_loader.dataset)
    test_loss /= ndata
    test_acc = 100.*correct/ndata

    print('Test set: Average loss: {:.4f} Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, ndata, test_acc))

    return test_acc, test_loss

if __name__ == '__main__':
    # parse args
    args = args_parser()
    torch.manual_seed(args.seed)

    summary = SummaryWriter('local')

    # load dataset and split users
    if args.dataset == 'mnist':
	transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ), (0.3081, ))])

        dataset_train = datasets.MNIST('../data/mnist/', train=True, 
					download=True,
					transform=transform)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)

        dataset_test = datasets.MNIST('../data/mnist/', train=False,
                                       download=True,
                                       transform=transform)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    elif args.dataset == 'cifar':
	transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        dataset_train = datasets.CIFAR10('../data/cifar', train=True, 
					  transform=transform, 
					  target_transform=None, 
					  download=True)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
        
        dataset_test = datasets.CIFAR10('../data/cifar', train=False,
                                        transform=transform,
                                        target_transform=None,
                                        download=True)

        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    else:
        exit('Error: unrecognized dataset')

    img_size = dataset_train[0][0].shape
    # print 'dataset_train: ', dataset_train.shape
    # print 'dataset_test: ', dataset_test.shape
    print 'image size:', img_size
    print('training on', len(dataset_train), 'samples')
    print('testing  on', len(dataset_test), 'samples')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNCifar(args=args).cuda()
        else:
            net_glob = CNNCifar(args=args)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = CNNMnist(args=args).cuda()
        else:
            net_glob = CNNMnist(args=args)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        if args.gpu != -1:
            torch.cuda.set_device(args.gpu)
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).cuda()
        else:
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    ep = 0
    loss_train = []
    cv_loss, cv_acc = [], []
    test_loss_list = []
    test_acc_list = []
    val_loss_pre, counter = 0, 0
    net_best = None
    val_acc_list, net_list = [], []
    for iter in tqdm(range(args.epochs)):
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
            w, loss = local.update_weights(net=copy.deepcopy(net_glob))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = average_weights(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        print 'Epoch %3d: Train loss %.2f' % (len(loss_train), loss_avg)

        # test on testing dataset
#        net_glob.eval()
 	test_acc, test_loss = test(net_glob, test_loader, args)
	test_acc_list.append(test_acc)
	test_loss_list.append(test_loss)
	print 'Epoch %3d: Testing Accuracy %6.2f' % (ep, test_acc)

	ep += 1

    'save to txtfile'
    K = args.num_users
    C = args.frac
    txtfile = 'fed_{}_{}_{}_K{}_C{}_iid{}.txt'.format(args.dataset, args.model, args.epochs, K, C, args.iid)
    with open(txtfile, "w") as f:
	for i, acc in enumerate(test_acc_list): 
            f.write("{:6d}  {:6.2f}\n".format(i, acc))

    # plot loss curve
    plt.figure()
    plt.subplot(211)
    plt.plot(range(len(loss_train)), loss_train)
    plt.plot(range(len(test_loss_list)), test_loss_list)
    plt.ylabel('train_loss')
    plt.subplot(212)
    plt.plot(test_acc_list)
    plt.savefig('../save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    list_acc, list_loss = [], []
#    net_glob.eval()
    for c in tqdm(range(args.num_users)):
        net_local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[c], tb=summary)
        acc, loss = net_local.test(net=net_glob)
        list_acc.append(acc)
        list_loss.append(loss)
    print("final average acc: {:.2f}%".format(100.*sum(list_acc)/len(list_acc)))

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader, args)

    '''
    # actual testing
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307, ), (0.3081, ))])

        dataset_test = datasets.MNIST('../data/mnist/', train=False,
                                       download=True,
                                       transform=transform)
        
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        dataset_test = datasets.CIFAR10('../data/cifar', train=False,
                                        transform=transform,
                                        target_transform=None,
                                        download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    else:
        exit('Error: unrecognized dataset')
    '''
    
#     print('test on', len(dataset_test), 'samples')
#     test_acc, test_loss = test(net_glob, test_loader, args)

