#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np

import random
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from options import args_parser
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset

from dataprocessing import *
import math
import sys
import torch.optim as optim
from PreResNet import *
from pathlib import Path
from noise_generation import *


args = args_parser()
args.method = 'CLC'
rootPath0 = 'saveDicts/'+str(args.dataset)
Path(os.path.join(rootPath0,)).mkdir(parents=True,exist_ok=True)
args.save_dicts_dir = os.path.join(rootPath0,args.method+'/noise_'+str(args.noise_rate))
Path(os.path.join(args.save_dicts_dir,)).mkdir(parents=True,exist_ok=True)

rootPath = 'data/'+args.dataset+'/noise_' + str(args.noise_rate)
serverPath = os.path.join(rootPath, 'server')
clientPath = os.path.join(rootPath, 'client')

all_noise_chart = []
noise_chart = []

class Server:
    def __init__(self,model,data_dir):
        self.model = model
        self.data_dir = data_dir
        with open(os.path.join(data_dir,'server/testdata.pkl'),'rb') as f:
            test_dataset = joblib.load(f)
        self.test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)
        with open(os.path.join(data_dir,'server/benchdata.pkl'),'rb') as f:
            bench_dataset = joblib.load(f)
        self.bench_loader = DataLoader(bench_dataset,batch_size=args.local_bs,shuffle=True)
        self.class_nums_each = [[] for i in range(args.num_users)]
        self.conflist_each = [[] for i in range(args.num_users)]

    def test(self,epoch):
        self.model.eval()
        correct = 0
        total = 0
        LOSS = 0
        criterion = nn.CrossEntropyLoss().to(args.device)

        with torch.no_grad():
            for batch_ixx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, targets)
                LOSS += loss

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        LOSS = LOSS / len(self.test_loader)


        print("\n| Global Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc))

    def receiveconf(self):
        for ix in range(args.num_users):
            with open(os.path.join(args.save_dicts_dir,'confidence_%d.pkl'%(ix)),'rb') as f:
                self.conflist_each[ix] = pickle.load(f)
            with open(os.path.join(args.save_dicts_dir, 'classnums_%d.pkl' % (ix)), 'rb') as f:
                self.class_nums_each[ix] = pickle.load(f)

    def conf_agg(self):
        conf_score = [0] * args.num_classes
        conf_wt = [[0] * args.num_users for i in range(args.num_classes)]
        class_nums = np.array(self.class_nums_each)
        sum_col = class_nums.sum(axis=0)
        for ix in range(args.num_users):
            for i in range(args.num_classes):
                denom = sum_col[i]
                nom = self.class_nums_each[ix][i]
                w = nom / denom
                conf_wt[i][ix] = w

            if ix == args.num_users - 1:

                for i in range(args.num_classes):
                    for j in range(args.num_users):
                        conf_score[i] += conf_wt[i][j] * self.conflist_each[j][i]
        return conf_score

    def pre_train(self,max_epochs=50):
        self.model.train()
        criterion = nn.CrossEntropyLoss().to(args.device)
        num_iter = (len(self.bench_loader.dataset) // self.bench_loader.batch_size) + 1
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.local_mom, weight_decay=args.local_decay)
        for epoch in range(max_epochs):
            for batch_idx, (inputs, labels) in enumerate(self.bench_loader):
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                sys.stdout.write('\r')
                sys.stdout.write('Server  | Pre-train | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                                 % (epoch, max_epochs, batch_idx + 1, num_iter,
                                    loss.item()))
                sys.stdout.flush()

            self.test(epoch)
            torch.save(self.model.state_dict(), os.path.join(rootPath0,
                                                               'benchmodel.json'))

    def FedAvg(self,w, userdatlen):
        w_avg = copy.deepcopy(w[0])
        sum = np.sum(userdatlen)
        for key in w_avg.keys():
            for i in range(len(w)):
                a = w[i][key]
                b = userdatlen[i] / sum
                tensor = torch.mul(w[i][key], userdatlen[i] / sum)
                if i == 0:
                    w_avg[key] = tensor.type(w[i][key].dtype)
                else:
                    w_avg[key] += tensor.type(w[i][key].dtype)
        self.model.load_state_dict(w_avg)

    def sendmodel(self):
        torch.save(self.model.state_dict(), os.path.join(args.save_dicts_dir,
                                                'servermodel.json' ))



class Client:
    def __init__(self,client_id,model,tao):
        self.tao = tao
        self.model = model
        self.data_dir = rootPath
        with open(os.path.join(rootPath,'client/dataset'+str(client_id)+'.pkl'),'rb') as f:
            self.dataset = joblib.load(f)
        self.data_loader = DataLoader(self.dataset,shuffle=True,batch_size=args.local_bs)
        self.client_id = client_id
        self.sfm_Mat = None
        self.avai_dataset = None
        self.keys = []
        self.sudo_labels = []

    def receivemodel(self):
        torchLoad = torch.load(os.path.join(args.save_dicts_dir,
                                                'servermodel.json'))
        self.model.load_state_dict(torchLoad)

    def sendconf(self):
        confListU, class_nums = self.confidence()
        sys.stdout.write('\r')
        sys.stdout.write('User = [%d/%d]  | confidence is computed '
                         % (self.client_id, args.num_users))
        sys.stdout.flush()
        with open(os.path.join(args.save_dicts_dir, 'confidence_%d.pkl' % (self.client_id)), 'wb') as f:
            pickle.dump(confListU, f)
        with open(os.path.join(args.save_dicts_dir, 'classnums_%d.pkl' % (self.client_id)), 'wb') as f:
            pickle.dump(class_nums, f)

    def outputSof(self):

        dataset = self.dataset
        s = dataset.labels
        s = np.array(s)

        self.model.eval()
        val_loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=False)
        outputs = []
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(val_loader):
                data, labels = data.to(args.device), labels.to(args.device)
                if len(outputs) == 0:
                    outputs = self.model(data)
                else:
                    outputs = torch.cat([outputs, self.model(data)], dim=0)

        psx_cv = F.softmax(outputs, dim=1)

        psx = psx_cv.cpu().numpy().reshape((-1, args.num_classes))
        s = s.reshape([s.size, 1])
        sfm_Mat = np.hstack((psx, s))

        return sfm_Mat

    def confidence(self):

        outputSofma = self.outputSof()
        r = outputSofma.shape[0]
        c = outputSofma.shape[1]
        prob_everyclass = [[] for i in range(c - 1)]
        class_nums = []
        confList = []
        for i in range(r):
            oriL = outputSofma[i][c - 1]
            oriL = int(oriL)
            pro = outputSofma[i, oriL]
            prob_everyclass[oriL].append(pro)

        for i in range(c - 1):
            confList.append(round(np.mean(prob_everyclass[i], axis=0), 3))
            class_nums.append(len(prob_everyclass[i]))
        self.sfm_Mat = outputSofma
        return confList, class_nums

    def data_holdout(self,conf_score):

        r = self.sfm_Mat.shape[0]
        delta_sort = {}
        naive_num = 0
        self.keys = []
        self.sudo_labels = []
        for idx in range(r):
            if idx == 26:
                debug = True
            softmax = self.sfm_Mat[idx]

            maxPro_Naive = -1
            preIndex_Naive = -1
            maxPro = -1
            preIndex = -1
            for j in range(args.num_classes):
                if softmax[j] > maxPro_Naive:
                    preIndex_Naive = j
                    maxPro_Naive = softmax[j]

                if softmax[j] > conf_score[j]:
                    if softmax[j] > maxPro:
                        maxPro = softmax[j]
                        preIndex = j

            label = int(softmax[-1])
            margin = maxPro_Naive - softmax[label]

            if preIndex == -1:
                preIndex = preIndex_Naive
                maxPro = maxPro_Naive
                naive_num += 1
            elif preIndex != label:
                delta_sort[idx] = margin

            self.sudo_labels.append(preIndex)

        delta_sorted = sorted(delta_sort.items(), key=lambda delta_sort: delta_sort[1], reverse=True)  # 降序
        reserve = []
        for (k, v) in delta_sorted:
            if v > self.tao:
                self.keys.append(k)

        for idx in range(r):
            if idx not in self.keys:
                reserve.append(idx)
        data = self.dataset.samples
        labels = self.dataset.labels
        data = data[reserve]
        labels = labels[reserve]
        self.avai_dataset = data_loader(data, labels, args)
        self.data_loader = DataLoader(self.avai_dataset,batch_size=args.local_bs,shuffle=True)

    def data_correct(self):
        data = self.dataset.samples
        self.avai_dataset = data_loader(data, self.sudo_labels, args)
        self.data_loader = DataLoader(self.avai_dataset, batch_size=args.local_bs, shuffle=True)

    def update_weights(self,global_ep, epoch):
        self.model.train()
        criterion = nn.CrossEntropyLoss().to(args.device)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.local_mom,
                                    weight_decay=args.local_decay)
        num_iter = (len(self.data_loader.dataset) // self.data_loader.batch_size) + 1

        for batch_idx, (inputs, labels) in enumerate(self.data_loader):
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            sys.stdout.write('\r')
            sys.stdout.write('User = %d  | Global Epoch %d | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.4f'
                             % (self.client_id, global_ep, epoch, args.local_ep, batch_idx + 1, num_iter,
                                loss.item()))
            sys.stdout.flush()

    def local_update(self,epoch):
        for iter in range(args.local_ep):
            self.update_weights(epoch, iter)
        return self.model.state_dict(),len(self.data_loader.dataset)

class CLC:
    def __init__(self,tao=0.1):
        self.tao = tao
        self.model = None
        self.ini_model()
        self.data_dir = rootPath
        self.clients = []
        for p_id in range(args.num_users):
            self.clients.append(Client(p_id, copy.deepcopy(self.model), self.tao))
        self.server = Server(self.model,self.data_dir)

        if args.benchmark:
            if not os.path.exists(os.path.join(rootPath0,'benchmodel.json')):
                self.server.pre_train()
            self.server.model.load_state_dict(torch.load(os.path.join(rootPath0,'benchmodel.json')))
        else:
            self.warmup()
        self.server.sendmodel()
        for ix in range(args.num_users):
            self.clients[ix].receivemodel()

    def create_model(self):
        model = ResNet18(num_classes=args.num_classes)
        model = model.to(args.device)
        return model

    def ini_model(self):
        if args.dataset == 'cifar10':
            self.model = self.create_model()

    def warmup(self):

        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]

        for ix in range(args.num_users):
            model_param[ix], Keep_size[ix] = self.clients[ix].local_update(-1)

        self.server.FedAvg(model_param, Keep_size)


    def holdout_stage(self):

        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]
        for epoch in range(args.first_epochs):

            for ix in range(args.num_users):
                self.clients[ix].sendconf()
            self.server.receiveconf()
            conf_score = self.server.conf_agg()
            for ix in range(args.num_users):
                self.clients[ix].data_holdout(conf_score)
                model_param[ix],Keep_size[ix] = self.clients[ix].local_update(epoch)

            self.server.FedAvg(model_param, Keep_size)
            self.server.sendmodel()
            self.server.test(epoch)
            for ix in range(args.num_users):
                self.clients[ix].receivemodel()

            if epoch %10==0:
                torch.save(self.server.model.state_dict(), os.path.join(args.save_dicts_dir,
                                                        'global_train_%d.json' % (epoch)))

    def correct_stage(self):
        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]
        correct_done = False
        for epoch in range(args.first_epochs,args.first_epochs+args.last_epochs):
            if not correct_done:
                for ix in range(args.num_users):
                    self.clients[ix].sendconf()
                self.server.receiveconf()
                conf_score = self.server.conf_agg()
                for ix in range(args.num_users):
                    self.clients[ix].data_holdout(conf_score)
                    self.clients[ix].data_correct()
                correct_done = True


            for ix in range(args.num_users):
                model_param[ix],Keep_size[ix] = self.clients[ix].local_update(epoch)

            self.server.FedAvg(model_param, Keep_size)
            self.server.sendmodel()
            self.server.test(epoch)
            for ix in range(args.num_users):
                self.clients[ix].receivemodel()

            if epoch%10==0:
                torch.save(self.server.model.state_dict(), os.path.join(args.save_dicts_dir,
                                                        'global_train_%d.json' % (epoch)))



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def make_noise(oriset):

    uni_nm = len(oriset)


    labels = oriset.labels
    images = oriset.samples
    class_nums = Counter(labels)
    py = []
    for i in range(args.num_classes):
        py.append(class_nums[i]/uni_nm)

    trace = (1 - args.noise_rate) * args.num_classes
    mat = generate_noise_matrix_from_trace(args.num_classes, trace=trace,py=py)
    new_labels = generate_noisy_labels(labels,mat)
    global all_noise_chart
    all_noise_chart = [new_labels[i]!=labels[i] for i in range(uni_nm)]
    all_noise_chart = np.array(all_noise_chart)
    newset = data_loader(images, new_labels)
    noise_cnt = sum(new_labels[i]!=labels[i] for i in range(uni_nm))
    return newset


def bench_left(train_dataset):
    bench_idxs = bench_assign(train_dataset)
    X,Y = train_dataset.samples[bench_idxs],train_dataset.labels[bench_idxs]
    idxs_all = list(range(len(train_dataset)))
    idxs_left = list(set(idxs_all)-set(bench_idxs))
    X_left,Y_left = train_dataset.samples[np.array(idxs_left)],train_dataset.labels[np.array(idxs_left)]
    left_dataset = data_loader(X_left,Y_left)
    bench_dataset = data_loader(X,Y,args)
    return  left_dataset,bench_dataset


def prepare_data():


    train_dataset, test_dataset = get_dataset(args)


    left_dataset, bench_dataset = bench_left(train_dataset)
    new_dataset = make_noise(left_dataset)
    Usr_dataset, noise_chart = get_Users_Data(args, new_dataset, noise_lst=all_noise_chart)

    Path(clientPath).mkdir(parents=True, exist_ok=True)
    Path(serverPath).mkdir(parents=True, exist_ok=True)
    for p_id in range(args.num_users):
        with open(os.path.join(clientPath, 'dataset' + str(p_id) + '.pkl'), 'wb') as f:
            pickle.dump(Usr_dataset[p_id], f)
    with open(os.path.join(serverPath, 'testdata.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)
    with open(os.path.join(serverPath, 'benchdata.pkl'), 'wb') as f:
        pickle.dump(bench_dataset, f)


if __name__ == '__main__':
    prepare_data()
    clc = CLC(tao=0.1)
    clc.holdout_stage()
    clc.correct_stage()



