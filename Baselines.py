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
from operator import itemgetter
from scipy import stats
import joblib

from collections import Counter
from dataprocessing import *
import math
import sys
import torch.optim as optim
from PreResNet import *
from pathlib import Path
from noise_generation import *

args = args_parser()
rootPath0 = 'saveDicts/'+str(args.dataset)
Path(os.path.join(rootPath0,)).mkdir(parents=True,exist_ok=True)
args.save_dicts_dir = os.path.join(rootPath0,args.method+'/'+'noise='+str(args.noise_rate))
Path(os.path.join(args.save_dicts_dir,)).mkdir(parents=True,exist_ok=True)

rootPath = 'data/'+args.dataset+'/noise_' + str(args.noise_rate)
serverPath = os.path.join(rootPath, 'server')
clientPath = os.path.join(rootPath, 'client')

all_noise_chart = []
noise_chart = []

class Server:
    def __init__(self,model,datapath,mode):
        self.mode = mode
        self.model = model
        self.data_dir = datapath
        with open(os.path.join(self.data_dir,'testdata.pkl'),'rb') as f:
            test_dataset = joblib.load(f)
        self.test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)
        with open(os.path.join(self.data_dir,'benchdata.pkl'),'rb') as f:
            bench_dataset = joblib.load(f)
        self.bench_loader = DataLoader(bench_dataset,batch_size=args.local_bs,shuffle=True)
        if self.mode == 'DS':
            all_idxs = [i for i in range(len(bench_dataset))]
            idx_train = np.random.choice(all_idxs, int(len(bench_dataset) / 10)*7,
                                         replace=False)
            idx_test = np.array(list(set(all_idxs) - set(idx_train)))

            x_bench = bench_dataset.samples
            y_bench = bench_dataset.labels
            train_bench = data_loader(x_bench[idx_train], y_bench[idx_train], args)
            test_bench = data_loader(x_bench[idx_test], y_bench[idx_test], args)
            self.bench_trloader = DataLoader(train_bench, batch_size=args.local_bs, shuffle=True)
            self.bench_tstloader = DataLoader(test_bench, batch_size=1, shuffle=True)
            savepath = os.path.join(rootPath0,'benchmodel_frombentr.json')
        else:
            savepath = os.path.join(rootPath0, 'benchmodel.json')

        if args.benchmark:
            if not os.path.exists(savepath):
                print(savepath)
                self.pre_train(savepath=savepath)
            self.model.load_state_dict(torch.load(savepath))
        self.sendmodel()

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

    def pre_train(self,max_epochs=50,savepath=''):
        self.model.train()
        criterion = nn.CrossEntropyLoss().to(args.device)

        if self.mode == 'DS':
            loader = self.bench_trloader
        else:
            loader = self.bench_loader
        num_iter = (len(loader.dataset) // loader.batch_size) + 1
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.local_mom, weight_decay=args.local_decay)
        for epoch in range(max_epochs):
            for batch_idx, (inputs, labels) in enumerate(loader):
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

            self.test(epoch-max_epochs)
        torch.save(self.model.state_dict(),savepath )

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
    def cmp_bench_loss(self):
        device = args.device
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        bench_loss = []
        for idx, (data, label) in enumerate(self.bench_tstloader):
            data, label = data.to(device), label.to(device)
            outputs = self.model(data)
            loss = criterion(outputs, label)
            bench_loss.extend(loss.tolist())
        return np.array(bench_loss)



class Client:
    def __init__(self,client_id,model,datapath,mode):
        self.mode = mode
        self.model = model
        self.data_dir = datapath
        with open(os.path.join(self.data_dir,'dataset'+str(client_id)+'.pkl'),'rb') as f:
            self.dataset = joblib.load(f)
        self.data_loader = DataLoader(self.dataset,shuffle=True,batch_size=args.local_bs)
        self.client_id = client_id
        self.sfm_Mat = None
        self.avai_dataset = None
        self.keys = []
        self.sudo_labels = []
        self.conf = []

    def receivemodel(self):
        torchLoad = torch.load(os.path.join(args.save_dicts_dir,
                                                'servermodel.json'))
        self.model.load_state_dict(torchLoad)


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
        self.conf = confList

    def data_filter(self):
        conf_score = self.conf
        r = self.sfm_Mat.shape[0]
        sudo_labels = []
        naive_num = 0
        keys = []
        reserve = []
        for idx in range(r):
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
                keys.append(idx)
            sudo_labels.append(preIndex)

        for idx in range(r):
            if idx not in keys:
                reserve.append(idx)
        data = self.dataset.samples
        labels = self.dataset.labels
        data = data[reserve]
        labels = labels[reserve]
        self.avai_dataset = data_loader(data, labels, args)
        self.data_loader = DataLoader(self.avai_dataset, batch_size=args.local_bs, shuffle=True)

    def reference(self):

        self.model.eval()
        device = args.device
        criterion = nn.CrossEntropyLoss(reduction='none').to(device)
        B_predits = []
        Loss = []
        for batch_idx, (data, label) in enumerate(self.data_loader):
            data, label = data.to(device), label.to(device)

            outputs = self.model(data)

            loss = criterion(outputs, label)
            Loss.extend(loss.tolist())
            _, predits = torch.max(outputs, 1)
            predits = predits.cpu().detach().numpy()
            B_predits.extend(predits)
        Loss = np.array(Loss)
        self.local_loss = Loss
        return B_predits, Loss

    def lamda1_detect(self, lamda):
        avai_idx = []
        nonavai_idx = []

        for i, v in enumerate(self.local_loss):
            if v <= lamda:
                avai_idx.append(i)
            else:
                nonavai_idx.append(i)

        x = self.dataset.samples
        y = self.dataset.labels
        newx = x[avai_idx]
        newy = y[avai_idx]

        self.avai_dataset = data_loader(newx, newy, args)
        self.data_loader = DataLoader(self.avai_dataset, batch_size=args.local_bs, shuffle=True)

    def one_hot_embedding(self, labels, num_classes=10):
        y = torch.eye(num_classes)
        neg = labels < 0  # negative labels
        labels[neg] = 0  # placeholder label to class-0
        y = y[labels]  # create one hot embedding
        y[neg, 0] = 0  # remove placeholder label
        return y

    def model_train_val(self, epoch, train_set, val_set):
        model = copy.deepcopy(self.model)
        device = args.device
        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        criterion_red = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.local_mom,
                                    weight_decay=args.local_decay)
        batVerbose = 9
        batch_nm = int(len(train_set.dataset) / args.local_bs)
        model.train()
        bestacc = 0
        acc_ls = []
        for iter in range(epoch):

            print('Epoch {}/{}'.format(iter + 1, epoch))
            print('-' * 10)
            batch_loss = []
            running_corrects = 0
            model.train()
            for batch_idx, (data, label) in enumerate(train_set):
                data, label = data.to(device), label.to(device)
                model.zero_grad()
                outputs = model(data)
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == label).item()
                lossback = criterion_red(outputs, label)

                lossback.backward()
                optimizer.step()
                if batch_idx % batVerbose == 0:
                    print('| Server Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(data),
                        len(train_set.dataset), 100. * batch_idx / batch_nm, lossback.item()))

                batch_loss.append(lossback.item())
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(val_set):
                    data, label = data.to(device), label.to(device)
                    outputs = model(data)
                    _, preds = torch.max(outputs.data, 1)
                    cross_entropy = criterion_red(outputs, label)
                    correct += torch.sum(torch.eq(preds, label)).item()
            val_acc = correct / len(val_set.dataset)
            acc_ls.append(val_acc)
            if val_acc > bestacc:
                bestacc = val_acc
                bestmodel = copy.deepcopy(model)

        # ----------------------

        # Select samples of 'True' prediction
        bestmodel.eval()
        PREDS = []
        CROSS_ENTROPY = []
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(val_set):
                data, label = data.to(device), label.to(device)
                log_probs = bestmodel(data)
                _, preds = torch.max(log_probs.data, 1)
                preds_ls = preds.tolist()
                PREDS.extend(preds_ls)
                cross_entropy = criterion(log_probs, label)
                CROSS_ENTROPY.extend(cross_entropy.tolist())

        return PREDS, CROSS_ENTROPY

    def INCV(self):
        """ parameters """

        Num_top = 1

        INCV_epochs = 50
        INCV_iter = 4


        #################################################################################################################################
        """ Data preparation """

        x_train = self.dataset.samples
        y_train_noisy_ori = np.array(self.dataset.labels)
        y_train_noisy = self.one_hot_embedding(np.array(y_train_noisy_ori))
        input_shape = list(x_train.shape[1:])
        n_classes = args.num_classes
        n_train = x_train.shape[0]

        #################################################################################################################################
        """ Build model """

        ##################################################################################################################################
        """ INCV iteration """
        train_idx = np.array([False for i in range(n_train)])
        val_idx = np.array([True for i in range(n_train)])

        for ITER in range(1, INCV_iter + 1):
            print('INCV iteration %d including first half and second half. In total %d iterations.' % (
            ITER, INCV_iter))
            val_idx_int = np.array([i for i in range(n_train) if val_idx[i]])  # integer index
            np.random.shuffle(val_idx_int)
            n_val_half = int(np.sum(val_idx) / 2)
            val1_idx = val_idx_int[:n_val_half]  # integer index
            val2_idx = val_idx_int[n_val_half:]  # integer index
            # candid_set1 =
            candid_set1 = DataLoader(data_loader(x_train[val1_idx, :], y_train_noisy_ori[val1_idx], args),
                                     batch_size=args.local_bs, shuffle=True, num_workers=10)
            # candid_set2 =
            candid_set2 = DataLoader(data_loader(x_train[val2_idx, :], y_train_noisy_ori[val2_idx], args),
                                     batch_size=args.local_bs, shuffle=True, num_workers=10)
            # Train model on the first half of dataset

            First_half = True
            print('Iteration ' + str(ITER) + ' - first half')
            # reset weights

            # --------------------------------------------------

            y_pred, cross_entropy = self.model_train_val(epoch=INCV_epochs, train_set=candid_set1,
                                                         val_set=candid_set2,
                                                         )


            y_true_noisy = np.argmax(y_train_noisy[val2_idx, :], axis=1)
            top_True = [y_true_noisy[i] == y_pred[i] for i in range(len(y_true_noisy))]
            # --------------------------------------------------

            val2train_idx = val2_idx[top_True]

            if ITER == 1:
                eval_ratio = 0.001
                product = np.sum(top_True) / (n_train / 2.)
                while (1 - eval_ratio) * (1 - eval_ratio) + eval_ratio * eval_ratio / (
                        n_classes / Num_top - 1) > product:
                    eval_ratio += 0.001
                    if eval_ratio >= 1:
                        break
                print('noisy ratio evaluation: %.4f\n' % eval_ratio)
                discard_ratio = min(2, eval_ratio / (1 - eval_ratio))
                discard_idx = val2_idx[
                    np.argsort(np.array(cross_entropy))[-int(discard_ratio * np.sum(top_True)):]]

            else:
                discard_idx = np.concatenate([discard_idx, val2_idx[
                    np.argsort(np.array(cross_entropy))[-int(discard_ratio * np.sum(top_True)):]]])

            print('%d samples selected\n' % (np.sum(train_idx) + val2train_idx.shape[0]))

            print('Iteration ' + str(ITER) + ' - second half')

            y_pred, cross_entropy = self.model_train_val(epoch=INCV_epochs, train_set=candid_set2,
                                                         val_set=candid_set1,
                                                         )

            del candid_set1, candid_set2
            gc.collect()

            y_true_noisy = np.argmax(y_train_noisy[val1_idx, :], axis=1)
            top_True = [y_true_noisy[i] == y_pred[i] for i in range(len(y_true_noisy))]

            val2train_idx = np.concatenate([val1_idx[top_True], val2train_idx])
            discard_idx = np.concatenate(
                [discard_idx, val1_idx[
                    np.argsort(np.array(cross_entropy))[-int(discard_ratio * np.sum(top_True)):]]])
            train_idx[val2train_idx] = True
            val_idx[val2train_idx] = False

            iter_save_best = 1

            if ITER == iter_save_best:
                INCV_save_best = False

        ##################################################################################################################################
        train_idx_int = np.array([i for i in range(n_train) if train_idx[i]])

        self.dataset = data_loader(x_train[train_idx_int, :], y_train_noisy_ori[train_idx_int], args)
        self.data_loader = DataLoader(self.dataset,batch_size=args.local_bs,shuffle=True)
        with open(os.path.join(self.data_dir, 'INCVset'+str(self.client_id)+'.pkl'), 'wb') as f:
            joblib.dump(self.dataset, f)

    def sce_loss(self, outputs, labels, reduce=True):
        sfm_probs = F.softmax(outputs)  # 添加激活层取正数
        loss_ce = F.nll_loss(F.log_softmax(outputs), labels, reduce=reduce)
        q = self.one_hot_embedding(labels)
        q = q.to(args.device)
        for i in range(len(q)):
            for j in range(len(q[i])):
                if int(q[i][j]) == 1:
                    q[i][j] = 0
                else:
                    q[i][j] = -4

        multi = q.mul(sfm_probs)
        # np.multiply(q,sfm_probs.cpu().detach().numpy())
        sum_Forrow = torch.sum(multi, dim=1)
        if reduce:
            rce = torch.mean(sum_Forrow, dim=0)
        else:
            rce = sum_Forrow
        loss_rce = (-1) * rce
        loss = 0.01 * loss_ce.detach() + loss_rce
        return loss.to(args.device)


    def update_weights(self,global_ep, epoch):
        self.model.train()
        if self.mode == 'SL':
            criterion = self.sce_loss
        else:
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

class bline:
    def __init__(self,mode):
        self.mode = mode
        self.model = None
        self.ini_model()
        self.clients = []
        for p_id in range(args.num_users):
            self.clients.append(Client(p_id, copy.deepcopy(self.model),clientPath,self.mode))
        self.server = Server(self.model,serverPath,self.mode)
        for ix in range(args.num_users):
            self.clients[ix].receivemodel()

    def create_model(self):
        model = ResNet18(num_classes=args.num_classes)
        model = model.to(args.device)
        return model

    def ini_model(self):
        if args.dataset == 'cifar10':
            self.model = self.create_model()



    def conf_FL(self,max_epochs=200):

        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]

        for epoch in range(max_epochs):

            for ix in range(args.num_users):
                if epoch == 0:
                    self.clients[ix].confidence()
                    self.clients[ix].data_filter()
                model_param[ix],Keep_size[ix] = self.clients[ix].local_update(epoch)

            self.server.FedAvg(model_param, Keep_size)
            self.server.sendmodel()
            self.server.test(epoch)
            for ix in range(args.num_users):
                self.clients[ix].receivemodel()

            if epoch %10==0:
                torch.save(self.server.model.state_dict(), os.path.join(args.save_dicts_dir,
                                                        'global_train_%d.json' % (epoch)))
    def compute_lamda(self,samp1,samp2):


        user_len = len(samp2)
        # samp2.squeeze()
        enu1 = sorted(enumerate(samp1), key=itemgetter(1))
        enu2 = sorted(enumerate(samp2), key=itemgetter(1))

        sp1 = [value for index, value in enu1]
        sp2 = [value for index, value in enu2]

        temp = -1

        D = stats.ks_2samp

        inixis = D(samp1, samp2)
        inixis = inixis[0]

        distance = []
        stp = 100
        STEP = np.arange(user_len / stp, user_len + 1, user_len / stp)
        STEP = STEP.astype(np.int)
        STEP_loss = []
        idx = 0
        # for t in range(1,len(samp2)):
        for t in STEP:
            trunsp2 = sp2[:t]
            STEP_loss.append(sp2[t - 1].item())
            dis = D(sp1, trunsp2)
            dis = dis[0]
            distance.append(dis)
            if dis <= inixis:
                inixis = dis
                temp = t - 1
                best_idx = idx
            idx += 1
        lamda = sp2[temp]
        with open(os.path.join(serverPath, 'lamda.pkl'), 'wb') as f:
            pickle.dump(lamda, f)
        return lamda


    def Data_selection(self,max_epochs=200):

        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]
        for epoch in range(max_epochs):
            if epoch == 0:
                bench_loss = self.server.cmp_bench_loss()
                uni_loss = []
                for ix in range(args.num_users):

                    _,eachloss = self.clients[ix].reference()
                    uni_loss.extend(eachloss.tolist())
                uni_loss = np.array(uni_loss)
                lamda = self.compute_lamda(bench_loss,uni_loss)
                for ix in range(args.num_users):
                    self.clients[ix].lamda1_detect(lamda)

            for ix in range(args.num_users):
                model_param[ix],Keep_size[ix] = self.clients[ix].local_update(epoch)

            self.server.FedAvg(model_param, Keep_size)
            self.server.sendmodel()
            self.server.test(epoch)
            for ix in range(args.num_users):
                self.clients[ix].receivemodel()
            if epoch %10==0:
                torch.save(self.server.model.state_dict(), os.path.join(args.save_dicts_dir,
                                                        'global_train_%d.json' % (epoch)))

    def federated_learning(self,max_epochs=200):
        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]
        for epoch in range(max_epochs):

            for ix in range(args.num_users):
                model_param[ix], Keep_size[ix] = self.clients[ix].local_update(epoch)

            self.server.FedAvg(model_param, Keep_size)
            self.server.sendmodel()
            self.server.test(epoch)
            for ix in range(args.num_users):
                self.clients[ix].receivemodel()

            if epoch % 10 == 0:
                torch.save(self.server.model.state_dict(), os.path.join(args.save_dicts_dir,
                                                        'global_train_%d.json' % (epoch)))

    def INCV_FL(self, max_epochs=200):
        Keep_size = [0] * args.num_users
        model_param = [[] for i in range(args.num_users)]
        for epoch in range(max_epochs):

            for ix in range(args.num_users):
                if epoch == 0:
                    self.clients[ix].INCV()
                model_param[ix], Keep_size[ix] = self.clients[ix].local_update(epoch)

            self.server.FedAvg(model_param, Keep_size)
            self.server.sendmodel()
            self.server.test(epoch)
            for ix in range(args.num_users):
                self.clients[ix].receivemodel()
            if epoch % 10 == 0:
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
    bl = bline(mode=args.method)
    if args.method == 'Conf':
        bl.conf_FL()
    elif args.method == 'DS':
        bl.Data_selection()
    elif args.method == 'DT':
        bl.federated_learning()
    elif args.method == 'INCV':
        bl.INCV_FL()
    elif args.method == 'SL':
        bl.federated_learning()




