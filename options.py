#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='CLC',
                        help="method option:CLC/DS/DT/INCV/SL/Conf")
    parser.add_argument('--first_epochs', type=int, default=150,
                        help="number of rounds before correction")
    parser.add_argument('--last_epochs', type=int, default=50,
                        help="number of rounds after correction")

    parser.add_argument('--benchmark', type=bool, default=True,
                        help="the server can use an benchmark dataset to pretrain model")
    parser.add_argument('--num_users', type=int, default=19,
                        help="number of users: K")
    parser.add_argument('--noise_rate', type=float, default=0.5,
                        help="noise rate")
    parser.add_argument('--device', type=str, default='cuda',
                        help='device name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')

    parser.add_argument('--local_ep', type=int, default=20,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size")
    parser.add_argument('--local_mom', type=int, default=0.9,
                        help="momentem for optimizer")  # 0.9
    parser.add_argument('--local_decay', type=int, default=5e-4,
                        help="momentem for optimizer")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                            of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                            of classes")


    args = parser.parse_args()
    return args
