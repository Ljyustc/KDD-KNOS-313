# -*- coding:utf-8 -*-

from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='RKLF-Graph2Tree')
    parser.add_argument('--cuda', type=str, dest='cuda_id', default=None)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='linear')
    parser.add_argument('--scheduler_steps', type=int, default=10, help='total number of step for the scheduler')
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--fixed_lr', action='store_true')
    parser.add_argument('--lr', type=float, default=0.000075, help='learning rate of T5')
    args = parser.parse_args()
    return args

args = get_args()