# Creat Time: 2019/12/17 19:20
# Edit Time: 2019/12/17 19:20
# Project: NR_Bulr_IQA_original
# Description: parse command line
# Author: chengyu
# coding = utf-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset_path',type = str,help='train annotaions path')
parser.add_argument('-model_init',type = str, default='kaiming',help='model initialization')
parser.add_argument('-continue',help='continue training')
parser.add_argument('-checkpoint_path',type = str,help='continue training checkpoint_path')
parser.add_argument('-checkpoint_epoch',type = int, default= 1,help='continue training checkpoint epoch')
parser.add_argument('-epoch_num',type = int, default=300,help='total epoch of training')
parser.add_argument('-batchsize',type = int, default=10)
parser.add_argument('-num_categories',type = int, default=6,help='number of categories')
parser.add_argument('-lr_init',type = float, default=0.001,help='initial learning rate')
parser.add_argument('-lr_decay',type = str, default='steplr',help='decay type of learning rate')
parser.add_argument('-lr_step',type = int, default=100,help='decay step of epoch')
parser.add_argument('-lr_gamma',type = float, default=0.1,help='decay coefficient of learning rate')
parser.add_argument('-gup_ids',type = str, default='0',help='gpu ids for trainning')