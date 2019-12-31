# Creat Time: 2019/12/17 20:37
# Edit Time: 2019/12/17 20:37
# Project: NR_Bulr_IQA_original
# Description: 
# Author: chengyu
# coding = utf-8
from visdom import Visdom
import numpy as np

class visualizer(object):
    def __init__(self, env_name):
        self.viz = Visdom(server='http://127.0.0.1', port=8097)
        assert self.viz.check_connection()
        self.curve = {}
        self.env = env_name

    def plot_curve(self,x_data,y_data):
        for k,v in y_data.items():
            if k not in self.curve.keys():
                self.curve[k] = []
                self.curve[k + '_epoch'] = []

            self.curve[k + '_epoch'].append(x_data)
            self.curve[k].append(v)

            X = np.array(self.curve[k + '_epoch'])
            Y = np.array(self.curve[k])
            self.viz.line(
                X = X,
                Y = Y,
                opts={
                    'title': k,
                    'legend': ['loss'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win = k,
            env=self.env)