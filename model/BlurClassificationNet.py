# Creat Time: 2019/12/13 21:27
# Edit Time: 2019/12/13 21:27
# Project: NR_Bulr_IQA_original
# Description: network model
# Author: chengyu
# coding = utf-8
from torch import nn
import torch
#model construction
class BlurClassificationNet(nn.Module):
    def __init__(self, init = None, blurtype = 6):
        super(BlurClassificationNet,self).__init__()
        #conv stage
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7,padding=3)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=7,padding=3)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,padding=2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,padding=2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.maxpool4 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.maxpool5 = nn.MaxPool2d(2,2)

        #dilation conv stage
        self.dilaconv1_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1, dilation=1)
        self.dilaconv1_1_relu = nn.ReLU()
        self.dilaconv1_1_bn = nn.BatchNorm2d(num_features=128)
        self.dilaconv1_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding = 1, dilation=1)
        self.dilaconv1_2_relu = nn.ReLU()
        self.dilaconv1_2_bn = nn.BatchNorm2d(num_features=64)
        self.dilaconv1_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding = 1, dilation=1)
        self.dilaconv1_3_relu = nn.ReLU()
        self.dilaconv1_3_bn = nn.BatchNorm2d(num_features=32)
        self.dilaconv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 2, dilation=2)
        self.dilaconv2_1_relu = nn.ReLU()
        self.dilaconv2_1_bn = nn.BatchNorm2d(num_features=128)
        self.dilaconv2_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding = 2, dilation=2)
        self.dilaconv2_2_relu = nn.ReLU()
        self.dilaconv2_2_bn = nn.BatchNorm2d(num_features=64)
        self.dilaconv2_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding = 2, dilation=2)
        self.dilaconv2_3_relu = nn.ReLU()
        self.dilaconv2_3_bn = nn.BatchNorm2d(num_features=32)
        self.dilaconv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 4, dilation=4)
        self.dilaconv3_1_relu = nn.ReLU()
        self.dilaconv3_1_bn = nn.BatchNorm2d(num_features=128)
        self.dilaconv3_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding = 4, dilation=4)
        self.dilaconv3_2_relu = nn.ReLU()
        self.dilaconv3_2_bn = nn.BatchNorm2d(num_features=64)
        self.dilaconv3_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding = 4, dilation=4)
        self.dilaconv3_3_relu = nn.ReLU()
        self.dilaconv3_3_bn = nn.BatchNorm2d(num_features=32)
        self.dilaconv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 8, dilation=8)
        self.dilaconv4_1_relu = nn.ReLU()
        self.dilaconv4_1_bn = nn.BatchNorm2d(num_features=128)
        self.dilaconv4_2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding = 8, dilation=8)
        self.dilaconv4_2_relu = nn.ReLU()
        self.dilaconv4_2_bn = nn.BatchNorm2d(num_features=64)
        self.dilaconv4_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding = 8, dilation=8)
        self.dilaconv4_3_relu = nn.ReLU()
        self.dilaconv4_3_bn = nn.BatchNorm2d(num_features=32)
        self.dilaconv_maxpool = nn.MaxPool2d(16,16)
        #fc and dropout, category num is 6
        self.fc1 = nn.Linear(32*4,256)
        self.dropout1 = nn.Dropout2d(0.5)#Dropout layer actually set the random selected nuerons to zero, but the shape of vector is not changed
        # self.fc2 = nn.Linear(128,256)
        # self.dropout2 = nn.Dropout2d(0.5)
        self.fc3 = nn.Linear(256,blurtype)
        self.softmax = nn.Softmax()
        #package
        self.conv = nn.Sequential(
            self.conv1, self.relu1, self.bn1, self.conv2, self.relu2, self.bn2, self.maxpool2, self.conv3,
             self.relu3, self.bn3, self.maxpool3, self.conv4, self.relu4, self.bn4, self.maxpool4,self.conv5,self.relu5,self.bn5,self.maxpool5)
        self.dilaconv1 = nn.Sequential(
            self.dilaconv1_1, self.dilaconv1_1_relu, self.dilaconv1_1_bn, self.dilaconv1_2, self.dilaconv1_2_relu, self.dilaconv1_2_bn, self.dilaconv1_3,
             self.dilaconv1_3_relu, self.dilaconv1_3_bn)
        self.dilaconv2 = nn.Sequential(
            self.dilaconv2_1, self.dilaconv2_1_relu, self.dilaconv2_1_bn, self.dilaconv2_2, self.dilaconv2_2_relu, self.dilaconv2_2_bn, self.dilaconv2_3,
             self.dilaconv2_3_relu, self.dilaconv2_3_bn)
        self.dilaconv3 = nn.Sequential(
            self.dilaconv3_1, self.dilaconv3_1_relu, self.dilaconv3_1_bn, self.dilaconv3_2, self.dilaconv3_2_relu, self.dilaconv3_2_bn, self.dilaconv3_3,
             self.dilaconv3_3_relu, self.dilaconv3_3_bn)
        self.dilaconv4 = nn.Sequential(
            self.dilaconv4_1, self.dilaconv4_1_relu, self.dilaconv4_1_bn, self.dilaconv4_2, self.dilaconv4_2_relu, self.dilaconv4_2_bn, self.dilaconv4_3,
             self.dilaconv4_3_relu, self.dilaconv4_3_bn)
        self.fc = nn.Sequential(self.fc1,self.dropout1,self.fc3,self.softmax)#,self.fc2,self.dropout2
        #parameters initial  kaiming init
        if init is not None:# conv2d has a built-in initialization
            print("No model parameter initial now!")
            assert(False)
    def forward(self, x):
        conv_ft = self.conv(x)
        dilaconv1_ft = self.dilaconv1(conv_ft)
        dilaconv2_ft = self.dilaconv2(conv_ft)
        dilaconv3_ft = self.dilaconv3(conv_ft)
        dilaconv4_ft = self.dilaconv4(conv_ft)
        concat_ft = self.dilaconv_maxpool(torch.cat((dilaconv1_ft,dilaconv2_ft,dilaconv3_ft,dilaconv4_ft),dim=1))
        output = self.fc(concat_ft.squeeze())
        return output


