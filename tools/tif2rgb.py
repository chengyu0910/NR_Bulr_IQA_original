#
# #导入cv模块
# import cv2 as cv
# import numpy as np
# #读取图像，支持 bmp、jpg、png、tiff 等常用格式
# #第二个参数是通道数和位深的参数，有四种选择，参考https://www.cnblogs.com/goushibao/p/6671079.html
# img = cv.imread("m_4007423_ne_18_1_20170801.tif",-1)
#
# #在这里一开始我写成了img.shape（），报错因为img是一个数组不是一个函数，只有函数才可以加()表示请求执行，
# #参考http://blog.csdn.net/a19990412/article/details/78283742
# print(img.shape)
# print(img.dtype)
# print(img.max().max().max())
# img_rgb = np.zeros(shape=(7590, 5946, 3))
# img_rgb[:,:,0:3] = img[:,:,0:3]/2
# img_rgb = img_rgb.astype(np.uint8)
# #创建窗口并显示图像
# cv.namedWindow("Image")
# cv.imshow("Image",img_rgb)
# cv.waitKey(0)
# #释放窗口


# from osgeo import gdal
# file_path="G:/spatial_resolution_estimation/m_4007423_ne_18_1_20170801/m_4007423_ne_18_1_20170801.tif"
# ds=gdal.Open(file_path)
# driver=gdal.GetDriverByName('PNG')
# dst_ds = driver.CreateCopy('G:/spatial_resolution_estimation/m_4007423_ne_18_1_20170801//example.png', ds)
# dst_ds = None
# src_ds = None

from PIL import Image
import os
SrcFilePath=r"G:\NAIP_Dataset\NewYork City"
DesFilePath=r"G:\NAIP_Dataset\NewYork City"
ftype = '.tif'
FilesList = []
for file in os.listdir(SrcFilePath):
    if ftype in file and os.path.isfile(os.path.join(SrcFilePath, file[:-4] + '.png')) is False and os.path.getsize(os.path.join(SrcFilePath, file))<278956970:
        print('Processing file %s'%(file))
        FilesList.append(file)
        img = Image.open(os.path.join(SrcFilePath, file))
        img = img.convert("RGB")
        img.save(os.path.join(DesFilePath, file[:-4] + '.png'))

print(FilesList)

