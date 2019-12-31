# Creat Time: 2019/12/17 19:20
# Edit Time: 2019/12/17 19:20
# Project: NR_Bulr_IQA_original
# Description: transform the tif format to rgb format
# Author: chengyu
# coding = utf-8

from PIL import Image
import os
SrcFilePath=r"G:\NAIP_Dataset\TestData\06m"
DesFilePath=r"G:\NAIP_Dataset\TestData\06m"
ftype = '.tif'
FilesList = []
for file in os.listdir(SrcFilePath):
    if ftype in file and os.path.isfile(os.path.join(SrcFilePath, file[:-4] + '.png')) is False:# and os.path.getsize(os.path.join(SrcFilePath, file))<278956970
        print('Processing file %s'%(file))
        FilesList.append(file)
        img = Image.open(os.path.join(SrcFilePath, file))
        img = img.convert("RGB")
        img.save(os.path.join(DesFilePath, file[:-4] + '.png'))

print(FilesList)

