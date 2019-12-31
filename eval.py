# Creat Time: 2019/12/20 14:46
# Edit Time: 2019/12/20 14:46
# Project: NR_Bulr_IQA_original
# Description: evaluate the performance of model
# Author: chengyu
# coding = utf-8

import torch
import torch.cuda
import torch.nn
from data.Dataset import MyDataloader
from torchvision import transforms
from model.BlurClassificationNet import BlurClassificationNet
from tools.scriptparse import parser

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_path = args.dataset_path
    model_params_path = args.checkpoint_path
    batchsize = args.batchsize
    gpu_ids = [int(id) for id in args.gup_ids.split(',')]
    preprocess = transforms.Compose([transforms.ToTensor(),
         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224,0.225))])  # data augment: sub-mean/std(normalization), up-down/left-right flip, tranferred to torchtensor
    blurtestloader = MyDataloader(dataset_path, 'test', batchsize, shuffle=True, num_workers=4,transform=preprocess)
    # test  phase
    BlurClsNet = BlurClassificationNet(blurtype = 6)
    if torch.cuda.is_available():
        BlurClsNet = torch.nn.DataParallel(BlurClsNet, device_ids=gpu_ids).cuda(device=gpu_ids[0])  # specify the available GPUs, depoly the model on gpu_ids[0]
    BlurClsNet.load_state_dict(torch.load(model_params_path, map_location='cuda:' + str(gpu_ids[0])))
    BlurClsNet.eval()

    num_wrong = 0
    num_correct = 0

    for i, batch in enumerate(blurtestloader):  # 9600 images
        img, label = batch
        img = img.cuda(device=gpu_ids[0])
        result = BlurClsNet(img)
        [score, category] = result.max(dim=1)
        category = category.cpu()
        for j,r in enumerate(category):
            if  r==label[j]:
                num_correct += 1
            else:
                num_wrong += 1
        print('Iter %d, correct number is %d, error number is %d'%(i+1,num_correct,num_wrong))

    print('Class Accuracy is %f, correct number is %d, error number is %d'%(num_correct/(num_correct+num_wrong),num_correct,num_wrong))
