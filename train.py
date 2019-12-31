#coding=utf-8
import torch
import torch.nn
import torch.optim
import torch.cuda
from data.Dataset import MyDataloader
from torchvision import transforms
from model.BlurClassificationNet import BlurClassificationNet
from tools.scriptparse import parser
from tools.visualizer import visualizer
import os
import time


def checkpoint_dir(checkpoint_path, epoch):
    return os.path.join(checkpoint_path,str(epoch)+'.pth')
def calcu_time(epoch,iter,epoch_num,blurtrainloader,blurtestloader,time_stamp):
    rest_sec = (epoch_num - epoch) * (len(blurtrainloader) + len(blurtestloader)) * (time.time() - time_stamp) / iter
    res_d = int(rest_sec / (24 * 60 * 60))
    res_h = int((rest_sec - res_d * (24 * 60 * 60)) / (60 * 60))
    res_m = int((rest_sec - res_d * (24 * 60 * 60) - res_h * (60 * 60)) / 60)
    return '%d day %d hour %d min'%(res_d,res_h,res_m)

if __name__ == '__main__':
    args = parser.parse_args()
    #training setting
    model_init, epoch, epoch_num, checkpoint_path, batchsize, num_categories, \
    lr_init, lr_decay, lr_step, lr_gamma, dataset_path, gpu_ids = \
        args.model_init, args.checkpoint_epoch, args.epoch_num, args.checkpoint_path, \
        args.batchsize, args.num_categories, args.lr_init, args.lr_decay, \
        args.lr_step, args.lr_gamma, args.dataset_path, [int(id) for id in args.gup_ids.split(',')]
    #check GPU
    cuda_avail = torch.cuda.is_available()
    #new checkpoint path when training from scratch
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.join(os.getcwd(),'checkpoint'),time.strftime("%Y-%m-%d %H:%M:%S").replace(':','-').replace(' ','-'))
        os.makedirs(checkpoint_path)
        train_setting = open(os.path.join(checkpoint_path, 'train_setting.txt'), mode='x')# save trainig setting in checkpoint folder
        train_setting.write(str(args))
        train_setting.close()
    #construct dataset and model
    preprocess = transforms.Compose([transforms.RandomHorizontalFlip(0.5),transforms.RandomVerticalFlip(0.5),transforms.ToTensor(),transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))])#data augment: sub-mean/std(normalization), up-down/left-right flip, tranferred to torchtensor
    blurtrainloader, blurtestloader = MyDataloader(dataset_path, 'train', batchsize, shuffle=True, num_workers=4,transform=preprocess), \
                                      MyDataloader(dataset_path, 'test', batchsize, shuffle=True, num_workers=4,transform=preprocess)
    BlurClsNet = BlurClassificationNet(blurtype = 6)
    #model init
    if cuda_avail is True:
        BlurClsNet = torch.nn.DataParallel(BlurClsNet, device_ids=gpu_ids).cuda(device=gpu_ids[0])#specify the available GPUs, depoly the model on gpu_ids[0]
    if epoch is not 1:# load model checkpoint
        BlurClsNet.load_state_dict(torch.load(checkpoint_dir(checkpoint_path,epoch)))
    #loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([{'params': BlurClsNet.parameters(), 'initial_lr': lr_init}],lr = lr_init)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma, last_epoch=-1 if epoch is 1 else epoch)
    #visualizaer and loss logger
    viz = visualizer(env_name='BlurClassification')
    loss_logger = open(os.path.join(checkpoint_path, 'loss_logger.txt'), mode='w')
    #training
    log_info = 'Start Traning, Epoch: %d/%d, Time:%s' % (epoch, epoch_num, time.strftime("%Y-%m-%d %H:%M:%S"))
    print(log_info)
    loss_logger.write(log_info)
    while(epoch <= epoch_num):
        lr_scheduler.step()
        time_stamp = time.time()
        #train  phase
        train_loss = 0
        BlurClsNet.train()
        for i,batch in enumerate(blurtrainloader):#14412 images
            img, label = batch
            label = torch.zeros(len(label), num_categories).scatter_(1, label.unsqueeze(dim=1), 1)# transfer the label num to one-hot vector
            img, label = img.cuda(device=gpu_ids[0]), label.cuda(device=gpu_ids[0])# assign data on gpu_id[0]
            result = BlurClsNet(img)
            loss = criterion(result,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
            rest_time = calcu_time(epoch, i+1, epoch_num, blurtrainloader, blurtestloader, time_stamp)
            print('Train, Epoch: %d-%d/%d, Loss: %f, Rest Time: %s, Clock:%s' % (epoch, i, epoch_num, loss.data.item(),rest_time,time.strftime("%Y-%m-%d %H:%M:%S")))
        train_loss /= len(blurtrainloader)

        #test  phase
        test_loss = 0
        BlurClsNet.eval()
        for i,batch in enumerate(blurtestloader):#9600 images
            img, label = batch
            label = torch.zeros(len(label), num_categories).scatter_(1, label.unsqueeze(dim=1), 1)# transfer the label num to one-hot vector
            img, label = img.cuda(device=gpu_ids[0]), label.cuda(device=gpu_ids[0])# assign data on gpu_id[0]
            result = BlurClsNet(img)
            loss = criterion(result,label)
            test_loss += loss.data.item()
            print('Test, Epoch: %d-%d/%d, Loss: %f, Clock: %s' % (epoch, i, epoch_num, loss.data.item(),time.strftime("%Y-%m-%d %H:%M:%S")))
        test_loss /= len(blurtestloader)

        log_info = 'Epoch: %d/%d, Train Loss: %f, Test Loss: %f, Time:%s'%(epoch, epoch_num, train_loss, test_loss, time.strftime("%Y-%m-%d %H:%M:%S"))
        print(log_info)
        loss_logger.write(log_info)
        viz.plot_curve(epoch,{'loss_t':train_loss,'loss_v':test_loss})
        torch.save(BlurClsNet.state_dict(),checkpoint_dir(checkpoint_path,epoch))
        epoch += 1

    loss_logger.close()