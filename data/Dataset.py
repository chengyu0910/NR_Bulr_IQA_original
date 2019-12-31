from PIL import  Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import os
class MyDataset(Dataset):
    def __init__(self, dataset_path, branch, transform): #transform由外部定义
        self.imgset_info = []
        self.transform = transform
        self.imgset_path = os.path.join(os.path.join(dataset_path,'images'),branch)
        imginfo_list = os.listdir(self.imgset_path)
        for img_info in imginfo_list:
            img_name = img_info
            strs = img_info.strip().split('_')
            img_label = int(strs[-1][:-4])
            self.imgset_info.append((img_name, img_label))
        # txt_file = os.path.join(os.path.join(os.path.join(dataset_path,'annotations'),branch),'annotations.txt')
        # imginfo_list = open(txt_file,'r')
        # self.imgset_info = []
        # self.transform = transform
        # self.imgset_path = os.path.join(os.path.join(dataset_path,'images'),branch)
        # for img_info in imginfo_list:
        #     img_name, img_label = img_info.strip().split(' ')
        #     self.imgset_info.append((img_name, img_label))
    def __getitem__(self, index):
        name, label = self.imgset_info[index]
        img = Image.open(os.path.join(self.imgset_path, name)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)
    def __len__(self):
        return len(self.imgset_info)

def MyDataloader(dataset_path, branch, batchsize, shuffle=True, num_workers=4, transform=None):
    dataset = MyDataset(dataset_path, branch, transform)
    return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers)