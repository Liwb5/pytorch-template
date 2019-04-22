import torch
from torch.utils.data import DataLoader
import numpy as np 
import scipy.misc, os

class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, data_quota):
        """
        args:
            data_path:
            data_quota: 训练集的图片数量
        """
        self.data_path = data_path 
        if data_quota == -1:
            data_quota = len([name for name in os.listdir(data_path)])

        self.data_quota = data_quota
        self.data_files = []
        for i in range(data_quota):
            self.data_files.append(os.path.join(data_path,"{}.jpg".format(i)))

    def __getitem__(self, ind):
        path = self.data_files[ind]
        img = scipy.misc.imread(path)
        #  img = img.transpose(2,0,1)-127.5/127.5
        return img

    def __len__(self):
        return len(self.data_files)

if __name__ == "__main__":
    data_path = '../data/anime/'
    dataset = AnimeDataset(data_path, -1)
    loader = DataLoader(dataset, batch_size=10, shuffle=True,num_workers=4)
    for i, inp in enumerate(loader):
        print(i,inp.size())
