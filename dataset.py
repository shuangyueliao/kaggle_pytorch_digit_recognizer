from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from PIL import Image
import os
from skimage import io
class MyDataset(Dataset):
    def __init__(self, datatxt,train=True,transform=None, target_transform=None):
        self.data = pd.read_csv(datatxt)
        self.transform = transform
        self.train = train
        if self.train:
            self.X = self.data.iloc[:, 1:]
            self.X = np.array(self.X)
            self.y = self.data.iloc[:, 0]
        else:
            self.X = self.data
            self.X = np.array(self.X)

    def __getitem__(self, index):
        im = self.X[index]
        im=im.reshape(28, 28)
        im = Image.fromarray(im.astype(np.uint8), mode='L')
        if self.transform is not None:
            im = self.transform(im)
        if self.train:
            label = self.y[index]
            label=label.item()
            return im, label
        else:
            return im

    def __len__(self):
        return len(self.data)

# def convert_to_img(train=True):
#     root='E:/kaggledigit/savemnist/'
#     train_data = MyDataset(datatxt='./all/train_set.csv', transform=None)
#     test_data = MyDataset(datatxt='./all/val_set.csv', transform=None)
#     i=0
#     if(train):
#         f=open(root+'train.txt','w')
#         data_path=root+'/train/'
#         if(not os.path.exists(data_path)):
#             os.makedirs(data_path)
#         for (img,label) in train_data:
#             img_path=data_path+str(i)+'.jpg'
#             img.save(img_path)
#             # io.imsave(img_path,img)
#             f.write(img_path+' '+str(label)+'\n')
#             i=i+1
#             print(i)
#         f.close()
#     else:
#         f = open(root + 'test.txt', 'w')
#         data_path = root + '/test/'
#         if (not os.path.exists(data_path)):
#             os.makedirs(data_path)
#         for (img,label) in test_data:
#             img_path = data_path+ str(i) + '.jpg'
#             img.save(img_path)
#             f.write(img_path + ' ' + str(label) + '\n')
#             i=i+1
#             print(i)
#         f.close()
#
# if __name__ == '__main__':
#     convert_to_img(False)