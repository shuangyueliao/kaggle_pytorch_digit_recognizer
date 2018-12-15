from model import CNN
from torch.autograd import Variable
import torch
import torch.nn as nn
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import torchvision
import csv
cnn = CNN()
state_dict=torch.load('./checkpoints/params_7_585.pkl')
cnn.load_state_dict(state_dict)
print(cnn)  # net architecture

test_data = MyDataset(datatxt='./all/test.csv',train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=1)
with open('./all/sample_submission.csv', 'w',newline='') as myFile:
    myWriter = csv.writer(myFile)
    tmp=[]
    tmp.append('ImageId')
    tmp.append('Label')
    myWriter.writerow(tmp)

    for step, (b_x ) in enumerate(test_loader):  # gives batch data, normalize x when iterate train_loader
        cnn.eval()
        output = cnn(b_x)[0]
        pred = torch.max(output, 1)[1]
        tmp = []
        tmp.append(step+1)
        tmp.append(int(pred))
        myWriter.writerow(tmp)
        print(step)
