from model import CNN
from torch.autograd import Variable
import torch
import torch.nn as nn
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
import torchvision
EPOCH = 200  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
cnn = CNN()
print(cnn)  # net architecture

train_data = MyDataset(datatxt='./all/train_set.csv', transform=torchvision.transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

test_data = MyDataset(datatxt='./all/val_set.csv', transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)  # batch x
        b_y = Variable(y)  # batch y

        output = cnn(b_x)[0]  # cnn output
        loss = loss_func(output, b_y)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            cnn.eval()
            eval_loss = 0.
            eval_acc = 0.
            for i, (tx, ty) in enumerate(test_loader):
                t_x = Variable(tx)
                t_y = Variable(ty)
                output = cnn(t_x)[0]
                loss = loss_func(output, t_y)
                eval_loss += loss.item()
                pred = torch.max(output, 1)[1]
                # print(pred)
                num_correct = (pred == t_y).sum()
                eval_acc += float(num_correct.item())
            acc_rate = eval_acc / float(len(test_data))
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), acc_rate))
        torch.save(cnn.state_dict(),'./checkpoints/params_{}_{}.pkl'.format(epoch,step))
