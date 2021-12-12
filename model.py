# ANN模型的训练和推理
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import IO

from torch.storage import T

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    path = load_dict['model_path']
    bs = load_dict["batch_size"]
    is_trained = load_dict["is_trained"]

# ANN的实现
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output, ch):
        self.n_hidden = n_hidden
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden//2,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
        self.conv = nn.Conv1d(ch, ch, 2)

    def forward(self,input):
        out = self.hidden1(input)
        out = torch.relu(out)
        s1 = out.shape[0]
        out = torch.reshape(out, (s1, self.n_hidden//2, 2))
        out = self.conv(out)
        out = torch.reshape(out, (s1, self.n_hidden//2))
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out =self.predict(out)
        return out


# 训练并保存ANN模型
def train(model, features, labels, epochs, path):
    assert len(features) == len(labels)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.05)
    loss_func = torch.nn.MSELoss()
    # train batch by batch
    for j in range(epochs):
        print("Training in %s epochs"%{j})
        for i in range(len(features)):
            feature = features[i]
            label = labels[i]
            prediction = model(feature)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, path) 

# ANN测试
def test(features, labels, path):
    model = torch.load(path)
    loss_func = torch.nn.MSELoss()
    tot_loss = []
    for i in range(len(features)):
        feature = features[i]
        label = labels[i]
        prediction = model(feature)
        loss = loss_func(prediction, label)
        tot_loss.append(float(loss))
    print("tot avg loss =", sum(tot_loss)/len(tot_loss))

# ANN推理
def inference(genes):
    input = torch.tensor(genes,dtype=torch.float32)
    mod = torch.load(path)
    return mod(input).detach().numpy()


if __name__ == "__main__":
    x=[]
    y=[]
    _x, _y = IO.get_data("./data/chopin_nocturnes_train.txt")
    for i in range(len(_x)):
        x.append(torch.tensor(torch.from_numpy(_x[i]), dtype=torch.float32))
        y.append(torch.tensor(torch.from_numpy(_y[i]), dtype=torch.float32))

    x_test=[]
    y_test=[]
    _x_test, _y_test = IO.get_data("./data/chopin_nocturnes_test.txt")
    for i in range(len(_x_test)):
        x_test.append(torch.tensor(torch.from_numpy(_x_test[i]), dtype=torch.float32))
        y_test.append(torch.tensor(torch.from_numpy(_y_test[i]), dtype=torch.float32))

    net = Net(32,32,1,16)
    print(net.children)
    if not is_trained:
        train(net, x, y ,500, path)
    else:
        test(x_test, y_test, path)
