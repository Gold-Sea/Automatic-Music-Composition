import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json


class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self,input):
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out =self.predict(out)
        return out



def train(model, features, labels, epochs, path):
    assert len(features) == len(labels)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.05)
    loss_func = torch.nn.MSELoss()
    for i in epochs:
        for i in range(len(features)):
            input = features[i]
            label = labels[i]
            prediction = model(input)
            loss = loss_func(prediction, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, path) 



if __name__ == "__main__":
    net = Net(32,64,1)
    print(net.children)
    with open("./GA.json",'r') as load_f:
        load_dict = json.load(load_f)
        path = load_dict['model_path']
    train(net, )
