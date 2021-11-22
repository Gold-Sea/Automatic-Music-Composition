import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json

from torch.storage import T

with open("./GA.json",'r') as load_f:
    load_dict = json.load(load_f)
    path = load_dict['model_path']
    bs = load_dict["batch_size"]
    is_trained = load_dict["is_trained"]


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


# train and save the model checkpoint
def train(model, features, labels, epochs, path):
    assert len(features) == len(labels)
    optimizer = torch.optim.SGD(model.parameters(),lr = 0.05)
    loss_func = torch.nn.MSELoss()
    for j in range(epochs):
        print("Training in %s epochs"%{j})
        prediction = model(features)
        loss = loss_func(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model, path) 

# model inference
def inference(genes):
    assert isinstance(genes, np.ndarray)
    input = torch.tensor(genes,dtype=torch.float32)
    mod = torch.load(path)
    return mod(input).detach().numpy()


if __name__ == "__main__":
    x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    y = x * 2 + 1
    print(x.shape)
    net = Net(1,16,1)
    print(net.children)
    if not is_trained:
        train(net, x, y ,500, path)

    # test model inference
    # x = []
    # for i in range(100):
    #     x.append(i/100)
    # x = np.array(x)
    # x = np.reshape(x, (100,1))
    # print(x)
    # print(inference(x))

    # with open("./GA.json",'w') as f:
    #     load_dict["is_trained"] = True
    #     json.dump(load_dict, f)
    
