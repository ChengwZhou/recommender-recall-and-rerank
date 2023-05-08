import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec


import torch
import torch.nn as nn
import torch.utils.data as Data
from MovieClass import MovieClass, MultiMovieClass





class MLP_3layer(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2):  # Define layers in the constructor
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, 2)

    def forward(self, x):  # Define forward pass in the forward method
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP_6layer(nn.Module):
    def __init__(self, n_input):  # Define layers in the constructor
        super().__init__()
        self.fc1 = nn.Linear(n_input, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 30)
        self.fc4 = nn.Linear(30, 20)
        self.fc5 = nn.Linear(20, 10)
        self.fc6 = nn.Linear(10, 2)

    def forward(self, x):  # Define forward pass in the forward method
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

def train(model, epoch_num, trainloader, optimizer, loss_func):
    device = torch.device("cpu")
    model.train()
    model.to(device)
    train_loss_all = []
    for epoch in range(epoch_num):
        train_loss = 0
        train_num = 0
        for step, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)# Move batch to device
            optimizer.zero_grad()
            output = model(x)
#             print(y.shape, output.shape)
            loss = loss_func(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)
        train_loss_all.append(train_loss / train_num)
        if epoch % 10 == 0:
            print(f'epoch:{epoch}, loss:{train_loss / train_num}')

# def test(model, testloader, loss_func):
#     device = torch.device("cpu")
#     model.eval()  # set model to evaluation mode
#     running_loss = 0
#     num = 0
#     with torch.no_grad():  # no need to compute gradients for testing
#         for step, (x, y) in enumerate(testloader):
#             x, y = x.to(device), y.to(device)
#             output = model(x)
#             # if step == 1:
#             #     print(output[:10], y[:10])
#             loss = loss_func(output, y)  # Compute loss
#             running_loss += loss.item() * x.size(0)
#             num += x.size(0)
#     return running_loss / num

def predict(model, testloader, loss_func):
    device = torch.device("cpu")
    model.eval()  # set model to evaluation mode
    prediction = []
    with torch.no_grad():  # no need to compute gradients for testing
        for step, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            output = model(x).cpu().numpy().tolist()
            prediction += output

    return prediction


def get_label(df_recall, df_y):
    groudtruth = df_y.groupby('userId').movieId.apply(list).to_dict()
    label = []
    for i in range(len(df_recall)):
        movieid = int(df_recall.iloc[i].movieId)
        userid = int(df_recall.iloc[i].userId)
        if movieid in groudtruth[userid]:
            label.append([0,1])
        else:
            label.append([1,0])
    return np.array(label)



if __name__ == "__main__":
    df_y = pd.read_csv('data/ground_truth.csv')
    df_recall = pd.read_csv('data/dataset_feature.csv')
    df_x = df_recall.iloc[:, 2:]
    label = get_label(df_recall, df_y)

    x_train = df_x[:27447].values[:, 1:]
    x_test = df_x[27447:].values[:, 1:]
    y_train = label[:27447]
    y_test = label[27447:]

    x_train, x_test = torch.from_numpy(x_train.astype(np.float32)), torch.from_numpy(x_test.astype(np.float32))
    y_train, y_test = torch.from_numpy(y_train.astype(np.float32)), torch.from_numpy(y_test.astype(np.float32))
    train_data = Data.TensorDataset(x_train, y_train)
    test_data = Data.TensorDataset(x_test, y_test)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=64)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=64)

    model = MLP_6layer(97)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    train(model, 100, train_loader, optimizer, criterion)
    test(model, test_loader, criterion)
    prediction = predict(model, test_loader, criterion)