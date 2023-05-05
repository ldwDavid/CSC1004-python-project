from __future__ import print_function
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch, seed):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    training_acc = correct / len(train_loader.dataset)
    training_loss = total_loss / len(train_loader)
    print("training: seed: %d epoch: %d acc: %.4f loss:%.4f" % (seed, epoch, training_acc, training_loss))
    return training_acc, training_loss


def test(model, device, test_loader, seed):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    testing_acc = correct / len(test_loader.dataset)
    testing_loss = test_loss / len(test_loader.dataset)
    print("testing: seed: %d acc: %.4f loss:%.4f" % (seed, testing_acc, testing_loss))
    return testing_acc, testing_loss


def plot(epoches, performance, ylabel, save_path):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    plt.figure()
    plt.plot(epoches, performance)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs Epoch')
    plt.savefig(save_path)


def run(config, Seed):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs,
                                               worker_init_fn=lambda worker_id: np.random.seed(config.seed + worker_id),
                                               generator=torch.Generator().manual_seed(Seed))
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs,
                                              worker_init_fn=lambda worker_id: np.random.seed(config.seed + worker_id),
                                              generator=torch.Generator().manual_seed(Seed))

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch, Seed)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        with open("train_results.txt", "a") as Train:
            Train.write("%.4f,%.4f" % (train_acc, train_loss))
            Train.write("\n")
            Train.close()
        """record training info"""
        test_acc, test_loss = test(model, device, test_loader, Seed)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        with open("test_results.txt", "a") as Test:
            Test.write("%.4f,%.4f" % (test_acc, test_loss))
            Test.write("\n")
            Test.close()
        """record testing info"""
        epoches.append(epoch)
        scheduler.step()
        """update the records, Fill your code"""

    """plotting training performance with the records"""
    plot(epoches, training_accuracies, "training accuracies", "training accuracies"+str(Seed)+".png")
    plot(epoches, training_loss, "training loss", "training loss"+str(Seed)+".png")

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, "testing_accuracies", "testing_accuracies"+str(Seed)+".png")
    plot(epoches, testing_loss, "testing_loss", "testing_loss"+str(Seed)+".png")

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean():
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    epoches = [i+1 for i in range(config.epochs)]
    testing_accuracies = [0 for i in range(config.epochs)]
    testing_loss = [0 for i in range(config.epochs)]
    training_accuracies = [0 for i in range(config.epochs)]
    training_loss = [0 for i in range(config.epochs)]

    with open("test_results.txt", "r") as Test:
        with open("train_results.txt", "r") as Train:
            for j in range(3):
                for i in range(config.epochs):
                    test_result = Test.readline().strip().split(",")
                    train_result = Train.readline().strip().split(",")
                    testing_accuracies[i] += float(test_result[0]) / 3
                    testing_loss[i] += float(test_result[1]) / 3
                    training_accuracies[i] += float(train_result[0]) / 3
                    training_loss[i] += float(train_result[1]) / 3
    print(testing_accuracies)
    print(testing_loss)
    print(training_loss)
    print(training_accuracies)
    plot(epoches, training_accuracies, 'Mean Training Accuracy', "MeanTrainingAccuracy.png")
    plot(epoches, training_loss, 'Mean Training Loss', "MeanTrainingLoss.png")
    plot(epoches, testing_accuracies, 'Mean Testing Accuracy', "MeanTestingAccuracy.png")
    plot(epoches, testing_loss, 'Mean Testing Loss', "MeanTestingLoss.png")


if __name__ == '__main__':
    start_time = time.time()
    matplotlib.use('Agg')

    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    f = open("train_results.txt", "w")  # erase previous content
    f.close()
    f = open("test_results.txt", "w")
    f.close()
    run(config, 123)
    run(config, 321)
    run(config, 666)

    print(time.time() - start_time)

    """plot the mean results"""
    plot_mean()
